import wfdb
import os
import glob
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt, cheb2ord, cheby2, firwin, find_peaks
from sklearn.decomposition import FastICA
from scipy.interpolate import RectBivariateSpline
from scipy import interpolate
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.signal import welch
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
import matplotlib.animation as animation
import matplotlib.cm as cm
from IPython.display import HTML
from pyclustering.cluster.gmeans import gmeans
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils.metric import type_metric, distance_metric
from sklearn.svm import SVC
import itertools
import json
from pathlib import Path
from collections.abc import Mapping
from typing import List, Sequence, Any, Optional
import hdbscan
import warnings
np.warnings = warnings
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- バンドパスフィルタ -----
def _ensure_2d(x: np.ndarray) -> tuple[np.ndarray, bool]:
    """(n_samples, n_channels) に整形して処理し、元形状を覚えて戻す。"""
    x = np.asarray(x)
    was_1d = False
    if x.ndim == 1:
        x = x[:, None]
        was_1d = True
    elif x.ndim != 2:
        raise ValueError("Input must be 1D or 2D array (samples[, channels]).")
    return x, was_1d

def band_edges_safe(low_hz: float, high_hz: float, fs: float, margin: float = 0.01) -> tuple[float,float]:
    """Nyquist超えや非正を避けて安全なカットオフに丸める。"""
    nyq = fs * 0.5
    low = max(0.0, float(low_hz))
    high = min(float(high_hz), nyq*(1.0 - margin))
    if not (0.0 < low < high < nyq):
        raise ValueError(f"Invalid band: low={low_hz}, high={high_hz}, fs={fs}")
    return low, high

# ------------------------------
# 1) Butterworth（0位相filtfilt）推奨
# ------------------------------
def butter_bandpass_filter(x, fs: float, low_hz: float, high_hz: float, order: int = 4):
    """
    ゼロ位相IIRバンドパス。EMGの一般処理に無難。
    x: 1D (n,) または 2D (n, ch)
    """
    low, high = band_edges_safe(low_hz, high_hz, fs)
    sos = butter(order, [low, high], btype='band', fs=fs, output='sos')
    x2, was_1d = _ensure_2d(x)
    y = sosfiltfilt(sos, x2, axis=0)
    return y.ravel() if was_1d else y


# ----- ガウス関数定義 -----
def gaussian_2d(coord, A, x0, y0, sigma_x, sigma_y, theta, offset):
    x, y = coord #coord[:, 0], coord[:, 1]
    a = (np.cos(theta)**2) / (2*sigma_x**2) + (np.sin(theta)**2) / (2*sigma_y**2)
    b = -(np.sin(2*theta)) / (4*sigma_x**2) + (np.sin(2*theta)) / (4*sigma_y**2)
    c = (np.sin(theta)**2) / (2*sigma_x**2) + (np.cos(theta)**2) / (2*sigma_y**2)
    return A * np.exp(-(a*(x - x0)**2 + 2*b*(x - x0)*(y - y0) + c*(y - y0)**2)) + offset


# ------ 時間領域の特徴量 -----
def waveform_length(signal):
    return np.sum(np.abs(np.diff(signal, axis=0)), axis=0)

def ptp(signal):
    return np.ptp(signal, axis=0)

def arv(signal):
    return np.mean(np.abs(signal), axis=0)

def rms(signal):
    return np.sqrt(np.mean(signal**2, axis=0))

def zc(signal):
    s = np.sign(signal)
    s_prev = s[:-1]
    s_next = s[1:]
    sign_change = (s_prev * s_next) < 0
    return np.sum(sign_change, axis=0)


# ----- 周波数領域の特徴量 -----
def _ensure_shape_samples_64(window_2d: np.ndarray) -> np.ndarray:
    """(samples, 64) 形状に統一"""
    if window_2d.ndim != 2:
        raise ValueError("window_2d は2次元配列である必要があります。")
    if window_2d.shape[0] == 64 and window_2d.shape[1] != 64:
        window_2d = window_2d.T
    if window_2d.shape[1] != 64:
        raise ValueError(f"2次元配列の片方の次元は64である必要があります。現在: {window_2d.shape}")
    return window_2d

def _compute_psd(window_2d: np.ndarray, fs: float, nperseg=None, noverlap=None, detrend='constant'):
    """Welch法でPSDを計算し (freqs, psd) を返す。psd.shape=(n_freqs, 64)"""
    w = _ensure_shape_samples_64(window_2d)
    n_samples = w.shape[0]
    if nperseg is None:
        nperseg = min(n_samples, 1024)
    if noverlap is None:
        noverlap = nperseg // 2
    freqs, psd = welch(w, fs=fs, axis=0, nperseg=nperseg, noverlap=noverlap, detrend=detrend)
    return freqs, psd



# 1) 平均周波数（Mean Frequency）
def mean_frequency(window_2d, fs=2000, nperseg=None, noverlap=None, detrend='constant'):
    """
    単一時間窓の64ch EMGから各chの平均周波数を計算する

    Parameters
    ----------
    window_2d : np.ndarray, shape = (samples, 64) or (64, samples)
        単一時間窓のEMGデータ
    fs : float
        サンプリング周波数 [Hz]
    nperseg : int or None
        Welch法のセグメント長（Noneなら自動設定：min(ウィンドウ長, 1024)）
    noverlap : int or None
        セグメント重なり（Noneなら自動設定：nperseg//2）
    detrend : str
        Welch法のdetrend方式（'constant' など）

    Returns
    -------
    mf : np.ndarray, shape = (64,)
        各チャンネルの平均周波数 [Hz]
    """

    freqs, psd = _compute_psd(window_2d, fs, nperseg, noverlap, detrend)

    # 0除算対策
    power = np.sum(psd, axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        mf = np.sum(freqs[:, None] * psd, axis=0) / power
    mf[~np.isfinite(mf)] = np.nan  # 全パワー0などのときNaNに

    return mf


# 2) 中央周波数（Median Frequency）
def median_frequency(window_2d: np.ndarray, fs: float, nperseg=None, noverlap=None,
                          detrend='constant', fmin: float=None, fmax: float=None) -> np.ndarray:
    """
    各chの中央周波数[Hz]（総パワーの50%となる周波数）を返す。shape=(64,)
    """
    freqs, psd = _compute_psd(window_2d, fs, nperseg, noverlap, detrend)

    # 総パワーと累積和
    power = np.sum(psd, axis=0)  # (64,)
    cumsum = np.cumsum(psd, axis=0)  # (n_freqs, 64)
    half_power = 0.5 * power

    # 周波数ごとに線形補間して中央値を推定
    mf = np.full(64, np.nan, dtype=float)
    for ch in range(64):
        if power[ch] <= 0 or np.all(psd[:, ch] == 0):
            continue
        idx = np.searchsorted(cumsum[:, ch], half_power[ch])
        if idx == 0:
            mf[ch] = freqs[0]
        elif idx >= len(freqs):
            mf[ch] = freqs[-1]
        else:
            # 線形補間
            f0, f1 = freqs[idx-1], freqs[idx]
            c0, c1 = cumsum[idx-1, ch], cumsum[idx, ch]
            # c0 -> c1 の間で half_power に達する位置
            if c1 == c0:
                mf[ch] = f1
            else:
                t = (half_power[ch] - c0) / (c1 - c0)
                mf[ch] = f0 + t * (f1 - f0)
    return mf

# 3) ピーク周波数（Peak Frequency）
def peak_frequency(window_2d: np.ndarray, fs: float, nperseg=None, noverlap=None,
                        detrend='constant', fmin: float=None, fmax: float=None) -> np.ndarray:
    """
    各chのピーク周波数[Hz]（PSDが最大となる周波数）を返す。shape=(64,)
    """
    freqs, psd = _compute_psd(window_2d, fs, nperseg, noverlap, detrend)

    peak_idx = np.argmax(psd, axis=0)  # (64,)
    pf = freqs[peak_idx] if len(freqs) > 0 else np.full(64, np.nan)
    # 全パワー0のチャンネルを NaN に
    power = np.sum(psd, axis=0)
    pf = np.where(power > 0, pf, np.nan)
    return pf

# 4) スペクトルエントロピー（Spectral Entropy）
def spectral_entropy(window_2d: np.ndarray, fs: float, nperseg=None, noverlap=None,
                          detrend='constant', fmin: float=None, fmax: float=None,
                          base: float=2.0, normalized: bool=True) -> np.ndarray:
    """
    各chのスペクトルエントロピーを返す。shape=(64,)
    - PSDを確率分布 p_k = P(f_k)/sum_k P(f_k) とみなした Shannon entropy: H = -sum p_k log_b p_k
    - normalized=True の場合、log_b(N) で割って 0〜1 に正規化
    """
    freqs, psd = _compute_psd(window_2d, fs, nperseg, noverlap, detrend)

    N = psd.shape[0]
    if N == 0:
        return np.full(64, np.nan)

    power = np.sum(psd, axis=0)  # (64,)
    H = np.full(64, np.nan, dtype=float)

    # 分布を規格化してエントロピー
    with np.errstate(divide='ignore', invalid='ignore'):
        p = psd / power  # (N, 64)
        # p=0 のとき p*log(p) は 0 とみなす
        plogp = np.where(p > 0, p * (np.log(p) / np.log(base)), 0.0)
        H = -np.sum(plogp, axis=0)

    # 非正規化 or 正規化
    if normalized:
        H = H / (np.log(N) / np.log(base))  # 0〜1にスケーリング

    # 無信号チャンネルは NaN
    H = np.where(power > 0, H, np.nan)
    return H

# ----- 角度を-90°~+90°に変更する関数 -----
def ninety2ninety_radian(theta_radian):
  return (theta_radian + np.pi/2) % np.pi - np.pi/2

# ----- 筋活動中心・方向の取得 -----
# ピーク検出とガウス関数フィッティング
def gaussian_fitting(emg_data, curve, feature_func, fs=2048, window_ms=25, threshold=0, func_type=True):
  # パラメータ設定
  emg_data = emg_data  # shape: (n_samples, n_channels)
  curve = curve  # 近似曲面
  feature_func = feature_func  # 特徴量抽出関数
  fs = fs  # サンプリング周波数
  window_ms = window_ms    # ウィンドウ幅 [ms]
  window_size = int(fs * (window_ms / 1000))  # サンプル数に変換
  threshold = threshold   # featureの最大 - 平均がこの値以上なら採用
  half_win = window_size // 2

  # 出力先
  valid_indices = []

  # ---------- 64チャネルすべてに対してピーク検出 ----------
  peak_mask = np.zeros(emg_data.shape[0], dtype=bool)

  for ch in range(64):
      rms_signal = np.sqrt(emg_data[:,ch]**2)
      peaks, _ = find_peaks(rms_signal, distance=50,
                        height=np.mean(rms_signal) + np.std(rms_signal))
      # peaks, _ = find_peaks(emg_data[:, ch], distance=window_size//2, height=np.std(emg_data[:,ch]) * 1)
      peak_mask[peaks] = True  # どこか1チャネルでもピークがあればTrue

  # 全体でのピーク位置で条件を評価
  for t in np.where(peak_mask)[0]:
      if t - half_win < 0 or t + half_win >= emg_data.shape[0]:
          continue  # ウィンドウが境界を超えるならスキップ

      snippet = emg_data[t - half_win : t + half_win, :]  # shape: [window_size, 64]
      if func_type:
          feature_per_ch = feature_func(snippet, fs=fs)  # 各チャネルのfeature
      else:
          feature_per_ch = feature_func(snippet)  # 各チャネルのfeature
      feature_mean = np.mean(feature_per_ch)
      feature_max = np.max(feature_per_ch)

      if feature_max - feature_mean >= threshold:
          valid_indices.append(t)

  print(f"検出されたピーク数（条件を満たすもの）: {len(valid_indices)}")

  # MUAP波形の切り出し
  snippets = []
  valid_peaks = []
  for peak in valid_indices:
      if peak - window_size//2 >= 0 and peak + window_size//2 < emg_data.shape[0]:
          snippet = emg_data[peak - window_size//2 : peak + window_size//2, :]
          snippets.append(snippet)
          valid_peaks.append(peak)

  snippets = np.array(snippets)
  valid_peaks = np.array(valid_peaks)


  # 座標グリッド
  x = np.arange(8)
  y = np.arange(8)
  xv, yv = np.meshgrid(x, y)
  coords = np.vstack((xv.ravel(), yv.ravel()))

  centers = []
  directions = []
  theta_1s = []
  features = []
  for snippet in snippets[:]:
    try:
      segment = snippet
      #ピーク間振幅
      if func_type:
          feature = feature_func(segment, fs=fs) #shape:(64,)
      else:
          feature = feature_func(segment) #shape:(64,)
      map_2d = feature.reshape(8, 8)

      # フィッティング
      max_index = np.unravel_index(np.argmax(map_2d), map_2d.shape)
      initial = [np.max(feature)-np.min(feature), max_index[0], max_index[1], 1, 1, 0, np.min(feature)]
      bounds = ([0, 0, 0, 0.1, 0.1, -np.pi/2, -np.inf],
              [np.inf, 7, 7, 5, 5, np.pi/2, np.inf])
      popt, _ = curve_fit(curve, coords, feature, p0=initial, bounds=bounds, maxfev=5000)
      A, x0, y0, sigma_x, sigma_y, theta, offset = popt #パラメータ取得
      center = (x0, y0)
      if sigma_x > sigma_y:
        theta_1 = ninety2ninety_radian(theta)
        # print('σx > σy')
      else:
        theta_1 = ninety2ninety_radian(theta + np.pi/2)
        # print('σy >= σx')
      direction = (np.cos(theta_1), np.sin(theta_1))

      centers.append(center)
      directions.append(direction)
      theta_1s.append(np.degrees(theta_1))

      feature = []
      feature.append(center[0])
      feature.append(center[1])
      feature.append(np.degrees(theta_1))
      features.append(feature)
    except RuntimeError:
       pass

  return features

# gmm
def fit_gmm_auto(X, max_components=5, criterion='bic', covariance_type='full', sample_weight=None, upsample_factor=5, map_visualize=True, criterion_visualize=True):
    lowest_crit = np.inf
    best_gmm = None
    crits = []

    if sample_weight is not None:
      p = sample_weight / sample_weight.sum()
      n_draw = max(len(X), upsample_factor * len(X))
      idx = np.random.choice(len(X), size=n_draw, replace=True, p=p)
      X_fit = X[idx]

      if map_visualize:
          x = X_fit[:, 0]
          y = X_fit[:, 1]
          plt.figure(figsize=(6, 6))
          plt.hist2d(x, y, bins=70, cmap='Blues')
          plt.colorbar(label='Counts')
          # plt.xticks(np.arange(0, 71, 1))
          # plt.yticks(np.arange(0, 71, 1))
          plt.xlim(0, 70)
          plt.ylim(0, 70)
          plt.xlabel('X')
          plt.ylabel('Y')
          plt.title('2D Histogram of Upsampled Data for GMM Fitting')
          plt.show()

    for n in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n, covariance_type=covariance_type, random_state=0)
        if sample_weight is not None:
          gmm.fit(X_fit)
        else:
          gmm.fit(X)

        crit_val = gmm.bic(X) if criterion == 'bic' else gmm.aic(X)
        crits.append(crit_val)

        if crit_val < lowest_crit:
            lowest_crit = crit_val
            best_gmm = gmm

    if criterion_visualize:
      # 可視化（任意）
      plt.plot(range(1, max_components + 1), crits, marker='o')
      plt.xlabel("Number of components")
      plt.ylabel(criterion.upper())
      plt.title(f"{criterion.upper()} for GMM model selection")
      plt.grid(True)
      plt.show()
      print(f"Selected n_components = {best_gmm.n_components}")

    return best_gmm

def gmm(emg_data, curve, feature_func, fs=2048, window_ms=25, threshold=0, percent=95, max_components=4, criterion='bic', upsample_factor=5, func_type=True):
    # ---------- パラメータ設定 ----------
    emg_data = emg_data  # shape: (n_samples, n_channels)
    curve = curve  # 近似曲面
    feature_func = feature_func  # 特徴量抽出関数
    fs = fs  # サンプリング周波数
    window_ms = window_ms    # ウィンドウ幅 [ms]
    window_size = int(fs * (window_ms / 1000))  # サンプル数に変換
    threshold = threshold   # featureの最大 - 平均がこの値以上なら採用
    half_win = window_size // 2

    # ---------- 出力先 ----------
    valid_indices = []

    # ---------- 64チャネルすべてに対してピーク検出 ----------
    peak_mask = np.zeros(emg_data.shape[0], dtype=bool)

    for ch in range(64):
        rms_signal = np.sqrt(emg_data[:,ch]**2)
        peaks, _ = find_peaks(rms_signal, distance=50,
                        height=np.mean(rms_signal) + np.std(rms_signal))
        # peaks, _ = find_peaks(emg_data[:, ch], distance=window_size//2, height=np.std(emg_data[:,ch]) * 1)
        peak_mask[peaks] = True  # どこか1チャネルでもピークがあればTrue

    # ---------- 全体でのピーク位置で条件を評価 ----------
    for t in np.where(peak_mask)[0]:
        if t - half_win < 0 or t + half_win >= emg_data.shape[0]:
            continue  # ウィンドウが境界を超えるならスキップ

        snippet = emg_data[t - half_win : t + half_win, :]  # shape: [window_size, 64]
        if func_type:
            feature_per_ch = feature_func(snippet, fs=fs)  # 各チャネルのfeature
        else:
            feature_per_ch = feature_func(snippet)  # 各チャネルのfeature
        feature_mean = np.mean(feature_per_ch)
        feature_max = np.max(feature_per_ch)

        if feature_max - feature_mean >= threshold:
            valid_indices.append(t)

    print(f"検出されたピーク数（条件を満たすもの）: {len(valid_indices)}")

    # ----- MUAP波形の切り出し -----
    snippets = []
    valid_peaks = []
    for peak in valid_indices:
        if peak - window_size//2 >= 0 and peak + window_size//2 < emg_data.shape[0]:
            snippet = emg_data[peak - window_size//2 : peak + window_size//2, :]
            snippets.append(snippet)
            valid_peaks.append(peak)

    snippets = np.array(snippets)
    valid_peaks = np.array(valid_peaks)


    # 座標グリッド
    x = np.arange(8)
    y = np.arange(8)
    xv, yv = np.meshgrid(x, y)
    coords = np.vstack((xv.ravel(), yv.ravel()))

    # ==== 描画用補間座標 ====
    n_interp = 10  # チャネル間の補間数（1チャネルを10分割）
    x_interp = np.linspace(0, 7, (8 - 1) * n_interp + 1)
    y_interp = np.linspace(0, 7, (8 - 1) * n_interp + 1)
    x_mesh, y_mesh = np.meshgrid(x_interp, y_interp)
    coords_interp = np.vstack((x_mesh.ravel(), y_mesh.ravel()))  # shape: (2, N)


    features = []
    for snippet in snippets[:]:
        try:
            segment = snippet
            if func_type:
                feature = feature_func(segment, fs=fs) #shape:(64,)
            else:
                feature = feature_func(segment) #shape:(64,)
            map_2d = feature.reshape(8, 8)

            interp_spline = RectBivariateSpline(y, x, map_2d)
            map_2d = interp_spline(y_interp, x_interp)  # shape: (Nx, Ny)

            percent = percent
            gmm_coords = np.column_stack(np.where(map_2d >= np.percentile(map_2d, percent)))[:,::-1]
            weights = map_2d[map_2d >= np.percentile(map_2d, percent)].flatten()

            # GMMを自動選定でフィット
            gmm = fit_gmm_auto(gmm_coords, max_components=max_components, criterion=criterion, sample_weight=weights, upsample_factor=upsample_factor, map_visualize=False, criterion_visualize=False)


            for n, (mean, cov) in enumerate(zip(gmm.means_, gmm.covariances_)):
                center = mean / n_interp  # (x, y)
                # print(f'mean:{mean}, center:{center}')
                eigvals, eigvecs = np.linalg.eigh(cov)
                main_dir = eigvecs[:, np.argmax(eigvals)]
                angle = np.arctan2(main_dir[1], main_dir[0])  # 方向（ラジアン）
                angle = ninety2ninety_radian(angle)

                feature = []
                feature.append(center[0])
                feature.append(center[1])
                feature.append(np.degrees(angle))
                features.append(feature)
        except ValueError:
            pass

    return features

# ----- クラスタリング -----
# kmeansクラスタリング
def kmeans_clustering(features, k1=4, k2=3):
    # 1. 特徴ベクトルを構築
    features = np.array(features)
    results_df = pd.DataFrame(data=features, columns=['center_x', 'center_y', 'theta_deg'])

    # --- 第1段階：中心座標でクラスタリング ---
    # 特徴量：center_x, center_y
    center_features = results_df[['center_x', 'center_y']].dropna()
    k1 = k1  # 中心クラスタ数（例）
    kmeans1 = KMeans(n_clusters=k1, random_state=0)
    center_labels = kmeans1.fit_predict(center_features)

    # 結果に追加
    results_df['center_cluster'] = -1
    results_df.loc[center_features.index, 'center_cluster'] = center_labels

    # # --- 第2段階：方向角でクラスタリング（各中心クラスタ内で） ---
    # # θの周期性を考慮 → sinθ, cosθでクラスタリング
    # results_df['theta_sin'] = np.sin(results_df['theta_rad'])
    # results_df['theta_cos'] = np.cos(results_df['theta_rad'])

    direction_cluster_labels = np.full(len(results_df), -1)  # 初期化

    k2 = k2  # 各中心クラスタ内の方向クラスタ数（例）

    for group_id in range(k1):
        group_df = results_df[results_df['center_cluster'] == group_id]
        idx = group_df.index
        if len(group_df) >= k2:  # クラスタ数以上あるか確認
            dir_features = group_df[['theta_deg']].values
            kmeans2 = KMeans(n_clusters=k2, random_state=0)  # 固定乱数
            sub_labels = kmeans2.fit_predict(dir_features)
            direction_cluster_labels[idx] = sub_labels + group_id * 10  # 固有ラベル化

    # 結果に追加
    results_df['direction_cluster'] = direction_cluster_labels
    results_df = results_df[results_df['direction_cluster'] != -1].reset_index(drop=True)


    # クラスタごとに平均・標準偏差を集計
    cluster_stats = results_df.groupby('direction_cluster').agg({
        'center_x': ['mean', 'std'],
        'center_y': ['mean', 'std'],
        'theta_deg': ['mean', 'std'],
        'direction_cluster': 'count'
    })

    # 結果をまとめ直す
    summary_df = pd.DataFrame({
        'cluster': cluster_stats.index,
        'center_x_mean': cluster_stats[('center_x', 'mean')],
        'center_x_std': cluster_stats[('center_x', 'std')],
        'center_y_mean': cluster_stats[('center_y', 'mean')],
        'center_y_std': cluster_stats[('center_y', 'std')],
        'theta_deg_mean': cluster_stats[('theta_deg', 'mean')],
        'theta_deg_std': cluster_stats[('theta_deg', 'std')],
        # 'count': results_df['cluster'].value_counts().sort_index(),
        'count': cluster_stats[('direction_cluster', 'count')]
    }).reset_index(drop=True)

    # 結果表示
    summary_df

    return results_df, summary_df

# normalizeを使ったkmeansクラスタリング
def normalized_kmeans_clustering(features, k1=4):
    # 1. 特徴ベクトルを構築
    features = np.array(features)
    results_df = pd.DataFrame(data=features, columns=['center_x', 'center_y', 'theta_deg'])

    # --- 第1段階：中心座標でクラスタリング ---
    # 特徴量：
    features = results_df[['center_x', 'center_y', 'theta_deg']].dropna()
    #正規化のクラスを準備
    ms = MinMaxScaler()
    #特徴量の最大値と最小値を計算し変換
    features = ms.fit_transform(features)
    features = pd.DataFrame(features, columns=['center_x', 'center_y', 'theta_deg'])
    k1 = k1  # 中心クラスタ数（例）
    kmeans1 = KMeans(n_clusters=k1, random_state=0)
    center_labels = kmeans1.fit_predict(features)

    # 結果に追加
    results_df['direction_cluster'] = -1
    results_df.loc[features.index, 'direction_cluster'] = center_labels

    # クラスタごとに平均・標準偏差を集計
    cluster_stats = results_df.groupby('direction_cluster').agg({
        'center_x': ['mean', 'std'],
        'center_y': ['mean', 'std'],
        'theta_deg': ['mean', 'std'],
        'direction_cluster': 'count'
    })

    # 結果をまとめ直す
    summary_df = pd.DataFrame({
        'cluster': cluster_stats.index,
        'center_x_mean': cluster_stats[('center_x', 'mean')],
        'center_x_std': cluster_stats[('center_x', 'std')],
        'center_y_mean': cluster_stats[('center_y', 'mean')],
        'center_y_std': cluster_stats[('center_y', 'std')],
        'theta_deg_mean': cluster_stats[('theta_deg', 'mean')],
        'theta_deg_std': cluster_stats[('theta_deg', 'std')],
        # 'count': results_df['cluster'].value_counts().sort_index(),
        'count': cluster_stats[('direction_cluster', 'count')]
    }).reset_index(drop=True)

    # 結果表示
    summary_df

    return results_df, summary_df

from sklearn.preprocessing import MinMaxScaler

# xmeans用ラベル変換関数
def indices_to_labels(
    index_lists: Sequence[Sequence[int]],
    num_samples: Optional[int] = None,
    class_names: Optional[Sequence[Any]] = None,
    fill: Any = None,
    on_conflict: str = "error",  # "error" or "last_wins"
) -> List[Any]:

    # クラス名の既定は 0..C-1
    if class_names is None:
        class_names = list(range(len(index_lists)))
    else:
        if len(class_names) != len(index_lists):
            raise ValueError("class_names の長さは index_lists と一致させてください。")

    # 配列長を自動推定（最大インデックス+1）
    if num_samples is None:
        max_idx = -1
        for idxs in index_lists:
            if idxs:
                m = max(idxs)
                if m > max_idx:
                    max_idx = m
        num_samples = max_idx + 1 if max_idx >= 0 else 0

    labels: List[Any] = [fill] * num_samples

    for c, idxs in zip(class_names, index_lists):
        for i in idxs:
            if i < 0 or i >= num_samples:
                raise IndexError(f"インデックス {i} が範囲外です (0..{num_samples-1}).")
            if labels[i] is not None and labels[i] != fill and labels[i] != c:
                if on_conflict == "error":
                    raise ValueError(f"インデックス {i} が複数クラスに割り当てられています: "
                                     f"{labels[i]} と {c}")
                elif on_conflict == "last_wins":
                    labels[i] = c
                else:
                    raise ValueError("on_conflict は 'error' か 'last_wins' を指定してください。")
            else:
                labels[i] = c
    return labels

# xmeansクラスタリング
def xmeans_clustering(features, kmax=5):
    # 1. 特徴ベクトルを構築
    features = np.array(features)
    results_df = pd.DataFrame(data=features, columns=['center_x', 'center_y', 'theta_deg'])

    # --- 第1段階：中心座標でクラスタリング ---
    # 特徴量：
    center_features = results_df[['center_x', 'center_y']].dropna()
    # ----- X-meansクラスタリング -----
    initial_centers = kmeans_plusplus_initializer(center_features, 2).initialize()
    xmeans_instance = xmeans(
        data=center_features,
        initial_centers=initial_centers,
        kmax=kmax,  # 最大クラスタ数（必要に応じて調整）
        ccore=True,
        metric=distance_metric(type_metric.EUCLIDEAN)
    )
    xmeans_instance.process()

    clusters = xmeans_instance.get_clusters()
    centers = np.array(xmeans_instance.get_centers())
    num_clusters = len(clusters)
    center_labels = indices_to_labels(clusters)

    # 結果に追加
    results_df['center_cluster'] = -1
    results_df.loc[center_features.index, 'center_cluster'] = center_labels

    direction_cluster_labels = np.full(len(results_df), -1)  # 初期化

    for group_id in range(num_clusters):
        group_df = results_df[results_df['center_cluster'] == group_id]
        if len(group_df) >= 2:
            idx = group_df.index
            dir_features = group_df[['theta_deg']].values
            initial_centers = kmeans_plusplus_initializer(dir_features, 2).initialize()
            xmeans_instance = xmeans(
                data=dir_features,
                initial_centers=initial_centers,
                kmax=kmax,  # 最大クラスタ数（必要に応じて調整）
                ccore=True,
                metric=distance_metric(type_metric.EUCLIDEAN)
            )
            xmeans_instance.process()

            clusters = xmeans_instance.get_clusters()
            centers = np.array(xmeans_instance.get_centers())
            num_clusters = len(clusters)
            sub_labels = np.array(indices_to_labels(clusters))
            direction_cluster_labels[idx] = sub_labels + group_id * 10  # 固有ラベル化

    # 結果に追加
    results_df['direction_cluster'] = direction_cluster_labels
    results_df = results_df[results_df['direction_cluster'] != -1].reset_index(drop=True)

    # クラスタごとに平均・標準偏差を集計
    cluster_stats = results_df.groupby('direction_cluster').agg({
        'center_x': ['mean', 'std'],
        'center_y': ['mean', 'std'],
        'theta_deg': ['mean', 'std'],
        'direction_cluster': 'count'
    })

    # 結果をまとめ直す
    summary_df = pd.DataFrame({
        'cluster': cluster_stats.index,
        'center_x_mean': cluster_stats[('center_x', 'mean')],
        'center_x_std': cluster_stats[('center_x', 'std')],
        'center_y_mean': cluster_stats[('center_y', 'mean')],
        'center_y_std': cluster_stats[('center_y', 'std')],
        'theta_deg_mean': cluster_stats[('theta_deg', 'mean')],
        'theta_deg_std': cluster_stats[('theta_deg', 'std')],
        # 'count': results_df['cluster'].value_counts().sort_index(),
        'count': cluster_stats[('direction_cluster', 'count')]
    }).reset_index(drop=True)

    # 結果表示
    summary_df

    return results_df, summary_df

# normalizeを使ったxmeansクラスタリング
def normalized_xmeans_clustering(features, kmax=5):
    # 1. 特徴ベクトルを構築
    features = np.array(features)
    results_df = pd.DataFrame(data=features, columns=['center_x', 'center_y', 'theta_deg'])

    # --- 第1段階：中心座標でクラスタリング ---
    # 特徴量：
    features = results_df[['center_x', 'center_y', 'theta_deg']].dropna()
    #正規化のクラスを準備
    ms = MinMaxScaler()
    #特徴量の最大値と最小値を計算し変換
    features = ms.fit_transform(features)
    features = pd.DataFrame(features, columns=['center_x', 'center_y', 'theta_deg'])
    # ----- X-meansクラスタリング -----
    initial_centers = kmeans_plusplus_initializer(features, 2).initialize()
    xmeans_instance = xmeans(
        data=features,
        initial_centers=initial_centers,
        kmax=kmax,  # 最大クラスタ数（必要に応じて調整）
        ccore=True,
        metric=distance_metric(type_metric.EUCLIDEAN)
    )
    xmeans_instance.process()

    clusters = xmeans_instance.get_clusters()
    centers = np.array(xmeans_instance.get_centers())
    num_clusters = len(clusters)
    center_labels = indices_to_labels(clusters)

    # 結果に追加
    results_df['direction_cluster'] = -1
    results_df.loc[features.index, 'direction_cluster'] = center_labels

    # クラスタごとに平均・標準偏差を集計
    cluster_stats = results_df.groupby('direction_cluster').agg({
        'center_x': ['mean', 'std'],
        'center_y': ['mean', 'std'],
        'theta_deg': ['mean', 'std'],
        'direction_cluster': 'count'
    })

    # 結果をまとめ直す
    summary_df = pd.DataFrame({
        'cluster': cluster_stats.index,
        'center_x_mean': cluster_stats[('center_x', 'mean')],
        'center_x_std': cluster_stats[('center_x', 'std')],
        'center_y_mean': cluster_stats[('center_y', 'mean')],
        'center_y_std': cluster_stats[('center_y', 'std')],
        'theta_deg_mean': cluster_stats[('theta_deg', 'mean')],
        'theta_deg_std': cluster_stats[('theta_deg', 'std')],
        # 'count': results_df['cluster'].value_counts().sort_index(),
        'count': cluster_stats[('direction_cluster', 'count')]
    }).reset_index(drop=True)

    # 結果表示
    summary_df

    return results_df, summary_df

# HDBSCAN
def run_hdbscan(X, min_cluster_size=10, min_samples=None):
    clus = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                           min_samples=min_samples)
    labels = clus.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return {'labels': labels, 'n_clusters': int(n_clusters), 'probabilities': clus.probabilities_}

# hdbscanクラスタリング
def hdbscan_clustering(features, min_cluster_size=10, min_samples=None):
    # 1. 特徴ベクトルを構築
    features = np.array(features)
    results_df = pd.DataFrame(data=features, columns=['center_x', 'center_y', 'theta_deg'])

    # --- 第1段階：中心座標でクラスタリング ---
    # 特徴量：
    center_features = results_df[['center_x', 'center_y']].dropna()
    # ----- hdbscan -----
    results = run_hdbscan(center_features, min_cluster_size=min_cluster_size, min_samples=min_samples)
    center_labels = results['labels']
    n_clusters = results['n_clusters']

    # 結果に追加
    results_df['center_cluster'] = -1
    results_df.loc[center_features.index, 'center_cluster'] = center_labels
    results_df = results_df[results_df['center_cluster'] != -1].reset_index(drop=True)

    direction_cluster_labels = np.full(len(results_df), -1)  # 初期化

    for group_id in range(n_clusters):
        group_df = results_df[results_df['center_cluster'] == group_id]
        if len(group_df) >= min_cluster_size:
            idx = group_df.index
            dir_features = group_df[['theta_deg']].values
            results = run_hdbscan(dir_features, min_cluster_size=min_cluster_size, min_samples=min_samples)
            sub_labels = results['labels']
            for i, sub_label in zip(idx, sub_labels):
                if sub_label != -1:
                    direction_cluster_labels[i] = sub_label + group_id * 10  # 固有ラベル化

    # 結果に追加
    results_df['direction_cluster'] = direction_cluster_labels
    results_df = results_df[results_df['direction_cluster'] != -1].reset_index(drop=True)

    # クラスタごとに平均・標準偏差を集計
    cluster_stats = results_df.groupby('direction_cluster').agg({
        'center_x': ['mean', 'std'],
        'center_y': ['mean', 'std'],
        'theta_deg': ['mean', 'std'],
        'direction_cluster': 'count'
    })

    # 結果をまとめ直す
    summary_df = pd.DataFrame({
        'cluster': cluster_stats.index,
        'center_x_mean': cluster_stats[('center_x', 'mean')],
        'center_x_std': cluster_stats[('center_x', 'std')],
        'center_y_mean': cluster_stats[('center_y', 'mean')],
        'center_y_std': cluster_stats[('center_y', 'std')],
        'theta_deg_mean': cluster_stats[('theta_deg', 'mean')],
        'theta_deg_std': cluster_stats[('theta_deg', 'std')],
        # 'count': results_df['cluster'].value_counts().sort_index(),
        'count': cluster_stats[('direction_cluster', 'count')]
    }).reset_index(drop=True)

    # 結果表示
    summary_df

    return results_df, summary_df

# normalizeを使ったhdbscanクラスタリング
def normalized_hdbscan_clustering(features, min_cluster_size=10, min_samples=None):
    # 1. 特徴ベクトルを構築
    features = np.array(features)
    results_df = pd.DataFrame(data=features, columns=['center_x', 'center_y', 'theta_deg'])

    # --- 第1段階：中心座標でクラスタリング ---
    # 特徴量：
    features = results_df[['center_x', 'center_y', 'theta_deg']].dropna()
    #正規化のクラスを準備
    ms = MinMaxScaler()
    #特徴量の最大値と最小値を計算し変換
    features = ms.fit_transform(features)
    features = pd.DataFrame(features, columns=['center_x', 'center_y', 'theta_deg'])
    # ----- hdbscan -----
    results = run_hdbscan(features, min_cluster_size=min_cluster_size, min_samples=min_samples)
    center_labels = results['labels']

    # 結果に追加
    results_df['direction_cluster'] = -1
    results_df.loc[features.index, 'direction_cluster'] = center_labels
    results_df = results_df[results_df['direction_cluster'] != -1]

    # クラスタごとに平均・標準偏差を集計
    cluster_stats = results_df.groupby('direction_cluster').agg({
        'center_x': ['mean', 'std'],
        'center_y': ['mean', 'std'],
        'theta_deg': ['mean', 'std'],
        'direction_cluster': 'count'
    })

    # 結果をまとめ直す
    summary_df = pd.DataFrame({
        'cluster': cluster_stats.index,
        'center_x_mean': cluster_stats[('center_x', 'mean')],
        'center_x_std': cluster_stats[('center_x', 'std')],
        'center_y_mean': cluster_stats[('center_y', 'mean')],
        'center_y_std': cluster_stats[('center_y', 'std')],
        'theta_deg_mean': cluster_stats[('theta_deg', 'mean')],
        'theta_deg_std': cluster_stats[('theta_deg', 'std')],
        # 'count': results_df['cluster'].value_counts().sort_index(),
        'count': cluster_stats[('direction_cluster', 'count')]
    }).reset_index(drop=True)

    # 結果表示
    summary_df

    return results_df, summary_df

# クラスタ数に応じた色リスト取得関数
def get_cluster_colors(n_clusters, cmap_name='tab20'):
    cmap = cm.get_cmap(cmap_name, n_clusters)  # 'tab20', 'nipy_spectral', etc.
    return [cmap(i) for i in range(n_clusters)]

# ----- 筋活動情報取得関数 -----
def get_virtual_bipolars(results_df, ied=2, arrow_scale=0.5, show_plot=True):
  #仮想双極電極の電極間距離（cm）
  a = ied

  if show_plot:
    n_clusters = results_df['direction_cluster'].nunique() #
    cluster_colors = get_cluster_colors(n_clusters) #

  virtual_bipolars = []
  labels = []
  center_direction = []
  n_virtual_bipolars_checker = []
  if show_plot:
    plt.figure(figsize=(7, 6)) #
  for n, direction_cluster in enumerate(sorted(results_df['direction_cluster'].unique())):
    sub_df = results_df[results_df['direction_cluster'] == direction_cluster]
    # if sub_df['center_cluster'].count() >= 20 and sub_df['center_x'].std() < 1 and sub_df['center_y'].std() < 1 and sub_df['theta_deg'].std() < 20:
    x, y = sub_df['center_x'].mean(), sub_df['center_y'].mean()
    theta = sub_df['theta_deg'].mean()
    dx, dy = np.cos(np.radians(theta)) * arrow_scale, np.sin(np.radians(theta)) * arrow_scale #
    # print(f'cluster:{direction_cluster}, x={x}, y={y}, θ={theta}')
    x1 = x - a/2*np.cos(np.radians(theta))
    y1 = y - a/2*np.sin(np.radians(theta))
    x2 = x + a/2*np.cos(np.radians(theta))
    y2 = y + a/2*np.sin(np.radians(theta))
    virtual_bipolars.append([x1, y1, x2, y2])
    labels.append(direction_cluster)
    center_direction.append([x, y, theta])
    n_virtual_bipolars_checker.append(True)
    if show_plot:
      # 色で中心クラスタを、マーカーサイズで方向クラスタを示す
      color = cluster_colors[n]
      plt.arrow(x, y, dx, dy, head_width=0.15, color=color, alpha=0.8)
      plt.plot(x, y, 'o', color=color,
                label=f"Center Cl {direction_cluster}" if f"Center Cl {direction_cluster}" not in plt.gca().get_legend_handles_labels()[1] else "")
  # if len(n_virtual_bipolars_checker) == 1:
  #   virtual_bipolars.append([0,0,0,0])
  #   labels.append(999)
  #   center_direction.append([0,0,0])
  if show_plot:
    # ラベル・軸
    plt.xlabel("x")
    plt.ylabel("y")
    # plt.title("2-Stage Clustering of Muscle Fiber Direction")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.axis('equal')
    plt.tight_layout()
    plt.xticks(np.arange(0, 8, 1))
    plt.yticks(np.arange(0, 8, 1))
    plt.xlim(0, 7)
    plt.ylim(0, 7)
    plt.show()

  return virtual_bipolars, labels, center_direction, len(n_virtual_bipolars_checker)


# # csvファイル保存用の関数
# FIELDS = [
#     'file_name', 'gesture', 'trial', 'subject', 'session',
#     'electrode_place', 'emg_data', 'virtual_bipolars',
#     'labels', 'center_direction', 'n_virtual_bipolars'
# ]

# def _to_jsonable(x):
#     """CSVに入れる前段階としてJSON化可能な素に変換（NumPy対応）。"""
#     if isinstance(x, np.ndarray):
#         return x.tolist()
#     if isinstance(x, (np.integer, np.floating, np.bool_)):
#         return x.item()
#     if isinstance(x, Mapping):
#         return {k: _to_jsonable(v) for k, v in x.items()}
#     if isinstance(x, (list, tuple)):
#         return [_to_jsonable(v) for v in x]
#     return x  # 文字列/数値/None などはそのまま

# def _serialize_for_csv(x):
#     """リスト/辞書/タプルはJSON文字列に、その他はそのまま返す。"""
#     x = _to_jsonable(x)
#     if isinstance(x, (list, dict, tuple)):
#         return json.dumps(x, ensure_ascii=False, separators=(',', ':'))
#     return x

# def save_records_to_csv(records, out_path, fields=FIELDS):
#     """
#     records: 上記フォーマットの辞書のリスト
#     out_path: 保存パス（例: 'output/records.csv'）
#     fields: 列順（recordsに無いキーは空欄、余分なキーは末尾に追加）
#     """
#     rows = []
#     for rec in records:
#         row = {k: _serialize_for_csv(rec.get(k, None)) for k in rec.keys()}
#         # 列順を固定したい場合に欠損キーを補完
#         for k in fields:
#             if k not in row:
#                 row[k] = None
#         rows.append(row)

#     # 列順をfields優先に（未知のキーがあれば後ろに付ける）
#     extra_cols = [c for c in rows[0].keys() if c not in fields] if rows else []
#     columns = list(fields) + extra_cols

#     df = pd.DataFrame(rows, columns=columns)
#     Path(out_path).parent.mkdir(parents=True, exist_ok=True)
#     df.to_csv(out_path, index=False, encoding='utf-8-sig')
#     return out_path

# 標準化・正規化
def scaler(vpolars):
    x_list = []
    y_list = []
    theta_list = []
    for vpolar_list in vpolars:
        for vpolar in vpolar_list['center_direction']:
            x_list.append(vpolar[0])
            y_list.append(vpolar[1])
            theta_list.append(vpolar[2])
            # print(f"x: {vpolar[0]}, y: {vpolar[1]}, θ: {vpolar[2]}")
    # 標準化
    x_mean = np.mean(x_list)
    y_mean = np.mean(y_list)
    theta_mean = np.mean(theta_list)
    x_std = np.std(x_list)
    y_std = np.std(y_list)
    theta_std = np.std(theta_list)
    # 正規化
    x_max = np.max(x_list)
    x_min = np.min(x_list)
    y_max = np.max(y_list)
    y_min = np.min(y_list)
    theta_max = np.max(theta_list)
    theta_min = np.min(theta_list)
    return x_mean, y_mean, theta_mean, x_std, y_std, theta_std, x_max, x_min, y_max, y_min, theta_max, theta_min

def distances_across_sessions(vpolars, x_std, y_std, theta_std, x_max, x_min, y_max, y_min, theta_max, theta_min, scaling=2):
    distances = []
    for list_session1 in vpolars:
        for list_session2 in vpolars:
            if list_session1['subject'] == list_session2['subject'] and list_session1['session'] == 1 and list_session2['session'] == 2 and list_session1['gesture'] == list_session2['gesture'] and list_session1['trial'] == list_session2['trial'] and list_session1['electrode_place'] == list_session2['electrode_place']:
                # print(f'file_name: {list_session1["file_name"]}, gesture: {list_session1["gesture"]}, trial: {list_session1["trial"]}, subject: {list_session1["subject"]}, session: {list_session1["session"]}')
                # print(f'file_name: {list_session2["file_name"]}, gesture: {list_session2["gesture"]}, trial: {list_session2["trial"]}, subject: {list_session2["subject"]}, session: {list_session2["session"]}')
                x_distance = {}
                y_distance = {}
                theta_distance = {}
                result_distance = []
                if list_session1['n_virtual_bipolars'] > 0 and list_session2['n_virtual_bipolars'] > 0:
                    for i in range(len(list_session1['center_direction'])):
                        x_distance[i] = []
                        y_distance[i] = []
                        theta_distance[i] = []
                        for j in range(len(list_session2['center_direction'])):
                            x_distance[i].append(list_session2['center_direction'][j][0] - list_session1['center_direction'][i][0])
                            y_distance[i].append(list_session2['center_direction'][j][1] - list_session1['center_direction'][i][1])
                            theta_distance[i].append(list_session2['center_direction'][j][2] - list_session1['center_direction'][i][2])
                        # print(f'j={j},x_distance[{i}]: {x_distance[i]}, y_distance[{i}]: {y_distance[i]}, theta_distance[{i}]: {theta_distance[i]}')
                        if scaling == 0:
                            xy_distance = np.sqrt(np.array(x_distance[i])**2 + np.array(y_distance[i])**2)
                        elif scaling == 1:
                            # 標準化
                            x_norm = np.array(x_distance[i]) / x_std
                            y_norm = np.array(y_distance[i]) / y_std
                            theta_norm = np.array(theta_distance[i]) / theta_std
                            xy_distance = np.sqrt(x_norm**2 + y_norm**2 + theta_norm**2) # θの距離も考慮
                        elif scaling == 2:
                            # min-max正規化
                            x_norm = np.array(x_distance[i]) / (x_max - x_min)
                            y_norm = np.array(y_distance[i]) / (y_max - y_min)
                            theta_norm = np.array(theta_distance[i]) / (theta_max - theta_min)
                            xy_distance = np.sqrt(x_norm**2 + y_norm**2 + theta_norm**2) # θの距離も考慮
                        session2_id = np.argmin(xy_distance) #最小値のインデックス
                        result_distance.append({
                            'session1_cluster': list_session1['labels'][i],
                            'session2_cluster': list_session2['labels'][session2_id],
                            'x_distance': x_distance[i][session2_id],
                            'y_distance': y_distance[i][session2_id],
                            'theta_distance': theta_distance[i][session2_id]
                        })
                        # print(f"subject={list_session1['subject']}, gesture={list_session1['gesture']}, trial={list_session1['trial']}, electrode_place={list_session1['electrode_place']},session1_id={i}, session2_id={session2_id}: session1_cluster={list_session1['labels'][i]}, session2_cluster={list_session2['labels'][session2_id]},x_distance={x_distance[i][session2_id]}, y_distance={y_distance[i][session2_id]}, theta_distance={theta_distance[i][session2_id]}")
                else:
                    result_distance.append(None)
                distances.append({'subject': list_session1['subject'], 'gesture': list_session1['gesture'], 'trial': list_session1['trial'], 'electrode_place': list_session1['electrode_place'], 'result_distance': result_distance})
    return distances

def distances_across_trials(vpolars, x_std, y_std, theta_std, x_max, x_min, y_max, y_min, theta_max, theta_min, scaling=2):
    distances_between_trials = []
    for list_1 in vpolars:
        for list_2 in vpolars:
            if list_1['subject'] == list_2['subject'] and list_1['session'] == list_2['session']  and list_1['gesture'] == list_2['gesture'] and list_1['trial'] == 1 and list_2['trial'] == 2 and list_1['electrode_place'] == list_2['electrode_place']:
                # print(f'file_name: {list_1["file_name"]}, gesture: {list_1["gesture"]}, trial: {list_1["trial"]}, subject: {list_1["subject"]}, session: {list_1["session"]}')
                # print(f'file_name: {list_2["file_name"]}, gesture: {list_2["gesture"]}, trial: {list_2["trial"]}, subject: {list_2["subject"]}, session: {list_2["session"]}')
                x_distance = {}
                y_distance = {}
                theta_distance = {}
                result_distance = []
                if list_1['n_virtual_bipolars'] > 0 and list_2['n_virtual_bipolars'] > 0:
                    for i in range(len(list_1['center_direction'])):
                        x_distance[i] = []
                        y_distance[i] = []
                        theta_distance[i] = []
                        for j in range(len(list_2['center_direction'])):
                            x_distance[i].append(list_2['center_direction'][j][0] - list_1['center_direction'][i][0])
                            y_distance[i].append(list_2['center_direction'][j][1] - list_1['center_direction'][i][1])
                            theta_distance[i].append(list_2['center_direction'][j][2] - list_1['center_direction'][i][2])
                        # print(f'j={j},x_distance[{i}]: {x_distance[i]}, y_distance[{i}]: {y_distance[i]}, theta_distance[{i}]: {theta_distance[i]}')
                        if scaling == 0:
                            xy_distance = np.sqrt(np.array(x_distance[i])**2 + np.array(y_distance[i])**2)
                        elif scaling == 1:
                            # 標準化
                            x_norm = np.array(x_distance[i]) / x_std
                            y_norm = np.array(y_distance[i]) / y_std
                            theta_norm = np.array(theta_distance[i]) / theta_std
                            xy_distance = np.sqrt(x_norm**2 + y_norm**2 + theta_norm**2) # θの距離も考慮
                        elif scaling == 2:
                            # min-max正規化
                            x_norm = np.array(x_distance[i]) / (x_max - x_min)
                            y_norm = np.array(y_distance[i]) / (y_max - y_min)
                            theta_norm = np.array(theta_distance[i]) / (theta_max - theta_min)
                            xy_distance = np.sqrt(x_norm**2 + y_norm**2 + theta_norm**2) # θの距離も考慮
                        session2_id = np.argmin(xy_distance) #最小値のインデックス
                        result_distance.append({
                            'session1_cluster': list_1['labels'][i],
                            'session2_cluster': list_2['labels'][session2_id],
                            'x_distance': x_distance[i][session2_id],
                            'y_distance': y_distance[i][session2_id],
                            'theta_distance': theta_distance[i][session2_id]
                        })
                        # print(f"subject={list_1['subject']}, gesture={list_1['gesture']}, trial={list_1['trial']}, electrode_place={list_1['electrode_place']},session1_id={i}, session2_id={session2_id}: session1_cluster={list_1['labels'][i]}, session2_cluster={list_2['labels'][session2_id]},x_distance={x_distance[i][session2_id]}, y_distance={y_distance[i][session2_id]}, theta_distance={theta_distance[i][session2_id]}")
                else:
                    result_distance.append(None)
                distances_between_trials.append({'subject': list_1['subject'], 'gesture': list_1['gesture'], 'session': list_1['session'], 'electrode_place': list_1['electrode_place'], 'result_distance': result_distance})
    return distances_between_trials

def distances_across_sessions_and_trials(vpolars, x_std, y_std, theta_std, x_max, x_min, y_max, y_min, theta_max, theta_min, scaling=2):
    distances_across_session_trials = []
    for list_1 in vpolars:
        for list_2 in vpolars:
            if list_1['subject'] == list_2['subject'] and list_1['session'] == 1 and list_2['session'] == 2  and list_1['gesture'] == list_2['gesture'] and list_1['trial'] != list_2['trial'] and list_1['electrode_place'] == list_2['electrode_place']:
                # print(f'file_name: {list_1["file_name"]}, gesture: {list_1["gesture"]}, trial: {list_1["trial"]}, subject: {list_1["subject"]}, session: {list_1["session"]}')
                # print(f'file_name: {list_2["file_name"]}, gesture: {list_2["gesture"]}, trial: {list_2["trial"]}, subject: {list_2["subject"]}, session: {list_2["session"]}')
                x_distance = {}
                y_distance = {}
                theta_distance = {}
                result_distance = []
                if list_1['n_virtual_bipolars'] > 0 and list_2['n_virtual_bipolars'] > 0:
                    for i in range(len(list_1['center_direction'])):
                        x_distance[i] = []
                        y_distance[i] = []
                        theta_distance[i] = []
                        for j in range(len(list_2['center_direction'])):
                            x_distance[i].append(list_2['center_direction'][j][0] - list_1['center_direction'][i][0])
                            y_distance[i].append(list_2['center_direction'][j][1] - list_1['center_direction'][i][1])
                            theta_distance[i].append(list_2['center_direction'][j][2] - list_1['center_direction'][i][2])
                        # print(f'j={j},x_distance[{i}]: {x_distance[i]}, y_distance[{i}]: {y_distance[i]}, theta_distance[{i}]: {theta_distance[i]}')
                        if scaling == 0:
                            xy_distance = np.sqrt(np.array(x_distance[i])**2 + np.array(y_distance[i])**2)
                        elif scaling == 1:
                            # 標準化
                            x_norm = np.array(x_distance[i]) / x_std
                            y_norm = np.array(y_distance[i]) / y_std
                            theta_norm = np.array(theta_distance[i]) / theta_std
                            xy_distance = np.sqrt(x_norm**2 + y_norm**2 + theta_norm**2) # θの距離も考慮
                        elif scaling == 2:
                            # min-max正規化
                            x_norm = np.array(x_distance[i]) / (x_max - x_min)
                            y_norm = np.array(y_distance[i]) / (y_max - y_min)
                            theta_norm = np.array(theta_distance[i]) / (theta_max - theta_min)
                            xy_distance = np.sqrt(x_norm**2 + y_norm**2 + theta_norm**2) # θの距離も考慮
                        session2_id = np.argmin(xy_distance) #最小値のインデックス
                        result_distance.append({
                            'session1_cluster': list_1['labels'][i],
                            'session2_cluster': list_2['labels'][session2_id],
                            'x_distance': x_distance[i][session2_id],
                            'y_distance': y_distance[i][session2_id],
                            'theta_distance': theta_distance[i][session2_id]
                        })
                        # print(f"subject={list_1['subject']}, gesture={list_1['gesture']}, trial={list_1['trial']}, electrode_place={list_1['electrode_place']},session1_id={i}, session2_id={session2_id}: session1_cluster={list_1['labels'][i]}, session2_cluster={list_2['labels'][session2_id]},x_distance={x_distance[i][session2_id]}, y_distance={y_distance[i][session2_id]}, theta_distance={theta_distance[i][session2_id]}")
                else:
                    result_distance.append(None)
                distances_across_session_trials.append({'subject': list_1['subject'], 'gesture': list_1['gesture'], 'session': list_1['session'], 'electrode_place': list_1['electrode_place'], 'result_distance': result_distance})
    return distances_across_session_trials

def df_maker(distances):
    n_sessions_list = []
    for distance in distances:
        n_sessions_list.append(distance['subject'])
    subjects = set(n_sessions_list)

    x_ED_gesture = {}
    y_ED_gesture = {}
    theta_ED_gesture = {}
    x_EP_gesture = {}
    y_EP_gesture = {}
    theta_EP_gesture = {}
    x_FD_gesture = {}
    y_FD_gesture = {}
    theta_FD_gesture = {}
    x_FP_gesture = {}
    y_FP_gesture = {}
    theta_FP_gesture = {}
    n_gestures = 34
    for subject in subjects:
        x_ED_gesture[subject] = {}
        y_ED_gesture[subject] = {}
        theta_ED_gesture[subject] = {}
        x_EP_gesture[subject] = {}
        y_EP_gesture[subject] = {}
        theta_EP_gesture[subject] = {}
        x_FD_gesture[subject] = {}
        y_FD_gesture[subject] = {}
        theta_FD_gesture[subject] = {}
        x_FP_gesture[subject] = {}
        y_FP_gesture[subject] = {}
        theta_FP_gesture[subject] = {}
        for i in range(n_gestures):
            x_ED_gesture[subject][i+1] = []
            y_ED_gesture[subject][i+1] = []
            theta_ED_gesture[subject][i+1] = []
            x_EP_gesture[subject][i+1] = []
            y_EP_gesture[subject][i+1] = []
            theta_EP_gesture[subject][i+1] = []
            x_FD_gesture[subject][i+1] = []
            y_FD_gesture[subject][i+1] = []
            theta_FD_gesture[subject][i+1] = []
            x_FP_gesture[subject][i+1] = []
            y_FP_gesture[subject][i+1] = []
            theta_FP_gesture[subject][i+1] = []
    for diff_list in distances:
        # print(diff_list)
        if diff_list['result_distance'] != [None]:
            subject = diff_list['subject']
            gesture = int(diff_list['gesture'])
            electrode_place = diff_list['electrode_place']
            if electrode_place == 'ED':
                for diff in diff_list['result_distance']:
                    x_ED_gesture[subject][gesture].append(diff['x_distance'])
                    y_ED_gesture[subject][gesture].append(diff['y_distance'])
                    theta_ED_gesture[subject][gesture].append(diff['theta_distance'])
            elif electrode_place == 'EP':
                for diff in diff_list['result_distance']:
                    x_EP_gesture[subject][gesture].append(diff['x_distance'])
                    y_EP_gesture[subject][gesture].append(diff['y_distance'])
                    theta_EP_gesture[subject][gesture].append(diff['theta_distance'])
            elif electrode_place == 'FD':
                for diff in diff_list['result_distance']:
                    x_FD_gesture[subject][gesture].append(diff['x_distance'])
                    y_FD_gesture[subject][gesture].append(diff['y_distance'])
                    theta_FD_gesture[subject][gesture].append(diff['theta_distance'])
            elif electrode_place == 'FP':
                for diff in diff_list['result_distance']:
                    x_FP_gesture[subject][gesture].append(diff['x_distance'])
                    y_FP_gesture[subject][gesture].append(diff['y_distance'])
                    theta_FP_gesture[subject][gesture].append(diff['theta_distance'])
    # 差分の絶対値
    x_gesture_mean = {}
    y_gesture_mean = {}
    theta_gesture_mean = {}
    for subject in subjects:
        x_gesture_mean[subject] = {'value':[], 'mean':0, 'std':0}
        y_gesture_mean[subject] = {'value':[], 'mean':0, 'std':0}
        theta_gesture_mean[subject] = {'value':[], 'mean':0, 'std':0}
        for i in range(n_gestures):
            for l in range(len(x_ED_gesture[subject][i+1])):
                x_gesture_mean[subject]['value'].append(np.abs(x_ED_gesture[subject][i+1][l]))
                y_gesture_mean[subject]['value'].append(np.abs(y_ED_gesture[subject][i+1][l]))
                theta_gesture_mean[subject]['value'].append(np.abs(theta_ED_gesture[subject][i+1][l]))
            for l in range(len(x_EP_gesture[subject][i+1])):
                x_gesture_mean[subject]['value'].append(np.abs(x_EP_gesture[subject][i+1][l]))
                y_gesture_mean[subject]['value'].append(np.abs(y_EP_gesture[subject][i+1][l]))
                theta_gesture_mean[subject]['value'].append(np.abs(theta_EP_gesture[subject][i+1][l]))
            for l in range(len(x_FD_gesture[subject][i+1])):
                x_gesture_mean[subject]['value'].append(np.abs(x_FD_gesture[subject][i+1][l]))
                y_gesture_mean[subject]['value'].append(np.abs(y_FD_gesture[subject][i+1][l]))
                theta_gesture_mean[subject]['value'].append(np.abs(theta_FD_gesture[subject][i+1][l]))
            for l in range(len(x_FP_gesture[subject][i+1])):
                x_gesture_mean[subject]['value'].append(np.abs(x_FP_gesture[subject][i+1][l]))
                y_gesture_mean[subject]['value'].append(np.abs(y_FP_gesture[subject][i+1][l]))
                theta_gesture_mean[subject]['value'].append(np.abs(theta_FP_gesture[subject][i+1][l]))
        x_gesture_mean[subject]['mean'] = np.mean(x_gesture_mean[subject]['value'])
        x_gesture_mean[subject]['std'] = np.std(x_gesture_mean[subject]['value'])
        y_gesture_mean[subject]['mean'] = np.mean(y_gesture_mean[subject]['value'])
        y_gesture_mean[subject]['std'] = np.std(y_gesture_mean[subject]['value'])
        theta_gesture_mean[subject]['mean'] = np.mean(theta_gesture_mean[subject]['value'])
        theta_gesture_mean[subject]['std'] = np.std(theta_gesture_mean[subject]['value'])
    # 表の作成
    rows = []
    for subject in subjects:
        rows.append([subject, 
                    x_gesture_mean[subject]['mean'], x_gesture_mean[subject]['std'],
                    y_gesture_mean[subject]['mean'], y_gesture_mean[subject]['std'],
                    theta_gesture_mean[subject]['mean'], theta_gesture_mean[subject]['std']])
    abs_df = pd.DataFrame(rows, columns=['Subject', 
                                        'x_mean', 'x_std', 'y_mean', 'y_std',
                                        'theta_mean', 'theta_std'])
    # ジェスチャーごとの差分の絶対値
    n_gestures_list = []
    for distance in distances:
        n_gestures_list.append(int(distance['gesture']))
    gestures = set(n_gestures_list)

    x_gesture_mean = {}
    y_gesture_mean = {}
    theta_gesture_mean = {}
    for gesture in gestures:
        x_gesture_mean[gesture] = {'value':[], 'mean':0, 'std':0}
        y_gesture_mean[gesture] = {'value':[], 'mean':0, 'std':0}
        theta_gesture_mean[gesture] = {'value':[], 'mean':0, 'std':0}
        for subject in subjects:
            for l in range(len(x_ED_gesture[subject][gesture])):
                x_gesture_mean[gesture]['value'].append(np.abs(x_ED_gesture[subject][gesture][l]))
                y_gesture_mean[gesture]['value'].append(np.abs(y_ED_gesture[subject][gesture][l]))
                theta_gesture_mean[gesture]['value'].append(np.abs(theta_ED_gesture[subject][gesture][l]))
            for l in range(len(x_EP_gesture[subject][gesture])):
                x_gesture_mean[gesture]['value'].append(np.abs(x_EP_gesture[subject][gesture][l]))
                y_gesture_mean[gesture]['value'].append(np.abs(y_EP_gesture[subject][gesture][l]))
                theta_gesture_mean[gesture]['value'].append(np.abs(theta_EP_gesture[subject][gesture][l]))
            for l in range(len(x_FD_gesture[subject][gesture])):
                x_gesture_mean[gesture]['value'].append(np.abs(x_FD_gesture[subject][gesture][l]))
                y_gesture_mean[gesture]['value'].append(np.abs(y_FD_gesture[subject][gesture][l]))
                theta_gesture_mean[gesture]['value'].append(np.abs(theta_FD_gesture[subject][gesture][l]))
            for l in range(len(x_FP_gesture[subject][gesture])):
                x_gesture_mean[gesture]['value'].append(np.abs(x_FP_gesture[subject][gesture][l]))
                y_gesture_mean[gesture]['value'].append(np.abs(y_FP_gesture[subject][gesture][l]))
                theta_gesture_mean[gesture]['value'].append(np.abs(theta_FP_gesture[subject][gesture][l]))
        x_gesture_mean[gesture]['mean'] = np.mean(x_gesture_mean[gesture]['value'])
        x_gesture_mean[gesture]['std'] = np.std(x_gesture_mean[gesture]['value'])
        y_gesture_mean[gesture]['mean'] = np.mean(y_gesture_mean[gesture]['value'])
        y_gesture_mean[gesture]['std'] = np.std(y_gesture_mean[gesture]['value'])
        theta_gesture_mean[gesture]['mean'] = np.mean(theta_gesture_mean[gesture]['value'])
        theta_gesture_mean[gesture]['std'] = np.std(theta_gesture_mean[gesture]['value'])
    # 表の作成
    rows = []
    for gesture in gestures:
        rows.append([gesture, 
                    x_gesture_mean[gesture]['mean'], x_gesture_mean[gesture]['std'],
                    y_gesture_mean[gesture]['mean'], y_gesture_mean[gesture]['std'],
                    theta_gesture_mean[gesture]['mean'], theta_gesture_mean[gesture]['std']])
    gesture_df = pd.DataFrame(rows, columns=['Gesture', 
                                        'x_mean', 'x_std', 'y_mean', 'y_std',
                                        'theta_mean', 'theta_std'])
    return abs_df, gesture_df

def featurename(feature_func):
    if feature_func == ptp:
        file_name_prefix = 'ptp'
    elif feature_func == rms:
        file_name_prefix = 'rms'
    elif feature_func == zc:
        file_name_prefix = 'zc'
    elif feature_func == waveform_length:
        file_name_prefix = 'waveformlength'
    elif feature_func == mean_frequency:
        file_name_prefix = 'meanfrequency'
    elif feature_func == median_frequency:
        file_name_prefix = 'medianfrequency'
    elif feature_func == peak_frequency:
        file_name_prefix = 'peakfrequency'
    elif feature_func == spectral_entropy:
        file_name_prefix = 'sepectralentropy'
    return file_name_prefix

def csv_saver(muscle_activity_informations, file_name_prefix = 'test3_ptp_gaussianfitting'):
    x_mean, y_mean, theta_mean, x_std, y_std, theta_std, x_max, x_min, y_max, y_min, theta_max, theta_min = scaler(muscle_activity_informations)
    distances = distances_across_sessions(muscle_activity_informations, scaling=2, x_std=x_std, y_std=y_std, theta_std=theta_std, x_max=x_max, x_min=x_min, y_max=y_max, y_min=y_min, theta_max=theta_max, theta_min=theta_min)
    abs_df, gesture_df = df_maker(distances)
    abs_df.to_csv('output/' + file_name_prefix + '_abs_sessions.csv', index=False, encoding="utf-8-sig")
    gesture_df.to_csv('output/' + file_name_prefix + '_gesture_sessions.csv', index=False, encoding="utf-8-sig")
    distances = distances_across_trials(muscle_activity_informations, scaling=2, x_std=x_std, y_std=y_std, theta_std=theta_std, x_max=x_max, x_min=x_min, y_max=y_max, y_min=y_min, theta_max=theta_max, theta_min=theta_min)
    abs_df, gesture_df = df_maker(distances)
    abs_df.to_csv('output/' + file_name_prefix + '_abs_trials.csv', index=False, encoding="utf-8-sig")
    gesture_df.to_csv('output/' + file_name_prefix + '_gesture_trials.csv', index=False, encoding="utf-8-sig")
    distances = distances_across_sessions_and_trials(muscle_activity_informations, scaling=2, x_std=x_std, y_std=y_std, theta_std=theta_std, x_max=x_max, x_min=x_min, y_max=y_max, y_min=y_min, theta_max=theta_max, theta_min=theta_min)
    abs_df, gesture_df = df_maker(distances)
    abs_df.to_csv('output/' + file_name_prefix + '_abs_sessions_and_trials.csv', index=False, encoding="utf-8-sig")
    gesture_df.to_csv('output/' + file_name_prefix + '_gesture_sessions_and_trials.csv', index=False, encoding="utf-8-sig")


# ----- メイン -----
# if __name__ == "__main__":
def main(feature_func, clustering=True, func_type=False):
    # 分析
    preprosess = False #前処理を行うかどうか
    n_subjects = 5 #20
    n_sessions = 2

    muscle_activity_informations_kmeans32 = []
    muscle_activity_informations_kmeans42 = []
    muscle_activity_informations_kmeans43 = []
    muscle_activity_informations_kmeans52 = []
    muscle_activity_informations_kmeans53 = []
    muscle_activity_informations_kmeans54 = []
    muscle_activity_informations_normalized_kmeans2 = []
    muscle_activity_informations_normalized_kmeans3 = []
    muscle_activity_informations_normalized_kmeans4 = []
    muscle_activity_informations_normalized_kmeans5 = []
    muscle_activity_informations_xmeans = []
    muscle_activity_informations_normalized_xmeans = []
    muscle_activity_informations_hdbscan = []
    muscle_activity_informations_normalized_hdbscan = []
    for i in range(n_subjects):
        for j in range(n_sessions):
            lines = []
            gestures= open('pr_dataset/subject{:02}'.format(i+1) + '_session' + str(j+1) + '/label_maintenance.txt', 'r')
            line = gestures.read()
            for l in line.split(','):
                lines.append(l.strip())
            gestures.close()
            gesture = 0
            for k, line in enumerate(lines): #for k, line in enumerate(lines):
                if gesture == line:
                    trial = 2
                else:
                    trial = 1
                gesture = line
                record_name = 'pr_dataset/subject{:02}'.format(i+1) + '_session' + str(j+1) + '/maintenance_preprocess_sample'+str(k+1)
                # lists.append({'file_name': record_name, 'gesture': gesture, 'trial': trial, 'subject': i+1, 'session': j+1})
                print(record_name) # ファイル名
                try:
                    record = wfdb.rdrecord(record_name)

                    filtered_emg_ED = record.p_signal[:,:64] #Extensor Distal
                    filtered_emg_EP = record.p_signal[:,64:128] #Extensor Proximal
                    filtered_emg_FD = record.p_signal[:,128:192] #Flexor Distal
                    filtered_emg_FP = record.p_signal[:,192:256] #Flexor Proximal

                    electrode_places = [[filtered_emg_ED, 'ED'],
                                        [filtered_emg_EP, 'EP'],
                                        [filtered_emg_FD, 'FD'],
                                        [filtered_emg_FP, 'FP']]

                    for electrode_place in electrode_places:
                        print(electrode_place[1]) #電極位置
                        
                        try:
                            emg_data = electrode_place[0]
                            if preprosess:
                                emg_data = butter_bandpass_filter(emg_data, fs=2048, low_hz=20.0, high_hz=400.0, order=4)
                            features = gaussian_fitting(emg_data, gaussian_2d, feature_func, fs=2048, window_ms=25, threshold=0, func_type=func_type)
                            if clustering:
                                # 32kmeansクラスタリング
                                results_df, summary_df = kmeans_clustering(features, k1=3, k2=2)
                                virtual_bipolars, labels, center_direction, n_virtual_bipolars = get_virtual_bipolars(results_df, show_plot=False)
                                muscle_activity_informations_kmeans32.append({'file_name': record_name, 'gesture': gesture, 'trial': trial, 'subject': i+1, 'session': j+1, 'electrode_place':electrode_place[1], 'emg_data': emg_data, 'virtual_bipolars': virtual_bipolars, 'labels': labels, 'center_direction': center_direction, 'n_virtual_bipolars': n_virtual_bipolars})
                                # 42kmeansクラスタリング
                                results_df, summary_df = kmeans_clustering(features, k1=4, k2=2)
                                virtual_bipolars, labels, center_direction, n_virtual_bipolars = get_virtual_bipolars(results_df, show_plot=False)
                                muscle_activity_informations_kmeans42.append({'file_name': record_name, 'gesture': gesture, 'trial': trial, 'subject': i+1, 'session': j+1, 'electrode_place':electrode_place[1], 'emg_data': emg_data, 'virtual_bipolars': virtual_bipolars, 'labels': labels, 'center_direction': center_direction, 'n_virtual_bipolars': n_virtual_bipolars})
                                # 43kmeansクラスタリング
                                results_df, summary_df = kmeans_clustering(features, k1=4, k2=3)
                                virtual_bipolars, labels, center_direction, n_virtual_bipolars = get_virtual_bipolars(results_df, show_plot=False)
                                muscle_activity_informations_kmeans43.append({'file_name': record_name, 'gesture': gesture, 'trial': trial, 'subject': i+1, 'session': j+1, 'electrode_place':electrode_place[1], 'emg_data': emg_data, 'virtual_bipolars': virtual_bipolars, 'labels': labels, 'center_direction': center_direction, 'n_virtual_bipolars': n_virtual_bipolars})
                                # 52kmeansクラスタリング
                                results_df, summary_df = kmeans_clustering(features, k1=5, k2=2)
                                virtual_bipolars, labels, center_direction, n_virtual_bipolars = get_virtual_bipolars(results_df, show_plot=False)
                                muscle_activity_informations_kmeans52.append({'file_name': record_name, 'gesture': gesture, 'trial': trial, 'subject': i+1, 'session': j+1, 'electrode_place':electrode_place[1], 'emg_data': emg_data, 'virtual_bipolars': virtual_bipolars, 'labels': labels, 'center_direction': center_direction, 'n_virtual_bipolars': n_virtual_bipolars})
                                # 53kmeansクラスタリング
                                results_df, summary_df = kmeans_clustering(features, k1=5, k2=3)
                                virtual_bipolars, labels, center_direction, n_virtual_bipolars = get_virtual_bipolars(results_df, show_plot=False)
                                muscle_activity_informations_kmeans53.append({'file_name': record_name, 'gesture': gesture, 'trial': trial, 'subject': i+1, 'session': j+1, 'electrode_place':electrode_place[1], 'emg_data': emg_data, 'virtual_bipolars': virtual_bipolars, 'labels': labels, 'center_direction': center_direction, 'n_virtual_bipolars': n_virtual_bipolars})
                                # 54kmeansクラスタリング
                                results_df, summary_df = kmeans_clustering(features, k1=5, k2=4)
                                virtual_bipolars, labels, center_direction, n_virtual_bipolars = get_virtual_bipolars(results_df, show_plot=False)
                                muscle_activity_informations_kmeans54.append({'file_name': record_name, 'gesture': gesture, 'trial': trial, 'subject': i+1, 'session': j+1, 'electrode_place':electrode_place[1], 'emg_data': emg_data, 'virtual_bipolars': virtual_bipolars, 'labels': labels, 'center_direction': center_direction, 'n_virtual_bipolars': n_virtual_bipolars})
                            else:
                                # 2normalized_kmeansクラスタリング
                                results_df, summary_df = normalized_kmeans_clustering(features, k1=2)
                                virtual_bipolars, labels, center_direction, n_virtual_bipolars = get_virtual_bipolars(results_df, show_plot=False)
                                muscle_activity_informations_normalized_kmeans2.append({'file_name': record_name, 'gesture': gesture, 'trial': trial, 'subject': i+1, 'session': j+1, 'electrode_place':electrode_place[1], 'emg_data': emg_data, 'virtual_bipolars': virtual_bipolars, 'labels': labels, 'center_direction': center_direction, 'n_virtual_bipolars': n_virtual_bipolars})
                                # 3normalized_kmeansクラスタリング
                                results_df, summary_df = normalized_kmeans_clustering(features, k1=3)
                                virtual_bipolars, labels, center_direction, n_virtual_bipolars = get_virtual_bipolars(results_df, show_plot=False)
                                muscle_activity_informations_normalized_kmeans3.append({'file_name': record_name, 'gesture': gesture, 'trial': trial, 'subject': i+1, 'session': j+1, 'electrode_place':electrode_place[1], 'emg_data': emg_data, 'virtual_bipolars': virtual_bipolars, 'labels': labels, 'center_direction': center_direction, 'n_virtual_bipolars': n_virtual_bipolars})
                                # 4normalized_kmeansクラスタリング
                                results_df, summary_df = normalized_kmeans_clustering(features, k1=4)
                                virtual_bipolars, labels, center_direction, n_virtual_bipolars = get_virtual_bipolars(results_df, show_plot=False)
                                muscle_activity_informations_normalized_kmeans4.append({'file_name': record_name, 'gesture': gesture, 'trial': trial, 'subject': i+1, 'session': j+1, 'electrode_place':electrode_place[1], 'emg_data': emg_data, 'virtual_bipolars': virtual_bipolars, 'labels': labels, 'center_direction': center_direction, 'n_virtual_bipolars': n_virtual_bipolars})
                                # 5normalized_kmeansクラスタリング
                                results_df, summary_df = normalized_kmeans_clustering(features, k1=5)
                                virtual_bipolars, labels, center_direction, n_virtual_bipolars = get_virtual_bipolars(results_df, show_plot=False)
                                muscle_activity_informations_normalized_kmeans5.append({'file_name': record_name, 'gesture': gesture, 'trial': trial, 'subject': i+1, 'session': j+1, 'electrode_place':electrode_place[1], 'emg_data': emg_data, 'virtual_bipolars': virtual_bipolars, 'labels': labels, 'center_direction': center_direction, 'n_virtual_bipolars': n_virtual_bipolars})
                                # xmeansクラスタリング
                                results_df, summary_df = xmeans_clustering(features, kmax=5)
                                virtual_bipolars, labels, center_direction, n_virtual_bipolars = get_virtual_bipolars(results_df, show_plot=False)
                                muscle_activity_informations_xmeans.append({'file_name': record_name, 'gesture': gesture, 'trial': trial, 'subject': i+1, 'session': j+1, 'electrode_place':electrode_place[1], 'emg_data': emg_data, 'virtual_bipolars': virtual_bipolars, 'labels': labels, 'center_direction': center_direction, 'n_virtual_bipolars': n_virtual_bipolars})
                                # normalized_xmeansクラスタリング
                                results_df, summary_df = normalized_xmeans_clustering(features, kmax=5)
                                virtual_bipolars, labels, center_direction, n_virtual_bipolars = get_virtual_bipolars(results_df, show_plot=False)
                                muscle_activity_informations_normalized_xmeans.append({'file_name': record_name, 'gesture': gesture, 'trial': trial, 'subject': i+1, 'session': j+1, 'electrode_place':electrode_place[1], 'emg_data': emg_data, 'virtual_bipolars': virtual_bipolars, 'labels': labels, 'center_direction': center_direction, 'n_virtual_bipolars': n_virtual_bipolars})
                                # hdbscanクラスタリング
                                results_df, summary_df = hdbscan_clustering(features, min_cluster_size=10)
                                virtual_bipolars, labels, center_direction, n_virtual_bipolars = get_virtual_bipolars(results_df, show_plot=False)
                                muscle_activity_informations_hdbscan.append({'file_name': record_name, 'gesture': gesture, 'trial': trial, 'subject': i+1, 'session': j+1, 'electrode_place':electrode_place[1], 'emg_data': emg_data, 'virtual_bipolars': virtual_bipolars, 'labels': labels, 'center_direction': center_direction, 'n_virtual_bipolars': n_virtual_bipolars})
                                # normalized_hdbscanクラスタリング
                                results_df, summary_df = normalized_hdbscan_clustering(features, min_cluster_size=10)
                                virtual_bipolars, labels, center_direction, n_virtual_bipolars = get_virtual_bipolars(results_df, show_plot=False)
                                muscle_activity_informations_normalized_hdbscan.append({'file_name': record_name, 'gesture': gesture, 'trial': trial, 'subject': i+1, 'session': j+1, 'electrode_place':electrode_place[1], 'emg_data': emg_data, 'virtual_bipolars': virtual_bipolars, 'labels': labels, 'center_direction': center_direction, 'n_virtual_bipolars': n_virtual_bipolars})
                        except RuntimeError:
                            if clustering:
                                muscle_activity_informations_kmeans32.append({'file_name': record_name, 'gesture': gesture, 'trial': trial, 'subject': i+1, 'session': j+1, 'electrode_place':electrode_place[1], 'emg_data': [], 'virtual_bipolars': [], 'labels': [], 'center_direction': [], 'n_virtual_bipolars': 0})
                                muscle_activity_informations_kmeans42.append({'file_name': record_name, 'gesture': gesture, 'trial': trial, 'subject': i+1, 'session': j+1, 'electrode_place':electrode_place[1], 'emg_data': [], 'virtual_bipolars': [], 'labels': [], 'center_direction': [], 'n_virtual_bipolars': 0})
                                muscle_activity_informations_kmeans43.append({'file_name': record_name, 'gesture': gesture, 'trial': trial, 'subject': i+1, 'session': j+1, 'electrode_place':electrode_place[1], 'emg_data': [], 'virtual_bipolars': [], 'labels': [], 'center_direction': [], 'n_virtual_bipolars': 0})
                                muscle_activity_informations_kmeans52.append({'file_name': record_name, 'gesture': gesture, 'trial': trial, 'subject': i+1, 'session': j+1, 'electrode_place':electrode_place[1], 'emg_data': [], 'virtual_bipolars': [], 'labels': [], 'center_direction': [], 'n_virtual_bipolars': 0})
                                muscle_activity_informations_kmeans53.append({'file_name': record_name, 'gesture': gesture, 'trial': trial, 'subject': i+1, 'session': j+1, 'electrode_place':electrode_place[1], 'emg_data': [], 'virtual_bipolars': [], 'labels': [], 'center_direction': [], 'n_virtual_bipolars': 0})
                                muscle_activity_informations_kmeans54.append({'file_name': record_name, 'gesture': gesture, 'trial': trial, 'subject': i+1, 'session': j+1, 'electrode_place':electrode_place[1], 'emg_data': [], 'virtual_bipolars': [], 'labels': [], 'center_direction': [], 'n_virtual_bipolars': 0})
                            else:
                                muscle_activity_informations_normalized_kmeans2.append({'file_name': record_name, 'gesture': gesture, 'trial': trial, 'subject': i+1, 'session': j+1, 'electrode_place':electrode_place[1], 'emg_data': [], 'virtual_bipolars': [], 'labels': [], 'center_direction': [], 'n_virtual_bipolars': 0})
                                muscle_activity_informations_normalized_kmeans3.append({'file_name': record_name, 'gesture': gesture, 'trial': trial, 'subject': i+1, 'session': j+1, 'electrode_place':electrode_place[1], 'emg_data': [], 'virtual_bipolars': [], 'labels': [], 'center_direction': [], 'n_virtual_bipolars': 0})
                                muscle_activity_informations_normalized_kmeans4.append({'file_name': record_name, 'gesture': gesture, 'trial': trial, 'subject': i+1, 'session': j+1, 'electrode_place':electrode_place[1], 'emg_data': [], 'virtual_bipolars': [], 'labels': [], 'center_direction': [], 'n_virtual_bipolars': 0})
                                muscle_activity_informations_normalized_kmeans5.append({'file_name': record_name, 'gesture': gesture, 'trial': trial, 'subject': i+1, 'session': j+1, 'electrode_place':electrode_place[1], 'emg_data': [], 'virtual_bipolars': [], 'labels': [], 'center_direction': [], 'n_virtual_bipolars': 0})
                                muscle_activity_informations_normalized_xmeans.append({'file_name': record_name, 'gesture': gesture, 'trial': trial, 'subject': i+1, 'session': j+1, 'electrode_place':electrode_place[1], 'emg_data': [], 'virtual_bipolars': [], 'labels': [], 'center_direction': [], 'n_virtual_bipolars': 0})
                                muscle_activity_informations_normalized_hdbscan.append({'file_name': record_name, 'gesture': gesture, 'trial': trial, 'subject': i+1, 'session': j+1, 'electrode_place':electrode_place[1], 'emg_data': [], 'virtual_bipolars': [], 'labels': [], 'center_direction': [], 'n_virtual_bipolars': 0})
                except FileNotFoundError:
                    pass

    file_name_prefix = 'test0_' + featurename(feature_func) + '_gaussianfitting'
    # csvファイルに保存
    if clustering:
        csv_saver(muscle_activity_informations_kmeans32, file_name_prefix = file_name_prefix + '/kmeans32')
        csv_saver(muscle_activity_informations_kmeans42, file_name_prefix = file_name_prefix + '/kmeans42')
        csv_saver(muscle_activity_informations_kmeans43, file_name_prefix = file_name_prefix + '/kmeans43')
        csv_saver(muscle_activity_informations_kmeans52, file_name_prefix = file_name_prefix + '/kmeans52')
        csv_saver(muscle_activity_informations_kmeans53, file_name_prefix = file_name_prefix + '/kmeans53')
        csv_saver(muscle_activity_informations_kmeans54, file_name_prefix = file_name_prefix + '/kmeans54')
    else:
        csv_saver(muscle_activity_informations_normalized_kmeans2, file_name_prefix = file_name_prefix + '/normalized_kmeans2')
        csv_saver(muscle_activity_informations_normalized_kmeans3, file_name_prefix = file_name_prefix + '/normalized_kmeans3')
        csv_saver(muscle_activity_informations_normalized_kmeans4, file_name_prefix = file_name_prefix + '/normalized_kmeans4')
        csv_saver(muscle_activity_informations_normalized_kmeans5, file_name_prefix = file_name_prefix + '/normalized_kmeans5')
        csv_saver(muscle_activity_informations_xmeans, file_name_prefix = file_name_prefix + '/xmeans')
        csv_saver(muscle_activity_informations_normalized_xmeans, file_name_prefix = file_name_prefix + '/normalized_xmeans')
        csv_saver(muscle_activity_informations_hdbscan, file_name_prefix = file_name_prefix + '/hdbscan')
        csv_saver(muscle_activity_informations_normalized_hdbscan, file_name_prefix = file_name_prefix + '/normalized_hdbscan')

    # save_records_to_csv(muscle_activity_informations_kmeans32, out_path='output/test3_ptp_gaussianfitting_kmeans32.csv')
    # save_records_to_csv(muscle_activity_informations_kmeans42, out_path='output/test3_ptp_gaussianfitting_kmeans42.csv')
    # save_records_to_csv(muscle_activity_informations_kmeans43, out_path='output/test3_ptp_gaussianfitting_kmeans43.csv')
    # save_records_to_csv(muscle_activity_informations_kmeans52, out_path='output/test3_ptp_gaussianfitting_kmeans52.csv')
    # save_records_to_csv(muscle_activity_informations_kmeans53, out_path='output/test3_ptp_gaussianfitting_kmeans53.csv')
    # save_records_to_csv(muscle_activity_informations_kmeans54, out_path='output/test3_ptp_gaussianfitting_kmeans54.csv')
    # save_records_to_csv(muscle_activity_informations_normalized_kmeans2, out_path='output/test3_ptp_gaussianfitting_normalized_kmeans2.csv')
    # save_records_to_csv(muscle_activity_informations_normalized_kmeans3, out_path='output/test3_ptp_gaussianfitting_normalized_kmeans3.csv')
    # save_records_to_csv(muscle_activity_informations_normalized_kmeans4, out_path='output/test3_ptp_gaussianfitting_normalized_kmeans4.csv')
    # save_records_to_csv(muscle_activity_informations_normalized_kmeans5, out_path='output/test3_ptp_gaussianfitting_normalized_kmeans5.csv')
    # save_records_to_csv(muscle_activity_informations_normalized_xmeans, out_path='output/test3_ptp_gaussianfitting_xmeans.csv')
    # save_records_to_csv(muscle_activity_informations_normalized_xmeans, out_path='output/test3_ptp_gaussianfitting_normalized_xmeans.csv')
    # save_records_to_csv(muscle_activity_informations_normalized_hdbscan, out_path='output/test3_ptp_gaussianfitting_hdbscan.csv')
    # save_records_to_csv(muscle_activity_informations_normalized_hdbscan, out_path='output/test3_ptp_gaussianfitting_normalized_hdbscan.csv')