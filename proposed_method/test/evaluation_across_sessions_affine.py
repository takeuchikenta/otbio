import wfdb
import os
import glob
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy import signal
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
from sklearn.decomposition import PCA
import matplotlib.animation as animation
import matplotlib.cm as cm
from IPython.display import HTML
from pyclustering.cluster.gmeans import gmeans
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils.metric import type_metric, distance_metric
import cv2
from scipy.stats import pearsonr
import pingouin as pg  # for ICC
from sklearn.svm import SVC
import itertools
import json
from pathlib import Path
from collections.abc import Mapping
from typing import List, Sequence, Any, Optional
import hdbscan
import itk
from scipy.signal import correlate
from scipy.optimize import linear_sum_assignment, minimize
import warnings
np.warnings = warnings
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import glmdtps


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

# ノッチフィルタ
def remove_power_line_harmonics(data, fs, fundamental=60.0, Q=30.0):
    # 出力用データのコピー
    filtered_data = data.copy()
    
    # ナイキスト周波数
    nyquist = fs / 2.0
    
    # 除去すべき周波数のリストを作成 (60, 120, 180 ... < nyquist)
    # np.arange(start, stop, step)
    target_freqs = np.arange(fundamental, nyquist, fundamental)
    
    # print(f"Removing harmonics at: {target_freqs} Hz")
    
    for freq in target_freqs:
        # --- iirnotchフィルタの設計 ---
        # w0: 除去したい周波数
        # Q: 鋭さ (例: Q=30 なら 60Hz±1Hz 程度が削られるイメージ)
        b, a = signal.iirnotch(w0=freq, Q=Q, fs=fs)
        
        # --- フィルタ適用 (ゼロ位相) ---
        # filtfiltを使うことで位相ズレを防ぐ
        # axis=0 (時間方向) に対して適用
        filtered_data = signal.filtfilt(b, a, filtered_data, axis=0)
        
    return filtered_data

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

# ファイル名出力関数
def file_name_output(subject, hand="right", electrode_place="original", gesture=1, trial=1):
    hand = "right"

    if electrode_place == "original":
        filename = "1-original"
    elif electrode_place == "upright":
        filename = "2-upright"
    elif electrode_place == "downright":
        filename = "3-downright"
    elif electrode_place == "downleft":
        filename = "4-downleft"
    elif electrode_place == "upleft":
        filename = "5-upleft"
    elif electrode_place == "clockwise":
        filename = "6-clockwise"
    elif electrode_place == "anticlockwise":
        filename = "7-anticlockwise"
    elif electrode_place == "original2":
        filename = "original2"
    elif electrode_place == "downleft5mm":
        filename = "downleft5mm"
    elif electrode_place == "downleft10mm":
        filename = "downleft10mm"
    file_name = subject + '/' + hand + '/' + filename + '/set' + str(trial) + '/' + electrode_place + '-g' + str(gesture) + '-' + str(trial) + '.csv'
    # print(file_name)
    return file_name

#===筋線維方向を棒で描画するユーティリティ===
def wrap_half_pi(theta_rad):
    """
    軸データ(θ≡θ+π)を [-pi/2, pi/2] に丸める。
    棒の向きとしてはこの軸方向がちょうど良い。
    """
    return (theta_rad + np.pi/2) % np.pi - np.pi/2


def scale_keypoints(keypoints, scale):
    """
    元の keypoints の座標(x,y)を scale 倍して新しいリストを返す。
    角度やampはそのままコピーする。
    keypoints は dict型 [{"x":..,"y":..,"angle":..,"amp":..}, ...]
    または tuple型 (x,y,angle[,amp]) の混在でもOKにする。
    """
    scaled = []
    for kp in keypoints:
        if isinstance(kp, dict):
            new_kp = kp.copy()
            new_kp["x"] = kp["x"] * scale
            new_kp["y"] = kp["y"] * scale
            # angleやampなどはそのまま
            scaled.append(new_kp)
        else:
            # tuple/list想定
            if len(kp) == 3:
                x, y, ang = kp
                scaled.append((x*scale, y*scale, ang))
            elif len(kp) >= 4:
                x, y, ang, val = kp[0], kp[1], kp[2], kp[3]
                # 4つ目以降(ampなど)もそのまま残す
                scaled.append((x*scale, y*scale, ang, val))
            else:
                raise ValueError("keypoint tuple must be at least (x,y,angle)")
    return scaled

def draw_keypoints_as_sticks(
    keypoints,
    canvas_h=71,
    canvas_w=71,
    stick_len=10.0,
    stick_thickness=2,
    angle_is_deg=False,
    intensity_key=None,
    intensity_default=1.0,
    use_wrap=True,
    combine_mode="add",
    clip_output=True
):
    # float32のキャンバス（加算用）
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    half_len = stick_len / 2.0

    for kp in keypoints:
        # --- 1) x, y, angle, strength を取り出す ---
        if isinstance(kp, dict):
            x = float(kp["x"])
            y = float(kp["y"])
            if "angle" in kp:
                ang = float(kp["angle"])
            elif "angle_rad" in kp:
                ang = float(kp["angle_rad"])
            else:
                raise ValueError("dictにangle (rad) がありません")

            if angle_is_deg:
                ang = np.deg2rad(ang)

            if use_wrap:
                ang = wrap_half_pi(ang)

            # 強度（明るさ）
            if intensity_key is not None:
                strength = float(kp.get(intensity_key, intensity_default))
            else:
                strength = float(intensity_default)

        else:
            # tuple/list想定: (x, y, angle, [strength?])
            if len(kp) < 3:
                raise ValueError("tupleは少なくとも(x,y,angle)が必要")
            x = float(kp[0])
            y = float(kp[1])
            ang = float(kp[2])

            if angle_is_deg:
                ang = np.deg2rad(ang)
            if use_wrap:
                ang = wrap_half_pi(ang)

            if len(kp) >= 4:
                strength = float(kp[3])
            else:
                strength = float(intensity_default)

        # --- 2) 棒の両端座標を計算 ---
        # 方向ベクトル (dx, dy) を angle から作る
        dx = np.cos(ang) * half_len
        dy = np.sin(ang) * half_len

        # OpenCV は (col=x, row=y) なので注意
        x1 = x - dx
        y1 = y - dy
        x2 = x + dx
        y2 = y + dy

        # 描画先はuint8画像を想定するが、加算したいので一旦別レイヤに描く
        layer = np.zeros((canvas_h, canvas_w), dtype=np.float32)

        # cv2.line の引数は整数座標なので丸める
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))

        # 線を描画（強度はとりあえず 255*strength を期待値に）
        # 後で正規化するならここは strength だけでもいいけど
        val = float(strength)

        # OpenCVはBGRイメージを想定することが多いけど、ここは1ch灰色なので
        cv2.line(
            layer,
            p1,
            p2,
            color=val,          # floatでもOK。layerはfloat32
            thickness=stick_thickness,
            lineType=cv2.LINE_AA
        )

        # --- 3) layer を canvas に合成 ---
        if combine_mode == "add":
            canvas += layer
        elif combine_mode == "max":
            canvas = np.maximum(canvas, layer)
        else:
            raise ValueError("combine_mode must be 'add' or 'max'")

    # --- 4) 正規化して0-255にスケール ---
    vmax = canvas.max()
    if vmax > 0:
        img_f = canvas / vmax
    else:
        img_f = canvas.copy()

    img_u8 = (img_f * 255.0).astype(np.uint8)

    if clip_output:
        img_u8 = np.clip(img_u8, 0, 255).astype(np.uint8)

    return img_u8, img_f

# -*- coding: utf-8 -*-
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Literal, Optional, Any, Dict, Callable, List
from scipy.interpolate import RectBivariateSpline

# ===== 可搬な import（PyTorchは任意） =====
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# -*- coding: utf-8 -*-
ArrayLike = np.ndarray

def extract_center_6x6(emg: ArrayLike) -> ArrayLike:
    """
    元の8×8から中央6×6(行1..6, 列1..6)を抽出。
    emg: (n_samples, 8, 8)
    return: (n_samples, 6, 6)
    """
    if emg.ndim != 3 or emg.shape[1:] != (8, 8):
        raise ValueError("emg must have shape (n_samples, 8, 8)")
    return emg[:, 1:7, 1:7]


@dataclass
class GridSubsetMapper:
    """
    8×8グリッドのEMGから、既知の並進・回転だけずらした“中央6×6サブセット”位置の
    EMG時系列を、空間スプライン補間で取得するクラス。
    """
    n_rows: int = 8
    n_cols: int = 8
    spacing: float = 1.0  # 電極間距離の空間単位（mmでも1でも可）
    rotate_about: str = "grid_center"  # 'grid_center'（推奨）/ 'subset_center'
    # 将来: 'custom' にして任意中心にしたい場合は center=(cx,cy) を追加実装

    def _grid_coords(self) -> Tuple[ArrayLike, ArrayLike]:
        """8×8グリッドの座標（行y, 列x）。"""
        y = np.arange(self.n_rows) * self.spacing  # 行インデックス方向
        x = np.arange(self.n_cols) * self.spacing  # 列インデックス方向
        return y, x

    def _subset_coords(self) -> Tuple[ArrayLike, ArrayLike]:
        """中央6×6サブセットの“元の”座標（行y, 列x）を (6,6) で返す。"""
        y_full, x_full = self._grid_coords()
        y6 = y_full[1:7]  # 行1..6
        x6 = x_full[1:7]  # 列1..6
        Y6, X6 = np.meshgrid(y6, x6, indexing="ij")  # (6,6)
        return Y6, X6

    def _rotation_center(self) -> Tuple[float, float]:
        """回転中心座標 (cy, cx) を返す。"""
        if self.rotate_about == "grid_center":
            cy = (self.n_rows - 1) * 0.5 * self.spacing
            cx = (self.n_cols - 1) * 0.5 * self.spacing
        elif self.rotate_about == "subset_center":
            # 中央6×6の中心（行/列1..6の中心）= 3.5 → グリッド座標では index=1..6 の中心は (3.5, 3.5)
            cy = 3.5 * self.spacing
            cx = 3.5 * self.spacing
        else:
            raise ValueError("rotate_about must be 'grid_center' or 'subset_center'")
        return cy, cx

    @staticmethod
    def _apply_se2(Y: ArrayLike, X: ArrayLike, cy: float, cx: float,
                   dy: float, dx: float, theta: float) -> Tuple[ArrayLike, ArrayLike]:
        """
        2Dの相似変換（回転+並進、スケール=1）。
        入出力: (6,6)の座標格子。回転は (cx,cy) を中心。dx,dy は並進（x右+, y下+）。
        """
        # 平行移動して原点を回転中心へ
        Y0 = Y - cy
        X0 = X - cx
        c, s = np.cos(theta), np.sin(theta)
        # 反時計回り回転
        Xr =  c * X0 - s * Y0
        Yr =  s * X0 + c * Y0
        # 元の位置に戻して並進
        Xp = Xr + cx + dx
        Yp = Yr + cy + dy
        return Yp, Xp

    def _build_spline(self, frame: ArrayLike) -> RectBivariateSpline:
        """
        1フレーム(8×8)からRectBivariateSplineを作成。
        RectBivariateSplineの引数順に注意: (y, x, z)
        """
        if frame.shape != (self.n_rows, self.n_cols):
            raise ValueError("frame must be (8,8)")
        y, x = self._grid_coords()
        # kx=3, ky=3 → bicubic。s=0で補間（平滑化なし）
        return RectBivariateSpline(y, x, frame, kx=3, ky=3, s=0)

    def transform(
        self,
        emg: ArrayLike,
        dx: float,
        dy: float,
        theta: float,
        mode: Literal["extrapolate", "clip"] = "extrapolate",
    ) -> ArrayLike:
        """
        既知の位置ずれ (dx,dy,theta) を中央6×6サブセットに適用し、
        その“ずらした位置”でのEMG時系列をスプライン補間で取得する。

        emg : (n_samples, 8, 8)
        dx, dy : 並進。+xは列方向(右)、+yは行方向(下)。単位は self.spacing に合わせる。
        theta : 反時計回りの回転 [radian]。
        mode : 'extrapolate'（外挿許容） or 'clip'（境界にクリップ）

        return : (n_samples, 6, 6)
        """
        if emg.ndim != 3 or emg.shape[1:] != (self.n_rows, self.n_cols):
            raise ValueError("emg must have shape (n_samples, 8, 8)")

        n_samples = emg.shape[0]
        Y6, X6 = self._subset_coords()           # (6,6) 元のサブセット座標
        cy, cx = self._rotation_center()

        # サブセット座標に対してずれ(回転+並進)を適用
        Yt, Xt = self._apply_se2(Y6, X6, cy, cx, dy, dx, theta)  # (6,6)

        # クリップ指定なら境界内に丸め込む
        if mode == "clip":
            y0, y1 = 0.0, (self.n_rows - 1) * self.spacing
            x0, x1 = 0.0, (self.n_cols - 1) * self.spacing
            Yq = np.clip(Yt, y0, y1)
            Xq = np.clip(Xt, x0, x1)
        elif mode == "extrapolate":
            Yq, Xq = Yt, Xt
        else:
            raise ValueError("mode must be 'extrapolate' or 'clip'")

        # (n_samples, 6, 6) を格納
        out = np.empty((n_samples, 6, 6), dtype=emg.dtype)

        # 各時間サンプルごとに2Dスプラインで補間
        # RectBivariateSpline.ev はベクトル化可能：ev(Y.flatten(), X.flatten()) → reshape
        Y_flat = Yq.ravel()
        X_flat = Xq.ravel()
        for i in range(n_samples):
            spl = self._build_spline(emg[i])         # frame=(8,8)
            zi = spl.ev(Y_flat, X_flat).reshape(6, 6)
            out[i] = zi

        return out

def segment_time_series(emg_6x6: np.ndarray, window: int, hop: int) -> np.ndarray:
    """
    (n_samples, 6, 6) を窓分割して (n_windows, window, 6, 6) へ。
    """
    n, _, _ = emg_6x6.shape
    idx_starts = np.arange(0, max(1, n - window + 1), hop)
    segs = np.stack([emg_6x6[s:s+window] for s in idx_starts if s + window <= n], axis=0)
    # segs: (n_windows, window, 6, 6)
    return segs

def rms_feat(x):
    return np.sqrt(np.mean(np.square(x), axis=0))

def wl_feat(x):
    return np.sum(np.abs(np.diff(x, axis=0)), axis=0)

def zero_crossings(x, thresh):
    # count sign changes where both samples exceed threshold in magnitude
    x1 = x[:-1, :]
    x2 = x[1:, :]
    cond = (np.abs(x1) > thresh) & (np.abs(x2) > thresh) & (np.sign(x1) != np.sign(x2))
    return np.sum(cond, axis=0).astype(float)

def slope_sign_changes(x, thresh):
    # count SSC where middle sample forms a slope sign change and all magnitudes exceed threshold
    x_prev = x[:-2, :]
    x_mid  = x[1:-1, :]
    x_next = x[2:, :]
    cond_mag = (np.abs(x_prev) > thresh) & (np.abs(x_mid) > thresh) & (np.abs(x_next) > thresh)
    cond_ssc = (((x_mid - x_prev) * (x_mid - x_next)) > 0)
    return np.sum(cond_mag & cond_ssc, axis=0).astype(float)

def wamp_feat(x, thresh):
    """
    Willison Amplitude (WAMP)
    x: (N, C) 窓データ
    thresh: (C,) しきい値（各チャネル）
    return: (C,) 各チャネルのWAMP
    """
    dx = np.abs(np.diff(x, axis=0))      # (N-1, C)
    return np.sum(dx > thresh, axis=0).astype(float)

# === TD-PSD特徴量 ===

_EPS = 1e-12

def _hoyer_sparsity(x):
    """0〜1の疎度。0=等値ベクトル、1=スパース"""
    x = np.asarray(x)
    n = x.size
    l1 = np.linalg.norm(x, 1)
    l2 = np.linalg.norm(x, 2) + _EPS
    return (np.sqrt(n) - (l1 / l2)) / (np.sqrt(n) - 1 + _EPS)

def _moments_and_derivatives(sig, alpha=0.1):
    """0,1,2,4次微分に基づくモーメント（L2ノルム）を計算し、alpha乗で正規化。"""
    # 微分
    dx1 = np.gradient(sig)
    dx2 = np.gradient(dx1)
    dx4 = np.gradient(np.gradient(dx2))

    # Root-squared moments (式(4)〜(6))
    m0 = np.sqrt(np.sum(sig**2))
    m2 = np.sqrt(np.sum(dx1**2))
    m4 = np.sqrt(np.sum(dx2**2))

    # ノイズ感度低減のための power 変換（式(7)）
    m0 = np.power(m0 + _EPS, alpha)
    m2 = np.power(m2 + _EPS, alpha)
    m4 = np.power(m4 + _EPS, alpha)

    return m0, m2, m4, dx1, dx2

def _six_base_features(sig, alpha=0.1):
    """
    基底6特徴（式(8)〜(11)）:
      f1 = m0
      f2 = m2/m0
      f3 = m4/m2
      f4 = Sparseness（Hoyer型）
      f5 = Irregularity Factor（momentsのみで表現）
      f6 = Waveform Length Ratio = WL(dx1)/WL(dx2)
    """
    m0, m2, m4, dx1, dx2 = _moments_and_derivatives(sig, alpha=alpha)

    # (8) 系: 比特徴
    f1 = m0
    f2 = m2 / (m0 + _EPS)
    f3 = m4 / (m2 + _EPS)

    # (9) 疎度：信号の振幅分布に対するHoyer疎度
    f4 = _hoyer_sparsity(np.abs(sig))

    # (10) IF：ゼロ交差頻度/ピーク頻度 ≈ sqrt( (m2^2)/(m0*m4) )
    f5 = np.sqrt((m2**2) / (m0 * m4 + _EPS))

    # (11) WL比
    wl1 = np.sum(np.abs(dx1))
    wl2 = np.sum(np.abs(dx2)) + _EPS
    f6 = wl1 / wl2

    return np.array([f1, f2, f3, f4, f5, f6], dtype=float)

def _log_map(x):
    """対数スケール（符号は保持）。"""
    return np.sign(x) * np.log1p(np.abs(x))

def td_psd_multichannel(
    window,
    fs=2000,
    alpha=0.1,
    mode="vector",
    m0_channel_norm=False
):
    """
    TD-PSD（式(4)〜(12)）を複数チャネル窓データに適用。

    Parameters
    ----------
    window : ndarray, shape (n_samples, n_channels)
        1窓ぶんのEMG。チャネルは列方向。
    fs : float
        サンプリング周波数（式自体はfsに依存しない）
    alpha : float
        式(7)のpower正規化指数（論文は 0.1）
    mode : {"scalar", "vector"}
        "scalar": 各ch 1次元（cosθ）
        "vector": 各ch 6次元（正規化f,gの要素積。総和はcosθ）
    m0_channel_norm : bool
        式(4)注記の「0次モーメントの全チャネル正規化」を有効化

    Returns
    -------
    feats : ndarray, shape (n_channels, D)
        D = 1（scalar）または 6（vector）
    """
    X = np.asarray(window)
    if X.ndim == 2:
        n, C = X.shape
    elif X.ndim == 3:
        X = X.reshape(X.shape[0], -1)
        n, C = X.shape
    else:
        raise ValueError("window must be 2D or 3D array")

    # 平均除去
    Xz = X - X.mean(axis=0, keepdims=True)
    Xlog = _log_map(Xz)

    # まず各chの基底6特徴（原）と（log）を作る
    F = np.zeros((C, 6), dtype=float)
    G = np.zeros((C, 6), dtype=float)
    for ch in range(C):
        F[ch] = _six_base_features(Xz[:, ch], alpha=alpha)
        G[ch] = _six_base_features(Xlog[:, ch], alpha=alpha)

    # オプション：0次モーメントの全チャネル正規化（式(4)注）
    if m0_channel_norm:
        sum_m0_F = np.sum(F[:, 0]) + _EPS
        sum_m0_G = np.sum(G[:, 0]) + _EPS
        F[:, 0] = F[:, 0] / sum_m0_F
        G[:, 0] = G[:, 0] / sum_m0_G

    # （式12）“方向”の抽出：f,gを正規化し、cosθ = <f̂,ĝ>
    F_norm = F / (np.linalg.norm(F, axis=1, keepdims=True) + _EPS)
    G_norm = G / (np.linalg.norm(G, axis=1, keepdims=True) + _EPS)

    if mode == "scalar":
        # 各chの cosθ を1値で返す
        cos_theta = np.sum(F_norm * G_norm, axis=1, keepdims=True)
        return cos_theta  # (C,1)

    elif mode == "vector":
        # “方向ベクトル成分”：要素ごとの積（和をとると cosθ ）
        orient_vec = F_norm * G_norm  # (C,6)
        return orient_vec

    else:
        raise ValueError("mode must be 'scalar' or 'vector'")

# --- 使い方例 ---
# emg_win: shape (n_samples, n_channels) の1窓データ
# feats_scalar = td_psd_multichannel(emg_win, mode="scalar")  # -> (C,1)
# feats_vector = td_psd_multichannel(emg_win, mode="vector")  # -> (C,6)


def _json_load_if_needed(x):
    """セル文字列がJSONならPythonオブジェクトに戻す。空欄/NaNはNone。"""
    if pd.isna(x):
        return None
    if isinstance(x, (list, dict)):
        return x
    if isinstance(x, str):
        s = x.strip()
        if s.startswith('[') or s.startswith('{') or s.startswith('"'):
            try:
                return json.loads(s)
            except Exception:
                return x  # JSONでなければそのまま
    return x

def load_records_from_csv(path, json_cols=None, numpy_cols=None, encoding='utf-8-sig'):
    """
    path: CSVパス（.gzなど圧縮もOK：compressionは自動推定）
    json_cols: JSON文字列として保存したカラム名のリスト（例: 配列/辞書）
    numpy_cols: その中でNumPy配列に戻したいカラム名（リスト→np.arrayへ）

    return: (df, records_list_of_dicts)
    """
    df = pd.read_csv(path, encoding=encoding)  # 圧縮は自動推定されます（.gzなど）
    json_cols = json_cols or []
    numpy_cols = numpy_cols or []

    # JSON復元
    for c in json_cols:
        if c in df.columns:
            df[c] = df[c].apply(_json_load_if_needed)

    # NumPy配列化
    for c in numpy_cols:
        if c in df.columns:
            df[c] = df[c].apply(lambda v: np.array(v) if isinstance(v, list) else v)

    return df, df.to_dict(orient='records')



# 強度ベースNCCによるレジストレーション結果の表示
def ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return np.sum(a * b) / (
        np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b**2)) + 1e-8
    )

def affine_transform(img, params):
    a, b, c, d, tx, ty = params
    M = np.array([[a, b, tx],
                  [c, d, ty]], dtype=np.float32)
    h, w = img.shape
    return cv2.warpAffine(img, M, (w, h))

def objective_affine(params, ref, mov):
    warped = affine_transform(mov, params)
    return -ncc(ref, warped)

# スケーリングなしアフィンレジストレーション

def warp_rot_shear_trans(img, params):
    tx, ty, theta, k = params

    c, s = np.cos(theta), np.sin(theta)

    # 回転 + せん断（スケールなし）
    A = np.array([
        [c, k*c - s],
        [s, k*s + c]
    ], dtype=np.float32)

    M = np.hstack([A, [[tx],[ty]]])

    h, w = img.shape
    return cv2.warpAffine(img, M, (w, h))

def objective(params, ref, mov):
    warped = warp_rot_shear_trans(mov, params)
    return -ncc(ref, warped)

def segment_time_series(emg_6x6: np.ndarray, window: int, hop: int) -> np.ndarray:
    """
    (n_samples, 6, 6) を窓分割して (n_windows, window, 6, 6) へ。
    """
    n, _, _ = emg_6x6.shape
    idx_starts = np.arange(0, max(1, n - window + 1), hop)
    segs = np.stack([emg_6x6[s:s+window] for s in idx_starts if s + window <= n], axis=0)
    # segs: (n_windows, window, 6, 6)
    return segs




# === main ===
def main(subject="takeuchi", registration='affine', icc_r = False):
    print("セッション間")
    emg_list_original = []  # 各要素: shape=(T,8,8)
    y_list_original   = []  # ファイル名から抽出したラベル（長さ = 試行数）
    for j in range(7):
        for k in range(5):
            try:
                file_name = file_name_output(subject=subject, hand='right', electrode_place="original", gesture=j+1, trial=k+1)
                path = '../../data/highMVC/' + file_name
                encoding = 'utf-8-sig'  # または 'utf-16'
                df = pd.read_csv(path, encoding=encoding, sep=';', header=None) 

                # ==== EMGデータの抽出 ====
                time = df.iloc[:, 0].values  # 時刻 [s]
                emg_data = df.iloc[:, 1:65].values  # shape: (time, 64)

                # ==== 基本パラメータ ====
                fs = int(1 / np.mean(np.diff(time)))  # サンプリング周波数
                emg_data = remove_power_line_harmonics(emg_data, fs=fs, fundamental=60.0, Q=30.0)
                filtered_emg = butter_bandpass_filter(emg_data, fs=fs, low_hz=20.0, high_hz=450.0, order=4)
                emg_data = filtered_emg.reshape(-1,8,8)  # shape: (time, 64)
                
                emg_list_original.append(emg_data)
                y_list_original.append(j+1)
            except FileNotFoundError:
                pass

    emg_list_original2 = []
    y_list_original2   = []
    for j in range(7):
        for k in range(5):
            try:
                file_name = file_name_output(subject=subject, hand='right', electrode_place="original2", gesture=j+1, trial=k+1)
                path = '../../data/highMVC/' + file_name
                encoding = 'utf-8-sig'  # または 'utf-16'
                df = pd.read_csv(path, encoding=encoding, sep=';', header=None) 

                # ==== EMGデータの抽出 ====
                time = df.iloc[:, 0].values  # 時刻 [s]
                emg_data = df.iloc[:, 1:65].values  # shape: (time, 64)

                # ==== 基本パラメータ ====
                fs = int(1 / np.mean(np.diff(time)))  # サンプリング周波数
                emg_data = remove_power_line_harmonics(emg_data, fs=fs, fundamental=60.0, Q=30.0)
                filtered_emg = butter_bandpass_filter(emg_data, fs=fs, low_hz=20.0, high_hz=450.0, order=4)
                emg_data = filtered_emg.reshape(-1,8,8)  # shape: (time, 64)
                
                emg_list_original2.append(emg_data)
                y_list_original2.append(j+1)
            except FileNotFoundError:
                pass

    emg_list_downleft5mm = []
    y_list_downleft5mm   = []
    for j in range(7):
        for k in range(5):
            try:
                file_name = file_name_output(subject=subject, hand='right', electrode_place="downleft5mm", gesture=j+1, trial=k+1)
                path = '../../data/highMVC/' + file_name
                encoding = 'utf-8-sig'  # または 'utf-16'
                df = pd.read_csv(path, encoding=encoding, sep=';', header=None)

                # ==== EMGデータの抽出 ====
                time = df.iloc[:, 0].values  # 時刻 [s]
                emg_data = df.iloc[:, 1:65].values  # shape: (time, 64)

                # ==== 基本パラメータ ====
                fs = int(1 / np.mean(np.diff(time)))  # サンプリング周波数
                emg_data = remove_power_line_harmonics(emg_data, fs=fs, fundamental=60.0, Q=30.0)
                filtered_emg = butter_bandpass_filter(emg_data, fs=fs, low_hz=20.0, high_hz=450.0, order=4)
                emg_data = filtered_emg.reshape(-1,8,8)  # shape: (time, 64)

                emg_list_downleft5mm.append(emg_data)
                y_list_downleft5mm.append(j+1)
            except FileNotFoundError:
                pass

    emg_list_downleft10mm = []
    y_list_downleft10mm   = []
    for j in range(7):
        for k in range(5):
            try:
                file_name = file_name_output(subject=subject, hand='right', electrode_place="downleft10mm", gesture=j+1, trial=k+1)
                path = '../../data/highMVC/' + file_name
                encoding = 'utf-8-sig'  # または 'utf-16'
                df = pd.read_csv(path, encoding=encoding, sep=';', header=None)

                # ==== EMGデータの抽出 ====
                time = df.iloc[:, 0].values  # 時刻 [s]
                emg_data = df.iloc[:, 1:65].values  # shape: (time, 64)

                # ==== 基本パラメータ ====
                fs = int(1 / np.mean(np.diff(time)))  # サンプリング周波数
                emg_data = remove_power_line_harmonics(emg_data, fs=fs, fundamental=60.0, Q=30.0)
                filtered_emg = butter_bandpass_filter(emg_data, fs=fs, low_hz=20.0, high_hz=450.0, order=4)
                emg_data = filtered_emg.reshape(-1,8,8)  # shape: (time, 64)

                emg_list_downleft10mm.append(emg_data)
                y_list_downleft10mm.append(j+1)
            except FileNotFoundError:
                pass

    emg_list_clockwise = []
    y_list_clockwise   = []
    for j in range(7):
        for k in range(5):
            try:
                file_name = file_name_output(subject=subject, hand='right', electrode_place="clockwise", gesture=j+1, trial=k+1)
                path = '../../data/highMVC/' + file_name
                encoding = 'utf-8-sig'  # または 'utf-16'
                df = pd.read_csv(path, encoding=encoding, sep=';', header=None)

                # ==== EMGデータの抽出 ====
                time = df.iloc[:, 0].values  # 時刻 [s]
                emg_data = df.iloc[:, 1:65].values  # shape: (time, 64)

                # ==== 基本パラメータ ====
                fs = int(1 / np.mean(np.diff(time)))  # サンプリング周波数
                emg_data = remove_power_line_harmonics(emg_data, fs=fs, fundamental=60.0, Q=30.0)
                filtered_emg = butter_bandpass_filter(emg_data, fs=fs, low_hz=20.0, high_hz=450.0, order=4)
                emg_data = filtered_emg.reshape(-1,8,8)  # shape: (time, 64)

                emg_list_clockwise.append(emg_data)
                y_list_clockwise.append(j+1)
            except FileNotFoundError:
                pass

    window = 200   # サンプル幅（例：100ms @ 2kHz）
    hop    = 50    # ホップ（例：25ms）
    kind = 'rms'  # 'rms' or 'mav' or 'waveform_length' etc.
    fs = 2000
    window = int(window * (fs/1000))
    hop = int(hop * (fs/1000))
    # X_train = []
    # y_train = []
    sizes_te = []
    threshold = 0
    threshold2 = 0.0013

    # 学習データのアライメントマップ作成
    for i, (emg_original_8x8, label) in enumerate(zip(emg_list_original, y_list_original)):
        # --- ラベル（学習用）：あなたのラベリングに置き換えてください ---
        # 中央6×6から特徴抽出した個数に合わせる必要があります。
        tmp_X = emg_original_8x8  # (n_samples, 8, 8)
        # tmp_X = extract_center_6x6(tmp_X)  # (n_samples, 6, 6)
        tmp_X = segment_time_series(tmp_X, window=window, hop=hop)  # (n_windows, window, 8, 8)

        # ========= チャネルごとに正規化 =========
        mean = np.mean(tmp_X.reshape(-1,8,8), axis=0)
        std = np.std(tmp_X.reshape(-1,8,8), axis=0) + 1e-8
        tmp_X = (tmp_X - mean.reshape(1, 1, 8, 8)) / std.reshape(1, 1, 8, 8)
        # ====================================

        # tmp_X = tmp_X.reshape(tmp_X.shape[0], tmp_X.shape[1], 36)  # (n_windows, window, 36) #

        # 特徴量抽出
        rms = [rms_feat(x) for x in tmp_X]
        wl = [waveform_length(x) for x in tmp_X]
        zc = [zero_crossings(x, threshold) for x in tmp_X]
        ssc = [slope_sign_changes(x, threshold) for x in tmp_X]
        wamp = [wamp_feat(x, threshold2) for x in tmp_X]
        tmp_X = rms
        # tmp_X = extract_features(emg_original_8x8, FeatureSpec(kind=kind, window=window, hop=hop))  # (n_windows, 36)
        n_windows = len(tmp_X)
        # 例：ダミーの 3 クラスを周回（実際はジェスチャーIDに差し替え）
        tmp_y = [int(label)-1 for i in range(n_windows)]
        sizes_te.append(n_windows)

        if len(tmp_y) != len(tmp_X):
            raise ValueError(f"y_original length ({len(tmp_y)}) must match number of windows ({len(tmp_X)})")
        
        if i == 0:
            X_original = tmp_X
            y_original = tmp_y
        else:
            X_original = np.vstack([X_original, tmp_X])
            y_original = np.hstack([y_original, tmp_y])


    X_original_aligment = X_original

    #  推論時のアライメントマップ平均を1秒毎に計算し、３秒間で平均(補間なし)
    gestures = np.unique(y_original) + 1

    original_trial_alignment_maps_without_interp_1sec = []
    for gesture in gestures:
        z_list = []
        trial_list = []
        iterator = X_original_aligment[y_original==gesture-1]
        i = 0
        for tmp_X_original_aligment in iterator[:]:
            z_list.append(tmp_X_original_aligment)
            i += 1
            if i == 19:  # トライアル数で分割
                trial_list.append(np.mean(np.array(z_list), axis=0))
                z_list = []
                i = 0
        original_trial_alignment_maps_without_interp_1sec.append(trial_list)

    # 推論データのアライメントマップ作成
    for i, (emg_original2_8x8, label) in enumerate(zip(emg_list_original2, y_list_original2)):
        # --- ラベル（学習用）：あなたのラベリングに置き換えてください ---
        # 中央6×6から特徴抽出した個数に合わせる必要があります。
        tmp_X = emg_original2_8x8  # (n_samples, 8, 8)
        # tmp_X = extract_center_6x6(tmp_X)  # (n_samples, 6, 6)
        tmp_X = segment_time_series(tmp_X, window=window, hop=hop)  # (n_windows, window, 8, 8)

        # ========= チャネルごとに正規化 =========
        mean = np.mean(tmp_X.reshape(-1,8,8), axis=0)
        std = np.std(tmp_X.reshape(-1,8,8), axis=0) + 1e-8
        tmp_X = (tmp_X - mean.reshape(1, 1, 8, 8)) / std.reshape(1, 1, 8, 8)
        # ====================================

        # tmp_X = tmp_X.reshape(tmp_X.shape[0], tmp_X.shape[1], 36)  # (n_windows, window, 36) #

        # 特徴量抽出
        rms = [rms_feat(x) for x in tmp_X]
        wl = [waveform_length(x) for x in tmp_X]
        zc = [zero_crossings(x, threshold) for x in tmp_X]
        ssc = [slope_sign_changes(x, threshold) for x in tmp_X]
        wamp = [wamp_feat(x, threshold2) for x in tmp_X]
        tmp_X = rms
        # tmp_X = extract_features(emg_original2_8x8, FeatureSpec(kind=kind, window=window, hop=hop))  # (n_windows, 36)
        n_windows = len(tmp_X)
        # 例：ダミーの 3 クラスを周回（実際はジェスチャーIDに差し替え）
        tmp_y = [int(label)-1 for i in range(n_windows)]
        sizes_te.append(n_windows)

        if len(tmp_y) != len(tmp_X):
            raise ValueError(f"y_original2 length ({len(tmp_y)}) must match number of windows ({len(tmp_X)})")
        
        if i == 0:
            X_original2 = tmp_X
            y_original2 = tmp_y
        else:
            X_original2 = np.vstack([X_original2, tmp_X])
            y_original2 = np.hstack([y_original2, tmp_y])


    X_original2_aligment = X_original2

    #  推論時のアライメントマップ平均を1秒毎に計算し、３秒間で平均(補間なし)
    gestures = np.unique(y_original2) + 1

    original2_trial_alignment_maps_without_interp_1sec = []
    for gesture in gestures:
        z_list = []
        trial_list = []
        iterator = X_original2_aligment[y_original2==gesture-1]
        i = 0
        for tmp_X_original2_aligment in iterator[:]:
            z_list.append(tmp_X_original2_aligment)
            i += 1
            if i == 19:  # トライアル数で分割
                trial_list.append(np.mean(np.array(z_list), axis=0))
                z_list = []
                i = 0
        original2_trial_alignment_maps_without_interp_1sec.append(trial_list)


    # 推論データのアライメントマップ作成
    for i, (emg_downleft5mm_8x8, label) in enumerate(zip(emg_list_downleft5mm, y_list_downleft5mm)):
        # --- ラベル（学習用）：あなたのラベリングに置き換えてください ---
        # 中央6×6から特徴抽出した個数に合わせる必要があります。
        tmp_X = emg_downleft5mm_8x8  # (n_samples, 8, 8)
        # tmp_X = extract_center_6x6(tmp_X)  # (n_samples, 6, 6)
        tmp_X = segment_time_series(tmp_X, window=window, hop=hop)  # (n_windows, window, 8, 8)

        # ========= チャネルごとに正規化 =========
        mean = np.mean(tmp_X.reshape(-1,8,8), axis=0)
        std = np.std(tmp_X.reshape(-1,8,8), axis=0) + 1e-8
        tmp_X = (tmp_X - mean.reshape(1, 1, 8, 8)) / std.reshape(1, 1, 8, 8)
        # ====================================

        # tmp_X = tmp_X.reshape(tmp_X.shape[0], tmp_X.shape[1], 36)  # (n_windows, window, 36) #

        # 特徴量抽出
        rms = [rms_feat(x) for x in tmp_X]
        wl = [waveform_length(x) for x in tmp_X]
        zc = [zero_crossings(x, threshold) for x in tmp_X]
        ssc = [slope_sign_changes(x, threshold) for x in tmp_X]
        wamp = [wamp_feat(x, threshold2) for x in tmp_X]
        tmp_X = rms
        # tmp_X = extract_features(emg_downleft5mm_8x8, FeatureSpec(kind=kind, window=window, hop=hop))  # (n_windows, 36)
        n_windows = len(tmp_X)
        # 例：ダミーの 3 クラスを周回（実際はジェスチャーIDに差し替え）
        tmp_y = [int(label)-1 for i in range(n_windows)]
        sizes_te.append(n_windows)

        if len(tmp_y) != len(tmp_X):
            raise ValueError(f"y_downleft5mm length ({len(tmp_y)}) must match number of windows ({len(tmp_X)})")
        
        if i == 0:
            X_downleft5mm = tmp_X
            y_downleft5mm = tmp_y
        else:
            X_downleft5mm = np.vstack([X_downleft5mm, tmp_X])
            y_downleft5mm = np.hstack([y_downleft5mm, tmp_y])


    X_downleft5mm_aligment = X_downleft5mm

    #  推論時のアライメントマップ平均を1秒毎に計算し、３秒間で平均(補間なし)
    gestures = np.unique(y_downleft5mm) + 1

    downleft5mm_trial_alignment_maps_without_interp_1sec = []
    for gesture in gestures:
        z_list = []
        trial_list = []
        iterator = X_downleft5mm_aligment[y_downleft5mm==gesture-1]
        i = 0
        for tmp_X_downleft5mm_aligment in iterator[:]:
            z_list.append(tmp_X_downleft5mm_aligment)
            i += 1
            if i == 19:  # トライアル数で分割
                trial_list.append(np.mean(np.array(z_list), axis=0))
                z_list = []
                i = 0
        downleft5mm_trial_alignment_maps_without_interp_1sec.append(trial_list)

    # trans10
    for i, (emg_downleft10mm_8x8, label) in enumerate(zip(emg_list_downleft10mm, y_list_downleft10mm)):
        # --- ラベル（学習用）：あなたのラベリングに置き換えてください ---
        # 中央6×6から特徴抽出した個数に合わせる必要があります。
        tmp_X = emg_downleft10mm_8x8  # (n_samples, 8, 8)
        # tmp_X = extract_center_6x6(tmp_X)  # (n_samples, 6, 6)
        tmp_X = segment_time_series(tmp_X, window=window, hop=hop)  # (n_windows, window, 8, 8)

        # ========= チャネルごとに正規化 =========
        mean = np.mean(tmp_X.reshape(-1,8,8), axis=0)
        std = np.std(tmp_X.reshape(-1,8,8), axis=0) + 1e-8
        tmp_X = (tmp_X - mean.reshape(1, 1, 8, 8)) / std.reshape(1, 1, 8, 8)
        # ====================================

        # tmp_X = tmp_X.reshape(tmp_X.shape[0], tmp_X.shape[1], 36)  # (n_windows, window, 36) #

        # 特徴量抽出
        rms = [rms_feat(x) for x in tmp_X]
        wl = [waveform_length(x) for x in tmp_X]
        zc = [zero_crossings(x, threshold) for x in tmp_X]
        ssc = [slope_sign_changes(x, threshold) for x in tmp_X]
        wamp = [wamp_feat(x, threshold2) for x in tmp_X]
        tmp_X = rms
        # tmp_X = extract_features(emg_downleft10mm_8x8, FeatureSpec(kind=kind, window=window, hop=hop))  # (n_windows, 36)
        n_windows = len(tmp_X)
        # 例：ダミーの 3 クラスを周回（実際はジェスチャーIDに差し替え）
        tmp_y = [int(label)-1 for i in range(n_windows)]
        sizes_te.append(n_windows)

        if len(tmp_y) != len(tmp_X):
            raise ValueError(f"y_downleft10mm length ({len(tmp_y)}) must match number of windows ({len(tmp_X)})")
        
        if i == 0:
            X_downleft10mm = tmp_X
            y_downleft10mm = tmp_y
        else:
            X_downleft10mm = np.vstack([X_downleft10mm, tmp_X])
            y_downleft10mm = np.hstack([y_downleft10mm, tmp_y])


    X_downleft10mm_aligment = X_downleft10mm

    #  推論時のアライメントマップ平均を1秒毎に計算し、３秒間で平均(補間なし)
    gestures = np.unique(y_downleft10mm) + 1

    downleft10mm_trial_alignment_maps_without_interp_1sec = []
    for gesture in gestures:
        z_list = []
        trial_list = []
        iterator = X_downleft10mm_aligment[y_downleft10mm==gesture-1]
        i = 0
        for tmp_X_downleft10mm_aligment in iterator[:]:
            z_list.append(tmp_X_downleft10mm_aligment)
            i += 1
            if i == 19:  # トライアル数で分割
                trial_list.append(np.mean(np.array(z_list), axis=0))
                z_list = []
                i = 0
        downleft10mm_trial_alignment_maps_without_interp_1sec.append(trial_list)

    
    # rotation
    for i, (emg_clockwise_8x8, label) in enumerate(zip(emg_list_clockwise, y_list_clockwise)):
        # --- ラベル（学習用）：あなたのラベリングに置き換えてください ---
        # 中央6×6から特徴抽出した個数に合わせる必要があります。
        tmp_X = emg_clockwise_8x8  # (n_samples, 8, 8)
        # tmp_X = extract_center_6x6(tmp_X)  # (n_samples, 6, 6)
        tmp_X = segment_time_series(tmp_X, window=window, hop=hop)  # (n_windows, window, 8, 8)

        # ========= チャネルごとに正規化 =========
        mean = np.mean(tmp_X.reshape(-1,8,8), axis=0)
        std = np.std(tmp_X.reshape(-1,8,8), axis=0) + 1e-8
        tmp_X = (tmp_X - mean.reshape(1, 1, 8, 8)) / std.reshape(1, 1, 8, 8)
        # ====================================

        # tmp_X = tmp_X.reshape(tmp_X.shape[0], tmp_X.shape[1], 36)  # (n_windows, window, 36) #

        # 特徴量抽出
        rms = [rms_feat(x) for x in tmp_X]
        wl = [waveform_length(x) for x in tmp_X]
        zc = [zero_crossings(x, threshold) for x in tmp_X]
        ssc = [slope_sign_changes(x, threshold) for x in tmp_X]
        wamp = [wamp_feat(x, threshold2) for x in tmp_X]
        tmp_X = rms
        # tmp_X = extract_features(emg_clockwise_8x8, FeatureSpec(kind=kind, window=window, hop=hop))  # (n_windows, 36)
        n_windows = len(tmp_X)
        # 例：ダミーの 3 クラスを周回（実際はジェスチャーIDに差し替え）
        tmp_y = [int(label)-1 for i in range(n_windows)]
        sizes_te.append(n_windows)

        if len(tmp_y) != len(tmp_X):
            raise ValueError(f"y_clockwise length ({len(tmp_y)}) must match number of windows ({len(tmp_X)})")
        
        if i == 0:
            X_clockwise = tmp_X
            y_clockwise = tmp_y
        else:
            X_clockwise = np.vstack([X_clockwise, tmp_X])
            y_clockwise = np.hstack([y_clockwise, tmp_y])


    X_clockwise_aligment = X_clockwise

    #  推論時のアライメントマップ平均を1秒毎に計算し、３秒間で平均(補間なし)
    gestures = np.unique(y_clockwise) + 1

    clockwise_trial_alignment_maps_without_interp_1sec = []
    for gesture in gestures:
        z_list = []
        trial_list = []
        iterator = X_clockwise_aligment[y_clockwise==gesture-1]
        i = 0
        for tmp_X_clockwise_aligment in iterator[:]:
            z_list.append(tmp_X_clockwise_aligment)
            i += 1
            if i == 19:  # トライアル数で分割
                trial_list.append(np.mean(np.array(z_list), axis=0))
                z_list = []
                i = 0
        clockwise_trial_alignment_maps_without_interp_1sec.append(trial_list)

    # セッション間レジストレーションの実行
    theta_list_all = []
    dx_list_all = []
    dy_list_all = []
    scale_list_all = []
    icc2_list_all = []
    icc2k_list_all = []
    r_list_all = []
    for electrode_place in ["original2", "downleft5mm", "downleft10mm", "clockwise"]:
        theta_list = []
        dx_list = []
        dy_list = []
        scale_list = []
        icc2_list = []
        icc2k_list = []
        r_list = []
        for gesture in range(1,8):
            theta_gesture_list = []
            dx_gesture_list = []
            dy_gesture_list = []
            scale_gesture_list = []
            for trial1 in range(1,16):  # トライアル数
                for trial2 in range(trial1,16):
                    scalar = MinMaxScaler()
                    img_ref = (scalar.fit_transform(original_trial_alignment_maps_without_interp_1sec[gesture-1][trial1-1].reshape(-1,1)).reshape(original_trial_alignment_maps_without_interp_1sec[gesture-1][trial1-1].shape)*255).astype(np.float32)
                    if electrode_place == "original2":
                        img_test = (scalar.fit_transform(original2_trial_alignment_maps_without_interp_1sec[gesture-1][trial2-1].reshape(-1,1)).reshape(original2_trial_alignment_maps_without_interp_1sec[gesture-1][trial2-1].shape)*255).astype(np.float32)
                    elif electrode_place == "downleft5mm":
                        img_test = (scalar.fit_transform(downleft5mm_trial_alignment_maps_without_interp_1sec[gesture-1][trial2-1].reshape(-1,1)).reshape(downleft5mm_trial_alignment_maps_without_interp_1sec[gesture-1][trial2-1].shape)*255).astype(np.float32)
                    elif electrode_place == "downleft10mm":
                        img_test = (scalar.fit_transform(downleft10mm_trial_alignment_maps_without_interp_1sec[gesture-1][trial2-1].reshape(-1,1)).reshape(downleft10mm_trial_alignment_maps_without_interp_1sec[gesture-1][trial2-1].shape)*255).astype(np.float32)
                    elif electrode_place == "clockwise":
                        img_test = (scalar.fit_transform(clockwise_trial_alignment_maps_without_interp_1sec[gesture-1][trial2-1].reshape(-1,1)).reshape(clockwise_trial_alignment_maps_without_interp_1sec[gesture-1][trial2-1].shape)*255).astype(np.float32)
                    
                    # try:
                    if  1 == 1:
                        if registration == 'affine':
                            # 初期値：単位行列（≒剛体）
                            init = [1, 0, 0, 1, 0, 0]

                            res = minimize(
                                objective_affine,
                                x0=init,
                                args=(img_test, img_ref),
                                method="Powell" #L-BFGS-B",   # bounds対応
                                #bounds=bounds
                            )

                            a, b, c, d, tx, ty = res.x

                            theta_rad = np.arctan2(c, a)
                            theta_deg = np.degrees(theta_rad)
                            sx = np.sqrt(a**2 + c**2)
                            sy = np.sqrt(b**2 + d**2)
                            shear = (a*b + c*d) / (sx*sy)
                            

                            theta_gesture_list.append(theta_deg)
                            dx_gesture_list.append(tx*10)  # mm単位
                            dy_gesture_list.append(ty*10)  # mm単位
                            scale_gesture_list.append(sx)
                            
                            theta_list.append(theta_deg)
                            dx_list.append(tx*10)  # mm単位
                            dy_list.append(ty*10)  # mm単位
                            scale_list.append(sx)

                            theta_list_all.append(theta_deg)
                            dx_list_all.append(tx*10)  # mm単位
                            dy_list_all.append(ty*10)  # mm単位
                            scale_list_all.append(sx)


                        if icc_r:
                            feature_name = "wl"
                            subject_name = subject

                            window = 200
                            hop = 50
                            threshold = 0
                            threshold2 = 0.0013

                            spacing = 1.0 # 電極間距離の単位
                            rotate_about = "grid_center"  # or "top_left"
                            mode = "extrapolate"   # or "clip"

                            if electrode_place == "original":
                                dx_ref = 0.0
                                dy_ref = 0.0
                                theta_ref = 0.0
                            elif electrode_place == "upright":
                                dx_ref = 1.0
                                dy_ref = 1.0
                                theta_ref = 0.0
                            elif electrode_place == "downright":
                                dx_ref = -1.0
                                dy_ref = 1.0
                                theta_ref = 0.0
                            elif electrode_place == "downleft":
                                dx_ref = -1.0
                                dy_ref = -1.0
                                theta_ref = 0.0
                            elif electrode_place == "upleft":
                                dx_ref = 1.0
                                dy_ref = -1.0
                                theta_ref = 0.0
                            elif electrode_place == "clockwise":
                                dx_ref = 0.0
                                dy_ref = 0.0
                                theta_ref = np.radians(10.0)
                            elif electrode_place == "anticlockwise":
                                dx_ref = 0.0
                                dy_ref = 0.0
                                theta_ref = np.radians(-10.0)

                            dx_test = tx/n_interp*10  # mm→チャネル単位
                            dy_test = ty/n_interp*10  # mm→チャネル単位
                            theta_test = np.radians(theta_deg)  # degree

                            session_list= [(dx_ref, dy_ref, theta_ref), (dx_test, dy_test, theta_test)]
                            features_ref_icc = []
                            for i, session in enumerate(session_list):
                                for j in range(7):
                                    for k in range(5):
                                        file_name = file_name_output(subject=subject_name, hand='right', electrode_place=electrode_place, gesture=j+1, trial=k+1, output=False)
                                        path = '../../data/highMVC/' + file_name
                                        encoding = 'utf-8-sig'  # または 'utf-16'
                                        df = pd.read_csv(path, encoding=encoding, sep=';', header=None)

                                        # ==== EMGデータの抽出 ====
                                        time = df.iloc[:, 0].values  # 時刻 [s]
                                        emg_data = df.iloc[:, 1:65].values.T  # shape: (64, time)

                                        # ==== 基本パラメータ ====
                                        fs = int(1 / np.mean(np.diff(time)))  # サンプリング周波数

                                        filtered_emg = butter_bandpass_filter(emg_data, fs=fs, low_hz=20.0, high_hz=400.0, order=4)
                                        emg_data = filtered_emg.T.reshape(-1,8,8)  # shape: (time, 64)

                                        # 移動
                                        mapper = GridSubsetMapper(spacing=spacing, rotate_about=rotate_about)
                                        tmp_X = mapper.transform(emg_data, dx=session[0], dy=session[1], theta=session[2], mode=mode)  # (n,6,6)

                                        tmp_X = segment_time_series(tmp_X, window=window, hop=hop)  # (n_windows, window, 6, 6)
                                        tmp_X = tmp_X.reshape(tmp_X.shape[0], tmp_X.shape[1], 36)  # (n_windows, window, 36) #

                                        if feature_name == "rms":
                                            feature = [rms_feat(x) for x in tmp_X]
                                        elif feature_name == "wl":
                                            feature = [waveform_length(x) for x in tmp_X]
                                        elif feature_name == "zc":
                                            feature = [zero_crossings(x, threshold) for x in tmp_X]
                                        elif feature_name == "ssc":
                                            feature = [slope_sign_changes(x, threshold) for x in tmp_X]
                                        elif feature_name == "wamp":
                                            feature = [wamp_feat(x, threshold2) for x in tmp_X]

                                        if i==0 and j==0 and k==0:
                                            feature_df_r = pd.DataFrame(feature, columns=[f'ch{k+1}' for k in range(len(feature[0]))])
                                            feature_df_r['window'] = np.arange(feature_df_r.shape[0])
                                            feature_df_r['subject'] = subject_name
                                            feature_df_r['electrode_place'] = electrode_place
                                            feature_df_r['gesture'] = j+1
                                            feature_df_r['trial'] = k+1
                                            feature_df_r['session'] = i+1
                                        else:
                                            tmp_df_r = pd.DataFrame(feature, columns=[f'ch{k+1}' for k in range(len(feature[0]))])
                                            tmp_df_r['window'] = np.arange(tmp_df_r.shape[0])
                                            tmp_df_r['subject'] = subject_name
                                            tmp_df_r['electrode_place'] = electrode_place
                                            tmp_df_r['gesture'] = j+1
                                            tmp_df_r['trial'] = k+1
                                            tmp_df_r['session'] = i+1
                                            feature_df_r = pd.concat([feature_df_r, tmp_df_r], axis=0)

                                        # MinMaxScaler
                                        feature_list = []
                                        for ch in range(np.array(feature).shape[1]):
                                            scaler = MinMaxScaler()
                                            feature_1ch = scaler.fit_transform(np.array(feature)[:,ch].reshape(-1,1))
                                            feature_list.append(feature_1ch)
                                        feature = np.hstack(feature_list)

                                        if i==0 and j==0 and k==0:
                                            feature_df_icc = pd.DataFrame(feature, columns=[f'ch{k+1}' for k in range(len(feature[0]))])
                                            feature_df_icc['window'] = np.arange(feature_df_icc.shape[0])
                                            feature_df_icc['subject'] = subject_name
                                            feature_df_icc['electrode_place'] = electrode_place
                                            feature_df_icc['gesture'] = j+1
                                            feature_df_icc['trial'] = k+1
                                            feature_df_icc['session'] = i+1
                                        else:
                                            tmp_df_icc = pd.DataFrame(feature, columns=[f'ch{k+1}' for k in range(len(feature[0]))])
                                            tmp_df_icc['window'] = np.arange(tmp_df_icc.shape[0])
                                            tmp_df_icc['subject'] = subject_name
                                            tmp_df_icc['electrode_place'] = electrode_place
                                            tmp_df_icc['gesture'] = j+1
                                            tmp_df_icc['trial'] = k+1
                                            tmp_df_icc['session'] = i+1
                                            feature_df_icc = pd.concat([feature_df_icc, tmp_df_icc], axis=0)

                            feature_df_r_long = pd.melt(feature_df_r, id_vars=['subject', 'electrode_place', 'gesture', 'trial', 'session', 'window'],
                                                    value_vars=[f'ch{k+1}' for k in range(36)], var_name='channel', value_name='value')
                            feature_df_icc_long = pd.melt(feature_df_icc, id_vars=['subject', 'electrode_place', 'gesture', 'trial', 'session', 'window'],
                                                    value_vars=[f'ch{k+1}' for k in range(36)], var_name='channel', value_name='value')
                            # icc = pg.intraclass_corr(data=feature_df_long, targets='window', raters='session', ratings='value')
                            # icc2 = icc[icc['Type']=='ICC2']['ICC'].values[0]
                            # icc2_list.append(icc2)
                            # icc2_list_all.append(icc2)
                            # r, p = pearsonr(feature_df_long[feature_df_long['session']==1]['value'],
                            #     feature_df_long[feature_df_long['session']==2]['value'])
                            # r_list.append(r)
                            # r_list_all.append(r)
                            warnings.simplefilter('ignore')
                            r_list_each = []
                            for gesture in feature_df_r_long['gesture'].unique():
                                for channel in feature_df_r_long['channel'].unique():
                                    for trial in feature_df_r_long['trial'].unique():
                                        data = feature_df_r_long[feature_df_r_long['gesture']==gesture][feature_df_r_long['channel']==channel][feature_df_r_long['trial']==trial]
                                        data_session1 = data[data['session']==1]['value']
                                        data_session2 = data[data['session']==2]['value']
                                        min_len = min(len(data_session1), len(data_session2))
                                        if min_len > 0:
                                            r, _ = pearsonr(data_session1[:min_len], data_session2[:min_len])
                                            r_list_each.append(r)
                            icc2_list_each = []
                            icc2k_list_each = []
                            for gesture in feature_df_icc_long['gesture'].unique():
                                for channel in feature_df_icc_long['channel'].unique():
                                    data = feature_df_icc_long[feature_df_icc_long['gesture']==gesture][feature_df_icc_long['channel']==channel]
                                    icc = pg.intraclass_corr(data=data, targets='trial', raters='session', ratings='value')
                                    icc2_value = icc[icc['Type']=='ICC2']['ICC'].values[0]
                                    icc2k_value = icc[icc['Type']=='ICC2k']['ICC'].values[0]
                                    if not np.isnan(icc2_value):
                                        icc2_list_each.append(icc2_value)
                                    if not np.isnan(icc2k_value):
                                        icc2k_list_each.append(icc2k_value)
                            icc2_list.append(icc2_list_each)
                            icc2_list_all.append(icc2_list_each)
                            icc2k_list.append(icc2k_list_each)
                            icc2k_list_all.append(icc2k_list_each)
                            r_list.append(r_list_each)
                            r_list_all.append(r_list_each)
                    # except:
                    #     print(f"電極配置: {electrode_place}, ジェスチャ: {gesture}でエラー")
                    #     continue
            coordinate_list = [theta_gesture_list, dx_gesture_list, dy_gesture_list]
            fig, ax = plt.subplots(figsize=(40,20), tight_layout=True)
            ax.boxplot(coordinate_list)
            ax.set_xticklabels(['theta','x','y'])
            plt.title('across_session_gesture' + str(gesture) + '_' + electrode_place)
            plt.rcParams['font.size'] = 58
            plt.ylim([-15,15])
            plt.grid()
            plt.savefig('line_plot_across_sessions_' + subject + '_gesture' + str(gesture) + '_' + electrode_place + '.png')
            # plt.show()
        theta_mean = np.mean(theta_list)
        dx_mean = np.mean(dx_list)
        dy_mean = np.mean(dy_list)
        scale_mean = np.mean(scale_list)
        print(f"電極配置: {electrode_place}")
        print("平均 回転角 [deg]:", theta_mean)
        print("平均 並進 [mm]:   dx =", dx_mean, ", dy =", dy_mean)
        print("平均 スケール   :", scale_mean)

        coordinate_list = [theta_list, dx_list, dy_list]
        fig, ax = plt.subplots(figsize=(40,20), tight_layout=True)
        ax.boxplot(coordinate_list)
        ax.set_xticklabels(['theta','x','y'])
        plt.title('across_session_' + electrode_place)
        plt.rcParams['font.size'] = 58
        plt.ylim([-15,15])
        plt.grid()
        plt.savefig('line_plot_across_sessions_' + subject + '_' + electrode_place + '.png')
        # plt.show()

        if icc_r:
            icc2_mean = np.mean(icc2_list)
            icc2k_mean = np.mean(icc2k_list)
            r_mean = np.mean(r_list)
            print("平均 ICC2:", icc2_mean)
            print("平均 ICC2k:", icc2k_mean)
            print("平均 Pearson r:", r_mean)

    theta_mean_all = np.mean(theta_list_all)
    dx_mean_all = np.mean(dx_list_all)
    dy_mean_all = np.mean(dy_list_all)
    scale_mean_all = np.mean(scale_list_all)
    print("all electrode places:")
    print("平均 回転角 [deg]:", theta_mean_all)
    print("平均 並進 [mm]:   dx =", dx_mean_all, ", dy =", dy_mean_all)
    print("平均 スケール   :", scale_mean_all)

    coordinate_list = [theta_list, dx_list, dy_list]
    fig, ax = plt.subplots(figsize=(40,20), tight_layout=True)
    ax.boxplot(coordinate_list)
    ax.set_xticklabels(['theta','x','y'])
    plt.title('across_session_all electrode places')
    plt.rcParams['font.size'] = 58
    plt.ylim([-15,15])
    plt.grid()
    plt.savefig('line_plot_across_sessions_' + subject + '_all.png')
    # plt.show()

    if icc_r:
        icc2_mean_all = np.mean(icc2_list_all)
        icc2k_mean_all = np.mean(icc2k_list_all)
        r_mean_all = np.mean(r_list_all)
        print("平均 ICC2:", icc2_mean_all)
        print("平均 ICC2k:", icc2k_mean_all)
        print("平均 Pearson r:", r_mean_all)