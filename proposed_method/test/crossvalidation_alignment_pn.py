import wfdb
import os
import glob
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy import signal
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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


# ----- ガウス関数定義 -----
def gaussian_2d(coord, A, x0, y0, sigma_x, sigma_y, theta, offset):
    x, y = coord #coord[:, 0], coord[:, 1]
    a = (np.cos(theta)**2) / (2*sigma_x**2) + (np.sin(theta)**2) / (2*sigma_y**2)
    b = -(np.sin(2*theta)) / (4*sigma_x**2) + (np.sin(2*theta)) / (4*sigma_y**2)
    c = (np.sin(theta)**2) / (2*sigma_x**2) + (np.cos(theta)**2) / (2*sigma_y**2)
    return A * np.exp(-(a*(x - x0)**2 + 2*b*(x - x0)*(y - y0) + c*(y - y0)**2)) + offset

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
    subject_list = ["garu"]
    hand = "right"
    dir_list = ["1-original", "2-upright", "3-downright", "4-downleft", "5-upleft", "6-clockwise", "7-anticlockwise"]

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

def ptp_feat(signal):
    return np.array(np.ptp(signal, axis=0))

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

# mixecc

# ==========================================
# 1. コスト関数計算用のパーツ（修正版）
# ==========================================

def compute_gradient_magnitude(image):
    dx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(dx, dy)

def thresholding(img, threshold=90):
    scalar = MinMaxScaler()
    img_scalared = (scalar.fit_transform(img.reshape(-1,1)).reshape(img.shape)*255).astype(np.float32)
    return ((img_scalared>threshold)*255).astype(np.uint8) # ((img_scalared>threshold)*255).astype(np.uint8)

def soft_thresholding(img, threshold=0.35, steepness=10.0):
    """
    シグモイド関数を用いて、微分可能な形で疑似的に二値化する。
    img: 入力画像 (正規化済みを想定)
    threshold: 閾値 (0.0 - 1.0)
    steepness: 変化の急峻さ (大きいほど硬い二値化に近づくが、最適化は不安定になる)
    """
    # 画像を0-1に正規化 (MinMaxではなく、想定最大値で割るのが安全だが今回は簡易実装)
    if img.max() > 0:
        img_norm = img / img.max()
    else:
        img_norm = img

    # シグモイド関数: 1 / (1 + exp(-k * (x - t)))
    # これにより、閾値付近で滑らかに0から1へ変化する
    return 1.0 / (1.0 + np.exp(-steepness * (img_norm - threshold))) * 255

def ncc(a, b):
    if np.std(a) == 0 or np.std(b) == 0: return 0.0
    a_mean = a - np.mean(a)
    b_mean = b - np.mean(b)
    num = np.sum(a_mean * b_mean)
    den = np.sqrt(np.sum(a_mean**2)) * np.sqrt(np.sum(b_mean**2)) + 1e-8
    return float(num / den)

def affine_transform(img, params):
    # params: [a, b, c, d, tx, ty]
    # 行列: [[a, b, tx], [c, d, ty]]
    # ※ 単位行列に近い初期値: [1, 0, 0, 1, 0, 0]
    M = np.array([[params[0], params[1], params[4]],
                  [params[2], params[3], params[5]]], dtype=np.float32)
    h, w = img.shape
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)

def calculate_hybrid_score(params, ref_img, mov_img, ref_grad_cache=None, alpha=0.1, beta=0.45, threshold=0.35, steepness=10.0):
    """
    目的関数: 輝度NCC × 勾配NCC
    """
    # 1. 画像を変形
    warped_mov = affine_transform(mov_img, params)
    
    # 2. 輝度NCC
    score_intensity = ncc(ref_img, warped_mov)
    
    # 3. 勾配計算 (refの勾配はキャッシュ可能)
    if ref_grad_cache is None:
        ref_grad = compute_gradient_magnitude(ref_img)
    else:
        ref_grad = ref_grad_cache
        
    # ワープ後の画像から勾配を計算
    warped_grad = compute_gradient_magnitude(warped_mov)
    
    # 4. 勾配NCC
    score_gradient = ncc(ref_grad, warped_grad)

    # 二値化
    # ref_thresholded = thresholding(ref_img)
    # warped_thresholded = thresholding(warped_mov)
    # score_thresholded = ncc(ref_thresholded, warped_thresholded)
    ref_thresholded = soft_thresholding(ref_img, threshold=threshold, steepness=steepness)
    warped_thresholded = soft_thresholding(warped_mov, threshold=threshold, steepness=steepness)
    score_thresholded = ncc(ref_thresholded, warped_thresholded)

    return float((1 - alpha - beta) * score_intensity + alpha * score_gradient + beta * score_thresholded)

# ==========================================
# 2. スクラッチ実装: 数値微分による最適化
# ==========================================

class CustomECCOptimizer:
    def __init__(self, learning_rate=1.0, max_iter=50, tolerance=1e-4, epsilon=np.array([1e-3, 1e-3, 1e-3, 1e-3, 0.1, 0.1])):
        self.lr = learning_rate       # 学習率（更新ステップの大きさ）
        self.max_iter = max_iter      # 最大反復回数
        self.tol = tolerance          # 収束判定閾値
        # 数値微分のための微小変動値
        self.epsilon = epsilon
        # 回転/スケール(a,b,c,d)は小さく、移動(tx,ty)は少し大きく変動させて傾きを見る

    def compute_numerical_gradient(self, params, ref, mov, current_score, ref_grad):
        """
        パラメータを少しずらしてスコアの変化量（勾配）を計算する
        """
        grads = np.zeros_like(params)
        
        for i in range(len(params)):
            # パラメータを +epsilon ずらす
            perturbed_params = params.copy()
            perturbed_params[i] += self.epsilon[i]
            
            # ずらした先でのスコアを計算
            new_score = calculate_hybrid_score(perturbed_params, ref, mov, ref_grad_cache=ref_grad)
            
            # 勾配（傾き）= (変化後のスコア - 現在のスコア) / epsilon
            grads[i] = (new_score - current_score) / self.epsilon[i]
            
        return grads

    def run(self, ref_img, mov_img, init_params):
        # 現在のパラメータ
        params = np.array(init_params, dtype=np.float32)
        
        # 高速化のためReferenceの勾配は事前計算
        ref_grad = compute_gradient_magnitude(ref_img)
        
        print(f"Start Optimization. Init Params: {params}")
        
        prev_score = -1.0
        
        for i in range(self.max_iter):
            # 1. 現在のスコア計算
            current_score = calculate_hybrid_score(params, ref_img, mov_img, ref_grad)
            
            # 収束判定 (スコアの向上がごく僅かなら終了)
            if abs(current_score - prev_score) < self.tol:
                print(f"Iter {i}: Converged. Score={current_score:.5f}")
                break
            
            # 2. 勾配（進むべき方向）の計算
            gradients = self.compute_numerical_gradient(params, ref_img, mov_img, current_score, ref_grad)
            
            # 3. パラメータ更新 (Gradient Ascent: 勾配方向へ進む)
            # パラメータごとに感度が違うため、学習率を調整しても良いが、ここでは単純化
            # スケール成分と移動成分でオーダーが違うため重み付け推奨
            # [a, b, c, d, tx, ty]
            weights = np.array([0.05, 0.05, 0.05, 0.05, 10.0, 10.0]) # 回転等は慎重に、移動は大胆に
            
            update_step = self.lr * gradients * weights
            params += update_step
            
            print(f"Iter {i}: Score={current_score:.5f} | Change={np.linalg.norm(update_step):.5f}")
            prev_score = current_score

        return params, current_score

# -----------------------
# 変換関数
# -----------------------
def warp_emg_8x8(emg, tx, ty, theta, sx, sy, shear):
    H, W = emg.shape

    # === 座標定義 ===
    x = np.arange(H)
    y = np.arange(W)
    interp = RectBivariateSpline(y, x, emg, kx=3, ky=3)

    # === 中心座標 ===
    cx = (H - 1) / 2
    cy = (W - 1) / 2

    # === 各変換行列 ===
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    S = np.array([[sx, 0],
                  [0, sy]])
    Sh = np.array([[1, shear],
                   [0, 1]])

    A = R @ Sh @ S

    # === 同次座標変換行列 ===
    T_center = np.array([[1, 0, -cx],
                          [0, 1, -cy],
                          [0, 0, 1]])

    T_center_inv = np.array([[1, 0, cx],
                              [0, 1, cy],
                              [0, 0, 1]])

    T_shift = np.array([[1, 0, tx],
                        [0, 1, ty],
                        [0, 0, 1]])

    M_affine = np.eye(3)
    M_affine[:2, :2] = A

    # === 合成変換（中心回転） ===
    M = T_shift @ T_center_inv @ M_affine @ T_center
    M_inv = np.linalg.inv(M)

    # === グリッド生成 ===
    grid_x, grid_y = np.meshgrid(x, y)
    ones = np.ones_like(grid_x)

    coords = np.stack([grid_x, grid_y, ones], axis=-1)
    coords = coords.reshape(-1, 3).T

    src = M_inv @ coords
    src_x, src_y = src[0], src[1]

    emg_new = interp.ev(src_y, src_x).reshape(H, W)

    # 6×6 サブセット抽出
    return emg_new

# === バッチ適用関数 ===
def warp_batch_images(batch_data, tx, ty, theta, sx, sy, shear):
    """
    (35, 6000, 8, 8) のような多次元リストに warp_emg_8x8 を適用する関数
    """
    # 1. 元の形状を保存
    batch_data = np.array(batch_data)
    original_shape = batch_data.shape  # (35, 6000, 8, 8)
    
    # 2. 画像の枚数(N) × 縦(H) × 横(W) に平坦化
    # これにより (210000, 8, 8) になります
    flat_data = batch_data.reshape(-1, original_shape[-2], original_shape[-1])
    
    # 3. 結果を格納するリスト
    processed_list = []
    
    # 4. ループ処理 (tqdmで進捗を表示すると安心です)
    # print(f"Processing {len(flat_data)} images...")
    for img in flat_data:
        # 個別の画像に関数を適用
        warped_img = warp_emg_8x8(img, tx, ty, theta, sx, sy, shear)
        processed_list.append(warped_img)
    
    # 5. numpy配列に戻し、元の形状 (35, 6000, 8, 8) に戻す
    result_data = np.array(processed_list).reshape(original_shape)
    
    return result_data

# 目的関数
def ncc(a, b):
    if np.std(a) == 0 or np.std(b) == 0: return 0.0
    a_mean = a - np.mean(a)
    b_mean = b - np.mean(b)
    num = np.sum(a_mean * b_mean)
    den = np.sqrt(np.sum(a_mean**2)) * np.sqrt(np.sum(b_mean**2)) + 1e-8
    return float(num / den)

def affine_transform(img, params):
    a, b, c, d, tx, ty = params
    M = np.array([[a, b, tx],
                  [c, d, ty]], dtype=np.float32)
    h, w = img.shape
    return cv2.warpAffine(img, M, (w, h))

def objective_ncc(params, ref, mov):
    warped = affine_transform(mov, params)
    return 1 - ncc(ref, warped)



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# --- 1. モデル定義 (Encoder) ---
class EMGEncoder(nn.Module):
    """
    HD-sEMG画像から特徴ベクトルを抽出するネットワーク
    """
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=64):
        super(EMGEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # どんなサイズでも 1x1 にする
        )
        
    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

# --- 2. 距離計算などのユーティリティ ---
def compute_prototypes(embeddings, labels, n_way):
    """
    サポートセット(適応データ)からプロトタイプを計算する関数
    """
    prototypes = []
    for c in range(n_way):
        # クラスcに属するデータの平均をとる
        mask = (labels == c)
        if mask.sum() > 0:
            p = embeddings[mask].mean(0)
            prototypes.append(p)
        else:
            # 万が一データがない場合はゼロベクトルなどを入れる（エラー回避）
            prototypes.append(torch.zeros_like(embeddings[0]))
    return torch.stack(prototypes)

def euclidean_metric(query, prototypes):
    """
    クエリとプロトタイプ間のユークリッド距離を計算
    """
    n = query.size(0)
    m = prototypes.size(0)
    d = query.size(1)
    
    query = query.unsqueeze(1).expand(n, m, d)
    prototypes = prototypes.unsqueeze(0).expand(n, m, d)
    
    # 距離の二乗（Squared Euclidean Distance）
    dists = torch.pow(query - prototypes, 2).sum(2)
    return dists

def preprocess_ndarray(x, y):
    # 1. Tensorに変換
    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)
    
    # 2. チャネル次元の追加 (Batch, Height, Width) -> (Batch, Channel, Height, Width)
    # CNNは4次元入力を期待するため、2番目に次元を追加します
    x_tensor = x_tensor.unsqueeze(1) 
    
    return x_tensor, y_tensor

import time

# === main ===
def main(subject='nojima'):

    print(f'subject name:{subject}')
    # 学習データ
    emg_list_train = []  # 各要素: shape=(T,8,8)
    y_list_train   = []  # ファイル名から抽出したラベル（長さ = 試行数）
    for j in range(7):
        for k in range(5):
            try:
                file_name = file_name_output(subject=subject, hand='right', electrode_place="original", gesture=j+1, trial=k+1)
                path = '../../data/highMVC/' + file_name
                encoding = 'utf-8-sig'  # または 'utf-16'
                df = pd.read_csv(path, encoding=encoding, sep=';', header=None) 

                # ==== EMGデータの抽出 ====
                time_emg = df.iloc[:, 0].values  # 時刻 [s]
                emg_data = df.iloc[:, 1:65].values  # shape: (time, 64)

                # ==== 基本パラメータ ====
                fs = int(1 / np.mean(np.diff(time_emg)))  # サンプリング周波数
                emg_data = remove_power_line_harmonics(emg_data, fs=fs, fundamental=60.0, Q=30.0)
                filtered_emg = butter_bandpass_filter(emg_data, fs=fs, low_hz=20.0, high_hz=450.0, order=4)
                emg_data = filtered_emg.reshape(-1,8,8)  # shape: (time, 64)
                
                emg_list_train.append(emg_data)
                y_list_train.append(j+1)
            except FileNotFoundError:
                pass
    
    # ==============学習フェーズ=============#
    window = 200   # サンプル幅（例：100ms @ 2kHz）
    hop    = 50    # ホップ（例：25ms）
    window = int(window * (fs/1000))
    hop = int(hop * (fs/1000))
    # X_train = []
    # y_train = []
    sizes_te = []
    threshold = 0
    threshold2 = 0.0013
    ch_size = 8
    # 8×8channel
    for i, (emg_train_8x8, label) in enumerate(zip(emg_list_train, y_list_train)):
        # --- ラベル（学習用）：あなたのラベリングに置き換えてください ---
        # 中央6×6から特徴抽出した個数に合わせる必要があります。
        tmp_X = emg_train_8x8  # (n_samples, 8, 8)
        # tmp_X = extract_center_6x6(tmp_X)  # (n_samples, 6, 6)
        tmp_X = segment_time_series(tmp_X, window=window, hop=hop)  # (n_windows, window, 8, 8)

        # ========= チャネルごとに正規化 =========
        mean = np.mean(tmp_X.reshape(-1,8,8), axis=0)
        std = np.std(tmp_X.reshape(-1,8,8), axis=0) + 1e-8
        tmp_X = (tmp_X - mean.reshape(1, 1, 8, 8)) / std.reshape(1, 1, 8, 8)
        # ====================================

        # tmp_X = tmp_X.reshape(tmp_X.shape[0], tmp_X.shape[1], 36)  # (n_windows, window, 36) #

        # 特徴量抽出
        ptp = [ptp_feat(x) for x in tmp_X]
        rms = [rms_feat(x) for x in tmp_X]
        wl = [waveform_length(x) for x in tmp_X]
        zc = [zero_crossings(x, threshold) for x in tmp_X]
        ssc = [slope_sign_changes(x, threshold) for x in tmp_X]
        wamp = [wamp_feat(x, threshold2) for x in tmp_X]
        td_psd = [td_psd_multichannel(x, fs=fs, mode="vector") for x in tmp_X]
        td_psd = np.array(td_psd)
        f1 = td_psd[:,:,0].reshape(-1,8,8)
        f2 = td_psd[:,:,1].reshape(-1,8,8)
        f3 = td_psd[:,:,2].reshape(-1,8,8)
        f4 = td_psd[:,:,3].reshape(-1,8,8)
        f5 = td_psd[:,:,4].reshape(-1,8,8)
        f6 = td_psd[:,:,5].reshape(-1,8,8)
        # td_psd = td_psd.reshape(td_psd.shape[0], -1)
        # tmp_X = medianfilter_and_hstack([wl, f1, f6], kernel_size=2, shape=6)
        # tmp_X = np.hstack([rms, wl, zc]) # tmp_X = np.hstack([rms, wl, zc, ssc]) #
        # tmp_X = rms
        # tmp_X = np.stack([rms, wl, zc], axis=1)
        # tmp_X = np.stack([wl, rms, zc], axis=1)
        tmp_X = wl
        # tmp_X = extract_features(emg_train_8x8, FeatureSpec(kind=kind, window=window, hop=hop))  # (n_windows, 36)
        n_windows = len(tmp_X)
        # 例：ダミーの 3 クラスを周回（実際はジェスチャーIDに差し替え）
        tmp_y = [int(label)-1 for i in range(n_windows)]
        sizes_te.append(n_windows)

        if len(tmp_y) != len(tmp_X):
            raise ValueError(f"y_train length ({len(tmp_y)}) must match number of windows ({len(tmp_X)})")
        
        if i == 0:
            X_train = tmp_X
            y_train = tmp_y
        else:
            X_train = np.vstack([X_train, tmp_X])
            y_train = np.hstack([y_train, tmp_y])

    X_train_aligment = X_train
    # 学習時のアライメントマップ平均を1秒毎に計算し、３秒間で平均(補間なし)
    gestures = np.unique(y_train) + 1
    train_trial_alignment_maps_without_interp_1sec = []
    for gesture in gestures:
        z_list = []
        trial_list = []
        iterator = X_train_aligment[y_train==gesture-1]
        i = 0
        for tmp_X_train_aligment in iterator[:]:
            z_list.append(tmp_X_train_aligment)
            i += 1
            if i == 19:  # トライアル数で分割
                trial_list.append(np.mean(np.array(z_list), axis=0))
                z_list = []
                i = 0
        train_trial_alignment_maps_without_interp_1sec.append(trial_list)
    

    # 学習時のアライメントマップ平均を時間窓毎に計算 (補間なし)
    train_window_alignment_maps = []
    for gesture in gestures:
        z_list = []
        iterator = X_train_aligment[y_train==gesture-1]
        for tmp_X_train_aligment in iterator[:]:
            z_list.append(tmp_X_train_aligment)
        train_window_alignment_maps.append(z_list)

    # --- 設定パラメータ ---
    N_WAY = 7       # クラス数 (ジェスチャーの種類)
    IMG_SIZE = 8   # HD-sEMGのグリッドサイズ (16x16)
    BATCH_SIZE = 32

    # 「1秒間」の定義
    # 例: サンプリング周波数とウィンドウ幅による。
    # 仮に 1つのウィンドウが時系列データの一部で、1秒間に20個のウィンドウ(フレーム)が得られるとする
    FRAMES_PER_SEC = 19 

    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tr = np.array(train_window_alignment_maps).reshape(-1,8,8)
    y_tr = np.array([[i for j in range(285)] for i in range(7)]).reshape(-1)

    source_train_x, source_train_y = preprocess_ndarray(X_tr.astype(np.float32), y_tr.astype(np.int64))
    source_train_x = source_train_x.to(device)
    source_train_y = source_train_y.to(device)

    # ==========================================
    # Step 1: Sourceデータでの事前学習 (Pre-training)
    # ==========================================
    encoder = EMGEncoder().to(device)
    optimizer = optim.Adam(encoder.parameters(), lr=0.001)

    # ここでは簡易的に、Sourceデータ上でプロトタイプ学習を行うループ
    # (本来はもっと多くのエピソードを回します)
    encoder.train()
    for epoch in range(100): # 簡易的なので10エピソードのみ
        # ランダムサンプリングしてエピソード作成
        perm = torch.randperm(len(source_train_x))
        sx = source_train_x[perm[:100]] # Support (一部)
        sy = source_train_y[perm[:100]]
        qx = source_train_x[perm[100:200]] # Query (一部)
        qy = source_train_y[perm[100:200]]
        
        optimizer.zero_grad()
        
        # 特徴抽出
        z_support = encoder(sx)
        z_query = encoder(qx)
        
        # プロトタイプ計算
        protos = compute_prototypes(z_support, sy, N_WAY)
        
        # 距離計算 & 分類
        dists = euclidean_metric(z_query, protos)
        log_p_y = F.log_softmax(-dists, dim=1)
        
        loss = F.nll_loss(log_p_y, qy)
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
            
    print("Done pre-training.")


    # ==============位置合わせ + 適応 + 推論=============#
    accracy_all = []
    time_alignment_list = []
    time_adoptation_list = []
    for electrode_place in ["original2"]:#["downleft5mm", "downleft10mm", "clockwise"]:
        print(f'electrode_place: {electrode_place}')
        # ずれデータ
        emg_list_test = []
        y_list_test   = []
        for j in range(7):
            for k in range(5):
                try:
                    file_name = file_name_output(subject=subject, hand='right', electrode_place=electrode_place, gesture=j+1, trial=k+1)
                    path = '../../data/highMVC/' + file_name
                    encoding = 'utf-8-sig'  # または 'utf-16'
                    df = pd.read_csv(path, encoding=encoding, sep=';', header=None) 

                    # ==== EMGデータの抽出 ====
                    time_emg = df.iloc[:, 0].values  # 時刻 [s]
                    emg_data = df.iloc[:, 1:65].values  # shape: (time, 64)

                    # ==== 基本パラメータ ====
                    fs = int(1 / np.mean(np.diff(time_emg)))  # サンプリング周波数
                    emg_data = remove_power_line_harmonics(emg_data, fs=fs, fundamental=60.0, Q=30.0)
                    filtered_emg = butter_bandpass_filter(emg_data, fs=fs, low_hz=20.0, high_hz=450.0, order=4)
                    emg_data = filtered_emg.reshape(-1,8,8)  # shape: (time, 64)
                    
                    emg_list_test.append(emg_data)
                    y_list_test.append(j+1)
                except FileNotFoundError:
                    pass
        # 位置合わせ用特徴量マップ抽出
        for i, (emg_test_8x8, label) in enumerate(zip(emg_list_test, y_list_test)):
            # --- ラベル（学習用）：あなたのラベリングに置き換えてください ---
            # 中央6×6から特徴抽出した個数に合わせる必要があります。
            tmp_X = emg_test_8x8  # (n_samples, 8, 8)
            # tmp_X = extract_center_6x6(tmp_X)  # (n_samples, 6, 6)
            tmp_X = segment_time_series(tmp_X, window=window, hop=hop)  # (n_windows, window, 8, 8)

            # ========= チャネルごとに正規化 =========
            mean = np.mean(tmp_X.reshape(-1,ch_size,ch_size), axis=0)
            std = np.std(tmp_X.reshape(-1,ch_size,ch_size), axis=0) + 1e-8
            tmp_X = (tmp_X - mean.reshape(1, 1, ch_size, ch_size)) / std.reshape(1, 1, ch_size, ch_size)
            # ====================================

            # tmp_X = tmp_X.reshape(tmp_X.shape[0], tmp_X.shape[1], 36)  # (n_windows, window, 36) #

            # 特徴量抽出
            ptp = [ptp_feat(x) for x in tmp_X]
            rms = [rms_feat(x) for x in tmp_X]
            wl = [waveform_length(x) for x in tmp_X]
            zc = [zero_crossings(x, threshold) for x in tmp_X]
            ssc = [slope_sign_changes(x, threshold) for x in tmp_X]
            wamp = [wamp_feat(x, threshold2) for x in tmp_X]
            td_psd = [td_psd_multichannel(x, fs=fs, mode="vector") for x in tmp_X]
            td_psd = np.array(td_psd)
            f1 = td_psd[:,:,0].reshape(-1,ch_size,ch_size)
            f2 = td_psd[:,:,1].reshape(-1,ch_size,ch_size)
            f3 = td_psd[:,:,2].reshape(-1,ch_size,ch_size)
            f4 = td_psd[:,:,3].reshape(-1,ch_size,ch_size)
            f5 = td_psd[:,:,4].reshape(-1,ch_size,ch_size)
            f6 = td_psd[:,:,5].reshape(-1,ch_size,ch_size)
            # td_psd = td_psd.reshape(td_psd.shape[0], -1)
            # tmp_X = medianfilter_and_hstack([wl, f1, f6], kernel_size=2, shape=6)
            # tmp_X = np.hstack([rms, wl, zc]) # tmp_X = np.hstack([rms, wl, zc, ssc]) #
            # tmp_X = rms
            # tmp_X = np.stack([rms, wl, zc], axis=1)
            tmp_X = np.stack([wl, rms, zc], axis=1)
            # tmp_X = rms
            # tmp_X = extract_features(emg_test_8x8, FeatureSpec(kind=kind, window=window, hop=hop))  # (n_windows, 36)
            n_windows = len(tmp_X)
            # 例：ダミーの 3 クラスを周回（実際はジェスチャーIDに差し替え）
            tmp_y = [int(label)-1 for i in range(n_windows)]
            sizes_te.append(n_windows)

            if len(tmp_y) != len(tmp_X):
                raise ValueError(f"y_test length ({len(tmp_y)}) must match number of windows ({len(tmp_X)})")
            
            if i == 0:
                X_test = tmp_X
                y_test = tmp_y
            else:
                X_test = np.vstack([X_test, tmp_X])
                y_test = np.hstack([y_test, tmp_y])

        X_test_aligment = X_test

        # 推論時のアライメントマップ平均を1秒毎に計算し、３秒間で平均(補間なし)
        gestures = np.unique(y_test) + 1
        test_trial_alignment_maps_without_interp_1sec = []
        for gesture in gestures:
            z_list = []
            trial_list = []
            iterator = X_test_aligment[y_test==gesture-1, 0]
            i = 0
            for tmp_X_test_aligment in iterator[:]:
                z_list.append(tmp_X_test_aligment)
                i += 1
                if i == 19:  # トライアル数で分割
                    trial_list.append(np.mean(np.array(z_list), axis=0))
                    z_list = []
                    i = 0
            test_trial_alignment_maps_without_interp_1sec.append(trial_list)
 
        #-----------------------------------------
        # 位置合わせ
        #-----------------------------------------
        accuracy_each_position = []
        accuracy_each_position_normal = []
        for gesture in range(1,8):
            accuracy_each_gesture_alignment = []
            for trial in range(1,6):
                start_alignment = time.time() # 時間計測開始
                scalar = MinMaxScaler()
                img_ref = (scalar.fit_transform(train_trial_alignment_maps_without_interp_1sec[gesture-1][trial*3-1].reshape(-1,1)).reshape(train_trial_alignment_maps_without_interp_1sec[gesture-1][trial*3-1].shape)*255).astype(np.float32)
                img_test = (scalar.fit_transform(test_trial_alignment_maps_without_interp_1sec[gesture-1][trial*3-1].reshape(-1,1)).reshape(test_trial_alignment_maps_without_interp_1sec[gesture-1][trial*3-1].shape)*255).astype(np.float32)

                if 1 ==1:
                # try:
                    # 初期値：単位行列（≒剛体）
                    init = [1, 0, 0, 1, 0, 0]

                    res = minimize(
                        objective_ncc,
                        x0=init,
                        args=(img_test, img_ref),
                        method="Powell"
                    )

                    a, b, c, d, tx, ty = res.x

                    theta_rad = np.arctan2(c, a)
                    theta_deg = np.degrees(theta_rad)
                    sx = np.sqrt(a**2 + c**2)
                    sy = np.sqrt(b**2 + d**2)
                    shear = (a*b + c*d) / (sx*sy)

                    end_alignment = time.time()
                    time_alignment_list.append(end_alignment - start_alignment)
                        

                    # ==============推論フェーズ=============#
                    # 推論用特徴量マップ抽出
                    tx = -tx  # mm→チャネル単位
                    ty = -ty  # mm→チャネル単位
                    theta = theta_rad #theta_rad  # degree
                    sx = sx
                    sy = sy
                    shear = shear
                    # ==============推論フェーズ=============#
                    for i, (emg_test_8x8, label) in enumerate(zip(emg_list_test, y_list_test)):
                        # --- ラベル（学習用）：あなたのラベリングに置き換えてください ---
                        # 中央6×6から特徴抽出した個数に合わせる必要があります。
                        tmp_X = warp_batch_images(emg_test_8x8, tx, ty, theta, sx, sy, shear)
                        # tmp_X = extract_center_6x6(tmp_X)  # (n_samples, 6, 6)
                        tmp_X = segment_time_series(tmp_X, window=window, hop=hop)  # (n_windows, window, 8, 8)

                        # ========= チャネルごとに正規化 =========
                        mean = np.mean(tmp_X.reshape(-1,8,8), axis=0)
                        std = np.std(tmp_X.reshape(-1,8,8), axis=0) + 1e-8
                        tmp_X = (tmp_X - mean.reshape(1, 1, 8, 8)) / std.reshape(1, 1, 8, 8)
                        # ====================================

                        # tmp_X = tmp_X.reshape(tmp_X.shape[0], tmp_X.shape[1], 36)  # (n_windows, window, 36) #

                        # 特徴量抽出
                        ptp = [ptp_feat(x) for x in tmp_X]
                        rms = [rms_feat(x) for x in tmp_X]
                        wl = [waveform_length(x) for x in tmp_X]
                        zc = [zero_crossings(x, threshold) for x in tmp_X]
                        ssc = [slope_sign_changes(x, threshold) for x in tmp_X]
                        wamp = [wamp_feat(x, threshold2) for x in tmp_X]
                        td_psd = [td_psd_multichannel(x, fs=fs, mode="vector") for x in tmp_X]
                        td_psd = np.array(td_psd)
                        f1 = td_psd[:,:,0].reshape(-1,8,8)
                        f2 = td_psd[:,:,1].reshape(-1,8,8)
                        f3 = td_psd[:,:,2].reshape(-1,8,8)
                        f4 = td_psd[:,:,3].reshape(-1,8,8)
                        f5 = td_psd[:,:,4].reshape(-1,8,8)
                        f6 = td_psd[:,:,5].reshape(-1,8,8)
                        # td_psd = td_psd.reshape(td_psd.shape[0], -1)
                        # tmp_X = medianfilter_and_hstack([wl, f1, f6], kernel_size=2, shape=6)
                        # tmp_X = np.hstack([rms, wl, zc]) # tmp_X = np.hstack([rms, wl, zc, ssc]) #
                        # tmp_X = rms
                        # tmp_X = np.stack([rms, wl, zc], axis=1)
                        # tmp_X = np.stack([wl, rms, zc], axis=1)
                        tmp_X = wl
                        # tmp_X = extract_features(emg_test_8x8, FeatureSpec(kind=kind, window=window, hop=hop))  # (n_windows, 36)
                        n_windows = len(tmp_X)
                        # 例：ダミーの 3 クラスを周回（実際はジェスチャーIDに差し替え）
                        tmp_y = [int(label)-1 for i in range(n_windows)]
                        sizes_te.append(n_windows)

                        if len(tmp_y) != len(tmp_X):
                            raise ValueError(f"y_test length ({len(tmp_y)}) must match number of windows ({len(tmp_X)})")
                        
                        if i == 0:
                            X_test = tmp_X
                            y_test = tmp_y
                        else:
                            X_test = np.vstack([X_test, tmp_X])
                            y_test = np.hstack([y_test, tmp_y])


                    # 学習時のアライメントマップ平均を時間窓毎に計算 (補間なし)
                    X_test_aligment = X_test
                    gestures = np.unique(y_test) + 1

                    test_window_alignment_maps = []
                    for ges in gestures:
                        z_list = []
                        iterator = X_test_aligment[y_test==ges-1]
                        for tmp_X_test_aligment in iterator[:]:
                            z_list.append(tmp_X_test_aligment)
                        test_window_alignment_maps.append(z_list)
                    
                    
                    # 適応データ + テストデータ作成
                    X_adap = np.array(test_window_alignment_maps).reshape(7,285,64)[:, (trial-1)*57:(trial-1)*57+19].reshape(-1,8,8)
                    y_adap = np.array([[i for j in range(19)] for i in range(7)]).reshape(-1)
                    X_te = np.array(test_window_alignment_maps).reshape(7,285,64)[:, np.r_[0:(trial-1)*57, trial*57:285]].reshape(-1,8,8)
                    y_te = np.array([[i for j in range(228)] for i in range(7)]).reshape(-1)

                    target_adapt_x, target_adapt_y = preprocess_ndarray(X_adap.astype(np.float32), y_adap.astype(np.int64))
                    target_test_x, target_test_y = preprocess_ndarray(X_te.astype(np.float32), y_te.astype(np.int64))

                    target_adapt_x = target_adapt_x.to(device)
                    target_adapt_y = target_adapt_y.to(device)
                    target_test_x = target_test_x.to(device)
                    target_test_y = target_test_y.to(device)
                        
                    # ==========================================
                    # Step 2: 再装着時の適応 (Prototypical Adaptation)
                    # ==========================================
                    start_adoptation = time.time()
                    encoder.eval() # エンコーダは固定 (評価モード)

                    with torch.no_grad():
                        # 1. 適応データ(1秒分)をエンコード
                        z_adapt = encoder(target_adapt_x)
                        
                        # 2. 新しいセッション専用のプロトタイプを計算
                        # これが「キャリブレーション」の正体です
                        new_prototypes = compute_prototypes(z_adapt, target_adapt_y, N_WAY)

                    end_adoptation = time.time()

                    time_adoptation_list.append(end_adoptation - start_adoptation)

                    # ==========================================
                    # Step 3: 再装着後の推論 (Inference)
                    # ==========================================
                    with torch.no_grad():
                        # テストデータをエンコード
                        z_test = encoder(target_test_x)
                        
                        # 「新しいプロトタイプ」を使って距離計算
                        dists = euclidean_metric(z_test, new_prototypes)
                        
                        # 予測 (距離が一番近いクラス)
                        predictions = torch.argmin(dists, dim=1)
                        
                        # 精度計算
                        acc = (predictions == target_test_y).float().mean()
                        score = acc.to('cpu').detach().numpy().copy()

                    accuracy_each_gesture_alignment.append(score)
                    accuracy_each_position.append(score)
                    accracy_all.append(score)
            print(f'accuracy of gesture{gesture}: {np.mean(accuracy_each_gesture_alignment)}')
        print(f'accuracy of {electrode_place}: {np.mean(accuracy_each_position)}')
    print(f'accracy_all: {np.mean(accracy_all)}')
    print(f'average time for alignment: {np.mean(time_alignment_list)} sec')
    print(f'average time for adoptation: {np.mean(time_adoptation_list)} sec')