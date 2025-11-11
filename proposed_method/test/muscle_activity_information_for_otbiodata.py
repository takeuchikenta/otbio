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
from scipy.interpolate import RectBivariateSpline, griddata
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

# Butterworth（0位相filtfilt）推奨
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

# ==== バンドパスフィルタ ====
def bandpass(signal, fs=2000, low=5, high=500, order=3):
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype='band')
    return filtfilt(b, a, signal, axis=1)

# ----- 2次元データ中のNaN補間関数 -----
def fill_nan_griddata(data_2d):
    """8×8の2次元データ中のNaNを3次補間(griddata)で補う"""
    x = np.arange(data_2d.shape[1])
    y = np.arange(data_2d.shape[0])
    xx, yy = np.meshgrid(x, y)
    mask = ~np.isnan(data_2d)
    points = np.column_stack((xx[mask], yy[mask]))
    values = data_2d[mask]
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))

    interp_values = griddata(points, values, grid_points, method='cubic')
    data_interp = interp_values.reshape(data_2d.shape)

    filled = data_2d.copy()
    filled[np.isnan(data_2d)] = data_interp[np.isnan(data_2d)]
    return filled

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
            map_2d = fill_nan_griddata(map_2d) # NaN補間

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




# csvファイル保存用の関数
FIELDS = [
    'file_name', 'gesture', 'trial', 'subject', 'session',
    'electrode_place', 'virtual_bipolars',
    'labels', 'center_direction', 'n_virtual_bipolars', 'features'
]

def _to_jsonable(x):
    """CSVに入れる前段階としてJSON化可能な素に変換（NumPy対応）。"""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()
    if isinstance(x, Mapping):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x  # 文字列/数値/None などはそのまま

def _serialize_for_csv(x):
    """リスト/辞書/タプルはJSON文字列に、その他はそのまま返す。"""
    x = _to_jsonable(x)
    if isinstance(x, (list, dict, tuple)):
        return json.dumps(x, ensure_ascii=False, separators=(',', ':'))
    return x

def save_records_to_csv(records, out_path, fields=FIELDS):
    """
    records: 上記フォーマットの辞書のリスト
    out_path: 保存パス（例: 'output/records.csv'）
    fields: 列順（recordsに無いキーは空欄、余分なキーは末尾に追加）
    """
    rows = []
    for rec in records:
        row = {k: _serialize_for_csv(rec.get(k, None)) for k in rec.keys()}
        # 列順を固定したい場合に欠損キーを補完
        for k in fields:
            if k not in row:
                row[k] = None
        rows.append(row)

    # 列順をfields優先に（未知のキーがあれば後ろに付ける）
    extra_cols = [c for c in rows[0].keys() if c not in fields] if rows else []
    columns = list(fields) + extra_cols

    df = pd.DataFrame(rows, columns=columns)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    return out_path





# ----- メイン -----
# if __name__ == "__main__":
def main(feature_func, func_type=False, feature_name='ptp', muscle_activity_informations_measurere='gmm', subject_list=["garu"]):
    print(feature_name +'_' + muscle_activity_informations_measurere)

    muscle_activity_informations = []
    subject_list = subject_list
    hand = "right"
    dir_list = ["1-original", "2-upright", "3-downright", "4-downleft", "5-upleft", "6-clockwise", "7-anticlockwise"]

    for subject in subject_list:
        for dir in dir_list:
            if dir == "1-original":
                filename = "original"
            elif dir == "2-upright":
                filename = "upright"
            elif dir == "3-downright":
                filename = "downright"
            elif dir == "4-downleft":
                filename = "downleft"
            elif dir == "5-upleft":
                filename = "upleft"
            elif dir == "6-clockwise":
                filename = "clockwise"
            elif dir == "7-anticlockwise":
                filename = "anticlockwise"
            for i in range(1, 6):
                for j in range(1, 8):
                    file_name = subject + '/' + hand + '/' + dir + '/set' + str(i) + '/' + filename + '-g' + str(j) + '-' + str(i) + '.csv'
                    path = '../../data/' + file_name
                    encoding = 'utf-8-sig'  # または 'utf-16'
                    df = pd.read_csv(path, encoding=encoding, sep=';', header=None) 
                    print(file_name)

                    # ==== EMGデータの抽出 ====
                    time = df.iloc[:, 0].values  # 時刻 [s]
                    emg_data = df.iloc[:, 1:65].values.T  # shape: (64, time)

                    # ==== 基本パラメータ ====
                    fs = int(1 / np.mean(np.diff(time)))  # サンプリング周波数

                    filtered_emg = emg_data = butter_bandpass_filter(emg_data, fs=2048, low_hz=20.0, high_hz=400.0, order=4) #bandpass(emg_data, fs=fs)
                    emg_data = filtered_emg.T  # shape: (time, 64)
                        
                    try:
                        if muscle_activity_informations_measurere == 'gaussianfitting':
                            features = gaussian_fitting(emg_data, gaussian_2d, feature_func, fs=2000, window_ms=25, threshold=0, func_type=func_type)
                        elif muscle_activity_informations_measurere == 'gmm':
                            features = gmm(emg_data, gaussian_2d, feature_func, fs=2000, window_ms=25, threshold=0, percent=95, max_components=4, criterion='bic', upsample_factor=5, func_type=func_type)
                        muscle_activity_informations.append({'file_name': file_name, 'gesture': j, 'trial': i, 'subject': subject, 'electrode_place':filename, 'features': features})
                    except RuntimeError:
                        muscle_activity_informations.append({'file_name': file_name, 'gesture': j, 'trial': i, 'subject': subject, 'electrode_place':filename, 'features': []})


    file_name_prefix = 'otbio_test1_' + feature_name + '_' + muscle_activity_informations_measurere
    # csvファイルに保存
    save_records_to_csv(muscle_activity_informations, out_path='output/features/' + file_name_prefix + '_features.csv')