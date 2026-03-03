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

from dataclasses import dataclass, field
from typing import Tuple, Literal, Optional, Any, Dict, Callable, List
from scipy.interpolate import RectBivariateSpline


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




import time


# === main ===
def main():

    # ### 修正点1: 全被験者リストと結果保存用リスト ###
    subjects = ['nojima', 'takeuchi2', 'yamamoto', 'stefan']
    test_positions = ["original2", "downleft5mm", "downleft10mm", "clockwise"]
    all_results_data = []

    print(f"Starting LDA Analysis for: {subjects}")

    for subject in subjects:
        print(f"\n{'='*20} Processing Subject: {subject} {'='*20}")
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
            # td_psd = [td_psd_multichannel(x, fs=fs, mode="vector") for x in tmp_X]
            # td_psd = np.array(td_psd)
            # f1 = td_psd[:,:,0].reshape(-1,ch_size,ch_size)
            # f2 = td_psd[:,:,1].reshape(-1,ch_size,ch_size)
            # f3 = td_psd[:,:,2].reshape(-1,ch_size,ch_size)
            # f4 = td_psd[:,:,3].reshape(-1,ch_size,ch_size)
            # f5 = td_psd[:,:,4].reshape(-1,ch_size,ch_size)
            # f6 = td_psd[:,:,5].reshape(-1,ch_size,ch_size)
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
        # 学習
        X_train_nonclopped = X_train
        X_train_nonclopped = X_train_nonclopped.reshape(-1, 8*8)

        clf = LinearDiscriminantAnalysis()
        clf.fit(X_train_nonclopped, y_train)

        # ==============位置合わせ=============#
        accracy_all = []
        time_alignment_list = []
        for electrode_place in ["original2","downleft5mm", "downleft10mm", "clockwise"]:
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
            #-----------------------------------------
            # 推論
            #-----------------------------------------
            accuracy_each_position = []
            for trial in range(1,6):
                ignore_no = []
                no = 0
                for g in range(1,8):
                    for t in range(1,6):
                        no += 1
                        if t == trial:
                            ignore_no.append(no)
                # ==============推論フェーズ=============#
                j = 0
                for i, (emg_test_8x8, label) in enumerate(zip(emg_list_test, y_list_test)):
                    if i+1 in ignore_no:
                        continue
                    tmp_X = emg_test_8x8  # (n_samples, 8, 8)

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
                    # td_psd = [td_psd_multichannel(x, fs=fs, mode="vector") for x in tmp_X]
                    # td_psd = np.array(td_psd)
                    # f1 = td_psd[:,:,0].reshape(-1,ch_size,ch_size)
                    # f2 = td_psd[:,:,1].reshape(-1,ch_size,ch_size)
                    # f3 = td_psd[:,:,2].reshape(-1,ch_size,ch_size)
                    # f4 = td_psd[:,:,3].reshape(-1,ch_size,ch_size)
                    # f5 = td_psd[:,:,4].reshape(-1,ch_size,ch_size)
                    # f6 = td_psd[:,:,5].reshape(-1,ch_size,ch_size)
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
                    
                    if j == 0:
                        X_test = tmp_X
                        y_test = tmp_y
                    else:
                        X_test = np.vstack([X_test, tmp_X])
                        y_test = np.hstack([y_test, tmp_y])

                    j += 1

                X_test_nonclopped = X_test.reshape(-1, 8*8)
                y_pred = clf.predict(X_test_nonclopped)
                proba = clf.predict_proba(X_test_nonclopped)
                score = clf.score(X_test_nonclopped, y_test)
                # print(f"accuracy = {score}")
                accuracy_each_position.append(score)
                accracy_all.append(score)

                # ### 修正点3: 結果保存 ###
                all_results_data.append({
                    "Subject": subject,
                    "Method": "LDA(SourceOnly)",
                    "Position": electrode_place,
                    "Fold": trial-1,
                    "Accuracy": score
                })
            # except:
            #     print(f"電極配置: {electrode_place}, ジェスチャ: {gesture}, trial1: {trial1}, trial2: {trial2} でエラー")
        
            print(f'accuracy of {electrode_place}: {np.mean(accuracy_each_position)}')
        print(f'accracy_all: {np.mean(accracy_all)}')
    # ### 修正点4: CSV出力 ###
    print("\n" + "="*40)
    print("Saving results to 'lda_results.csv'...")
    df_results = pd.DataFrame(all_results_data)
    df_results.to_csv("lda_results.csv", index=False)
    print("Done.")