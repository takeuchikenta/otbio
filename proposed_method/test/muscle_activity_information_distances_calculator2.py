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

import muscle_activity_information
import muscle_activity_information_pca



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


# muscle_activity_information_pca.main(ptp, func_type=False, feature_name='ptp')

# muscle_activity_information.main(ptp, func_type=False, feature_name='ptp', muscle_activity_informations_measurere='gmm', distance_results=False)
# muscle_activity_information.main(rms, func_type=False, feature_name='rms', muscle_activity_informations_measurere='gmm', distance_results=False)
# muscle_activity_information.main(zc, func_type=False, feature_name='zc', muscle_activity_informations_measurere='gmm', distance_results=False)
# muscle_activity_information.main(waveform_length, func_type=False, feature_name='waveformlength', muscle_activity_informations_measurere='gmm', distance_results=False)
# muscle_activity_information.main(mean_frequency, func_type=True, feature_name='meanfrequency', muscle_activity_informations_measurere='gmm', distance_results=False)
# muscle_activity_information.main(median_frequency, func_type=True, feature_name='medianfrequency', muscle_activity_informations_measurere='gmm', distance_results=False)
# muscle_activity_information.main(peak_frequency, func_type=True, feature_name='peakfrequency', muscle_activity_informations_measurere='gmm', distance_results=False)
# muscle_activity_information.main(spectral_entropy, func_type=True, feature_name='spectralentropy', muscle_activity_informations_measurere='gmm', distance_results=False)

muscle_activity_information.main(ptp, func_type=False, feature_name='ptp', muscle_activity_informations_measurere='gaussianfitting', preprocess=False, n_subjects=5, n_sessions=2, distance_results=False, kmeans32=False, kmeans42=False, kmeans43=True, kmeans52=False, kmeans53=False, kmeans54=False, normalized_kmeans2=False, normalized_kmeans3=False, normalized_kmeans4=False, normalized_kmeans5=False, xmeans=False, normalized_xmeans=False, hdbscan=False, normalized_hdbscan=False)
# muscle_activity_information.main(zc, func_type=False, feature_name='zc', muscle_activity_informations_measurere='gaussianfitting', distance_results=False)
# muscle_activity_information.main(rms, func_type=False, feature_name='rms', muscle_activity_informations_measurere='gaussianfitting', distance_results=False)
# muscle_activity_information.main(waveform_length, func_type=False, feature_name='waveformlength', muscle_activity_informations_measurere='gaussianfitting', distance_results=False)
# muscle_activity_information.main(mean_frequency, func_type=True, feature_name='meanfrequency', muscle_activity_informations_measurere='gaussianfitting', distance_results=False)
# muscle_activity_information.main(median_frequency, func_type=True, feature_name='medianfrequency', muscle_activity_informations_measurere='gaussianfitting', distance_results=False)
# muscle_activity_information.main(peak_frequency, func_type=True, feature_name='peakfrequency', muscle_activity_informations_measurere='gaussianfitting', distance_results=False)
# muscle_activity_information.main(spectral_entropy, func_type=True, feature_name='spectralentropy', muscle_activity_informations_measurere='gaussianfitting', distance_results=False)