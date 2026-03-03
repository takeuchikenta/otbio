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
from scipy.ndimage import gaussian_filter, median_filter
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
import torch.optim as optim
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

def arv_feat(x):
    return np.mean(np.abs(x), axis=0)

def zc_feat(signal):
    s = np.sign(signal)
    s_prev = s[:-1]
    s_next = s[1:]
    sign_change = (s_prev * s_next) < 0
    return np.sum(sign_change, axis=0)

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



#===========================================
# Prototypical Adaptation
#===========================================

# ==========================================
# 1. 前処理と特徴量抽出 
# ==========================================

def extract_feature_map(window_data, feature_name='wl'):
    """
    Waveform Length (WL) を抽出し、8x8のマップとして返す
    Input: (Time, Channels_Flat)
    Output: (8, 8) Feature Map
    """
    # 1. 各チャンネルのWLを計算 (Flat)
    if feature_name == 'wl':
        feature_flat = np.sum(np.abs(np.diff(window_data, axis=0)), axis=0)
    elif feature_name == 'ptp':
        feature_flat = ptp_feat(window_data)
    elif feature_name == 'rms':
        feature_flat = rms_feat(window_data)
    elif feature_name == 'zc':
        feature_flat = zc_feat(window_data)
    else:
        raise ValueError(f"Unknown feature name: {feature_name}")
    
    # 2. 空間配置 (8x8) にリシェイプ
    # データセットの並び順に依存しますが、ここでは一般的な8x8配列と仮定
    feature_map = feature_flat.reshape(8, 8)
    
    return feature_map

def apply_median_filter(feature_map, kernel_size=3):
    """
    論文 Eq.(2) に基づくメディアンフィルタ
    Input: (8, 8)
    Output: (8, 8)
    """
    # 3x3 カーネルでフィルタリング [cite: 976]
    filtered_map = median_filter(feature_map, size=kernel_size, mode='reflect')
    return filtered_map

def prepare_data_indices(n_trials, n_time, fs, is_support_set=False, shot_trial_idx=0):
    """
    修正: shot_trial_idx (0~4) を引数に追加
    
    shot_trial_idx: 0始まりのインデックス。0なら1施行目、4なら5施行目を指定。
    """
    # 全データ形状: (35トライアル, 6000サンプル, 8, 8)
    # 35トライアル = 7ジェスチャー * 5施行
    trials_per_gesture = 5
    n_gestures = 7
    
    indices = []
    
    # バリデーション: インデックスが範囲内か確認
    if not (0 <= shot_trial_idx < trials_per_gesture):
        raise ValueError(f"shot_trial_idx must be between 0 and {trials_per_gesture - 1}")

    for g in range(n_gestures):
        start_trial_idx = g * trials_per_gesture
        
        # ターゲットとなる試行の全体インデックス
        target_idx = start_trial_idx + shot_trial_idx
        
        if is_support_set:
            # 修正: 指定された shot_trial_idx の試行のみを選択
            # さらに「最初の1秒間」のみ使用
            trial_indices = [target_idx] 
            time_limit = 2000 
        else:
            # 修正: 指定された shot_trial_idx 「以外」の残りの施行を選択
            # 全試行インデックスのリストを作成
            all_indices = list(range(start_trial_idx, start_trial_idx + trials_per_gesture))
            # ターゲットを除外
            all_indices.remove(target_idx)
            
            trial_indices = all_indices
            time_limit = n_time # 最後まで使う
            
        indices.append((g, trial_indices, time_limit))
            
    return indices

def create_dataset(emg_list, y_list, fs=2000, window_ms=200, step_ms=50, mode='full', shot_trial_idx=0, feature_name='wl'):
    """
    データセット作成関数 (修正: 特徴抽出後にメディアンフィルタ適用)
    """
    data = np.array(emg_list) # (35, 6000, 8, 8)
    n_trials, n_time, _, _ = data.shape
    # チャンネルをフラット化して処理
    data = data.reshape(n_trials, n_time, -1)
    
    window_samples = int((window_ms / 1000) * fs)
    step_samples = int((step_ms / 1000) * fs)
    
    X_feat = []
    y_labels = []
    
    if mode == 'full':
        target_config = [(g, range(g*5, (g+1)*5), n_time) for g in range(7)]
    elif mode == 'support':
        target_config = prepare_data_indices(5, n_time, fs, is_support_set=True, shot_trial_idx=shot_trial_idx)
    elif mode == 'query':
        target_config = prepare_data_indices(5, n_time, fs, is_support_set=False, shot_trial_idx=shot_trial_idx)
    
    for gesture_label, trial_indices, time_limit in target_config:
        for t_idx in trial_indices:
            trial_data = data[t_idx]
            limit = min(time_limit, trial_data.shape[0])
            valid_data = trial_data[:limit, :]
            
            for start in range(0, limit - window_samples + 1, step_samples):
                end = start + window_samples
                window = valid_data[start:end, :]
                
                # --- 変更箇所: 特徴抽出 -> フィルタ -> フラット化 ---
                
                # 1. WL特徴量マップ抽出 (8x8)
                feat_map = extract_feature_map(window, feature_name=feature_name)
                
                # 2. メディアンフィルタ適用 [cite: 968]
                # feat_map_filtered = apply_median_filter(feat_map, kernel_size=3)
                
                # 3. フラット化してリストに追加
                feat_flat = feat_map.flatten()
                
                X_feat.append(feat_flat)
                y_labels.append(gesture_label)
    # 正規化のため、以下を追加
    X_feat = np.array(X_feat)
    y_labels = np.array(y_labels)

    # 各列(チャネル)ごとに正規化を行います。
    if len(X_feat) > 0:
        scaler = StandardScaler()
        X_feat = scaler.fit_transform(X_feat)
                
    return X_feat, y_labels
    # return np.array(X_feat), np.array(y_labels)
                

# ==========================================
# 2. モデル定義 
# ==========================================

class ProtoNet(nn.Module):
    def __init__(self, input_dim):
        super(ProtoNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
    def forward(self, x):
        return self.net(x)

def euclidean_dist(x, prototypes):
    n = x.size(0)
    m = prototypes.size(0)
    d = x.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    prototypes = prototypes.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - prototypes, 2).sum(2)

import time

# === main ===
def main(subject='nojima', feature_name='wl'):

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
    
    # --- A. データセット作成 ---
    print("Processing Training Data (with Median Filter)...")
    X_train, y_train = create_dataset(emg_list_train, y_list_train, fs=2000, window_ms=200, step_ms=50, mode='full', feature_name=feature_name)
    # Tensor変換
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    # --- C. モデル学習 (Source Domain) ---
    model = ProtoNet(input_dim=X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    n_epochs = 300
    n_classes = 7

    print("\nTraining Embedding Network...")
    model.train()
    for epoch in range(n_epochs):
        embeddings = model(X_train_t)
        prototypes = []
        for c in range(n_classes):
            mask = (y_train_t == c)
            if mask.sum() > 0:
                prototypes.append(embeddings[mask].mean(0))
            else:
                prototypes.append(torch.zeros(128))
        prototypes = torch.stack(prototypes)
        
        dists = euclidean_dist(embeddings, prototypes)
        log_p_y = torch.log_softmax(-dists, dim=1)
        loss = -log_p_y.gather(1, y_train_t.view(-1, 1)).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")

    # ==============適応 + 推論=============#
    accracy_all = []
    time_alignment_list = []
    time_adaptation_list = []
    for electrode_place in ["original2","downleft5mm", "downleft10mm", "clockwise"]: 
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
 
        accuracy_each_position = []
        for target_shot_index in range(5):
            # print("Processing Test Data (One-Shot Split)...")
            X_test_support, y_test_support = create_dataset(emg_list_test, y_list_test, mode='support', shot_trial_idx=target_shot_index, feature_name=feature_name)
            X_test_query, y_test_query = create_dataset(emg_list_test, y_list_test, mode='query', shot_trial_idx=target_shot_index, feature_name=feature_name)

            X_support_t = torch.FloatTensor(X_test_support)
            y_support_t = torch.LongTensor(y_test_support)
            X_query_t = torch.FloatTensor(X_test_query)
            y_query_t = torch.LongTensor(y_test_query)

            # --- D. One-Shot 推論 (Target Domain) ---
            # print("\nPerforming One-Shot Adaptation...")
            model.eval()
            with torch.no_grad():
                start_adap = time.time()
                # サポートセットでプロトタイプ計算
                support_embeddings = model(X_support_t)
                new_prototypes = []
                for c in range(n_classes):
                    mask = (y_support_t == c)
                    if mask.sum() > 0:
                        new_prototypes.append(support_embeddings[mask].mean(0))
                    else:
                        new_prototypes.append(torch.zeros(128))
                new_prototypes = torch.stack(new_prototypes)
                end_adap = time.time()
                time_adaptation_list.append(end_adap - start_adap)
                
                # クエリセットで評価
                query_embeddings = model(X_query_t)
                dists = euclidean_dist(query_embeddings, new_prototypes)
                y_pred = torch.argmin(dists, dim=1)
                score = (y_pred == y_query_t).float().mean().item()

            accuracy_each_position.append(score)
            accracy_all.append(score)
        print(f'accuracy of {electrode_place}: {np.mean(accuracy_each_position)}')
    print(f'accracy_all: {np.mean(accracy_all)}')
    print(f'average time for adoptation: {np.mean(time_adaptation_list)} sec')