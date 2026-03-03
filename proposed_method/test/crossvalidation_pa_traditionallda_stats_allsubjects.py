import wfdb
import os
import glob
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, sosfiltfilt, iirnotch
from scipy.stats import ttest_rel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import json

# 不要な警告を無視
warnings.filterwarnings('ignore')
np.warnings = warnings

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. Helper Functions (Signal Processing)
# ==========================================

def _ensure_2d(x: np.ndarray) -> tuple[np.ndarray, bool]:
    x = np.asarray(x)
    was_1d = False
    if x.ndim == 1:
        x = x[:, None]
        was_1d = True
    elif x.ndim != 2:
        raise ValueError("Input must be 1D or 2D array (samples[, channels]).")
    return x, was_1d

def band_edges_safe(low_hz: float, high_hz: float, fs: float, margin: float = 0.01):
    nyq = fs * 0.5
    low = max(0.0, float(low_hz))
    high = min(float(high_hz), nyq*(1.0 - margin))
    return low, high

def butter_bandpass_filter(x, fs: float, low_hz: float, high_hz: float, order: int = 4):
    low, high = band_edges_safe(low_hz, high_hz, fs)
    sos = butter(order, [low, high], btype='band', fs=fs, output='sos')
    x2, was_1d = _ensure_2d(x)
    y = sosfiltfilt(sos, x2, axis=0)
    return y.ravel() if was_1d else y

def remove_power_line_harmonics(data, fs, fundamental=60.0, Q=30.0):
    filtered_data = data.copy()
    nyquist = fs / 2.0
    target_freqs = np.arange(fundamental, nyquist, fundamental)
    for freq in target_freqs:
        b, a = signal.iirnotch(w0=freq, Q=Q, fs=fs)
        filtered_data = signal.filtfilt(b, a, filtered_data, axis=0)
    return filtered_data

def file_name_output(subject, hand="right", electrode_place="original", gesture=1, trial=1):
    filename_map = {
        "original": "1-original", "upright": "2-upright", "downright": "3-downright",
        "downleft": "4-downleft", "upleft": "5-upleft", "clockwise": "6-clockwise",
        "anticlockwise": "7-anticlockwise", "original2": "original2",
        "downleft5mm": "downleft5mm", "downleft10mm": "downleft10mm"
    }
    filename = filename_map.get(electrode_place, electrode_place)
    file_name = f"{subject}/{hand}/{filename}/set{trial}/{electrode_place}-g{gesture}-{trial}.csv"
    return file_name

# ==========================================
# 2. Feature Extraction (From Your Code)
# ==========================================

def segment_time_series(emg_8x8: np.ndarray, window: int, hop: int) -> np.ndarray:
    """(n_samples, 8, 8) -> (n_windows, window, 8, 8)"""
    n, _, _ = emg_8x8.shape
    idx_starts = np.arange(0, max(1, n - window + 1), hop)
    segs = np.stack([emg_8x8[s:s+window] for s in idx_starts if s + window <= n], axis=0)
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


def process_emg_data(emg_list, y_list, fs=2000, window_ms=200, hop_ms=50, feature_name="wl"):
    """
    提示されたコードのロジックに基づいて特徴量抽出を行う関数
    1. 窓分割
    2. チャネル毎のZ-score正規化 (平均0, 分散1)
    3. RMS特徴量抽出
    """
    window_samples = int((window_ms / 1000) * fs)
    hop_samples = int((hop_ms / 1000) * fs)
    ch_size = 8
    
    X_features = []
    y_labels = []
    
    # リスト内の各ファイル（試行）ごとに処理
    for emg_data, label in zip(emg_list, y_list):
        # 1. 窓分割
        tmp_X = segment_time_series(emg_data, window=window_samples, hop=hop_samples) # (n_windows, window, 8, 8)
        
        if tmp_X.shape[0] == 0:
            continue

        # 2. 正規化 (提示コード準拠: ファイル単位での正規化)
        # shape変形: (n_windows*window, 8, 8) で平均分散を計算
        reshaped_for_stats = tmp_X.reshape(-1, ch_size, ch_size)
        mean = np.mean(reshaped_for_stats, axis=0)
        std = np.std(reshaped_for_stats, axis=0) + 1e-8
        
        # Broadcasting: (n_win, win, 8, 8) - (1, 1, 8, 8)
        tmp_X = (tmp_X - mean.reshape(1, 1, ch_size, ch_size)) / std.reshape(1, 1, ch_size, ch_size)
        
        # 3. 特徴量抽出 (RMS) -> (n_windows, 8, 8)
        if feature_name == 'rms':
            feats = np.array([rms_feat(x) for x in tmp_X])
        elif feature_name == 'wl':
            feats = np.array([wl_feat(x) for x in tmp_X])
        elif feature_name == 'ptp':
            feats = np.array([ptp_feat(x) for x in tmp_X])
        elif feature_name == 'arv':
            feats = np.array([arv_feat(x) for x in tmp_X])
        elif feature_name == 'zc':
            feats = np.array([zc_feat(x) for x in tmp_X])
        
        # Flatten -> (n_windows, 64)
        feats_flat = feats.reshape(feats.shape[0], -1)
        
        X_features.append(feats_flat)
        # ラベルの拡張
        y_labels.extend([int(label)-1] * feats_flat.shape[0]) # 0-indexed labels
        
    if len(X_features) > 0:
        return np.vstack(X_features), np.array(y_labels)
    else:
        return np.array([]), np.array([])

# ==========================================
# 3. Model Definitions
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

def calculate_prototypes(embeddings, labels, n_classes):
    prototypes = []
    for c in range(n_classes):
        mask = (labels == c)
        if mask.sum() > 0:
            prototypes.append(embeddings[mask].mean(0))
        else:
            prototypes.append(torch.zeros(embeddings.size(1)).to(embeddings.device))
    return torch.stack(prototypes)

# ==========================================
# 4. Main Comparison Logic
# ==========================================

def main(feature_name='wl'):
    subjects = ['nojima', 'takeuchi2', 'yamamoto', 'stefan']
    test_positions = ["original2", "downleft5mm", "downleft10mm", "clockwise"]
    
    # 結果保存用コンテナ
    global_results = {
        "Method1 (Adaptive)": {pos: [] for pos in test_positions},
        "Method2 (LDA)":      {pos: [] for pos in test_positions}
    }
    
    print(f"Comparison started for subjects: {subjects}")

    for subject in subjects:
        print(f"\n{'='*20} Processing Subject: {subject} {'='*20}")
        
        # --- 1. 学習データの準備 (Source Domain: Original) ---
        emg_list_train = []
        y_list_train = []
        for j in range(7):
            for k in range(5):
                try:
                    file_name = file_name_output(subject=subject, hand='right', electrode_place="original", gesture=j+1, trial=k+1)
                    path = '../../data/highMVC/' + file_name
                    df = pd.read_csv(path, encoding='utf-8-sig', sep=';', header=None) 
                    time_emg = df.iloc[:, 0].values
                    emg_data = df.iloc[:, 1:65].values
                    fs = int(1 / np.mean(np.diff(time_emg)))
                    emg_data = remove_power_line_harmonics(emg_data, fs=fs)
                    filtered_emg = butter_bandpass_filter(emg_data, fs=fs, low_hz=20.0, high_hz=450.0)
                    emg_data = filtered_emg.reshape(-1,8,8)
                    emg_list_train.append(emg_data)
                    y_list_train.append(j+1)
                except FileNotFoundError:
                    pass
        
        if not emg_list_train:
            print(f"  [Error] No training data found for {subject}. Skipping.")
            continue

        # 特徴量抽出 (RMS + 正規化)
        X_train, y_train = process_emg_data(emg_list_train, y_list_train, feature_name=feature_name)
        
        # --- Model 1: ProtoNet Training ---
        print("  Training ProtoNet (Method 1)...")
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.LongTensor(y_train).to(device)
        
        model = ProtoNet(input_dim=X_train.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0005)
        n_epochs = 300
        n_classes = 7
        
        model.train()
        for epoch in range(n_epochs):
            embeddings = model(X_train_t)
            prototypes = calculate_prototypes(embeddings, y_train_t, n_classes)
            dists = euclidean_dist(embeddings, prototypes)
            log_p_y = torch.log_softmax(-dists, dim=1)
            loss = -log_p_y.gather(1, y_train_t.view(-1, 1)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # --- Model 2: LDA Training (Source) ---
        print("  Training LDA (Method 2)...")
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train, y_train)

        # --- 2. テストデータの評価 (Target Domains) ---
        print("  Evaluating...")
        model.eval()
        
        for electrode_place in test_positions:
            emg_list_test = []
            y_list_test = []
            for j in range(7):
                for k in range(5):
                    try:
                        file_name = file_name_output(subject=subject, electrode_place=electrode_place, gesture=j+1, trial=k+1)
                        path = '../../data/highMVC/' + file_name
                        df = pd.read_csv(path, encoding='utf-8-sig', sep=';', header=None) 
                        time_emg = df.iloc[:, 0].values
                        emg_data = df.iloc[:, 1:65].values
                        fs = int(1 / np.mean(np.diff(time_emg)))
                        emg_data = remove_power_line_harmonics(emg_data, fs=fs)
                        filtered_emg = butter_bandpass_filter(emg_data, fs=fs, low_hz=20.0, high_hz=450.0)
                        emg_data = filtered_emg.reshape(-1,8,8)
                        emg_list_test.append(emg_data)
                        y_list_test.append(j+1)
                    except FileNotFoundError:
                        pass
            
            if not emg_list_test:
                continue

            acc_buffer_m1 = []
            acc_buffer_m2 = []
            
            # One-Shot Split Evaluation (5-fold)
            # ProtoNetは適応のためSplit必須。LDAも比較のため同じQueryセットを使用。
            for target_shot_index in range(5):
                # Split indices
                trials_per_gesture = 5
                support_indices_flat = [] # リスト内のインデックス
                query_indices_flat = []
                
                # ジェスチャーごとにSupport(1試行)とQuery(4試行)を分ける
                for g_idx in range(7):
                    start = g_idx * trials_per_gesture
                    target = start + target_shot_index
                    if target < len(emg_list_test):
                         support_indices_flat.append(target)
                         # 残りをQueryに
                         others = list(range(start, start + trials_per_gesture))
                         others.remove(target)
                         query_indices_flat.extend([idx for idx in others if idx < len(emg_list_test)])

                # データ抽出
                emg_support = [emg_list_test[i] for i in support_indices_flat]
                y_support_raw = [y_list_test[i] for i in support_indices_flat]
                emg_query = [emg_list_test[i] for i in query_indices_flat]
                y_query_raw = [y_list_test[i] for i in query_indices_flat]

                # 特徴量抽出 (Supportは最初の2秒分のみ使う設定を継承)
                # 注: process_emg_data内で全データを処理するため、ここでは簡易的に全データ渡す
                # 厳密なOne-Shot設定(Supportは短時間)にする場合、emgデータをスライスしてから渡す必要がありますが
                # ここでは提示コードの簡略化のため全データを渡して処理します。
                
                X_support, y_support = process_emg_data(emg_support, y_support_raw, feature_name=feature_name)
                X_query, y_query = process_emg_data(emg_query, y_query_raw, feature_name=feature_name)

                if len(X_query) == 0: continue

                # --- Method 1: ProtoNet Adaptation ---
                X_support_t = torch.FloatTensor(X_support).to(device)
                y_support_t = torch.LongTensor(y_support).to(device)
                X_query_t = torch.FloatTensor(X_query).to(device)
                y_query_t = torch.LongTensor(y_query).to(device)

                with torch.no_grad():
                    query_embeddings = model(X_query_t)
                    support_embeddings = model(X_support_t)
                    adapted_prototypes = calculate_prototypes(support_embeddings, y_support_t, n_classes)
                    
                    dists_m1 = euclidean_dist(query_embeddings, adapted_prototypes)
                    y_pred_m1 = torch.argmin(dists_m1, dim=1)
                    score_m1 = (y_pred_m1 == y_query_t).float().mean().item()
                    
                # --- Method 2: LDA (Source Only) ---
                # LDAはSupportセットを見ずに、Source学習済みのモデルでPredictする
                y_pred_m2 = lda.predict(X_query)
                score_m2 = np.mean(y_pred_m2 == y_query)

                global_results["Method1 (Adaptive)"][electrode_place].append(score_m1)
                global_results["Method2 (LDA)"][electrode_place].append(score_m2)
                
                acc_buffer_m1.append(score_m1)
                acc_buffer_m2.append(score_m2)

            print(f"    {electrode_place}: M1(Proto)={np.mean(acc_buffer_m1):.3f}, M2(LDA)={np.mean(acc_buffer_m2):.3f}")

    # ==========================================
    # 5. Statistical Analysis
    # ==========================================
    print("\n\n" + "#"*60)
    print("FINAL STATISTICAL RESULTS (All Subjects Aggregated)")
    print("#"*60)
    
    all_acc_m1_mean = []
    all_acc_m2_mean = []

    # --- Analysis 1: Method Comparison per Position ---
    print("\n>>> Comparison: Method 1 (Adaptation) vs Method 2 (LDA) per Position")
    
    for electrode_place in test_positions:
        acc_list_m1 = global_results["Method1 (Adaptive)"][electrode_place]
        acc_list_m2 = global_results["Method2 (LDA)"][electrode_place]
        
        m1_mean = np.mean(acc_list_m1)
        m2_mean = np.mean(acc_list_m2)
        all_acc_m1_mean.append(m1_mean)
        all_acc_m2_mean.append(m2_mean)
        
        print(f'\n[Method1: Adaptation] accuracy of {electrode_place}: {m1_mean:.4f}')
        print(f'[Method2: LDA       ] accuracy of {electrode_place}: {m2_mean:.4f}')

        if len(acc_list_m1) > 1:
            t_stat, p_val = ttest_rel(acc_list_m1, acc_list_m2)
            sig_mark = "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"   => Paired t-test: p-value = {p_val:.4e} ({sig_mark})")
        else:
            print("   => Not enough samples for t-test.")

    print("\n" + "-"*60)
    print(f'Method1 (Adaptation) accracy_all: {np.mean(all_acc_m1_mean):.4f}')
    print(f'Method2 (LDA)        accracy_all: {np.mean(all_acc_m2_mean):.4f}')
    print("-"*60)

    # --- Analysis 2: Position Comparison within each Method ---
    print("\n\n>>> Comparison: Between Electrode Positions (Tukey HSD)")
    
    for method_name in ["Method1 (Adaptive)", "Method2 (LDA)"]:
        print(f"\n[{method_name}] Multiple Comparison Results:")
        
        all_scores = []
        all_groups = []
        
        for pos in test_positions:
            scores = global_results[method_name][pos]
            all_scores.extend(scores)
            all_groups.extend([pos] * len(scores))
            
        if len(all_scores) > 0:
            try:
                tukey = pairwise_tukeyhsd(endog=all_scores, groups=all_groups, alpha=0.05)
                print(tukey)
            except Exception as e:
                print(f"Error in Tukey HSD: {e}")

if __name__ == "__main__":
    main(feature_name='wl')