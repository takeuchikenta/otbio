import wfdb
import os
import glob
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, sosfiltfilt, iirnotch
from scipy.ndimage import median_filter
import warnings
import torch
import torch.nn as nn
import torch.optim as optim

# 不要な警告を無視
warnings.filterwarnings('ignore')
np.warnings = warnings

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. Helper Functions (共通処理)
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
    # パスは環境に合わせてください
    file_name = f"{subject}/{hand}/{filename}/set{trial}/{electrode_place}-g{gesture}-{trial}.csv"
    return file_name

# ==========================================
# 2. Dataset & Feature Extraction
# ==========================================

def segment_time_series(emg_6x6: np.ndarray, window: int, hop: int) -> np.ndarray:
    n, _, _ = emg_6x6.shape
    idx_starts = np.arange(0, max(1, n - window + 1), hop)
    segs = np.stack([emg_6x6[s:s+window] for s in idx_starts if s + window <= n], axis=0)
    return segs

def extract_wl_feature_map(window_data):
    """Waveform Length (Time, 8, 8) -> (8, 8)"""
    return np.sum(np.abs(np.diff(window_data, axis=0)), axis=0)

def apply_median_filter(feature_map, kernel_size=3):
    return median_filter(feature_map, size=kernel_size, mode='reflect')

def prepare_data_indices(n_trials, n_time, is_support_set=False, shot_trial_idx=0):
    trials_per_gesture = 5
    n_gestures = 7
    indices = []
    
    for g in range(n_gestures):
        start_trial_idx = g * trials_per_gesture
        target_idx = start_trial_idx + shot_trial_idx
        
        if is_support_set:
            trial_indices = [target_idx] 
            time_limit = 2000 # Supportは短く
        else:
            all_indices = list(range(start_trial_idx, start_trial_idx + trials_per_gesture))
            all_indices.remove(target_idx)
            trial_indices = all_indices
            time_limit = n_time
            
        indices.append((g, trial_indices, time_limit))
    return indices

def create_dataset_proto(emg_list, y_list, fs=2000, window_ms=200, step_ms=50, mode='full', shot_trial_idx=0):
    """
    ProtoNet用データセット作成: WL特徴量 + Median Filter
    """
    data = np.array(emg_list) 
    n_trials, n_time, _, _ = data.shape
    
    window_samples = int((window_ms / 1000) * fs)
    step_samples = int((step_ms / 1000) * fs)
    
    X_feat = []
    y_labels = []
    
    if mode == 'full':
        target_config = [(g, range(g*5, (g+1)*5), n_time) for g in range(7)]
    elif mode == 'support':
        target_config = prepare_data_indices(5, n_time, is_support_set=True, shot_trial_idx=shot_trial_idx)
    elif mode == 'query':
        target_config = prepare_data_indices(5, n_time, is_support_set=False, shot_trial_idx=shot_trial_idx)
    
    for gesture_label, trial_indices, time_limit in target_config:
        for t_idx in trial_indices:
            trial_data = data[t_idx]
            limit = min(time_limit, trial_data.shape[0])
            valid_data = trial_data[:limit, :]
            
            # Segment
            tmp_X = segment_time_series(valid_data, window=window_samples, hop=step_samples)
            
            if tmp_X.shape[0] == 0: continue

            # Feature Extraction
            for i in range(tmp_X.shape[0]):
                win_data = tmp_X[i]
                # WL
                feat_map = extract_wl_feature_map(win_data)
                # Median Filter
                feat_map_filtered = apply_median_filter(feat_map, kernel_size=3)
                
                X_feat.append(feat_map_filtered.flatten())
                y_labels.append(gesture_label)
                
    return np.array(X_feat), np.array(y_labels)

# ==========================================
# 3. Model
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
# 4. Main Execution (Multi-Subject & Data Export)
# ==========================================

def main():
    # ### 修正点1: 全被験者リスト ###
    subjects = ['nojima', 'takeuchi2', 'yamamoto', 'stefan']
    test_positions = ["original2", "downleft5mm", "downleft10mm", "clockwise"]
    
    # ### 修正点2: 結果保存用リスト (DataFrame化してCSV保存するため) ###
    all_results_data = [] 

    print(f"Starting ProtoNet Analysis for: {subjects}")

    for subject in subjects:
        print(f"\n{'='*20} Processing Subject: {subject} {'='*20}")
        
        # --- A. Load Training Data ---
        emg_list_train = []
        y_list_train = []
        for j in range(7):
            for k in range(5):
                try:
                    file_name = file_name_output(subject, "right", "original", j+1, k+1)
                    path = '../../data/highMVC/' + file_name
                    df = pd.read_csv(path, encoding='utf-8-sig', sep=';', header=None) 
                    time_emg = df.iloc[:, 0].values
                    emg_data = df.iloc[:, 1:65].values
                    fs = int(1 / np.mean(np.diff(time_emg)))
                    emg_data = remove_power_line_harmonics(emg_data, fs, 60.0)
                    filtered_emg = butter_bandpass_filter(emg_data, fs, 20.0, 450.0)
                    emg_data = filtered_emg.reshape(-1,8,8)
                    emg_list_train.append(emg_data)
                    y_list_train.append(j+1)
                except FileNotFoundError:
                    pass
        
        if not emg_list_train:
            print("Skipping (No Data)")
            continue

        # --- B. Train ProtoNet (Source Domain) ---
        print("  Training Embedding Network...")
        # Create dataset (Full training set)
        X_train, y_train = create_dataset_proto(emg_list_train, [y-1 for y in y_list_train], fs=fs, mode='full')
        
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.LongTensor(y_train).to(device)
        
        model = ProtoNet(input_dim=64).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0005)
        model.train()
        
        for epoch in range(300):
            embeddings = model(X_train_t)
            prototypes = calculate_prototypes(embeddings, y_train_t, 7)
            dists = euclidean_dist(embeddings, prototypes)
            loss = -torch.log_softmax(-dists, dim=1).gather(1, y_train_t.view(-1, 1)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # --- C. Test (Target Domain Adaptation) ---
        print("  Testing & Adapting...")
        model.eval()
        
        for electrode_place in test_positions:
            emg_list_test = []
            y_list_test = []
            for j in range(7):
                for k in range(5):
                    try:
                        file_name = file_name_output(subject, "right", electrode_place, j+1, k+1)
                        path = '../../data/highMVC/' + file_name
                        df = pd.read_csv(path, encoding='utf-8-sig', sep=';', header=None) 
                        time_emg = df.iloc[:, 0].values
                        emg_data = df.iloc[:, 1:65].values
                        fs = int(1 / np.mean(np.diff(time_emg)))
                        emg_data = remove_power_line_harmonics(emg_data, fs, 60.0)
                        filtered_emg = butter_bandpass_filter(emg_data, fs, 20.0, 450.0)
                        emg_data = filtered_emg.reshape(-1,8,8)
                        emg_list_test.append(emg_data)
                        y_list_test.append(j+1)
                    except FileNotFoundError:
                        pass
            
            if not emg_list_test: continue

            # One-Shot Split Evaluation (5-fold)
            acc_list = []
            for shot_idx in range(5):
                y_corr = [y-1 for y in y_list_test]
                
                # Support: shot_idx, Query: others
                X_supp, y_supp = create_dataset_proto(emg_list_test, y_corr, fs=fs, mode='support', shot_trial_idx=shot_idx)
                X_query, y_query = create_dataset_proto(emg_list_test, y_corr, fs=fs, mode='query', shot_trial_idx=shot_idx)
                
                if len(X_query) == 0: continue

                with torch.no_grad():
                    supp_t = torch.FloatTensor(X_supp).to(device)
                    ysupp_t = torch.LongTensor(y_supp).to(device)
                    query_t = torch.FloatTensor(X_query).to(device)
                    yquery_t = torch.LongTensor(y_query).to(device)
                    
                    # Adapt
                    supp_emb = model(supp_t)
                    query_emb = model(query_t)
                    
                    new_prototypes = calculate_prototypes(supp_emb, ysupp_t, 7)
                    dists = euclidean_dist(query_emb, new_prototypes)
                    preds = torch.argmin(dists, dim=1)
                    
                    score = (preds == yquery_t).float().mean().item()
                    acc_list.append(score)
                    
                    # ### 修正点3: データの記録 ###
                    # 後で他手法と比較分析するために生データを保存
                    all_results_data.append({
                        "Subject": subject,
                        "Method": "ProtoNet(Adaptive)",
                        "Position": electrode_place,
                        "Fold": shot_idx,
                        "Accuracy": score
                    })

            print(f"    {electrode_place}: Mean Acc = {np.mean(acc_list):.4f}")

    # ### 修正点4: CSV出力 ###
    print("\n" + "="*40)
    print("Saving results to 'protonet_results.csv'...")
    df_results = pd.DataFrame(all_results_data)
    df_results.to_csv("protonet_results.csv", index=False)
    print(df_results.groupby(["Subject", "Position"])["Accuracy"].mean())
    print("Done.")

if __name__ == "__main__":
    main()