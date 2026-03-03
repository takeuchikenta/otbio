import wfdb
import os
import glob
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, sosfiltfilt, iirnotch
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import median_filter
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import cv2
import warnings
import time
import torch
import torch.nn as nn
import torch.optim as optim

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
    # パス構成は環境に合わせて調整してください
    file_name = f"{subject}/{hand}/{filename}/set{trial}/{electrode_place}-g{gesture}-{trial}.csv"
    return file_name

# ==========================================
# 2. Features & Warping
# ==========================================

def segment_time_series(emg_6x6: np.ndarray, window: int, hop: int) -> np.ndarray:
    n, _, _ = emg_6x6.shape
    idx_starts = np.arange(0, max(1, n - window + 1), hop)
    segs = np.stack([emg_6x6[s:s+window] for s in idx_starts if s + window <= n], axis=0)
    return segs

def wl_feat(x):
    return np.sum(np.abs(np.diff(x, axis=0)), axis=0)

def extract_wl_feature_map(window_data):
    """(Time, 8, 8) -> (8, 8) WL map"""
    # チャンネル軸(Time, 8, 8)に対して時間軸(axis=0)で計算
    # 入力が(Time, 64)の場合は reshapeが必要だが、
    # 今回のコードでは (Time, 8, 8) で回している想定
    if window_data.ndim == 3:
        wl = np.sum(np.abs(np.diff(window_data, axis=0)), axis=0)
    else:
        # channel flat case
        wl_flat = np.sum(np.abs(np.diff(window_data, axis=0)), axis=0)
        wl = wl_flat.reshape(8, 8)
    return wl

def apply_median_filter(feature_map, kernel_size=3):
    return median_filter(feature_map, size=kernel_size, mode='reflect')

# --- Warping Functions ---

def warp_emg_8x8(emg, tx, ty, theta, sx, sy, shear):
    H, W = emg.shape
    x = np.arange(H)
    y = np.arange(W)
    interp = RectBivariateSpline(y, x, emg, kx=3, ky=3)

    cx = (H - 1) / 2
    cy = (W - 1) / 2

    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    S = np.array([[sx, 0],
                  [0, sy]])
    Sh = np.array([[1, shear],
                   [0, 1]])

    A = R @ Sh @ S

    T_center = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
    T_center_inv = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
    T_shift = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

    M_affine = np.eye(3)
    M_affine[:2, :2] = A
    M = T_shift @ T_center_inv @ M_affine @ T_center
    M_inv = np.linalg.inv(M)

    grid_x, grid_y = np.meshgrid(x, y)
    ones = np.ones_like(grid_x)
    coords = np.stack([grid_x, grid_y, ones], axis=-1).reshape(-1, 3).T
    src = M_inv @ coords
    src_x, src_y = src[0], src[1]

    emg_new = interp.ev(src_y, src_x).reshape(H, W)
    return emg_new

def warp_batch_images(batch_data, tx, ty, theta, sx, sy, shear):
    batch_data = np.array(batch_data)
    original_shape = batch_data.shape
    flat_data = batch_data.reshape(-1, original_shape[-2], original_shape[-1])
    processed_list = []
    for img in flat_data:
        warped_img = warp_emg_8x8(img, tx, ty, theta, sx, sy, shear)
        processed_list.append(warped_img)
    return np.array(processed_list).reshape(original_shape)

def warp_emg_list(emg_list, tx, ty, theta, sx, sy, shear):
    emg_list_warped = []
    for emg_test_8x8 in emg_list:
        tmp_X = warp_batch_images(emg_test_8x8, tx, ty, theta, sx, sy, shear)
        emg_list_warped.append(tmp_X)
    return emg_list_warped

# --- Optimization Objectives ---

def affine_transform_cv2(img, params):
    a, b, c, d, tx, ty = params
    M = np.array([[a, b, tx],
                  [c, d, ty]], dtype=np.float32)
    h, w = img.shape
    return cv2.warpAffine(img, M, (w, h))

def ncc(a, b):
    if np.std(a) == 0 or np.std(b) == 0: return 0.0
    a_mean = a - np.mean(a)
    b_mean = b - np.mean(b)
    num = np.sum(a_mean * b_mean)
    den = np.sqrt(np.sum(a_mean**2)) * np.sqrt(np.sum(b_mean**2)) + 1e-8
    return float(num / den)

def objective_ncc(params, ref, mov):
    warped = affine_transform_cv2(mov, params)
    return 1 - ncc(ref, warped)

# ==========================================
# 3. ProtoNet Definitions
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

def prepare_data_indices(n_trials, n_time, fs, is_support_set=False, shot_trial_idx=0):
    trials_per_gesture = 5
    n_gestures = 7
    indices = []
    if not (0 <= shot_trial_idx < trials_per_gesture):
        raise ValueError(f"shot_trial_idx must be between 0 and {trials_per_gesture - 1}")

    for g in range(n_gestures):
        start_trial_idx = g * trials_per_gesture
        target_idx = start_trial_idx + shot_trial_idx
        if is_support_set:
            trial_indices = [target_idx] 
            time_limit = 2000 
        else:
            all_indices = list(range(start_trial_idx, start_trial_idx + trials_per_gesture))
            all_indices.remove(target_idx)
            trial_indices = all_indices
            time_limit = n_time
        indices.append((g, trial_indices, time_limit))
    return indices

def create_dataset_proto(emg_list, y_list, fs=2000, window_ms=200, step_ms=50, mode='full', shot_trial_idx=0):
    """
    ProtoNet用のデータセット作成
    segment -> normalize -> WL -> Median Filter -> Flatten
    """
    data = np.array(emg_list) # (trials, time, 8, 8)
    n_trials, n_time, _, _ = data.shape
    # Flatten channels for indexing logic, but we need 2D for feature extraction
    # Logic in previous code extracted features from (Window, 8, 8).
    
    window_samples = int((window_ms / 1000) * fs)
    step_samples = int((step_ms / 1000) * fs)
    ch_size = 8

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
            trial_data = data[t_idx] # (Time, 8, 8)
            limit = min(time_limit, trial_data.shape[0])
            valid_data = trial_data[:limit]
            
            # Segment
            tmp_X = segment_time_series(valid_data, window=window_samples, hop=step_samples) # (n_win, win, 8, 8)
            
            if tmp_X.shape[0] == 0: continue

            # Normalize (Channel-wise per batch/window)
            # 提示コードのロジック: バッチ全体でのチャネル毎正規化
            # mean = np.mean(tmp_X.reshape(-1,8,8), axis=0)
            # std = np.std(tmp_X.reshape(-1,8,8), axis=0) + 1e-8
            # tmp_X = (tmp_X - mean.reshape(1, 1, 8, 8)) / std.reshape(1, 1, 8, 8)
            
            # ここではProtoNet用なので、元のコードに従い「WL抽出 -> Median Filter」の流れ
            # 正規化は特徴抽出前に行うかコードによるが、提示されたProtoNetの元のコードでは
            # 特徴抽出後に特に正規化していないか、学習ループ内で処理されている。
            # ここでは提示されたLDA/Alignment用コードの正規化を適用してからWLを抽出する形で統一します。
            mean = np.mean(tmp_X.reshape(-1,8,8), axis=0)
            std = np.std(tmp_X.reshape(-1,8,8), axis=0) + 1e-8
            tmp_X = (tmp_X - mean.reshape(1, 1, 8, 8)) / std.reshape(1, 1, 8, 8)

            for i in range(tmp_X.shape[0]):
                win_data = tmp_X[i]
                # 1. WL
                feat_map = extract_wl_feature_map(win_data)
                # 2. Median Filter
                feat_map_filtered = apply_median_filter(feat_map, kernel_size=3)
                # 3. Flatten
                X_feat.append(feat_map_filtered.flatten())
                y_labels.append(gesture_label)
                
    return np.array(X_feat), np.array(y_labels)


# ==========================================
# 4. Main Process
# ==========================================

def main():
    subjects = ['nojima', 'takeuchi2', 'yamamoto', 'stefan']
    test_positions = ["original2", "downleft5mm", "downleft10mm", "clockwise"]
    
    # 統計用データ蓄積 {position: [accuracy_list_from_all_subjects]}
    stats_data = {pos: [] for pos in test_positions}
    
    print(f"Comparison started for subjects: {subjects}")

    for subject in subjects:
        print(f"\n{'='*20} Processing Subject: {subject} {'='*20}")
        
        # --- A. Load Training Data (Original) ---
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
                    emg_data = remove_power_line_harmonics(emg_data, fs=fs, fundamental=60.0, Q=30.0)
                    filtered_emg = butter_bandpass_filter(emg_data, fs=fs, low_hz=20.0, high_hz=450.0, order=4)
                    emg_data = filtered_emg.reshape(-1,8,8)
                    emg_list_train.append(emg_data)
                    y_list_train.append(j+1)
                except FileNotFoundError:
                    pass
        
        if not emg_list_train:
            print(f"  No training data for {subject}. Skipping.")
            continue

        # --- B. Train ProtoNet & Build Alignment Maps ---
        # 1. Train ProtoNet
        print("  Training ProtoNet...")
        # ProtoNet学習用データセット (WL, MedianFilter, Flattened)
        X_train_proto, y_train_proto = create_dataset_proto(emg_list_train, y_list_train, fs=fs, mode='full')
        X_train_t = torch.FloatTensor(X_train_proto).to(device)
        y_train_t = torch.LongTensor(y_train_proto).to(device)
        
        model = ProtoNet(input_dim=X_train_proto.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0005)
        model.train()
        for epoch in range(300): # 300 epochs
            embeddings = model(X_train_t)
            prototypes = []
            for c in range(7):
                mask = (y_train_t == c)
                if mask.sum() > 0:
                    prototypes.append(embeddings[mask].mean(0))
                else:
                    prototypes.append(torch.zeros(128).to(device))
            prototypes = torch.stack(prototypes)
            dists = euclidean_dist(embeddings, prototypes)
            loss = -torch.log_softmax(-dists, dim=1).gather(1, y_train_t.view(-1, 1)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 2. Build Alignment Maps (Reference)
        # アライメント用の特徴量(WL)を抽出。ProtoNet用とは別に、Map(8x8)のまま保持する必要がある。
        # 提示コードのロジック: Segment -> Normalize -> WL -> (No Median Filter for Alignment?) -> Average over 19 windows
        
        window = 200
        hop = 50
        window_samples = int(window * (fs/1000))
        hop_samples = int(hop * (fs/1000))
        ch_size = 8
        
        X_train_alignment_raw = []
        y_train_alignment_raw = []

        for i, (emg_train_8x8, label) in enumerate(zip(emg_list_train, y_list_train)):
            tmp_X = segment_time_series(emg_train_8x8, window=window_samples, hop=hop_samples)
            mean = np.mean(tmp_X.reshape(-1,8,8), axis=0)
            std = np.std(tmp_X.reshape(-1,8,8), axis=0) + 1e-8
            tmp_X = (tmp_X - mean.reshape(1, 1, 8, 8)) / std.reshape(1, 1, 8, 8)
            
            # Extract WL (Result: n_win, 8, 8)
            # ここではflattenせず、Mapとして保持
            wl_maps = np.array([wl_feat(x) for x in tmp_X]) # (n_win, 8, 8)
            
            X_train_alignment_raw.append(wl_maps)
            y_train_alignment_raw.extend([int(label)-1] * len(wl_maps))

        X_train_alignment_flat = np.vstack(X_train_alignment_raw)
        y_train_alignment_flat = np.array(y_train_alignment_raw)

        # 1秒(約19窓)ごとの平均マップを作成
        train_trial_alignment_maps = []
        gestures = np.unique(y_train_alignment_flat) + 1
        for gesture in gestures:
            trial_list = []
            z_list = []
            iterator = X_train_alignment_flat[y_train_alignment_flat==gesture-1]
            idx = 0
            for val in iterator:
                z_list.append(val)
                idx += 1
                if idx == 19:
                    trial_list.append(np.mean(np.array(z_list), axis=0))
                    z_list = []
                    idx = 0
            train_trial_alignment_maps.append(trial_list)

        # --- C. Test Phase (Loop Positions) ---
        model.eval()
        for electrode_place in test_positions:
            print(f'  Testing: {electrode_place}')
            emg_list_test = []
            y_list_test = []
            for j in range(7):
                for k in range(5):
                    try:
                        file_name = file_name_output(subject=subject, hand='right', electrode_place=electrode_place, gesture=j+1, trial=k+1)
                        path = '../../data/highMVC/' + file_name
                        df = pd.read_csv(path, encoding='utf-8-sig', sep=';', header=None) 
                        time_emg = df.iloc[:, 0].values
                        emg_data = df.iloc[:, 1:65].values
                        fs = int(1 / np.mean(np.diff(time_emg)))
                        emg_data = remove_power_line_harmonics(emg_data, fs=fs, fundamental=60.0, Q=30.0)
                        filtered_emg = butter_bandpass_filter(emg_data, fs=fs, low_hz=20.0, high_hz=450.0, order=4)
                        emg_data = filtered_emg.reshape(-1,8,8)
                        emg_list_test.append(emg_data)
                        y_list_test.append(j+1)
                    except FileNotFoundError:
                        pass
            
            if not emg_list_test:
                continue

            # 1. Prepare Test Alignment Maps (Pre-Warping)
            X_test_alignment_raw = []
            y_test_alignment_raw = []
            
            for i, (emg_test_8x8, label) in enumerate(zip(emg_list_test, y_list_test)):
                tmp_X = segment_time_series(emg_test_8x8, window=window_samples, hop=hop_samples)
                mean = np.mean(tmp_X.reshape(-1,8,8), axis=0)
                std = np.std(tmp_X.reshape(-1,8,8), axis=0) + 1e-8
                tmp_X = (tmp_X - mean.reshape(1, 1, 8, 8)) / std.reshape(1, 1, 8, 8)
                wl_maps = np.array([wl_feat(x) for x in tmp_X])
                X_test_alignment_raw.append(wl_maps)
                y_test_alignment_raw.extend([int(label)-1] * len(wl_maps))

            X_test_alignment_flat = np.vstack(X_test_alignment_raw)
            y_test_alignment_flat = np.array(y_test_alignment_raw)

            test_trial_alignment_maps = []
            for gesture in gestures:
                trial_list = []
                z_list = []
                iterator = X_test_alignment_flat[y_test_alignment_flat==gesture-1]
                idx = 0
                for val in iterator:
                    z_list.append(val)
                    idx += 1
                    if idx == 19:
                        trial_list.append(np.mean(np.array(z_list), axis=0))
                        z_list = []
                        idx = 0
                test_trial_alignment_maps.append(trial_list)

            # 2. Alignment & Inference Loop
            # 各試行(One-Shot Trial)について最適化 -> Warping -> Adaptation -> Accuracy
            subject_pos_accuracies = []

            for gesture in range(1, 8):
                for trial in range(1, 6): # 1-based index
                    try:
                        # --- Optimization ---
                        # Train(Reference) と Test(Target) のMapを取得
                        ref_len = len(train_trial_alignment_maps[gesture-1])
                        test_len = len(test_trial_alignment_maps[gesture-1])
                        if ref_len == 0 or test_len == 0: continue
                        
                        ref_idx = min(ref_len-1, trial*3-1)
                        test_idx = min(test_len-1, trial*3-1)

                        scalar = MinMaxScaler()
                        img_ref = (scalar.fit_transform(train_trial_alignment_maps[gesture-1][ref_idx].reshape(-1,1)).reshape(8,8)*255).astype(np.float32)
                        img_test = (scalar.fit_transform(test_trial_alignment_maps[gesture-1][test_idx].reshape(-1,1)).reshape(8,8)*255).astype(np.float32)

                        init = [1, 0, 0, 1, 0, 0]
                        res = minimize(objective_ncc, x0=init, args=(img_test, img_ref), method="Powell")
                        a, b, c, d, tx, ty = res.x

                        theta_rad = np.arctan2(c, a)
                        sx = np.sqrt(a**2 + c**2)
                        sy = np.sqrt(b**2 + d**2)
                        shear = (a*b + c*d) / (sx*sy)

                        # Params for Warping
                        p_tx, p_ty = -tx, -ty
                        p_theta = theta_rad
                        p_sx, p_sy = sx, sy
                        p_shear = shear

                        # --- Warping ---
                        # 指定されたtrial *以外* のデータをWarpしてテストセットとする (Leave-one-out logic)
                        # trial: 1..5. emg_list_test index: 0..34.
                        # (i % 5) + 1 == trial ならば、それは最適化に使った試行なので除外(Support相当)してQueryを評価
                        # または、One-Shot Adaptationなので、Support(最適化に使った試行)とQuery(それ以外)を用意する
                        
                        # 生データをWarp
                        emg_list_warped = warp_emg_list(emg_list_test, p_tx, p_ty, p_theta, p_sx, p_sy, p_shear)
                        
                        # Create Dataset for ProtoNet (Support: trial, Query: others)
                        X_supp, y_supp = create_dataset_proto(emg_list_warped, y_list_test, mode='support', shot_trial_idx=trial-1)
                        X_query, y_query = create_dataset_proto(emg_list_warped, y_list_test, mode='query', shot_trial_idx=trial-1)

                        if len(X_query) == 0: continue

                        # --- Adaptation & Inference ---
                        X_supp_t = torch.FloatTensor(X_supp).to(device)
                        y_supp_t = torch.LongTensor(y_supp).to(device)
                        X_query_t = torch.FloatTensor(X_query).to(device)
                        y_query_t = torch.LongTensor(y_query).to(device)

                        with torch.no_grad():
                            supp_emb = model(X_supp_t)
                            query_emb = model(X_query_t)
                            
                            # Adapt Prototypes using Support Set
                            new_prototypes = []
                            for c_idx in range(7):
                                mask = (y_supp_t == c_idx)
                                if mask.sum() > 0:
                                    new_prototypes.append(supp_emb[mask].mean(0))
                                else:
                                    new_prototypes.append(torch.zeros(128).to(device))
                            new_prototypes = torch.stack(new_prototypes)
                            
                            # Predict Query
                            dists = euclidean_dist(query_emb, new_prototypes)
                            y_pred = torch.argmin(dists, dim=1)
                            score = (y_pred == y_query_t).float().mean().item()
                            
                            subject_pos_accuracies.append(score)

                    except Exception as e:
                        # print(f"Error: {e}")
                        pass
            
            # Store results for this position
            if subject_pos_accuracies:
                mean_acc = np.mean(subject_pos_accuracies)
                print(f"    Mean Acc: {mean_acc:.4f}")
                # 統計検定用に全スコアを保存
                stats_data[electrode_place].extend(subject_pos_accuracies)

    # ==========================================
    # 5. Statistical Analysis (Tukey HSD)
    # ==========================================
    print("\n" + "#"*60)
    print("STATISTICAL ANALYSIS: Between Electrode Positions (Tukey HSD)")
    print("#"*60)

    all_scores = []
    all_groups = []

    for pos in test_positions:
        scores = stats_data[pos]
        if scores:
            all_scores.extend(scores)
            all_groups.extend([pos] * len(scores))
            print(f"Position: {pos}, N={len(scores)}, Mean={np.mean(scores):.4f}, Std={np.std(scores):.4f}")

    if all_scores:
        try:
            tukey = pairwise_tukeyhsd(endog=all_scores, groups=all_groups, alpha=0.05)
            print("\n>>> Tukey's HSD Test Results:")
            print(tukey)
        except Exception as e:
            print(f"Could not perform Tukey HSD: {e}")
    else:
        print("No valid data collected for statistics.")

if __name__ == "__main__":
    main()