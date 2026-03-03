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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import ttest_rel
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

def segment_time_series(emg_6x6: np.ndarray, window: int, hop: int) -> np.ndarray:
    n, _, _ = emg_6x6.shape
    idx_starts = np.arange(0, max(1, n - window + 1), hop)
    segs = np.stack([emg_6x6[s:s+window] for s in idx_starts if s + window <= n], axis=0)
    return segs

# ==========================================
# 2. Feature Extraction & Augmentation
# ==========================================

def extract_wl_feature_map(window_data):
    """
    (Window, 8, 8) -> WL Map (8, 8)
    """
    # 時間軸(axis=0)方向に差分絶対値和
    wl_map = np.sum(np.abs(np.diff(window_data, axis=0)), axis=0)
    return wl_map

def apply_median_filter(feature_map, kernel_size=3):
    """
    (8, 8) -> (8, 8) Median Filtered
    """
    return median_filter(feature_map, size=kernel_size, mode='reflect')

def warp_feature_map(feature_map, theta, tx, ty):
    """
    8x8特徴量マップに対するアフィン変換 (回転+並進)
    theta: rad, tx/ty: pixels
    """
    H, W = feature_map.shape
    x = np.arange(H)
    y = np.arange(W)
    interp = RectBivariateSpline(y, x, feature_map, kx=3, ky=3)

    cx = (H - 1) / 2
    cy = (W - 1) / 2

    # 回転行列
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    
    # 中心基準の座標変換
    T_center = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
    T_center_inv = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
    T_shift = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    
    M_affine = np.eye(3)
    M_affine[:2, :2] = R
    
    # 合成変換行列: Shift -> Center_Inv -> Rotate -> Center
    # 逆変換してサンプリング座標を求めるため、順序に注意
    M = T_shift @ T_center_inv @ M_affine @ T_center
    M_inv = np.linalg.inv(M)

    grid_x, grid_y = np.meshgrid(x, y)
    ones = np.ones_like(grid_x)
    coords = np.stack([grid_x, grid_y, ones], axis=-1).reshape(-1, 3).T
    
    src = M_inv @ coords
    src_x, src_y = src[0], src[1]

    warped = interp.ev(src_y, src_x).reshape(H, W)
    return warped

def augment_data(X_maps, y_labels, n_aug=3):
    """
    X_maps: List of (8, 8) feature maps
    y_labels: List of labels
    Return: Augmented arrays (Original + Augmented) flattened
    """
    X_aug_flat = []
    y_aug = []
    
    # パラメータ範囲
    angle_limit = np.radians(15) # ±15度
    shift_limit = 1.0 # ±1チャネル
    
    for fm, label in zip(X_maps, y_labels):
        # オリジナル追加
        X_aug_flat.append(fm.flatten())
        y_aug.append(label)
        
        # 拡張データ生成
        for _ in range(n_aug):
            theta = np.random.uniform(-angle_limit, angle_limit)
            tx = np.random.uniform(-shift_limit, shift_limit)
            ty = np.random.uniform(-shift_limit, shift_limit)
            
            warped = warp_feature_map(fm, theta, tx, ty)
            X_aug_flat.append(warped.flatten())
            y_aug.append(label)
            
    return np.array(X_aug_flat), np.array(y_aug)

def process_features(emg_list, y_list, fs=2000, window_ms=200, hop_ms=50):
    """
    共通の特徴抽出処理: Segment -> WL -> Median Filter
    戻り値: 特徴マップのリスト(flatten前), ラベルリスト
    """
    window = int((window_ms / 1000) * fs)
    hop = int((hop_ms / 1000) * fs)
    
    X_maps = []
    y_labels = []
    
    for emg, label in zip(emg_list, y_list):
        # Segment (n_win, win, 8, 8)
        tmp_X = segment_time_series(emg, window=window, hop=hop)
        if tmp_X.shape[0] == 0: continue
        
        for i in range(tmp_X.shape[0]):
            win_data = tmp_X[i]
            # WL
            wl = extract_wl_feature_map(win_data)
            # Median Filter (Previous Logic)
            filtered = apply_median_filter(wl, kernel_size=3)
            
            X_maps.append(filtered)
            y_labels.append(int(label)-1)
            
    return X_maps, y_labels

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
# 4. Main Processing Logic
# ==========================================

def main():
    subjects = ['nojima', 'takeuchi2', 'yamamoto', 'stefan']
    test_positions = ["original2", "downleft5mm", "downleft10mm", "clockwise"]
    
    # 統計用データ蓄積
    stats_data = {
        "Method1 (Aug+Proto)": {pos: [] for pos in test_positions},
        "Method2 (LDA Only)":  {pos: [] for pos in test_positions}
    }
    
    print(f"Comparison started for subjects: {subjects}")

    for subject in subjects:
        print(f"\n{'='*20} Processing Subject: {subject} {'='*20}")
        
        # --- A. Load Training Data (Original) ---
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

        # 1. Extract Base Features (WL + Median)
        X_maps_train, y_labels_train = process_features(emg_list_train, y_list_train, fs)
        
        # --- B. Train Models ---
        
        # Method 1: ProtoNet with Augmentation
        print("  Training Method 1 (Augmented ProtoNet)...")
        # Augment Data (Original + 4 variations)
        X_train_aug, y_train_aug = augment_data(X_maps_train, y_labels_train, n_aug=4)
        
        X_train_p_t = torch.FloatTensor(X_train_aug).to(device)
        y_train_p_t = torch.LongTensor(y_train_aug).to(device)
        
        proto_model = ProtoNet(input_dim=64).to(device)
        optimizer = optim.Adam(proto_model.parameters(), lr=0.0005)
        proto_model.train()
        
        for epoch in range(300):
            embeddings = proto_model(X_train_p_t)
            prototypes = calculate_prototypes(embeddings, y_train_p_t, 7)
            dists = euclidean_dist(embeddings, prototypes)
            loss = -torch.log_softmax(-dists, dim=1).gather(1, y_train_p_t.view(-1, 1)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Method 2: LDA (No Augmentation, Same Features)
        print("  Training Method 2 (LDA)...")
        # Flatten original maps without augmentation
        X_train_lda = np.array([m.flatten() for m in X_maps_train])
        y_train_lda = np.array(y_labels_train)
        
        lda_model = LinearDiscriminantAnalysis()
        lda_model.fit(X_train_lda, y_train_lda)

        # --- C. Test Phase (Target Positions) ---
        print("  Testing...")
        proto_model.eval()
        
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

            acc_m1_list = []
            acc_m2_list = []

            # 5-fold (One-Shot Split) Evaluation
            for shot_idx in range(5):
                query_emg = []
                query_y = []
                support_emg = []
                support_y = []
                
                # Split raw data into Support and Query based on trial index
                for i, (raw_emg, label) in enumerate(zip(emg_list_test, y_list_test)):
                    current_file_trial = i % 5 
                    if current_file_trial == shot_idx:
                        support_emg.append(raw_emg)
                        support_y.append(label)
                    else:
                        query_emg.append(raw_emg)
                        query_y.append(label)
                
                if not query_emg: continue

                # Extract features (WL + Median)
                X_supp_maps, y_supp_raw = process_features(support_emg, support_y, fs)
                X_query_maps, y_query_raw = process_features(query_emg, query_y, fs)
                
                if len(X_query_maps) == 0: continue

                # Flatten
                X_supp_flat = np.array([m.flatten() for m in X_supp_maps])
                y_supp_flat = np.array(y_supp_raw)
                X_query_flat = np.array([m.flatten() for m in X_query_maps])
                y_query_flat = np.array(y_query_raw)

                # --- Method 1: ProtoNet (One-Shot Adaptation) ---
                with torch.no_grad():
                    supp_t = torch.FloatTensor(X_supp_flat).to(device)
                    ysupp_t = torch.LongTensor(y_supp_flat).to(device)
                    query_t = torch.FloatTensor(X_query_flat).to(device)
                    yquery_t = torch.LongTensor(y_query_flat).to(device)
                    
                    # Embed & Adapt Prototypes using Support Set
                    supp_emb = proto_model(supp_t)
                    query_emb = proto_model(query_t)
                    
                    new_protos = calculate_prototypes(supp_emb, ysupp_t, 7)
                    dists = euclidean_dist(query_emb, new_protos)
                    preds = torch.argmin(dists, dim=1)
                    
                    score_m1 = (preds == yquery_t).float().mean().item()
                    acc_m1_list.append(score_m1)

                # --- Method 2: LDA (No Adaptation) ---
                # Predict Query directly using Source Model
                pred_lda = lda_model.predict(X_query_flat)
                score_m2 = np.mean(pred_lda == y_query_flat)
                acc_m2_list.append(score_m2)

            # Store averages per position
            if acc_m1_list:
                stats_data["Method1 (Aug+Proto)"][electrode_place].extend(acc_m1_list)
                stats_data["Method2 (LDA Only)"][electrode_place].extend(acc_m2_list)
                print(f"    {electrode_place}: M1(Aug+Proto)={np.mean(acc_m1_list):.4f}, M2(LDA)={np.mean(acc_m2_list):.4f}")

    # ==========================================
    # 5. Statistical Analysis
    # ==========================================
    print("\n" + "#"*60)
    print("FINAL STATISTICAL RESULTS")
    print("#"*60)

    # 1. Compare Methods per Position (Paired t-test)
    print("\n>>> Comparison 1: Method 1 (Aug+Proto) vs Method 2 (LDA Only)")
    for pos in test_positions:
        m1_scores = stats_data["Method1 (Aug+Proto)"][pos]
        m2_scores = stats_data["Method2 (LDA Only)"][pos]
        
        m1_mean = np.mean(m1_scores) if m1_scores else 0
        m2_mean = np.mean(m2_scores) if m2_scores else 0
        
        print(f"\nPosition: {pos}")
        print(f"  Method 1 Mean: {m1_mean:.4f}")
        print(f"  Method 2 Mean: {m2_mean:.4f}")

        if len(m1_scores) > 1:
            t_stat, p_val = ttest_rel(m1_scores, m2_scores)
            sig = "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"  Paired t-test: p-value = {p_val:.4e} ({sig})")
        else:
            print("  Not enough samples for t-test.")

    # 2. Compare Positions within Method (Tukey HSD)
    print("\n>>> Comparison 2: Between Electrode Positions (Tukey HSD)")
    for method_name in stats_data.keys():
        print(f"\n[{method_name}]")
        all_vals = []
        all_groups = []
        
        for pos in test_positions:
            scores = stats_data[method_name][pos]
            all_vals.extend(scores)
            all_groups.extend([pos] * len(scores))
            
        if all_vals:
            try:
                tukey = pairwise_tukeyhsd(endog=all_vals, groups=all_groups, alpha=0.05)
                print(tukey)
            except Exception as e:
                print(f"Error in Tukey HSD: {e}")

if __name__ == "__main__":
    main()