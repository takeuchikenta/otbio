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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import ttest_rel
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
    # 実際のパスに合わせて調整してください
    file_name = f"{subject}/{hand}/{filename}/set{trial}/{electrode_place}-g{gesture}-{trial}.csv"
    return file_name

def segment_time_series(emg_6x6: np.ndarray, window: int, hop: int) -> np.ndarray:
    n, _, _ = emg_6x6.shape
    idx_starts = np.arange(0, max(1, n - window + 1), hop)
    segs = np.stack([emg_6x6[s:s+window] for s in idx_starts if s + window <= n], axis=0)
    return segs

# --- Feature Extraction (prototypical.py logic: WL) ---
def wl_feat(x):
    """(Time, 8, 8) -> (8, 8)"""
    return np.sum(np.abs(np.diff(x, axis=0)), axis=0)

def apply_median_filter(feature_map, kernel_size=3):
    return median_filter(feature_map, size=kernel_size, mode='reflect')

# ==========================================
# 2. Warping & Optimization Logic
# ==========================================

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
    coords = np.stack([grid_x, grid_y, np.ones_like(grid_x)], axis=-1).reshape(-1, 3).T
    src = M_inv @ coords
    src_x, src_y = src[0], src[1]

    emg_new = interp.ev(src_y, src_x).reshape(H, W)
    return emg_new

def warp_batch_images(batch_data, tx, ty, theta, sx, sy, shear):
    """Batch processing for warping"""
    batch_data = np.array(batch_data)
    original_shape = batch_data.shape # (N, Time, 8, 8) or (N, 8, 8)
    
    # Flatten to list of images
    if batch_data.ndim == 4:
        flat_data = batch_data.reshape(-1, 8, 8)
    else:
        flat_data = batch_data
        
    processed_list = []
    for img in flat_data:
        warped_img = warp_emg_8x8(img, tx, ty, theta, sx, sy, shear)
        processed_list.append(warped_img)
        
    return np.array(processed_list).reshape(original_shape)

# ==========================================
# 3. Model & Dataset
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

def process_features(emg_list, y_list, fs=2000, window_ms=200, hop_ms=50):
    """
    Common feature extraction: 
    Segment -> WL -> Median Filter -> Flatten
    """
    window = int((window_ms / 1000) * fs)
    hop = int((hop_ms / 1000) * fs)
    X_feat = []
    y_labels = []
    
    # Alignment map用の未Flattenデータも返す
    X_maps = []
    
    for emg, label in zip(emg_list, y_list):
        tmp_X = segment_time_series(emg, window=window, hop=hop)
        if tmp_X.shape[0] == 0: continue
        
        # WL Calculation (Time axis integration)
        # Note: prototypical.py applies median filter after WL
        for i in range(tmp_X.shape[0]):
            win_data = tmp_X[i]
            wl_map = wl_feat(win_data) # (8, 8)
            
            # Median Filter
            filtered_map = apply_median_filter(wl_map, kernel_size=3)
            
            X_maps.append(filtered_map) # For alignment
            X_feat.append(filtered_map.flatten()) # For classifier
            y_labels.append(int(label)-1)
            
    return np.array(X_feat), np.array(X_maps), np.array(y_labels)

# ==========================================
# 4. Main Processing
# ==========================================

def main():
    subjects = ['nojima', 'takeuchi2', 'yamamoto', 'stefan']
    test_positions = ["original2", "downleft5mm", "downleft10mm", "clockwise"]
    
    stats_data = {
        "Method1 (Align+Proto)": {pos: [] for pos in test_positions},
        "Method2 (LDA Only)":    {pos: [] for pos in test_positions}
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

        # --- B. Train Models (Source) ---
        
        # 1. Process Training Features
        X_train_flat, X_train_maps, y_train_labels = process_features(emg_list_train, y_list_train, fs)
        
        # 2. Train ProtoNet (Method 1)
        print("  Training ProtoNet...")
        X_train_t = torch.FloatTensor(X_train_flat).to(device)
        y_train_t = torch.LongTensor(y_train_labels).to(device)
        
        proto_model = ProtoNet(input_dim=64).to(device)
        optimizer = optim.Adam(proto_model.parameters(), lr=0.0005)
        proto_model.train()
        for epoch in range(300):
            embeddings = proto_model(X_train_t)
            prototypes = calculate_prototypes(embeddings, y_train_t, 7)
            dists = euclidean_dist(embeddings, prototypes)
            loss = -torch.log_softmax(-dists, dim=1).gather(1, y_train_t.view(-1, 1)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 3. Train LDA (Method 2)
        print("  Training LDA...")
        lda_model = LinearDiscriminantAnalysis()
        lda_model.fit(X_train_flat, y_train_labels)

        # 4. Prepare Reference Maps for Alignment
        # Average maps every ~19 frames (~1 sec) per gesture
        train_ref_maps = []
        gestures = np.unique(y_train_labels) + 1
        for g in gestures:
            g_idx = (y_train_labels == g-1)
            g_maps = X_train_maps[g_idx]
            
            trial_list = []
            buf = []
            for i, val in enumerate(g_maps):
                buf.append(val)
                if len(buf) == 19:
                    trial_list.append(np.mean(np.array(buf), axis=0))
                    buf = []
            train_ref_maps.append(trial_list)

        # --- C. Test Phase (Loop Positions) ---
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

            # Pre-calculate Test Maps for Alignment Optimization (Pre-Warping)
            # Use raw segmentation to find alignment parameters
            _, X_test_maps_raw, y_test_labels_raw = process_features(emg_list_test, y_list_test, fs)
            
            # Prepare Target Maps for Optimization
            test_target_maps = []
            for g in gestures:
                g_idx = (y_test_labels_raw == g-1)
                g_maps = X_test_maps_raw[g_idx]
                trial_list = []
                buf = []
                for val in g_maps:
                    buf.append(val)
                    if len(buf) == 19:
                        trial_list.append(np.mean(np.array(buf), axis=0))
                        buf = []
                test_target_maps.append(trial_list)

            # Loop 5-fold (One-Shot Split)
            acc_m1_list = []
            acc_m2_list = []

            for g_idx in range(7):
                for t_idx in range(5):
                    try:
                        # --- 1. Alignment Optimization (For Method 1) ---
                        ref_idx = min(len(train_ref_maps[g_idx])-1, t_idx*3)
                        tgt_idx = min(len(test_target_maps[g_idx])-1, t_idx*3)
                        
                        scalar = MinMaxScaler()
                        img_ref = (scalar.fit_transform(train_ref_maps[g_idx][ref_idx].reshape(-1,1)).reshape(8,8)*255).astype(np.float32)
                        img_tgt = (scalar.fit_transform(test_target_maps[g_idx][tgt_idx].reshape(-1,1)).reshape(8,8)*255).astype(np.float32)

                        res = minimize(objective_ncc, x0=[1,0,0,1,0,0], args=(img_tgt, img_ref), method="Powell")
                        a, b, c, d, tx, ty = res.x
                        theta = np.arctan2(c, a)
                        sx = np.sqrt(a**2 + c**2)
                        sy = np.sqrt(b**2 + d**2)
                        shear = (a*b + c*d) / (sx*sy)
                        
                        p_tx, p_ty = -tx, -ty

                        # --- 2. Data Preparation (Query & Support) ---
                        # Method 1: Use Warped Support to Adapt, Warped Query to Test
                        # Method 2: Use Raw Query to Test (Source Model)
                        
                        query_emg_warped = []
                        query_y_warped = []
                        
                        query_emg_raw = []
                        query_y_raw = []
                        
                        support_emg_warped = []
                        support_y_warped = []

                        for i, (raw_emg, label) in enumerate(zip(emg_list_test, y_list_test)):
                            # Assuming standard file order: 5 trials per gesture
                            current_trial = i % 5
                            current_gesture = i // 5
                            
                            # Warp Raw Signal (Method 1)
                            warped_emg = warp_batch_images(raw_emg, p_tx, p_ty, theta, sx, sy, shear)
                            
                            # Support Set (Specific Trial)
                            if current_gesture == g_idx and current_trial == t_idx:
                                support_emg_warped.append(warped_emg)
                                support_y_warped.append(label)
                            
                            # Query Set (Others)
                            elif current_trial != t_idx:
                                query_emg_warped.append(warped_emg)
                                query_y_warped.append(label)
                                
                                query_emg_raw.append(raw_emg)
                                query_y_raw.append(label)

                        if not query_emg_warped: continue

                        # --- Method 1: Align + Proto ---
                        X_supp_p, _, y_supp_p = process_features(support_emg_warped, support_y_warped, fs)
                        X_query_p, _, y_query_p = process_features(query_emg_warped, query_y_warped, fs)
                        
                        if len(X_supp_p) > 0 and len(X_query_p) > 0:
                            with torch.no_grad():
                                supp_t = torch.FloatTensor(X_supp_p).to(device)
                                ysupp_t = torch.LongTensor(y_supp_p).to(device)
                                query_t = torch.FloatTensor(X_query_p).to(device)
                                yquery_t = torch.LongTensor(y_query_p).to(device)
                                
                                supp_emb = proto_model(supp_t)
                                query_emb = proto_model(query_t)
                                
                                new_protos = calculate_prototypes(supp_emb, ysupp_t, 7)
                                dists = euclidean_dist(query_emb, new_protos)
                                preds = torch.argmin(dists, dim=1)
                                
                                score_m1 = (preds == yquery_t).float().mean().item()
                                acc_m1_list.append(score_m1)

                        # --- Method 2: LDA Only ---
                        X_query_l, _, y_query_l = process_features(query_emg_raw, query_y_raw, fs)
                        if len(X_query_l) > 0:
                            pred_lda = lda_model.predict(X_query_l)
                            score_m2 = np.mean(pred_lda == y_query_l)
                            acc_m2_list.append(score_m2)

                    except Exception:
                        pass
            
            # Store Result
            if acc_m1_list:
                stats_data["Method1 (Align+Proto)"][electrode_place].extend(acc_m1_list)
                stats_data["Method2 (LDA Only)"][electrode_place].extend(acc_m2_list)
                print(f"    {electrode_place}: M1={np.mean(acc_m1_list):.4f}, M2={np.mean(acc_m2_list):.4f}")

    # ==========================================
    # 5. Statistical Analysis
    # ==========================================
    print("\n" + "#"*60)
    print("FINAL STATISTICAL RESULTS")
    print("#"*60)

    # 1. Method Comparison per Position (Paired t-test)
    print("\n>>> Comparison 1: Method 1 (Align+Proto) vs Method 2 (LDA Only)")
    for pos in test_positions:
        m1_scores = stats_data["Method1 (Align+Proto)"][pos]
        m2_scores = stats_data["Method2 (LDA Only)"][pos]
        
        if len(m1_scores) > 1:
            t_stat, p_val = ttest_rel(m1_scores, m2_scores)
            sig = "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"Position: {pos:12s} | M1: {np.mean(m1_scores):.4f}, M2: {np.mean(m2_scores):.4f} | p-value: {p_val:.4e} ({sig})")
        else:
            print(f"Position: {pos:12s} | Not enough data")

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