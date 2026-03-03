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
from scipy.optimize import minimize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import cv2
import warnings
import time

# 不要な警告を無視
warnings.filterwarnings('ignore')
np.warnings = warnings

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
# 2. Features & Warping Logic
# ==========================================

def segment_time_series(emg_6x6: np.ndarray, window: int, hop: int) -> np.ndarray:
    n, _, _ = emg_6x6.shape
    idx_starts = np.arange(0, max(1, n - window + 1), hop)
    segs = np.stack([emg_6x6[s:s+window] for s in idx_starts if s + window <= n], axis=0)
    return segs

def wl_feat(x):
    return np.sum(np.abs(np.diff(x, axis=0)), axis=0)

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
    coords = np.stack([grid_x, grid_y, np.ones_like(grid_x)], axis=-1).reshape(-1, 3).T
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
# 3. Main Processing Logic
# ==========================================

def main():
    subjects = ['nojima', 'takeuchi2', 'yamamoto', 'stefan']
    test_positions = ["original2", "downleft5mm", "downleft10mm", "clockwise"]
    
    # 統計用データ蓄積 {method: {position: [accuracy_list]}}
    stats_data = {
        "Method1 (Align+LDA)": {pos: [] for pos in test_positions},
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

        # --- B. Train Phase & Alignment Map Creation ---
        window = 200
        hop    = 50
        window = int(window * (fs/1000))
        hop = int(hop * (fs/1000))
        ch_size = 8
        
        # Feature Extraction for Training
        X_train = None
        y_train = None
        X_train_map = [] # For alignment

        for i, (emg_train_8x8, label) in enumerate(zip(emg_list_train, y_list_train)):
            tmp_X = segment_time_series(emg_train_8x8, window=window, hop=hop)

            # Normalization
            mean = np.mean(tmp_X.reshape(-1,ch_size,ch_size), axis=0)
            std = np.std(tmp_X.reshape(-1,ch_size,ch_size), axis=0) + 1e-8
            tmp_X = (tmp_X - mean.reshape(1, 1, ch_size, ch_size)) / std.reshape(1, 1, ch_size, ch_size)

            # Feature Extraction (WL)
            wl = [wl_feat(x) for x in tmp_X]
            tmp_X_feat = np.array(wl) 
            
            n_windows = len(tmp_X_feat)
            tmp_y = [int(label)-1 for _ in range(n_windows)]

            if i == 0:
                X_train = tmp_X_feat
                y_train = tmp_y
            else:
                X_train = np.vstack([X_train, tmp_X_feat])
                y_train = np.hstack([y_train, tmp_y])
            
            X_train_map.append(tmp_X_feat)

        # Train LDA
        X_train_flat = X_train.reshape(-1, 64)
        clf = LinearDiscriminantAnalysis()
        clf.fit(X_train_flat, y_train)
        
        # Calculate Training Alignment Maps (Average every ~1 sec)
        X_train_map_flat = np.vstack(X_train_map)
        y_train_flat = np.array(y_train)
        gestures = np.unique(y_train_flat) + 1
        train_trial_maps = []
        
        for gesture in gestures:
            trial_list = []
            z_list = []
            iterator = X_train_map_flat[y_train_flat==gesture-1]
            i = 0
            for tmp_val in iterator:
                z_list.append(tmp_val)
                i += 1
                if i == 19: 
                    trial_list.append(np.mean(np.array(z_list), axis=0))
                    z_list = []
                    i = 0
            train_trial_maps.append(trial_list)

        # --- C. Test Phase ---
        for electrode_place in test_positions:
            # print(f'  Processing: {electrode_place}')
            emg_list_test = []
            y_list_test   = []
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
            
            if not emg_list_test: continue

            # Create Test Feature Maps (Pre-alignment)
            X_test_map = []
            y_test_map = []
            for i, (emg_test_8x8, label) in enumerate(zip(emg_list_test, y_list_test)):
                tmp_X = segment_time_series(emg_test_8x8, window=window, hop=hop)
                mean = np.mean(tmp_X.reshape(-1,ch_size,ch_size), axis=0)
                std = np.std(tmp_X.reshape(-1,ch_size,ch_size), axis=0) + 1e-8
                tmp_X = (tmp_X - mean.reshape(1, 1, ch_size, ch_size)) / std.reshape(1, 1, ch_size, ch_size)
                
                wl = [wl_feat(x) for x in tmp_X]
                tmp_X_feat = np.array(wl)
                n_windows = len(tmp_X_feat)
                
                X_test_map.append(tmp_X_feat)
                y_test_map.extend([int(label)-1] * n_windows)

            X_test_map_flat = np.vstack(X_test_map)
            y_test_map_flat = np.array(y_test_map)

            test_trial_maps = []
            for gesture in gestures:
                trial_list = []
                z_list = []
                iterator = X_test_map_flat[y_test_map_flat==gesture-1]
                i = 0
                for tmp_val in iterator:
                    z_list.append(tmp_val)
                    i += 1
                    if i == 19:
                        trial_list.append(np.mean(np.array(z_list), axis=0))
                        z_list = []
                        i = 0
                test_trial_maps.append(trial_list)

            # D. Loop per Trial (Leave-One-Out for alignment optimization)
            acc_m1_list = []
            acc_m2_list = []

            for gesture in range(1, 8):
                for trial in range(1, 6): # 5 trials per gesture
                    try:
                        # 1. Alignment Optimization (For Method 1)
                        ref_idx = min(len(train_trial_maps[gesture-1])-1, trial*3-1)
                        test_idx = min(len(test_trial_maps[gesture-1])-1, trial*3-1)
                        
                        scalar = MinMaxScaler()
                        img_ref = (scalar.fit_transform(train_trial_maps[gesture-1][ref_idx].reshape(-1,1)).reshape(8,8)*255).astype(np.float32)
                        img_test = (scalar.fit_transform(test_trial_maps[gesture-1][test_idx].reshape(-1,1)).reshape(8,8)*255).astype(np.float32)

                        init = [1, 0, 0, 1, 0, 0]
                        res = minimize(objective_ncc, x0=init, args=(img_test, img_ref), method="Powell")
                        a, b, c, d, tx, ty = res.x

                        theta_rad = np.arctan2(c, a)
                        sx = np.sqrt(a**2 + c**2)
                        sy = np.sqrt(b**2 + d**2)
                        shear = (a*b + c*d) / (sx*sy)
                        
                        p_tx, p_ty = -tx, -ty
                        
                        # 2. Prepare Test Set (Exclude the trial used for optimization)
                        X_test_m1_list = [] # Warped
                        X_test_m2_list = [] # Raw
                        y_test_list = []

                        for i, (emg_test_8x8, label) in enumerate(zip(emg_list_test, y_list_test)):
                            current_trial_num = (i % 5) + 1
                            if current_trial_num == trial: continue 
                            
                            # Method 1: Warp -> Segment -> Normalize -> Feature
                            tmp_X_warp = warp_batch_images(emg_test_8x8, p_tx, p_ty, theta_rad, sx, sy, shear)
                            tmp_X_warp = segment_time_series(tmp_X_warp, window=window, hop=hop)
                            mean_w = np.mean(tmp_X_warp.reshape(-1,8,8), axis=0)
                            std_w = np.std(tmp_X_warp.reshape(-1,8,8), axis=0) + 1e-8
                            tmp_X_warp = (tmp_X_warp - mean_w) / std_w
                            wl_warp = np.array([wl_feat(x) for x in tmp_X_warp])
                            X_test_m1_list.append(wl_warp)

                            # Method 2: Segment -> Normalize -> Feature (No Warp)
                            tmp_X_raw = segment_time_series(emg_test_8x8, window=window, hop=hop)
                            mean_r = np.mean(tmp_X_raw.reshape(-1,8,8), axis=0)
                            std_r = np.std(tmp_X_raw.reshape(-1,8,8), axis=0) + 1e-8
                            tmp_X_raw = (tmp_X_raw - mean_r) / std_r
                            wl_raw = np.array([wl_feat(x) for x in tmp_X_raw])
                            X_test_m2_list.append(wl_raw)

                            y_test_list.extend([int(label)-1] * len(wl_raw))

                        if not X_test_m1_list: continue

                        X_test_m1 = np.vstack(X_test_m1_list).reshape(-1, 64)
                        X_test_m2 = np.vstack(X_test_m2_list).reshape(-1, 64)
                        y_test_final = np.array(y_test_list)

                        # Inference
                        score_m1 = clf.score(X_test_m1, y_test_final)
                        score_m2 = clf.score(X_test_m2, y_test_final)

                        acc_m1_list.append(score_m1)
                        acc_m2_list.append(score_m2)

                    except Exception as e:
                        pass
            
            # Store per subject/position averages
            if acc_m1_list:
                stats_data["Method1 (Align+LDA)"][electrode_place].extend(acc_m1_list)
                stats_data["Method2 (LDA Only)"][electrode_place].extend(acc_m2_list)
                print(f"    {electrode_place}: M1={np.mean(acc_m1_list):.4f}, M2={np.mean(acc_m2_list):.4f}")

    # ==========================================
    # 4. Statistical Analysis
    # ==========================================
    from scipy.stats import ttest_rel

    print("\n" + "#"*60)
    print("FINAL STATISTICAL RESULTS (All Subjects Aggregated)")
    print("#"*60)
    
    # 1. Method Comparison per Position
    print("\n>>> Comparison 1: Method 1 (Align+LDA) vs Method 2 (LDA Only)")
    for pos in test_positions:
        acc1 = stats_data["Method1 (Align+LDA)"][pos]
        acc2 = stats_data["Method2 (LDA Only)"][pos]
        
        if len(acc1) > 1:
            t_stat, p_val = ttest_rel(acc1, acc2)
            sig = "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"Position: {pos:12s} | M1: {np.mean(acc1):.4f}, M2: {np.mean(acc2):.4f} | p-value: {p_val:.4e} ({sig})")
        else:
            print(f"Position: {pos:12s} | Not enough data")

    # 2. Position Comparison within Method
    print("\n>>> Comparison 2: Between Electrode Positions (Tukey HSD)")
    
    for method in stats_data.keys():
        print(f"\n[{method}]")
        all_scores = []
        all_groups = []
        
        for pos in test_positions:
            scores = stats_data[method][pos]
            all_scores.extend(scores)
            all_groups.extend([pos] * len(scores))
            
        if all_scores:
            try:
                tukey = pairwise_tukeyhsd(endog=all_scores, groups=all_groups, alpha=0.05)
                print(tukey)
            except Exception as e:
                print(f"Error in Tukey HSD: {e}")

if __name__ == "__main__":
    main()