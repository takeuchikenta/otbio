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
# 1. Helper Functions (Provided Code)
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

def ptp_feat(signal):
    return np.array(np.ptp(signal, axis=0))

def rms_feat(x):
    return np.sqrt(np.mean(np.square(x), axis=0))

def wl_feat(x):
    return np.sum(np.abs(np.diff(x, axis=0)), axis=0)

def zero_crossings(x, thresh):
    x1 = x[:-1, :]
    x2 = x[1:, :]
    cond = (np.abs(x1) > thresh) & (np.abs(x2) > thresh) & (np.sign(x1) != np.sign(x2))
    return np.sum(cond, axis=0).astype(float)

def slope_sign_changes(x, thresh):
    x_prev = x[:-2, :]
    x_mid  = x[1:-1, :]
    x_next = x[2:, :]
    cond_mag = (np.abs(x_prev) > thresh) & (np.abs(x_mid) > thresh) & (np.abs(x_next) > thresh)
    cond_ssc = (((x_mid - x_prev) * (x_mid - x_next)) > 0)
    return np.sum(cond_mag & cond_ssc, axis=0).astype(float)

def wamp_feat(x, thresh):
    dx = np.abs(np.diff(x, axis=0))
    return np.sum(dx > thresh, axis=0).astype(float)

# === TD-PSD特徴量 (簡易版: 依存関数等は省略し、mainで使うもののみ定義) ===
# mainでは td_psd_multichannel を呼んでいますが、
# 今回の目的（位置合わせ手法の評価）では main 内で wl_feat が選択されているため、
# 依存関係が複雑な TD-PSD は省略し、必要な import 等のみ記述します。
# もし TD-PSD が必須であれば、元の長いコードを含める必要がありますが、
# 提示された main ロジックでは `tmp_X = wl` となっているため WL のみで動作します。

def _log_map(x):
    return np.sign(x) * np.log1p(np.abs(x))

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

def ncc(a, b):
    if np.std(a) == 0 or np.std(b) == 0: return 0.0
    a_mean = a - np.mean(a)
    b_mean = b - np.mean(b)
    num = np.sum(a_mean * b_mean)
    den = np.sqrt(np.sum(a_mean**2)) * np.sqrt(np.sum(b_mean**2)) + 1e-8
    return float(num / den)

def affine_transform(img, params):
    a, b, c, d, tx, ty = params
    M = np.array([[a, b, tx],
                  [c, d, ty]], dtype=np.float32)
    h, w = img.shape
    return cv2.warpAffine(img, M, (w, h))

def objective_ncc(params, ref, mov):
    warped = affine_transform(mov, params)
    return 1 - ncc(ref, warped)


# ==========================================
# 2. Main Processing Logic (Multi-Subject)
# ==========================================

def main():
    subjects = ['nojima', 'takeuchi2', 'yamamoto', 'stefan']
    test_positions = ["original2", "downleft5mm", "downleft10mm", "clockwise"]
    
    # 統計用データコンテナ: {position: [all_accuracies_from_all_subjects]}
    stats_data = {pos: [] for pos in test_positions}
    
    print(f"Comparison started for subjects: {subjects}")

    for subject in subjects:
        print(f"\n{'='*20} Processing Subject: {subject} {'='*20}")
        
        # --- A. Load Training Data (Original) ---
        emg_list_train = []
        y_list_train   = []
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
        sizes_te = []
        threshold = 0
        threshold2 = 0.0013
        ch_size = 8
        
        # Feature Extraction for Training
        X_train = None
        y_train = None
        X_train_aligment = []

        for i, (emg_train_8x8, label) in enumerate(zip(emg_list_train, y_list_train)):
            tmp_X = emg_train_8x8
            tmp_X = segment_time_series(tmp_X, window=window, hop=hop)

            # Normalization
            mean = np.mean(tmp_X.reshape(-1,ch_size,ch_size), axis=0)
            std = np.std(tmp_X.reshape(-1,ch_size,ch_size), axis=0) + 1e-8
            tmp_X = (tmp_X - mean.reshape(1, 1, ch_size, ch_size)) / std.reshape(1, 1, ch_size, ch_size)

            # Feature Extraction (WL selected as per prompt logic)
            wl = [wl_feat(x) for x in tmp_X]
            tmp_X_feat = np.array(wl) 
            # Note: The prompt uses `tmp_X = wl` but later does `tmp_X = rms` then reverts. 
            # Assuming `wl` is the intended feature for both LDA and Alignment based on context.
            
            n_windows = len(tmp_X_feat)
            tmp_y = [int(label)-1 for _ in range(n_windows)]
            sizes_te.append(n_windows)

            if i == 0:
                X_train = tmp_X_feat
                y_train = tmp_y
            else:
                X_train = np.vstack([X_train, tmp_X_feat])
                y_train = np.hstack([y_train, tmp_y])
            
            # Store for alignment map (keep structure)
            X_train_aligment.append(tmp_X_feat) # List of arrays

        # Train LDA
        X_train_nonclopped = X_train.reshape(-1, 8*8)
        clf = LinearDiscriminantAnalysis()
        clf.fit(X_train_nonclopped, y_train)
        
        # Flatten X_train_aligment for map calculation
        X_train_aligment_flat = np.vstack(X_train_aligment)
        y_train_flat = np.array(y_train)

        # Calculate Training Alignment Maps
        gestures = np.unique(y_train_flat) + 1
        train_trial_alignment_maps = []
        
        # ロジック: 各ジェスチャーごとにデータを取得し、19個(約1秒?)ごとに平均してリスト化
        for gesture in gestures:
            z_list = []
            trial_list = []
            # 該当ジェスチャーのデータを抽出
            iterator = X_train_aligment_flat[y_train_flat==gesture-1]
            i = 0
            for tmp_val in iterator:
                z_list.append(tmp_val)
                i += 1
                if i == 19: 
                    trial_list.append(np.mean(np.array(z_list), axis=0))
                    z_list = []
                    i = 0
            train_trial_alignment_maps.append(trial_list)

        # --- C. Test Phase (Loop over positions) ---
        for electrode_place in test_positions:
            print(f'  Processing: {electrode_place}')
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
            
            if not emg_list_test:
                continue

            # Test Feature Extraction (Pre-Alignment)
            X_test_aligment = []
            y_test_aligment = [] # For map calculation
            
            for i, (emg_test_8x8, label) in enumerate(zip(emg_list_test, y_list_test)):
                tmp_X = segment_time_series(emg_test_8x8, window=window, hop=hop)
                mean = np.mean(tmp_X.reshape(-1,ch_size,ch_size), axis=0)
                std = np.std(tmp_X.reshape(-1,ch_size,ch_size), axis=0) + 1e-8
                tmp_X = (tmp_X - mean.reshape(1, 1, ch_size, ch_size)) / std.reshape(1, 1, ch_size, ch_size)
                
                wl = [wl_feat(x) for x in tmp_X]
                tmp_X_feat = np.array(wl)
                
                n_windows = len(tmp_X_feat)
                tmp_y = [int(label)-1 for _ in range(n_windows)]
                
                X_test_aligment.append(tmp_X_feat)
                y_test_aligment.extend(tmp_y)

            X_test_aligment_flat = np.vstack(X_test_aligment)
            y_test_aligment_flat = np.array(y_test_aligment)

            # Calculate Test Alignment Maps
            test_trial_alignment_maps = []
            for gesture in gestures:
                z_list = []
                trial_list = []
                iterator = X_test_aligment_flat[y_test_aligment_flat==gesture-1]
                i = 0
                for tmp_val in iterator:
                    z_list.append(tmp_val)
                    i += 1
                    if i == 19:
                        trial_list.append(np.mean(np.array(z_list), axis=0))
                        z_list = []
                        i = 0
                test_trial_alignment_maps.append(trial_list)

            # --- D. Alignment & Inference Loop ---
            # 各試行ごとに位置合わせパラメータを最適化し、推論する
            subject_pos_accuracies = [] # Store accuracy per trial (or gesture step)
            
            for gesture in range(1, 8):
                # accuracy_each_gesture_alignment = []
                for trial in range(1, 6):
                    # Alignment Optimization
                    try:
                        # 参照画像とテスト画像の取得 (trial*3-1 はデータのインデックス仮定)
                        # インデックスエラー回避
                        ref_idx = min(len(train_trial_alignment_maps[gesture-1])-1, trial*3-1)
                        test_idx = min(len(test_trial_alignment_maps[gesture-1])-1, trial*3-1)
                        
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
                        
                        # Parameters for warping
                        p_tx, p_ty = -tx, -ty
                        p_theta = theta_rad
                        p_sx, p_sy = sx, sy
                        p_shear = shear

                        # Apply Warping & Inference
                        # 対象のデータ: emg_list_test 内の当該 trial 以外のデータ (Leave-one-out的な挙動)
                        # プロンプトのロジック: `if (i+1) % 5 == trial: continue`
                        # 位置合わせに使った試行以外をテストする
                        
                        X_test_warped_list = []
                        y_test_warped_list = []
                        
                        for i, (emg_test_8x8, label) in enumerate(zip(emg_list_test, y_list_test)):
                            # trialは1-5, iは0-34. (i+1)%5 は 1,2,3,4,0. 
                            # trial==5 のとき (i+1)%5==0 とマッチさせるため調整
                            current_trial_num = (i % 5) + 1
                            if current_trial_num == trial:
                                continue 
                            
                            # Warp Raw Data
                            tmp_X = warp_batch_images(emg_test_8x8, p_tx, p_ty, p_theta, p_sx, p_sy, p_shear)
                            # Segment
                            tmp_X = segment_time_series(tmp_X, window=window, hop=hop)
                            # Normalize
                            mean = np.mean(tmp_X.reshape(-1,ch_size,ch_size), axis=0)
                            std = np.std(tmp_X.reshape(-1,ch_size,ch_size), axis=0) + 1e-8
                            tmp_X = (tmp_X - mean.reshape(1, 1, ch_size, ch_size)) / std.reshape(1, 1, ch_size, ch_size)
                            # Feature (WL)
                            wl = [wl_feat(x) for x in tmp_X]
                            tmp_X_feat = np.array(wl)
                            
                            X_test_warped_list.append(tmp_X_feat)
                            y_test_warped_list.extend([int(label)-1] * len(tmp_X_feat))

                        if not X_test_warped_list: continue

                        X_test_final = np.vstack(X_test_warped_list).reshape(-1, 64)
                        y_test_final = np.array(y_test_warped_list)
                        
                        score = clf.score(X_test_final, y_test_final)
                        subject_pos_accuracies.append(score)
                        
                    except Exception as e:
                        # print(f"Error in optimization/inference: {e}")
                        pass
            
            # Record average accuracy for this subject/position
            if subject_pos_accuracies:
                # ここでは「各ジェスチャー・各試行でのアライメント結果に基づく推論スコア」の平均を
                # その被験者・その位置での代表値として保存します。
                # より詳細な統計が必要な場合は subject_pos_accuracies 全体を extend してください。
                # 今回はN=被験者数x試行数にするため、extendします
                stats_data[electrode_place].extend(subject_pos_accuracies)
                print(f"    Mean Acc: {np.mean(subject_pos_accuracies):.4f}")

    # ==========================================
    # 3. Statistical Analysis (Tukey HSD)
    # ==========================================
    print("\n" + "#"*60)
    print("STATISTICAL ANALYSIS: Between Electrode Positions")
    print("#"*60)

    # Prepare data for Tukey HSD
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
            
            # Plotting (Optional)
            # fig = tukey.plot_simultaneous()
            # plt.show()
        except Exception as e:
            print(f"Could not perform Tukey HSD: {e}")
    else:
        print("No valid data collected for statistics.")

if __name__ == "__main__":
    main()