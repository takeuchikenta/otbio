import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.ndimage import affine_transform
import cv2

# ==========================================
# 1. 設定エリア：ここで解析対象を指定してください
# ==========================================

BASE_DIR = "output"

# 基準とする画像（Ref）の電極名
REF_ELECTRODE_NAME = "original"

# 解析対象とする画像（Test）の電極名 ★ここで指定★
# 例: "original2", "E1", "E2" など
TARGET_ELECTRODE_NAME = "clockwise"

# ==========================================
# 2. 関数定義
# ==========================================

def ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return np.sum(a * b) / (
        np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b**2)) + 1e-8
    )

def affine_transform(img, params):
    a, b, c, d, tx, ty = params
    M = np.array([[a, b, tx],
                  [c, d, ty]], dtype=np.float32)
    h, w = img.shape
    return cv2.warpAffine(img, M, (w, h))

def objective_affine(params, ref, mov):
    warped = affine_transform(mov, params)
    return -ncc(ref, warped)

# ==========================================
# 3. データの読み込み
# ==========================================

results = []
search_pattern = os.path.join(BASE_DIR, "*", "EMG_map_array", "wl", "*.npy")
file_paths = glob.glob(search_pattern)

data_groups = {}
filename_pattern = re.compile(r'(.+?)_gesture(\d+)_sec(\d+)\.npy')

print(f"Found {len(file_paths)} files. Grouping data...")

for path in file_paths:
    filename = os.path.basename(path)
    subject = path.split(os.sep)[-4]
    
    match = filename_pattern.match(filename)
    if match:
        elec_pos = match.group(1)
        gesture_num = match.group(2)
        sec_num = int(match.group(3))
        
        # 必要な電極（Ref または Target）以外はメモリ節約のため読み込まない
        if elec_pos not in [REF_ELECTRODE_NAME, TARGET_ELECTRODE_NAME]:
            continue

        key = (subject, gesture_num)
        if key not in data_groups:
            data_groups[key] = {}
        if elec_pos not in data_groups[key]:
            data_groups[key][elec_pos] = {}
            
        data_groups[key][elec_pos][sec_num] = path

# ==========================================
# 4. 特定電極ペアでの総当たり解析
# ==========================================

print(f"Processing comparison: {REF_ELECTRODE_NAME} (Ref) vs {TARGET_ELECTRODE_NAME} (Test)...")

for (subject, gesture), electrodes_data in data_groups.items():
    
    # RefとTargetの両方が揃っている場合のみ解析
    if (REF_ELECTRODE_NAME in electrodes_data) and (TARGET_ELECTRODE_NAME in electrodes_data):
        
        ref_files_dict = electrodes_data[REF_ELECTRODE_NAME]
        target_files_dict = electrodes_data[TARGET_ELECTRODE_NAME]
        
        # --- 総当たりループ (15秒 x 15秒) ---
        for ref_sec, ref_path in ref_files_dict.items():
            for test_sec, test_path in target_files_dict.items():
                
                try:
                    img_ref = np.load(ref_path)
                    img_test = np.load(test_path)
                    
                    # 初期値 [a, b, c, d, tx, ty]
                    init = [1, 0, 0, 1, 0, 0]

                    res = minimize(
                        objective_affine,
                        x0=init,
                        args=(img_test, img_ref),
                        method="Powell"
                    )

                    a, b, c, d, tx, ty = res.x
                    theta_deg = np.degrees(np.arctan2(c, a))
                    
                    results.append({
                        "Subject": subject,
                        "Gesture": gesture,
                        "Ref_Sec": ref_sec,
                        "Target_Sec": test_sec,
                        "x": tx*10,
                        "y": ty*10,
                        "theta": theta_deg
                    })
                    
                except Exception as e:
                    print(f"Error: {subject} G{gesture} ({test_sec}s vs {ref_sec}s): {e}")

# ==========================================
# 5. 可視化
# ==========================================

if results:
    df_results = pd.DataFrame(results)
    
    print("\n--- Processing Complete ---")
    print(f"Total combinations processed: {len(df_results)}")
    
    df_melted = df_results.melt(
        id_vars=["Subject", "Gesture", "Ref_Sec", "Target_Sec"],
        value_vars=["x", "y", "theta"],
        var_name="Parameter",
        value_name="Value"
    )

    plt.figure(figsize=(10, 6))
    
    sns.boxplot(data=df_melted, x="Parameter", y="Value", showfliers=False)
    
    plt.title(f"Estimated Displacement: '{REF_ELECTRODE_NAME}' vs '{TARGET_ELECTRODE_NAME}'", fontsize=18)
    plt.xlabel("Parameter (x, y, theta)", fontsize=18)
    plt.ylabel("Estimated Value", fontsize=18)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim([-40, 40])
    #labelsizeで軸の数字の文字サイズ変更
    plt.tick_params(labelsize=18)
    plt.tight_layout()
    plt.show()
    
else:
    print(f"No matching data found for pair: {REF_ELECTRODE_NAME} and {TARGET_ELECTRODE_NAME}")