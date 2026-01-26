import numpy as np
import glob
import re
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

# -----------------------
# Settings
# -----------------------
DATA_DIR = "output/nojima/EMG_map_array/wl"
SESSIONS = ["original", "original2", "downleft5mm", "downleft10mm", "clockwise"]
GESTURES = range(1, 8)
EXPECTED_SECS = set(range(1, 16))

# -----------------------
# 変換関数
# -----------------------
def warp_emg_8x8(emg, tx, ty, theta, sx, sy, shear):
    H, W = emg.shape

    # === 座標定義 ===
    x = np.arange(H)
    y = np.arange(W)
    interp = RectBivariateSpline(y, x, emg, kx=3, ky=3)

    # === 中心座標 ===
    cx = (H - 1) / 2
    cy = (W - 1) / 2

    # === 各変換行列 ===
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    S = np.array([[sx, 0],
                  [0, sy]])
    Sh = np.array([[1, shear],
                   [0, 1]])

    A = R @ Sh @ S

    # === 同次座標変換行列 ===
    T_center = np.array([[1, 0, -cx],
                          [0, 1, -cy],
                          [0, 0, 1]])

    T_center_inv = np.array([[1, 0, cx],
                              [0, 1, cy],
                              [0, 0, 1]])

    T_shift = np.array([[1, 0, tx],
                        [0, 1, ty],
                        [0, 0, 1]])

    M_affine = np.eye(3)
    M_affine[:2, :2] = A

    # === 合成変換（中心回転） ===
    M = T_shift @ T_center_inv @ M_affine @ T_center
    M_inv = np.linalg.inv(M)

    # === グリッド生成 ===
    grid_x, grid_y = np.meshgrid(x, y)
    ones = np.ones_like(grid_x)

    coords = np.stack([grid_x, grid_y, ones], axis=-1)
    coords = coords.reshape(-1, 3).T

    src = M_inv @ coords
    src_x, src_y = src[0], src[1]

    emg_new = interp.ev(src_y, src_x).reshape(H, W)

    # 6×6 サブセット抽出
    return emg_new[1:7,1:7]

# -----------------------
# Helpers
# -----------------------
sec_re = re.compile(r"_sec(\d+)\.npy$")

def extract_sec_num(filepath: str) -> int:
    m = sec_re.search(filepath)
    if not m:
        raise ValueError(f"Could not parse sec number from: {filepath}")
    return int(m.group(1))

def pearson_corr(map_a: np.ndarray, map_b: np.ndarray) -> float:
    a = map_a.reshape(-1)
    b = map_b.reshape(-1)
    return float(np.corrcoef(a, b)[0, 1])

def load_gesture_stack(prefix: str, gesture: int) -> np.ndarray:
    """Load all sec*.npy for one (session prefix, gesture). Returns (N, H, W)."""
    pattern = f"{DATA_DIR}/{prefix}_gesture{gesture}_sec*.npy"
    files = glob.glob(pattern)
    if len(files) == 0:
        raise FileNotFoundError(f"No files found: {pattern}")

    files = sorted(files, key=extract_sec_num)
    secs_found = [extract_sec_num(f) for f in files]

    missing = EXPECTED_SECS - set(secs_found)
    extra = set(secs_found) - EXPECTED_SECS
    if missing:
        print(f"[WARN] {prefix} gesture{gesture}: missing secs {sorted(missing)}")
    if extra:
        print(f"[WARN] {prefix} gesture{gesture}: extra secs {sorted(extra)}")

    data_list = [np.load(f) for f in files]
    shapes = {arr.shape for arr in data_list}
    if len(shapes) != 1:
        raise ValueError(f"Inconsistent shapes for {prefix} gesture{gesture}: {shapes}")

    return np.stack(data_list, axis=0)

# -----------------------
# Build templates per session and gesture
# -----------------------
templates = {s: {} for s in SESSIONS}

for s in SESSIONS:
    for g in GESTURES:
        X = load_gesture_stack(s, g)        # (secs, 8, 8)
        if s == 'original':
            dx = 0
            dy = 0
            theta = np.radians(0)
            tmp_X = warp_emg_8x8(X.mean(axis=0),dx,dy,theta,1,1,0)
        elif s == 'original2':
            dx = 0
            dy = 0
            theta = np.radians(0)
            tmp_X = warp_emg_8x8(X.mean(axis=0),dx,dy,theta,1,1,0)
        elif s == 'downleft5mm':
            dx = -0.5
            dy = -0.5
            theta = np.radians(0)
            tmp_X = warp_emg_8x8(X.mean(axis=0),dx,dy,theta,1,1,0)
        elif s == 'downleft10mm':
            dx = -1
            dy = -1
            theta = np.radians(0)
            tmp_X = warp_emg_8x8(X.mean(axis=0),dx,dy,theta,1,1,0)
        elif s == 'clockwise':
            dx = 0
            dy = 0
            theta = np.radians(10)
            tmp_X = warp_emg_8x8(X.mean(axis=0),dx,dy,theta,1,1,0)
        templates[s][g] = tmp_X    # (8, 8)

# -----------------------
# 集計: 1つの箱ひげ図用のデータ作成
# -----------------------
ref_session = "original"
compare_sessions = [s for s in SESSIONS if s != ref_session]

# 各ジェスチャーごとに、全比較対象セッションの相関係数をリストに格納する
data_to_plot = [] 
inter_correlations = []

for g in GESTURES:
    g_corrs = []
    T_ref = templates[ref_session][g]
    for s in compare_sessions:
        corr = pearson_corr(T_ref, templates[s][g])
        g_corrs.append(corr)
        inter_correlations.append(corr)
    data_to_plot.append(g_corrs)

# -----------------------
# Plot: 箱ひげ図
# -----------------------
plt.figure(figsize=(10, 6))

# # 箱ひげ図の描画
# # labels引数にジェスチャー名を指定
# plt.boxplot(data_to_plot, labels=[f"G{g}" for g in GESTURES], patch_artist=True,
#             boxprops=dict(facecolor='lightblue', color='blue'),
#             medianprops=dict(color='red'))
plt.boxplot(inter_correlations, labels=['inter session'], patch_artist=True, 
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red'))

plt.ylim(-1.0, 1.0)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8) # 0の線
plt.xlabel("Gesture")
plt.ylabel("Pearson Correlation Coefficient")
plt.title(f"Cross-session Stability relative to '{ref_session}'\n(Aggregated over: {', '.join(compare_sessions)})")
plt.grid(axis='y', linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()