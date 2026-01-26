import numpy as np
import glob
import re
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import cv2

# -----------------------
# Settings
# -----------------------
DATA_DIR = "output/nojima/EMG_map_array/wl"
SESSIONS = ["original", "original2", "downleft5mm", "downleft10mm", "clockwise"]
GESTURES = range(1, 8)
EXPECTED_SECS = set(range(1, 16))

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
# NGC
# -----------------------
from scipy.ndimage import gaussian_filter

def preprocess_wl(img: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    """
    WLマップの前処理: 対数変換で非線形性を緩和し、ガウシアンで高周波ノイズを除去。
    """
    # log(1+x)でゼロ除算回避しつつ対数変換
    img_log = np.log1p(img)
    return gaussian_filter(img_log, sigma=sigma)

def calc_ngc(img1: np.ndarray, img2: np.ndarray, eps: float = 1e-8) -> float:
    """
    正規化勾配相関 (NGC) を計算。
    値は [-1, 1] の範囲。1に近いほど勾配の向きと強度の分布が一致。
    """
    # 1. 勾配の計算 (中央差分)
    # g1_y, g1_x = np.gradient(img1) # 行列サイズが大きい場合
    # 行列サイズが小さいHD-sEMG(例:8x16)ではSobelの方が安定する場合があるが、gradientで十分
    # img1 = np.pad(img1, ((1,1),(1,1)))
    # img2 = np.pad(img2, ((1,1),(1,1)))
    # img1 = cv2.GaussianBlur(img1,(3,3),0)
    # img2 = cv2.GaussianBlur(img2,(3,3),0)

    dy1, dx1 = np.gradient(img1)
    dy2, dx2 = np.gradient(img2)

    # 2. 各点での勾配ベクトルの内積 (Numerator)
    # A . B = |A||B|cos(theta)
    dot_product = (dx1 * dx2) + (dy1 * dy2)

    # 3. 勾配の大きさ (Magnitude)
    mag1_sq = dx1**2 + dy1**2
    mag2_sq = dx2**2 + dy2**2

    # 4. 全体の相関を計算 (Global NGC)
    # 分母: 全エネルギーの平方根の積 (Cauchy-Schwarz)
    numerator = np.sum(dot_product)
    denominator = np.sqrt(np.sum(mag1_sq)) * np.sqrt(np.sum(mag2_sq))

    return numerator / (denominator + eps)
# -----------------------
# parameters
# -----------------------
alpha=0.0
beta=1.0
threshold=0.5

# -----------------------
# intra session similarity
# -----------------------
gestures = {}  # g -> array shape (num_secs, 8, 8)
session_templates = {s: {} for s in SESSIONS}
intra_correlations = []

for s in SESSIONS:
    for g in GESTURES:
        pattern = f"{DATA_DIR}/{s}_gesture{g}_sec*.npy"
        files = glob.glob(pattern)

        if len(files) == 0:
            raise FileNotFoundError(f"No files found for gesture {g} with pattern: {pattern}")

        # Sort by sec number (not lexicographic)
        files = sorted(files, key=extract_sec_num)

        # Optional: sanity check that we have exactly sec1..sec15
        secs_found = [extract_sec_num(f) for f in files]
        missing = EXPECTED_SECS - set(secs_found)
        extra = set(secs_found) - EXPECTED_SECS
        if missing:
            print(f"[WARN] Gesture {g}: missing secs: {sorted(missing)}")
        if extra:
            print(f"[WARN] Gesture {g}: unexpected extra secs: {sorted(extra)}")

        # Load
        data_list = [np.load(f) for f in files]

        # Sanity: shapes consistent
        shapes = {arr.shape for arr in data_list}
        if len(shapes) != 1:
            raise ValueError(f"Gesture {g} has inconsistent shapes: {shapes}")

        gestures[g] = np.stack(data_list, axis=0)
        stack = np.stack(data_list, axis=0)
        template = stack.mean(axis=0)

        # セッション内相関: 各秒数 vs テンプレート
        for i in range(stack.shape[0]):
            # corr = ncc(stack[i], template)
            corr = calc_ngc(stack[i], template)
            intra_correlations.append(corr)

# -----------------------
# inter session similarity
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


ref_session = "original"
compare_sessions = [s for s in SESSIONS if s != ref_session]

# 各ジェスチャーごとに、全比較対象セッションの相関係数をリストに格納する
data_to_plot = [] 
inter_correlations = []

for g in GESTURES:
    T_ref = templates[ref_session][g]
    for s in compare_sessions:
        # corr = ncc(T_ref, templates[s][g])
        corr = calc_ngc(T_ref, templates[s][g])
        inter_correlations.append(corr)

# -----------------------
# Visualization
# -----------------------
plt.figure(figsize=(7, 6))
data_to_plot = [intra_correlations, inter_correlations]
labels = ["Intra-session", "Inter-session"]

plt.boxplot(data_to_plot, labels=labels, patch_artist=True, 
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red'))

plt.ylabel("Pearson Correlation Coefficient", fontsize=16)
# plt.title("EMG Map Stability: Intra-session vs Inter-session")
plt.ylim(0.0, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(fontsize=16) 
plt.yticks(fontsize=16)

plt.tight_layout()
plt.show()

# 数値のサマリーを表示
print(f"Intra-session: mean={np.mean(intra_correlations):.3f}, std={np.std(intra_correlations):.3f}, n={len(intra_correlations)}")
print(f"Inter-session: mean={np.mean(inter_correlations):.3f}, std={np.std(inter_correlations):.3f}, n={len(inter_correlations)}")