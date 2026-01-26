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
# NCC
# -----------------------
import numpy as np
import cv2

def local_ncc(img1, img2, win_size=11, sigma=1.5, C=1e-5):
    """
    Locally Regularized NCC (LR-NCC) を計算する関数
    
    Parameters:
    -----------
    img1, img2 : np.ndarray
        比較する2つの画像（グレースケール, float型推奨）
    win_size : int
        局所領域のウィンドウサイズ（奇数推奨, 例: 11）
    sigma : float
        ガウシアンカーネルの標準偏差
    C : float
        正則化定数（ゼロ除算防止および無信号領域の安定化用）
        入力画像のダイナミックレンジに合わせて調整が必要。
        画像が0-1に正規化されている場合、1e-5 ~ 1e-3程度が目安。

    Returns:
    --------
    score : float
        画像全体の平均スコア (-1.0 ~ 1.0)
    map : np.ndarray
        局所スコアのマップ（どの場所が一致しているかの可視化用）
    """
    
    # 画像がfloat型でない場合は変換
    if img1.dtype != np.float64 and img1.dtype != np.float32:
        img1 = img1.astype(np.float64)
    if img2.dtype != np.float64 and img2.dtype != np.float32:
        img2 = img2.astype(np.float64)

    # 1. 局所平均 (mu) の計算
    mu1 = cv2.GaussianBlur(img1, (win_size, win_size), sigma)
    mu2 = cv2.GaussianBlur(img2, (win_size, win_size), sigma)

    # 2. 二乗および積の局所平均の計算
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    # 3. 局所分散 (sigma^2) と共分散 (sigma_12) の計算
    # E[x^2] - (E[x])^2 の公式を利用
    sigma1_sq = cv2.GaussianBlur(img1 * img1, (win_size, win_size), sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (win_size, win_size), sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (win_size, win_size), sigma) - mu1_mu2

    # 計算誤差で分散がわずかに負になるのを防ぐ
    sigma1_sq = np.maximum(sigma1_sq, 0)
    sigma2_sq = np.maximum(sigma2_sq, 0)

    # 4. LR-NCCの計算 (SSIMの構造項に相当)
    # 分母: std1 * std2 + C
    # 分子: covariance + C
    
    # 標準偏差を取得
    std1 = np.sqrt(sigma1_sq)
    std2 = np.sqrt(sigma2_sq)
    
    numerator = sigma12 + C
    denominator = (std1 * std2) + C

    # 局所スコアのマップ
    lr_ncc_map = numerator / denominator

    # 画像全体の平均値をスコアとする
    score = np.mean(lr_ncc_map)

    return score#, lr_ncc_map

# -----------------------
# parameters
# -----------------------
win_size = 7
sigma=1.5
C=1e-5

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
            corr = local_ncc(stack[i].astype(np.float32), template.astype(np.float32), win_size=win_size, sigma=sigma, C=C)
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
        corr = local_ncc(np.pad(T_ref, ((1,1),(1,1))).astype(np.float32), np.pad(templates[s][g], ((1,1),(1,1))).astype(np.float32), win_size=win_size, sigma=sigma, C=C)
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