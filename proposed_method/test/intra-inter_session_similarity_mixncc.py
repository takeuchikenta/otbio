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
def ncc(a, b):
    if np.std(a) == 0 or np.std(b) == 0: return 0.0
    a_mean = a - np.mean(a)
    b_mean = b - np.mean(b)
    num = np.sum(a_mean * b_mean)
    den = np.sqrt(np.sum(a_mean**2)) * np.sqrt(np.sum(b_mean**2)) + 1e-8
    return float(num / den)

def compute_gradient_magnitude(image):
    dx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(dx, dy)

def soft_thresholding(img, threshold=0.35, steepness=10):
    """
    シグモイド関数を用いて、微分可能な形で疑似的に二値化する。
    img: 入力画像 (正規化済みを想定)
    threshold: 閾値 (0.0 - 1.0)
    steepness: 変化の急峻さ (大きいほど硬い二値化に近づくが、最適化は不安定になる)
    """
    # 画像を0-1に正規化 (MinMaxではなく、想定最大値で割るのが安全だが今回は簡易実装)
    if img.max() > 0:
        img_norm = img / img.max()
    else:
        img_norm = img

    # シグモイド関数: 1 / (1 + exp(-k * (x - t)))
    # これにより、閾値付近で滑らかに0から1へ変化する
    return 1.0 / (1.0 + steepness*np.exp(- (img_norm - threshold))) * 255

def mix_ncc(ref_img, mov_img, ref_grad_cache=None, alpha=0.1, beta=0.45, threshold=0.35, steepness=10):
    """
    目的関数: 輝度NCC × 勾配NCC
    """
    # 2. 輝度NCC
    score_intensity = ncc(ref_img, mov_img)
    
    # 3. 勾配計算 (refの勾配はキャッシュ可能)
    if ref_grad_cache is None:
        ref_grad = compute_gradient_magnitude(ref_img)
    else:
        ref_grad = ref_grad_cache
        
    # ワープ後の画像から勾配を計算
    warped_grad = compute_gradient_magnitude(mov_img)
    
    # 4. 勾配NCC
    score_gradient = ncc(ref_grad, warped_grad)

    # 二値化
    # ref_thresholded = thresholding(ref_img)
    # warped_thresholded = thresholding(warped_mov)
    # score_thresholded = ncc(ref_thresholded, warped_thresholded)
    ref_thresholded = soft_thresholding(ref_img, threshold=threshold, steepness=steepness)
    warped_thresholded = soft_thresholding(mov_img, threshold=threshold, steepness=steepness)
    score_thresholded = ncc(ref_thresholded, warped_thresholded)

    return float((1 - alpha - beta) * score_intensity + alpha * score_gradient + beta * score_thresholded)
# -----------------------
# parameters
# -----------------------
alpha=0.0
beta=1.0
threshold=0.35
steepness=10

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
            corr = mix_ncc(stack[i].astype(np.float32), template.astype(np.float32), alpha=alpha, beta=beta, threshold=threshold, steepness=steepness)
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
        corr = mix_ncc(T_ref.astype(np.float32), templates[s][g].astype(np.float32), alpha=alpha, beta=beta, threshold=threshold, steepness=steepness)
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