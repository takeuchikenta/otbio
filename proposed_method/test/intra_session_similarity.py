import numpy as np
import glob
import matplotlib.pyplot as plt
import re
from pathlib import Path

# -----------------------
# Settings
# -----------------------
DATA_DIR = "output/nojima/EMG_map_array/wl"      # change if needed
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

# -----------------------
# Load data for all gestures
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
            corr = pearson_corr(stack[i], template)
            intra_correlations.append(corr)


# -----------------------
# Build templates (mean over seconds)
# -----------------------
# templates = {g: gestures[g].mean(axis=0) for g in gestures}

# -----------------------
# Compute sec -> template correlations per gesture
# -----------------------
# box_data = []
# labels = []

# for g in GESTURES:
#     sims = [pearson_corr(gestures[g][i], templates[g]) for i in range(gestures[g].shape[0])]
#     box_data.append(sims)
#     labels.append(f"G{g}")

# -----------------------
# Plot box plot
# -----------------------
plt.figure(figsize=(9, 5))
# plt.boxplot(box_data, labels=labels)
# plt.xlabel("Gesture")
# plt.ylabel("Correlation (sec → template)")
# plt.title(f"Intra-gesture spatial stability ({PREFIX} session, sec1–sec15)")
# plt.tight_layout()
# plt.show()
plt.boxplot(intra_correlations, labels=['intra session'], patch_artist=True, 
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red'))
plt.ylim(-1.0, 1.0)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8) # 0の線
plt.xlabel("Gesture")
plt.ylabel("Pearson Correlation Coefficient")
# plt.title(f"Cross-session Stability relative to '{ref_session}'\n(Aggregated over: {', '.join(compare_sessions)})")
plt.grid(axis='y', linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()