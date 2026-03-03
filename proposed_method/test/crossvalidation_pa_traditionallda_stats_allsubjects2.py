import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import signal
from scipy.signal import butter, sosfiltfilt
from scipy.stats import ttest_rel
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import warnings

# 不要な警告を無視
warnings.filterwarnings('ignore')
np.warnings = warnings

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 0. Common Helper Functions (Raw Data Load)
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

def load_raw_data(subject, electrode_place):
    """指定された条件の生データを読み込み、フィルタリングして返す共通関数"""
    emg_list = []
    y_list = []
    for j in range(7):
        for k in range(5):
            try:
                file_name = file_name_output(subject=subject, electrode_place=electrode_place, gesture=j+1, trial=k+1)
                path = '../../data/highMVC/' + file_name
                df = pd.read_csv(path, encoding='utf-8-sig', sep=';', header=None) 
                time_emg = df.iloc[:, 0].values
                emg_data = df.iloc[:, 1:65].values
                fs = int(1 / np.mean(np.diff(time_emg)))
                emg_data = remove_power_line_harmonics(emg_data, fs=fs)
                filtered_emg = butter_bandpass_filter(emg_data, fs=fs, low_hz=20.0, high_hz=450.0)
                emg_data = filtered_emg.reshape(-1,8,8)
                emg_list.append(emg_data)
                y_list.append(j+1)
            except FileNotFoundError:
                pass
    return emg_list, y_list

def ptp_feat(signal):
    return np.array(np.ptp(signal, axis=0))

def rms_feat(x):
    return np.sqrt(np.mean(np.square(x), axis=0))

def wl_feat(x):
    return np.sum(np.abs(np.diff(x, axis=0)), axis=0)

def arv_feat(x):
    return np.mean(np.abs(x), axis=0)

def zc_feat(signal):
    s = np.sign(signal)
    s_prev = s[:-1]
    s_next = s[1:]
    sign_change = (s_prev * s_next) < 0
    return np.sum(sign_change, axis=0)

def zero_crossings(x, thresh):
    # count sign changes where both samples exceed threshold in magnitude
    x1 = x[:-1, :]
    x2 = x[1:, :]
    cond = (np.abs(x1) > thresh) & (np.abs(x2) > thresh) & (np.sign(x1) != np.sign(x2))
    return np.sum(cond, axis=0).astype(float)

def slope_sign_changes(x, thresh):
    # count SSC where middle sample forms a slope sign change and all magnitudes exceed threshold
    x_prev = x[:-2, :]
    x_mid  = x[1:-1, :]
    x_next = x[2:, :]
    cond_mag = (np.abs(x_prev) > thresh) & (np.abs(x_mid) > thresh) & (np.abs(x_next) > thresh)
    cond_ssc = (((x_mid - x_prev) * (x_mid - x_next)) > 0)
    return np.sum(cond_mag & cond_ssc, axis=0).astype(float)

def wamp_feat(x, thresh):
    """
    Willison Amplitude (WAMP)
    x: (N, C) 窓データ
    thresh: (C,) しきい値（各チャネル）
    return: (C,) 各チャネルのWAMP
    """
    dx = np.abs(np.diff(x, axis=0))      # (N-1, C)
    return np.sum(dx > thresh, axis=0).astype(float)

def extract_feature_map(window_data, feature_name='wl'):
    if feature_name == 'wl':
        feature_flat = np.sum(np.abs(np.diff(window_data, axis=0)), axis=0)
    elif feature_name == 'ptp':
        feature_flat = ptp_feat(window_data)
    elif feature_name == 'rms':
        feature_flat = rms_feat(window_data)
    elif feature_name == 'zc':
        feature_flat = zc_feat(window_data)
    else:
        raise ValueError(f"Unknown feature name: {feature_name}")
    return feature_flat.reshape(8, 8)

# ==========================================
# Method 1: Prototypical Network (Specifics)
# ==========================================

def extract_feature_map_proto(window_data, feature_name='wl'):
    # 元のPrototypical用特徴抽出 (WLのみ)
    if feature_name == 'wl':
        feature_flat = wl_feat(window_data)
    elif feature_name == 'ptp':
        feature_flat = ptp_feat(window_data)
    elif feature_name == 'rms':
        feature_flat = rms_feat(window_data)
    elif feature_name == 'zc':
        feature_flat = zc_feat(window_data)
    else:
        raise ValueError(f"Unknown feature name: {feature_name}")
    return feature_flat.reshape(8, 8)

def prepare_indices_proto(n_trials, n_time, is_support_set=False, shot_trial_idx=0):
    trials_per_gesture = 5
    n_gestures = 7
    indices = []
    for g in range(n_gestures):
        start_trial_idx = g * trials_per_gesture
        target_idx = start_trial_idx + shot_trial_idx
        if is_support_set:
            trial_indices = [target_idx] 
            time_limit = 2000 
        else:
            all_indices = list(range(start_trial_idx, start_trial_idx + trials_per_gesture))
            all_indices.remove(target_idx)
            trial_indices = all_indices
            time_limit = n_time
        indices.append((g, trial_indices, time_limit))
    return indices

def create_dataset_proto(emg_list, y_list, fs=2000, window_ms=200, step_ms=50, mode='full', shot_trial_idx=0, feature_name='wl'):
    """ProtoNet用のデータセット作成（WL特徴量、正規化なし）"""
    data = np.array(emg_list) 
    n_trials, n_time, _, _ = data.shape
    data = data.reshape(n_trials, n_time, -1)
    
    window_samples = int((window_ms / 1000) * fs)
    step_samples = int((step_ms / 1000) * fs)
    
    X_feat = []
    y_labels = []
    
    if mode == 'full':
        target_config = [(g, range(g*5, (g+1)*5), n_time) for g in range(7)]
    elif mode == 'support':
        target_config = prepare_indices_proto(5, n_time, is_support_set=True, shot_trial_idx=shot_trial_idx)
    elif mode == 'query':
        target_config = prepare_indices_proto(5, n_time, is_support_set=False, shot_trial_idx=shot_trial_idx)
    
    for gesture_label, trial_indices, time_limit in target_config:
        for t_idx in trial_indices:
            trial_data = data[t_idx]
            limit = min(time_limit, trial_data.shape[0])
            valid_data = trial_data[:limit, :]
            
            for start in range(0, limit - window_samples + 1, step_samples):
                end = start + window_samples
                window = valid_data[start:end, :]
                # 特徴抽出 (Proto用: WL)
                feat_map = extract_feature_map_proto(window, feature_name=feature_name)
                X_feat.append(feat_map.flatten())
                y_labels.append(gesture_label)
            
    X_feat = np.array(X_feat)
    y_labels = np.array(y_labels)

    # チャネル毎のZ-score正規化
    if len(X_feat) > 0:
        scaler = StandardScaler()
        X_feat = scaler.fit_transform(X_feat)
                
    return X_feat, y_labels

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

def calculate_prototypes(embeddings, labels, n_classes):
    prototypes = []
    for c in range(n_classes):
        mask = (labels == c)
        if mask.sum() > 0:
            prototypes.append(embeddings[mask].mean(0))
        else:
            prototypes.append(torch.zeros(embeddings.size(1)).to(embeddings.device))
    return torch.stack(prototypes)

def euclidean_dist(x, prototypes):
    n = x.size(0)
    m = prototypes.size(0)
    d = x.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    prototypes = prototypes.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - prototypes, 2).sum(2)

# ==========================================
# Method 2: LDA (Specifics from provided code)
# ==========================================

def segment_time_series_lda(emg_8x8: np.ndarray, window: int, hop: int) -> np.ndarray:
    """(n_samples, 8, 8) -> (n_windows, window, 8, 8)"""
    n, _, _ = emg_8x8.shape
    idx_starts = np.arange(0, max(1, n - window + 1), hop)
    segs = np.stack([emg_8x8[s:s+window] for s in idx_starts if s + window <= n], axis=0)
    return segs

def create_dataset_lda(emg_list, y_list, fs=2000, window_ms=200, hop_ms=50, is_test_split=False, shot_trial_idx=0, feature_name='rms'):
    """
    LDA用のデータセット作成（提示された特定の正規化＋RMS）
    is_test_split=True の場合、5-fold one-shot の Query 相当のデータのみを抽出する
    """
    window_samples = int((window_ms / 1000) * fs)
    hop_samples = int((hop_ms / 1000) * fs)
    ch_size = 8
    
    X_features = []
    y_labels = []
    
    # イテレーションロジック
    trials_per_gesture = 5
    
    # フィルタリングすべきインデックス（テスト時のOne-Shot Split用）
    target_indices = []
    if is_test_split:
        for g in range(7):
            start = g * trials_per_gesture
            # shot_trial_idx は学習に使わない（Adaptationなしだが、比較のため評価セットを合わせる）
            # Queryセット相当（shot_trial_idx以外）を使用
            for t in range(trials_per_gesture):
                if t != shot_trial_idx:
                    target_indices.append(start + t)
    else:
        # 学習時は全データ使用
        target_indices = range(len(emg_list))

    for idx in target_indices:
        emg_data = emg_list[idx]
        label = y_list[idx]
        
        # 1. 窓分割
        tmp_X = segment_time_series_lda(emg_data, window=window_samples, hop=hop_samples)
        
        if tmp_X.shape[0] == 0: continue

        # 2. 正規化 (提示コードのロジック: バッチ全体でのチャネル毎正規化)
        # Broadcasting: (n_windows, window, 8, 8) に対し (1, 1, 8, 8) の平均・分散
        mean = np.mean(tmp_X.reshape(-1, ch_size, ch_size), axis=0)
        std = np.std(tmp_X.reshape(-1, ch_size, ch_size), axis=0) + 1e-8
        tmp_X = (tmp_X - mean.reshape(1, 1, ch_size, ch_size)) / std.reshape(1, 1, ch_size, ch_size)

        # 3. 特徴抽出
        if feature_name == 'rms':
            feats = np.array([rms_feat(x) for x in tmp_X]) # (n_win, 8, 8)
        elif feature_name == 'wl':
            feats = np.array([wl_feat(x) for x in tmp_X]) # (n_win, 8, 8)
        elif feature_name == 'ptp':
            feats = np.array([ptp_feat(x) for x in tmp_X]) # (n_win, 8, 8)
        elif feature_name == 'arv':
            feats = np.array([arv_feat(x) for x in tmp_X]) # (n_win, 8, 8)
        elif feature_name == 'zc':
            feats = np.array([zc_feat(x) for x in tmp_X]) # (n_win, 8, 8)
        else:
            raise ValueError(f"Unknown feature name: {feature_name}")
        feats_flat = feats.reshape(feats.shape[0], -1)     # (n_win, 64)
        
        X_features.append(feats_flat)
        y_labels.extend([int(label)-1] * feats_flat.shape[0])

    if len(X_features) > 0:
        return np.vstack(X_features), np.array(y_labels)
    else:
        return np.array([]), np.array([])


# ==========================================
# Main Execution Logic
# ==========================================

def main(feature_name='wl'):
    subjects = ['nojima', 'takeuchi2', 'yamamoto', 'stefan']
    test_positions = ["original2", "downleft5mm", "downleft10mm", "clockwise"]
    
    # 結果保存
    global_results = {
        "Method1 (Adaptive)": {pos: [] for pos in test_positions},
        "Method2 (LDA)":      {pos: [] for pos in test_positions}
    }
    
    print(f"Comparison started for subjects: {subjects}")

    for subject in subjects:
        print(f"\n{'='*20} Processing Subject: {subject} {'='*20}")
        
        # --- A. Load Training Data (Original Position) ---
        emg_train, y_train_raw = load_raw_data(subject, "original")
        
        if not emg_train:
            print("  No training data. Skipping.")
            continue

        # -----------------------------------------------
        # 1. Train Method 1 (ProtoNet) - Original Logic
        # -----------------------------------------------
        print("  [Method 1] Preparing Data & Training ProtoNet...")
        # WL特徴量, 正規化なし
        X_train_proto, y_train_proto = create_dataset_proto(emg_train, [y-1 for y in y_train_raw], mode='full', feature_name=feature_name)
        
        X_train_p_t = torch.FloatTensor(X_train_proto).to(device)
        y_train_p_t = torch.LongTensor(y_train_proto).to(device)
        
        proto_model = ProtoNet(input_dim=X_train_proto.shape[1]).to(device)
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

        # -----------------------------------------------
        # 2. Train Method 2 (LDA) - New Logic
        # -----------------------------------------------
        print("  [Method 2] Preparing Data & Training LDA...")
        # RMS特徴量, 特定の正規化
        X_train_lda, y_train_lda = create_dataset_lda(emg_train, y_train_raw, is_test_split=False, feature_name=feature_name)
        
        lda_model = LinearDiscriminantAnalysis()
        lda_model.fit(X_train_lda, y_train_lda)

        # -----------------------------------------------
        # 3. Evaluation Loop (Target Positions)
        # -----------------------------------------------
        print("  Evaluating both methods...")
        proto_model.eval()
        
        for electrode_place in test_positions:
            emg_test, y_test_raw = load_raw_data(subject, electrode_place)
            if not emg_test: continue

            acc_buf_m1 = []
            acc_buf_m2 = []
            
            # 5-fold evaluation
            for shot_idx in range(5):
                # --- Method 1 Eval: One-Shot Adaptation ---
                # Support: shot_idx, Query: others
                y_corr = [y-1 for y in y_test_raw]
                X_supp_p, y_supp_p = create_dataset_proto(emg_test, y_corr, mode='support', shot_trial_idx=shot_idx, feature_name=feature_name)
                X_query_p, y_query_p = create_dataset_proto(emg_test, y_corr, mode='query', shot_trial_idx=shot_idx, feature_name=feature_name)
                
                if len(X_query_p) == 0: continue

                with torch.no_grad():
                    supp_t = torch.FloatTensor(X_supp_p).to(device)
                    ysupp_t = torch.LongTensor(y_supp_p).to(device)
                    query_t = torch.FloatTensor(X_query_p).to(device)
                    yquery_t = torch.LongTensor(y_query_p).to(device)
                    
                    # Adaptation
                    emb_supp = proto_model(supp_t)
                    emb_query = proto_model(query_t)
                    new_protos = calculate_prototypes(emb_supp, ysupp_t, 7)
                    
                    dists = euclidean_dist(emb_query, new_protos)
                    preds = torch.argmin(dists, dim=1)
                    score_m1 = (preds == yquery_t).float().mean().item()

                # --- Method 2 Eval: LDA Prediction ---
                # Target: Query set corresponding to the same split (to be fair)
                # create_dataset_lda で is_test_split=True, shot_trial_idx=shot_idx を指定して
                # shot_idx以外の試行(Query相当)のみを取得
                X_query_lda, y_query_lda = create_dataset_lda(emg_test, y_test_raw, is_test_split=True, shot_trial_idx=shot_idx, feature_name=feature_name)
                
                if len(X_query_lda) == 0: continue

                preds_lda = lda_model.predict(X_query_lda)
                score_m2 = np.mean(preds_lda == y_query_lda)

                # Store
                global_results["Method1 (Adaptive)"][electrode_place].append(score_m1)
                global_results["Method2 (LDA)"][electrode_place].append(score_m2)
                acc_buf_m1.append(score_m1)
                acc_buf_m2.append(score_m2)
            
            print(f"    {electrode_place}: M1(Proto)={np.mean(acc_buf_m1):.3f}, M2(LDA)={np.mean(acc_buf_m2):.3f}")

    # ==========================================
    # 4. Statistical Analysis
    # ==========================================
    print("\n" + "#"*60)
    print("FINAL STATISTICAL RESULTS")
    print("#"*60)
    
    all_acc_m1_mean = []
    all_acc_m2_mean = []

    print("\n>>> Method Comparison (Paired t-test)")
    for pos in test_positions:
        acc1 = global_results["Method1 (Adaptive)"][pos]
        acc2 = global_results["Method2 (LDA)"][pos]
        
        m1 = np.mean(acc1)
        m2 = np.mean(acc2)
        all_acc_m1_mean.append(m1)
        all_acc_m2_mean.append(m2)
        
        print(f'\n[Method1: Adaptation] accuracy of {pos}: {m1:.4f}')
        print(f'[Method2: LDA       ] accuracy of {pos}: {m2:.4f}')
        
        if len(acc1) > 1:
            t, p = ttest_rel(acc1, acc2)
            sig = "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"   => Paired t-test: p={p:.4e} ({sig})")

    print("-" * 60)
    print(f'Method1 (Adaptation) all: {np.mean(all_acc_m1_mean):.4f}')
    print(f'Method2 (LDA)        all: {np.mean(all_acc_m2_mean):.4f}')

    print("\n>>> Position Comparison (Tukey HSD)")
    for method in ["Method1 (Adaptive)", "Method2 (LDA)"]:
        print(f"\n[{method}]")
        vals, groups = [], []
        for pos in test_positions:
            scores = global_results[method][pos]
            vals.extend(scores)
            groups.extend([pos]*len(scores))
        if vals:
            try:
                print(pairwise_tukeyhsd(vals, groups))
            except:
                print("Error in HSD")

if __name__ == "__main__":
    main(feature_name='wl')