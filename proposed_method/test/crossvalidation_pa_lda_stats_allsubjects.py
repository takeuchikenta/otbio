import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import signal
from scipy.signal import butter, sosfiltfilt
from scipy.stats import ttest_rel
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
import matplotlib.pyplot as plt

# ### 修正点: LDAとStandardScalerのインポート ###
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 不要な警告を無視
warnings.filterwarnings('ignore')
np.warnings = warnings

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. Helper Functions (共通処理)
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

def prepare_data_indices(n_trials, n_time, fs, is_support_set=False, shot_trial_idx=0):
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

def create_dataset(emg_list, y_list, fs=2000, window_ms=200, step_ms=50, mode='full', shot_trial_idx=0, feature_name='wl'):
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
        target_config = prepare_data_indices(5, n_time, fs, is_support_set=True, shot_trial_idx=shot_trial_idx)
    elif mode == 'query':
        target_config = prepare_data_indices(5, n_time, fs, is_support_set=False, shot_trial_idx=shot_trial_idx)
    
    for gesture_label, trial_indices, time_limit in target_config:
        for t_idx in trial_indices:
            trial_data = data[t_idx]
            limit = min(time_limit, trial_data.shape[0])
            valid_data = trial_data[:limit, :]
            
            for start in range(0, limit - window_samples + 1, step_samples):
                end = start + window_samples
                window = valid_data[start:end, :]
                
                feat_map = extract_feature_map(window, feature_name=feature_name)
                X_feat.append(feat_map.flatten())
                y_labels.append(gesture_label)
    
    X_feat = np.array(X_feat)
    y_labels = np.array(y_labels)

    # チャネル毎のZ-score正規化
    if len(X_feat) > 0:
        scaler = StandardScaler()
        X_feat = scaler.fit_transform(X_feat)
                
    return X_feat, y_labels

# ==========================================
# 2. Model (ProtoNet Only)
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

# ==========================================
# 3. Main Logic (Multi-Subject)
# ==========================================

def main(feature_name='wl'):
    subjects = ['nojima', 'takeuchi2', 'yamamoto', 'stefan']
    test_positions = ["original2", "downleft5mm", "downleft10mm", "clockwise"]
    
    # 結果保存用コンテナ
    # Method 2 を Source Only ProtoNet から LDA に変更
    global_results = {
        "Method1 (Adaptive)": {pos: [] for pos in test_positions},
        "Method2 (LDA)":      {pos: [] for pos in test_positions}
    }
    
    print(f"Comparison started for subjects: {subjects}")

    # --- Subject Loop ---
    for subject in subjects:
        print(f"\n{'='*20} Processing Subject: {subject} {'='*20}")
        
        # 1. Training Phase (Source Domain: Original)
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
                    emg_data = remove_power_line_harmonics(emg_data, fs=fs)
                    filtered_emg = butter_bandpass_filter(emg_data, fs=fs, low_hz=20.0, high_hz=450.0)
                    emg_data = filtered_emg.reshape(-1,8,8)
                    emg_list_train.append(emg_data)
                    y_list_train.append(j+1)
                except FileNotFoundError:
                    pass
        
        if not emg_list_train:
            print(f"  [Error] No training data found for {subject}. Skipping.")
            continue

        X_train, y_train = create_dataset(emg_list_train, [y-1 for y in y_list_train], mode='full', feature_name=feature_name)
        
        # --- Model 1: ProtoNet Training ---
        print("  Training ProtoNet (Method 1)...")
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.LongTensor(y_train).to(device)
        
        model = ProtoNet(input_dim=X_train.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0005)
        n_epochs = 300
        n_classes = 7
        
        model.train()
        for epoch in range(n_epochs):
            embeddings = model(X_train_t)
            prototypes = calculate_prototypes(embeddings, y_train_t, n_classes)
            dists = euclidean_dist(embeddings, prototypes)
            log_p_y = torch.log_softmax(-dists, dim=1)
            loss = -log_p_y.gather(1, y_train_t.view(-1, 1)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # --- Model 2: LDA Training (Source) ---
        print("  Training LDA (Method 2)...")
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train, y_train)

        # 2. Test Phase (Target Domains)
        print("  Evaluating...")
        model.eval()
        
        for electrode_place in test_positions:
            emg_list_test = []
            y_list_test = []
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
                        emg_list_test.append(emg_data)
                        y_list_test.append(j+1)
                    except FileNotFoundError:
                        pass
            
            if not emg_list_test:
                continue

            acc_buffer_m1 = []
            acc_buffer_m2 = []
            
            for target_shot_index in range(5):
                y_corrected = [y-1 for y in y_list_test]
                
                # データセット作成 (正規化込み)
                X_test_support, y_test_support = create_dataset(emg_list_test, y_corrected, mode='support', shot_trial_idx=target_shot_index, feature_name=feature_name)
                X_test_query, y_test_query = create_dataset(emg_list_test, y_corrected, mode='query', shot_trial_idx=target_shot_index, feature_name=feature_name)

                # --- Method 1: ProtoNet Adaptation ---
                X_support_t = torch.FloatTensor(X_test_support).to(device)
                y_support_t = torch.LongTensor(y_test_support).to(device)
                X_query_t = torch.FloatTensor(X_test_query).to(device)
                y_query_t = torch.LongTensor(y_test_query).to(device)

                with torch.no_grad():
                    # QueryのEmbedding
                    query_embeddings = model(X_query_t)
                    # Supportを用いてプロトタイプ適応
                    support_embeddings = model(X_support_t)
                    adapted_prototypes = calculate_prototypes(support_embeddings, y_support_t, n_classes)
                    
                    dists_m1 = euclidean_dist(query_embeddings, adapted_prototypes)
                    y_pred_m1 = torch.argmin(dists_m1, dim=1)
                    score_m1 = (y_pred_m1 == y_query_t).float().mean().item()
                    
                # --- Method 2: LDA (Source Only) ---
                # LDAはnumpy配列を入力に取ります
                y_pred_m2 = lda.predict(X_test_query)
                score_m2 = np.mean(y_pred_m2 == y_test_query)

                # Global Containerに追加
                global_results["Method1 (Adaptive)"][electrode_place].append(score_m1)
                global_results["Method2 (LDA)"][electrode_place].append(score_m2)
                
                acc_buffer_m1.append(score_m1)
                acc_buffer_m2.append(score_m2)

            print(f"    {electrode_place}: M1(Proto)={np.mean(acc_buffer_m1):.3f}, M2(LDA)={np.mean(acc_buffer_m2):.3f}")

    # ==========================================
    # 4. Statistical Analysis (Aggregated)
    # ==========================================
    print("\n\n" + "#"*60)
    print("FINAL STATISTICAL RESULTS (All Subjects Aggregated)")
    print("#"*60)
    
    all_acc_m1_mean = []
    all_acc_m2_mean = []

    # --- Analysis 1: Method Comparison per Position ---
    print("\n>>> Comparison: Method 1 (Adaptation) vs Method 2 (LDA) per Position")
    
    for electrode_place in test_positions:
        acc_list_m1 = global_results["Method1 (Adaptive)"][electrode_place]
        acc_list_m2 = global_results["Method2 (LDA)"][electrode_place]
        
        m1_mean = np.mean(acc_list_m1)
        m2_mean = np.mean(acc_list_m2)
        all_acc_m1_mean.append(m1_mean)
        all_acc_m2_mean.append(m2_mean)
        
        print(f'\n[Method1: Adaptation] accuracy of {electrode_place}: {m1_mean:.4f}')
        print(f'[Method2: LDA       ] accuracy of {electrode_place}: {m2_mean:.4f}')

        if len(acc_list_m1) > 1:
            t_stat, p_val = ttest_rel(acc_list_m1, acc_list_m2)
            sig_mark = "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"   => Paired t-test: p-value = {p_val:.4e} ({sig_mark})")
        else:
            print("   => Not enough samples for t-test.")

    print("\n" + "-"*60)
    print(f'Method1 (Adaptation) accracy_all: {np.mean(all_acc_m1_mean):.4f}')
    print(f'Method2 (LDA)        accracy_all: {np.mean(all_acc_m2_mean):.4f}')
    print("-"*60)

    # --- Analysis 2: Position Comparison within each Method ---
    print("\n\n>>> Comparison: Between Electrode Positions (Tukey HSD)")
    
    for method_name in ["Method1 (Adaptive)", "Method2 (LDA)"]:
        print(f"\n[{method_name}] Multiple Comparison Results:")
        
        all_scores = []
        all_groups = []
        
        for pos in test_positions:
            scores = global_results[method_name][pos]
            all_scores.extend(scores)
            all_groups.extend([pos] * len(scores))
            
        if len(all_scores) > 0:
            try:
                tukey = pairwise_tukeyhsd(endog=all_scores, groups=all_groups, alpha=0.05)
                print(tukey)
            except Exception as e:
                print(f"Error in Tukey HSD: {e}")
        else:
            print("No data available for statistics.")

if __name__ == "__main__":
    main()