import wfdb
import os
import glob
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import median_filter
from sklearn.preprocessing import StandardScaler
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

np.warnings = warnings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# ユーティリティ関数（フィルタリング・ファイル読み込み）
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

def band_edges_safe(low_hz: float, high_hz: float, fs: float, margin: float = 0.01) -> tuple[float,float]:
    nyq = fs * 0.5
    low = max(0.0, float(low_hz))
    high = min(float(high_hz), nyq*(1.0 - margin))
    if not (0.0 < low < high < nyq):
        raise ValueError(f"Invalid band: low={low_hz}, high={high_hz}, fs={fs}")
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
    # ファイル名生成ロジック（元のコードと同様）
    if electrode_place == "original": filename = "1-original"
    elif electrode_place == "upright": filename = "2-upright"
    elif electrode_place == "downright": filename = "3-downright"
    elif electrode_place == "downleft": filename = "4-downleft"
    elif electrode_place == "upleft": filename = "5-upleft"
    elif electrode_place == "clockwise": filename = "6-clockwise"
    elif electrode_place == "anticlockwise": filename = "7-anticlockwise"
    elif electrode_place == "original2": filename = "original2"
    elif electrode_place == "downleft5mm": filename = "downleft5mm"
    elif electrode_place == "downleft10mm": filename = "downleft10mm"
    
    file_name = subject + '/' + hand + '/' + filename + '/set' + str(trial) + '/' + electrode_place + '-g' + str(gesture) + '-' + str(trial) + '.csv'
    return file_name

# ==========================================
# 1. 前処理とデータセット作成
# ==========================================

def extract_feature_map(window_data, feature_name='wl'):
    """
    (Time, Channels_Flat) -> (8, 8) Feature Map
    """
    if feature_name == 'wl':
        feature_flat = np.sum(np.abs(np.diff(window_data, axis=0)), axis=0)
    elif feature_name == 'ptp':
        feature_flat = np.ptp(window_data, axis=0)
    elif feature_name == 'rms':
        feature_flat = np.sqrt(np.mean(np.square(window_data), axis=0))
    else:
        raise ValueError(f"Unknown feature name: {feature_name}")
    
    # 8x8 マップに変形
    feature_map = feature_flat.reshape(8, 8)
    return feature_map

def prepare_data_indices(n_trials, n_time, fs, is_support_set=False, shot_trial_idx=0):
    trials_per_gesture = 5
    n_gestures = 7
    indices = []
    
    if not (0 <= shot_trial_idx < trials_per_gesture):
        raise ValueError(f"shot_trial_idx must be between 0 and {trials_per_gesture - 1}")

    for g in range(n_gestures):
        start_trial_idx = g * trials_per_gesture
        target_idx = start_trial_idx + shot_trial_idx
        
        if is_support_set:
            # サポートセット: 指定された1試行の最初の1秒間 (2000 samples)
            trial_indices = [target_idx] 
            time_limit = 2000 
        else:
            # クエリセット: 残りの試行すべて
            all_indices = list(range(start_trial_idx, start_trial_idx + trials_per_gesture))
            all_indices.remove(target_idx)
            trial_indices = all_indices
            time_limit = n_time
            
        indices.append((g, trial_indices, time_limit))
            
    return indices

def create_dataset(emg_list, y_list, fs=2000, window_ms=200, step_ms=50, mode='full', shot_trial_idx=0, feature_name='wl'):
    """
    データセット作成: (N_samples, 1, 8, 8) の画像形式で返す
    """
    data = np.array(emg_list) # (Trials, Time, 8, 8)
    n_trials, n_time, _, _ = data.shape
    data = data.reshape(n_trials, n_time, -1) # Flatten channels for processing
    
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
                
                # 特徴抽出 (8, 8)
                feat_map = extract_feature_map(window, feature_name=feature_name)
                
                # [cite_start]メディアンフィルタ (論文 Eq.2 [cite: 313])
                feat_map_filtered = median_filter(feat_map, size=3, mode='reflect')
                
                X_feat.append(feat_map_filtered)
                y_labels.append(gesture_label)

    X_feat = np.array(X_feat) # (N, 8, 8)
    y_labels = np.array(y_labels)

    # 標準化 (全体に対して適用)
    if len(X_feat) > 0:
        # channelごとに正規化するため (N, 64) にしてscaler適用後戻す
        N, H, W = X_feat.shape
        X_flat = X_feat.reshape(N, -1)
        scaler = StandardScaler()
        X_flat = scaler.fit_transform(X_flat)
        X_feat = X_flat.reshape(N, 1, H, W) # (N, Channel=1, 8, 8) for ConvNet

    return X_feat, y_labels

# ==========================================
# 2. モデル定義 (Deep Domain Adaptation)
# ==========================================
# [cite_start]論文 [cite: 212-216, 236] のアーキテクチャに基づく
# Input: 8x8 (論文は8x16だがデータに合わせて調整)
# Structure: 
#   Conv(3x3) -> BN -> ReLU
#   Conv(3x3) -> BN -> ReLU
#   LocallyConnected(1x1) -> BN -> ReLU (Approximated by Conv1x1)
#   LocallyConnected(1x1) -> BN -> ReLU
#   FC(512) -> BN -> ReLU -> Dropout
#   FC(512) -> BN -> ReLU -> Dropout
#   FC(128) -> BN -> ReLU
#   Softmax

class AdaBNConvNet(nn.Module):
    def __init__(self, num_classes=7):
        super(AdaBNConvNet, self).__init__()
        
        # Layer 1: Conv 3x3, 64 filters
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Layer 2: Conv 3x3, 64 filters
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Layer 3: Locally Connected (approx with Conv 1x1), 64 filters
        self.lc1 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Layer 4: Locally Connected (approx with Conv 1x1), 64 filters
        self.lc2 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        # Flatten size: 64 * 8 * 8
        self.flatten_dim = 64 * 8 * 8
        
        # Layer 5: FC 512
        self.fc1 = nn.Linear(self.flatten_dim, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        
        # Layer 6: FC 512
        self.fc2 = nn.Linear(512, 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.5)
        
        # Layer 7: FC 128
        self.fc3 = nn.Linear(512, 128)
        self.bn7 = nn.BatchNorm1d(128)
        
        # Layer 8: Output
        self.fc4 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (Batch, 1, 8, 8)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.lc1(x)))
        x = F.relu(self.bn4(self.lc2(x)))
        
        x = x.view(x.size(0), -1) # Flatten
        
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn6(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn7(self.fc3(x)))
        
        x = self.fc4(x) # CrossEntropyLoss includes Softmax
        return x

# ==========================================
# 3. 学習・適応・評価ロジック
# ==========================================

import copy
import time

def main():
    feature_name = 'wl'
    subjects = ['nojima', 'takeuchi2', 'yamamoto', 'stefan']
    test_positions = ["original2", "downleft5mm", "downleft10mm", "clockwise"]
    
    all_results_data = []

    print(f"Comparison started for subjects: {subjects}")

    for subject in subjects:
        print(f"\n{'='*20} Processing Subject: {subject} {'='*20}")
        
        # --- 1. 学習データの読み込み (Source Domain) ---
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
                    emg_data = remove_power_line_harmonics(emg_data, fs=fs)
                    filtered_emg = butter_bandpass_filter(emg_data, fs=fs, low_hz=20.0, high_hz=450.0, order=4)
                    emg_data = filtered_emg.reshape(-1,8,8)
                    
                    emg_list_train.append(emg_data)
                    y_list_train.append(j+1)
                except FileNotFoundError:
                    pass
        
        # Dataset作成 (Source)
        print("Creating Source Dataset...")
        X_train, y_train = create_dataset(emg_list_train, y_list_train, mode='full', feature_name=feature_name)
        
        # Tensor化
        X_train_t = torch.FloatTensor(X_train).to(device)
        # 【修正】 create_datasetは既に0-indexed (0~6) を返すため、 -1 は不要
        y_train_t = torch.LongTensor(y_train).to(device)
        
        train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

        # --- 2. Source Modelの学習 ---
        model = AdaBNConvNet(num_classes=7).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001) # L2 reg
        
        print("Training Source Model...")
        model.train()
        n_epochs = 50 
        for epoch in range(n_epochs):
            total_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(train_loader):.4f}")
        
        # 学習済みモデルの状態辞書を保存（コピー）
        source_model_state = copy.deepcopy(model.state_dict())

        # --- 3. テスト（位置ずれデータに対する適応） ---
        accracy_all = []
        time_adaptation_list = []

        for electrode_place in test_positions: 
            print(f'\n--- Testing on position: {electrode_place} ---')
            
            # テストデータの読み込み
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
                        emg_data = remove_power_line_harmonics(emg_data, fs=fs)
                        filtered_emg = butter_bandpass_filter(emg_data, fs=fs, low_hz=20.0, high_hz=450.0, order=4)
                        emg_data = filtered_emg.reshape(-1,8,8)
                        emg_list_test.append(emg_data)
                        y_list_test.append(j+1)
                    except FileNotFoundError:
                        pass
            
            accuracy_each_position = []
            
            # 5-fold cross validation (One-shot scenario)
            for target_shot_index in range(5):
                # データの分割 (Support: Labeled, Query: Unlabeled Stream)
                X_support, y_support = create_dataset(emg_list_test, y_list_test, mode='support', shot_trial_idx=target_shot_index, feature_name=feature_name)
                X_query, y_query = create_dataset(emg_list_test, y_list_test, mode='query', shot_trial_idx=target_shot_index, feature_name=feature_name)
                
                # Tensor化
                X_support_t = torch.FloatTensor(X_support).to(device)
                # 【修正】 -1 は不要
                y_support_t = torch.LongTensor(y_support).to(device)
                X_query_t = torch.FloatTensor(X_query).to(device)
                # 【修正】 -1 は不要
                y_query_t = torch.LongTensor(y_query).to(device)

                # モデルのリセット（Sourceの重みに戻す）
                model.load_state_dict(source_model_state)
                
                start_adap = time.time()
                
                # --- Step A: Fine-tuning on Support Set (Labeled) ---
                # ラベル付きサポートデータを使って重みを更新
                model.train()
                ft_optimizer = optim.Adam(model.parameters(), lr=0.0001) # 低い学習率
                batch_size_ft = 16
                ft_epochs = 10
                
                # データローダー作成
                if len(X_support_t) > 0:
                    support_dataset = torch.utils.data.TensorDataset(X_support_t, y_support_t)
                    support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=batch_size_ft, shuffle=True)
                    
                    for _ in range(ft_epochs):
                        for X_b, y_b in support_loader:
                            ft_optimizer.zero_grad()
                            out = model(X_b)
                            loss = criterion(out, y_b)
                            loss.backward()
                            ft_optimizer.step()

                # --- Step B: Continuous Adaptation on Query Set (Unlabeled) ---
                # クエリデータを逐次処理し、BatchNorm統計量を更新しながら推論する
                
                model.train() # BN更新のためTrainモード
                correct_counts = 0
                total_counts = 0
                
                if len(X_query_t) > 0:
                    query_batch_size = 32
                    query_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_query_t, y_query_t), batch_size=query_batch_size, shuffle=False)
                    
                    with torch.no_grad(): # 重み更新はしない
                        for X_b, y_b in query_loader:
                            # 1. Adaptation Step: BN statistics update
                            _ = model(X_b)
                            
                            # 2. Prediction Step
                            outputs = model(X_b)
                            preds = torch.argmax(outputs, dim=1)
                            
                            correct_counts += (preds == y_b).sum().item()
                            total_counts += y_b.size(0)

                end_adap = time.time()
                time_adaptation_list.append(end_adap - start_adap)
                
                score = correct_counts / total_counts if total_counts > 0 else 0
                
                all_results_data.append({
                    "Subject": subject,
                    "Method": "ConvNet+AdaBN+FT",
                    "Position": electrode_place,
                    "Fold": target_shot_index,
                    "Accuracy": score
                })
                
                accuracy_each_position.append(score)
                accracy_all.append(score)
                
            print(f'Average Accuracy for {electrode_place}: {np.mean(accuracy_each_position):.4f}')

        print(f'Overall Accuracy for {subject}: {np.mean(accracy_all):.4f}')
        print(f'Average Time for Adaptation: {np.mean(time_adaptation_list):.4f} sec')

    # 結果出力
    print("\n" + "="*40)
    print("Saving results to 'domain_adaptation_results.csv'...")
    df_results = pd.DataFrame(all_results_data)
    df_results.to_csv("domain_adaptation_results.csv", index=False)
    # Group by Subject and Position to check means
    if not df_results.empty:
        print(df_results.groupby(["Subject", "Position"])["Accuracy"].mean())
    print("Done.")

if __name__ == '__main__':
    main()