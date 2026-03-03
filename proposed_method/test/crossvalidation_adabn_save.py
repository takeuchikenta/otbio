import wfdb
import os
import glob
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, sosfiltfilt
import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----- バンドパスフィルタ等の基本関数 (変更なし) -----
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

# ファイル名出力関数
def file_name_output(subject, hand="right", electrode_place="original", gesture=1, trial=1):
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
# 1. データセット作成 (CNN用フレーム画像に変更)
# ==========================================

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
            # Adaptation/Calibration用
            trial_indices = [target_idx] 
            time_limit = 2000 
        else:
            # Test用
            all_indices = list(range(start_trial_idx, start_trial_idx + trials_per_gesture))
            all_indices.remove(target_idx)
            trial_indices = all_indices
            time_limit = n_time 
            
        indices.append((g, trial_indices, time_limit))
    return indices

def create_dataset_frames(emg_list, y_list, fs=2000, window_ms=200, step_ms=50, mode='full', shot_trial_idx=0):
    """
    CNN用に (Batch, 1, 8, 8) の形式で返す
    各時点を1枚の画像として扱う (Per-frame recognition)
    """
    data = np.array(emg_list) # (Trials, Time, 8, 8)
    n_trials, n_time, _, _ = data.shape
    
    # 画像データとして扱うため、空間構造 (8x8) を保持
    # PDF[cite: 589]: linear transformation to grayscale. 
    # ここではStandardScaler等で正規化を行う
    
    # ダウンサンプリング (学習高速化のため間引く)
    downsample_rate = 10 
    
    X_frames = []
    y_labels = []
    
    if mode == 'full':
        target_config = [(g, range(g*5, (g+1)*5), n_time) for g in range(7)]
    elif mode == 'support':
        target_config = prepare_data_indices(5, n_time, fs, is_support_set=True, shot_trial_idx=shot_trial_idx)
    elif mode == 'query':
        target_config = prepare_data_indices(5, n_time, fs, is_support_set=False, shot_trial_idx=shot_trial_idx)
    
    for gesture_label, trial_indices, time_limit in target_config:
        for t_idx in trial_indices:
            trial_data = data[t_idx] # (Time, 8, 8)
            limit = min(time_limit, trial_data.shape[0])
            
            # 指定範囲のデータを取得し、ダウンサンプリング
            valid_data = trial_data[:limit:downsample_rate, :, :]
            
            # リストに追加
            # shape: (N_frames, 8, 8)
            for frame in valid_data:
                X_frames.append(frame)
                y_labels.append(gesture_label)
                
    X_frames = np.array(X_frames) # (Total_Frames, 8, 8)
    y_labels = np.array(y_labels)

    # チャンネル次元を追加 (Total_Frames, 1, 8, 8)
    if len(X_frames) > 0:
        X_frames = X_frames[:, np.newaxis, :, :]
        
        # 正規化 (全体に対して)
        # PDFでは [0, 255] への線形変換だが、NN学習では平均0分散1が一般的
        mean = np.mean(X_frames)
        std = np.std(X_frames)
        X_frames = (X_frames - mean) / (std + 1e-8)
                
    return X_frames, y_labels

# ==========================================
# 2. モデル定義 (Deep Domain Adaptation CNN)
# ==========================================

class ConvNetAdaBN(nn.Module):
    """
    PDF [cite: 615] Figure 2 に基づくアーキテクチャ
    Input: 1@8x16 (本コードでは 1@8x8 に適応)
    Layers:
      1. Conv 3x3 (64) + BN + ReLU
      2. Conv 3x3 (64) + BN + ReLU
      3. Locally Connected (approx. by Conv 1x1) (64) + BN + ReLU
      4. Locally Connected (approx. by Conv 1x1) (64) + BN + ReLU + Dropout
      5. FC (512) + BN + ReLU + Dropout
      6. FC (512) + BN + ReLU + Dropout
      7. FC (128) + BN + ReLU
      8. FC (Output) + Softmax
    """
    def __init__(self, num_classes=7):
        super(ConvNetAdaBN, self).__init__()
        
        # Feature Extractor (Convolutional Blocks)
        self.features = nn.Sequential(
            # Layer 1: Conv 3x3
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Layer 2: Conv 3x3
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Layer 3: Locally Connected (Approx with 1x1 Conv) [cite: 592]
            nn.Conv2d(64, 64, kernel_size=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Layer 4: Locally Connected + Dropout
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5) # [cite: 595]
        )
        
        # Classifier (Fully Connected Blocks)
        # 8x8 input -> features output is 64x8x8
        flatten_dim = 64 * 8 * 8
        
        self.classifier = nn.Sequential(
            # Layer 5: FC 512
            nn.Flatten(),
            nn.Linear(flatten_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), # [cite: 595]
            
            # Layer 6: FC 512
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), # [cite: 595]
            
            # Layer 7: FC 128
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            # Layer 8: Output
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def adabn_adaptation(model, support_loader, n_epochs=1, device='cpu'):
    """
    AdaBN (Adaptive Batch Normalization) 
    ターゲットドメインのデータを使って、BatchNormの統計量(mean, var)のみを更新する。
    重み(weight, bias)は更新しない。
    """
    model.train() # BNの統計量を更新するため trainモードにする
    
    # 全パラメータを固定 (勾配計算なし)
    for param in model.parameters():
        param.requires_grad = False
        
    # PDF[cite: 662]: update statistics using unlabeled calibration data
    # 複数回通して統計量を安定させる (moving average)
    for epoch in range(n_epochs):
        for batch_x, _ in support_loader:
            batch_x = batch_x.to(device)
            # Forward passのみ実行（統計量が更新される）
            with torch.no_grad():
                _ = model(batch_x)
                
    # 推論用にパラメータ固定のまま eval に戻さない
    # 注: PyTorchの仕様上、eval()にすると記録された統計量を使うようになる。
    # AdaBNでは「ターゲットデータで更新した統計量」を使いたいので、更新後はeval()にする。
    model.eval() 

# ==========================================
# メイン処理
# ==========================================

def main():
    subjects = ['nojima', 'takeuchi2', 'yamamoto', 'stefan']
    test_positions = ["original2", "downleft5mm", "downleft10mm", "clockwise"]
    n_classes = 7
    
    # PDF[cite: 699]: SGD, batch 1000 (ここではデータ量に合わせて調整), LR 0.1
    lr_train = 0.001 # Adam用に調整
    batch_size = 64
    n_epochs_pretrain = 20
    
    all_results_data = []

    print(f"Deep Domain Adaptation (AdaBN) Comparison started for subjects: {subjects}")

    for subject in subjects:
        print(f"\n{'='*20} Processing Subject: {subject} {'='*20}")
        
        # --- データ読み込み (Training) ---
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
                    y_list_train.append(j)
                except FileNotFoundError:
                    pass
        
        # データセット作成 (CNN用フレーム)
        print("Creating Training Frames...")
        X_train, y_train = create_dataset_frames(emg_list_train, y_list_train, fs=2000, mode='full')
        
        # DataLoader化
        train_tensor_x = torch.FloatTensor(X_train)
        train_tensor_y = torch.LongTensor(y_train)
        train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # --- Stage 1: Pre-training (Source Domain) ---
        print("\n[Stage 1] Pre-training ConvNet on Source Data...")
        
        model = ConvNetAdaBN(num_classes=n_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr_train)
        
        model.train()
        for epoch in range(n_epochs_pretrain):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch+1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{n_epochs_pretrain}, Loss: {total_loss/len(train_loader):.4f}")

        # Pre-training完了後のモデル状態を保存
        pretrained_state = model.state_dict()

        # --- Test Loops ---
        accracy_all = []
        
        for electrode_place in test_positions:
            print(f'Test Position: {electrode_place}')
            
            # テストデータ読み込み
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
                        y_list_test.append(j)
                    except FileNotFoundError:
                        pass

            accuracy_each_position = []
            
            for target_shot_index in range(5):
                # データ分割 (Support=Calibration, Query=Test)
                X_support, _ = create_dataset_frames(emg_list_test, y_list_test, mode='support', shot_trial_idx=target_shot_index)
                X_query, y_query = create_dataset_frames(emg_list_test, y_list_test, mode='query', shot_trial_idx=target_shot_index)
                
                if len(X_support) == 0 or len(X_query) == 0:
                    continue

                # DataLoader作成
                support_tensor_x = torch.FloatTensor(X_support)
                support_tensor_y = torch.LongTensor(np.zeros(len(X_support))) # ラベル不要
                support_dataset = torch.utils.data.TensorDataset(support_tensor_x, support_tensor_y)
                support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=batch_size, shuffle=True)

                query_tensor_x = torch.FloatTensor(X_query)
                query_tensor_y = torch.LongTensor(y_query)
                query_dataset = torch.utils.data.TensorDataset(query_tensor_x, query_tensor_y)
                query_loader = torch.utils.data.DataLoader(query_dataset, batch_size=batch_size, shuffle=False)

                # --- Stage 2: Domain Adaptation (AdaBN) ---
                # 1. モデルのリセット
                model.load_state_dict(pretrained_state)
                
                # 2. AdaBN実行 (Unsupervised) [cite: 662]
                # ターゲットデータ(X_support)を流してBN統計量のみ更新
                start_adap = time.time()
                adabn_adaptation(model, support_loader, n_epochs=1, device=device)
                end_adap = time.time()

                # --- Inference ---
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch_x, batch_y in query_loader:
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        outputs = model(batch_x)
                        _, predicted = torch.max(outputs.data, 1)
                        total += batch_y.size(0)
                        correct += (predicted == batch_y).sum().item()

                score = correct / total
                
                all_results_data.append({
                    "Subject": subject,
                    "Method": "ConvNet+AdaBN",
                    "Position": electrode_place,
                    "Fold": target_shot_index,
                    "Accuracy": score
                })
                
                accuracy_each_position.append(score)
                accracy_all.append(score)
            
            print(f'  Avg Accuracy ({electrode_place}): {np.mean(accuracy_each_position):.4f}')

        print(f'Subject Average Accuracy: {np.mean(accracy_all):.4f}')

    # 結果保存
    print("\n" + "="*40)
    print("Saving results to 'adabn_results.csv'...")
    df_results = pd.DataFrame(all_results_data)
    df_results.to_csv("adabn_results.csv", index=False)
    print(df_results.groupby(["Subject", "Position"])["Accuracy"].mean())
    print("Done.")

if __name__ == "__main__":
    main()