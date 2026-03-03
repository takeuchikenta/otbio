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
import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.preprocessing import StandardScaler

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
# 1. データセット作成 (RNN用に変更)
# ==========================================

def prepare_data_indices(n_trials, n_time, fs, is_support_set=False, shot_trial_idx=0):
    """
    shot_trial_idx: 0始まりのインデックス。ターゲットとなる試行番号。
    """
    trials_per_gesture = 5
    n_gestures = 7
    indices = []
    
    if not (0 <= shot_trial_idx < trials_per_gesture):
        raise ValueError(f"shot_trial_idx must be between 0 and {trials_per_gesture - 1}")

    for g in range(n_gestures):
        start_trial_idx = g * trials_per_gesture
        target_idx = start_trial_idx + shot_trial_idx
        
        if is_support_set:
            # Domain Adaptation用: 指定された1試行のみを使用
            trial_indices = [target_idx] 
            # PDF [cite: 284] では "50% of trials" などのシナリオがあるが、
            # ここではOne-shot想定のため、shot_trial_idxの試行全体、あるいは
            # 元コードのロジックに従い前半を使用します。
            time_limit = 2000 # 1秒分
        else:
            # 評価用: 残りの試行
            all_indices = list(range(start_trial_idx, start_trial_idx + trials_per_gesture))
            all_indices.remove(target_idx)
            trial_indices = all_indices
            time_limit = n_time 
            
        indices.append((g, trial_indices, time_limit))
    return indices

def create_dataset_rnn(emg_list, y_list, fs=2000, window_ms=200, step_ms=50, mode='full', shot_trial_idx=0):
    """
    RNN用に (Batch, Sequence, Features) の形式で返す
    PDF  に従い、スライディングウィンドウでシーケンスを作成。
    """
    data = np.array(emg_list) # (Trials, Time, 8, 8)
    n_trials, n_time, _, _ = data.shape
    # チャンネルをフラット化: (Trials, Time, 64)
    data = data.reshape(n_trials, n_time, -1)
    
    window_samples = int((window_ms / 1000) * fs)
    step_samples = int((step_ms / 1000) * fs)
    
    # RNNへの入力シーケンス長を減らすためのダウンサンプリング
    # 例: 200ms @ 2000Hz = 400サンプル -> 重すぎるため 1/10 にする (40シーケンス)
    downsample_rate = 10 
    
    X_seq = []
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
                
                # (Window_Size, 64) のデータを切り出し
                window_data = valid_data[start:end, :]
                
                # ダウンサンプリングしてシーケンスとする
                # shape: (Seq_Len, 64)
                seq_data = window_data[::downsample_rate, :]
                
                X_seq.append(seq_data)
                y_labels.append(gesture_label)
                
    X_seq = np.array(X_seq) # (N_samples, Seq_Len, 64)
    y_labels = np.array(y_labels)

    # 正規化 (StandardScaler)
    # (N, T, F) -> (N*T, F) でfitして戻す
    if len(X_seq) > 0:
        N, T, F = X_seq.shape
        scaler = StandardScaler()
        X_flat = X_seq.reshape(-1, F)
        X_flat = scaler.fit_transform(X_flat)
        X_seq = X_flat.reshape(N, T, F)
                
    return X_seq, y_labels

# ==========================================
# 2. モデル定義 (PDFに基づき変更)
# ==========================================

class DomainAdaptationLayer(nn.Module):
    """
    PDF [cite: 156-158] 参照:
    線形変換層 x' = Mx + b
    Stage 1では M=Identity, b=0 に固定。
    Stage 2では M, b を学習する。
    """
    def __init__(self, input_dim):
        super(DomainAdaptationLayer, self).__init__()
        # input_dim x input_dim の全結合層 (バイアスあり)
        self.linear = nn.Linear(input_dim, input_dim, bias=True)
        
        # 初期化: 単位行列とゼロバイアス [cite: 172]
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(input_dim))
            self.linear.bias.zero_()

    def forward(self, x):
        # x: (Batch, Seq, Feat)
        # Linearは最後の次元に適用されるため、時系列データの各ステップに適用される
        return self.linear(x)

class GestureClassifierRNN(nn.Module):
    """
    PDF [cite: 163, 255-257] 参照:
    2-stack RNN (LSTM) followed by FC and Softmax
    """
    def __init__(self, input_dim, num_classes=7, hidden_dim=512, dropout=0.5):
        super(GestureClassifierRNN, self).__init__()
        
        # LSTM: 2 layers, 512 hidden units [cite: 256]
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # Fully Connected Layer [cite: 257]
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 512),
            nn.ReLU(), # PDFには明記ないがFCの後に通常活性化関数を入れる。ただし論文図では直接Softmax前段へ。
                       # 文脈的に "G-way fully-connected layer with 512 units ... and a softmax" とあるので
                       # 512ユニットの層 -> 出力層(G) が正しい解釈
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes) 
        )

    def forward(self, x):
        # x: (Batch, Seq, Feat)
        # LSTM出力: (Batch, Seq, Hidden)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Many-to-One: 最後のタイムステップの出力を使用 
        last_out = lstm_out[:, -1, :]
        
        logits = self.fc(last_out)
        return logits

class TwoStageRNN(nn.Module):
    """
    全体のモデル: Domain Adaptation Layer -> RNN Classifier
    """
    def __init__(self, input_dim, num_classes=7):
        super(TwoStageRNN, self).__init__()
        self.da_layer = DomainAdaptationLayer(input_dim)
        self.classifier = GestureClassifierRNN(input_dim, num_classes)
        
    def forward(self, x):
        # 1. Domain Adaptation (Linear Transform)
        x_adapted = self.da_layer(x)
        # 2. Sequence Classification
        logits = self.classifier(x_adapted)
        return logits

# ==========================================
# メイン処理
# ==========================================

def main():
    # パラメータ
    subjects = ['nojima', 'takeuchi2', 'yamamoto', 'stefan']
    test_positions = ["original2", "downleft5mm", "downleft10mm", "clockwise"]
    input_dim = 64 # 8x8 pixels flattened
    n_classes = 7
    
    # PDF[cite: 258]: Adam lr=0.001
    lr_pretrain = 0.001
    lr_adapt = 0.001 
    
    # PDF[cite: 278, 332]: Pretrain 100 epochs, Adapt 5-100 epochs (ここでは短めに設定)
    n_epochs_pretrain = 50 # 時間短縮のため50 (論文は100)
    n_epochs_adapt = 10     # 論文では5 epochsで十分な効果とある [cite: 332]
    
    all_results_data = []

    print(f"2-Stage RNN (2SRNN) Comparison started for subjects: {subjects}")

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
                    # データ読み込み処理は既存と同じ
                    df = pd.read_csv(path, encoding='utf-8-sig', sep=';', header=None) 
                    time_emg = df.iloc[:, 0].values
                    emg_data = df.iloc[:, 1:65].values
                    fs = int(1 / np.mean(np.diff(time_emg)))
                    emg_data = remove_power_line_harmonics(emg_data, fs=fs, fundamental=60.0, Q=30.0)
                    filtered_emg = butter_bandpass_filter(emg_data, fs=fs, low_hz=20.0, high_hz=450.0, order=4)
                    emg_data = filtered_emg.reshape(-1,8,8)
                    
                    emg_list_train.append(emg_data)
                    y_list_train.append(j) # 0-indexed labels
                except FileNotFoundError:
                    pass
        
        # データセット作成 (RNN用Sequence)
        print("Creating Training Sequences...")
        X_train, y_train = create_dataset_rnn(emg_list_train, y_list_train, fs=2000, window_ms=200, step_ms=50, mode='full')
        
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.LongTensor(y_train).to(device)
        
        # --- Stage 1: Pre-training (Source Domain) ---
        # PDF: Domain adaptation weights frozen (Identity), train classifier.
        print("\n[Stage 1] Pre-training RNN on Source Data...")
        
        model = TwoStageRNN(input_dim=input_dim, num_classes=n_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        
        # Optimizer: Train ONLY classifier parameters
        optimizer_pre = optim.Adam(model.classifier.parameters(), lr=lr_pretrain)
        
        model.train()
        # DAレイヤーは固定（初期値は単位行列）
        for param in model.da_layer.parameters():
            param.requires_grad = False
            
        # Batch学習の簡易実装
        batch_size = 128
        n_samples = len(X_train_t)
        
        for epoch in range(n_epochs_pretrain):
            perm = torch.randperm(n_samples)
            total_loss = 0
            for i in range(0, n_samples, batch_size):
                indices = perm[i:i+batch_size]
                batch_x, batch_y = X_train_t[indices], y_train_t[indices]
                
                optimizer_pre.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer_pre.step()
                total_loss += loss.item()
            
            if (epoch+1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{n_epochs_pretrain}, Loss: {total_loss/(n_samples/batch_size):.4f}")

        # Pre-training完了後のモデル状態を保存（各テストポジションでリセットするため）
        # state_dictをコピー
        pretrained_state = model.classifier.state_dict()

        # --- Test Loops ---
        accracy_all = []
        time_adaptation_list = []
        
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
            
            # 各施行(Shot)をターゲットとして適応
            for target_shot_index in range(5):
                # データ分割 (Support=Adaptation data, Query=Test data)
                X_support, y_support = create_dataset_rnn(emg_list_test, y_list_test, mode='support', shot_trial_idx=target_shot_index)
                X_query, y_query = create_dataset_rnn(emg_list_test, y_list_test, mode='query', shot_trial_idx=target_shot_index)
                
                if len(X_support) == 0 or len(X_query) == 0:
                    continue

                X_support_t = torch.FloatTensor(X_support).to(device)
                y_support_t = torch.LongTensor(y_support).to(device)
                X_query_t = torch.FloatTensor(X_query).to(device)
                y_query_t = torch.LongTensor(y_query).to(device)

                # --- Stage 2: Domain Adaptation ---
                # PDF: Freeze classifier, Train DA layer.
                
                # 1. モデルのリセット: ClassifierはPre-trainedに戻し、DAレイヤーは初期化
                model.classifier.load_state_dict(pretrained_state)
                # DAレイヤーをIdentityにリセット
                with torch.no_grad():
                    model.da_layer.linear.weight.copy_(torch.eye(input_dim).to(device))
                    model.da_layer.linear.bias.zero_()

                # 2. 設定: Classifier固定、DA学習可能
                for param in model.classifier.parameters():
                    param.requires_grad = False
                for param in model.da_layer.parameters():
                    param.requires_grad = True
                
                optimizer_adapt = optim.Adam(model.da_layer.parameters(), lr=lr_adapt)
                
                # 3. 適応学習 (One-shot / Few-shot)
                model.train() # BNやDropoutの挙動に注意(今回はFCとLSTM)。論文では明記ないが通常Adapt時もTrainモード
                
                start_adap = time.time()
                for epoch in range(n_epochs_adapt):
                    # Support set全体を使って勾配更新
                    optimizer_adapt.zero_grad()
                    outputs = model(X_support_t)
                    loss = criterion(outputs, y_support_t)
                    loss.backward()
                    optimizer_adapt.step()
                end_adap = time.time()
                time_adaptation_list.append(end_adap - start_adap)

                # --- Inference ---
                model.eval()
                with torch.no_grad():
                    outputs_query = model(X_query_t)
                    _, y_pred = torch.max(outputs_query, 1)
                    score = (y_pred == y_query_t).float().mean().item()

                    all_results_data.append({
                        "Subject": subject,
                        "Method": "2SRNN(Adaptive)",
                        "Position": electrode_place,
                        "Fold": target_shot_index,
                        "Accuracy": score
                    })
                
                accuracy_each_position.append(score)
                accracy_all.append(score)
            
            print(f'  Avg Accuracy ({electrode_place}): {np.mean(accuracy_each_position):.4f}')

        print(f'Subject Average Accuracy: {np.mean(accracy_all):.4f}')
        print(f'Avg Adaptation Time: {np.mean(time_adaptation_list):.4f} sec')

    # 結果保存
    print("\n" + "="*40)
    print("Saving results to '2srnn_results.csv'...")
    df_results = pd.DataFrame(all_results_data)
    df_results.to_csv("2srnn_results.csv", index=False)
    print(df_results.groupby(["Subject", "Position"])["Accuracy"].mean())
    print("Done.")

if __name__ == "__main__":
    main()