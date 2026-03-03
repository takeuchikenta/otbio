import pandas as pd

def process_gesture_averages(file_path_1, file_path_2, output_prefix="processed"):
    files = [file_path_1, file_path_2]
    
    for i, file_path in enumerate(files):
        # 1. ファイルの読み込み
        df = pd.read_csv(file_path)
        
        # 2. gestureラベルの付与
        # fold(0-4)が7回繰り返されているため、35行周期で0~6のラベルを割り当て
        # 各ブロック内（35行）で、5行ごとにgesture番号が変わる計算
        num_folds = 5
        df['gesture'] = (df.index % (num_folds * 7)) // num_folds
        
        # 3. gestureごとの平均値を計算
        # 数値列のみを対象にするため numeric_only=True を指定
        gesture_means = df.groupby('gesture').mean(numeric_only=True)
        
        # 4. CSVとして出力
        output_name = f"{output_prefix}_file_{i+1}.csv"
        gesture_means.to_csv(output_name)
        print(f"Saved: {output_name}")

# 実行
process_gesture_averages("align_lda_results.csv", "align_pa_results_allgestures.csv")