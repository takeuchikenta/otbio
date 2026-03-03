import pandas as pd

def print_position_averages(*file_paths):
    
    for i, file_path in enumerate(file_paths):
        # 1. ファイルの読み込み
        df = pd.read_csv(file_path)
        
        # 2. positionラベルの付与
        # CSVに'position'列がない場合、35行（5 folds × 7 gestures）ごとにpositionが切り替わると仮定
        if 'position' not in df.columns:
            df['position'] = df.index // 35
            
        # 3. positionごとの平均値を計算
        pos_means = df.groupby('Position').mean(numeric_only=True)
        
        # 4. 結果の表示
        print(f"\n{'='*30}")
        print(f" 結果: ファイル {i+1} ({file_path})")
        print(f"{'='*30}")
        print(pos_means)
        print("\n")

# 実行（実際のファイル名を指定してください）
# print_position_averages("2srnn_results.csv","lda_results.csv", "filtered_align_lda_results.csv", "protonet_results.csv", "align_pa_results.csv", "pn_results.csv")
print_position_averages("adabn_results.csv","lda_results.csv", "protonet_results.csv")