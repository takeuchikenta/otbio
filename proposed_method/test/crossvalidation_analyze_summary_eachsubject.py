import pandas as pd

# 1. 読み込むファイル名のリストを指定します（実際のファイル名に変更してください）
file_names = [
    'pn_results.csv',
    'align_pn_results.csv',
    'protonet_results.csv',
    'align_pa_results.csv'
]

summary_df = None

for i, file in enumerate(file_names):
    # CSVファイルを読み込む
    df = pd.read_csv(file)
    
    # SubjectとPositionごとにAccuracyの平均値を計算
    # .reset_index() で結果をデータフレーム形式に戻す
    avg_df = df.groupby(['Subject', 'Position'])['Accuracy'].mean().reset_index()
    
    # 列名を分かりやすく変更（例: Accuracy -> Mean_file1）
    avg_df = avg_df.rename(columns={'Accuracy': f'Mean_{file}'})
    
    # 2. データを統合する
    if summary_df is None:
        # 最初のファイルの場合はそのまま代入
        summary_df = avg_df
    else:
        # 2つ目以降のファイルは Subject と Position をキーにして横に結合（Merge）
        summary_df = pd.merge(summary_df, avg_df, on=['Subject', 'Position'], how='outer')

# 3. 統合された結果を新しいCSVファイルとして出力
output_filename = 'subject_position_average_summary.csv'
summary_df.to_csv(output_filename, index=False)

print(f"集計が完了しました。出力ファイル: {output_filename}")
# 最初の数行を表示して確認
print(summary_df.head())