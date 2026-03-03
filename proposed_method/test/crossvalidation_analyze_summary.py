import pandas as pd

# 1. 処理したいファイル名のリストを指定します
# 実際のファイル名に合わせて書き換えてください
file_names = [
    'pn_results.csv',
    'align_pn_results.csv',
    'protonet_results.csv',
    'align_pa_results.csv',
    'pn_intra_results.csv'
]

summary_list = []

for file in file_names:
    # CSVファイルを読み込み
    df = pd.read_csv(file)
    
    # PositionごとにAccuracyの平均値を計算
    # (被験者やFoldをまたいだ全データの平均になります)
    avg_series = df.groupby('Position')['Accuracy'].mean()
    
    # 結果をデータフレーム形式に変換し、列名をファイル名にする
    avg_df = avg_series.to_frame(name=f'Mean_{file}')
    summary_list.append(avg_df)

# 2. 全ての集計結果をPositionを軸に横に結合する
final_summary = pd.concat(summary_list, axis=1).reset_index()

# 3. 新しいCSVファイルとして出力
output_filename = 'positions_average_summary3.csv'
final_summary.to_csv(output_filename, index=False)

print(f"集計が完了しました。出力ファイル: {output_filename}")
print(final_summary)