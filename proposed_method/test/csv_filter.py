# import pandas as pd

# # CSVファイルを読み込む
# df = pd.read_csv('align_protonet_results.csv')

# # Fold列が0の行のみを抽出
# df_filtered = df[df['Fold'] == 0]

# # 抽出したデータを新しいCSVファイルとして保存
# output_file = 'filtered_align_protonet_results.csv'
# df_filtered.to_csv(output_file, index=False)

# print(f"抽出完了: {output_file}")

# import pandas as pd

# # CSVファイルを読み込む
# df = pd.read_csv('align_protonet_results.csv')

# # Position列でグループ化し、各グループの最初の5行を抽出
# df_top5 = df.groupby('Position').head(5)

# # 抽出したデータを新しいCSVファイルとして保存
# output_file = 'filtered_align_protonet_results.csv'
# df_top5.to_csv(output_file, index=False)

# print(f"抽出完了: {output_file}")

import pandas as pd

# CSVファイルを読み込む
df = pd.read_csv('align_protonet_results.csv')

# SubjectとPositionでグループ化し、各グループの最初の5行を抽出
df_top5_sub_pos = df.groupby(['Subject', 'Position']).head(5)

# 抽出したデータを新しいCSVファイルとして保存
output_file = 'filtered_align_protonet_results.csv'
df_top5_sub_pos.to_csv(output_file, index=False)

print(f"抽出完了: {output_file}")