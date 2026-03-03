import pandas as pd

def summarize_icc_results(file_paths, output_csv="averaged_icc_results.csv"):
    """
    複数のCSVファイルを読み込み、Subject, Position, channel ごとに
    icc の平均値を計算してCSVに出力する。
    """
    data_frames = []
    
    # 1. すべてのファイルを読み込んでリストに格納
    for file in file_paths:
        try:
            df = pd.read_csv(file)
            data_frames.append(df)
            print(f"読み込み完了: {file}")
        except Exception as e:
            print(f"エラー (ファイル {file}): {e}")
            
    if not data_frames:
        print("読み込めるデータがありませんでした。")
        return

    # 2. 全データを1つのデータフレームに結合
    df_all = pd.concat(data_frames, ignore_index=True)
    
    # 3. Subject, Position, channel でグループ化し、icc の平均を計算
    # numeric_only=True を指定して、数値列である icc のみを対象にします
    summary = df_all.groupby(['Subject', 'Position', 'channel'], as_index=False)['icc'].mean()
    
    # 4. (オプション) チャンネル番号順などに並び替えたい場合
    # channelが 'ch1', 'ch2' のような形式の場合、自然順でソートされます
    summary = summary.sort_values(by=['Subject', 'Position', 'channel'])
    
    # 5. CSVファイルに出力
    summary.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*40)
    print(f"処理が完了しました。")
    print(f"保存先: {output_csv}")
    print("="*40)
    
    # 最初の数行を表示して確認
    print("\n--- 出力内容のプレビュー ---")
    print(summary.head(10))

# 使い方：4つのファイル名をリストに入れて実行してください
file_list = [
    "icc2_resultsnojima.csv", 
    "icc2_resultstakeuchi2.csv", 
    "icc2_resultsyamamoto.csv", 
    "icc2_resultsstefan.csv"
]

# 実行
summarize_icc_results(file_list, "total_icc_summary.csv")