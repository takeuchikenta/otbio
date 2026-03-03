import pandas as pd
from scipy import stats

def calculate_wilcoxon_significance(file_path):
    # 1. データの読み込み
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("ファイルが見つかりません。パスを確認してください。")
        return

    # 基準となるポジション
    baseline_pos = 'original2'
    
    # 基準データ（original2）の抽出
    df_base = df[df['Position'] == baseline_pos]
    
    # 比較対象となるその他のポジションを取得
    other_positions = [p for p in df['Position'].unique() if p != baseline_pos]
    
    print(f"=== Wilcoxon Signed-Rank Test: {baseline_pos} vs Others ===")
    print(f"{'Comparison':<30} | {'n':<3} | {'Statistic':<10} | {'p-value':<10} | {'Significance'}")
    print("-" * 80)

    results = []

    for pos in other_positions:
        # 比較対象データの抽出
        df_target = df[df['Position'] == pos]
        
        # 2. データのペアリング（SubjectとFoldで結合）
        #    inner joinにより、欠損なく両方のデータが揃っているペアのみを残します
        merged = pd.merge(df_base, df_target, on=['Subject', 'Fold'], suffixes=('_base', '_target'))
        
        n = len(merged)
        
        if n < 2:
            print(f"{baseline_pos} vs {pos:20s} | データ不足のため計算不可")
            continue

        # 3. Wilcoxon検定の実行
        #    両側検定 (two-sided) をデフォルトとして使用
        stat, p_val = stats.wilcoxon(merged['Accuracy_base'], merged['Accuracy_target'])
        
        # 有意差の判定マーク
        if p_val < 0.001: sig = "***"
        elif p_val < 0.01: sig = "**"
        elif p_val < 0.05: sig = "*"
        else: sig = "n.s."
        
        print(f"{baseline_pos} vs {pos:20s} | {n:<3} | {stat:<10.1f} | {p_val:.2e}   | {sig}")
        
        results.append({
            'Comparison': f"{baseline_pos} vs {pos}",
            'n': n,
            'statistic': stat,
            'p_value': p_val,
            'significance': sig
        })

    return pd.DataFrame(results)

# 実行
if __name__ == "__main__":
    # calculate_wilcoxon_significance('lda_results.csv')
    # calculate_wilcoxon_significance('filtered_align_lda_results.csv')
    # calculate_wilcoxon_significance('protonet_results.csv')
    # calculate_wilcoxon_significance('align_pa_results.csv')
    # calculate_wilcoxon_significance('pn_results.csv')
    calculate_wilcoxon_significance('2srnn_results.csv')