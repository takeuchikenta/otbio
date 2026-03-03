import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# 1. データの読み込み
df_base = pd.read_csv('lda_results.csv')
df_pa = pd.read_csv('protonet_results.csv')
df_align = pd.read_csv('filtered_align_lda_results.csv')
df_align_pa = pd.read_csv('align_pa_results.csv')

# ラベル付け
df_base['Method'] = 'Base'
df_pa['Method'] = 'PA'
df_align['Method'] = 'Align'
df_align_pa['Method'] = 'AlignPA'

# 2. データの結合（可視化用）
df_viz = pd.concat([df_base, df_pa, df_align, df_align_pa])
method_order = ['Base', 'PA', 'Align', 'AlignPA']  # 表示順序

# 3. 統計検定（位置ごとのFriedman + Post-hoc Wilcoxon）
print("\n=== ④ Ablation Study: Friedman & Post-hoc by Position ===")

# ペアデータ作成（横持ち形式）
df_merged = df_base[['Subject', 'Position', 'Fold', 'Accuracy']].rename(columns={'Accuracy': 'Base'})
for name, df in zip(['PA', 'Align', 'AlignPA'], [df_pa, df_align, df_align_pa]):
    temp = df[['Subject', 'Position', 'Fold', 'Accuracy']].rename(columns={'Accuracy': name})
    df_merged = pd.merge(df_merged, temp, on=['Subject', 'Position', 'Fold'])

positions = df_base['Position'].unique()
comparisons = ['PA', 'Align', 'AlignPA']

for pos in positions:
    subset = df_merged[df_merged['Position'] == pos]
    
    # Friedman検定
    stat_f, p_f = stats.friedmanchisquare(subset['Base'], subset['PA'], subset['Align'], subset['AlignPA'])
    print(f"\n[Position: {pos}] Friedman p-value: {p_f:.5e}")
    
    if p_f < 0.05:
        # Post-hoc: Wilcoxon with Bonferroni correction (vs Base)
        for comp in comparisons:
            s, p_raw = stats.wilcoxon(subset['Base'], subset[comp])
            p_bonf = min(p_raw * 3, 1.0) # 3回検定するので3倍
            sig = "*" if p_bonf < 0.05 else ""
            print(f"  vs {comp:8s}: p_corr={p_bonf:.5e} {sig}")

# 4. 箱ひげ図の描画
plt.figure(figsize=(12, 6))
sns.boxplot(x='Position', y='Accuracy', hue='Method', data=df_viz, 
            hue_order=method_order, palette="Set3")
plt.title('Ablation Study by Position')
plt.ylabel('Accuracy')
plt.xlabel('Electrode Position')
plt.legend(title='Method', loc='lower right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 保存または表示
plt.savefig('boxplot_ablation_by_pos.png')
plt.show()