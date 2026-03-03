import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------------------------------------
# ④ Ablation Study (Wilcoxon Signed-Rank Test Only)
# ---------------------------------------------------------

# 1. データの読み込み
df_base = pd.read_csv('lda_results.csv')         # Base (Control)
df_pa = pd.read_csv('protonet_results.csv')     # PA
df_align = pd.read_csv('filtered_align_lda_results.csv')  # Align
df_align_pa = pd.read_csv('align_pa_results.csv') # Align + PA

# ラベル付け
df_base['Method'] = 'Base'
df_pa['Method'] = 'PA'
df_align['Method'] = 'Align'
df_align_pa['Method'] = 'AlignPA'

# 2. データの結合（可視化用）
df_viz = pd.concat([df_base, df_pa, df_align, df_align_pa])
method_order = ['Base', 'PA', 'Align', 'AlignPA']  # グラフの並び順

# 3. 統計検定（位置ごとのWilcoxon検定）
print("=== Ablation Study: Wilcoxon Signed-Rank Test (Base vs Others) ===")

# ペア比較用にデータをマージ（Baseを基準に結合）
df_merged = df_base[['Subject', 'Position', 'Fold', 'Accuracy']].rename(columns={'Accuracy': 'Base'})
methods = {'PA': df_pa, 'Align': df_align, 'AlignPA': df_align_pa}

for name, df in methods.items():
    temp = df[['Subject', 'Position', 'Fold', 'Accuracy']].rename(columns={'Accuracy': name})
    df_merged = pd.merge(df_merged, temp, on=['Subject', 'Position', 'Fold'], how='inner')

positions = df_base['Position'].unique()
comparisons = ['PA', 'Align', 'AlignPA']

for pos in positions:
    subset = df_merged[df_merged['Position'] == pos]
    n = len(subset)
    print(f"\n[Position: {pos}] (n={n})")
    
    for comp in comparisons:
        # Wilcoxon検定実行
        stat, p_val = stats.wilcoxon(subset['Base'], subset[comp])
        
        # p値の判定マーク
        if p_val < 0.01:
            sig = "**"
        elif p_val < 0.05:
            sig = "*"
        else:
            sig = "n.s."
            
        print(f"  Base vs {comp:8s}: p-value = {p_val:.5e}  {sig}")

# 4. 箱ひげ図の描画
plt.figure(figsize=(12, 6))
sns.boxplot(x='Position', y='Accuracy', hue='Method', data=df_viz, 
            hue_order=method_order, palette="Set3")

plt.title('Ablation Study: Accuracy by Position')
plt.ylabel('Accuracy')
plt.xlabel('Electrode Position')
plt.legend(title='Method', loc='lower right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 保存・表示
plt.tight_layout()
plt.savefig('boxplot_ablation_wilcoxon_only.png')
plt.show()