import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# 1. データの読み込み
# df_base = pd.read_csv('pn_results.csv')
df_base = pd.read_csv('lda_results.csv')
# df_base = pd.read_csv('2srnn_results.csv')
# df_base = pd.read_csv('adabn_results.csv')
df_pa = pd.read_csv('protonet_results.csv')

# ラベル付け
df_base['Method'] = 'Base'
df_pa['Method'] = 'PA'

# 2. データの結合（可視化用）
df_viz = pd.concat([df_base, df_pa])

# 3. 統計検定（位置ごとのWilcoxon検定）
print("=== ② Base vs PA: Wilcoxon Test by Position ===")
positions = df_base['Position'].unique()

# ペアデータ作成
df_paired = pd.merge(df_base, df_pa, on=['Subject', 'Position', 'Fold'], suffixes=('_Base', '_PA'))

for pos in positions:
    subset = df_paired[df_paired['Position'] == pos]
    stat, p = stats.wilcoxon(subset['Accuracy_Base'], subset['Accuracy_PA'])
    print(f"Position: {pos:15s} | p-value: {p:.5e} {'*' if p<0.05 else ''}")

# 4. 箱ひげ図の描画
plt.figure(figsize=(10, 6))
sns.boxplot(x='Position', y='Accuracy', hue='Method', data=df_viz, palette="Set2")
plt.title('Comparison of Accuracy by Position (Base vs PA)')
plt.ylabel('Accuracy')
plt.xlabel('Electrode Position')
plt.legend(title='Method')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 保存または表示
plt.savefig('boxplot_base_vs_pa_by_pos.png')
plt.show()