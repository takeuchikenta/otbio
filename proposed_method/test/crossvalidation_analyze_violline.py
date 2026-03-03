# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import wilcoxon # 分布の比較用（ノンパラメトリック）

# def analyze_shift_distributions(base_csv, pa_csv):
#     # データの読み込み
#     df_base = pd.read_csv(base_csv)
#     df_pa = pd.read_csv(pa_csv)
    
#     # マージ（Subject, Position, Fold で紐付け）
#     df = pd.merge(df_base, df_pa, on=['Subject', 'Position', 'Fold'], suffixes=('_base', '_pa'))
    
#     # 基準となる original2 の PA 精度を取得して紐付け
#     df_orig2 = df_pa[df_pa['Position'] == 'original2'][['Subject', 'Fold', 'Accuracy']]
#     df = pd.merge(df, df_orig2, on=['Subject', 'Fold'], suffixes=('', '_pa_orig2'))
#     df.rename(columns={'Accuracy': 'Acc_pa_orig2'}, inplace=True)
    
#     # --- 1サンプルごとのインパクト計算 ---
#     # CSの影響: PAで改善した分
#     df['Impact_CS'] = df['Accuracy_pa'] - df['Accuracy_base']
#     # CovSの影響: orig2(PA) から shifted(PA) で落ちた分
#     df['Impact_CovS'] = df['Acc_pa_orig2'] - df['Accuracy_pa']
    
#     positions = [p for p in df['Position'].unique() if p != 'original2']
    
#     print("\n" + "="*70)
#     print(f"{'Position':<15} | {'Mean CS':<10} | {'Mean CovS':<10} | {'Dominant Factor'}")
#     print("-" * 70)
    
#     plot_data = []

#     for pos in positions:
#         sub_df = df[df['Position'] == pos]
#         cs_vals = sub_df['Impact_CS'].values
#         covs_vals = sub_df['Impact_CovS'].values
        
#         # 統計検定: Wilcoxonの符号付順位検定（2つのインパクト分布の差を比較）
#         # 「CSの影響度分布」と「CovSの影響度分布」に有意差があるか
#         stat, p_val = wilcoxon(cs_vals, covs_vals)
        
#         mean_cs = np.mean(cs_vals)
#         mean_covs = np.mean(covs_vals)
#         dominant = "Concept Shift" if mean_cs > mean_covs else "Covariate Shift"
#         sig = "*" if p_val < 0.05 else "ns"
        
#         print(f"{pos:15s} | {mean_cs:.4f}    | {mean_covs:.4f}    | {dominant} ({sig})")
        
#         # 描画用データの整形
#         for v in cs_vals: plot_data.append({'Position': pos, 'Impact': v, 'Type': 'Concept Shift'})
#         for v in covs_vals: plot_data.append({'Position': pos, 'Impact': v, 'Type': 'Covariate Shift'})

#     # --- 可視化: バイオリンプロット（分布の比較） ---
#     plt.figure(figsize=(12, 6))
#     df_plot = pd.DataFrame(plot_data)
#     sns.violinplot(x='Position', y='Impact', hue='Type', data=df_plot, split=True, inner="quart")
    
#     plt.title('Distribution of Impact: Concept Shift vs Covariate Shift')
#     plt.ylabel('Accuracy Drop / Improvement Magnitude')
#     plt.axhline(0, color='black', linestyle='--', alpha=0.3)
#     plt.grid(axis='y', alpha=0.3)
#     plt.tight_layout()
#     plt.show()

# # 実行（ファイル名は各自の出力に合わせてください）
# analyze_shift_distributions('pn_results.csv', 'protonet_results.csv')

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import wilcoxon, norm
# import warnings

# warnings.filterwarnings('ignore')

# def calculate_effect_size(stat, n):
#     """
#     Wilcoxonの検定統計量から効果量 r を算出する
#     r = Z / sqrt(N)
#     """
#     # 検定統計量TからZ値を近似計算
#     # E[T] = n(n+1)/4, Var[T] = n(n+1)(2n+1)/24
#     mu = n * (n + 1) / 4
#     sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
#     z = (stat - mu) / sigma
    
#     r = abs(z) / np.sqrt(n)
#     return r

# def analyze_shift_impact(pa_file='protonet_results.csv', base_file='lda_results.csv'):
#     # 1. データの読み込み
#     try:
#         df_pa = pd.read_csv(pa_file)
#         df_base = pd.read_csv(base_file)
#     except FileNotFoundError as e:
#         print(f"Error: {e}")
#         print("Please ensure 'protonet_results.csv' and 'lda_results.csv' exist.")
#         return

#     # Method列のリネームや不要列の整理
#     df_pa = df_pa[['Subject', 'Position', 'Fold', 'Accuracy']].rename(columns={'Accuracy': 'Acc_PA'})
#     df_base = df_base[['Subject', 'Position', 'Fold', 'Accuracy']].rename(columns={'Accuracy': 'Acc_Base'})

#     # 2. データの結合 (Subject, Position, Fold で紐付け)
#     df = pd.merge(df_pa, df_base, on=['Subject', 'Position', 'Fold'], how='inner')
    
#     # 3. 基準値 (original2 の PA精度) の取得
#     # 各被験者・各Foldにおける「理想的な精度（Shiftなし）」として original2 の PA精度を使用
#     df_orig2 = df[df['Position'] == 'original2'][['Subject', 'Fold', 'Acc_PA']]
#     df_orig2 = df_orig2.rename(columns={'Acc_PA': 'Acc_PA_Orig2'})
    
#     # 全データに基準値を紐付け
#     df = pd.merge(df, df_orig2, on=['Subject', 'Fold'], how='left')

#     # 4. インパクト（精度低下要因）の計算
#     # Concept Shift (CS) = PAで救済できた量 = Acc_PA - Acc_Base
#     df['Impact_CS'] = df['Acc_PA'] - df['Acc_Base']
    
#     # Covariate Shift (CovS) = 位置ずれで失われた量 = Acc_PA(Orig2) - Acc_PA(Pos)
#     # ※ PAはCSを除去済みなので、残る差分はCovSに由来するとみなす
#     df['Impact_CovS'] = df['Acc_PA_Orig2'] - df['Acc_PA']

#     # 5. 統計分析 & 表示
#     target_positions = [p for p in df['Position'].unique() if p != 'original2']
    
#     print("\n" + "="*90)
#     print(f"{'Position':<15} | {'Mean CS':<9} | {'Mean CovS':<9} | {'Dominant Factor':<18} | {'p-value':<10} | {'Effect(r)':<9}")
#     print("="*90)

#     plot_data = []

#     for pos in target_positions:
#         sub_df = df[df['Position'] == pos].copy()
        
#         # データのペアを取得
#         cs_vals = sub_df['Impact_CS'].values
#         covs_vals = sub_df['Impact_CovS'].values
        
#         # サンプル数
#         n = len(cs_vals)
#         if n < 2: continue

#         # --- Wilcoxon Signed-Rank Test ---
#         # 帰無仮説: CSとCovSの影響度分布に差はない
#         try:
#             stat, p_val = wilcoxon(cs_vals, covs_vals)
#             r = calculate_effect_size(stat, n)
#         except ValueError:
#             # 全て同じ値の場合など
#             p_val = 1.0
#             r = 0.0

#         # 代表値（平均）
#         mean_cs = np.mean(cs_vals)
#         mean_covs = np.mean(covs_vals)
        
#         # 判定
#         if p_val < 0.05:
#             if mean_cs > mean_covs:
#                 dom = "Concept Shift"
#             else:
#                 dom = "Covariate Shift"
#             sig_mark = "*" if p_val < 0.05 else ""
#             if p_val < 0.01: sig_mark = "**"
#             if p_val < 0.001: sig_mark = "***"
#         else:
#             dom = "Comparable (ns)"
#             sig_mark = "ns"

#         print(f"{pos:15s} | {mean_cs:.4f}    | {mean_covs:.4f}    | {dom:<18} | {p_val:.2e} {sig_mark:<3} | {r:.3f}")

#         # プロット用データ整形
#         for v in cs_vals:
#             plot_data.append({'Position': pos, 'Impact': v, 'Factor': 'Concept Shift'})
#         for v in covs_vals:
#             plot_data.append({'Position': pos, 'Impact': v, 'Factor': 'Covariate Shift'})

#     # 6. 可視化 (バイオリンプロット)
#     print("-" * 90)
#     print("Generating Plot...")
    
#     if not plot_data:
#         print("No data to plot.")
#         return

#     df_plot = pd.DataFrame(plot_data)
    
#     plt.figure(figsize=(12, 6))
#     sns.set(style="whitegrid")
    
#     # Split Violin Plot
#     ax = sns.violinplot(x='Position', y='Impact', hue='Factor', data=df_plot, 
#                         split=True, inner='quartile', palette="muted", cut=0)
    
#     plt.title("Comparison of Impact Distributions: Concept Shift vs Covariate Shift", fontsize=14)
#     plt.ylabel("Impact on Accuracy (Magnitude of Drop/Recovery)", fontsize=12)
#     plt.xlabel("Electrode Position", fontsize=12)
#     plt.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
#     plt.legend(title="Factor", loc='upper right')
    
#     # 結果の保存も可能
#     # plt.savefig("shift_impact_analysis.png", dpi=300)
#     plt.show()

# if __name__ == "__main__":
#     # ファイル名は保存したCSVに合わせてください
#     analyze_shift_impact(pa_file='protonet_results.csv', base_file='pn_results.csv')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
import warnings

warnings.filterwarnings('ignore')

def calculate_effect_size(stat, n):
    """
    Wilcoxonの検定統計量から効果量 r を算出
    """
    mu = n * (n + 1) / 4
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (stat - mu) / sigma
    r = abs(z) / np.sqrt(n)
    return r

def analyze_shift_impact_boxplot(pa_file='protonet_results.csv', base_file='lda_results.csv'):
    # 1. データの読み込み
    try:
        df_pa = pd.read_csv(pa_file)
        df_base = pd.read_csv(base_file)
    except FileNotFoundError:
        print("Error: Files not found.")
        return

    # Method列のリネームや不要列の整理
    df_pa = df_pa[['Subject', 'Position', 'Fold', 'Accuracy']].rename(columns={'Accuracy': 'Acc_PA'})
    df_base = df_base[['Subject', 'Position', 'Fold', 'Accuracy']].rename(columns={'Accuracy': 'Acc_Base'})

    # 2. データの結合
    df = pd.merge(df_pa, df_base, on=['Subject', 'Position', 'Fold'], how='inner')
    
    # 3. 基準値 (original2 の PA精度) の取得
    df_orig2 = df[df['Position'] == 'original2'][['Subject', 'Fold', 'Acc_PA']]
    df_orig2 = df_orig2.rename(columns={'Acc_PA': 'Acc_PA_Orig2'})
    df = pd.merge(df, df_orig2, on=['Subject', 'Fold'], how='left')

    # 4. インパクト計算
    # CS = PAで改善した分
    df['Impact_CS'] = df['Acc_PA'] - df['Acc_Base']
    # CovS = 位置ずれで失われた分
    df['Impact_CovS'] = df['Acc_PA_Orig2'] - df['Acc_PA']

    # 5. 統計分析 & プロット用データ作成
    target_positions = [p for p in df['Position'].unique() if p != 'original2']
    target_positions = ["downleft5mm", "downleft10mm", "clockwise"]
    plot_data = []

    print("\n" + "="*90)
    print(f"{'Position':<15} | {'Mean CS':<9} | {'Mean CovS':<9} | {'Dominant Factor':<18} | {'p-value':<10} | {'Effect(r)':<9}")
    print("="*90)

    for pos in target_positions:
        sub_df = df[df['Position'] == pos].copy()
        cs_vals = sub_df['Impact_CS'].values
        covs_vals = sub_df['Impact_CovS'].values
        n = len(cs_vals)
        if n < 2: continue

        # 統計検定
        try:
            stat, p_val = wilcoxon(cs_vals, covs_vals)
            r = calculate_effect_size(stat, n)
        except ValueError:
            p_val = 1.0; r = 0.0

        mean_cs = np.mean(cs_vals)
        mean_covs = np.mean(covs_vals)
        
        # 判定
        if p_val < 0.05:
            dom = "Physiological variation" if mean_cs > mean_covs else "Electrode shift"
            sig_mark = "*" if p_val < 0.05 else ""
            if p_val < 0.01: sig_mark = "**"
            if p_val < 0.001: sig_mark = "***"
        else:
            dom = "Comparable (ns)"
            sig_mark = "ns"

        print(f"{pos:15s} | {mean_cs:.4f}    | {mean_covs:.4f}    | {dom:<18} | {p_val:.2e} {sig_mark:<3} | {r:.3f}")

        # プロット用データ整形
        for v in cs_vals:
            plot_data.append({'Position': pos, 'Impact': v, 'Factor': 'Physiological variation'})
        for v in covs_vals:
            plot_data.append({'Position': pos, 'Impact': v, 'Factor': 'Electrode shift'})

    # 6. 可視化 (箱ひげ図)
    if not plot_data: return

    df_plot = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")
    
    # 箱ひげ図 (Box Plot)
    # showfliers=False: 外れ値を箱ひげ図側では非表示にし、stripplotで全点を描画する
    ax = sns.boxplot(x='Position', y='Impact', hue='Factor', data=df_plot, 
                     palette="muted", showfliers=False, width=0.6)
    
    # 個別データのプロット (Strip Plot) を重ねる
    # これにより、分布の密度やサンプルごとのばらつきも可視化できる
    sns.stripplot(x='Position', y='Impact', hue='Factor', data=df_plot, 
                  dodge=True, jitter=True, color='black', alpha=0.3, size=3, ax=ax)
    
    ax.set_xticklabels(['trans5', 'trans10', 'rotation'], fontsize=16)

    # 凡例の整理 (boxplotとstripplotで重複するため調整)
    handles, labels = ax.get_legend_handles_labels()
    # 最初の2つ(Boxplotの凡例)だけを使用
    ax.legend(handles[:2], labels[:2], title="Factor", loc='upper right', fontsize=16, title_fontsize=16)

    # plt.title("Comparison of Impact: Concept Shift vs Covariate Shift (Box Plot)", fontsize=14)
    plt.ylabel("Impact Magnitude (Accuracy Change)", fontsize=16)
    plt.xlabel("Electrode Position", fontsize=16)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_shift_impact_boxplot(pa_file='protonet_results.csv', base_file='pn_results.csv')

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import wilcoxon
# import warnings

# warnings.filterwarnings('ignore')

# def analyze_shift_impact_pvalue(pa_file='protonet_results.csv', base_file='lda_results.csv'):
#     # 1. データの読み込み
#     try:
#         df_pa = pd.read_csv(pa_file)
#         df_base = pd.read_csv(base_file)
#     except FileNotFoundError:
#         print("Error: Files not found. Please ensure CSV files exist.")
#         return

#     # Method列のリネームや不要列の整理
#     df_pa = df_pa[['Subject', 'Position', 'Fold', 'Accuracy']].rename(columns={'Accuracy': 'Acc_PA'})
#     df_base = df_base[['Subject', 'Position', 'Fold', 'Accuracy']].rename(columns={'Accuracy': 'Acc_Base'})

#     # 2. データの結合 (Subject, Position, Fold で紐付け)
#     df = pd.merge(df_pa, df_base, on=['Subject', 'Position', 'Fold'], how='inner')
    
#     # 3. 基準値 (original2 の PA精度) の取得
#     # 各被験者・各Foldにおける「理想的な精度（Shiftなし）」として original2 の PA精度を使用
#     df_orig2 = df[df['Position'] == 'original2'][['Subject', 'Fold', 'Acc_PA']]
#     df_orig2 = df_orig2.rename(columns={'Acc_PA': 'Acc_PA_Orig2'})
    
#     # 全データに基準値を紐付け
#     df = pd.merge(df, df_orig2, on=['Subject', 'Fold'], how='left')

#     # 4. インパクト（精度低下要因）の計算
#     # Concept Shift (CS) = PAで救済できた量 = Acc_PA - Acc_Base
#     df['Impact_CS'] = df['Acc_PA'] - df['Acc_Base']
    
#     # Covariate Shift (CovS) = 位置ずれで失われた量 = Acc_PA(Orig2) - Acc_PA(Pos)
#     # ※ PAはCSを除去済みなので、残る差分はCovSに由来するとみなす
#     df['Impact_CovS'] = df['Acc_PA_Orig2'] - df['Acc_PA']

#     # 5. 統計分析 & 表示
#     # target_positions = [p for p in df['Position'].unique() if p != 'original2']
#     target_positions = ["downleft5mm", "downleft10mm", "clockwise"]
    
#     print("\n" + "="*85)
#     print(f"{'Position':<15} | {'Mean CS':<9} | {'Mean CovS':<9} | {'Dominant Factor':<18} | {'p-value':<12}")
#     print("="*85)

#     plot_data = []

#     for pos in target_positions:
#         sub_df = df[df['Position'] == pos].copy()
        
#         # データのペアを取得
#         cs_vals = sub_df['Impact_CS'].values
#         covs_vals = sub_df['Impact_CovS'].values
        
#         n = len(cs_vals)
#         if n < 2: continue

#         # --- Wilcoxon Signed-Rank Test ---
#         # 帰無仮説: CSとCovSの影響度分布に差はない
#         try:
#             stat, p_val = wilcoxon(cs_vals, covs_vals)
#         except ValueError:
#             # 全て同じ値の場合など
#             p_val = 1.0

#         # 代表値（平均）
#         mean_cs = np.mean(cs_vals)
#         mean_covs = np.mean(covs_vals)
        
#         # 判定 (有意水準 5%)
#         if p_val < 0.05:
#             if mean_cs > mean_covs:
#                 dom = "Concept Shift"
#             else:
#                 dom = "Covariate Shift"
            
#             # 有意水準のマーク
#             if p_val < 0.001: sig_mark = "***"
#             elif p_val < 0.01: sig_mark = "**"
#             else: sig_mark = "*"
#         else:
#             dom = "Comparable"
#             sig_mark = "ns"

#         print(f"{pos:15s} | {mean_cs:.4f}    | {mean_covs:.4f}    | {dom:<18} | {p_val:.2e} {sig_mark:<3}")

#         # プロット用データ整形
#         for v in cs_vals:
#             plot_data.append({'Position': pos, 'Impact': v, 'Factor': 'Concept Shift'})
#         for v in covs_vals:
#             plot_data.append({'Position': pos, 'Impact': v, 'Factor': 'Covariate Shift'})

#     # 6. 可視化 (箱ひげ図)
#     if not plot_data:
#         print("No data to plot.")
#         return

#     df_plot = pd.DataFrame(plot_data)
    
#     plt.figure(figsize=(12, 6))
#     sns.set(style="whitegrid")
    
#     # 箱ひげ図 (Box Plot)
#     # showfliers=False: 外れ値は非表示（stripplotで表示するため）
#     ax = sns.boxplot(x='Position', y='Impact', hue='Factor', data=df_plot, 
#                      palette="muted", showfliers=False, width=0.6)
    
#     # 個別データのプロット (Strip Plot) を重ねる
#     sns.stripplot(x='Position', y='Impact', hue='Factor', data=df_plot, 
#                   dodge=True, jitter=True, color='black', alpha=0.3, size=3, ax=ax)
    
#     # 修正箇所（末尾の plt.show() の直前に追加）
#     ax.set_xticklabels(['trans5', 'trans10', 'rotation'], fontsize=16)

#     # 凡例の整理
#     handles, labels = ax.get_legend_handles_labels()
#     ax.legend(handles[:2], labels[:2], title="Factor", loc='upper right', fontsize=16, title_fontsize=16)

#     # plt.title("Comparison of Impact: Concept Shift vs Covariate Shift", fontsize=14)
#     plt.ylabel("Impact Magnitude (Accuracy Change)", fontsize=16)
#     plt.xlabel("Electrode Position", fontsize=16)
#     plt.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     analyze_shift_impact_pvalue(pa_file='protonet_results.csv', base_file='pn_results.csv')