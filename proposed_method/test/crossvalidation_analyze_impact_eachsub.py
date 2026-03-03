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
    if n == 0: return 0
    mu = n * (n + 1) / 4
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if sigma == 0: return 0
    z = (stat - mu) / sigma
    r = abs(z) / np.sqrt(n)
    return r

def analyze_shift_impact_by_subject(pa_file='protonet_results.csv', base_file='pn_results.csv'):
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
    # Subject, Position, Fold をキーにして結合
    df = pd.merge(df_pa, df_base, on=['Subject', 'Position', 'Fold'], how='inner')
    
    # 3. 基準値 (original2 の PA精度) の取得
    # Subjectごとに基準値が異なるため、SubjectとFoldでマージして各行に紐付ける
    df_orig2 = df[df['Position'] == 'original2'][['Subject', 'Fold', 'Acc_PA']]
    df_orig2 = df_orig2.rename(columns={'Acc_PA': 'Acc_PA_Orig2'})
    df = pd.merge(df, df_orig2, on=['Subject', 'Fold'], how='left')

    # 4. インパクト計算
    # Impact_CS (Concept Shift) = 生理的変動への適応による改善分 (PA - Base)
    df['Impact_CS'] = df['Acc_PA'] - df['Acc_Base']
    
    # Impact_CovS (Covariate Shift) = 位置ずれによる性能低下分 (Orig2_PA - Pos_PA)
    # 値が大きいほど「位置ずれの悪影響が大きい」ことを意味する
    df['Impact_CovS'] = df['Acc_PA_Orig2'] - df['Acc_PA']

    # 5. Subjectごとの分析ループ
    subjects = df['Subject'].unique()
    target_positions = ["downleft5mm", "downleft10mm", "clockwise"]
    
    # 全体のプロット用データ（必要に応じて使用）
    all_plot_data = []

    for subject in subjects:
        print(f"\n{'='*30} Results for Subject: {subject} {'='*30}")
        print(f"{'Position':<15} | {'Mean CS':<9} | {'Mean CovS':<9} | {'Dominant Factor':<22} | {'p-value':<10} | {'Effect(r)':<9}")
        print("-" * 105)

        # 特定の被験者のデータを抽出
        df_sub = df[df['Subject'] == subject]
        
        for pos in target_positions:
            sub_pos_df = df_sub[df_sub['Position'] == pos].copy()
            cs_vals = sub_pos_df['Impact_CS'].values
            covs_vals = sub_pos_df['Impact_CovS'].values
            n = len(cs_vals)
            
            # データ数が少なすぎる場合はスキップ
            if n < 2: continue

            # 統計検定 (Wilcoxon Signed-Rank Test)
            try:
                # 全ての値が全く同じ場合などのエラー回避
                if np.allclose(cs_vals, covs_vals):
                    p_val = 1.0; stat = 0
                else:
                    stat, p_val = wilcoxon(cs_vals, covs_vals)
                r = calculate_effect_size(stat, n)
            except ValueError:
                p_val = 1.0; r = 0.0

            mean_cs = np.mean(cs_vals)
            mean_covs = np.mean(covs_vals)
            
            # 判定ロジック
            dom = "Comparable (ns)"
            sig_mark = "ns"
            if p_val < 0.05:
                # 平均値が大きい方を支配的要因とする
                dom = "Physiological var." if mean_cs > mean_covs else "Electrode shift"
                if p_val < 0.001: sig_mark = "***"
                elif p_val < 0.01: sig_mark = "**"
                else: sig_mark = "*"
            
            print(f"{pos:15s} | {mean_cs:.4f}    | {mean_covs:.4f}    | {dom:<22} | {p_val:.2e} {sig_mark:<3} | {r:.3f}")

            # プロット用データの蓄積
            for v in cs_vals:
                all_plot_data.append({'Subject': subject, 'Position': pos, 'Impact': v, 'Factor': 'Physiological variation'})
            for v in covs_vals:
                all_plot_data.append({'Subject': subject, 'Position': pos, 'Impact': v, 'Factor': 'Electrode shift'})

        # 6. 可視化 (全体または被験者ごとにプロット)
        if not all_plot_data: return

        df_plot = pd.DataFrame(all_plot_data)
        
        plt.figure(figsize=(14, 6))
        sns.set(style="whitegrid")
        
        # Position x Factor の箱ひげ図
        ax = sns.boxplot(x='Position', y='Impact', hue='Factor', data=df_plot, 
                        palette="muted", showfliers=False, width=0.6)
        
        # 個別データのプロット (Stripplot)
        sns.stripplot(x='Position', y='Impact', hue='Factor', data=df_plot, 
                    dodge=True, jitter=True, color='black', alpha=0.3, size=3, ax=ax)

        # 凡例の調整 (重複を削除)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], labels[:2], title="Factor", loc='upper right')
        
        plt.ylabel("Impact Magnitude (Accuracy Change)")
        plt.title("Impact Analysis: Physiological Variation vs Electrode Shift (All Subjects Aggregated)")
        plt.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # ファイル名を適切に指定してください
    analyze_shift_impact_by_subject(pa_file='protonet_results.csv', base_file='pn_results.csv')