import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings

warnings.filterwarnings('ignore')

def load_and_combine_data(file_paths):
    """
    複数のCSVファイルを読み込んで統合する
    file_paths: 辞書 {'MethodName': 'path/to/csv'}
    """
    combined_df = pd.DataFrame()
    
    for method_label, path in file_paths.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            # もしCSV内に'Method'列がなければラベルを追加/上書き
            df['Method'] = method_label
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        else:
            print(f"Warning: File not found: {path}")
            
    return combined_df

def analyze_method_comparison(df, methods_to_compare, positions):
    """
    指定された2つの手法間で、各電極位置ごとにペアのあるt検定を行う
    """
    print("\n" + "="*60)
    print(f"Statistical Test: Method Comparison ({methods_to_compare[0]} vs {methods_to_compare[1]})")
    print("="*60)
    
    m1_name, m2_name = methods_to_compare
    
    for pos in positions:
        # データの抽出（被験者・Fold順にソートしてペアを作る）
        df_pos = df[df['Position'] == pos]
        
        data_m1 = df_pos[df_pos['Method'] == m1_name].sort_values(by=['Subject', 'Fold'])
        data_m2 = df_pos[df_pos['Method'] == m2_name].sort_values(by=['Subject', 'Fold'])
        
        acc1 = data_m1['Accuracy'].values
        acc2 = data_m2['Accuracy'].values
        
        # サンプル数チェック
        if len(acc1) != len(acc2) or len(acc1) == 0:
            print(f"Position: {pos:15s} | Error: Sample sizes do not match or are zero.")
            continue
            
        # 平均値
        mean1 = np.mean(acc1)
        mean2 = np.mean(acc2)
        
        # 対応のあるt検定
        t_stat, p_val = ttest_rel(acc1, acc2)
        
        # 有意水準の判定
        sig = ""
        if p_val < 0.001: sig = "***"
        elif p_val < 0.01: sig = "**"
        elif p_val < 0.05: sig = "*"
        else: sig = "ns"
        
        print(f"Position: {pos:15s} | {m1_name}: {mean1:.4f} vs {m2_name}: {mean2:.4f} | p-value: {p_val:.4e} ({sig})")

def analyze_position_comparison(df, methods):
    """
    各手法内で、電極位置間の多重比較（Tukey HSD）を行う
    """
    print("\n" + "="*60)
    print("Statistical Test: Position Comparison (Tukey HSD)")
    print("="*60)
    
    for method in methods:
        print(f"\n>>> Method: {method}")
        df_method = df[df['Method'] == method]
        
        if df_method.empty:
            print("  No data.")
            continue
            
        try:
            tukey = pairwise_tukeyhsd(endog=df_method['Accuracy'], 
                                      groups=df_method['Position'], 
                                      alpha=0.05)
            print(tukey)
        except Exception as e:
            print(f"  Error in Tukey HSD: {e}")

def plot_results(df):
    """
    結果の可視化（箱ひげ図）
    """
    plt.figure(figsize=(12, 6))
    
    # スタイル設定
    sns.set(style="whitegrid")
    
    # 箱ひげ図
    ax = sns.boxplot(x='Position', y='Accuracy', hue='Method', data=df, showfliers=False)
    # データ点を重ねて表示（ストリッププロット）
    sns.stripplot(x='Position', y='Accuracy', hue='Method', data=df, 
                  dodge=True, jitter=True, color='black', alpha=0.5, size=4, ax=ax)
    
    ax.set_xticklabels(['original2', 'trans5', 'trans10', 'rotation'], fontsize=18)
    
    # 凡例の調整（重複削除）
    handles, labels = ax.get_legend_handles_labels()
    # 前半半分だけ使用（boxplotとstripplotで重複するため）
    l = len(handles)//2
    ax.legend(handles[0:l], labels[0:l], title="Method", loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=18)
    
    # plt.title('Accuracy Comparison by Position and Method')
    plt.tick_params(labelsize=18)
    plt.ylim(0, 1.05)
    plt.ylabel('Classification Accuracy', fontsize=18)
    plt.xlabel('Electrode Position', fontsize=18)
    plt.tight_layout()
    plt.show()

from scipy.stats import dunnett
import pandas as pd
import numpy as np

def analyze_position_dunnett(df, methods, control_position='original2'):
    """
    各手法内で、指定したコントロール位置(original2)に対するDunnettの検定を行う
    """
    print("\n" + "="*80)
    print(f"Statistical Test: Position Comparison (Dunnett's Test vs '{control_position}')")
    print("="*80)
    
    for method in methods:
        print(f"\n>>> Method: {method}")
        df_method = df[df['Method'] == method]
        
        # コントロール群のデータ抽出
        control_data = df_method[df_method['Position'] == control_position]['Accuracy'].values
        
        if len(control_data) == 0:
            print(f"  Warning: Control position '{control_position}' not found.")
            continue

        # 比較対象群のデータ抽出
        # Positionがcontrol_positionではないものをユニークに取得
        other_positions = [p for p in df_method['Position'].unique() if p != control_position]
        
        # Dunnett検定用にデータをリストにまとめる
        samples = []
        position_labels = []
        
        for pos in other_positions:
            data = df_method[df_method['Position'] == pos]['Accuracy'].values
            if len(data) > 0:
                samples.append(data)
                position_labels.append(pos)
        
        if not samples:
            print("  No comparison groups found.")
            continue

        # --- Dunnett検定の実行 ---
        # samples: 比較したい群のデータのリスト
        # control: 基準群のデータ
        try:
            res = dunnett(*samples, control=control_data)
            
            # 結果の表示
            print(f"{'Comparison':<30} | {'Diff':<10} | {'p-value':<12} | {'Sig.'}")
            print("-" * 70)
            
            control_mean = np.mean(control_data)
            
            for i, pos in enumerate(position_labels):
                # 平均値の差
                comp_mean = np.mean(samples[i])
                diff = comp_mean - control_mean
                
                # p値
                p_val = res.pvalue[i]
                
                # 有意判定
                if p_val < 0.001: sig = "***"
                elif p_val < 0.01: sig = "**"
                elif p_val < 0.05: sig = "*"
                else: sig = "ns"  # Not Significant (有意差なし = 頑健性あり)
                
                print(f"{control_position} vs {pos:<18} | {diff:+.4f}    | {p_val:.4e}   | {sig}")
                
        except AttributeError:
            print("Error: 'scipy.stats.dunnett' not found. Please upgrade scipy (pip install --upgrade scipy).")
        except Exception as e:
            print(f"Error during Dunnett test: {e}")
    

# ==========================================
# Main Analysis Script
# ==========================================
import os

if __name__ == "__main__":
    # 1. 読み込むファイル（前のステップで保存したファイル名を指定）
    # 例として3つのファイルを想定していますが、存在するファイルのみでOKです
    files = {
        'PN': 'pn_results.csv',
        # 'Align+Baseline': 'align_pn_results.csv',
        'PA': 'protonet_results.csv',
        # 'Align + PA':    'filtered_align_protonet_results.csv',
        # 'PN(IntraSession)': 'pn_intrasession_results.csv'
        # 'LDA':'lda_results.csv',
        # 'Alignment + LDA': 'filtered_align_lda_results.csv',
        # 'PA': 'protonet_results.csv',
        # # 'Alignment + PA': 'align_pa_results.csv',
        # 'LDA': 'lda_results.csv',
        # 'AdaBN': 'adabn_results.csv',
        # '2SRNN': '2srnn_results.csv'
    }
    
    # 2. データのロード
    df_all = load_and_combine_data(files)
    
    if df_all.empty:
        print("No data found. Please run the experiments first.")
    else:
        # 電極位置の順序を定義（グラフ表示順）
        positions = ["original2", "downleft5mm", "downleft10mm", "clockwise"]
        # データに含まれる位置のみにフィルタリング
        positions = [p for p in positions if p in df_all['Position'].unique()]
        
        # 3. 統計分析
        # 例: ProtoNet (Adaptive) vs LDAの比較
        if 'ProtoNet (Adaptive)' in df_all['Method'].unique() and 'LDA (SourceOnly)' in df_all['Method'].unique():
            analyze_method_comparison(df_all, ['ProtoNet (Adaptive)', 'LDA (SourceOnly)'], positions)
        # if 'ProtoNet (Adaptive)' in df_all['Method'].unique() and 'LDA (SourceOnly)' in df_all['Method'].unique():
        #     analyze_position_dunnett(df_all, ['ProtoNet (Adaptive)', 'LDA (SourceOnly)'], positions)

        # 例: Align + ProtoNet vs LDAの比較
        if 'Align + ProtoNet' in df_all['Method'].unique() and 'LDA (SourceOnly)' in df_all['Method'].unique():
            analyze_method_comparison(df_all, ['Align + ProtoNet', 'LDA (SourceOnly)'], positions)
            
        # 例: Align+LDA vs LDAの比較
        if 'Align + LDA' in df_all['Method'].unique() and 'LDA (SourceOnly)' in df_all['Method'].unique():
            analyze_method_comparison(df_all, ['Align + LDA', 'LDA (SourceOnly)'], positions)

        if 'PA' in df_all['Method'].unique() and 'Baseline' in df_all['Method'].unique():
            analyze_method_comparison(df_all, ['PA', 'Baseline'], positions)
        
        if 'Align+Baseline' in df_all['Method'].unique() and 'Baseline' in df_all['Method'].unique():
            analyze_method_comparison(df_all, ['Align+Baseline', 'Baseline'], positions)
        
        if 'Align + PA' in df_all['Method'].unique() and 'Baseline' in df_all['Method'].unique():
            analyze_method_comparison(df_all, ['Align + PA', 'Baseline'], positions)
        
        if 'df_all' in locals() and not df_all.empty:
            analyze_position_dunnett(df_all, df_all['Method'].unique(), control_position='original2')

        # 位置間の比較（全手法）
        analyze_position_comparison(df_all, df_all['Method'].unique())
        
        # 4. 可視化
        plot_results(df_all)