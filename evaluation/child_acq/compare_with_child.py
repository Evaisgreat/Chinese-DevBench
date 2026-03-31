

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from pathlib import Path

# ================== 配置 ==================
class Config:
    GPT2_AOA_FILE = Path("results/word_aoa_gpt2.csv")
    CHILD_AOA_FILE = Path("aoa2.xlsx") 
    OUTPUT_DIR = Path("results/comparison")
    
config = Config()
config.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


# ================== 分析函数 ==================

def compute_percentile_ranks(values):
    """计算 percentile ranks (0-100)"""
    ranks = pd.Series(values).rank(pct=True) * 100
    return ranks.values


def main():
    print("=" * 60)
    print("对比 GPT-2 和儿童的 AoA")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n[1] 加载数据...")
    gpt2_df = pd.read_csv(config.GPT2_AOA_FILE)
    child_df = pd.read_excel(config.CHILD_AOA_FILE)
    
    # 合并
    merged_df = gpt2_df.merge(child_df[['Name', 'AoA_o']], 
                             left_on='word', right_on='Name', how='inner')
    
    # 过滤掉 NaN
    merged_df = merged_df[~merged_df['aoa_log10'].isna() & ~merged_df['AoA_o'].isna()]

    
    print(f"  GPT-2 词数: {len(gpt2_df)}")
    print(f"  儿童词数: {len(child_df)}")
    print(f"  匹配词数: {len(merged_df)}")
    
    # 2. 转换为 percentile ranks
    print("\n[2] 计算 percentile ranks...")
    merged_df['gpt2_percentile'] = compute_percentile_ranks(merged_df['aoa_log10'])
    merged_df['child_percentile'] = compute_percentile_ranks(merged_df['AoA_o'])
    
    # 3. 相关性分析
    print("\n[3] 相关性分析...")
    
    # Pearson 相关（原始值）
    pearson_r, pearson_p = pearsonr(merged_df['aoa_log10'], merged_df['AoA_o'])
    print(f"  Pearson r (原始 AoA): {pearson_r:.3f} (p={pearson_p:.4f})")
    
    # Spearman 相关（percentile）
    spearman_r, spearman_p = spearmanr(merged_df['gpt2_percentile'], 
                                       merged_df['child_percentile'])
    print(f"  Spearman r (percentile): {spearman_r:.3f} (p={spearman_p:.4f})")
    
    # 4. 可视化
    print("\n[4] 生成可视化...")
    
    # 4.1 Scatter plot - Percentile ranks
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：Percentile ranks
    ax = axes[0]
    ax.scatter(merged_df['child_percentile'], merged_df['gpt2_percentile'], 
              alpha=0.6, s=50)
    ax.plot([0, 100], [0, 100], 'r--', alpha=0.5, label='y=x')
    ax.set_xlabel('Child AoA (percentile)', fontsize=12)
    ax.set_ylabel('GPT-2 AoA (percentile)', fontsize=12)
    ax.set_title(f'AoA Comparison (Percentile)\nSpearman r = {spearman_r:.3f}', 
                fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 右图：原始值
    ax = axes[1]
    ax.scatter(merged_df['AoA_o'], merged_df['aoa_log10'], alpha=0.6, s=50)
    ax.set_xlabel('Child AoA (years)', fontsize=12)
    ax.set_ylabel('GPT-2 AoA (log10 steps)', fontsize=12)
    ax.set_title(f'AoA Comparison (Raw)\nPearson r = {pearson_r:.3f}', 
                fontsize=14)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plot_path = config.OUTPUT_DIR / "aoa_comparison.png"
    plt.savefig(plot_path, dpi=150)
    print(f"  已保存: {plot_path}")
    plt.close()
    
    # 4.2 分布对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax = axes[0]
    ax.hist(merged_df['child_percentile'], bins=20, alpha=0.6, label='Children')
    ax.hist(merged_df['gpt2_percentile'], bins=20, alpha=0.6, label='GPT-2')
    ax.set_xlabel('AoA (percentile)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('AoA Distribution', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[1]
    ax.boxplot([merged_df['child_percentile'], merged_df['gpt2_percentile']], 
               labels=['Children', 'GPT-2'])
    ax.set_ylabel('AoA (percentile)', fontsize=12)
    ax.set_title('AoA Distribution (Boxplot)', fontsize=14)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    dist_path = config.OUTPUT_DIR / "aoa_distribution.png"
    plt.savefig(dist_path, dpi=150)
    print(f"  已保存: {dist_path}")
    plt.close()
    
    # 5. 保存对比结果
    print("\n[5] 保存对比数据...")
    output_csv = config.OUTPUT_DIR / "aoa_comparison.csv"
    merged_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"  已保存: {output_csv}")
    
    # 6. 找出差异最大的词
    print("\n[6] 差异最大的词...")
    merged_df['percentile_diff'] = abs(merged_df['gpt2_percentile'] - merged_df['child_percentile'])
    top_diff = merged_df.nlargest(10, 'percentile_diff')[['word', 'gpt2_percentile', 'child_percentile', 'percentile_diff']]
    print("\n差异最大的10个词:")
    print(top_diff.to_string(index=False))
    
    print("\n✅ 完成！")


if __name__ == "__main__":
    main()