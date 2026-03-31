

import os
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import jieba
import jieba.posseg as pseg
import glob
import re

# ==================== 配置 ====================
class Config:
    RESULTS_DIR = r"/Volumes/Lexar/Dev-cn/files/results"
    OUTPUT_DIR = r"/Volumes/Lexar/Dev-cn/files/results"

# ==================== 1. 重新分词 ====================
def clean_and_reseg(text):
    # 去特殊token
    text = text.replace('<|endoftext|>', '。')  
    
    # 取第一句
    sentences = re.split('[。！？\n]', text)
    text = sentences[0] if sentences else text
    

    text = ''.join([c for c in text.replace(' ', '') if '\u4e00' <= c <= '\u9fff'])
    

    words = jieba.lcut(text)
    return [w for w in words if len(w) > 0]
# ==================== 2. 文本分析器 ====================
class ReSegAnalyzer:
    def __init__(self):
        print("Initializing jieba...")
        jieba.initialize()
        
    def analyze_corpus(self, texts):
        results = {
            'n_utterances': 0,
            'total_words': 0,
            'word_length': Counter(),
            'pos_count': Counter(),
            'all_words': []  
        }
        
        for text in tqdm(texts, desc="Re-segmenting & Analyzing"):
            if not text.strip():
                continue
            
            try:
                # 重新分词
                words = clean_and_reseg(text)
                
                if len(words) == 0:
                    continue
                
                results['n_utterances'] += 1
                results['total_words'] += len(words)
                
                # 词性标注
                for word in words:
                    # 词长统计
                    word_len = len(word)
                    results['word_length'][word_len] += 1
                    results['all_words'].append(word)
                    
                    # 词性标注
                    word_pos = pseg.lcut(word)
                    if word_pos:
                        pos = word_pos[0].flag
                        pos_cat = self.map_pos_category(pos)
                        results['pos_count'][pos_cat] += 1
                
            except Exception as e:
                continue
        
        metrics = self.calculate_metrics(results)
        return results, metrics
    
    def calculate_metrics(self, results):
        total_words = results['total_words']
        
        metrics = {
            'n_utterances': results['n_utterances'],
            'total_words': total_words
        }
        
        # 词长分布
        for length in [1, 2, 3, 4, 5, 6]:
            count = results['word_length'].get(length, 0)
            metrics[f'len_{length}_pct'] = (count / total_words * 100) if total_words > 0 else 0
        
        # 3+音节词
        count_3plus = sum(results['word_length'].get(i, 0) for i in range(3, 7))
        metrics['len_3plus_pct'] = (count_3plus / total_words * 100) if total_words > 0 else 0
        
        # 词性分布
        for pos_cat in ['Noun', 'Verb', 'Adjective', 'Adverb', 'Pronoun', 'Proper_noun']:
            count = results['pos_count'].get(pos_cat, 0)
            metrics[f'pos_{pos_cat.lower()}_pct'] = (count / total_words * 100) if total_words > 0 else 0
        
        return metrics
    
    @staticmethod
    def map_pos_category(jieba_pos):
        pos_map = {
            'n': 'Noun', 'nr': 'Proper_noun', 'ns': 'Noun', 'nt': 'Noun', 'nz': 'Noun',
            'v': 'Verb', 'vn': 'Verb', 'vd': 'Verb', 'vshi': 'Verb',
            'a': 'Adjective', 'ad': 'Adjective', 'an': 'Adjective',
            'd': 'Adverb', 'df': 'Adverb',
            'r': 'Pronoun', 'rr': 'Pronoun', 'rz': 'Pronoun',
        }
        if jieba_pos in pos_map:
            return pos_map[jieba_pos]
        if jieba_pos and jieba_pos[0] in pos_map:
            return pos_map[jieba_pos[0]]
        return 'Others'

# ==================== 3. 论文参考数据 ====================

def get_paper_reference():
    return {
        'K1': {
            'len_1': 17.48, 'len_2': 66.17, 'len_3': 14.36, 'len_3plus': 16.13,
            'noun': 43.03, 'verb': 29.22, 'adj': 6.03, 'adv': 4.39
        },
        'K2': {
            'len_1': 14.12, 'len_2': 67.42, 'len_3': 15.74, 'len_3plus': 18.12,
            'noun': 44.21, 'verb': 28.34, 'adj': 5.78, 'adv': 3.63
        },
        'K3': {
            'len_1': 10.94, 'len_2': 67.17, 'len_3': 18.81, 'len_3plus': 21.47,
            'noun': 44.71, 'verb': 27.02, 'adj': 5.96, 'adv': 3.47
        },
        # 新增 HF-GPT2 数据
        'HF_GPT2': {
            'len_1': 46.94, 'len_2': 47.54, 'len_3': 4.40, 'len_3plus': 5.48,
            'noun': 19.44, 'verb': 20.60, 'adj': 5.24, 'adv': 9.44
        }
    }

# ==================== 4. 主流程 ====================
def main():
    print("\n" + "=" * 70)
    print("Re-segmentation Analysis (Character-level → Word-level)")
    print("=" * 70 + "\n")
    
    
    csv_pattern = os.path.join(Config.RESULTS_DIR, "gen_*.csv")
    csv_files = sorted(glob.glob(csv_pattern))
    
    if len(csv_files) == 0:
        print(f"❌ No CSV files found in {Config.RESULTS_DIR}")
        return
    
    print(f"Found {len(csv_files)} CSV files\n")
    

    analyzer = ReSegAnalyzer()
    all_results = []
    

    for csv_file in csv_files:
        print("\n" + "=" * 70)
        basename = os.path.basename(csv_file)
        print(f"Processing: {basename}")
        print("=" * 70)
        

        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        
        # 提取checkpoint信息
        match = re.search(r'gen_(Stage\d+)_(.+)\.csv', basename)
        if match:
            stage = match.group(1)
            step = match.group(2)
        else:
            stage = 'Unknown'
            step = 'Unknown'
        
        checkpoint_name = f"{stage}_{step}"
        

        texts = df['text'].dropna().tolist()
        print(f"Processing {len(texts)} texts...")
        

        raw_results, metrics = analyzer.analyze_corpus(texts)
        

        print(f"\n✓ Re-segmented sample (first 10 words):")
        sample_words = raw_results['all_words'][:10]
        print(f"  {' / '.join(sample_words)}")
        

        metrics['checkpoint'] = checkpoint_name
        metrics['stage'] = stage
        metrics['step'] = step
        
        all_results.append(metrics)
    

    results_df = pd.DataFrame(all_results)
    

    def sort_key(row):
        stage_order = 0 if row['stage'] == 'Stage1' else 1
        if row['step'] == 'final_model':
            step_order = 99999
        else:
            try:
                step_order = int(row['step'])
            except:
                step_order = 99999
        return (stage_order, step_order)
    
    results_df['sort_key'] = results_df.apply(sort_key, axis=1)
    results_df = results_df.sort_values('sort_key').drop('sort_key', axis=1)
    results_df = results_df.reset_index(drop=True)
    
    # 保存
    output_csv = os.path.join(Config.OUTPUT_DIR, "reseg_analysis_results.csv")
    results_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n✓ Results saved to: {output_csv}")
    

    visualize_results(results_df)
    

    print_conclusions(results_df)
    
    print("\n" + "=" * 70)
    print("Re-segmentation Analysis Complete!")
    print("=" * 70)

# ==================== 5. 可视化 ====================

def visualize_results(df):
    print("\nGenerating visualization...")
    
    paper_ref = get_paper_reference()
    
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    
    fig = plt.figure(figsize=(15, 22), constrained_layout=True)
    gs = fig.add_gridspec(5, 2)
    
    x = range(len(df))
    

    def create_checkpoint_label(row):
        """将 Stage1_77 -> S1_C1, Stage1_final_model -> S1_C11 等"""
        stage = row['stage']
        step = row['step']
        

        stage_num = '1' if stage == 'Stage1' else '2'
        
        # 确定checkpoint编号
        if step == 'final_model':
            # final_model 对应 C11
            checkpoint_num = '11'
        else:
            stage_df = df[df['stage'] == stage]
            stage_steps = stage_df[stage_df['step'] != 'final_model']['step'].tolist()
            
            if step in stage_steps:
                # C1-C10 对应该stage的前10个checkpoint
                checkpoint_num = str(stage_steps.index(step) + 1)
            else:
                checkpoint_num = step
        
        return f'S{stage_num}_C{checkpoint_num}'
    
    x_labels = [create_checkpoint_label(row) for _, row in df.iterrows()]
    
    n_points = len(df)
    k_positions = [0, n_points // 2, n_points - 1]
    
    # 1. 1-syllable
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, df['len_1_pct'], marker='o', linewidth=2.5, markersize=8, 
             color='#e74c3c', label='child-trained GPT-2')
    ax1.scatter(k_positions, 
                [paper_ref['K1']['len_1'], paper_ref['K2']['len_1'], paper_ref['K3']['len_1']], 
                s=200, color='#2ecc71', marker='*', zorder=5, 
                label='CPCSLD (K1/K2/K3)', edgecolors='black', linewidths=1.5)
    ax1.axhline(y=paper_ref['HF_GPT2']['len_1'], color='green', linestyle='--', 
                linewidth=2, alpha=0.7, label='HF-GPT2', zorder=3)
    ax1.set_title('1-Syllable Words (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Percentage (%)', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 2. 2-syllable
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(x, df['len_2_pct'], marker='s', linewidth=2.5, markersize=8, 
             color='#3498db', label='child-trained GPT-2')
    ax2.scatter(k_positions,
                [paper_ref['K1']['len_2'], paper_ref['K2']['len_2'], paper_ref['K3']['len_2']],
                s=200, color='#2ecc71', marker='*', zorder=5,
                label='CPCSLD (K1/K2/K3)', edgecolors='black', linewidths=1.5)
    ax2.axhline(y=paper_ref['HF_GPT2']['len_2'], color='green', linestyle='--',
                linewidth=2, alpha=0.7, label='HF-GPT2', zorder=3)
    ax2.set_title('2-Syllable Words (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Percentage (%)', fontsize=12)
    ax2.legend(fontsize=12, loc='center right')
    ax2.grid(True, alpha=0.3)
    
    # 3. 3+ syllable
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(x, df['len_3plus_pct'], marker='^', linewidth=2.5, markersize=8,
             color='#9b59b6', label='child-trained GPT-2')
    ax3.scatter(k_positions,
                [paper_ref['K1']['len_3plus'], paper_ref['K2']['len_3plus'], paper_ref['K3']['len_3plus']],
                s=200, color='#2ecc71', marker='*', zorder=5,
                label='CPCSLD (K1/K2/K3)', edgecolors='black', linewidths=1.5)
    ax3.axhline(y=paper_ref['HF_GPT2']['len_3plus'], color='green', linestyle='--',
                linewidth=2, alpha=0.7, label='HF-GPT2', zorder=3)
    ax3.set_title('3+ Syllable Words (%)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Percentage (%)', fontsize=12)
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 4. Noun
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(x, df['pos_noun_pct'], marker='o', linewidth=2.5, markersize=8,
             color='#f39c12', label='child-trained GPT-2')
    ax4.scatter(k_positions,
                [paper_ref['K1']['noun'], paper_ref['K2']['noun'], paper_ref['K3']['noun']],
                s=200, color='#2ecc71', marker='*', zorder=5, label='CPCSLD (K1/K2/K3)', edgecolors='black', linewidths=1.5)
    ax4.axhline(y=paper_ref['HF_GPT2']['noun'], color='green', linestyle='--',
                linewidth=2, alpha=0.7, label='HF-GPT2', zorder=3)
    ax4.set_title('Noun (%)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Percentage (%)', fontsize=12)
    ax4.legend(fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # 5. Verb
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(x, df['pos_verb_pct'], marker='s', linewidth=2.5, markersize=8,
             color='#1abc9c', label='child-trained GPT-2')
    ax5.scatter(k_positions,
                [paper_ref['K1']['verb'], paper_ref['K2']['verb'], paper_ref['K3']['verb']],
                s=200, color='#2ecc71', marker='*', zorder=5, label='CPCSLD (K1/K2/K3)', edgecolors='black', linewidths=1.5)
    ax5.axhline(y=paper_ref['HF_GPT2']['verb'], color='green', linestyle='--',
                linewidth=2, alpha=0.7, label='HF-GPT2', zorder=3)
    ax5.set_title('Verb (%)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Percentage (%)', fontsize=12)
    ax5.legend(fontsize=12)
    ax5.grid(True, alpha=0.3)
    
    # 6. Adjective
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(x, df['pos_adjective_pct'], marker='^', linewidth=2.5, markersize=8,
             color='#e67e22', label='child-trained GPT-2')
    ax6.scatter(k_positions,
                [paper_ref['K1']['adj'], paper_ref['K2']['adj'], paper_ref['K3']['adj']],
                s=200, color='#2ecc71', marker='*', zorder=5, label='CPCSLD (K1/K2/K3)', edgecolors='black', linewidths=1.5)
    ax6.axhline(y=paper_ref['HF_GPT2']['adj'], color='green', linestyle='--',
                linewidth=2, alpha=0.7, label='HF-GPT2', zorder=3)
    ax6.set_title('Adjective (%)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Percentage (%)', fontsize=12)
    ax6.legend(fontsize=12)
    ax6.grid(True, alpha=0.3)
    
    # 7. Adverb
    ax7 = fig.add_subplot(gs[3, 0])
    ax7.plot(x, df['pos_adverb_pct'], marker='v', linewidth=2.5, markersize=8,
             color='#34495e', label='child-trained GPT-2')
    ax7.scatter(k_positions,
                [paper_ref['K1']['adv'], paper_ref['K2']['adv'], paper_ref['K3']['adv']],
                s=200, color='#2ecc71', marker='*', zorder=5, label='CPCSLD (K1/K2/K3)', edgecolors='black', linewidths=1.5)
    ax7.axhline(y=paper_ref['HF_GPT2']['adv'], color='green', linestyle='--',
                linewidth=2, alpha=0.7, label='HF-GPT2', zorder=3)
    ax7.set_title('Adverb (%)', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Percentage (%)', fontsize=12)
    ax7.legend(fontsize=12)
    ax7.grid(True, alpha=0.3)
    
    # 8. Word Length Overview
    ax8 = fig.add_subplot(gs[3, 1])
    ax8.plot(x, df['len_1_pct'], marker='o', linewidth=2, markersize=6, label='1-syl', alpha=0.8)
    ax8.plot(x, df['len_2_pct'], marker='s', linewidth=2, markersize=6, label='2-syl', alpha=0.8)
    ax8.plot(x, df['len_3plus_pct'], marker='^', linewidth=2, markersize=6, label='3+-syl', alpha=0.8)
    ax8.set_title('Word Length Overview', fontsize=12, fontweight='bold')
    ax8.set_ylabel('Percentage (%)', fontsize=12)
    ax8.legend(fontsize=12)
    ax8.grid(True, alpha=0.3)
    
    # 9. POS Overview
    ax9 = fig.add_subplot(gs[4, 0])
    ax9.plot(x, df['pos_noun_pct'], marker='o', linewidth=2, markersize=6, label='Noun', alpha=0.8)
    ax9.plot(x, df['pos_verb_pct'], marker='s', linewidth=2, markersize=6, label='Verb', alpha=0.8)
    ax9.plot(x, df['pos_adjective_pct'], marker='^', linewidth=2, markersize=6, label='Adj', alpha=0.8)
    ax9.plot(x, df['pos_adverb_pct'], marker='v', linewidth=2, markersize=6, label='Adv', alpha=0.8)
    ax9.set_title('POS Overview', fontsize=12, fontweight='bold')
    ax9.set_ylabel('Percentage (%)', fontsize=12)
    ax9.legend(fontsize=12, ncol=2)
    ax9.grid(True, alpha=0.3)
    

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
        step = max(1, len(x) // 10)  # 最多显示10-12个标签
        ax.set_xticks(x[::step])
        ax.set_xticklabels([x_labels[i] for i in x[::step]], 
                          rotation=45, ha='right', fontsize=12)
        ax.set_xlabel('Checkpoint', fontsize=12)
    
    fig.suptitle(
        'GPT-2 Developmental Trends vs. CPCSLD & HF-GPT2',
        fontsize=15,
        fontweight='bold',
        y=1.03
    )
    
    plt.subplots_adjust(top=0.90)
    
    output_png = os.path.join(Config.OUTPUT_DIR, "cihui.png")
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    output_pdf = os.path.join(Config.OUTPUT_DIR, "cihui.pdf")
    plt.savefig(output_pdf, bbox_inches="tight")
    print(f"\u2713 PDF saved to: {output_pdf}")
    print(f"✓ Visualization saved to: {output_png}")
    plt.close()
# ==================== 6. 打印结论 ====================
def print_conclusions(df):
    print("\n" + "=" * 70)
    print("QUALITATIVE CONCLUSIONS (Re-segmented)")
    print("=" * 70)
    
    print("\n📊 WORD LENGTH TRENDS:")
    print("-" * 70)
    
    change_1 = df['len_1_pct'].iloc[-1] - df['len_1_pct'].iloc[0]
    change_2 = df['len_2_pct'].iloc[-1] - df['len_2_pct'].iloc[0]
    change_3 = df['len_3plus_pct'].iloc[-1] - df['len_3plus_pct'].iloc[0]
    
    match_1 = "✓ MATCH" if change_1 < 0 else "✗ MISMATCH"
    match_2 = "✓ MATCH" if abs(change_2) < 2 else "✗ MISMATCH"
    match_3 = "✓ MATCH" if change_3 > 0 else "✗ MISMATCH"
    
    print(f"  1-syllable:   {change_1:+6.2f}%  (expect: ↓)  {match_1}")
    print(f"  2-syllable:   {change_2:+6.2f}%  (expect: →)  {match_2}")
    print(f"  3+ syllable:  {change_3:+6.2f}%  (expect: ↑)  {match_3}")
    
    print("\n\n📊 POS TRENDS:")
    print("-" * 70)
    
    change_noun = df['pos_noun_pct'].iloc[-1] - df['pos_noun_pct'].iloc[0]
    change_verb = df['pos_verb_pct'].iloc[-1] - df['pos_verb_pct'].iloc[0]
    change_adj = df['pos_adjective_pct'].iloc[-1] - df['pos_adjective_pct'].iloc[0]
    change_adv = df['pos_adverb_pct'].iloc[-1] - df['pos_adverb_pct'].iloc[0]
    
    match_noun = "✓ MATCH" if change_noun > 0 else "✗ MISMATCH"
    match_verb = "✓ MATCH" if change_verb < 0 else "✗ MISMATCH"
    match_adj = "✓ MATCH" if change_adj < 0 else "✗ MISMATCH"
    match_adv = "✓ MATCH" if change_adv < 0 else "✗ MISMATCH"
    
    print(f"  Noun:       {change_noun:+6.2f}%  (expect: ↑)  {match_noun}")
    print(f"  Verb:       {change_verb:+6.2f}%  (expect: ↓)  {match_verb}")
    print(f"  Adjective:  {change_adj:+6.2f}%  (expect: ↓)  {match_adj}")
    print(f"  Adverb:     {change_adv:+6.2f}%  (expect: ↓)  {match_adv}")
    
    matches = [match_1, match_3, match_noun, match_verb, match_adj, match_adv]
    n_match = sum(1 for m in matches if "MATCH" in m)
    
    print("\n" + "=" * 70)
    print(f"SUMMARY: {n_match}/6 trends match (Re-segmented)")
    print("=" * 70)
    
    # 保存
    with open(os.path.join(Config.OUTPUT_DIR, "reseg_conclusions.txt"), 'w', encoding='utf-8') as f:
        f.write("Re-segmentation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"1-syllable:  {change_1:+6.2f}%  {match_1}\n")
        f.write(f"3+-syllable: {change_3:+6.2f}%  {match_3}\n")
        f.write(f"Noun:        {change_noun:+6.2f}%  {match_noun}\n")
        f.write(f"Verb:        {change_verb:+6.2f}%  {match_verb}\n")
        f.write(f"Adjective:   {change_adj:+6.2f}%  {match_adj}\n")
        f.write(f"Adverb:      {change_adv:+6.2f}%  {match_adv}\n\n")
        f.write(f"Summary: {n_match}/6 match\n")

if __name__ == '__main__':
    main()