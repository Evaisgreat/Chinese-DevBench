import os
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import jieba
import glob
import re
import stanza

# ==================== 配置 ====================
class Config:
    RESULTS_DIR = r"E:\Dev-cn\files\results"
    OUTPUT_DIR = r"E:\Dev-cn\files\results_seed"

# ==================== 1. 文本清洗 ====================
def clean_text(text):
    """清洗并只取第一个完整句子"""
    # 去除特殊token
    text = text.replace('<|endoftext|>', '。')
    text = text.replace('[SEP]', '')
    text = text.replace('[CLS]', '')
    
    # 取第一句
    sentences = re.split('[。！？\n]', text)
    first_sentence = sentences[0] if sentences else text
    
    # 去空格，只保留中文
    cleaned = ''.join([c for c in first_sentence.replace(' ', '') 
                       if '\u4e00' <= c <= '\u9fff'])
    
    return cleaned if len(cleaned) >= 2 else None

# ==================== 2. 句法分析器 ====================
class SyntaxAnalyzer:
    def __init__(self):
        print("Initializing Stanza for Chinese...")
        # stanza.download('zh')
        
        self.nlp = stanza.Pipeline('zh', processors='tokenize,pos,lemma,depparse')
        jieba.initialize()
        
        # 从属连词列表
        self.sconj_words = {
            '因为', '如果', '虽然', '所以', '但是', '可是', '当', '而', 
            '就', '才', '则', '只要', '除非', '假如', '尽管', '既然',
            '要是', '倘若', '以免', '以便', '由于', '为了'
        }
        
        # 并列连词列表
        self.cconj_words = {
            '和', '与', '或', '及', '以及', '而且', '并且', '还是',
            '或者', '不但', '而且', '又'
        }
    
    def analyze_corpus(self, texts):
        """分析所有文本的句法特征"""
        results = {
            'n_utterances': 0,
            'total_words': 0,
            'ccomp_count': 0,      # 从句作宾语
            'sconj_sent_count': 0, # 含从属连词的句子数
            'cconj_count': 0,      # 并列连词总数
            'sconj_count': 0,      # 从属连词总数
        }
        
        for text in tqdm(texts, desc="Syntax parsing"):
            if not text:
                continue
            
            try:
                # 清洗文本
                cleaned = clean_text(text)
                if not cleaned or len(cleaned) < 2:
                    continue
                
                # 用jieba分词统计词数
                words = jieba.lcut(cleaned)
                if len(words) < 2:
                    continue
                
                results['n_utterances'] += 1
                results['total_words'] += len(words)
                
                # Stanza句法分析
                doc = self.nlp(cleaned)
                

                for sent in doc.sentences:
                    # 检测ccomp（从句作宾语）
                    if self.has_ccomp(sent):
                        results['ccomp_count'] += 1
                    
                    # 检测连词（用jieba分词结果）
                    has_sconj = False
                    for word in words:
                        if word in self.sconj_words:
                            results['sconj_count'] += 1
                            has_sconj = True
                        
                        if word in self.cconj_words:
                            results['cconj_count'] += 1
                    
                    if has_sconj:
                        results['sconj_sent_count'] += 1
                
            except Exception as e:
                print(f"Error parsing: {e}")
                continue
        
        # 计算指标
        metrics = self.calculate_metrics(results)
        return results, metrics
    
    def has_ccomp(self, sent):
        """检测是否有ccomp（从句作宾语）依存关系 - 加强版"""
        try:
            ccomp_found = False
            ccomp_head_word = None
            
            for word in sent.words:
                if word.deprel == 'ccomp':
                    ccomp_found = True
                    # 获取支配词（主句动词）
                    if word.head > 0:
                        ccomp_head_word = sent.words[word.head - 1].text
                    break
            

            if ccomp_found and ccomp_head_word:
                cognitive_verbs = {
                    '说', '认为', '觉得', '知道', '相信', '希望', '想', 
                    '以为', '发现', '证明', '表明', '显示', '表示',
                    '担心', '害怕', '怀疑', '估计', '猜', '料'
                }
                return ccomp_head_word in cognitive_verbs
            
            return False
        except:
            return False
    def calculate_metrics(self, results):
        """计算句法复杂度指标"""
        n_utt = results['n_utterances']
        total_words = results['total_words']
        
        metrics = {
            'n_utterances': n_utt,
            'total_words': total_words,
            
            # 从句作宾语比例
            'ccomp_rate': (results['ccomp_count'] / n_utt * 100) if n_utt > 0 else 0,
            
            # 从属连词句比例
            'sconj_sent_rate': (results['sconj_sent_count'] / n_utt * 100) if n_utt > 0 else 0,
            
            # 并列连词密度（每百词）
            'cconj_per_100': (results['cconj_count'] / total_words * 100) if total_words > 0 else 0,
            
            # 从属连词密度（每百词）
            'sconj_per_100': (results['sconj_count'] / total_words * 100) if total_words > 0 else 0,
        }
        
        return metrics

# ==================== 3. CHILDES参考数据 ====================

def get_childes_reference():
    """CHILDES的句法复杂度数据（8个年龄段）"""
    return pd.DataFrame({
        'age_bin': ['<24', '24-30', '30-36', '36-42', '42-48', '48-60', '60-72', '72+'],
        'stage': ['Stage1', 'Stage1', 'Stage1', 'Stage1', 'Stage2', 'Stage2', 'Stage2', 'Stage2'],
        'ccomp_rate': [1.0, 1.0, 3.0, 3.0, 3.0, 5.0, 7.0, 11.0],
        'sconj_sent_rate': [5.0, 5.0, 9.0, 10.0, 10.0, 13.0, 17.0, 24.0],
        'cconj_per_100': [0.04, 0.02, 0.06, 0.05, 0.09, 0.14, 0.14, 0.18],
        'sconj_per_100': [2.35, 2.23, 3.04, 3.31, 3.33, 3.75, 4.50, 5.15],
    })


def get_hf_gpt2_reference():
    """HF-GPT2预训练模型的句法复杂度基线"""
    return {
        'ccomp_rate': 46.02,
        'sconj_sent_rate': 28.98,
        'cconj_per_100': 0.77,
        'sconj_per_100': 2.54,
    }

# ==================== 4. 主流程 ====================
def main():
    print("\n" + "=" * 70)
    print("Syntactic Complexity Analysis (Stanza)")
    print("=" * 70 + "\n")
    

    csv_pattern = os.path.join(Config.RESULTS_DIR, "gen_*.csv")
    csv_files = sorted(glob.glob(csv_pattern))
    
    if len(csv_files) == 0:
        print(f"❌ No CSV files found")
        return
    
    print(f"Found {len(csv_files)} CSV files\n")
    

    analyzer = SyntaxAnalyzer()
    all_results = []
    

    for csv_file in csv_files:
        print("\n" + "=" * 70)
        basename = os.path.basename(csv_file)
        print(f"Processing: {basename}")
        print("=" * 70)
        

        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        

        match = re.search(r'gen_(Stage\d+)_(.+)\.csv', basename)
        if match:
            stage = match.group(1)
            step = match.group(2)
        else:
            stage = 'Unknown'
            step = 'Unknown'
        
        checkpoint_name = f"{stage}_{step}"
        

        texts = df['text'].dropna().tolist()
        print(f"Analyzing {len(texts)} texts...")
        
        _, metrics = analyzer.analyze_corpus(texts)
        

        metrics['checkpoint'] = checkpoint_name
        metrics['stage'] = stage
        metrics['step'] = step
        
        all_results.append(metrics)
        
        
        print(f"\n✓ Summary:")
        print(f"  ccomp rate:      {metrics['ccomp_rate']:.2f}%")
        print(f"  sconj sent rate: {metrics['sconj_sent_rate']:.2f}%")
        print(f"  cconj/100 words: {metrics['cconj_per_100']:.2f}")
        print(f"  sconj/100 words: {metrics['sconj_per_100']:.2f}")
    

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
    

    output_csv = os.path.join(Config.OUTPUT_DIR, "syntax_analysis_results.csv")
    results_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n✓ Results saved to: {output_csv}")
    

    visualize_results(results_df)
    
    print_conclusions(results_df)
    
    print("\n" + "=" * 70)
    print("Syntactic Analysis Complete!")
    print("=" * 70)

# ==================== 5. 可视化 ====================
def visualize_results(df):
    print("\nGenerating visualization...")
    
    childes_ref = get_childes_reference()
    hf_gpt2_ref = get_hf_gpt2_reference()
    
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    x = range(len(df))
    

    def create_checkpoint_label(row):
        """将 Stage1_77 -> S1_C1, Stage1_final_model -> S1_C11 等"""
        stage = row['stage']
        step = row['step']
        

        stage_num = '1' if stage == 'Stage1' else '2'
        

        if step == 'final_model':
            # final_model 对应 C11
            checkpoint_num = '11'
        else:
            # 根据在该stage内的位置确定编号
            stage_df = df[df['stage'] == stage]
            stage_steps = stage_df[stage_df['step'] != 'final_model']['step'].tolist()
            
            if step in stage_steps:
                # C1-C10 对应该stage的前10个checkpoint
                checkpoint_num = str(stage_steps.index(step) + 1)
            else:
                checkpoint_num = step
        
        return f'S{stage_num}_C{checkpoint_num}'
    
    x_labels = [create_checkpoint_label(row) for _, row in df.iterrows()]
    
    # CHILDES参考点（8个年龄段）
    childes_x = np.linspace(0, len(df)-1, 8)
    
    # 1. ccomp rate
    ax = axes[0, 0]
    ax.plot(x, df['ccomp_rate'], marker='o', linewidth=2.5, markersize=8,
            color='#e74c3c', label='GPT-2')
    ax.plot(childes_x, childes_ref['ccomp_rate'], 'g*--', markersize=12,
            linewidth=2, label='CHILDES', alpha=0.7)
    ax.axhline(y=hf_gpt2_ref['ccomp_rate'], color='green', linestyle='--',
               linewidth=2, alpha=0.7, label='HF-GPT2', zorder=3)
    ax.set_title('Clausal Complement Rate (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rate (%)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2. sconj sentence rate
    ax = axes[0, 1]
    ax.plot(x, df['sconj_sent_rate'], marker='s', linewidth=2.5, markersize=8,
            color='#3498db', label='GPT-2')
    ax.plot(childes_x, childes_ref['sconj_sent_rate'], 'g*--', markersize=12,
            linewidth=2, label='CHILDES', alpha=0.7)
    ax.axhline(y=hf_gpt2_ref['sconj_sent_rate'], color='green', linestyle='--',
               linewidth=2, alpha=0.7, label='HF-GPT2', zorder=3)
    ax.set_title('Subordinating Conjunction Sentence Rate (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rate (%)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 3. cconj density
    ax = axes[1, 0]
    ax.plot(x, df['cconj_per_100'], marker='^', linewidth=2.5, markersize=8,
            color='#9b59b6', label='GPT-2')
    ax.plot(childes_x, childes_ref['cconj_per_100'], 'g*--', markersize=12,
            linewidth=2, label='CHILDES', alpha=0.7)
    ax.axhline(y=hf_gpt2_ref['cconj_per_100'], color='green', linestyle='--',
               linewidth=2, alpha=0.7, label='HF-GPT2', zorder=3)
    ax.set_title('Coordinating Conjunction Density (per 100 words)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 4. sconj density
    ax = axes[1, 1]
    ax.plot(x, df['sconj_per_100'], marker='v', linewidth=2.5, markersize=8,
            color='#f39c12', label='GPT-2')
    ax.plot(childes_x, childes_ref['sconj_per_100'], 'g*--', markersize=12,
            linewidth=2, label='CHILDES', alpha=0.7)
    ax.axhline(y=hf_gpt2_ref['sconj_per_100'], color='green', linestyle='--',
               linewidth=2, alpha=0.7, label='HF-GPT2', zorder=3)
    ax.set_title('Subordinating Conjunction Density (per 100 words)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    

    for ax in axes.flat:

        step = max(1, len(x) // 10)  
        ax.set_xticks(x[::step])
        ax.set_xticklabels([x_labels[i] for i in x[::step]], 
                          rotation=45, ha='right', fontsize=9)
        ax.set_xlabel('Checkpoint', fontsize=10)
    
    plt.suptitle('Syntactic Complexity Development: GPT-2 vs. CHILDES & HF-GPT2', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    output_png = os.path.join(Config.OUTPUT_DIR, "syntax_trends.png")
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_png}")
    plt.close()
# ==================== 6. 打印结论 ====================
def print_conclusions(df):
    print("\n" + "=" * 70)
    print("SYNTACTIC COMPLEXITY TRENDS")
    print("=" * 70)
    
    # 计算变化
    metrics = {
        'ccomp_rate': df['ccomp_rate'].iloc[-1] - df['ccomp_rate'].iloc[0],
        'sconj_sent_rate': df['sconj_sent_rate'].iloc[-1] - df['sconj_sent_rate'].iloc[0],
        'cconj_per_100': df['cconj_per_100'].iloc[-1] - df['cconj_per_100'].iloc[0],
        'sconj_per_100': df['sconj_per_100'].iloc[-1] - df['sconj_per_100'].iloc[0],
    }
    
    print("\n📊 Changes from first to last checkpoint:")
    print("-" * 70)
    print(f"  ccomp rate:          {metrics['ccomp_rate']:+6.2f}%  (expect: ↑)")
    print(f"    CHILDES: <24: 1.0% → 72+: 11.0%")
    match_1 = "✓ MATCH" if metrics['ccomp_rate'] > 0 else "✗ MISMATCH"
    print(f"    Result: {match_1}\n")
    
    print(f"  sconj sent rate:     {metrics['sconj_sent_rate']:+6.2f}%  (expect: ↑)")
    print(f"    CHILDES: <24: 5.0% → 72+: 24.0%")
    match_2 = "✓ MATCH" if metrics['sconj_sent_rate'] > 0 else "✗ MISMATCH"
    print(f"    Result: {match_2}\n")
    
    print(f"  cconj density:       {metrics['cconj_per_100']:+6.2f}   (expect: ↑)")
    print(f"    CHILDES: <24: 0.04 → 72+: 0.18")
    match_3 = "✓ MATCH" if metrics['cconj_per_100'] > 0 else "✗ MISMATCH"
    print(f"    Result: {match_3}\n")
    
    print(f"  sconj density:       {metrics['sconj_per_100']:+6.2f}   (expect: ↑)")
    print(f"    CHILDES: <24: 2.35 → 72+: 5.15")
    match_4 = "✓ MATCH" if metrics['sconj_per_100'] > 0 else "✗ MISMATCH"
    print(f"    Result: {match_4}")
    
    # 总结
    matches = [match_1, match_2, match_3, match_4]
    n_match = sum(1 for m in matches if "MATCH" in m)
    
    print("\n" + "=" * 70)
    print(f"SUMMARY: {n_match}/4 syntactic trends match CHILDES expectations")
    print("=" * 70)
    
    # 保存
    with open(os.path.join(Config.OUTPUT_DIR, "syntax_conclusions.txt"), 'w', encoding='utf-8') as f:
        f.write("Syntactic Complexity Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"ccomp rate:      {metrics['ccomp_rate']:+6.2f}%  {match_1}\n")
        f.write(f"sconj sent rate: {metrics['sconj_sent_rate']:+6.2f}%  {match_2}\n")
        f.write(f"cconj density:   {metrics['cconj_per_100']:+6.2f}   {match_3}\n")
        f.write(f"sconj density:   {metrics['sconj_per_100']:+6.2f}   {match_4}\n\n")
        f.write(f"Summary: {n_match}/4 match\n")

if __name__ == '__main__':
    main()