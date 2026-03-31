import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import jieba
import stanza
from collections import defaultdict

# ==================== 配置 ====================

class Config:
    INPUT_CSV = r"D:\A-UOA\Wang\child production\child_utterances_with_age_only_norm.csv"
    OUTPUT_DIR = r"E:\Dev-cn\files"
    

    AGE_BINS = [
        (0, 24, '<24'),
        (24, 30, '24-30'),
        (30, 36, '30-36'),
        (36, 42, '36-42'),
        (42, 48, '42-48'),
        (48, 60, '48-60'),
        (60, 72, '60-72'),
        (72, 999, '72+'),
    ]

# ==================== 1. 句法分析器 ====================
class ChildSyntaxAnalyzer:
    def __init__(self):
        print("Initializing Stanza for Chinese...")
        # stanza.download('zh')  
        
        self.nlp = stanza.Pipeline('zh', processors='tokenize,pos,lemma,depparse')
        jieba.initialize()
        
        # 从属连词
        self.sconj_words = {
            '因为', '如果', '虽然', '所以', '但是', '可是', '当', '而', 
            '就', '才', '则', '只要', '除非', '假如', '尽管', '既然',
            '要是', '倘若', '以免', '以便', '由于', '为了'
        }
        
        # 并列连词
        self.cconj_words = {
            '和', '与', '或', '及', '以及', '而且', '并且', '还是',
            '或者', '不但', '又'
        }
    
    def analyze_age_group(self, utterances):
        """分析一个年龄组的所有语句"""
        
        results = {
            # 基础统计
            'n_utterances': 0,
            'total_words': 0,
            'total_turns': 0,      # 话轮数（按file分组）
            'unique_words': set(),  # 不同词
            
            # 从句
            'ccomp_count': 0,      # 从句作宾语
            'xcomp_count': 0,      # 开放式补语从句
            'csubj_count': 0,      # 从句作主语
            'acl_count': 0,        # 定语从句
            
            # 句法结构
            'ba_count': 0,
            'bei_count': 0,
            'svc_count': 0,
            'pivot_count': 0,
            
            # 连词
            'cconj_count': 0,
            'sconj_count': 0,
            'cconj_sent_count': 0,
            'sconj_sent_count': 0,
        }
        

        files_seen = set()
        
        for utt in tqdm(utterances, desc="Analyzing utterances"):
            # 跳过空utterance或无汉字的
            if pd.isna(utt['utterance_norm']) or not utt['utterance_norm_has_hanzi']:
                continue
            
            text = str(utt['utterance_norm']).strip()
            if len(text) < 1:
                continue
            
            # 统计话轮
            file_path = utt['file_path']
            if file_path not in files_seen:
                results['total_turns'] += 1
                files_seen.add(file_path)
            
            try:
                # jieba分词
                # 去空格后用jieba重新分词
                text_no_space = text.replace(' ', '')
                words = jieba.lcut(text_no_space)
                words = [w for w in words if len(w.strip()) > 0]
                
                if len(words) < 1:
                    continue
                
                results['n_utterances'] += 1
                results['total_words'] += len(words)
                results['unique_words'].update(words)
                

                doc = self.nlp(text)
                
                # 连词句级别标记
                has_cconj_in_utt = False
                has_sconj_in_utt = False
                
                for sent in doc.sentences:
                    # 从句
                    if self.has_ccomp(sent):
                        results['ccomp_count'] += 1
                    
                    if self.has_xcomp(sent):
                        results['xcomp_count'] += 1
                    
                    if self.has_csubj(sent):
                        results['csubj_count'] += 1
                    
                    if self.has_acl(sent):
                        results['acl_count'] += 1
                    
                    # 句法结构
                    if self.has_ba_construction(sent):
                        results['ba_count'] += 1
                    
                    if self.has_bei_construction(sent):
                        results['bei_count'] += 1
                    
                    if self.has_svc(sent):
                        results['svc_count'] += 1
                    
                    if self.has_pivot_construction(sent):
                        results['pivot_count'] += 1
                
                # 连词检测
                for word in words:
                    if word in self.sconj_words:
                        results['sconj_count'] += 1
                        has_sconj_in_utt = True
                    
                    if word in self.cconj_words:
                        results['cconj_count'] += 1
                        has_cconj_in_utt = True
                
                if has_cconj_in_utt:
                    results['cconj_sent_count'] += 1
                if has_sconj_in_utt:
                    results['sconj_sent_count'] += 1
                
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        # 计算指标
        metrics = self.calculate_metrics(results)
        return metrics
    
    def has_ccomp(self, sent):
        """从句作宾语"""
        for word in sent.words:
            if word.deprel == 'ccomp':
                return True
        return False
    
    def has_xcomp(self, sent):
        """开放式补语从句"""
        for word in sent.words:
            if word.deprel == 'xcomp':
                return True
        return False
    
    def has_csubj(self, sent):
        """从句作主语"""
        for word in sent.words:
            if word.deprel == 'csubj':
                return True
        return False
    
    def has_acl(self, sent):
        """定语从句"""
        for word in sent.words:
            if word.deprel == 'acl':
                return True
        return False
    
    def has_ba_construction(self, sent):
        """把字句"""
        for word in sent.words:
            if word.text == '把' and word.deprel == 'case':
                if word.head > 0:
                    head_word = sent.words[word.head - 1]
                    if head_word.upos in ['VERB']:
                        return True
        return False
    
    def has_bei_construction(self, sent):
        """被字句"""
        for word in sent.words:
            if word.text == '被':
                if word.deprel in ['case', 'aux:pass', 'mark']:
                    return True
        return False
    
    def has_svc(self, sent):
        """连动句"""
        verb_indices = []
        for i, word in enumerate(sent.words):
            if word.upos == 'VERB':
                verb_indices.append(i)
        
        if len(verb_indices) < 2:
            return False
        
        for word in sent.words:
            if word.upos == 'VERB' and word.deprel in ['conj', 'parataxis']:
                has_conj_marker = False
                for w in sent.words:
                    if w.head == word.head and w.deprel in ['cc', 'mark']:
                        has_conj_marker = True
                        break
                
                if not has_conj_marker:
                    return True
        return False
    
    def has_pivot_construction(self, sent):
        """枢轴结构"""
        for word in sent.words:
            if word.deprel in ['xcomp', 'ccomp'] and word.upos == 'VERB':
                if word.head > 0:
                    v1 = sent.words[word.head - 1]
                    has_obj = False
                    for w in sent.words:
                        if w.head == word.head and w.deprel == 'obj':
                            has_obj = True
                            break
                    if has_obj:
                        return True
        return False
    
    def calculate_metrics(self, results):
        """计算所有指标"""
        n_utt = results['n_utterances']
        total_words = results['total_words']
        n_turns = results['total_turns']
        n_unique = len(results['unique_words'])
        
        metrics = {
            'n_utterances': n_utt,
            'total_words': total_words,
            'total_turns': n_turns,
            'unique_words': n_unique,
            
            # MLU, TNU, TNW, NDW
            'MLU': (total_words / n_utt) if n_utt > 0 else 0,
            'TNU': n_utt,  # Total Number of Utterances
            'TNW': total_words,  # Total Number of Words
            'NDW': n_unique,  # Number of Different Words
            
            # 从句比例（小数）
            'ccomp_rate': (results['ccomp_count'] / n_utt) if n_utt > 0 else 0,
            'xcomp_rate': (results['xcomp_count'] / n_utt) if n_utt > 0 else 0,
            'csubj_rate': (results['csubj_count'] / n_utt) if n_utt > 0 else 0,
            'acl_rate': (results['acl_count'] / n_utt) if n_utt > 0 else 0,
            
            # 句法结构比例（小数）
            'ba_rate': (results['ba_count'] / n_utt) if n_utt > 0 else 0,
            'bei_rate': (results['bei_count'] / n_utt) if n_utt > 0 else 0,
            'svc_rate': (results['svc_count'] / n_utt) if n_utt > 0 else 0,
            'pivot_rate': (results['pivot_count'] / n_utt) if n_utt > 0 else 0,
            
            # 连词句比例（小数）
            'cconj_sent_rate': (results['cconj_sent_count'] / n_utt) if n_utt > 0 else 0,
            'sconj_sent_rate': (results['sconj_sent_count'] / n_utt) if n_utt > 0 else 0,
            
            # 连词密度（每百词）
            'cconj_per_100': (results['cconj_count'] / total_words * 100) if total_words > 0 else 0,
            'sconj_per_100': (results['sconj_count'] / total_words * 100) if total_words > 0 else 0,
        }
        
        return metrics

# ==================== 2. 主流程 ====================
def main():
    print("\n" + "=" * 70)
    print("CHILDES Syntactic Complexity Analysis")
    print("=" * 70 + "\n")
    
    # 读取数据
    print(f"Loading data from: {Config.INPUT_CSV}")
    df = pd.read_csv(Config.INPUT_CSV, encoding='utf-8-sig')
    print(f"✓ Loaded {len(df)} utterances\n")
    
    # 确保年龄列存在
    if 'child_age_months' not in df.columns:
        print("❌ Column 'child_age_months' not found!")
        return
    
    # 初始化分析器
    analyzer = ChildSyntaxAnalyzer()
    
    # 按年龄分组分析
    all_results = []
    
    for min_age, max_age, age_label in Config.AGE_BINS:
        print("\n" + "=" * 70)
        print(f"Processing age group: {age_label} ({min_age}-{max_age} months)")
        print("=" * 70)
        
        # 筛选该年龄段
        age_group = df[
            (df['child_age_months'] >= min_age) & 
            (df['child_age_months'] < max_age)
        ]
        
        if len(age_group) == 0:
            print(f"⚠ No utterances found for {age_label}")
            continue
        
        print(f"Found {len(age_group)} utterances")
        
        # 转为字典列表便于处理
        utterances = age_group.to_dict('records')
        
        # 分析
        metrics = analyzer.analyze_age_group(utterances)
        metrics['age_bin'] = age_label
        metrics['age_min'] = min_age
        metrics['age_max'] = max_age
        
        all_results.append(metrics)
        
        # 打印摘要
        print(f"\n✓ Summary for {age_label}:")
        print(f"  MLU (平均语句长度):     {metrics['MLU']:.2f}")
        print(f"  TNU (语句数):           {metrics['TNU']}")
        print(f"  TNW (总词数):           {metrics['TNW']}")
        print(f"  NDW (不同词数):         {metrics['NDW']}")
        print(f"  从句作宾语比例:         {metrics['ccomp_rate']:.4f}")
        print(f"  开放式补语从句比例:     {metrics['xcomp_rate']:.4f}")
        print(f"  从句作主语比例:         {metrics['csubj_rate']:.4f}")
        print(f"  定语从句比例:           {metrics['acl_rate']:.4f}")
        print(f"  把字句比例:             {metrics['ba_rate']:.4f}")
        print(f"  被字句比例:             {metrics['bei_rate']:.4f}")
        print(f"  连动句比例:             {metrics['svc_rate']:.4f}")
        print(f"  枢轴结构比例:           {metrics['pivot_rate']:.4f}")
        print(f"  并列连词句比例:         {metrics['cconj_sent_rate']:.4f}")
        print(f"  从属连词句比例:         {metrics['sconj_sent_rate']:.4f}")
        print(f"  并列连词密度/100:       {metrics['cconj_per_100']:.2f}")
        print(f"  从属连词密度/100:       {metrics['sconj_per_100']:.2f}")
    

    results_df = pd.DataFrame(all_results)
    
    column_order = [
        'age_bin', 'age_min', 'age_max',
        'MLU', 'TNU', 'TNW', 'NDW',
        'n_utterances',
        'ccomp_rate', 'xcomp_rate', 'csubj_rate', 'acl_rate',
        'ba_rate', 'bei_rate', 'svc_rate', 'pivot_rate',
        'cconj_sent_rate', 'sconj_sent_rate',
        'cconj_per_100', 'sconj_per_100',
    ]
    
    results_df = results_df[column_order]
    

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    output_csv = os.path.join(Config.OUTPUT_DIR, "childes_syntax_analysis.csv")
    results_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n✓ Results saved to: {output_csv}")
    
    print("\n" + "=" * 70)
    print("COMPLETE RESULTS TABLE")
    print("=" * 70)
    print(results_df.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()