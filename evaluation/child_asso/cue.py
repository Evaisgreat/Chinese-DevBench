import os
import sys
import numpy as np
import gc

def check_environment():
    major_version = int(np.__version__.split('.')[0])
    if major_version >= 2:
        print("\n [致命错误] 检测到 ABI 不兼容的 Numpy 版本!")
        print("请在终端运行: pip install \"numpy<2\"")
        sys.exit(1)

check_environment()

import torch
import pandas as pd
from scipy.stats import spearmanr
from transformers import BertTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from tqdm import tqdm

INPUT_FILE = r"D:\AAAprojects\babylava\测试数据\cue 1.21.xlsx"
CHECKPOINT_ROOT = r"D:\AAAprojects\babylava\测试数据\checkpoints"
BASELINE_NAME = "uer/gpt2-chinese-cluecorpussmall"

OUTPUT_DIR = r"D:\AAAprojects\babylava\测试数据\最终脚本\cue_results"
LOCAL_MODEL_CACHE = r"D:\AAAprojects\babylava\model_cache"

os.makedirs(LOCAL_MODEL_CACHE, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

os.environ['TRANSFORMERS_CACHE'] = LOCAL_MODEL_CACHE
os.environ['HF_HOME'] = LOCAL_MODEL_CACHE
os.environ['HF_HUB_OFFLINE'] = '1'

OUTPUT_IMAGE_PDF = os.path.join(OUTPUT_DIR, "CHILD_ASSOC_Plot_LastLayer.pdf")
OUTPUT_IMAGE_PNG = os.path.join(OUTPUT_DIR, "CHILD_ASSOC_Plot_LastLayer.png")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "CHILD_ASSOC_Results_LastLayer.csv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AGE_GROUPS = ['2.5-3.5', '3.5-4.5', '4.5-5.5', '5.5-6.5', '6.5-7.5']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

def find_checkpoints_reindexed(root):
    ckpts = []
    if not os.path.exists(root):
        return []

    for root_dir, dirs, _ in os.walk(root):
        for d in dirs:
            if d.startswith("checkpoint-") or d == "final_model":
                path = os.path.join(root_dir, d)
                stage = 1 if ("stage1" in path.lower() or "s1" in path.lower()) else 2
                step = 999999999 if d == "final_model" else int(d.split("-")[-1]) if "-" in d else 0
                ckpts.append({"path": path, "stage": stage, "raw_step": step})
    
    ckpts.sort(key=lambda x: (x['stage'], x['raw_step']))
    s1_cnt, s2_cnt = 0, 0
    for c in ckpts:
        if c['stage'] == 1:
            s1_cnt += 1
            c['label'] = f"S1_C{s1_cnt}"
        else:
            s2_cnt += 1
            c['label'] = f"S2_C{s2_cnt}"
    return ckpts

def load_human_matrices(filepath):
    print(f" 正在构建人类语义网络: {os.path.basename(filepath)}")
    df = pd.read_excel(filepath)
    df.columns = [str(c).strip() for c in df.columns]
    available_cols = [c for c in AGE_GROUPS if c in df.columns]
    
    all_words = sorted(list(set(df['item'].astype(str)) | set(df['association'].astype(str))))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    n = len(all_words)
    matrices = {}

    for col in available_cols:
        mat = np.zeros((n, n))
        col_data = pd.to_numeric(df[col], errors='coerce').fillna(0)
        sub_idx = col_data[col_data > 0].index
        for idx in sub_idx:
            item, assoc = str(df.loc[idx, 'item']), str(df.loc[idx, 'association'])
            if item in word_to_idx and assoc in word_to_idx:
                mat[word_to_idx[item], word_to_idx[assoc]] = col_data[idx]
        matrices[col] = mat
    print(f" 成功提取 {n} 个独特词汇，构建了 {len(matrices)} 个 RSM 矩阵")
    return all_words, matrices

def get_last_layer_embeddings_fast(model, pre_tokenized_inputs, batch_size=256):
    model.eval()
    last_layer_embs = []
    n_samples = pre_tokenized_inputs['input_ids'].size(0)
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_inputs = {
                'input_ids': pre_tokenized_inputs['input_ids'][i:i+batch_size].to(DEVICE),
                'attention_mask': pre_tokenized_inputs['attention_mask'][i:i+batch_size].to(DEVICE)
            }
            
            out = model(**batch_inputs, output_hidden_states=True)
            hidden_state = out.hidden_states[-1] 
            last_token_indices = batch_inputs['attention_mask'].sum(1) - 1
            batch_indices = torch.arange(hidden_state.size(0)).to(DEVICE)
            pooled = hidden_state[batch_indices, last_token_indices]
            
            last_layer_embs.append(pooled.cpu().float())
            
    return torch.cat(last_layer_embs, dim=0).to(DEVICE)

def get_cosine_sim_gpu(embs_tensor):
    embs_norm = torch.nn.functional.normalize(embs_tensor, p=2, dim=1)
    cos_sim = torch.mm(embs_norm, embs_norm.t())
    return cos_sim.cpu().numpy()

def rsa_from_matrices(human_mat, model_cos):
    upper = np.triu_indices_from(human_mat, k=1)
    h_vec, m_vec = human_mat[upper], model_cos[upper]
    mask = h_vec > 0
    if mask.sum() < 3: return np.nan
    return spearmanr(h_vec[mask], m_vec[mask])[0]

def plot_child_assoc(df, ckpts_info):
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'Arial'],
        'axes.unicode_minus': False,
        'font.size': 12,
    })
    
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f9f9f9')
    
    model_df = df[df['Checkpoint'] != 'Baseline']
    base_df = df[df['Checkpoint'] == 'Baseline']
    
    x_labels = [c['label'] for c in ckpts_info]
    x_pos = list(range(len(x_labels)))
    baselines_info = []

    for i, age in enumerate(AGE_GROUPS):
        c = COLORS[i % len(COLORS)]
        
        age_model_df = model_df[model_df['Age'] == age].copy()
        age_model_df['Checkpoint'] = pd.Categorical(age_model_df['Checkpoint'], categories=x_labels, ordered=True)
        age_model_df = age_model_df.sort_values('Checkpoint')
        
        if not age_model_df.empty:
            ax.plot(x_pos, age_model_df['RSA'], marker='o', markersize=7, markeredgecolor='white', 
                    markeredgewidth=1.2, color=c, linestyle='-', linewidth=2.5, label=f"Child-{age}yrs", zorder=3)

        age_base_df = base_df[base_df['Age'] == age]
        if not age_base_df.empty:
            base_val = age_base_df.iloc[0]['RSA']
            ax.axhline(y=base_val, color=c, linestyle='--', linewidth=2.0, alpha=0.7, zorder=1)
            baselines_info.append({'val': base_val, 'color': c, 'age': age})

    if baselines_info and len(x_pos) > 0:
        baselines_info.sort(key=lambda x: x['val'])
        min_y_dist = 0.015
        current_y = -999

        for item in baselines_info:
            target_y = item['val']
            if target_y < current_y + min_y_dist:
                target_y = current_y + min_y_dist
            item['text_y'] = target_y
            current_y = target_y

        for item in baselines_info:
            val, text_y, c = item['val'], item['text_y'], item['color']
            ax.text(x_pos[-1] + 0.55, text_y, f"{val:.3f}", color=c, fontsize=12, fontweight='bold', va='center', ha='left',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=c, alpha=0.95, linewidth=1.5), zorder=5)
            ax.plot([x_pos[-1] + 0.1, x_pos[-1] + 0.50], [val, text_y], color=c, linestyle=':', linewidth=1.5, alpha=0.8, zorder=4)

    ax.set_title("CUE-ASSOCIATION", fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel("Checkpoints", fontsize=13, fontweight='bold')
    ax.set_ylabel("Spearman Correlation", fontsize=13, fontweight='bold')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=11)
    
    if len(x_pos) > 0:
        ax.set_xlim(x_pos[0] - 0.5, x_pos[-1] + 2.8)
        
    cur_ylim = ax.get_ylim()
    y_margin = (cur_ylim[1] - cur_ylim[0]) * 0.1
    ax.set_ylim(cur_ylim[0] - y_margin, cur_ylim[1] + y_margin)

    ax.grid(True, linestyle='--', alpha=0.6, color='#cfcfcf')
    
    pretrained_handle = mlines.Line2D([], [], color='black', linestyle='--', linewidth=2.0, label='Pretrained-GPT-2')
    handles, labels = ax.get_legend_handles_labels()
    handles.append(pretrained_handle)
    labels.append('Pretrained-GPT-2')
    
    leg = ax.legend(handles=handles, labels=labels, loc='lower right', frameon=True, edgecolor='black', facecolor='white', framealpha=0.95, fontsize=11, ncol=2)
    leg.set_zorder(10)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_PDF, dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_IMAGE_PNG, dpi=300, bbox_inches='tight')
    print(f"\n 图表已保存:\n   PDF: {OUTPUT_IMAGE_PDF}\n   PNG: {OUTPUT_IMAGE_PNG}")
    plt.close()

def main():
    print(f"\n 当前计算设备: {DEVICE.upper()}")
    print("\n" + "=" * 70)
    print(" Cue-Association 任务测试开始 (提速版)")
    print("=" * 70)

    try:
        words, human_mats = load_human_matrices(INPUT_FILE)
    except Exception as e:
        print(f" 加载数据失败: {e}")
        return
        
    ckpts_info = find_checkpoints_reindexed(CHECKPOINT_ROOT)
    if not ckpts_info: return

    print("\n 正在预处理词汇分词 (只需执行一次)...")
    try:
        tokenizer = BertTokenizer.from_pretrained(BASELINE_NAME, cache_dir=LOCAL_MODEL_CACHE, local_files_only=True)
        tokenizer.pad_token = tokenizer.sep_token
        pre_tokenized_inputs = tokenizer(words, return_tensors='pt', padding=True, truncation=True, max_length=16)
    except Exception as e:
        print(f" Tokenizer 加载失败: {e}")
        return

    results = []

    def process_model(model_obj, ckpt_label):
        emb_tensor = get_last_layer_embeddings_fast(model_obj, pre_tokenized_inputs, batch_size=256)
        cos_sim = get_cosine_sim_gpu(emb_tensor)
        for age in AGE_GROUPS:
            if age not in human_mats: continue
            rsa_val = rsa_from_matrices(human_mats[age], cos_sim)
            if not np.isnan(rsa_val):
                results.append({'Checkpoint': ckpt_label, 'Age': age, 'RSA': rsa_val, 'Best_Layer': 'Last_Layer'})

    print("\n" + "=" * 70)
    print(f" 评估预训练基准: {BASELINE_NAME} ...")
    print("=" * 70)
    
    try:
        base_mdl = AutoModelForCausalLM.from_pretrained(
            BASELINE_NAME, 
            cache_dir=LOCAL_MODEL_CACHE, 
            local_files_only=True,
            torch_dtype=torch.float16 if DEVICE == 'cuda' else torch.float32,
            low_cpu_mem_usage=True
        ).to(DEVICE)
        
        process_model(base_mdl, 'Baseline')
        
        del base_mdl
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f" 基准模型加载失败: {e}")
        return

    print("\n" + "=" * 70)
    print(f" 遍历评估 {len(ckpts_info)} 个微调模型 ...")
    print("=" * 70)
    
    for info in tqdm(ckpts_info, desc="Checkpoints"):
        label, path = info['label'], info['path']
        try:
            mdl = AutoModelForCausalLM.from_pretrained(
                path,
                cache_dir=LOCAL_MODEL_CACHE,
                local_files_only=True,
                torch_dtype=torch.float16 if DEVICE == 'cuda' else torch.float32,
                low_cpu_mem_usage=True
            ).to(DEVICE)
            
            process_model(mdl, label)
            
            del mdl
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e: 
            print(f" 跳过 {label}: {e}")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        
        print("\n" + "=" * 70)
        print(" 数据已保存:")
        print(f"   汇总表: {OUTPUT_CSV}")
        print("=" * 70)
        
        plot_child_assoc(df, ckpts_info)
        print(" Cue-Association 测试完成！")

if __name__ == "__main__":
    main()