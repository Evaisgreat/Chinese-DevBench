import os
import re
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import BertTokenizer, AutoModelForCausalLM

CHECKPOINT_ROOT = r"D:\AAAprojects\babylava\测试数据\checkpoints"
EXCEL_FILE = r"D:\AAAprojects\babylava\测试数据\Copy morphology&ppvt_agebin_avg.xlsx"
BASELINE_NAME = "uer/gpt2-chinese-cluecorpussmall"

OUTPUT_DIR = r"D:\AAAprojects\babylava\测试数据\最终脚本\morphology_results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AGE_GROUPS = ['3-4岁', '4-5岁', '5-6岁']
COLORS = ['#ff7f0e', '#2ca02c', '#9467bd', '#d62728', '#1f77b4']

LOCAL_MODEL_CACHE = r"D:\AAAprojects\babylava\model_cache"
os.makedirs(LOCAL_MODEL_CACHE, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

os.environ['TRANSFORMERS_CACHE'] = LOCAL_MODEL_CACHE
os.environ['HF_HOME'] = LOCAL_MODEL_CACHE
os.environ['HF_HUB_OFFLINE'] = '1' 


def optimize_prompt(text):
    text = str(text).strip()
    text = re.sub(r'你想一想我们[会]?怎么叫它[？\?]?$', '我们叫它', text)
    text = re.sub(r'我们又会怎么形容它们呢[？\?]?$', '我们叫它', text)
    text = re.sub(r'我们怎么叫它[呀呢]?[？\?]?$', '我们叫它', text)
    text = re.sub(r'我们会怎么叫它[呀呢]?[？\?]?$', '我们叫它', text)
    text = re.sub(r'我们会叫它什么[呀呢]?[？\?]?$', '我们叫它', text)
    text = re.sub(r'我们就叫它——[：:]?$', '我们叫它', text)
    if not text.endswith("我们叫它"):
        text = re.sub(r'[，。！？\?：:]+$', '', text) + "，我们叫它"
    return text


def load_data(filepath):
    try:
        xls = pd.ExcelFile(filepath)
        df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
        df.columns = [str(c).strip() for c in df.columns]
        df = df.dropna(subset=['引导', 'Item']).head(15).copy()
        df['optimized_prompt'] = df['引导'].apply(optimize_prompt)
        age_cols = [c for c in df.columns if '岁' in c or '-' in c or 'age' in c.lower()]
        return df, age_cols
    except Exception as e:
        print(f" 加载Excel失败: {e}")
        return None, None


def find_checkpoints(root):
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

    s1, s2 = 0, 0
    for c in ckpts:
        label = f"S1_C{s1 + 1}" if c['stage'] == 1 else f"S2_C{s2 + 1}"
        if c['stage'] == 1:
            s1 += 1
        else:
            s2 += 1
        c['label'] = label

    return ckpts


def load_model_safe(model_path, is_baseline=False):
    try:
        print(f" 加载模型: {os.path.basename(model_path) if not is_baseline else BASELINE_NAME}")

        tokenizer = BertTokenizer.from_pretrained(
            model_path,
            cache_dir=LOCAL_MODEL_CACHE,
            local_files_only=True
        )
        tokenizer.pad_token = tokenizer.sep_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            cache_dir=LOCAL_MODEL_CACHE,
            local_files_only=True,
            torch_dtype=torch.float16 if DEVICE == 'cuda' else torch.float32,
            low_cpu_mem_usage=True
        ).to(DEVICE).eval()

        print(" 加载成功")
        return tokenizer, model

    except Exception as e:
        if is_baseline:
            print(f"\n 基线模型加载失败: {e}")
            print(" 请先运行 download_models.py 下载模型")
            raise
        else:
            print(f" 加载失败: {e}")
            raise


def get_completion_logprob(model, tokenizer, prompt, target):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    text_ids = tokenizer.encode(prompt + target, add_special_tokens=False)

    inputs = torch.tensor([text_ids]).to(DEVICE)

    with torch.no_grad():
        outputs = model(inputs, labels=inputs)
        logits = outputs.logits[0, :-1, :]
        labels = inputs[0, 1:]

        start_idx = len(prompt_ids) - 1
        target_logits = logits[start_idx:]
        target_labels = labels[start_idx:]

        loss = torch.nn.functional.cross_entropy(target_logits, target_labels, reduction='sum')
        return -loss.item()


def process_model_logprob(model_obj, tokenizer_obj, df, age_cols, label="Model"):
    prompts_list = df['optimized_prompt'].tolist()
    targets_list = df['Item'].tolist()

    summary_res = []
    detail_res = []

    model_scores = []
    for p, t in zip(prompts_list, targets_list):
        score = get_completion_logprob(model_obj, tokenizer_obj, p, t)
        model_scores.append(score)

    for age in age_cols:
        human_sim = pd.to_numeric(df[age], errors='coerce').fillna(0).values
        rsa_val, _ = spearmanr(model_scores, human_sim)
        corr = rsa_val if not np.isnan(rsa_val) else 0.0

        summary_res.append({
            'Checkpoint': label,
            'Age': age,
            'Spearman_Corr': corr
        })

        for idx in range(len(prompts_list)):
            detail_res.append({
                'Checkpoint': label,
                'Age_Group': age,
                'Item_Idx': idx + 1,
                'Prompt': prompts_list[idx],
                'Target': targets_list[idx],
                'Log_Probability': model_scores[idx]
            })

    return summary_res, detail_res


def plot_alignment(df_res, base_scores_dict, age_cols):
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'Arial'],
        'axes.unicode_minus': False,
        'font.size': 14,
    })

    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)

    x_labels = df_res['Checkpoint'].unique().tolist()
    x_pos = list(range(len(x_labels)))
    baselines_info = []

    for i, col in enumerate(age_cols):
        c = COLORS[i % len(COLORS)]
        display_name = col.replace('岁', 'yrs')

        age_df = df_res[df_res['Age'] == col]
        y_vals = age_df['Spearman_Corr'].values

        ax.plot(
            x_pos, y_vals,
            marker='o',
            markersize=9,
            markeredgecolor='white',
            markeredgewidth=1.5,
            color=c,
            label=f"Child-{display_name}",
            linewidth=3.0
        )

        b_val = base_scores_dict.get(col, 0.0)
        ax.axhline(y=b_val, color=c, linestyle='--', linewidth=2.0, alpha=0.6)
        baselines_info.append({'val': b_val, 'color': c, 'age': col})

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
            val = item['val']
            text_y = item['text_y']
            c = item['color']

            ax.text(
                x_pos[-1] + 0.55, text_y, f"{val:.3f}",
                color=c, fontsize=13, fontweight='bold',
                va='center', ha='left',
                bbox=dict(
                    boxstyle='round,pad=0.2',
                    facecolor='white',
                    edgecolor=c,
                    alpha=0.9,
                    linewidth=1.5
                )
            )

            ax.plot(
                [x_pos[-1] + 0.1, x_pos[-1] + 0.50],
                [val, text_y],
                color=c,
                linestyle=':',
                linewidth=1.5,
                alpha=0.7
            )

    ax.set_title("CHILD-ANALOGY",
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel("Spearman Correlation", fontsize=15, fontweight='bold')
    ax.set_xlabel("Checkpoint", fontsize=15, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=40, ha='right', fontsize=12)
    ax.tick_params(axis='y', labelsize=12)

    if len(x_pos) > 0:
        ax.set_xlim(x_pos[0] - 0.5, x_pos[-1] + 3.0)

    cur_ylim = ax.get_ylim()
    y_margin = (cur_ylim[1] - cur_ylim[0]) * 0.15
    ax.set_ylim(cur_ylim[0] - y_margin, cur_ylim[1] + y_margin)

    pretrained_handle = mlines.Line2D(
        [], [], color='black', linestyle='--',
        linewidth=2.0, label='Pretrained-GPT-2'
    )
    handles, labels = ax.get_legend_handles_labels()
    handles.append(pretrained_handle)
    labels.append('Pretrained-GPT-2')

    leg = ax.legend(
        handles=handles,
        labels=labels,
        loc='lower right',
        ncol=2,
        frameon=True,
        edgecolor='black',
        facecolor='white',
        fontsize=12,
        shadow=True
    )
    leg.set_zorder(10)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)

    ax.grid(True, linestyle=':', alpha=0.4, color='gray')

    plt.tight_layout()

    save_path_pdf = os.path.join(OUTPUT_DIR, "Morphology_LogProb_Plot.pdf")
    save_path_png = os.path.join(OUTPUT_DIR, "Morphology_LogProb_Plot.png")

    plt.savefig(save_path_pdf, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight', facecolor='white')

    print(f"\n 图表已保存:")
    print(f"   PDF: {save_path_pdf}")
    print(f"   PNG: {save_path_png}")

    plt.close()


def main():
    print("\n" + "=" * 70)
    print(" 形态学测试开始")
    print("=" * 70)

    print("\n 加载Excel数据...")
    df, age_cols = load_data(EXCEL_FILE)
    if df is None:
        return
    print(f" 数据加载成功：{len(df)} 道题目, {len(age_cols)} 个年龄段")

    ckpts = find_checkpoints(CHECKPOINT_ROOT)
    print(f" 找到 {len(ckpts)} 个checkpoint")

    model_summary = []
    model_details = []

    print("\n" + "=" * 70)
    print(" 评估 Pretrained GPT-2...")
    print("=" * 70)

    try:
        base_tok, base_mdl = load_model_safe(BASELINE_NAME, is_baseline=True)
    except Exception:
        return

    base_sum, _ = process_model_logprob(base_mdl, base_tok, df, age_cols, label="Baseline")
    base_scores_dict = {row['Age']: row['Spearman_Corr'] for row in base_sum}

    print(f"\n 基线模型评估完成:")
    for row in base_sum:
        print(f"   {row['Age']}: {row['Spearman_Corr']:.4f}")

    del base_mdl, base_tok
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print(f" 评估 {len(ckpts)} 个微调模型...")
    print("=" * 70)

    for _, ckpt in enumerate(tqdm(ckpts, desc="Checkpoints"), 1):
        try:
            tok, mdl = load_model_safe(ckpt['path'])

            c_sum, c_det = process_model_logprob(mdl, tok, df, age_cols, label=ckpt['label'])
            model_summary.extend(c_sum)
            model_details.extend(c_det)

            del mdl, tok
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f" {ckpt['label']} 评估失败: {e}")

    csv_summary = os.path.join(OUTPUT_DIR, "LogProb_Summary_ModelOnly.csv")
    csv_details = os.path.join(OUTPUT_DIR, "LogProb_Item_Details_ModelOnly.csv")

    pd.DataFrame(model_summary).to_csv(csv_summary, index=False, encoding='utf-8-sig')
    pd.DataFrame(model_details).to_csv(csv_details, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 70)
    print(" 数据已保存（仅微调模型，不含 baseline）:")
    print(f"   汇总表: {csv_summary}")
    print(f"   详细表: {csv_details}")
    print("=" * 70)

    print("\n 生成可视化图表...")
    plot_alignment(
        pd.DataFrame(model_summary),
        base_scores_dict,
        age_cols
    )

    print("\n" + "=" * 70)
    print(" 形态学测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()