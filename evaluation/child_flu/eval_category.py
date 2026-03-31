import os
import torch
import pandas as pd
import numpy as np
import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import spearmanr
from transformers import BertTokenizerFast, GPT2LMHeadModel

CHECKPOINT_ROOT = "/home/czs/GPT2/output_curriculum"

DATA_DIR = "/home/czs/GPT2/processed_data"
ANIMAL_FILE = os.path.join(DATA_DIR, "Copy animal.xlsx")
FRUIT_FILE = os.path.join(DATA_DIR, "Copy fruit.xlsx")

OUTPUT_DIR = "category_graph_results"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

PROMPTS = {
    "animal": ["动物有", "常见的动物包括"],
    "fruit": ["水果有", "常见的水果包括"]
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def find_all_checkpoints(root_dir):
    checkpoints = []
    stage_dirs = glob.glob(os.path.join(root_dir, "stage*"))
    for stage_dir in stage_dirs:
        stage_name = os.path.basename(stage_dir)
        ckpt_dirs = glob.glob(os.path.join(stage_dir, "checkpoint-*"))
        for ckpt in ckpt_dirs:
            try:
                step = int(re.search(r'checkpoint-(\d+)', ckpt).group(1))
                checkpoints.append({"path": ckpt, "stage": stage_name, "step": step, "full_name": f"{stage_name}-{step}"})
            except: continue
    
    for stage_dir in stage_dirs:
        final_path = os.path.join(stage_dir, "final_model")
        if os.path.exists(final_path):
            stage_name = os.path.basename(stage_dir)
            checkpoints.append({"path": final_path, "stage": stage_name, "step": 999999, "full_name": f"{stage_name}-Final"})

    checkpoints.sort(key=lambda x: (x['stage'], x['step']))
    return checkpoints

def get_category_score(model, tokenizer, category_type, item_name):
    prompt = PROMPTS[category_type][0] 
    
    text = prompt + item_name
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    input_ids = inputs.input_ids
    
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_len = len(prompt_ids)
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        logits = outputs.logits[0, :-1, :]
        labels = input_ids[0, 1:]
        
        item_ids = tokenizer.encode(item_name, add_special_tokens=False)
        if not item_ids: return -999
        
        input_list = input_ids[0].tolist()
        item_start = -1
        for i in range(len(input_list) - len(item_ids), -1, -1):
            if input_list[i:i+len(item_ids)] == item_ids:
                item_start = i
                break
        
        if item_start == -1: return -999

        target_logits = logits[item_start-1 : item_start+len(item_ids)-1]
        target_labels = labels[item_start-1 : item_start+len(item_ids)-1]
        
        if len(target_labels) == 0: return -999
        
        loss = torch.nn.functional.cross_entropy(target_logits, target_labels, reduction='sum')
        score = -loss.item() 
        
    return score

def process_task(model, tokenizer, df, category_type):
    model_scores = []
    
    for item in df['item']:
        item = str(item).strip()
        score = get_category_score(model, tokenizer, category_type, item)
        model_scores.append(score)
    
    results = {}
    
    age_cols = [c for c in df.columns if re.search(r'\d\.\d-\d\.\d', c) or 'adult' in c.lower()]
    age_cols = [c for c in age_cols if '.1' not in c and '除以' not in c and 'Unnamed' not in c]
    age_cols = sorted(list(set(age_cols)))

    for age in age_cols:
        human_freq = pd.to_numeric(df[age], errors='coerce').fillna(0)
        
        if np.count_nonzero(human_freq) > 5:
            corr, _ = spearmanr(model_scores, human_freq)
            results[age] = corr if not np.isnan(corr) else 0
        else:
            results[age] = 0
            
    return results

def plot_results(all_results):
    df_res = pd.DataFrame(all_results)
    
    tasks = ['Animal', 'Fruit']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for i, task in enumerate(tasks):
        task_cols = [c for c in df_res.columns if task in c]
        ax = axes[i]
        
        for col in task_cols:
            age_label = col.replace(f"{task}_", "")
            ax.plot(df_res['name'], df_res[col], marker='o', markersize=4, label=age_label)
            
        ax.set_title(f"{task} Graph Correlation")
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Spearman Correlation")
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title="Age Group")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "category_graph_evolution.png"))
    print(f"图表已保存: {os.path.join(OUTPUT_DIR, 'category_graph_evolution.png')}")

def main():
    print("读取 Animal & Fruit 数据...")
    try:
        df_animal = pd.read_excel(ANIMAL_FILE, sheet_name=0)
        df_fruit = pd.read_excel(FRUIT_FILE, sheet_name=0)
        
        df_animal = df_animal[['item'] + [c for c in df_animal.columns if '-' in c or 'adult' in c]]
        df_fruit = df_fruit[['item'] + [c for c in df_fruit.columns if '-' in c or 'adult' in c]]
        
        print(f"  Animal: {len(df_animal)} items")
        print(f"  Fruit: {len(df_fruit)} items")
    except Exception as e:
        print(f"读取失败: {e}")
        return

    ckpts = find_all_checkpoints(CHECKPOINT_ROOT)
    print(f"找到 {len(ckpts)} 个模型存档...")
    
    all_results = []
    
    for ckpt in tqdm(ckpts, desc="Evaluating"):
        res = {"name": ckpt['full_name']}
        
        try:
            tokenizer = BertTokenizerFast.from_pretrained(ckpt['path'])
            model = GPT2LMHeadModel.from_pretrained(ckpt['path']).to(DEVICE)
            model.eval()
            
            animal_res = process_task(model, tokenizer, df_animal, "animal")
            for k, v in animal_res.items():
                res[f"Animal_{k}"] = v
                
            fruit_res = process_task(model, tokenizer, df_fruit, "fruit")
            for k, v in fruit_res.items():
                res[f"Fruit_{k}"] = v
                
            all_results.append(res)
            
        except Exception as e:
            print(f"跳过 {ckpt['full_name']}: {e}")

    if all_results:
        df_out = pd.DataFrame(all_results)
        df_out.to_csv(os.path.join(OUTPUT_DIR, "category_graph_data.csv"), index=False)
        plot_results(all_results)
        print("完成！")

if __name__ == "__main__":
    main()