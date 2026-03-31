import os
import torch
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import spearmanr
from transformers import GPT2LMHeadModel, BertTokenizerFast
import numpy as np

STS_DIR = "/data0/datasets/babylm_clean/ChineseSTS-master"

CHECKPOINT_ROOT = os.path.expanduser("~/GPT2/output_curriculum")

BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_sts_data(data_dir):
    print(f"扫描 STS 数据: {data_dir}")
    files = glob.glob(os.path.join(data_dir, "*.txt")) 
    all_data = []
    
    for f in files:
        filename = os.path.basename(f)
        if "readme" in filename.lower(): continue

        try:
            with open(f, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            valid_count = 0
            for line in lines:
                parts = line.strip().split('\t')
                if len(parts) >= 5:
                    s1 = parts[1].strip()
                    s2 = parts[3].strip()
                    try:
                        score = float(parts[4].strip())
                        all_data.append({'s1': s1, 's2': s2, 'score': score})
                        valid_count += 1
                    except ValueError:
                        continue 
            print(f" {filename}: 读取 {valid_count} 条")
        except Exception as e:
            print(f" {filename}: {e}")

    print(f"总共加载测试数据: {len(all_data)} 对句子")
    return all_data

def get_sentence_embeddings(model, tokenizer, sentences):
    inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=128).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        
        last_hidden = outputs.hidden_states[-1] 
        
        attention_mask = inputs['attention_mask'].unsqueeze(-1).float()
        
        sum_embeddings = torch.sum(last_hidden * attention_mask, dim=1)
        
        sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        
        mean_embeddings = sum_embeddings / sum_mask
        
    return mean_embeddings

def evaluate_sts(model, tokenizer, data_list):
    model.eval()
    
    s1_list = [d['s1'] for d in data_list]
    s2_list = [d['s2'] for d in data_list]
    gold_scores = [d['score'] for d in data_list]
    
    cosine_sims = []
    
    for i in range(0, len(data_list), BATCH_SIZE):
        batch_s1 = s1_list[i : i+BATCH_SIZE]
        batch_s2 = s2_list[i : i+BATCH_SIZE]
        
        emb1 = get_sentence_embeddings(model, tokenizer, batch_s1)
        emb2 = get_sentence_embeddings(model, tokenizer, batch_s2)
        
        sim = torch.nn.functional.cosine_similarity(emb1, emb2)
        cosine_sims.extend(sim.cpu().numpy())
        
    correlation, _ = spearmanr(gold_scores, cosine_sims)
    
    if np.isnan(correlation):
        return 0.0
    return correlation * 100

def find_checkpoints():
    checkpoints = []
    for stage in ["stage1_0-3", "stage2_3-6"]:
        stage_dir = os.path.join(CHECKPOINT_ROOT, stage)
        if not os.path.exists(stage_dir): continue
        
        for d in os.listdir(stage_dir):
            if d.startswith("checkpoint-"):
                try:
                    step = int(d.split("-")[1])
                    checkpoints.append({
                        "stage": stage, "step": step,
                        "path": os.path.join(stage_dir, d),
                        "sort_key": (1 if "stage1" in stage else 2, step)
                    })
                except: pass
        
        final_path = os.path.join(stage_dir, "final_model")
        if os.path.exists(final_path):
             step = 99999 
             checkpoints.append({
                "stage": stage, "step": step, "path": final_path,
                "name": f"{stage}-Final",
                "sort_key": (1 if "stage1" in stage else 2, step + 1)
            })
            
    checkpoints.sort(key=lambda x: x['sort_key'])
    return checkpoints

def main():
    sts_data = load_sts_data(STS_DIR)
    if not sts_data: 
        print("未找到数据，退出。")
        return

    ckpts = find_checkpoints()
    print(f"找到 {len(ckpts)} 个模型存档。")
    
    print("初始化 Tokenizer...")
    tokenizer = BertTokenizerFast.from_pretrained(ckpts[0]['path'])
    tokenizer.pad_token = tokenizer.sep_token 
    
    results = []
    pbar = tqdm(ckpts, desc="Evaluating STS")
    
    for ckpt in pbar:
        try:
            model = GPT2LMHeadModel.from_pretrained(ckpt['path']).to(DEVICE)
            
            score = evaluate_sts(model, tokenizer, sts_data)
            
            name = ckpt.get('name', f"{ckpt['stage']}-{ckpt['step']}")
            results.append({
                "name": name,
                "score": score,
                "stage": ckpt['stage']
            })
            
            pbar.set_postfix({"Spearman": f"{score:.2f}%"})
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\n跳过 {ckpt['path']}: {e}")

    if not results: return
    
    print("正在生成图表...")
    plt.figure(figsize=(14, 7))
    x = range(len(results))
    y = [r['score'] for r in results]
    names = [r['name'] for r in results]
    
    plt.plot(x, y, 'o-', linewidth=2, color='#2ca02c', label='Spearman Correlation')
    
    for i in range(1, len(results)):
        if results[i]['stage'] != results[i-1]['stage']:
            plt.axvline(x=i-0.5, color='red', linestyle='--', label='Stage Boundary')

    plt.title("Semantic Textual Similarity (STS) Performance", fontsize=14)
    plt.xlabel("Training Checkpoints", fontsize=12)
    plt.ylabel("Spearman Correlation (%)", fontsize=12)
    plt.xticks(x, names, rotation=45, ha='right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    save_path = "sts_result_final.png"
    plt.savefig(save_path, dpi=300)
    print(f"测试完成！结果图表已保存至: {save_path}")

if __name__ == "__main__":
    main()