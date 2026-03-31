import os
import torch
import pandas as pd
import numpy as np
import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, KFold
from transformers import BertTokenizerFast, GPT2LMHeadModel

CHECKPOINT_ROOT = "/home/czs/GPT2/output_curriculum"

DATA_PATH = "/home/czs/GPT2/processed_data/Rated_semantic_dimensions.csv"

OUTPUT_DIR = "semantic_evolution_results"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_DIMS = ['Vision', 'Motor', 'Socialness', 'Emotion', 'Time', 'Space']

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

def get_word_embedding(model, tokenizer, word):
    inputs = tokenizer(word, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1] 
        token_embeddings = last_hidden_state.squeeze(0)
        embedding = torch.mean(token_embeddings, dim=0).cpu().numpy()
    return embedding

def evaluate_checkpoint(model_path, df_data):
    try:
        tokenizer = BertTokenizerFast.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path).to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"加载失败 {model_path}: {e}")
        return None

    embeddings = []
    valid_indices = []
    
    for idx, row in df_data.iterrows():
        word = str(row['word']).strip()
        if not word: continue
        try:
            emb = get_word_embedding(model, tokenizer, word)
            embeddings.append(emb)
            valid_indices.append(idx)
        except:
            pass
            
    if not embeddings: return None

    X = np.array(embeddings)
    df_valid = df_data.iloc[valid_indices].reset_index(drop=True)

    scores = {}
    regressor = LinearRegression()
    cv = KFold(n_splits=5, shuffle=True, random_state=42) 
    
    for dim in TARGET_DIMS:
        if dim not in df_valid.columns: continue
        
        y = pd.to_numeric(df_valid[dim], errors='coerce').fillna(0).values
        try:
            y_pred = cross_val_predict(regressor, X, y, cv=cv, n_jobs=-1)
            corr, _ = pearsonr(y, y_pred)
            scores[dim] = corr
        except:
            scores[dim] = 0
            
    return scores

def plot_evolution(df_res):
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    palette = sns.color_palette("tab10", n_colors=len(TARGET_DIMS))
    
    for i, dim in enumerate(TARGET_DIMS):
        if dim in df_res.columns:
            plt.plot(df_res['name'], df_res[dim], marker='o', linewidth=2, label=dim, color=palette[i])

    plt.title('Evolution of Semantic Dimension Alignment', fontsize=16)
    plt.xlabel('Training Checkpoints', fontsize=12)
    plt.ylabel('Prediction Accuracy (Pearson r)', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title="Semantic Dimensions", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "semantic_evolution_curve.png")
    plt.savefig(save_path, dpi=300)
    print(f"进化曲线已保存: {save_path}")

def main():
    print(f"读取数据: {DATA_PATH}")
    try:
        try:
            df = pd.read_csv(DATA_PATH)
        except:
            df = pd.read_csv(DATA_PATH, encoding='gbk')
    except:
        print("数据读取失败")
        return
        
    ckpts = find_all_checkpoints(CHECKPOINT_ROOT)
    print(f"找到 {len(ckpts)} 个 Checkpoints")
    
    all_results = []
    
    pbar = tqdm(ckpts, desc="Evaluating Models")
    for ckpt in pbar:
        pbar.set_description(f"Eval: {ckpt['full_name']}")
        
        scores = evaluate_checkpoint(ckpt['path'], df)
        
        if scores:
            scores['name'] = ckpt['full_name']
            all_results.append(scores)
            
            msg = f"  {ckpt['full_name']}: "
            msg += f"Vis={scores.get('Vision',0):.2f}, Emo={scores.get('Emotion',0):.2f}"
            tqdm.write(msg)

    if all_results:
        df_res = pd.DataFrame(all_results)
        csv_path = os.path.join(OUTPUT_DIR, "semantic_evolution_data.csv")
        df_res.to_csv(csv_path, index=False)
        print(f"数据已保存: {csv_path}")
        
        plot_evolution(df_res)
    else:
        print("没有产生有效结果")

if __name__ == "__main__":
    main()