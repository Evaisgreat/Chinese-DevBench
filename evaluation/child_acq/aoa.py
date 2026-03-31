import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy.optimize import curve_fit
from tqdm import tqdm
import matplotlib.pyplot as plt

# ================== 配置 ==================
class Config:
    INPUT_FILE = Path("results/word_learning_curves.json")
    OUTPUT_DIR = Path("results")
    PLOT_DIR = Path("results/plots") 
    

    CUTOFF_PROPORTION = 0.5  # 50% cutoff
    

    SAVE_PLOTS = False  
    
config = Config()
config.PLOT_DIR.mkdir(exist_ok=True, parents=True)


# ================== Sigmoid 拟合 ==================

def sigmoid(x, L, x0, k, b):
    """
    Sigmoid 函数
    L: 曲线的最大值
    x0: sigmoid 的中点
    k: 曲线的陡峭程度
    b: y 轴偏移
    """
    return L / (1 + np.exp(-k * (x - x0))) + b


def fit_sigmoid_curve(steps, surprisals):
    """
    拟合 sigmoid 曲线
    
    Args:
        steps: List[float] - training steps (log10 scale)
        surprisals: List[float] - mean surprisals
    
    Returns:
        params: sigmoid 参数 (L, x0, k, b)
        success: 是否拟合成功
    """
    # 过滤 NaN
    valid_indices = [i for i, s in enumerate(surprisals) if not np.isnan(s)]
    if len(valid_indices) < 4:
        return None, False
    
    x_data = np.array([steps[i] for i in valid_indices])
    y_data = np.array([surprisals[i] for i in valid_indices])
    
    # 初始参数猜测
    L_init = y_data.max() - y_data.min()
    x0_init = x_data[len(x_data) // 2]
    k_init = 1.0
    b_init = y_data.min()
    
    try:
        params, _ = curve_fit(
            sigmoid, x_data, y_data,
            p0=[L_init, x0_init, k_init, b_init],
            maxfev=10000
        )
        return params, True
    except:
        return None, False


def compute_aoa(params, baseline_surprisal, min_surprisal, cutoff_proportion=0.5):
    """
    计算 Age of Acquisition
    
    AoA = 找到 sigmoid 曲线与 cutoff 交点对应的 x 值
    Cutoff = baseline - cutoff_proportion * (baseline - min)
    """
    if params is None:
        return np.nan
    
    L, x0, k, b = params
    
    # Cutoff surprisal
    cutoff_surprisal = baseline_surprisal - cutoff_proportion * (baseline_surprisal - min_surprisal)
    
    # 求解 sigmoid(x) = cutoff_surprisal
    # L / (1 + exp(-k(x - x0))) + b = cutoff
    # L / (1 + exp(-k(x - x0))) = cutoff - b
    # 1 + exp(-k(x - x0)) = L / (cutoff - b)
    # exp(-k(x - x0)) = L / (cutoff - b) - 1
    # -k(x - x0) = log(L / (cutoff - b) - 1)
    # x = x0 - log(L / (cutoff - b) - 1) / k
    
    try:
        ratio = L / (cutoff_surprisal - b) - 1
        if ratio <= 0:
            return np.nan
        aoa = x0 - np.log(ratio) / k
        return aoa
    except:
        return np.nan


def plot_learning_curve(word, steps, surprisals, params, baseline, min_surprisal, aoa, save_path):
    """绘制学习曲线和 sigmoid 拟合"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 原始数据点
    ax.scatter(steps, surprisals, alpha=0.6, label='Observed surprisal')
    
    # Sigmoid 拟合曲线
    if params is not None:
        x_smooth = np.linspace(min(steps), max(steps), 200)
        y_smooth = sigmoid(x_smooth, *params)
        ax.plot(x_smooth, y_smooth, 'b-', label='Fitted sigmoid', linewidth=2)
        
        # Cutoff 线
        cutoff = baseline - config.CUTOFF_PROPORTION * (baseline - min_surprisal)
        ax.axhline(y=cutoff, color='r', linestyle='--', label=f'50% cutoff ({cutoff:.2f})')
        
        # AoA 标记
        if not np.isnan(aoa):
            ax.axvline(x=aoa, color='g', linestyle='--', label=f'AoA ({aoa:.2f})')
            ax.scatter([aoa], [cutoff], color='g', s=100, zorder=5)
    
    # Baseline
    ax.axhline(y=baseline, color='gray', linestyle=':', label=f'Baseline ({baseline:.2f})')
    
    ax.set_xlabel('Training step (log10)', fontsize=12)
    ax.set_ylabel('Surprisal (bits)', fontsize=12)
    ax.set_title(f'Learning curve: {word}', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 反转 y 轴（surprisal 越低越好）
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ================== 主函数 ==================

def main():
    print("=" * 60)
    print("拟合 Sigmoid 曲线，计算 AoA")
    print("=" * 60)
    
    print("\n[1] 加载学习曲线数据...")
    with open(config.INPUT_FILE, 'r', encoding='utf-8') as f:
        learning_curves = json.load(f)
    print(f"  词数量: {len(learning_curves)}")
    
    print("\n[2] 拟合曲线...")
    results = []
    
    for word_data in tqdm(learning_curves, desc="拟合进度"):
        word = word_data['word']
        baseline = word_data['baseline_surprisal']
        
        # 提取 steps 和 surprisals
        checkpoints = word_data['checkpoints']
        steps = []
        surprisals = []
        
        for ckpt in checkpoints:
            step_num = ckpt['step']
            if step_num == -1:  # final_model
                # 根据 stage 推断最终 step
                if ckpt['stage'] == 'Stage1':
                    step_num = 770
                else:
                    step_num = 5700
            
            steps.append(np.log10(step_num + 1))  # log10 scale, +1 避免 log(0)
            surprisals.append(ckpt['mean_surprisal'])
        
        valid_surprisals = [s for s in surprisals if not np.isnan(s)]
        min_surprisal = min(valid_surprisals) if len(valid_surprisals) > 0 else np.nan
        
        params, success = fit_sigmoid_curve(steps, surprisals)
        


        # 计算 AoA（log10）
        if success:
            aoa_log = compute_aoa(params, baseline, min_surprisal, config.CUTOFF_PROPORTION)
        else:
            aoa_log = np.nan

        if not np.isnan(aoa_log):
            aoa_step = 10 ** aoa_log - 1
        else:
            aoa_step = np.nan


        results.append({
            'word': word,
            'n_samples': word_data['n_samples'],
            'baseline_surprisal': baseline,
            'min_surprisal': min_surprisal,
            'aoa_log10': aoa_log, 
            'aoa_step': aoa_step, 
            'fit_success': success,
            'sigmoid_params': params.tolist() if params is not None else None
        })

        if config.SAVE_PLOTS and success:
            plot_path = config.PLOT_DIR / f"{word}.png"
            plot_learning_curve(word, steps, surprisals, params, 
                              baseline, min_surprisal, aoa_log, plot_path)
    
    print("\n[3] 保存结果...")
    
    json_output = config.OUTPUT_DIR / "word_aoa_gpt2.json"
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  已保存: {json_output}")
    
    csv_df = pd.DataFrame(results)
    csv_output = config.OUTPUT_DIR / "word_aoa_gpt2.csv"
    csv_df.to_csv(csv_output, index=False, encoding='utf-8-sig')
    print(f"  已保存: {csv_output}")
    
    n_success = sum(r['fit_success'] for r in results)
    n_valid_aoa = sum(not np.isnan(r['aoa_log10']) for r in results)
    print(f"\n拟合成功: {n_success}/{len(results)}")
    print(f"有效 AoA (log10): {n_valid_aoa}/{len(results)}")

    
    print("\n✅ 完成！")


if __name__ == "__main__":
    main()