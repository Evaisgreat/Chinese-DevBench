import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

CSV_FILE = r"D:\AAAprojects\babylava\测试数据\最终脚本\cue_results\Cue_Association_Summary_ModelOnly.csv"
OUTPUT_DIR = os.path.dirname(CSV_FILE)

OUTPUT_IMAGE_PDF = os.path.join(OUTPUT_DIR, "CHILD_ASSOC_Plot_Staggered.pdf")
OUTPUT_IMAGE_PNG = os.path.join(OUTPUT_DIR, "CHILD_ASSOC_Plot_Staggered.png")

AGE_GROUPS = ['2.5-3.5', '3.5-4.5', '4.5-5.5', '5.5-6.5', '6.5-7.5']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

def main():
    if not os.path.exists(CSV_FILE):
        print(f"找不到数据文件: {CSV_FILE}")
        return

    print("正在读取测试结果数据...")
    df = pd.read_csv(CSV_FILE)

    model_df = df[df['Checkpoint'] != 'Baseline']
    base_df = df[df['Checkpoint'] == 'Baseline']

    if model_df.empty:
        print("CSV 中没有微调模型的数据！")
        return

    x_labels = model_df['Checkpoint'].unique().tolist()
    x_pos = list(range(len(x_labels)))

    print("启动双向防重叠(Staggered Layout)学术绘图...")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'Arial'],
        'axes.unicode_minus': False,
        'font.size': 12,
    })
    
    fig, ax = plt.subplots(figsize=(13, 7.5), dpi=200)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#fcfcfc')

    baselines_info = []

    for i, age in enumerate(AGE_GROUPS):
        c = COLORS[i % len(COLORS)]
        
        age_model_df = model_df[model_df['Age'] == age].copy()
        age_model_df['Checkpoint'] = pd.Categorical(age_model_df['Checkpoint'], categories=x_labels, ordered=True)
        age_model_df = age_model_df.sort_values('Checkpoint')
        
        if not age_model_df.empty:
            ax.plot(
                x_pos, age_model_df['RSA'], marker='o', markersize=7, 
                markeredgecolor='white', markeredgewidth=1.2, color=c, 
                linestyle='-', linewidth=2.5, label=f"Child-{age}yrs", zorder=3
            )

        age_base_df = base_df[base_df['Age'] == age]
        if not age_base_df.empty:
            base_val = age_base_df.iloc[0]['RSA']
            ax.axhline(y=base_val, color=c, linestyle='--', linewidth=2.0, alpha=0.6, zorder=1)
            baselines_info.append({'val': base_val, 'color': c, 'age': age})

    if baselines_info and len(x_pos) > 0:
        baselines_info.sort(key=lambda x: x['val'])
        
        min_y_dist = 0.012  
        current_y = -999

        for i, item in enumerate(baselines_info):
            target_y = item['val']
            if target_y < current_y + min_y_dist:
                target_y = current_y + min_y_dist
            
            item['text_y'] = target_y
            current_y = target_y
            
            item['text_x'] = x_pos[-1] + 0.6 + (i % 2) * 1.2

        for item in baselines_info:
            val = item['val']
            text_y = item['text_y']
            text_x = item['text_x']
            c = item['color']

            ax.text(
                text_x, text_y, f"{val:.3f}", color=c, fontsize=11, fontweight='bold',
                va='center', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=c, alpha=0.95, linewidth=1.5),
                zorder=5
            )
            
            ax.plot(
                [x_pos[-1] + 0.1, text_x - 0.05], [val, text_y], 
                color=c, linestyle=':', linewidth=1.5, alpha=0.8, zorder=4
            )

    ax.set_title("CHILD-ASSOC", fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel("Checkpoints", fontsize=14, fontweight='bold')
    ax.set_ylabel("Spearman Correlation", fontsize=14, fontweight='bold')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=11)
    
    if len(x_pos) > 0:
        ax.set_xlim(x_pos[0] - 0.5, x_pos[-1] + 3.5)
        
    cur_ylim = ax.get_ylim()
    y_margin = (cur_ylim[1] - cur_ylim[0]) * 0.1
    ax.set_ylim(cur_ylim[0] - y_margin, cur_ylim[1] + y_margin)

    ax.grid(True, linestyle='--', alpha=0.5, color='#d3d3d3')
    
    pretrained_handle = mlines.Line2D([], [], color='black', linestyle='--', linewidth=2.0, label='Pretrained-GPT-2')
    handles, labels = ax.get_legend_handles_labels()
    handles.append(pretrained_handle)
    labels.append('Pretrained-GPT-2')
    
    leg = ax.legend(
        handles=handles, labels=labels, loc='lower right', 
        frameon=True, edgecolor='black', facecolor='white', framealpha=0.95, fontsize=11, ncol=2
    )
    leg.set_zorder(10)

    for spine in ax.spines.values():
        spine.set_color('#333333')
        spine.set_linewidth(1.5)

    plt.tight_layout()
    
    plt.savefig(OUTPUT_IMAGE_PDF, dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_IMAGE_PNG, dpi=300, bbox_inches='tight')
    print(f"\n错位布局渲染完成！图片已保存至:")
    print(f"   [PDF] {OUTPUT_IMAGE_PDF}")
    print(f"   [PNG] {OUTPUT_IMAGE_PNG}")
    plt.close()

if __name__ == "__main__":
    main()