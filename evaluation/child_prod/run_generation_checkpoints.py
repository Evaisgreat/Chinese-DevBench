
import os
import re
import warnings
from collections import Counter

import jieba
import jieba.posseg as pseg
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel

warnings.filterwarnings("ignore")


# ==================== 配置 ====================
class Config:
    PROJECT_ROOT = "."
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")
    PROMPTS_PATH = r"E:\Dev-cn\files\prompt.xlsx"

    # 按需要删减 checkpoints
    STAGE1_STEPS = [77, 154, 231, 308, 385, 462, 539, 616, 693, 770, "final_model"]
    STAGE2_STEPS = [570, 1140, 1710, 2280, 2850, 3420, 3990, 4560, 5130, 5700, "final_model"]

    ALL_CHECKPOINTS = []
    for step in STAGE1_STEPS:
        path = f"models/stage1_0-3/{'final_model' if step == 'final_model' else f'checkpoint-{step}'}"
        ALL_CHECKPOINTS.append(("Stage1", step, path))
    for step in STAGE2_STEPS:
        path = f"models/stage2_3-6/{'final_model' if step == 'final_model' else f'checkpoint-{step}'}"
        ALL_CHECKPOINTS.append(("Stage2", step, path))

    N_SAMPLES_PER_CHECKPOINT = 150
    MAX_LENGTH = 50
    TOP_K = 50
    TOP_P = 0.95
    TEMPERATURE = 0.9

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


os.makedirs(Config.OUTPUT_DIR, exist_ok=True)


# ==================== 1. 加载 prompts ====================
def load_prompts():
    print("=" * 70)
    print("Loading prompts from Excel...")
    print("=" * 70)

    df = pd.read_excel(Config.PROMPTS_PATH)
    if "prompt" not in df.columns:
        raise ValueError("Excel 文件里没有 'prompt' 这一列。")

    prompts = df["prompt"].dropna().astype(str).tolist()

    if not prompts:
        raise ValueError("没有读到任何 prompts。")

    print(f"\nLoaded {len(prompts)} prompts:")
    for i, p in enumerate(prompts[:10], 1):
        print(f"  {i}. '{p}'")
    if len(prompts) > 10:
        print(f"  ... (total {len(prompts)})")

    return prompts


# ==================== 2. 文本生成 ====================
class TextGenerator:
    def __init__(self, model_path):
        print(f"\nLoading model: {model_path}")
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model.to(Config.DEVICE)
        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_batch(self, prompts, n_samples_per_prompt=5):
        all_texts = []
        all_prompts = []

        for prompt in tqdm(prompts, desc="Generating"):
            for _ in range(n_samples_per_prompt):
                try:
                    input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(Config.DEVICE)

                    with torch.no_grad():
                        output = self.model.generate(
                            input_ids,
                            max_length=Config.MAX_LENGTH,
                            num_return_sequences=1,
                            do_sample=True,
                            top_k=Config.TOP_K,
                            top_p=Config.TOP_P,
                            temperature=Config.TEMPERATURE,
                            repetition_penalty=1.2,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )

                    text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                    all_texts.append(text)
                    all_prompts.append(prompt)

                except Exception as e:
                    print(f"Error during generation: {e}")
                    continue

        return all_texts, all_prompts


def clean_generated_text(generated, prompt):
    """移除 prompt，尽量保留生成后的第一句。"""
    generated = str(generated).strip()
    prompt = str(prompt).strip()

    if generated.startswith(prompt):
        generated = generated[len(prompt):].strip()

    sentences = re.split(r"[。！？\n]", generated)
    if sentences and sentences[0].strip():
        result = sentences[0].strip()
        if len(result) >= 2:
            return result

    return generated[:20].strip() if len(generated) > 2 else generated.strip()


# ==================== 3. 文本分析 ====================
class SimpleAnalyzer:
    def __init__(self):
        print("\nInitializing jieba...")
        jieba.initialize()

    def analyze_corpus(self, texts):
        print("\nAnalyzing texts...")

        results = {
            "n_utterances": len(texts),
            "total_words": 0,
            "word_length": Counter(),
            "pos_count": Counter(),
        }

        for text in tqdm(texts, desc="Analyzing"):
            if not str(text).strip():
                continue

            try:
                words = list(pseg.cut(text))
                words = [(w, p) for w, p in words if w.strip() and p != "x"]

                results["total_words"] += len(words)

                for word, pos in words:
                    results["word_length"][len(word)] += 1
                    results["pos_count"][self.map_pos_category(pos)] += 1

            except Exception:
                continue

        metrics = self.calculate_metrics(results)
        return results, metrics

    def calculate_metrics(self, results):
        total_words = results["total_words"]
        metrics = {
            "n_utterances": results["n_utterances"],
            "total_words": total_words,
        }

        for length in [1, 2, 3, 4]:
            count = results["word_length"].get(length, 0)
            metrics[f"len_{length}_pct"] = (count / total_words * 100) if total_words > 0 else 0

        for pos_cat in ["Noun", "Verb", "Adjective", "Adverb"]:
            count = results["pos_count"].get(pos_cat, 0)
            metrics[f"pos_{pos_cat.lower()}_pct"] = (count / total_words * 100) if total_words > 0 else 0

        return metrics

    @staticmethod
    def map_pos_category(jieba_pos):
        pos_map = {
            "n": "Noun", "nr": "Proper_noun", "ns": "Noun", "nt": "Noun", "nz": "Noun",
            "v": "Verb", "vn": "Verb", "vd": "Verb",
            "a": "Adjective", "ad": "Adjective",
            "d": "Adverb",
        }
        if jieba_pos in pos_map:
            return pos_map[jieba_pos]
        if jieba_pos and jieba_pos[0] in pos_map:
            return pos_map[jieba_pos[0]]
        return "Others"


# ==================== 4. 论文参考数据 ====================
def get_paper_reference():
    return {
        "K1": {"len_1": 17.48, "len_2": 66.17, "len_3plus": 16.13,
               "noun": 43.03, "verb": 29.22, "adj": 6.03, "adv": 4.39},
        "K2": {"len_1": 14.12, "len_2": 67.42, "len_3plus": 18.46,
               "noun": 44.21, "verb": 28.34, "adj": 5.78, "adv": 3.63},
        "K3": {"len_1": 10.94, "len_2": 67.17, "len_3plus": 21.89,
               "noun": 44.71, "verb": 27.02, "adj": 5.96, "adv": 3.47},
    }


# ==================== 5. 可视化 + 定性结论 ====================
def visualize_and_conclude(df):
    if df.empty:
        print("⚠️ 没有结果可视化。")
        return

    paper_ref = get_paper_reference()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    x = range(len(df))
    labels = [f"{r['stage']}\n{r['step']}" for _, r in df.iterrows()]

    ax = axes[0, 0]
    ax.plot(x, df["len_1_pct"], "o-", linewidth=2, markersize=8, label="GPT-2")
    ax.scatter([0, 2, 4], [paper_ref["K1"]["len_1"], paper_ref["K2"]["len_1"], paper_ref["K3"]["len_1"]],
               s=150, color="red", marker="*", label="Child", zorder=5)
    ax.set_title("1-Syllable Words (%)", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(x, df["len_2_pct"], "s-", linewidth=2, markersize=8)
    ax.scatter([0, 2, 4], [paper_ref["K1"]["len_2"], paper_ref["K2"]["len_2"], paper_ref["K3"]["len_2"]],
               s=150, color="red", marker="*", zorder=5)
    ax.set_title("2-Syllable Words (%)", fontweight="bold")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(x, df["pos_noun_pct"], "o-", linewidth=2, markersize=8)
    ax.scatter([0, 2, 4], [paper_ref["K1"]["noun"], paper_ref["K2"]["noun"], paper_ref["K3"]["noun"]],
               s=150, color="red", marker="*", zorder=5)
    ax.set_title("Noun (%)", fontweight="bold")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(x, df["pos_adverb_pct"], "v-", linewidth=2, markersize=8)
    ax.scatter([0, 2, 4], [paper_ref["K1"]["adv"], paper_ref["K2"]["adv"], paper_ref["K3"]["adv"]],
               s=150, color="red", marker="*", zorder=5)
    ax.set_title("Adverb (%)", fontweight="bold")
    ax.grid(alpha=0.3)

    for ax in axes.flat:
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(f"{Config.OUTPUT_DIR}/full_trends.png", dpi=300, bbox_inches="tight")
    print("\n✓ Visualization saved: full_trends.png")

    print("\n" + "=" * 70)
    print("QUALITATIVE CONCLUSIONS")
    print("=" * 70)

    change_1 = df["len_1_pct"].iloc[-1] - df["len_1_pct"].iloc[0]
    change_noun = df["pos_noun_pct"].iloc[-1] - df["pos_noun_pct"].iloc[0]
    change_adv = df["pos_adverb_pct"].iloc[-1] - df["pos_adverb_pct"].iloc[0]

    print(f"\n1-syllable:  {change_1:+.2f}%  (expect: ↓)  {'✓' if change_1 < 0 else '✗'}")
    print(f"Noun:        {change_noun:+.2f}%  (expect: ↑)  {'✓' if change_noun > 0 else '✗'}")
    print(f"Adverb:      {change_adv:+.2f}%  (expect: ↓)  {'✓' if change_adv < 0 else '✗'}")
    print("=" * 70)


# ==================== 6. 主流程 ====================
def main():
    print("\n" + "=" * 70)
    print("RUN: Multi-checkpoint generation analysis")
    print("=" * 70 + "\n")

    prompts = load_prompts()
    n_samples_per_prompt = Config.N_SAMPLES_PER_CHECKPOINT // len(prompts) + 1

    analyzer = SimpleAnalyzer()
    all_results = []

    for stage, step, model_path in Config.ALL_CHECKPOINTS:
        if not os.path.exists(model_path):
            print(f"⚠️ 模型路径不存在，跳过: {model_path}")
            continue

        print("\n" + "=" * 70)
        print(f"Processing: {stage} - {step}")
        print("=" * 70)

        checkpoint_name = f"{stage}_{step}"

        generator = TextGenerator(model_path)
        raw_texts, used_prompts = generator.generate_batch(prompts, n_samples_per_prompt)

        clean_texts = [clean_generated_text(t, p) for t, p in zip(raw_texts, used_prompts)]
        clean_texts = [t for t in clean_texts if len(t) >= 2]

        print(f"Generated {len(clean_texts)} texts")

        gen_df = pd.DataFrame({
            "checkpoint": checkpoint_name,
            "prompt": used_prompts[:len(clean_texts)],
            "text": clean_texts
        })
        gen_df.to_csv(
            f"{Config.OUTPUT_DIR}/gen_{checkpoint_name}.csv",
            index=False,
            encoding="utf-8-sig"
        )

        _, metrics = analyzer.analyze_corpus(clean_texts)
        metrics["checkpoint"] = checkpoint_name
        metrics["stage"] = stage
        metrics["step"] = str(step)

        all_results.append(metrics)

        del generator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(all_results)

    if results_df.empty:
        print("⚠️ 没有任何 checkpoint 成功产出结果。")
        return

    results_df.to_csv(f"{Config.OUTPUT_DIR}/full_results.csv", index=False, encoding="utf-8-sig")
    print("✓ Saved: full_results.csv")

    visualize_and_conclude(results_df)

    print("\n" + "=" * 70)
    print("Complete! Check results/ folder")
    print("=" * 70)


if __name__ == "__main__":
    main()