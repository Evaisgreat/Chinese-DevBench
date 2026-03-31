import pandas as pd
import torch
import numpy as np
from pathlib import Path
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from tqdm import tqdm
import json

# ================== 配置 ==================
class Config:
    AOA_DATA_PATH = "aoa2.xlsx" #wordbank aoa/ Liu et al. (2011)
    CHILDES_PATH = "childe_mandarin_adult_clean.csv"
    OUTPUT_DIR = Path("results")
    CHECKPOINT_DIR = Path("results/checkpoints")

    STAGE1_STEPS = [77, 154, 231, 308, 385, 462, 539, 616, 693, 770]
    STAGE2_STEPS = [570, 1140, 1710, 2280, 2850, 3420, 3990, 4560, 5130, 5700]
    STAGE1_DIR = Path("/Volumes/Lexar/Dev-cn/GPT2checkpoints/stage1_0-3")
    STAGE2_DIR = Path("/Volumes/Lexar/Dev-cn/GPT2checkpoints/stage2_3-6")
    TOKENIZER_PATH = Path("/Volumes/Lexar/Dev-cn/GPT2checkpoints/stage1_0-3/checkpoint-77")

    MAX_SAMPLES = 512
    MIN_SAMPLES = 30
    MIN_CONTEXT = 6

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 32
    SAVE_INTERVAL = 30

config = Config()
config.OUTPUT_DIR.mkdir(exist_ok=True)
config.CHECKPOINT_DIR.mkdir(exist_ok=True)


# ================== 工具函数 ==================

def get_all_checkpoints():
    checkpoints = []
    for step in config.STAGE1_STEPS:
        path = config.STAGE1_DIR / f"checkpoint-{step}"
        checkpoints.append({'stage': 'Stage1', 'step': step, 'global_step': step, 'path': path})

    stage1_last = config.STAGE1_STEPS[-1]
    for step in config.STAGE2_STEPS:
        path = config.STAGE2_DIR / f"checkpoint-{step}"
        checkpoints.append({'stage': 'Stage2', 'step': step, 'global_step': stage1_last + step, 'path': path})

    return checkpoints


def load_model(checkpoint_path):
    model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
    model.to(config.DEVICE)
    model.eval()
    return model


def find_word_token_positions_by_offset(sentence, word_start_char, word_end_char, tokenizer):
    encoding = tokenizer(
        sentence,
        return_offsets_mapping=True,
        add_special_tokens=False,
        return_tensors=None
    )
    token_ids = encoding['input_ids']
    offset_mapping = encoding['offset_mapping']

    word_token_positions = [
        idx for idx, (start, end) in enumerate(offset_mapping)
        if start < word_end_char and end > word_start_char
    ]

    if not word_token_positions:
        return None, None

    return word_token_positions, token_ids


def extract_word_samples(word, childes_df, tokenizer, max_samples=512, min_samples=50):
    samples = []
    seen_sentences = set()

    for _, row in childes_df.iterrows():
        sentence = str(row['utterance_norm'])

        if word not in sentence or sentence in seen_sentences:
            continue
        seen_sentences.add(sentence)

        pos = sentence.find(word)
        word_start_char = pos
        word_end_char = pos + len(word)

        word_token_positions, token_ids = find_word_token_positions_by_offset(
            sentence, word_start_char, word_end_char, tokenizer
        )

        if word_token_positions is None:
            continue

        if min(word_token_positions) < config.MIN_CONTEXT:
            continue

        samples.append({
            'sentence': sentence,
            'token_ids': token_ids,
            'word_token_positions': word_token_positions
        })

        if len(samples) >= max_samples:
            break

    return samples if len(samples) >= min_samples else []


def compute_surprisal_batch(model, samples_token_ids, word_token_positions_list):
    surprisals = []

    max_len = max(len(s) for s in samples_token_ids)
    batch_size = len(samples_token_ids)

    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)

    for i, token_ids in enumerate(samples_token_ids):
        length = len(token_ids)
        input_ids[i, :length] = torch.tensor(token_ids, dtype=torch.long)
        attention_mask[i, :length] = 1

    input_ids = input_ids.to(config.DEVICE)
    attention_mask = attention_mask.to(config.DEVICE)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        log_probs = torch.log_softmax(outputs.logits, dim=-1)

    for i, word_positions in enumerate(word_token_positions_list):
        token_surprisals = []
        for pos in word_positions:
            if pos == 0:
                continue
            target_token = input_ids[i, pos].item()
            target_log_prob = log_probs[i, pos - 1, target_token].item()
            token_surprisals.append(-target_log_prob / np.log(2))

        surprisals.append(float(np.mean(token_surprisals)) if token_surprisals else np.nan)

    return surprisals


def process_word(word, childes_df, checkpoints, tokenizer):
    samples = extract_word_samples(
        word, childes_df, tokenizer,
        max_samples=config.MAX_SAMPLES,
        min_samples=config.MIN_SAMPLES
    )

    if not samples:
        return None

    print(f"  找到 {len(samples)} 个有效样本")

    sample_token_ids = [s['token_ids'] for s in samples]
    sample_word_positions = [s['word_token_positions'] for s in samples]
    baseline_surprisal = float(np.log2(tokenizer.vocab_size))

    checkpoint_results = []

    for ckpt in tqdm(checkpoints, desc="  Checkpoints", leave=False):
        model = load_model(ckpt['path'])

        all_surprisals = []
        for batch_start in range(0, len(samples), config.BATCH_SIZE):
            batch_end = min(batch_start + config.BATCH_SIZE, len(samples))
            batch_surprisals = compute_surprisal_batch(
                model,
                sample_token_ids[batch_start:batch_end],
                sample_word_positions[batch_start:batch_end]
            )
            all_surprisals.extend(batch_surprisals)

        valid = [s for s in all_surprisals if not np.isnan(s)]

        checkpoint_results.append({
            'stage': ckpt['stage'],
            'step': ckpt['step'],
            'global_step': ckpt['global_step'],
            'mean_surprisal': float(np.mean(valid)) if valid else None,
            'min_surprisal': float(np.min(valid)) if valid else None,
            'max_surprisal': float(np.max(valid)) if valid else None,
            'n_valid_samples': len(valid)
        })

        del model
        torch.cuda.empty_cache()

    return {
        'word': word,
        'n_samples': len(samples),
        'baseline_surprisal': baseline_surprisal,
        'vocab_size': tokenizer.vocab_size,
        'checkpoints': checkpoint_results
    }


def save_checkpoint_results(results, checkpoint_id):
    output_file = config.CHECKPOINT_DIR / f"results_checkpoint_{checkpoint_id}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"已保存中间结果: {output_file}")


def load_existing_results():
    all_results = {}
    for ckpt_file in config.CHECKPOINT_DIR.glob("results_checkpoint_*.json"):
        with open(ckpt_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                all_results[item['word']] = item
    return all_results


# ================== 主函数 ==================

def main():
    print("=" * 60)
    print("提取词汇学习曲线")
    print("=" * 60)

    print("\n[1] 加载数据...")
    aoa_df = pd.read_excel(config.AOA_DATA_PATH)
    childes_df = pd.read_csv(config.CHILDES_PATH)
    target_words = aoa_df['Name'].tolist()
    print(f"  目标词数量: {len(target_words)}")
    print(f"  CHILDES 句子数: {len(childes_df)}")

    checkpoints = get_all_checkpoints()
    print(f"\n[2] Checkpoints: {len(checkpoints)} 个")

    print(f"\n[3] 加载 tokenizer...")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(config.TOKENIZER_PATH / "tokenizer.json")
    )

    print("\n[4] 检查已有结果...")
    existing_results = load_existing_results()
    print(f"  已完成词数: {len(existing_results)}")

    remaining_words = [w for w in target_words if w not in existing_results]
    print(f"  剩余待处理: {len(remaining_words)}")

    if not remaining_words:
        print("\n所有词已处理完成！")
        return

    print("\n[5] 开始处理词汇...")
    all_results = list(existing_results.values())

    for i, word in enumerate(tqdm(remaining_words, desc="处理词汇")):
        print(f"\n处理词 [{i+1}/{len(remaining_words)}]: {word}")

        result = process_word(word, childes_df, checkpoints, tokenizer)

        if result is not None:
            all_results.append(result)
            if (i + 1) % config.SAVE_INTERVAL == 0:
                save_checkpoint_results(all_results, len(existing_results) + i + 1)
        else:
            print(f"  ⚠️  词 '{word}' 样本不足，已跳过")

    print("\n[6] 保存最终结果...")
    final_output = config.OUTPUT_DIR / "word_learning_curves.json"
    with open(final_output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"  已保存: {final_output}")

    csv_rows = []
    for word_data in all_results:
        for ckpt in word_data['checkpoints']:
            csv_rows.append({
                'word': word_data['word'],
                'n_samples': word_data['n_samples'],
                'baseline_surprisal': word_data['baseline_surprisal'],
                'stage': ckpt['stage'],
                'step': ckpt['step'],
                'global_step': ckpt['global_step'],
                'mean_surprisal': ckpt['mean_surprisal'],
                'min_surprisal': ckpt['min_surprisal'],
                'max_surprisal': ckpt['max_surprisal'],
                'n_valid_samples': ckpt['n_valid_samples']
            })

    csv_output = config.OUTPUT_DIR / "word_learning_curves.csv"
    pd.DataFrame(csv_rows).to_csv(csv_output, index=False, encoding='utf-8-sig')
    print(f"  已保存 CSV: {csv_output}")

    print("\n✅ 完成！")


if __name__ == "__main__":
    main()