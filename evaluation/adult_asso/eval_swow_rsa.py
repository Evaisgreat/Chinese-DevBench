import argparse
import os
import re
import sys
import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_distances

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from tqdm import tqdm
import matplotlib.pyplot as plt

def die(msg: str, code: int = 1):
    print(msg, file=sys.stderr)
    sys.exit(code)

def require_exists(path: str, name: str):
    if not os.path.exists(path):
        die(f"[FATAL] {name} not found: {path}")

def ckpt_name(path: str) -> str:
    return os.path.basename(os.path.normpath(path))

def extract_step_or_nan(path: str):
    base = ckpt_name(path)
    m = re.search(r"(checkpoint[-_]?)(\d+)", base)
    if m:
        return int(m.group(2))
    m = re.search(r"(step[-_]?)(\d+)", base)
    if m:
        return int(m.group(2))
    m = re.search(r"(\d+)", base)
    if m:
        return int(m.group(1))
    return np.nan

def is_hf_checkpoint_dir(d: str) -> bool:
    if not os.path.isdir(d):
        return False
    if not os.path.exists(os.path.join(d, "config.json")):
        return False

    files = set(os.listdir(d))
    if "pytorch_model.bin" in files or "model.safetensors" in files:
        return True

    if any(fn.startswith("pytorch_model-") and fn.endswith(".bin") for fn in files):
        return True
    if any(fn.endswith(".safetensors") for fn in files):
        return True

    return False

def find_checkpoints(stage_dir: str):
    cands = []
    for cur, dirs, files in os.walk(stage_dir):
        if is_hf_checkpoint_dir(cur):
            cands.append(cur)
    return cands

def load_swow_raw_wide_to_long(path: str, verbose: bool = True):
    require_exists(path, "raw_csv")
    df = pd.read_csv(path)

    if verbose:
        print(f"[check] raw_csv loaded: {path}")
        print(f"[check] shape={df.shape}")
        print(f"[check] columns={list(df.columns)}")

    if "cue" not in df.columns:
        die(f"[FATAL] 'cue' column missing. columns={list(df.columns)}")

    resp_cols = [c for c in df.columns if re.match(r"^R\d+Raw$", str(c))]
    if not resp_cols:
        die(f"[FATAL] response columns like R1Raw/R2Raw/R3Raw not found. columns={list(df.columns)}")

    if verbose:
        print(f"[check] detected response columns: {resp_cols}")

    long_df = df[["cue"] + resp_cols].melt(
        id_vars=["cue"],
        value_vars=resp_cols,
        var_name="slot",
        value_name="response"
    )

    before = len(long_df)
    long_df = long_df.dropna(subset=["cue", "response"])
    long_df["cue"] = long_df["cue"].astype(str).str.strip()
    long_df["response"] = long_df["response"].astype(str).str.strip()
    long_df = long_df[(long_df["cue"] != "") & (long_df["response"] != "")]
    long_df = long_df[~long_df["response"].str.lower().isin(["nan", "none", "null"])]

    after = len(long_df)
    if verbose:
        print(f"[check] responses kept: {after}/{before}")

    long_df["count"] = 1.0
    return long_df[["cue", "response", "count"]]

def build_human_vectors(long_df: pd.DataFrame, top_k: int = 5000, min_cue_count: int = 1, l1_normalize: bool = True):
    resp_freq = long_df.groupby("response")["count"].sum().sort_values(ascending=False)
    vocab = resp_freq.head(top_k).index.tolist()
    vocab_index = {w: i for i, w in enumerate(vocab)}

    agg = long_df.groupby(["cue", "response"])["count"].sum().reset_index()
    cue_total = agg.groupby("cue")["count"].sum()
    keep_cues = cue_total[cue_total >= min_cue_count].index.tolist()
    agg = agg[agg["cue"].isin(keep_cues)]

    cues = sorted(agg["cue"].unique().tolist())
    cue_index = {c: i for i, c in enumerate(cues)}

    human = np.zeros((len(cues), len(vocab)), dtype=np.float32)
    for row in agg.itertuples(index=False):
        j = vocab_index.get(row.response, None)
        if j is None:
            continue
        i = cue_index[row.cue]
        human[i, j] += float(row.count)

    if l1_normalize:
        s = human.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        human = human / s

    return cues, vocab, human

@torch.no_grad()
def embed_cues_batch(model, tokenizer, cues, device="cuda", pooling="mean", batch_size=64, max_length=32):
    embs = []
    for start in tqdm(range(0, len(cues), batch_size), desc="Embedding cues", unit="batch"):
        batch_cues = cues[start:start + batch_size]
        tok = tokenizer(
            batch_cues,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False
        )
        tok = {k: v.to(device) for k, v in tok.items()}

        out = model(**tok, output_hidden_states=True)
        last = out.hidden_states[-1] 

        attn = tok.get("attention_mask", None)
        if attn is None:
            attn = torch.ones(last.shape[:2], device=device)
        attn = attn.unsqueeze(-1).to(last.dtype)

        if pooling == "mean":
            summed = (last * attn).sum(dim=1)
            denom = attn.sum(dim=1).clamp_min(1.0)
            emb = summed / denom
        elif pooling == "last":
            lengths = tok["attention_mask"].sum(dim=1)
            idx = (lengths - 1).clamp_min(0)
            emb = last[torch.arange(last.size(0), device=device), idx]
        else:
            die("[FATAL] pooling must be 'mean' or 'last'")

        emb = emb / (emb.norm(p=2, dim=1, keepdim=True) + 1e-12)
        embs.append(emb.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(embs, axis=0)

def rsa_spearman(human_vectors: np.ndarray, model_vectors: np.ndarray):
    D_h = cosine_distances(human_vectors)
    D_m = cosine_distances(model_vectors)
    idx = np.triu_indices(len(D_h), k=1)
    rho, p = spearmanr(D_h[idx], D_m[idx])
    return float(rho), float(p)

def load_existing(out_csv: str):
    if not os.path.exists(out_csv):
        return None, set()
    try:
        df = pd.read_csv(out_csv)
        if "checkpoint" in df.columns:
            done = set(df["checkpoint"].astype(str).tolist())
            return df, done
    except Exception:
        pass
    return None, set()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_csv", type=str, required=True)
    ap.add_argument("--model_root", type=str, required=True)
    ap.add_argument("--stages", type=str, nargs="+", default=None)
    ap.add_argument("--top_k", type=int, default=5000)
    ap.add_argument("--min_cue_count", type=int, default=1)
    ap.add_argument("--cue_limit", type=int, default=None)
    ap.add_argument("--pooling", type=str, default="mean", choices=["mean", "last"])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=32)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out_csv", type=str, default="rsa_trend.csv")
    ap.add_argument("--out_png", type=str, default="rsa_trend.png")
    ap.add_argument("--skip_done", action="store_true")
    args = ap.parse_args()

    require_exists(args.raw_csv, "raw_csv")
    require_exists(args.model_root, "model_root")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("[env] cwd =", os.getcwd())
    print("[env] device =", device)
    print("[env] raw_csv =", args.raw_csv)
    print("[env] model_root =", args.model_root)

    print("[load] SWOW raw -> long...")
    long_df = load_swow_raw_wide_to_long(args.raw_csv, verbose=True)

    print("[human] building vectors...")
    cues, vocab, human = build_human_vectors(
        long_df, top_k=args.top_k, min_cue_count=args.min_cue_count, l1_normalize=True
    )
    print(f"[human] cues={len(cues)} dims(top_k)={len(vocab)}")

    if args.cue_limit is not None:
        cues = cues[:args.cue_limit]
        human = human[:args.cue_limit]
        print(f"[human] cue_limit applied -> {len(cues)} cues")

    stages = args.stages
    if stages is None:
        stages = [d for d in os.listdir(args.model_root) if os.path.isdir(os.path.join(args.model_root, d))]
        print("[scan] auto stages:", stages)

    all_ckpts = []
    for st in stages:
        stage_dir = os.path.join(args.model_root, st)
        require_exists(stage_dir, f"stage_dir({st})")
        ckpts = find_checkpoints(stage_dir)
        if not ckpts:
            print(f"[warn] no checkpoints under {stage_dir}")
        for c in ckpts:
            all_ckpts.append((st, c))

    if not all_ckpts:
        die("[FATAL] no HF checkpoints found (need config.json + weights).")

    def sort_key(item):
        st, path = item
        step = extract_step_or_nan(path)
        step_key = (0, step) if not (isinstance(step, float) and np.isnan(step)) else (1, 0)
        return (st, step_key, ckpt_name(path))

    all_ckpts = sorted(all_ckpts, key=sort_key)
    print(f"[scan] total checkpoints found: {len(all_ckpts)}")

    existing_df, done = load_existing(args.out_csv)
    results = [] if existing_df is None else existing_df.to_dict("records")
    if args.skip_done and done:
        print(f"[resume] skip_done enabled, already have {len(done)} checkpoints in {args.out_csv}")

    for i, (stage, ckpt_path) in enumerate(all_ckpts, start=1):
        if args.skip_done and ckpt_path in done:
            print(f"[skip] ({i}/{len(all_ckpts)}) {ckpt_path}")
            continue

        step = extract_step_or_nan(ckpt_path)
        name = ckpt_name(ckpt_path)
        print(f"\n[{i}/{len(all_ckpts)}] stage={stage} ckpt={name} step={step}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(ckpt_path, use_fast=True)
        except Exception as e:
            print(f"[error] tokenizer load failed: {e}")
            continue

        try:
            model = AutoModelForCausalLM.from_pretrained(ckpt_path)
            model.to(device)
            model.eval()
        except Exception as e:
            print(f"[error] model load failed: {e}")
            continue

        try:
            model_vecs = embed_cues_batch(
                model, tokenizer, cues,
                device=device, pooling=args.pooling,
                batch_size=args.batch_size, max_length=args.max_length
            )
        except RuntimeError as e:
            print(f"[error] embedding failed (likely OOM). Try --batch_size 32/16. Error: {e}")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"[error] embedding failed: {e}")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        try:
            rho, p = rsa_spearman(human, model_vecs)
        except Exception as e:
            print(f"[error] RSA failed: {e}")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        print(f"[rsa] rho={rho:.6f} p={p:.3e}")

        results.append({
            "stage": stage,
            "checkpoint_index": None,
            "step": step,
            "checkpoint_name": name,
            "checkpoint": ckpt_path,
            "n_cues": len(cues),
            "top_k_dims": args.top_k,
            "pooling": args.pooling,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "rsa_spearman_rho": rho,
            "p_value": p
        })

        out_df = pd.DataFrame(results)

        out_df["_step_is_nan"] = out_df["step"].isna()
        out_df = out_df.sort_values(["stage", "_step_is_nan", "step", "checkpoint_name"]).reset_index(drop=True)
        out_df["checkpoint_index"] = out_df.groupby("stage").cumcount()

        out_df = out_df.drop(columns=["_step_is_nan"])
        out_df.to_csv(args.out_csv, index=False)
        print(f"[save] {args.out_csv}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_df = pd.DataFrame(results)
    if len(out_df) == 0:
        die("[FATAL] no successful checkpoints evaluated.")

    out_df["_step_is_nan"] = out_df["step"].isna()
    out_df = out_df.sort_values(["stage", "_step_is_nan", "step", "checkpoint_name"]).reset_index(drop=True)
    out_df["checkpoint_index"] = out_df.groupby("stage").cumcount()
    out_df = out_df.drop(columns=["_step_is_nan"])
    out_df.to_csv(args.out_csv, index=False)
    print(f"[done] csv saved -> {args.out_csv}")

    try:
        plt.figure()
        for st in sorted(out_df["stage"].unique()):
            sub = out_df[out_df["stage"] == st].sort_values("checkpoint_index")
            x = sub["checkpoint_index"].values
            y = sub["rsa_spearman_rho"].values
            plt.plot(x, y, marker="o", label=st)
        plt.xlabel("Checkpoint index (training order)")
        plt.ylabel("RSA Spearman rho")
        plt.title("RSA trend over checkpoints")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.out_png, dpi=200)
        print(f"[done] plot saved -> {args.out_png}")
    except Exception as e:
        print(f"[warn] plot failed: {e}")

if __name__ == "__main__":
    main()