#!/usr/bin/env python3
"""
TPU-compatible LLaMA embedding extraction for routing_analysis.

This script targets a TPU runtime (torch-xla available) but will also
work on CPU/GPU if TPU isn't present. It runs inference and saves
per-task embeddings as compressed `.npz` files matching the layout used
by the rest of the repo.

Usage (on Kaggle with TPU accelerator enabled):
  python extract_embeddings_llama_tpu.py --model meta-llama/Llama-2-7b-hf --device tpu --token YOUR_HF_TOKEN --batch_size 64

Notes:
- For best TPU throughput, use larger `--batch_size` (e.g. 64/128).
- This script uses a single XLA device (`xm.xla_device()`). For full
  multi-core scaling you'd add `xmp.spawn` + distributed dataloader.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Try to import torch_xla (TPU support). If not available, proceed with CPU/GPU.
try:
    import torch_xla.core.xla_model as xm  # type: ignore
    HAS_XLA = True
except Exception:
    xm = None
    HAS_XLA = False


# ── Data loaders (robust to minor JSON format differences) ───────────
def load_long_seq(json_path: str):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    defs = data.get("Definition", [])
    if isinstance(defs, list) and len(defs) > 0:
        definition = defs[0].strip()
    elif isinstance(defs, str):
        definition = defs.strip()
    else:
        definition = ""

    template = f"{definition}\n{{0}}\nOutput: " if definition else "{0}"

    texts, labels = [], []
    for inst in data.get("Instances", []):
        if isinstance(inst, dict):
            inp = inst.get("input", "") or inst.get("text", "")
            out = inst.get("output", "")
        else:
            try:
                inp = inst[0]
                out = inst[1]
            except Exception:
                inp = str(inst)
                out = ""
        texts.append(template.format(inp))
        labels.append(out if isinstance(out, str) else (out[0] if isinstance(out, (list, tuple)) and out else str(out)))
    return texts, labels


def load_superni(json_path: str):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    defs = data.get("Definition", [])
    if isinstance(defs, list) and len(defs) > 0:
        definition = defs[0].strip()
    elif isinstance(defs, str):
        definition = defs.strip()
    else:
        definition = ""

    template = f"Definition: {definition}\n\nNow complete the following example -\nInput: {{0}}\nOutput: " if definition else "{0}"

    texts, labels = [], []
    for inst in data.get("Instances", []):
        if isinstance(inst, dict):
            inp = inst.get("input", "") or inst.get("text", "")
            out = inst.get("output", "")
        else:
            try:
                inp = inst[0]
                out = inst[1]
            except Exception:
                inp = str(inst)
                out = ""
        texts.append(template.format(inp))
        labels.append(out if isinstance(out, str) else (out[0] if isinstance(out, (list, tuple)) and out else str(out)))
    return texts, labels


BENCHMARKS = {
    "Long_Sequence": [
        "yelp", "amazon", "mnli", "cb", "copa", "qqp", "rte",
        "imdb", "sst2", "dbpedia", "agnews", "yahoo", "multirc", "boolq", "wic",
    ],
    "SuperNI": [
        "task1687_sentiment140_classification",
        "task363_sst2_polarity_classification",
        "task875_emotion_classification",
        "task073_commonsenseqa_answer_generation",
        "task591_sciq_answer_generation",
        "task002_quoref_answer_generation",
        "task1290_xsum_summarization",
        "task1572_samsum_summary",
        "task511_reddit_tifu_long_text_summarization",
        "task181_outcome_extraction",
        "task748_glucose_reverse_cause_event_detection",
        "task1510_evalution_relation_extraction",
        "task639_multi_woz_user_utterance_generation",
        "task1590_diplomacy_text_generation",
        "task1729_personachat_generate_next",
    ],
}


@torch.no_grad()
def extract_embeddings(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int = 32,
    max_length: int = 512,
    device: torch.device | str = "cpu",
    pool: str = "last",
    desc: str = "batches",
) -> np.ndarray:
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"    {desc}", unit="batch", leave=False, position=1):
        batch_texts = texts[i: i + batch_size]
        enc = tokenizer(
            batch_texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        # Move tensors to target device (XLA device uses xm.xla_device())
        if HAS_XLA and isinstance(device, str) and device.lower() in ("tpu", "xla"):
            dev = xm.xla_device()
        else:
            dev = torch.device(device)
        enc = {k: v.to(dev) for k, v in enc.items()}

        out = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            output_hidden_states=True,
        )
        hidden = out.hidden_states[-1]  # (B, L, d)
        mask = enc["attention_mask"]  # (B, L)

        if pool == "last":
            seq_lens = mask.sum(dim=1) - 1
            pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), seq_lens]
        else:
            mask_f = mask.unsqueeze(-1).float()
            pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

        all_embs.append(pooled.cpu().float().numpy())

    return np.concatenate(all_embs, axis=0) if all_embs else np.zeros((0, model.config.hidden_size), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Extract LLaMA embeddings (TPU-ready)")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--data_root", type=str, default="CL_Benchmark")
    parser.add_argument("--output_dir", type=str, default="embeddings")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--pool", type=str, default="last", choices=["last", "avg"])
    parser.add_argument("--benchmarks", type=str, nargs="+", default=["Long_Sequence", "SuperNI"], choices=["Long_Sequence", "SuperNI"])
    parser.add_argument("--device", type=str, default=None, help="cpu|cuda|tpu (or leave None to auto-detect)")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace token if required")
    args = parser.parse_args()

    # Device selection
    if args.device is None:
        if HAS_XLA:
            args.device = "tpu"
        else:
            args.device = "cuda" if torch.cuda.is_available() else "cpu"

    use_xla = args.device.lower() in ("tpu", "xla") and HAS_XLA
    if use_xla:
        dev = xm.xla_device()
        print(f"Using XLA device: {dev}")
    else:
        dev = torch.device(args.device)
        print(f"Using device: {dev}")

    # Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_auth_token=args.token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Model
    print("Loading model (may take a while)...")
    # Load without device_map then move to the target device (works with XLA)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_auth_token=args.token,
    )
    model.eval()
    # Move to device
    model.to(dev)

    d_model = model.config.hidden_size
    print(f"d_model = {d_model}")

    loader_fn = load_long_seq

    for bench_name in args.benchmarks:
        tasks = BENCHMARKS[bench_name]
        print(f"\n--- Benchmark: {bench_name} ({len(tasks)} tasks) ---")
        task_pbar = tqdm(tasks, total=len(tasks), unit="task", position=0, leave=True)
        for task_name in task_pbar:
            task_pbar.set_description(f"[{bench_name}] {task_name:50s}")
            task_dir = Path(args.data_root) / bench_name / task_name
            if not task_dir.exists():
                task_pbar.write(f"  [SKIP] {task_name}: not found at {task_dir}")
                continue

            out_dir = Path(args.output_dir) / model.__class__.__name__ / bench_name / task_name
            out_dir.mkdir(parents=True, exist_ok=True)

            for split in ["train", "dev", "test"]:
                json_path = task_dir / f"{split}.json"
                if not json_path.exists():
                    continue

                out_path = out_dir / f"{split}.npz"
                if out_path.exists():
                    task_pbar.write(f"  [EXISTS] {task_name}/{split} — skip")
                    continue

                texts, labels = loader_fn(str(json_path))
                if len(texts) == 0:
                    task_pbar.write(f"  [EMPTY] {task_name}/{split} — skip")
                    continue

                task_pbar.write(f"  {task_name}/{split}: {len(texts)} samples")

                embs = extract_embeddings(
                    model, tokenizer, texts,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    device=args.device,
                    pool=args.pool,
                    desc=f"{task_name}/{split}",
                )

                np.savez_compressed(out_path, embeddings=embs, labels=np.array(labels, dtype=object))
                task_pbar.write(f"  -> saved {out_path} {embs.shape}")

    print("\nDone.")


if __name__ == "__main__":
    main()
