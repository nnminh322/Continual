"""
Extract frozen embeddings from LLaMA models for routing analysis.
Models: meta-llama/Llama-2-7b-hf, Llama-2-7b-chat-hf, Llama-2-13b-hf,
        meta-llama/Llama-3.1-8B (or Meta-Llama-3-8B)

Strategy: Use last hidden state of the LAST non-padding token (LLaMA is decoder-only,
left-padded → rightmost real token carries the most context).
Alternative: average pool all non-padding tokens (set --pool avg).

Output: embeddings/{model_name}/{benchmark}/{task_name}/{split}.npz
  - embeddings: (N, d_model) float32
  - labels: (N,) object array

Usage:
  python extract_embeddings_llama.py --model meta-llama/Llama-2-7b-hf
  python extract_embeddings_llama.py --model meta-llama/Llama-2-7b-chat-hf
  python extract_embeddings_llama.py --model meta-llama/Llama-2-13b-hf
  python extract_embeddings_llama.py --model meta-llama/Llama-3.1-8B
  python extract_embeddings_llama.py --model meta-llama/Llama-2-7b-hf --pool avg
"""

import argparse
import json
import os
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# ── Benchmark definitions ──────────────────────────────────────────────

LONG_SEQ_TASKS = [
    "yelp", "amazon", "mnli", "cb", "copa", "qqp", "rte",
    "imdb", "sst2", "dbpedia", "agnews", "yahoo", "multirc", "boolq", "wic",
]

SUPERNI_TASKS = [
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
]

BENCHMARKS = {
    "Long_Sequence": LONG_SEQ_TASKS,
    "SuperNI": SUPERNI_TASKS,
}


# ── Data loading (mirrors cl_dataset.py logic) ────────────────────────

def load_long_seq(json_path: str):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    definition = data["Definition"][0].strip()
    template = f"{definition}\n{{0}}\nOutput: "
    texts, labels = [], []
    for inst in data["Instances"]:
        texts.append(template.format(inst["input"]))
        out = inst["output"]
        labels.append(out if isinstance(out, str) else out[0])
    return texts, labels


def load_superni(json_path: str):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    definition = data["Definition"][0].strip()
    template = f"Definition: {definition}\n\nNow complete the following example -\nInput: {{0}}\nOutput: "
    texts, labels = [], []
    for inst in data["Instances"]:
        texts.append(template.format(inst["input"]))
        out = inst["output"]
        labels.append(out if isinstance(out, str) else out[0])
    return texts, labels


# ── Embedding extraction ──────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int = 8,
    max_length: int = 512,
    device: str = "cuda",
    pool: str = "last",
    desc: str = "batches",
) -> np.ndarray:
    """
    Extract embeddings from LLaMA (decoder-only).
    pool='last': last non-padding token's hidden state
    pool='avg':  average pool over non-padding tokens
    Returns (N, d_model) float32 array (cast from bfloat16, no quantization).
    """
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size),
                  desc=f"    {desc}", unit="batch", leave=False, position=1):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        out = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            output_hidden_states=True,
        )
        hidden = out.hidden_states[-1]  # (B, L, d)
        mask = enc["attention_mask"]  # (B, L)

        if pool == "last":
            # Index of last non-padding token per sample
            seq_lens = mask.sum(dim=1) - 1  # (B,)
            pooled = hidden[torch.arange(hidden.size(0), device=device), seq_lens]
        elif pool == "avg":
            mask_f = mask.unsqueeze(-1).float()
            pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        else:
            raise ValueError(f"Unknown pool={pool}")

        all_embs.append(pooled.cpu().float().numpy())

    return np.concatenate(all_embs, axis=0)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract LLaMA embeddings")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="HF model name")
    parser.add_argument("--data_root", type=str, default="CL_Benchmark",
                        help="Path to CL_Benchmark/ directory")
    parser.add_argument("--output_dir", type=str, default="embeddings",
                        help="Output root directory")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--pool", type=str, default="last",
                        choices=["last", "avg"],
                        help="Pooling: 'last' (last token) or 'avg' (mean pool)")
    parser.add_argument("--benchmarks", type=str, nargs="+",
                        default=["Long_Sequence", "SuperNI"],
                        choices=["Long_Sequence", "SuperNI"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace access token for gated models")
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    model_short = args.model.split("/")[-1]
    pool_suffix = f"_pool{args.pool}" if args.pool != "last" else ""
    print(f"=== Model: {args.model} ({model_short}) on {args.device} ===")
    print(f"    Pooling: {args.pool}")

    # Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, token=args.token, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # LLaMA convention

    # Model — full precision: bfloat16 (LLaMA native dtype, no quantization)
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        token=args.token,
    ).eval()

    d_model = model.config.hidden_size
    print(f"d_model = {d_model}")

    # Process benchmarks
    for bench_name in args.benchmarks:
        tasks = BENCHMARKS[bench_name]
        print(f"\n--- Benchmark: {bench_name} ({len(tasks)} tasks) ---")

        loader_fn = load_long_seq if bench_name == "Long_Sequence" else load_superni

        task_pbar = tqdm(tasks, total=len(tasks), unit="task", position=0, leave=True)
        for task_name in task_pbar:
            task_pbar.set_description(f"[{bench_name}] {task_name:50s}")
            task_dir = Path(args.data_root) / bench_name / task_name
            if not task_dir.exists():
                task_pbar.write(f"  [SKIP] {task_name}: not found at {task_dir}")
                continue

            out_dir = (
                Path(args.output_dir) / f"{model_short}{pool_suffix}"
                / bench_name / task_name
            )
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
                task_pbar.write(f"  {task_name}/{split}: {len(texts)} samples")

                embs = extract_embeddings(
                    model, tokenizer, texts,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    device=args.device,
                    pool=args.pool,
                    desc=f"{task_name}/{split}",
                )
                np.savez_compressed(
                    out_path,
                    embeddings=embs,
                    labels=np.array(labels, dtype=object),
                )
                task_pbar.write(f"  -> saved {out_path} {embs.shape}")

    print("\nDone.")


if __name__ == "__main__":
    main()
