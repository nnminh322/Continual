"""
Extract frozen encoder embeddings from T5 models for routing analysis.
Models: flan-t5-large, flan-t5-xl
Benchmarks: Long_Sequence (15 tasks), SuperNI (15 tasks)

Output: embeddings/{model_name}/{benchmark}/{task_name}/{split}.npz
  - embeddings: (N, d_model) float32  — avg-pooled encoder last hidden state
  - labels: (N,) object array         — output labels as strings

Usage:
  python extract_embeddings_t5.py --model google/flan-t5-large
  python extract_embeddings_t5.py --model google/flan-t5-xl
  python extract_embeddings_t5.py --model google/flan-t5-small  # debug
"""

import argparse
import json
import os
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, T5EncoderModel


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
    """Load Long_Sequence format: Definition + instances."""
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
    """Load SuperNI format: Definition + 'Now complete...' + instances."""
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
    model: T5EncoderModel,
    tokenizer,
    texts: list[str],
    batch_size: int = 32,
    max_length: int = 512,
    device: str = "cuda",
    desc: str = "batches",
) -> np.ndarray:
    """
    Average-pooled encoder last hidden state.
    Returns (N, d_model) float32 array (full precision, no quantization).
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

        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        hidden = out.last_hidden_state  # (B, L, d)
        mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, L, 1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, d)
        all_embs.append(pooled.cpu().float().numpy())

    return np.concatenate(all_embs, axis=0)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract T5 encoder embeddings")
    parser.add_argument("--model", type=str, default="google/flan-t5-large",
                        help="HF model name: google/flan-t5-{small,large,xl}")
    parser.add_argument("--data_root", type=str, default="CL_Benchmark",
                        help="Path to CL_Benchmark/ directory")
    parser.add_argument("--output_dir", type=str, default="embeddings",
                        help="Output root directory")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--benchmarks", type=str, nargs="+",
                        default=["Long_Sequence", "SuperNI"],
                        choices=["Long_Sequence", "SuperNI"])
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detect if not set)")
    args = parser.parse_args()

    # Device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model name for output dir
    model_short = args.model.split("/")[-1]  # e.g. "flan-t5-large"
    print(f"=== Model: {args.model} ({model_short}) on {args.device} ===")

    # Warn if GPU compute capability is low (common cause of kernel-image errors)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        try:
            prop = torch.cuda.get_device_properties(0)
            if prop.major < 7:
                print(f"Warning: Found GPU compute capability {prop.major}.{prop.minor}. "
                      "Installed PyTorch may not support this device (sm_<7).")
        except Exception:
            pass

    # Load tokenizer and model with safe fallback to CPU on CUDA/kernel failures
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print("Loading encoder model...")
    try:
        # Load on CPU first then move to target device; makes retry simpler
        model = T5EncoderModel.from_pretrained(args.model)
        model.to(args.device)
        model.eval()
    except Exception as e:
        print(f"Warning: failed to load model on {args.device}: {e}")
        if args.device.startswith("cuda"):
            print("Falling back to CPU. To use GPU, install a PyTorch build compatible with your GPU (see README).")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            args.device = "cpu"
            model = T5EncoderModel.from_pretrained(args.model)
            model.eval()
        else:
            raise

    d_model = model.config.d_model
    print(f"d_model = {d_model}")

    # Process benchmarks
    for bench_name in args.benchmarks:
        tasks = BENCHMARKS[bench_name]
        print(f"\n--- Benchmark: {bench_name} ({len(tasks)} tasks) ---")

        loader_fn = load_long_seq if bench_name == "Long_Sequence" else load_superni

        task_pbar = tqdm(tasks, total=len(tasks), unit="task", position=0, leave=True)
        for task_name in task_pbar:
            task_pbar.set_description(f"[{bench_name}] {task_name:40s}")
            task_dir = Path(args.data_root) / bench_name / task_name
            if not task_dir.exists():
                task_pbar.write(f"  [SKIP] {task_name}: not found at {task_dir}")
                continue

            out_dir = Path(args.output_dir) / model_short / bench_name / task_name
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

                try:
                    embs = extract_embeddings(
                        model, tokenizer, texts,
                        batch_size=args.batch_size,
                        max_length=args.max_length,
                        device=args.device,
                        desc=f"{task_name}/{split}",
                    )
                except Exception as e:
                    err = str(e).lower()
                    # Common indicator of incompatible PyTorch/CUDA build
                    if args.device.startswith("cuda") and ("kernel image" in err or "no kernel image" in err or "cuda" in err or "acceleratorerror" in err):
                        task_pbar.write(f"  [WARN] CUDA error during extraction: {e}. Retrying on CPU...")
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        # move model to CPU and retry
                        model = model.to("cpu")
                        args.device = "cpu"
                        embs = extract_embeddings(
                            model, tokenizer, texts,
                            batch_size=args.batch_size,
                            max_length=args.max_length,
                            device="cpu",
                            desc=f"{task_name}/{split}",
                        )
                    else:
                        raise

                np.savez_compressed(
                    out_path,
                    embeddings=embs,                      # (N, d_model) float32
                    labels=np.array(labels, dtype=object), # (N,) strings
                )
                task_pbar.write(f"  -> saved {out_path} {embs.shape}")

    print("\nDone.")


if __name__ == "__main__":
    main()
