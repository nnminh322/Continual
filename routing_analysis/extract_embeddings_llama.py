"""
Extract frozen embeddings from LLaMA models for routing analysis.

Default behavior is aligned with the deployed LLaMA continual-learning router:
    - SuperNI prompts are built with the same zero-shot instruction template.
    - Tokenization uses add_special_tokens=False, matching the CL collator.
    - Embeddings come from the final hidden state at the last non-padding token.

Optional ablations remain available via --pool / --layer / --superni_prompt_style,
but they are not treated as runtime-aligned by default.

Output: embeddings/{model_name}/{benchmark}/{task_name}/{split}.npz
    - embeddings: (N, d_model) float32
    - labels: (N,) object array
    - metadata_json: scalar JSON string describing the extraction profile

Usage:
    python extract_embeddings_llama.py --model meta-llama/Llama-2-7b-hf
    python extract_embeddings_llama.py --model meta-llama/Llama-2-7b-hf --superni_prompt_style runtime_cl
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
    defs = data.get("Definition", [])
    if isinstance(defs, list) and len(defs) > 0:
        definition = defs[0].strip()
    elif isinstance(defs, str):
        definition = defs.strip()
    else:
        definition = ""

    if definition:
        template = f"{definition}\n{{0}}\nOutput: "
    else:
        template = "{0}"

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


def _normalize_output_label(output):
    if isinstance(output, str):
        return output
    if isinstance(output, (list, tuple)) and output:
        first = output[0]
        return first if isinstance(first, str) else str(first)
    return str(output)


def _build_superni_runtime_instruction(data: dict, input_mode: str = "zeroshot") -> str:
    """Mirror new_gainlora/src/cl_dataset.py load_SuperNI_dataset exactly."""
    definition = ""
    if input_mode in {"fewshot", "zeroshot"}:
        raw_definition = data.get("Definition", "")
        if isinstance(raw_definition, list):
            raw_definition = raw_definition[0] if raw_definition else ""
        if raw_definition:
            definition = "Definition: " + str(raw_definition).strip() + "\n\n"

    instruction = ""
    if input_mode in {"fewshot", "zeroshot"}:
        instruction += "Now complete the following example -\n"
    instruction += "Input: {0}\n"
    instruction += "Output: "

    pos_examples = []
    if input_mode == "fewshot":
        for idx, pos_example in enumerate(data.get("Positive Examples", [])[:1]):
            pos_example_str = f"Positive Example {idx + 1} -\n"
            pos_example_str += f"Input: {pos_example['input'].strip()}\n"
            pos_example_str += f"Output: {pos_example['output'].strip()}\n"
            pos_examples.append(pos_example_str)

    return definition + "".join(pos_examples) + instruction


def load_superni(json_path: str, prompt_style: str = "runtime_cl"):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    if prompt_style == "runtime_cl":
        template = _build_superni_runtime_instruction(data, input_mode="zeroshot")
    elif prompt_style == "simple":
        defs = data.get("Definition", [])
        if isinstance(defs, list) and len(defs) > 0:
            definition = defs[0].strip()
        elif isinstance(defs, str):
            definition = defs.strip()
        else:
            definition = ""

        if definition:
            template = f"Definition: {definition}\n\nNow complete the following example -\nInput: {{0}}\nOutput: "
        else:
            template = "{0}"
    else:
        raise ValueError(f"Unknown SuperNI prompt_style={prompt_style}")

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
        labels.append(_normalize_output_label(out))
    return texts, labels


# ── Embedding extraction ──────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int = 8,
    max_length: int = 1024,
    device: str = "cuda",
    pool: str = "last",
    layer: str = "hidden",
    desc: str = "batches",
) -> np.ndarray:
    """
    Extract embeddings from LLaMA (decoder-only).
    layer='hidden': last hidden state (after all transformer blocks, runtime-aligned)
    layer='embedding': word embedding layer (available for ablations only)
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
            add_special_tokens=False,
            return_tensors="pt",
        ).to(device)

        if layer == "embedding":
            # Word embedding layer only — useful for ablations, not runtime-equivalent routing
            hidden = model.model.embed_tokens(enc["input_ids"])  # (B, L, d)
        else:
            out = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
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


def resolve_torch_dtype(device: str, dtype_name: str):
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp32":
        return torch.float32

    if device == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device == "cuda":
        return torch.float16
    return torch.float32


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
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--pool", type=str, default="last",
                        choices=["last", "avg"],
                        help="Pooling: 'last' (last token) or 'avg' (mean pool)")
    parser.add_argument("--layer", type=str, default="hidden",
                        choices=["hidden", "embedding"],
                        help="'hidden'=runtime-aligned last hidden state (default), 'embedding'=ablation only")
    parser.add_argument("--superni_prompt_style", type=str, default="runtime_cl",
                        choices=["runtime_cl", "simple"],
                        help="SuperNI prompt construction. 'runtime_cl' mirrors the deployed CL dataset builder.")
    parser.add_argument("--benchmarks", type=str, nargs="+",
                        default=["Long_Sequence", "SuperNI"],
                        choices=["Long_Sequence", "SuperNI"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=["auto", "fp16", "bf16", "fp32"],
                        help="Model weight dtype. auto picks bf16 when supported, otherwise fp16 on CUDA.")
    parser.add_argument("--token", nargs='?', const='', default=None,
                        help="HuggingFace access token for gated models (omit to use HF_TOKEN env var)")
    args = parser.parse_args()

    # Token fallback: if user omitted --token or gave it without a value,
    # fall back to the HF_TOKEN environment variable (if present).
    if args.token in (None, ''):
        args.token = os.environ.get('HF_TOKEN')

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    model_dtype = resolve_torch_dtype(args.device, args.dtype)

    model_short = args.model.split("/")[-1]
    pool_suffix = f"_pool{args.pool}" if args.pool != "last" else ""
    layer_suffix = "_wordemb" if args.layer == "embedding" else ""
    print(f"=== Model: {args.model} ({model_short}) on {args.device} ===")
    print(f"    Pooling: {args.pool}")
    print(f"    Layer: {args.layer}")
    print(f"    DType: {model_dtype}")

    # Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_auth_token=args.token, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # LLaMA convention

    # Model — full precision: bfloat16 (LLaMA native dtype, no quantization)
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=model_dtype,
        device_map="auto" if args.device == "cuda" else None,
        use_auth_token=args.token,
    ).eval()

    d_model = model.config.hidden_size
    print(f"d_model = {d_model}")

    # Process benchmarks
    for bench_name in args.benchmarks:
        tasks = BENCHMARKS[bench_name]
        print(f"\n--- Benchmark: {bench_name} ({len(tasks)} tasks) ---")

        if bench_name == "Long_Sequence":
            loader_fn = load_long_seq
        else:
            loader_fn = lambda json_path: load_superni(
                json_path,
                prompt_style=args.superni_prompt_style,
            )

        task_pbar = tqdm(tasks, total=len(tasks), unit="task", position=0, leave=True)
        for task_name in task_pbar:
            task_pbar.set_description(f"[{bench_name}] {task_name:50s}")
            task_dir = Path(args.data_root) / bench_name / task_name
            if not task_dir.exists():
                task_pbar.write(f"  [SKIP] {task_name}: not found at {task_dir}")
                continue

            out_dir = (
                Path(args.output_dir) / f"{model_short}{pool_suffix}{layer_suffix}"
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
                    layer=args.layer,
                    desc=f"{task_name}/{split}",
                )
                metadata = {
                    "benchmark": bench_name,
                    "task_name": task_name,
                    "split": split,
                    "pool": args.pool,
                    "layer": args.layer,
                    "torch_dtype": str(model_dtype).replace("torch.", ""),
                    "max_length": args.max_length,
                    "padding_side": tokenizer.padding_side,
                    "add_special_tokens": False,
                    "superni_prompt_style": args.superni_prompt_style if bench_name == "SuperNI" else None,
                    "runtime_aligned": (
                        bench_name == "SuperNI"
                        and args.superni_prompt_style == "runtime_cl"
                        and args.layer == "hidden"
                        and args.pool == "last"
                        and args.max_length == 1024
                    ),
                }
                np.savez_compressed(
                    out_path,
                    embeddings=embs,
                    labels=np.array(labels, dtype=object),
                    metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
                )
                task_pbar.write(f"  -> saved {out_path} {embs.shape}")

    print("\nDone.")


if __name__ == "__main__":
    main()
