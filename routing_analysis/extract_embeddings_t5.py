"""
Extract frozen encoder embeddings from T5 models for routing analysis.

Default behavior is aligned with the deployed T5 continual-learning router:
    - SuperNI prompts use the runtime zero-shot instruction template.
    - Embeddings come from frozen T5EncoderModel last_hidden_state.
    - Pooling is mean over non-padding tokens.

Optional ablations remain available via --layer embedding, but runtime-aligned
SRT validation should use the default encoder layer.

Output: embeddings/{model_name}/{benchmark}/{task_name}/{split}.npz
    - embeddings: (N, d_model) float32
    - labels: (N,) object array

Usage:
    python extract_embeddings_t5.py --model google/flan-t5-large
    python extract_embeddings_t5.py --model google/flan-t5-xl
    python extract_embeddings_t5.py --model google/flan-t5-small  # debug
    python extract_embeddings_t5.py --model google/flan-t5-large --layer encoder_depth_suite
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

HIDDEN_LAYER_FRACTIONS = {
    "final": 1.0,
    "half": 0.5,
    "quarter": 0.25,
    "eighth": 0.125,
    "sixteenth": 0.0625,
}

HIDDEN_LAYER_LABELS = {
    "final": "final",
    "half": "1/2",
    "quarter": "1/4",
    "eighth": "1/8",
    "sixteenth": "1/16",
}

LAYER_CHOICES = [
    "encoder",
    "embedding",
    "encoder_final",
    "encoder_half",
    "encoder_quarter",
    "encoder_eighth",
    "encoder_sixteenth",
    "encoder_depth_suite",
]


# ── Data loading (mirrors cl_dataset.py logic) ────────────────────────

def load_long_seq(json_path: str):
    """Load Long_Sequence format: Definition + instances."""
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


def load_superni(json_path: str):
    """Load SuperNI format: Definition + 'Now complete...' + instances."""
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
        template = f"Definition: {definition}\n\nNow complete the following example -\nInput: {{0}}\nOutput: "
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


# ── Embedding extraction ──────────────────────────────────────────────

def resolve_layer_specs(layer: str):
    encoder_specs = {
        "encoder": [{
            "mode": "encoder",
            "profile": "final",
            "metadata_layer": "encoder",
            "dir_suffix": "",
        }],
        "encoder_final": [{
            "mode": "encoder",
            "profile": "final",
            "metadata_layer": "encoder_final",
            "dir_suffix": "_layer-final",
        }],
        "encoder_half": [{
            "mode": "encoder",
            "profile": "half",
            "metadata_layer": "encoder_half",
            "dir_suffix": "_layer-1of2",
        }],
        "encoder_quarter": [{
            "mode": "encoder",
            "profile": "quarter",
            "metadata_layer": "encoder_quarter",
            "dir_suffix": "_layer-1of4",
        }],
        "encoder_eighth": [{
            "mode": "encoder",
            "profile": "eighth",
            "metadata_layer": "encoder_eighth",
            "dir_suffix": "_layer-1of8",
        }],
        "encoder_sixteenth": [{
            "mode": "encoder",
            "profile": "sixteenth",
            "metadata_layer": "encoder_sixteenth",
            "dir_suffix": "_layer-1of16",
        }],
        "encoder_depth_suite": [
            {
                "mode": "encoder",
                "profile": "final",
                "metadata_layer": "encoder_final",
                "dir_suffix": "_layer-final",
            },
            {
                "mode": "encoder",
                "profile": "half",
                "metadata_layer": "encoder_half",
                "dir_suffix": "_layer-1of2",
            },
            {
                "mode": "encoder",
                "profile": "quarter",
                "metadata_layer": "encoder_quarter",
                "dir_suffix": "_layer-1of4",
            },
            {
                "mode": "encoder",
                "profile": "eighth",
                "metadata_layer": "encoder_eighth",
                "dir_suffix": "_layer-1of8",
            },
            {
                "mode": "encoder",
                "profile": "sixteenth",
                "metadata_layer": "encoder_sixteenth",
                "dir_suffix": "_layer-1of16",
            },
        ],
    }
    if layer == "embedding":
        return [{
            "mode": "embedding",
            "profile": None,
            "metadata_layer": "embedding",
            "dir_suffix": "_wordemb",
        }]
    if layer not in encoder_specs:
        raise ValueError(f"Unknown layer={layer}")
    return encoder_specs[layer]


def resolve_hidden_layer_index(num_hidden_layers: int, profile: str) -> int:
    if profile == "final":
        return num_hidden_layers
    fraction = HIDDEN_LAYER_FRACTIONS[profile]
    return max(1, min(num_hidden_layers, int(round(num_hidden_layers * fraction))))


def pool_hidden_states(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.unsqueeze(-1).float()
    return (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)


def build_output_backbone(model_short: str, layer_spec: dict) -> str:
    return f"{model_short}{layer_spec['dir_suffix']}"


def describe_layer_spec(layer_spec: dict, num_hidden_layers: int | None = None) -> str:
    if layer_spec["mode"] == "embedding":
        return "embedding"
    label = HIDDEN_LAYER_LABELS[layer_spec["profile"]]
    if num_hidden_layers is None:
        return f"encoder@{label}"
    layer_index = resolve_hidden_layer_index(num_hidden_layers, layer_spec["profile"])
    return f"encoder@{label} (L{layer_index}/{num_hidden_layers})"

@torch.no_grad()
def extract_embeddings(
    model: T5EncoderModel,
    tokenizer,
    texts: list[str],
    layer_specs: list[dict],
    batch_size: int = 32,
    max_length: int = 512,
    device: str = "cuda",
    desc: str = "batches",
) -> dict[str, np.ndarray]:
    """
    Extract embeddings from T5.
    layer='encoder': avg-pooled encoder final hidden state (runtime-aligned).
    encoder_depth_suite additionally extracts 1/2, 1/4, 1/8, and 1/16 depth.
    """
    all_embs = {spec["metadata_layer"]: [] for spec in layer_specs}
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

        encoder_specs = [spec for spec in layer_specs if spec["mode"] == "encoder"]
        if encoder_specs:
            out = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=True,
            )
            hidden_states = out.hidden_states
            num_hidden_layers = len(hidden_states) - 1
            for spec in encoder_specs:
                layer_index = resolve_hidden_layer_index(num_hidden_layers, spec["profile"])
                hidden = hidden_states[layer_index]
                pooled = pool_hidden_states(hidden, enc["attention_mask"])
                all_embs[spec["metadata_layer"]].append(pooled.cpu().float().numpy())

        for spec in layer_specs:
            if spec["mode"] != "embedding":
                continue
            hidden = model.encoder.embed_tokens(enc["input_ids"])
            pooled = pool_hidden_states(hidden, enc["attention_mask"])
            all_embs[spec["metadata_layer"]].append(pooled.cpu().float().numpy())

    return {
        key: np.concatenate(value, axis=0)
        for key, value in all_embs.items()
    }


def probe_cuda_runtime():
    """Verify that the installed torch wheel can run a trivial CUDA kernel."""
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() returned False"

    try:
        device = torch.device("cuda:0")
        probe = torch.zeros(8, device=device, dtype=torch.float32)
        probe = probe + 1.0
        _ = probe.sum().item()
        torch.cuda.synchronize(device)
        return True, None
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"


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
    parser.add_argument("--layer", type=str, default="encoder",
                        choices=LAYER_CHOICES,
                        help=(
                            "'encoder'=runtime-aligned final encoder state (default); "
                            "'encoder_depth_suite'=extract final, 1/2, 1/4, 1/8, and 1/16 encoder depth profiles; "
                            "'embedding'=word embedding ablation"
                        ))
    parser.add_argument("--allow_cpu_fallback", action="store_true",
                        help="Fallback to CPU if CUDA is visible but cannot execute kernels.")
    args = parser.parse_args()

    # Device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.device == "cuda":
        cuda_ok, cuda_error = probe_cuda_runtime()
        if not cuda_ok:
            gpu_name = "unknown GPU"
            try:
                if torch.cuda.device_count() > 0:
                    gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                pass

            message = (
                f"CUDA is visible but cannot execute kernels on {gpu_name}. "
                f"This usually means the installed PyTorch wheel does not support this GPU architecture. "
                f"Original probe error: {cuda_error}"
            )
            if args.allow_cpu_fallback:
                print(f"[WARN] {message}")
                print("[WARN] Falling back to CPU because --allow_cpu_fallback was set.")
                args.device = "cpu"
            else:
                raise RuntimeError(
                    message +
                    " Re-run with --device cpu or --allow_cpu_fallback, or switch Kaggle GPU to T4/L4/A100."
                )

    layer_specs = resolve_layer_specs(args.layer)

    model_short = args.model.split("/")[-1]  # e.g. "flan-t5-large"
    print(f"=== Model: {args.model} ({model_short}) on {args.device} ===")
    print(f"    Layer mode: {args.layer}")

    # Load tokenizer and model
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print("Loading encoder model...")
    model = T5EncoderModel.from_pretrained(args.model).to(args.device).eval()

    d_model = model.config.d_model
    num_hidden_layers = getattr(model.config, "num_layers", None)
    print(f"d_model = {d_model}")
    print("    Layer profiles: " + ", ".join(
        describe_layer_spec(spec, num_hidden_layers) for spec in layer_specs
    ))

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

            for split in ["train", "dev", "test"]:
                json_path = task_dir / f"{split}.json"
                if not json_path.exists():
                    continue

                out_paths = {}
                missing_specs = []
                for spec in layer_specs:
                    out_dir = (
                        Path(args.output_dir)
                        / build_output_backbone(model_short, spec)
                        / bench_name
                        / task_name
                    )
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{split}.npz"
                    out_paths[spec["metadata_layer"]] = out_path
                    if not out_path.exists():
                        missing_specs.append(spec)

                if not missing_specs:
                    task_pbar.write(f"  [EXISTS] {task_name}/{split} — skip")
                    continue

                texts, labels = loader_fn(str(json_path))
                task_pbar.write(f"  {task_name}/{split}: {len(texts)} samples")

                embs_by_layer = extract_embeddings(
                    model, tokenizer, texts,
                    layer_specs=missing_specs,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    device=args.device,
                    desc=f"{task_name}/{split}",
                )

                for spec in missing_specs:
                    hidden_layer_index = None
                    if spec["mode"] == "encoder" and num_hidden_layers is not None:
                        hidden_layer_index = resolve_hidden_layer_index(num_hidden_layers, spec["profile"])

                    metadata = {
                        "benchmark": bench_name,
                        "task_name": task_name,
                        "split": split,
                        "backbone_family": "t5",
                        "pool": "avg",
                        "layer": spec["metadata_layer"],
                        "layer_family": spec["mode"],
                        "hidden_layer_profile": spec["profile"],
                        "hidden_layer_fraction": (
                            HIDDEN_LAYER_FRACTIONS[spec["profile"]]
                            if spec["profile"] is not None else None
                        ),
                        "hidden_layer_index": hidden_layer_index,
                        "hidden_layer_label": (
                            HIDDEN_LAYER_LABELS[spec["profile"]]
                            if spec["profile"] is not None else None
                        ),
                        "num_hidden_layers": num_hidden_layers,
                        "max_length": args.max_length,
                        "add_special_tokens": True,
                        "superni_prompt_style": "runtime_cl" if bench_name == "SuperNI" else None,
                        "runtime_aligned": (
                            bench_name == "SuperNI"
                            and spec["metadata_layer"] == "encoder"
                            and args.max_length == 512
                        ),
                    }

                    out_path = out_paths[spec["metadata_layer"]]
                    embs = embs_by_layer[spec["metadata_layer"]]
                    np.savez_compressed(
                        out_path,
                        embeddings=embs,
                        labels=np.array(labels, dtype=object),
                        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
                    )
                    task_pbar.write(
                        f"  -> saved {out_path} {embs.shape} [{describe_layer_spec(spec, num_hidden_layers)}]"
                    )

    print("\nDone.")


if __name__ == "__main__":
    main()
