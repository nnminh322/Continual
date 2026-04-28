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
    python extract_embeddings_llama.py --model meta-llama/Llama-2-7b-hf --layer hidden_depth_suite
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


def resolve_data_root(data_root_arg: str, benchmarks: list[str]) -> Path:
    requested = Path(data_root_arg).expanduser()
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    candidate_paths = []
    if requested.is_absolute():
        candidate_paths.append(requested)
    else:
        candidate_paths.extend([
            requested,
            Path.cwd() / requested,
            script_dir / requested,
            repo_root / requested,
        ])
        if requested == Path("CL_Benchmark"):
            candidate_paths.extend([
                script_dir / "CL_Benchmark",
                repo_root / "CL_Benchmark",
                repo_root / "root_gainlora" / "CL_Benchmark",
                repo_root / "new_gainlora" / "CL_Benchmark",
            ])

    seen = set()
    deduped_candidates = []
    for candidate in candidate_paths:
        try:
            normalized = candidate.resolve(strict=False)
        except Exception:
            normalized = candidate
        key = str(normalized)
        if key in seen:
            continue
        seen.add(key)
        deduped_candidates.append(candidate)

    for candidate in deduped_candidates:
        if candidate.exists() and candidate.is_dir() and any((candidate / bench).exists() for bench in benchmarks):
            return candidate

    candidate_text = "\n".join(f"  - {candidate}" for candidate in deduped_candidates)
    raise FileNotFoundError(
        f"Could not locate data_root={data_root_arg!r}. Checked:\n{candidate_text}"
    )

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
    "hidden",
    "embedding",
    "hidden_final",
    "hidden_half",
    "hidden_quarter",
    "hidden_eighth",
    "hidden_sixteenth",
    "hidden_depth_suite",
]


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

def resolve_layer_specs(layer: str):
    hidden_specs = {
        "hidden": [{
            "mode": "hidden",
            "profile": "final",
            "metadata_layer": "hidden",
            "dir_suffix": "",
        }],
        "hidden_final": [{
            "mode": "hidden",
            "profile": "final",
            "metadata_layer": "hidden_final",
            "dir_suffix": "_layer-final",
        }],
        "hidden_half": [{
            "mode": "hidden",
            "profile": "half",
            "metadata_layer": "hidden_half",
            "dir_suffix": "_layer-1of2",
        }],
        "hidden_quarter": [{
            "mode": "hidden",
            "profile": "quarter",
            "metadata_layer": "hidden_quarter",
            "dir_suffix": "_layer-1of4",
        }],
        "hidden_eighth": [{
            "mode": "hidden",
            "profile": "eighth",
            "metadata_layer": "hidden_eighth",
            "dir_suffix": "_layer-1of8",
        }],
        "hidden_sixteenth": [{
            "mode": "hidden",
            "profile": "sixteenth",
            "metadata_layer": "hidden_sixteenth",
            "dir_suffix": "_layer-1of16",
        }],
        "hidden_depth_suite": [
            {
                "mode": "hidden",
                "profile": "final",
                "metadata_layer": "hidden_final",
                "dir_suffix": "_layer-final",
            },
            {
                "mode": "hidden",
                "profile": "half",
                "metadata_layer": "hidden_half",
                "dir_suffix": "_layer-1of2",
            },
            {
                "mode": "hidden",
                "profile": "quarter",
                "metadata_layer": "hidden_quarter",
                "dir_suffix": "_layer-1of4",
            },
            {
                "mode": "hidden",
                "profile": "eighth",
                "metadata_layer": "hidden_eighth",
                "dir_suffix": "_layer-1of8",
            },
            {
                "mode": "hidden",
                "profile": "sixteenth",
                "metadata_layer": "hidden_sixteenth",
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
    if layer not in hidden_specs:
        raise ValueError(f"Unknown layer={layer}")
    return hidden_specs[layer]


def resolve_hidden_layer_index(num_hidden_layers: int, profile: str) -> int:
    if profile == "final":
        return num_hidden_layers
    fraction = HIDDEN_LAYER_FRACTIONS[profile]
    return max(1, min(num_hidden_layers, int(round(num_hidden_layers * fraction))))


def pool_hidden_states(hidden: torch.Tensor, mask: torch.Tensor, pool: str) -> torch.Tensor:
    if pool == "last":
        seq_lens = mask.sum(dim=1) - 1
        return hidden[torch.arange(hidden.size(0), device=hidden.device), seq_lens]
    if pool == "avg":
        mask_f = mask.unsqueeze(-1).float()
        return (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
    raise ValueError(f"Unknown pool={pool}")


def build_output_backbone(model_short: str, pool: str, layer_spec: dict) -> str:
    pool_suffix = f"_pool{pool}" if pool != "last" else ""
    return f"{model_short}{pool_suffix}{layer_spec['dir_suffix']}"


def describe_layer_spec(layer_spec: dict, num_hidden_layers: int | None = None) -> str:
    if layer_spec["mode"] == "embedding":
        return "embedding"
    label = HIDDEN_LAYER_LABELS[layer_spec["profile"]]
    if num_hidden_layers is None:
        return f"hidden@{label}"
    layer_index = resolve_hidden_layer_index(num_hidden_layers, layer_spec["profile"])
    return f"hidden@{label} (L{layer_index}/{num_hidden_layers})"

@torch.no_grad()
def extract_embeddings(
    model,
    tokenizer,
    texts: list[str],
    layer_specs: list[dict],
    batch_size: int = 8,
    max_length: int = 1024,
    device: str = "cuda",
    pool: str = "last",
    desc: str = "batches",
) -> dict[str, np.ndarray]:
    """
    Extract embeddings from LLaMA (decoder-only).
    Hidden-layer specs can target the final state or shallower depths such as
    1/2, 1/4, 1/8, and 1/16 of the transformer depth.
    pool='last': last non-padding token's hidden state
    pool='avg':  average pool over non-padding tokens
    Returns a dict mapping metadata_layer -> (N, d_model) float32 arrays.
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
            add_special_tokens=False,
            return_tensors="pt",
        ).to(device)

        hidden_specs = [spec for spec in layer_specs if spec["mode"] == "hidden"]
        if hidden_specs:
            out = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
            hidden_states = out.hidden_states
            num_hidden_layers = len(hidden_states) - 1
            for spec in hidden_specs:
                layer_index = resolve_hidden_layer_index(num_hidden_layers, spec["profile"])
                hidden = hidden_states[layer_index]
                pooled = pool_hidden_states(hidden, enc["attention_mask"], pool)
                all_embs[spec["metadata_layer"]].append(pooled.cpu().float().numpy())

        for spec in layer_specs:
            if spec["mode"] != "embedding":
                continue
            hidden = model.model.embed_tokens(enc["input_ids"])
            pooled = pool_hidden_states(hidden, enc["attention_mask"], pool)
            all_embs[spec["metadata_layer"]].append(pooled.cpu().float().numpy())

    return {
        key: np.concatenate(value, axis=0)
        for key, value in all_embs.items()
    }


def resolve_torch_dtype(device: str, dtype_name: str):
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp32":
        return torch.float32

    if device == "cuda" and torch.cuda.is_available():
        try:
            major, _minor = torch.cuda.get_device_capability()
        except Exception:
            major = 0
        if major >= 8 and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def probe_cuda_runtime():
    """Verify that the selected torch wheel can execute a trivial CUDA kernel."""
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
                        choices=LAYER_CHOICES,
                        help=(
                            "'hidden'=runtime-aligned final hidden state (default); "
                            "'hidden_depth_suite'=extract final, 1/2, 1/4, 1/8, and 1/16 depth profiles; "
                            "'embedding'=word embedding ablation"
                        ))
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
    parser.add_argument("--allow_cpu_fallback", action="store_true",
                        help="Fallback to CPU if CUDA is visible but cannot execute kernels.")
    parser.add_argument("--token", nargs='?', const='', default=None,
                        help="HuggingFace access token for gated models (omit to use HF_TOKEN env var)")
    args = parser.parse_args()

    # Token fallback: if user omitted --token or gave it without a value,
    # fall back to the HF_TOKEN environment variable (if present).
    if args.token in (None, ''):
        args.token = os.environ.get('HF_TOKEN')

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

    model_dtype = resolve_torch_dtype(args.device, args.dtype)
    layer_specs = resolve_layer_specs(args.layer)
    data_root = resolve_data_root(args.data_root, args.benchmarks)

    model_short = args.model.split("/")[-1]
    print(f"=== Model: {args.model} ({model_short}) on {args.device} ===")
    print(f"    Pooling: {args.pool}")
    print(f"    Layer mode: {args.layer}")
    print(f"    DType: {model_dtype}")
    print(f"    Data root: {data_root}")

    # Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, token=args.token, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # LLaMA convention

    # Model — prefer bf16 on Ampere+, otherwise fp16 on older CUDA GPUs such as P100
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=model_dtype,
        device_map="auto" if args.device == "cuda" else None,
        token=args.token,
    ).eval()

    d_model = model.config.hidden_size
    num_hidden_layers = getattr(model.config, "num_hidden_layers", None)
    print(f"d_model = {d_model}")
    print("    Layer profiles: " + ", ".join(
        describe_layer_spec(spec, num_hidden_layers) for spec in layer_specs
    ))

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
            task_dir = data_root / bench_name / task_name
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
                        / build_output_backbone(model_short, args.pool, spec)
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
                    pool=args.pool,
                    desc=f"{task_name}/{split}",
                )
                for spec in missing_specs:
                    hidden_layer_index = None
                    if spec["mode"] == "hidden" and num_hidden_layers is not None:
                        hidden_layer_index = resolve_hidden_layer_index(num_hidden_layers, spec["profile"])

                    metadata = {
                        "benchmark": bench_name,
                        "task_name": task_name,
                        "split": split,
                        "pool": args.pool,
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
                        "torch_dtype": str(model_dtype).replace("torch.", ""),
                        "max_length": args.max_length,
                        "padding_side": tokenizer.padding_side,
                        "add_special_tokens": False,
                        "superni_prompt_style": args.superni_prompt_style if bench_name == "SuperNI" else None,
                        "runtime_aligned": (
                            bench_name == "SuperNI"
                            and args.superni_prompt_style == "runtime_cl"
                            and spec["metadata_layer"] == "hidden"
                            and args.pool == "last"
                            and args.max_length == 1024
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
