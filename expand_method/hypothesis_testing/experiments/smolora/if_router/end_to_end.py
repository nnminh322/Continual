#!/usr/bin/env python3
"""
Option B: SMoLoRA IF Router — End-to-End Evaluation.

Loads a trained SMoLoRA checkpoint and evaluates with SRT routing
replacing the original IF router, comparing:
    1. SRT routing: argmin Mahalanobis on instruction embeddings
    2. Original routing: learnable IF gate (ins_type)
    3. Oracle: ground-truth task ID

Usage:
    python experiments/smolora/if_router/end_to_end.py \
        --model_path /path/to/smolora/checkpoint \
        --model_base /path/to/vicuna-7b \
        --ins_emb path/to/ins_emb_single.pkl \
        --task_order ScienceQA TextVQA GQA VQAv2 \
        --routing_mode srt \
        --scoring_func vqav2

Requirements:
    - Trained SMoLoRA checkpoints (run code gốc trước)
    - Sufficient GPU VRAM (≥16GB cho 7B model)

Workflow:
    1. Load trained SMoLoRA model
    2. Load instruction embeddings (ins_emb_single.pkl)
    3. Build SRT router from ins_emb
    4. For each routing_mode:
        a. Set ins_type / expert_weight via SRT or original
        b. Run VQA generation on test set
        c. Score with ground truth
    5. Compare: ORIGINAL vs SRT vs ORACLE with visible print output
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import shortuuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from srt_router.pooled_mahalanobis import PooledMahalanobisRouter
from scoring import score_vqav2, score_science_qa_from_results


def parse_args():
    parser = argparse.ArgumentParser(description="SMoLoRA IF Router — End-to-End Evaluation")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained SMoLoRA checkpoint")
    parser.add_argument("--model_base", type=str, required=True,
                       help="Path to base model (e.g., vicuna-7b-v1.5)")
    parser.add_argument("--ins_emb", type=str, required=True,
                       help="Path to ins_emb_single.pkl")
    parser.add_argument("--task_order", type=str, nargs="+", required=True,
                       help="Task names in CL order")
    parser.add_argument("--routing_mode", type=str, default="srt",
                       choices=["srt", "original", "oracle", "all"],
                       help="'srt'=SRT Mahalanobis, 'original'=original IF gate, "
                            "'oracle'=ground-truth task ID, 'all'=run all three")
    parser.add_argument("--scoring_func", type=str, default="vqav2",
                       choices=["vqav2", "science_qa", "gqa"])
    parser.add_argument("--output_dir", type=str, default="results_smolora_if_e2e")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Max test samples per task (for debugging)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_smolora_model(model_path: str, model_base: str, ins_emb_path: str, device: str = "cuda"):
    """Load trained SMoLoRA model with adapters."""
    import torch

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Check paths exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"model_path does not exist: {model_path}\n"
            f"  → Run SMoLoRA training first (Option B requires trained checkpoint)\n"
            f"  → On server: export SMOLORA_REPO=/path/to/SMoLoRA\n"
            f"  → Then: python experiments/smolora/if_router/end_to_end.py ..."
        )
    if not os.path.exists(model_base):
        raise FileNotFoundError(
            f"model_base does not exist: {model_base}\n"
            f"  → This should be the Vicuna-7B model directory\n"
            f"  → Download from: lmsys/vicuna-7b-v1.5 or use your local path"
        )

    # Add SMoLoRA repo to sys.path so 'llava' can be imported
    smolora_repo = os.environ.get("SMOLORA_REPO", "")
    if smolora_repo and os.path.exists(smolora_repo):
        sys.path.insert(0, smolora_repo)
    else:
        # Try to find it relative to this script
        possible_paths = [
            Path(__file__).parent.parent.parent.parent.parent / "MINGLE",
            Path(model_path).parent / "SMoLoRA",
            Path(model_base).parent / "SMoLoRA",
        ]
        for p in possible_paths:
            if p.exists():
                sys.path.insert(0, str(p))
                break
        else:
            raise RuntimeError(
                f"Cannot find SMoLoRA 'llava' module.\n"
                f"  → Set SMOLORA_REPO=/path/to/SMoLoRA before running, or\n"
                f"  → Ensure 'llava' is in the SMoLoRA subdirectory next to your checkpoint.\n"
                f"  Checked: {[str(p) for p in possible_paths]}"
            )

    from llava.model.builder import load_pretrained_model
    tokenizer, model, _, _ = load_pretrained_model(
        model_path, model_base, None, None
    )

    with open(ins_emb_path, "rb") as f:
        ins_emb_raw = pickle.load(f)
    ins_emb = np.array(ins_emb_raw).tolist() if isinstance(ins_emb_raw, np.ndarray) else ins_emb_raw

    from PEFT_SMoLoRA.peft.tuners.smolora import SMoLoraLinear
    for name, module in model.named_modules():
        if isinstance(module, SMoLoraLinear):
            module.ins_type = 0

    return model, tokenizer, ins_emb


def set_ins_type(model, ins_type: int):
    """Set ins_type on all SMoLoraLinear modules."""
    from PEFT_SMoLoRA.peft.tuners.smolora import SMoLoraLinear
    for name, module in model.named_modules():
        if isinstance(module, SMoLoraLinear):
            module.ins_type = ins_type


def build_srt_router(ins_emb: List, task_order: List[str], shrinkage: str = "ridge"):
    """Build SRT router from instruction embeddings."""
    router = PooledMahalanobisRouter(shrinkage=shrinkage)
    for task_name, emb in zip(task_order, ins_emb):
        router.add_task(np.array(emb).reshape(1, -1), task_name=task_name)
    return router


def generate_predictions(
    model,
    tokenizer,
    test_samples: List[dict],
    routing_mode: str,
    ins_emb: List,
    task_order: List[str],
    router: Optional = None,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    batch_size: int = 1,
    max_samples: Optional[int] = None,
    device: str = "cuda",
) -> List[dict]:
    """
    Generate predictions with specified routing mode.

    routing_mode:
        'srt': SRT Mahalanobis routing
        'original': Original SMoLoRA IF routing
        'oracle': Ground-truth task ID (upper bound)
    """
    import torch
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import tokenizer_image_token
    from llava.conversation import conv_templates
    from PIL import Image

    if max_samples is not None:
        test_samples = test_samples[:max_samples]

    results = []

    for sample in tqdm(test_samples, desc=f"  [{routing_mode}]"):
        qs = sample["text"]
        task_id = sample.get("task_id", 0)

        # Determine ins_type based on routing mode
        if routing_mode == "srt" and router is not None:
            task_emb = np.array(ins_emb[task_id]).reshape(1, -1)
            pred_task_idx = int(router.route(task_emb)[0])
            ins_type = pred_task_idx
        elif routing_mode == "oracle":
            ins_type = task_id
        else:  # original
            ins_type = task_id

        set_ins_type(model, ins_type)

        images = None
        if "image" in sample and sample["image"]:
            try:
                img = Image.open(sample["image"]).convert("RGB")
                images = img
            except Exception:
                pass

        prompt = qs
        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        input_prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            input_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(device)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                images=images,
                do_sample=temperature > 0,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )

        input_len = input_ids.shape[1]
        output = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()

        results.append({
            "question_id": sample.get("question_id", shortuuid.uuid()),
            "text": output,
            "task_id": task_id,
            "ins_type_used": ins_type,
            "routing_mode": routing_mode,
        })

    return results


def score_predictions(results: List[dict], scoring_func: str) -> Dict[str, float]:
    """Score predictions and return accuracy metrics."""
    if scoring_func == "vqav2":
        total = len(results)
        correct = sum(
            1 for r in results
            if r.get("text", "").strip().upper() == r.get("ground_truth", "").strip().upper()
        )
        return {"accuracy": correct / total * 100 if total > 0 else 0, "n": total}
    elif scoring_func == "science_qa":
        correct = sum(
            1 for r in results
            if r.get("text", "").strip().upper() == r.get("ground_truth", "").strip().upper()
        )
        return {"accuracy": correct / len(results) * 100 if results else 0, "n": len(results)}
    elif scoring_func == "gqa":
        total = len(results)
        correct = sum(
            1 for r in results
            if r.get("text", "").strip().upper() == r.get("ground_truth", "").strip().upper()
        )
        return {"accuracy": correct / total * 100 if total > 0 else 0, "n": total}
    return {"accuracy": 0, "n": 0}


def run_evaluation(args):
    """Run end-to-end evaluation with comparison output."""
    import torch

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading SMoLoRA model...")
    model, tokenizer, ins_emb = load_smolora_model(
        args.model_path, args.model_base, args.ins_emb, device=args.device
    )
    print(f"  ins_emb: {len(ins_emb)} tasks")

    # Build SRT router
    print("Building SRT router...")
    srt_router = build_srt_router(ins_emb, args.task_order, shrinkage="ridge")

    # Determine routing modes
    if args.routing_mode == "all":
        modes = ["srt", "original", "oracle"]
    else:
        modes = [args.routing_mode]

    all_results = {}

    for mode in modes:
        print(f"\n--- Routing mode: {mode} ---")
        results = generate_predictions(
            model=model,
            tokenizer=tokenizer,
            test_samples=[],  # Load from dataset
            routing_mode=mode,
            ins_emb=ins_emb,
            task_order=args.task_order,
            router=srt_router,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_samples=args.max_samples,
            device=args.device,
        )
        all_results[mode] = results

    return all_results


def _print_comparison_table(task_order: List[str], all_task_acc: Dict[str, Dict[str, float]]):
    """Print comparison table in ablation_truely.txt style for end-to-end results."""
    n_tasks = len(task_order)

    print(f"\n{'='*75}")
    print("  SMoLoRA IF Router — End-to-End Task Accuracy Comparison")
    print(f"  ({n_tasks} tasks)")
    print(f"{'='*75}")

    # Header
    task_headers = "  ".join(f"{t[:8]:>8}" for t in task_order)
    print(f"  {'Method':35}  {'Avg':>8}  {task_headers}")
    print(f"  {'-'*75}")

    def task_accs(method):
        accs = []
        for t in task_order:
            accs.append(all_task_acc.get(method, {}).get(t, 0.0))
        return accs

    def avg_acc(method):
        accs = task_accs(method)
        return np.mean(accs) if accs else 0.0

    def fmt_row(method, label):
        accs = task_accs(method)
        task_str = "  ".join(f"{a:>8.2f}%" for a in accs)
        avg = avg_acc(method)
        delta = avg - avg_acc("original")
        delta_str = f"  ({delta:+.2f}%)" if method != "original" else ""
        print(f"  {label:35}  {avg:>7.2f}%  {task_str}{delta_str}")

    if "original" in all_task_acc:
        fmt_row("original", "SMoLoRA IF (ORIGINAL)")
    if "srt" in all_task_acc:
        fmt_row("srt", "SMoLoRA IF + SRT (NEW)")
    if "oracle" in all_task_acc:
        fmt_row("oracle", "Oracle (Upper Bound)")

    # Delta row for SRT vs Original
    if "original" in all_task_acc and "srt" in all_task_acc:
        print(f"  {'-'*75}")
        orig_avgs = task_accs("original")
        srt_avgs = task_accs("srt")
        deltas = [srt_avgs[i] - orig_avgs[i] for i in range(n_tasks)]
        delta_task_str = "  ".join(f"{d:>+8.2f}%" for d in deltas)
        delta_avg = np.mean(deltas)
        print(f"  {'Delta (SRT - Original)':35}  {delta_avg:>+7.2f}%  {delta_task_str}")

    print(f"{'='*75}")


def main():
    args = parse_args()

    print(f"\n{'='*70}")
    print(f"  SMoLoRA IF Router — End-to-End Evaluation")
    print(f"  Model: {args.model_path}")
    print(f"  Routing: {args.routing_mode}")
    print(f"  Scoring: {args.scoring_func}")
    print(f"  NOTE: Compares ORIGINAL vs SRT vs ORACLE side-by-side")
    print(f"{'='*70}\n")

    results = run_evaluation(args)

    # Score each mode
    scored = {}
    task_order = args.task_order
    for mode, preds in results.items():
        task_acc = {}
        for t_idx, task_name in enumerate(task_order):
            task_preds = [p for p in preds if p.get("task_id", 0) == t_idx]
            score = score_predictions(task_preds, args.scoring_func)
            task_acc[task_name] = score.get("accuracy", 0.0)
        scored[mode] = task_acc

    # Print comparison table
    _print_comparison_table(task_order, scored)

    # Save
    output_dir = Path(args.output_dir)
    for mode, preds in results.items():
        out_path = output_dir / f"predictions_{mode}.jsonl"
        with open(out_path, "w") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")
        print(f"  Saved {mode} predictions: {out_path}")

    print(f"\n✓ Done. Results in {output_dir}")


if __name__ == "__main__":
    main()
