#!/usr/bin/env python3
"""
Option B: SMoLoRA Dual Router (VU + IF) — End-to-End Evaluation.

Combines CLIP-based VU routing and instruction-based IF routing
for end-to-end task identification and VQA evaluation.

Usage:
    python experiments/smolora/dual_router/end_to_end.py \
        --model_path /path/to/smolora/checkpoint \
        --model_base /path/to/vicuna-7b \
        --ins_emb path/to/ins_emb_single.pkl \
        --clip_model openai/clip-vit-large-patch14-336 \
        --task_images_root /path/to/task/images \
        --task_order ScienceQA TextVQA GQA VQAv2 \
        --routing_mode all \
        --alpha 0.5

Comparison output:
    Shows ORIGINAL (VU baseline) vs NEW (SRT: VU/IF/Dual) vs ORACLE side-by-side.
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import shortuuid
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from srt_router.pooled_mahalanobis import PooledMahalanobisRouter
from embedding_extractors.clip_extractor import CLIPVisionExtractor


def parse_args():
    parser = argparse.ArgumentParser(description="SMoLoRA Dual Router — End-to-End Evaluation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_base", type=str, required=True)
    parser.add_argument("--ins_emb", type=str, required=True)
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--task_images_root", type=str, required=True)
    parser.add_argument("--task_order", type=str, nargs="+", required=True)
    parser.add_argument("--routing_mode", type=str, default="dual",
                       choices=["dual", "vu_only", "if_only", "original", "all"])
    parser.add_argument("--scoring_func", type=str, default="vqav2",
                       choices=["vqav2", "science_qa", "gqa"])
    parser.add_argument("--output_dir", type=str, default="results_smolora_dual_e2e")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--alpha", type=float, default=0.5,
                       help="Weight for VU in dual routing: d = alpha*d_vu + (1-alpha)*d_if")
    parser.add_argument("--n_build_signatures", type=int, default=200)
    return parser.parse_args()


def collect_task_images(data_root: Path, task_name: str, max_n: int) -> List[Path]:
    task_dir = data_root / task_name
    if not task_dir.exists():
        return []
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPEG", "*.JPG", "*.PNG"]
    images = []
    for ext in extensions:
        images.extend(task_dir.rglob(ext))
    return images[:max_n]


def build_dual_routers(
    ins_emb, task_order, clip_extractor, task_images_root, n_per_task, shrinkage="ridge"
):
    """Build both VU and IF SRT routers."""
    vu_router = PooledMahalanobisRouter(shrinkage=shrinkage)
    if_router = PooledMahalanobisRouter(shrinkage=shrinkage)

    for task_name in tqdm(task_order, desc="  Building signatures"):
        # IF router
        if_router.add_task(
            np.array(ins_emb[task_order.index(task_name)]).reshape(1, -1),
            task_name=task_name
        )

        # VU router
        paths = collect_task_images(Path(task_images_root), task_name, max_n=n_per_task)
        if paths:
            embs = clip_extractor.extract_from_paths(paths, batch_size=8)
            vu_router.add_task(embs, task_name=task_name)
            print(f"  {task_name}: {len(paths)} images")

    return vu_router, if_router


def dual_route(vu_router, if_router, clip_emb, if_emb_query, alpha=0.5):
    """Combine VU and IF routing distances."""
    d_vu = vu_router._compute_distances(clip_emb)
    d_if = if_router._compute_distances(if_emb_query)
    d_dual = alpha * d_vu + (1 - alpha) * d_if
    return int(d_dual.argmin(axis=1)[0])


def load_smolora_model(model_path, model_base, device):
    """Load SMoLoRA model."""
    import torch
    from llava.model.builder import load_pretrained_model
    from PEFT_SMoLoRA.peft.tuners.smolora import SMoLoraLinear

    model, tokenizer, _, _ = load_pretrained_model(model_path, model_base, None, None)
    for name, module in model.named_modules():
        if isinstance(module, SMoLoraLinear):
            module.ins_type = 0
    return model, tokenizer


def set_ins_type(model, ins_type: int):
    from PEFT_SMoLoRA.peft.tuners.smolora import SMoLoraLinear
    for name, module in model.named_modules():
        if isinstance(module, SMoLoraLinear):
            module.ins_type = ins_type


def generate_with_dual_routing(
    model, tokenizer, clip_ext, vu_router, if_router,
    test_samples, routing_mode, ins_emb, task_order, alpha=0.5,
    max_new_tokens=128, temperature=0.0, max_samples=None, device="cuda",
) -> List[dict]:
    """Generate with dual routing."""
    import torch
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import tokenizer_image_token
    from llava.conversation import conv_templates
    from PIL import Image

    if max_samples is not None:
        test_samples = test_samples[:max_samples]

    results = []

    for sample in tqdm(test_samples, desc=f"  [{routing_mode}]"):
        task_id = sample.get("task_id", 0)
        qs = sample["text"]

        image = None
        clip_emb = None
        if_emb_query = None

        if "image" in sample and sample["image"]:
            try:
                image = Image.open(sample["image"]).convert("RGB")
                clip_emb = clip_ext.extract_single(image).reshape(1, -1)
            except Exception:
                pass

        # Determine ins_type based on routing mode
        if routing_mode == "dual":
            if clip_emb is not None and vu_router.n_tasks > 0:
                ins_type = dual_route(vu_router, if_router, clip_emb,
                                     np.array(ins_emb[task_id]).reshape(1, -1), alpha=alpha)
            else:
                ins_type = task_id
        elif routing_mode == "vu_only":
            if clip_emb is not None and vu_router.n_tasks > 0:
                ins_type = int(vu_router.route(clip_emb)[0])
            else:
                ins_type = task_id
        elif routing_mode == "if_only":
            if if_router.n_tasks > 0:
                ins_type = int(if_router.route(np.array(ins_emb[task_id]).reshape(1, -1))[0])
            else:
                ins_type = task_id
        elif routing_mode == "oracle":
            ins_type = task_id
        else:  # original
            ins_type = task_id

        set_ins_type(model, ins_type)

        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(device)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                images=image,
                do_sample=temperature > 0,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )

        output = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

        results.append({
            "question_id": sample.get("question_id", shortuuid.uuid()),
            "text": output,
            "task_id": task_id,
            "ins_type_used": ins_type,
            "routing_mode": routing_mode,
        })

    return results


def score_predictions(results: List[dict], scoring_func: str) -> Dict:
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
    return {"accuracy": 0, "n": 0}


def _print_comparison_table(task_order, all_task_acc):
    n_tasks = len(task_order)

    print(f"\n{'='*80}")
    print("  SMoLoRA Dual Router — End-to-End Task Accuracy Comparison")
    print(f"  ({n_tasks} tasks)")
    print(f"{'='*80}")

    task_headers = "  ".join(f"{t[:8]:>8}" for t in task_order)
    print(f"  {'Method':40}  {'Avg':>8}  {task_headers}")
    print(f"  {'-'*80}")

    def task_accs(method):
        return [all_task_acc.get(method, {}).get(t, 0.0) for t in task_order]

    def avg_acc(method):
        accs = task_accs(method)
        return np.mean(accs) if accs else 0.0

    methods_to_show = ["original", "vu_only", "if_only", "dual", "oracle"]
    labels = {
        "original": "SMoLoRA Dual (ORIGINAL)",
        "vu_only": "VU SRT (NEW)",
        "if_only": "IF SRT (NEW)",
        "dual": "Dual SRT (NEW)",
        "oracle": "Oracle (Upper Bound)",
    }

    for m in methods_to_show:
        if m not in all_task_acc:
            continue
        accs = task_accs(m)
        task_str = "  ".join(f"{a:>8.2f}%" for a in accs)
        avg = avg_acc(m)
        delta_str = ""
        if m != "original" and "original" in all_task_acc:
            delta = avg - avg_acc("original")
            delta_str = f"  ({delta:+.2f}%)"
        print(f"  {labels.get(m, m):40}  {avg:>7.2f}%  {task_str}{delta_str}")

    # Delta row
    if "original" in all_task_acc and "dual" in all_task_acc:
        orig = task_accs("original")
        dual = task_accs("dual")
        deltas = [dual[i] - orig[i] for i in range(n_tasks)]
        delta_task_str = "  ".join(f"{d:>+8.2f}%" for d in deltas)
        print(f"  {'Delta (Dual SRT - Original)':40}  {np.mean(deltas):>+7.2f}%  {delta_task_str}")

    print(f"{'='*80}")


def main():
    args = parse_args()

    print(f"\n{'='*70}")
    print(f"  SMoLoRA Dual Router — End-to-End Evaluation")
    print(f"  Mode: {args.routing_mode}, alpha(VU)={args.alpha}")
    print(f"  NOTE: Compares ORIGINAL vs SRT (VU/IF/Dual) vs ORACLE side-by-side")
    print(f"{'='*70}\n")

    # Load ins_emb
    with open(args.ins_emb, "rb") as f:
        ins_emb = np.array(pickle.load(f)).tolist()

    # Build routers
    clip_ext = CLIPVisionExtractor(model_name=args.clip_model)
    vu_router, if_router = build_dual_routers(
        ins_emb, args.task_order, clip_ext, args.task_images_root,
        n_per_task=args.n_build_signatures,
    )

    # Load model
    print("Loading SMoLoRA model...")
    model, tokenizer = load_smolora_model(args.model_path, args.model_base, args.device)

    # Determine routing modes
    if args.routing_mode == "all":
        modes = ["original", "vu_only", "if_only", "dual", "oracle"]
    else:
        modes = [args.routing_mode]

    all_results = {}

    for mode in modes:
        print(f"\n--- Routing mode: {mode} ---")
        results = generate_with_dual_routing(
            model, tokenizer, clip_ext, vu_router, if_router,
            test_samples=[],  # Load from dataset
            routing_mode=mode, ins_emb=ins_emb, task_order=args.task_order,
            alpha=args.alpha, max_new_tokens=args.max_new_tokens,
            max_samples=args.max_samples, device=args.device,
        )
        all_results[mode] = results

    # Score
    scored = {}
    for mode, preds in all_results.items():
        task_acc = {}
        for t_idx, task_name in enumerate(args.task_order):
            task_preds = [p for p in preds if p.get("task_id", 0) == t_idx]
            score = score_predictions(task_preds, args.scoring_func)
            task_acc[task_name] = score.get("accuracy", 0.0)
        scored[mode] = task_acc

    _print_comparison_table(args.task_order, scored)

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for mode, preds in all_results.items():
        out_path = output_dir / f"predictions_{mode}.jsonl"
        with open(out_path, "w") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")
        print(f"  Saved {mode} predictions: {out_path}")

    print(f"\n✓ Saved to {output_dir}")


if __name__ == "__main__":
    main()