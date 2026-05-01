#!/usr/bin/env python3
"""
Option B: HiDe-LLaVA Cosine → SRT — End-to-End Evaluation.

Loads a trained HiDe-LLaVA checkpoint and evaluates with SRT
Mahalanobis routing replacing the cosine similarity routing.

Usage:
    python experiments/hide/cosine_router/end_to_end.py \
        --model_path /path/to/hide/checkpoint \
        --model_base /path/to/vicuna-7b \
        --clip_model openai/clip-vit-large-patch14-336 \
        --task_images_root /path/to/task/images \
        --task_order ScienceQA TextVQA GQA VQAv2 VizWiz TextCaps \
        --routing_mode all

Comparison output:
    Shows ORIGINAL (Cosine HiDe) vs NEW (SRT) vs ORACLE side-by-side.
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import shortuuid
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from srt_router.pooled_mahalanobis import PooledMahalanobisRouter
from embedding_extractors.clip_extractor import CLIPVisionExtractor


def parse_args():
    parser = argparse.ArgumentParser(description="HiDe-LLaVA Cosine → SRT End-to-End Evaluation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_base", type=str, required=True)
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--task_images_root", type=str, required=True)
    parser.add_argument("--task_order", type=str, nargs="+", required=True)
    parser.add_argument("--routing_mode", type=str, default="srt",
                       choices=["srt", "cosine", "oracle", "all"])
    parser.add_argument("--scoring_func", type=str, default="vqav2",
                       choices=["vqav2", "science_qa", "gqa"])
    parser.add_argument("--output_dir", type=str, default="results_hide_e2e")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_build_signatures", type=int, default=200)
    return parser.parse_args()


def collect_images(data_root: Path, task_name: str, max_n: int) -> List[Path]:
    task_dir = data_root / task_name
    if not task_dir.exists():
        return []
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPEG", "*.JPG", "*.PNG"]
    images = []
    for ext in exts:
        images.extend(task_dir.rglob(ext))
    return images[:max_n]


def build_srt_router(clip_ext, task_images_root, task_order, n_per_task, shrinkage="ridge"):
    router = PooledMahalanobisRouter(shrinkage=shrinkage)
    for task_name in tqdm(task_order, desc="  Building SRT signatures"):
        paths = collect_images(Path(task_images_root), task_name, max_n=n_per_task)
        if not paths:
            continue
        embs = clip_ext.extract_from_paths(paths, batch_size=8)
        router.add_task(embs, task_name=task_name)
        print(f"  {task_name}: {len(paths)} images")
    return router


def load_hide_model(model_path, model_base, device):
    """Load HiDe-LLaVA model."""
    import torch
    from llava.model.builder import load_pretrained_model
    from HiDe.peft.tuners.clitmoelora import HiDeMOELoraLinear

    model, tokenizer, _, _ = load_pretrained_model(model_path, model_base, None, None)

    for name, module in model.named_modules():
        if isinstance(module, HiDeMOELoraLinear):
            if not hasattr(module, "expert_weight"):
                module.expert_weight = [0.0]

    return model, tokenizer


def set_expert_weight(model, expert_weight: List[float]):
    """Set expert_weight on the last transformer layer's projection modules."""
    from HiDe.peft.tuners.clitmoelora import HiDeMOELoraLinear

    last_layer = model.get_model().layers[-1]
    for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
        proj = getattr(last_layer.self_attn, proj_name, None) or getattr(last_layer.mlp, proj_name, None)
        if proj is not None and hasattr(proj, "expert_weight"):
            proj.expert_weight = expert_weight


def cosine_routing(test_emb, anchors, temperature=0.1):
    """HiDe-style cosine routing."""
    C = np.stack(anchors)
    C_norm = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    test_norm = test_emb / (np.linalg.norm(test_emb, axis=1, keepdims=True) + 1e-12)
    sims = test_norm @ C_norm.T
    weights = np.exp(sims / temperature)
    weights = weights / weights.sum(axis=1, keepdims=True)
    return weights[0].tolist()


def generate_predictions(
    model, tokenizer, clip_ext, router,
    test_samples, routing_mode, task_order,
    max_new_tokens=128, temperature=0.0, max_samples=None, device="cuda",
) -> List[dict]:
    """Generate with specified routing mode: 'srt', 'cosine', or 'oracle'."""
    import torch
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import tokenizer_image_token
    from llava.conversation import conv_templates
    from PIL import Image

    if max_samples is not None:
        test_samples = test_samples[:max_samples]

    results = []
    cosine_anchors = []

    for sample in tqdm(test_samples, desc=f"  [{routing_mode}]"):
        task_id = sample.get("task_id", 0)
        qs = sample["text"]

        image = None
        clip_emb = None
        if "image" in sample and sample["image"]:
            try:
                image = Image.open(sample["image"]).convert("RGB")
                clip_emb = clip_ext.extract_single(image).reshape(1, -1)
            except Exception:
                pass

        # Determine expert_weight based on routing mode
        if routing_mode == "srt":
            if clip_emb is not None and router.n_tasks > 0:
                pred_idx = int(router.route(clip_emb)[0])
            else:
                pred_idx = task_id
        elif routing_mode == "cosine":
            if clip_emb is not None and len(cosine_anchors) > 0:
                weights = cosine_routing(clip_emb, cosine_anchors)
                pred_idx = int(np.argmax(weights))
            else:
                pred_idx = task_id
            if clip_emb is not None:
                cosine_anchors.append(clip_emb[0])
        elif routing_mode == "oracle":
            pred_idx = task_id
        else:
            pred_idx = task_id

        expert_weight = [0.0] * max(router.n_tasks, 1)
        if pred_idx < len(expert_weight):
            expert_weight[pred_idx] = 1.0
        set_expert_weight(model, expert_weight)

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
            "pred_idx": pred_idx,
            "expert_weight": expert_weight,
            "routing_mode": routing_mode,
        })

    return results


def score_predictions(results: List[dict], scoring_func: str) -> dict:
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

    print(f"\n{'='*75}")
    print("  HiDe-LLaVA Cosine → SRT — End-to-End Task Accuracy Comparison")
    print(f"  ({n_tasks} tasks)")
    print(f"{'='*75}")

    task_headers = "  ".join(f"{t[:8]:>8}" for t in task_order)
    print(f"  {'Method':42}  {'Avg':>8}  {task_headers}")
    print(f"  {'-'*75}")

    def task_accs(method):
        return [all_task_acc.get(method, {}).get(t, 0.0) for t in task_order]

    def avg_acc(method):
        accs = task_accs(method)
        return np.mean(accs) if accs else 0.0

    def fmt_row(method, label):
        accs = task_accs(method)
        task_str = "  ".join(f"{a:>8.2f}%" for a in accs)
        avg = avg_acc(method)
        delta_str = ""
        if method != "cosine" and "cosine" in all_task_acc:
            delta = avg - avg_acc("cosine")
            delta_str = f"  ({delta:+.2f}%)"
        print(f"  {label:42}  {avg:>7.2f}%  {task_str}{delta_str}")

    if "cosine" in all_task_acc:
        fmt_row("cosine", "HiDe Cosine (ORIGINAL)")
    if "srt" in all_task_acc:
        fmt_row("srt", "HiDe Cosine → SRT (NEW)")
    if "oracle" in all_task_acc:
        fmt_row("oracle", "Oracle (Upper Bound)")

    # Delta row
    if "cosine" in all_task_acc and "srt" in all_task_acc:
        orig = task_accs("cosine")
        srt = task_accs("srt")
        deltas = [srt[i] - orig[i] for i in range(n_tasks)]
        delta_task_str = "  ".join(f"{d:>+8.2f}%" for d in deltas)
        print(f"  {'Delta (SRT - Original)':42}  {np.mean(deltas):>+7.2f}%  {delta_task_str}")

    print(f"{'='*75}")


def main():
    args = parse_args()
    print(f"\n{'='*70}")
    print(f"  HiDe-LLaVA Cosine → SRT End-to-End")
    print(f"  Mode: {args.routing_mode}")
    print(f"  NOTE: Compares ORIGINAL (Cosine) vs NEW (SRT) vs ORACLE side-by-side")
    print(f"{'='*70}\n")

    # Build SRT router
    print("Building SRT router...")
    clip_ext = CLIPVisionExtractor(model_name=args.clip_model)
    srt_router = build_srt_router(
        clip_ext, args.task_images_root, args.task_order,
        n_per_task=args.n_build_signatures,
    )
    print(f"  SRT router: {srt_router.n_tasks} tasks")

    # Load model
    print("Loading HiDe-LLaVA model...")
    model, tokenizer = load_hide_model(args.model_path, args.model_base, args.device)

    # Determine routing modes
    if args.routing_mode == "all":
        modes = ["cosine", "srt", "oracle"]
    else:
        modes = [args.routing_mode]

    all_results = {}

    for mode in modes:
        print(f"\n--- Routing mode: {mode} ---")
        results = generate_predictions(
            model=model, tokenizer=tokenizer, clip_ext=clip_ext,
            router=srt_router, test_samples=[], routing_mode=mode,
            task_order=args.task_order,
            max_new_tokens=args.max_new_tokens,
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

    print(f"\n✓ Done. Results in {output_dir}")


if __name__ == "__main__":
    main()