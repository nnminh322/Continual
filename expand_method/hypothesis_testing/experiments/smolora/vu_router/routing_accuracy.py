#!/usr/bin/env python3
"""
Option A: SMoLoRA VU Router — Routing Accuracy Evaluation.
GPU-accelerated: CLIP extraction on CUDA, Mahalanobis routing on GPU.

Usage:
    python experiments/smolora/vu_router/routing_accuracy.py \\
        --data_root /path/to/training/images \\
        --task_names ScienceQA TextVQA GQA VQAv2 \\
        --clip_model openai/clip-vit-large-patch14-336 \\
        --device cuda
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from srt_router.pooled_mahalanobis import PooledMahalanobisRouter
from srt_router.metrics import (
    compute_routing_accuracy,
    baseline_cosine_routing,
    compute_per_task_accuracy,
)
from embedding_extractors.clip_extractor import CLIPVisionExtractor


def parse_args():
    parser = argparse.ArgumentParser(description="SMoLoRA VU Router — Routing Accuracy (GPU)")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--task_names", type=str, nargs="+", required=True)
    parser.add_argument("--clip_model", type=str,
                       default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--output_dir", type=str, default="results_smolora_vu")
    parser.add_argument("--shrinkage", type=str, default="ridge",
                       choices=["ridge", "lw"])
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"])
    parser.add_argument("--n_train", type=int, default=500)
    parser.add_argument("--n_test", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


def collect_images(data_root: Path, task_name: str, max_n: int) -> List[Path]:
    task_dir = data_root / task_name
    if not task_dir.exists():
        return []
    images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPEG", "*.JPG", "*.PNG"]:
        images.extend(task_dir.rglob(ext))
    return images[:max_n]


def main():
    args = parse_args()
    data_root = Path(args.data_root)

    # GPU header
    try:
        import torch
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        gpu_tag = f"[GPU] {gpu_name}"
    except Exception:
        gpu_tag = "[CPU]"

    print(f"\n{gpu_tag}  === CL Routing [SMoLoRA VU Router] ===")
    print(f"    CLIP: {args.clip_model}")
    print(f"    Tasks: {len(args.task_names)}")
    print(f"    Shrinkage: {args.shrinkage}")
    print(f"    Device: {args.device}")

    # Load CLIP (GPU)
    extractor = CLIPVisionExtractor(
        model_name=args.clip_model,
        device=args.device,
        dtype="float16",
    )
    d = extractor.embedding_dim
    print(f"\n  CLIP: {extractor}")

    # Pre-extract embeddings (GPU → torch tensors, then numpy for router)
    print("  Extracting CLIP embeddings...")
    emb_train: Dict[str, np.ndarray] = {}
    emb_test: Dict[str, np.ndarray] = {}

    for task_name in tqdm(args.task_names, desc="  Extracting"):
        paths = collect_images(data_root, task_name, args.n_train + args.n_test)
        if len(paths) < 2:
            print(f"  [WARN] {task_name}: only {len(paths)} images")
            continue
        rng = np.random.RandomState(hash(task_name) % (2**32))
        idx = rng.permutation(len(paths))
        split = max(2, int(len(idx) * 0.8))
        train_paths = [paths[i] for i in idx[:split]]
        test_paths = [paths[i] for i in idx[split: split + args.n_test]]

        emb_train[task_name] = extractor.extract_from_paths(train_paths, batch_size=args.batch_size)
        emb_test[task_name] = extractor.extract_from_paths(test_paths, batch_size=args.batch_size)
        print(f"  {task_name}: {len(train_paths)} train, {len(test_paths)} test  →  dim={d}")

    print(f"\n{'='*75}")
    print(f"  SMoLoRA VU Router — Routing Accuracy")
    print(f"  SRT-Mahal_{args.shrinkage.upper()} (NEW) vs Cosine (ORIGINAL)")
    print(f"  Device: {args.device}")
    print(f"{'='*75}\n")

    # SRT router on GPU
    srt_router = PooledMahalanobisRouter(shrinkage=args.shrinkage, device=args.device)
    print(f"  SRT Router initialized on {srt_router.device}")

    all_results = []
    centroids = []

    for step_idx, task_name in enumerate(args.task_names):
        if task_name not in emb_train:
            continue

        train_embs = emb_train[task_name]
        srt_router.add_task(train_embs, task_name=task_name)
        centroids.append(train_embs.mean(axis=0))

        all_preds_srt = []
        all_preds_cos = []
        all_gt = []

        for t_idx, seen_task in enumerate(args.task_names[:step_idx + 1]):
            if seen_task not in emb_test:
                continue
            test_embs = emb_test[seen_task]
            gts = np.full(len(test_embs), t_idx, dtype=np.int64)
            all_gt.append(gts)

            preds_srt = srt_router.route(test_embs)
            all_preds_srt.append(preds_srt)

            c_stacked = np.stack(centroids)
            preds_cos = baseline_cosine_routing(c_stacked, test_embs)
            all_preds_cos.append(preds_cos)

        all_preds_srt = np.concatenate(all_preds_srt)
        all_preds_cos = np.concatenate(all_preds_cos)
        all_gt = np.concatenate(all_gt)

        acc_srt = compute_routing_accuracy(all_preds_srt, all_gt)
        acc_cos = compute_routing_accuracy(all_preds_cos, all_gt)

        seen_names = args.task_names[:step_idx + 1]
        pt_srt = compute_per_task_accuracy(all_preds_srt, all_gt, seen_names)
        pt_cos = compute_per_task_accuracy(all_preds_cos, all_gt, seen_names)

        diag = srt_router.get_diagnostics()
        pool_n = diag["n_pool"]
        n_d_ratio = pool_n / d if d > 0 else 0

        all_results.append({
            "task": task_name,
            "n": pool_n,
            "n_d": n_d_ratio,
            "acc_srt": acc_srt,
            "acc_cos": acc_cos,
            "pt_srt": pt_srt,
            "pt_cos": pt_cos,
        })

        def fmt(pt):
            return "  ".join(f"{v*100:5.1f}%" for v in pt.values())

        print(
            f"  [{step_idx+1}/{len(args.task_names)}] {task_name}"
            f"  (n={pool_n}, n/d={n_d_ratio:.4f}, pool={pool_n})"
        )
        print("    Cosine (ORIGINAL)")
        print(f"    SRT-Mahal_{args.shrinkage.upper()} (NEW)")
        print(
            f"    RESULT Cosine (ORIGINAL)              macro={acc_cos*100:6.2f}%"
            f"  [{fmt(pt_cos)}]"
        )
        print(
            f"    RESULT SRT-Mahal_{args.shrinkage.upper()} (NEW)        macro={acc_srt*100:6.2f}%"
            f"  [{fmt(pt_srt)}]"
        )

    # Final summary
    n = len(all_results)
    print(f"\n{'='*75}")
    print(f"  SMoLoRA VU — Final Routing Accuracy ({n} tasks, {args.device})")
    print(f"{'='*75}")
    hdr = "  " + "  ".join(f"{'T'+str(i+1):>7}" for i in range(n))
    print(f"  {'Method':30}  {'Final':>7}  {'Avg':>7}{hdr}")
    print(f"  {'-'*75}")

    def ts(key):
        return "  ".join(f"{r[key]*100:6.1f}%" for r in all_results)
    def av(key):
        return f"{np.mean([r[key] for r in all_results])*100:6.1f}%"
    def fn(key):
        return f"{all_results[-1][key]*100:6.1f}%"

    print(f"  {'Cosine (ORIGINAL)':30}  {fn('acc_cos')}  {av('acc_cos')}  {ts('acc_cos')}")
    print(f"  {'SRT-Mahal_{args.shrinkage.upper()} (NEW)':30}  {fn('acc_srt')}  {av('acc_srt')}  {ts('acc_srt')}")

    deltas = [all_results[i]["acc_srt"] - all_results[i]["acc_cos"] for i in range(n)]
    ds = "  ".join(f"{d*100:+5.1f}%" for d in deltas)
    print(f"  {'Delta (SRT - Original)':30}  {deltas[-1]*100:+6.1f}%  {np.mean(deltas)*100:+6.1f}%  {ds}")
    print(f"{'='*75}")

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"smolora_vu_routing_{args.shrinkage}_{args.device}.json"

    report = {
        "method": "SMoLoRA VU Router",
        "clip_model": args.clip_model,
        "data_root": str(data_root),
        "task_names": args.task_names,
        "shrinkage": args.shrinkage,
        "device": args.device,
        "results": {
            f"step_{i+1}": {
                "task": r["task"],
                "accuracy_srt": r["acc_srt"],
                "accuracy_cosine": r["acc_cos"],
                "per_task_srt": r["pt_srt"],
                "per_task_cosine": r["pt_cos"],
            }
            for i, r in enumerate(all_results)
        },
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()
