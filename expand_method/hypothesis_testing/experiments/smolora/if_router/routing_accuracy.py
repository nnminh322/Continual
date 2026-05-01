#!/usr/bin/env python3
"""
Option A: SMoLoRA IF Router — Routing Accuracy Evaluation.
GPU-accelerated via torch CUDA for distance computation.

Usage:
    python experiments/smolora/if_router/routing_accuracy.py \\
        --ins_emb path/to/ins_emb_single.pkl \\
        --task_names ScienceQA TextVQA GQA VQAv2

Workflow:
    1. Load instruction embeddings from SMoLoRA's ins_emb pickle
    2. SRT router: Mahalanobis distance on GPU (Ridge + LW shrinkage)
    3. Baseline: Cosine similarity (original SMoLoRA IF-style)
    4. Incremental CL evaluation — prints ORIGINAL vs SRT side-by-side
"""
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from srt_router.pooled_mahalanobis import PooledMahalanobisRouter
from srt_router.metrics import compute_routing_accuracy, compute_per_task_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="SMoLoRA IF Router — Routing Accuracy (GPU)")
    parser.add_argument("--ins_emb", type=str, required=True,
                       help="Path to ins_emb_single.pkl")
    parser.add_argument("--task_names", type=str, nargs="+", default=None)
    parser.add_argument("--output_dir", type=str, default="results_smolora_if")
    parser.add_argument("--shrinkage", type=str, default="ridge",
                       choices=["ridge", "lw"])
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--noise_std", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="'cuda' = GPU-accelerated, 'cpu' = NumPy fallback")
    return parser.parse_args()


def generate_perturbed_samples(ins_emb: np.ndarray, n_samples: int,
                                noise_std: float, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    d = ins_emb.shape[-1]
    noise = rng.randn(n_samples, d) * noise_std
    return ins_emb.reshape(1, -1) + noise


def main():
    args = parse_args()

    # Check GPU
    try:
        import torch
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9 if torch.cuda.is_available() else 0
        gpu_tag = f"[GPU] {gpu_name}  {gpu_mem:.1f}GB"
    except Exception:
        gpu_tag = "[CPU]"

    print(f"\n{gpu_tag}  === CL Routing [SMoLoRA IF Router] ===")
    print(f"    Backbone: SMoLoRA (IF Router)")
    print(f"    Shrinkage: {args.shrinkage}")
    print(f"    Device: {args.device}")
    print(f"    NOTE: Gaussian noise (std={args.noise_std}) for synthetic test")

    # Load ins_emb
    with open(args.ins_emb, "rb") as f:
        ins_emb_raw = pickle.load(f)
    ins_emb = np.array(ins_emb_raw)
    if ins_emb.ndim == 1:
        ins_emb = ins_emb.reshape(1, -1)
    n_tasks, d = ins_emb.shape
    print(f"  Loaded ins_emb: ({n_tasks}, {d})")

    task_names = args.task_names if args.task_names else [f"task_{i}" for i in range(n_tasks)]
    if len(task_names) != n_tasks:
        task_names = [f"task_{i}" for i in range(n_tasks)]

    print(f"\n{'='*75}")
    print(f"  SMoLoRA IF Router — Routing Accuracy")
    print(f"  Tasks: {n_tasks}")
    print(f"  Shrinkage: {args.shrinkage}")
    print(f"  Device: {args.device}")
    print(f"{'='*75}\n")

    # Initialize SRT router (GPU-accelerated)
    srt_router = PooledMahalanobisRouter(shrinkage=args.shrinkage, device=args.device)
    print(f"  Router initialized on {srt_router.device}{' [GPU]' if srt_router.device == 'cuda' else ''}")

    all_results = []

    for step_idx, task_name in enumerate(task_names):
        # Add task
        emb = ins_emb[step_idx].reshape(1, -1)
        srt_router.add_task(emb, task_name=task_name)

        # Generate test samples for all seen tasks
        all_preds_srt = []
        all_preds_cos = []
        all_gt = []

        for t_idx in range(step_idx + 1):
            test_embs = generate_perturbed_samples(
                ins_emb[t_idx], n_samples=args.n_samples,
                noise_std=args.noise_std, seed=42 + t_idx,
            )
            gts = np.full(args.n_samples, t_idx, dtype=np.int64)
            all_gt.append(gts)

            # SRT Mahalanobis (GPU)
            preds_srt = srt_router.route(test_embs)
            all_preds_srt.append(preds_srt)

            # Cosine (ORIGINAL baseline)
            centroids = np.stack([ins_emb[i].reshape(1, -1) for i in range(step_idx + 1)])
            c_n = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
            t_n = test_embs / (np.linalg.norm(test_embs, axis=1, keepdims=True) + 1e-12)
            preds_cos = (t_n @ c_n.T).argmax(axis=1).astype(np.int64)
            all_preds_cos.append(preds_cos)

        all_preds_srt = np.concatenate(all_preds_srt)
        all_preds_cos = np.concatenate(all_preds_cos)
        all_gt = np.concatenate(all_gt)

        acc_srt = compute_routing_accuracy(all_preds_srt, all_gt)
        acc_cos = compute_routing_accuracy(all_preds_cos, all_gt)
        diag = srt_router.get_diagnostics()

        pt_srt = compute_per_task_accuracy(all_preds_srt, all_gt, task_names[:step_idx + 1])
        pt_cos = compute_per_task_accuracy(all_preds_cos, all_gt, task_names[:step_idx + 1])

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

        # Print RESULT lines
        def fmt(pt):
            return "  ".join(f"{v*100:5.1f}%" for v in pt.values())

        print(
            f"  [{step_idx+1}/{n_tasks}] {task_name}"
            f"  (n={pool_n}, n/d={n_d_ratio:.4f}, pool={pool_n})"
        )
        print(f"    Cosine (ORIGINAL)")
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
    print(f"  SMoLoRA IF — Final Routing Accuracy ({n} tasks, {args.device})")
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
    out_path = output_dir / f"smolora_if_routing_{args.shrinkage}_{args.device}.json"

    report = {
        "method": "SMoLoRA IF Router",
        "ins_emb_path": str(args.ins_emb),
        "n_tasks": n_tasks,
        "task_names": task_names,
        "shrinkage": args.shrinkage,
        "device": args.device,
        "n_samples_per_task": args.n_samples,
        "noise_std": args.noise_std,
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
