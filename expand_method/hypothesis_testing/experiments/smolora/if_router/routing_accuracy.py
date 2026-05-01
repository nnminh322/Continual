#!/usr/bin/env python3
"""
Option A: SMoLoRA IF Router — Routing Accuracy Evaluation.
GPU-accelerated. Runs standalone — generates synthetic ins_emb if no file provided.

Usage (no external files needed):
    python experiments/smolora/if_router/routing_accuracy.py \
        --task_names ScienceQA TextVQA GQA VQAv2

Usage (with real ins_emb):
    python experiments/smolora/if_router/routing_accuracy.py \
        --ins_emb path/to/ins_emb_single.pkl \
        --task_names ScienceQA TextVQA GQA VQAv2
"""
from __future__ import annotations
import argparse
import json
import pickle
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from srt_router.pooled_mahalanobis import PooledMahalanobisRouter
from srt_router.metrics import compute_routing_accuracy, compute_per_task_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="SMoLoRA IF Router — Routing Accuracy (GPU)")
    parser.add_argument("--ins_emb", type=str, default=None,
                       help="Path to ins_emb_single.pkl. If omitted, uses synthetic embeddings.")
    parser.add_argument("--task_names", type=str, nargs="+",
                       default=["ScienceQA", "TextVQA", "GQA", "VQAv2", "Flickr30k", "ImageNet", "Place365"])
    parser.add_argument("--output_dir", type=str, default="results_smolora_if")
    parser.add_argument("--shrinkage", type=str, default="ridge", choices=["ridge", "lw"])
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--noise_std", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--emb_dim", type=int, default=384,
                       help="Embedding dimension for synthetic ins_emb (default: 384)")
    parser.add_argument("--emb_sep", type=float, default=2.0,
                       help="Separation between task centroids for synthetic data (default: 2.0)")
    return parser.parse_args()


def _generate_synthetic_ins_emb(task_names, seed=42, emb_dim=384, sep=2.0):
    """
    Generate synthetic instruction embeddings for testing.
    Each task centroid is sep units away from others in a random direction.
    Returns list of (emb_dim,) arrays — one per task.
    """
    rng = np.random.RandomState(seed)
    n = len(task_names)
    embs = []
    for i in range(n):
        direction = rng.randn(emb_dim)
        direction /= (np.linalg.norm(direction) + 1e-12)
        centroid = direction * sep * i + rng.randn(emb_dim) * 0.1
        embs.append(centroid.astype(np.float32))
    return embs


def _load_ins_emb(path, task_names, emb_dim, sep):
    """
    Load ins_emb from pickle. If file missing or --ins_emb not given,
    generate synthetic embeddings.
    """
    if path is not None and Path(path).exists():
        print(f"  Loading ins_emb from: {path}")
        with open(path, "rb") as f:
            raw = pickle.load(f)
        ins_emb = np.array(raw)
        if ins_emb.ndim == 1:
            ins_emb = ins_emb.reshape(1, -1)
        return ins_emb
    else:
        print(f"  [Synthetic mode] Generating {len(task_names)} embeddings (dim={emb_dim}, sep={sep})")
        return _generate_synthetic_ins_emb(task_names, emb_dim=emb_dim, sep=sep)


def main():
    args = parse_args()
    task_names = args.task_names
    n_tasks = len(task_names)

    # GPU header
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            gpu_tag = f"[GPU] {gpu_name}  {gpu_mem:.1f}GB"
        else:
            gpu_tag = "[CPU]"
    except Exception:
        gpu_tag = "[CPU]"

    print(f"\n{gpu_tag}  === CL Routing [SMoLoRA IF Router] ===")
    print(f"    Tasks: {n_tasks}")
    print(f"    Shrinkage: {args.shrinkage}")
    print(f"    Device: {args.device}")
    print(f"    Noise std: {args.noise_std}")

    # Load or generate ins_emb
    ins_emb = _load_ins_emb(args.ins_emb, task_names, args.emb_dim, args.emb_sep)
    d = ins_emb.shape[-1]
    print(f"  ins_emb shape: ({len(ins_emb)}, {d})")

    print(f"\n{'='*75}")
    print(f"  SMoLoRA IF Router — Routing Accuracy")
    print(f"  Tasks: {n_tasks}  Shrinkage: {args.shrinkage}  Device: {args.device}")
    print(f"  Synthetic mode: {args.ins_emb is None or not Path(args.ins_emb).exists()}")
    print(f"{'='*75}\n")

    # SRT router (GPU-accelerated)
    srt_router = PooledMahalanobisRouter(shrinkage=args.shrinkage, device=args.device)
    print(f"  Router: {srt_router.device}  {'[GPU]' if srt_router.device == 'cuda' else '[CPU]'}")

    all_results = []

    for step_idx, task_name in enumerate(task_names):
        srt_router.add_task(ins_emb[step_idx].reshape(1, -1), task_name=task_name)

        all_preds_srt = []
        all_preds_cos = []
        all_gt = []

        for t_idx in range(step_idx + 1):
            # Synthetic test: Gaussian noise around each centroid
            rng = np.random.RandomState(42 + t_idx)
            test_embs = ins_emb[t_idx].reshape(1, -1) + rng.randn(args.n_samples, d) * args.noise_std
            gts = np.full(args.n_samples, t_idx, dtype=np.int64)
            all_gt.append(gts)

            preds_srt = srt_router.route(test_embs)
            all_preds_srt.append(preds_srt)

            # Cosine (ORIGINAL)
            centroids = np.stack([ins_emb[i] for i in range(step_idx + 1)])
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
        n_d = pool_n / d if d > 0 else 0
        all_results.append({
            "task": task_name, "n": pool_n, "n_d": n_d,
            "acc_srt": acc_srt, "acc_cos": acc_cos,
            "pt_srt": pt_srt, "pt_cos": pt_cos,
        })

        def fmt(pt):
            return "  ".join(f"{v*100:5.1f}%" for v in pt.values())

        print(f"  [{step_idx+1}/{n_tasks}] {task_name}  (n={pool_n}, n/d={n_d:.4f})")
        print("    Cosine (ORIGINAL)")
        print(f"    SRT-Mahal_{args.shrinkage.upper()} (NEW)")
        print(f"    RESULT Cosine (ORIGINAL)              macro={acc_cos*100:6.2f}%  [{fmt(pt_cos)}]")
        print(f"    RESULT SRT-Mahal_{args.shrinkage.upper()} (NEW)        macro={acc_srt*100:6.2f}%  [{fmt(pt_srt)}]")

    # Final summary
    n = len(all_results)
    print(f"\n{'='*75}")
    print(f"  SMoLoRA IF — Final Routing Accuracy ({n} tasks, {args.device})")
    print(f"{'='*75}")
    hdr = "  " + "  ".join(f"{'T'+str(i+1):>7}" for i in range(n))
    print(f"  {'Method':30}  {'Final':>7}  {'Avg':>7}{hdr}")
    print(f"  {'-'*75}")

    def ts(key): return "  ".join(f"{r[key]*100:6.1f}%" for r in all_results)
    def av(key): return f"{np.mean([r[key] for r in all_results])*100:6.1f}%"
    def fn(key): return f"{all_results[-1][key]*100:6.1f}%"

    print(f"  {'Cosine (ORIGINAL)':30}  {fn('acc_cos')}  {av('acc_cos')}  {ts('acc_cos')}")
    print(f"  {'SRT-Mahal_{args.shrinkage.upper()} (NEW)':30}  {fn('acc_srt')}  {av('acc_srt')}  {ts('acc_srt')}")

    deltas = [all_results[i]["acc_srt"] - all_results[i]["acc_cos"] for i in range(n)]
    ds = "  ".join(f"{d*100:+5.1f}%" for d in deltas)
    print(f"  {'Delta (SRT - Original)':30}  {deltas[-1]*100:+6.1f}%  {np.mean(deltas)*100:+6.1f}%  {ds}")
    print(f"{'='*75}")

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"smolora_if_{args.shrinkage}_{args.device}.json"

    _ins_emb_has = args.ins_emb is not None and (Path(str(args.ins_emb)).exists() if args.ins_emb else False)
    ins_emb_mode = "real" if _ins_emb_has else "synthetic"

    report = {
        "method": "SMoLoRA IF Router",
        "ins_emb_mode": ins_emb_mode,
        "n_tasks": n_tasks, "task_names": task_names,
        "shrinkage": args.shrinkage, "device": args.device,
        "noise_std": args.noise_std, "emb_dim": args.emb_dim, "emb_sep": args.emb_sep,
        "results": {f"step_{i+1}": {
            "task": r["task"], "accuracy_srt": r["acc_srt"],
            "accuracy_cosine": r["acc_cos"],
            "per_task_srt": r["pt_srt"], "per_task_cosine": r["pt_cos"],
        } for i, r in enumerate(all_results)},
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()