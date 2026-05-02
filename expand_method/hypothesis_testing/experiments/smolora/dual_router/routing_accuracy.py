#!/usr/bin/env python3
"""
Option A: SMoLoRA Dual Router (VU + IF) — Routing Accuracy Evaluation.
GPU-accelerated: CLIP on CUDA, Mahalanobis routing on GPU.

Usage:
    python experiments/smolora/dual_router/routing_accuracy.py \\
        --ins_emb path/to/ins_emb_single.pkl \\
        --data_root /path/to/images \\
        --task_names ScienceQA TextVQA GQA VQAv2 \\
        --clip_model openai/clip-vit-large-patch14-336 \\
        --device cuda
"""
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path
from typing import List

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


# ── Default instruction strings (from SMoLoRA ins_gen.py) ──────────────────────
DEFAULT_INSTRUCTIONS = [
    "Answer with the option's letter from the given choices directly.",   # 0: ScienceQA
    "Answer the question using a single word or phrase.",                # 1: TextVQA
    "What is happening in the presented picture?\nPlease describe it in one complete sentence.",  # 2: Flickr30k
    "What is the object in the image?\nAnswer the question using a single word or phrase.",       # 3: ImageNet
    "Answer the question using a single word or phrase.",                # 4: GQA
    "Answer the question using a single word or phrase.",              # 5: VQAv2
    "What is the background of the image?\nAnswer the question using a single word or phrase.",  # 6: Place365
]


def _generate_ins_emb_from_strings(instructions: list, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Generate ins_emb from instruction strings (same as SMoLoRA ins_gen.py)."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        return None
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    encoded = tokenizer(instructions, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoded)
    token_emb = output[0]
    mask_expanded = encoded["attention_mask"].unsqueeze(-1).expand(token_emb.size()).float()
    embeddings = torch.sum(token_emb * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)
    return embeddings.cpu().numpy().astype(np.float32)


def _load_ins_emb(path: str | None, n_tasks: int) -> np.ndarray:
    """Load or generate ins_emb. Falls back to synthetic if unavailable."""
    if path is not None and Path(path).exists():
        with open(path, "rb") as f:
            raw = pickle.load(f)
        ins_emb = np.array(raw)
        if ins_emb.ndim == 1:
            ins_emb = ins_emb.reshape(1, -1)
        return ins_emb
    generated = _generate_ins_emb_from_strings(DEFAULT_INSTRUCTIONS)
    if generated is not None:
        print(f"  [Generated] ins_emb from {len(DEFAULT_INSTRUCTIONS)} instruction strings")
        return generated
    print(f"  [Synthetic] ins_emb: {n_tasks} × 384 Gaussian centroids")
    rng = np.random.RandomState(42)
    d = 384
    embs = []
    for i in range(n_tasks):
        direction = rng.randn(d)
        direction /= (np.linalg.norm(direction) + 1e-12)
        centroid = direction * 2.0 * i + rng.randn(d) * 0.1
        embs.append(centroid.astype(np.float32))
    return np.array(embs)


def parse_args():
    parser = argparse.ArgumentParser(description="SMoLoRA Dual Router — Routing Accuracy (GPU)")
    parser.add_argument("--ins_emb", type=str, default="ins_emb_single.pkl",
                       help="Path to ins_emb_single.pkl (auto-generated if missing)")
    parser.add_argument("--data_root", type=str,
                       default="/data/zqwang/moe_cl_data/dataset",
                       help="Root with task image subdirs. Default: SMoLoRA server. "
                            "Set to 'synthetic' to skip VU routing.")
    parser.add_argument("--task_names", type=str, nargs="+",
                       default=["ScienceQA", "TextVQA", "GQA", "VQAv2", "Flickr30k", "ImageNet", "Place365"])
    parser.add_argument("--clip_model", type=str,
                       default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--output_dir", type=str, default="results_smolora_dual")
    parser.add_argument("--shrinkage", type=str, default="ridge",
                       choices=["ridge", "lw"])
    parser.add_argument("--alpha", type=float, default=0.5,
                       help="Weight for VU: d = alpha*d_vu + (1-alpha)*d_if")
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

    try:
        import torch
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        gpu_tag = f"[GPU] {gpu_name}"
    except Exception:
        gpu_tag = "[CPU]"

    print(f"\n{gpu_tag}  === CL Routing [SMoLoRA Dual Router] ===")
    print(f"    CLIP: {args.clip_model}")
    print(f"    Tasks: {len(args.task_names)}")
    print(f"    Shrinkage: {args.shrinkage}, alpha(VU)={args.alpha}")
    print(f"    Device: {args.device}")

    # Load ins_emb
    ins_emb = _load_ins_emb(args.ins_emb, len(args.task_names))
    print(f"  ins_emb: {ins_emb.shape}")

    # Load CLIP + images
    clip_ext = CLIPVisionExtractor(model_name=args.clip_model, device=args.device, dtype="float16")
    print(f"  CLIP: {clip_ext}")

    # Extract image embeddings
    task_img_train = {}
    task_img_test = {}

    use_vu = (args.data_root != "synthetic")

    if use_vu:
        for task_name in tqdm(args.task_names, desc="  Extracting images"):
            paths = collect_images(Path(args.data_root), task_name, args.n_train + args.n_test)
            if len(paths) < 2:
                print(f"  [WARN] {task_name}: only {len(paths)} images — skipping VU for this task")
                continue
            rng = np.random.RandomState(hash(task_name) % (2**32))
            idx = rng.permutation(len(paths))
            split = max(2, int(len(idx) * 0.8))
            train_paths = [paths[i] for i in idx[:split]]
            test_paths = [paths[i] for i in idx[split: split + args.n_test]]
            task_img_train[task_name] = clip_ext.extract_from_paths(train_paths, batch_size=args.batch_size)
            task_img_test[task_name] = clip_ext.extract_from_paths(test_paths, batch_size=args.batch_size)
            print(f"  {task_name}: {len(train_paths)} train, {len(test_paths)} test  dim={clip_ext.embedding_dim}")
    else:
        print("  [Synthetic mode] VU routing skipped — data_root=synthetic")

    print(f"\n{'='*75}")
    print(f"  SMoLoRA Dual Router — Routing Accuracy")
    print(f"  Cosine VU (ORIGINAL) vs SRT-VU (NEW) vs SRT-Dual (NEW)")
    print(f"  Device: {args.device}")
    print(f"{'='*75}\n")

    # SRT routers (GPU)
    vu_router = PooledMahalanobisRouter(shrinkage=args.shrinkage, device=args.device)
    if_router = PooledMahalanobisRouter(shrinkage=args.shrinkage, device=args.device)
    print(f"  Routers initialized on {args.device}")

    all_results = []

    for step_idx, task_name in enumerate(args.task_names):
        # Add task
        if task_name in task_img_train:
            vu_router.add_task(task_img_train[task_name], task_name=task_name)
        if_router.add_task(ins_emb[step_idx].reshape(1, -1), task_name=task_name)

        all_gt = []
        preds_vu_srt = []
        preds_vu_cos = []
        preds_dual = []

        for t_idx, seen_task in enumerate(args.task_names[:step_idx + 1]):
            if seen_task not in task_img_test:
                continue
            test_embs = task_img_test[seen_task]
            gts = np.full(len(test_embs), t_idx, dtype=np.int64)
            all_gt.append(gts)

            # SRT VU (GPU)
            preds_vu_srt.append(vu_router.route(test_embs))

            # Cosine VU (ORIGINAL baseline)
            vu_cents = [task_img_train[args.task_names[i]].mean(axis=0)
                        for i in range(step_idx + 1)]
            preds_vu_cos.append(baseline_cosine_routing(np.stack(vu_cents), test_embs))

            # SRT Dual: alpha*VU + (1-alpha)*IF
            d_vu = vu_router._compute_distances_cpu(test_embs) if args.device == "cpu" \
                else None  # GPU path below
            if_query = np.tile(ins_emb[t_idx].reshape(1, -1), (len(test_embs), 1))
            d_if = if_router._compute_distances_cpu(if_query) if args.device == "cpu" \
                else None

            if args.device == "cuda":
                import torch
                h_t = torch.from_numpy(test_embs.astype(np.float32)).cuda()
                if_emb_t = torch.from_numpy(if_query.astype(np.float32)).cuda()
                C_t = torch.from_numpy(np.stack(vu_cents)).cuda()
                Sinv_vu = vu_router._Sinv_t
                Sinv_if = if_router._Sinv_t
                with torch.no_grad():
                    d_vu_t = (h_t @ Sinv_vu * h_t).sum(1, keepdim=True) \
                             - 2 * (h_t @ Sinv_vu @ C_t.T) \
                             + (C_t @ Sinv_vu * C_t).sum(1)
                    d_if_t = (if_emb_t @ Sinv_if * if_emb_t).sum(1, keepdim=True) \
                             - 2 * (if_emb_t @ Sinv_if @ torch.from_numpy(if_query).cuda().T) \
                             + (torch.from_numpy(if_query).cuda() @ Sinv_if * torch.from_numpy(if_query).cuda()).sum(1)
                d_dual_t = args.alpha * d_vu_t + (1 - args.alpha) * d_if_t.T
                preds_dual.append(d_dual_t.argmin(1).cpu().numpy().astype(np.int64))
            else:
                d_dual = args.alpha * d_vu + (1 - args.alpha) * d_if
                preds_dual.append(d_dual.argmin(axis=1).astype(np.int64))

        if not all_gt:
            print(f"\n  [{step_idx+1}/{len(args.task_names)}] {task_name}"
                  f"  — no test images, skipping")
            continue

        all_gt = np.concatenate(all_gt)
        preds_vu_srt = np.concatenate(preds_vu_srt)
        preds_vu_cos = np.concatenate(preds_vu_cos)
        preds_dual = np.concatenate(preds_dual)

        acc_vu_srt = compute_routing_accuracy(preds_vu_srt, all_gt)
        acc_vu_cos = compute_routing_accuracy(preds_vu_cos, all_gt)
        acc_dual = compute_routing_accuracy(preds_dual, all_gt)

        seen_names = args.task_names[:step_idx + 1]
        pt_vu_srt = compute_per_task_accuracy(preds_vu_srt, all_gt, seen_names)
        pt_vu_cos = compute_per_task_accuracy(preds_vu_cos, all_gt, seen_names)
        pt_dual = compute_per_task_accuracy(preds_dual, all_gt, seen_names)

        diag = vu_router.get_diagnostics()
        pool_n = diag["n_pool"]

        all_results.append({
            "task": task_name,
            "acc_vu_srt": acc_vu_srt,
            "acc_vu_cos": acc_vu_cos,
            "acc_dual": acc_dual,
            "pt_vu_srt": pt_vu_srt,
            "pt_vu_cos": pt_vu_cos,
            "pt_dual": pt_dual,
        })

        def fmt(pt):
            return "  ".join(f"{v*100:5.1f}%" for v in pt.values())

        print(
            f"\n  [{step_idx+1}/{len(args.task_names)}] {task_name}"
            f"  (pool={pool_n})"
        )
        print("    Cosine VU (ORIGINAL)")
        print("    SRT-Mahal_VU (NEW)")
        print(f"    SRT-Mahal_Dual (NEW, alpha={args.alpha})")
        print(
            f"    RESULT Cosine VU (ORIGINAL)            macro={acc_vu_cos*100:6.2f}%"
            f"  [{fmt(pt_vu_cos)}]"
        )
        print(
            f"    RESULT SRT-Mahal_VU (NEW)             macro={acc_vu_srt*100:6.2f}%"
            f"  [{fmt(pt_vu_srt)}]"
        )
        print(
            f"    RESULT SRT-Mahal_Dual (NEW)           macro={acc_dual*100:6.2f}%"
            f"  [{fmt(pt_dual)}]"
        )

    # Final summary
    n = len(all_results)
    print(f"\n{'='*75}")
    print(f"  SMoLoRA Dual — Final Routing Accuracy ({n} tasks, {args.device})")
    print(f"{'='*75}")
    if n == 0:
        print("  No images found — check --data_root and --task_names.")
        print(f"{'='*75}")
        return
    hdr = "  " + "  ".join(f"{'T'+str(i+1):>7}" for i in range(n))
    print(f"  {'Method':30}  {'Final':>7}  {'Avg':>7}{hdr}")
    print(f"  {'-'*75}")

    def ts(key):
        return "  ".join(f"{r[key]*100:6.1f}%" for r in all_results)
    def av(key):
        return f"{np.mean([r[key] for r in all_results])*100:6.1f}%"
    def fn(key):
        return f"{all_results[-1][key]*100:6.1f}%"

    print(f"  {'Cosine VU (ORIGINAL)':30}  {fn('acc_vu_cos')}  {av('acc_vu_cos')}  {ts('acc_vu_cos')}")
    print(f"  {'SRT-Mahal_VU (NEW)':30}  {fn('acc_vu_srt')}  {av('acc_vu_srt')}  {ts('acc_vu_srt')}")
    print(f"  {'SRT-Mahal_Dual (NEW)':30}  {fn('acc_dual')}  {av('acc_dual')}  {ts('acc_dual')}")

    deltas = [all_results[i]["acc_vu_srt"] - all_results[i]["acc_vu_cos"] for i in range(n)]
    ds = "  ".join(f"{d*100:+5.1f}%" for d in deltas)
    print(f"  {'Delta (VU SRT - Cosine)':30}  {deltas[-1]*100:+6.1f}%  {np.mean(deltas)*100:+6.1f}%  {ds}")
    print(f"{'='*75}")

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"smolora_dual_{args.shrinkage}_{args.device}.json"
    report = {
        "method": "SMoLoRA Dual Router (VU+IF)",
        "ins_emb": str(args.ins_emb),
        "clip_model": args.clip_model,
        "task_names": args.task_names,
        "shrinkage": args.shrinkage,
        "alpha": args.alpha,
        "device": args.device,
        "results": {
            f"step_{i+1}": {
                "task": r["task"],
                "accuracy_vu_srt": r["acc_vu_srt"],
                "accuracy_vu_cosine": r["acc_vu_cos"],
                "accuracy_dual": r["acc_dual"],
            }
            for i, r in enumerate(all_results)
        },
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n✓ Saved to {out_path}")


if __name__ == "__main__":
    main()
