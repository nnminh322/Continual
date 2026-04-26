#!/usr/bin/env python3
"""
Continual Learning Routing Evaluation — CORRECTED design.

Design philosophy:
  1. Router chỉ thấy embeddings của các task ĐÃ train (zero-rehearsal).
  2. Tại mỗi step t, router phải route test samples của task j (j ≤ t)
     mà không biết ground-truth task label.
  3. ZCA whitening: trong kịch bản continual thực sự, whitening transform
     có thể được fit từ ALL training embeddings (vì whitening transform
     không chứa raw data, chỉ chứa statistics). Đây là approach của reference.
  4. Buffer-based ZCA: thử nghiệm approach incremental — chờ buffer đủ lớn
     rồi fit ZCA một lần.
  5. No-whitening: baseline không có whitening.

Metrics: Routing Accuracy (macro across all seen tasks at each step).
Tương ứng với score matrix trong CL: accuracy của row i = macro_avg(scores[i][:i+1]).
"""
from __future__ import annotations
import argparse, json, os, sys, time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from numpy.linalg import eigh

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ─── Shared constants ────────────────────────────────────────────────────────

SUPERNI_ORDER = [
    "task1572_samsum_summary",
    "task363_sst2_polarity_classification",
    "task1290_xsum_summarization",
    "task181_outcome_extraction",
    "task002_quoref_answer_generation",
    "task1510_evalution_relation_extraction",
    "task639_multi_woz_user_utterance_generation",
    "task1729_personachat_generate_next",
    "task073_commonsenseqa_answer_generation",
    "task1590_diplomacy_text_generation",
    "task748_glucose_reverse_cause_event_detection",
    "task511_reddit_tifu_long_text_summarization",
    "task591_sciq_answer_generation",
    "task1687_sentiment140_classification",
    "task875_emotion_classification"
]

LONG_SEQ_ORDER = [
    "yelp","amazon","mnli","cb","copa","qqp","rte",
    "imdb","sst2","dbpedia","agnews","yahoo","multirc","boolq","wic",
]

BENCHMARK_ORDER = {
    "SuperNI": SUPERNI_ORDER,
    "Long_Sequence": LONG_SEQ_ORDER,
}


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_split(emb_dir, benchmark, task, split):
    p = Path(emb_dir) / benchmark / task / f"{split}.npz"
    if not p.exists():
        return None
    data = np.load(str(p), allow_pickle=True)
    return data["embeddings"].astype(np.float64)


def load_all(emb_dir, benchmark, tasks, split, max_per_task=None):
    out = OrderedDict()
    for t in tasks:
        embs = load_split(emb_dir, benchmark, t, split)
        if embs is not None:
            if max_per_task is not None and embs.shape[0] > max_per_task:
                embs = embs[:max_per_task]
            out[t] = embs
    return out


# ─── ZCA Whitening (core math) ───────────────────────────────────────────────

def fit_zca(embs_list, shrink_factor=0.1):
    """
    Fit ZCA whitening from a list of embedding arrays (pooled).

    Returns: (mu_global, W_zca)
    """
    all_embs = np.vstack(embs_list)
    n, d = all_embs.shape
    mu = all_embs.mean(axis=0)
    cov = np.cov(all_embs, rowvar=False, ddof=1)

    if shrink_factor > 0:
        trace = np.trace(cov)
        target = (trace / d) * np.eye(d)
        cov = (1 - shrink_factor) * cov + shrink_factor * target

    eigvals, eigvecs = eigh(cov)
    eigvals = np.maximum(eigvals, 1e-8)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return mu, W


def apply_whitening(h, mu, W):
    if h.ndim == 1:
        h = h.reshape(1, -1)
    return (h - mu) @ W.T


# ─── Router implementations ───────────────────────────────────────────────────

class NearestCentroidRouter:
    """Raw L2 distance to task centroid."""
    def __init__(self):
        self.centroids = []

    def add_task(self, embs):
        self.centroids.append(embs.mean(axis=0))

    def route(self, h_batch):
        C = np.stack(self.centroids)
        H_sq = np.sum(h_batch ** 2, axis=1, keepdims=True)
        C_sq = np.sum(C ** 2, axis=1).T
        dists = H_sq + C_sq - 2 * (h_batch @ C.T)
        return dists.argmin(axis=1)


class CosineNearestCentroidRouter:
    """Cosine similarity to task centroid."""
    def __init__(self):
        self.centroids = []

    def add_task(self, embs):
        mu = embs.mean(0)
        self.centroids.append(mu / (np.linalg.norm(mu) + 1e-12))

    def route(self, h_batch):
        C = np.stack(self.centroids)
        h_norm = h_batch / (np.linalg.norm(h_batch, axis=1, keepdims=True) + 1e-12)
        sims = h_norm @ C.T
        return sims.argmax(axis=1)


class PSRRouter:
    """Probabilistic Subspace Routing (k principal components)."""
    def __init__(self, k=8):
        self.k = k
        self.sigs = []

    def add_task(self, embs):
        n, d = embs.shape
        mu = embs.mean(axis=0)
        cov = np.cov(embs, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        k_eff = min(self.k, d)
        V = eigvecs[:, :k_eff]
        lam = np.maximum(eigvals[:k_eff], 1e-12)
        sigma2 = max(eigvals[k_eff:].mean() if k_eff < d else 1e-12, 1e-12)
        self.sigs.append((mu, V, lam, sigma2, d))

    def route(self, h_batch):
        T = len(self.sigs)
        if T == 0:
            return np.zeros(h_batch.shape[0], dtype=np.int64)
        d = self.sigs[0][4]
        k_eff = min(self.k, d)
        C = np.stack([s[0] for s in self.sigs]).astype(np.float32)
        V = np.stack([s[1] for s in self.sigs]).astype(np.float32)
        lam = np.stack([s[2] for s in self.sigs]).astype(np.float32)
        s2 = np.array([s[3] for s in self.sigs], dtype=np.float32)
        W_psr = lam / (s2[:, None] * (lam + s2[:, None]))
        pen = np.sum(np.log(lam + s2[:, None]), axis=1) + (d - k_eff) * np.log(s2)
        H = h_batch.astype(np.float32)
        H_sq = np.sum(H ** 2, axis=1, keepdims=True)
        C_sq = np.sum(C ** 2, axis=1)
        l2 = H_sq + C_sq[None, :] - 2 * (H @ C.T)
        iso = l2 / (s2[None, :] + 1e-12)
        H_proj = np.einsum('nd,tdk->ntk', H, V)
        CV = np.einsum('td,tdk->tk', C, V)
        dp = H_proj - CV[None, :, :]
        dists = iso + np.sum(W_psr[None, :, :] * dp**2, axis=-1) + pen[None, :]
        return np.argmin(dists, axis=1).astype(np.int64)


class GlobalZCAWhitenedRouter:
    """
    Fit ZCA once from ALL accumulated train embeddings (n/d large enough).

    This matches the reference --whiten experiment: fit ZCA on all seen
    training data (n/d >= 0.5), then use fixed whitening for all future routing.

    Approach: At each step, re-fit ZCA from all seen train embeddings.
    For final evaluation, ZCA is fit from all 15 tasks (n/d=0.59).
    """
    def __init__(self, shrink_factor=0.1):
        self.shrink_factor = shrink_factor
        self.train_embs_seen = []  # list of (task_name, emb_array)
        self.mu_global = None
        self.W_zca = None

    def _refit_zca(self):
        all_embs = [e for _, e in self.train_embs_seen]
        if len(all_embs) == 0:
            return
        self.mu_global, self.W_zca = fit_zca(all_embs, self.shrink_factor)

    def add_task(self, embs, task_name=None):
        self.train_embs_seen.append((task_name, embs.copy()))
        self._refit_zca()

        # Compute whitened centroids for all seen tasks
        self.centroids_whitened = []
        for _, emb in self.train_embs_seen:
            mu = emb.mean(axis=0)
            mu_w = apply_whitening(mu, self.mu_global, self.W_zca)
            self.centroids_whitened.append(mu_w)

    def route(self, h_batch):
        if self.W_zca is None:
            return np.zeros(h_batch.shape[0], dtype=np.int64)
        H_w = apply_whitening(h_batch, self.mu_global, self.W_zca)
        C_w = np.stack(self.centroids_whitened)
        H_sq = np.sum(H_w ** 2, axis=1, keepdims=True)
        C_sq = np.sum(C_w ** 2, axis=1).T
        dists = H_sq + C_sq - 2 * (H_w @ C_w.T)
        return dists.argmin(axis=1)


class IncrementalShrinkageWhitenedRouter:
    """
    Refit ZCA incrementally after each task (n/d grows from 0.04 upward).
    Matches ShrinkageWhitenedRouter in the original routing_class.py.

    This is the naive approach: ZCA is refit from ALL seen embeddings
    after each task. At task 1, n/d=0.04 → uninformative whitening.
    """
    def __init__(self, shrink_factor=0.1):
        self.shrink_factor = shrink_factor
        self.raw_centroids = []
        self.seen_embs = []
        self.mu_g = None
        self.W_zca = None

    def add_task(self, embs):
        self.raw_centroids.append(embs.mean(axis=0))
        self.seen_embs.append(embs.copy())
        all_embs = np.vstack(self.seen_embs)

        self.mu_g = all_embs.mean(axis=0)
        cov = np.cov(all_embs, rowvar=False, ddof=1)

        d = cov.shape[0]
        target = (np.trace(cov) / d) * np.eye(d)
        cov = (1 - self.shrink_factor) * cov + self.shrink_factor * target

        eigvals, eigvecs = eigh(cov)
        eigvals = np.maximum(eigvals, 1e-8)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        self.W_zca = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        self.signatures = [(mu_r - self.mu_g) @ self.W_zca.T for mu_r in self.raw_centroids]

    def route(self, h_batch):
        if not hasattr(self, 'signatures') or not self.signatures:
            return np.zeros(h_batch.shape[0], dtype=np.int64)
        H = np.array(h_batch, dtype=np.float64)
        H_w = (H - self.mu_g) @ self.W_zca.T
        C_w = np.stack(self.signatures)
        H_sq = np.sum(H_w ** 2, axis=1, keepdims=True)
        C_sq = np.sum(C_w ** 2, axis=1).T
        dists = H_sq + C_sq - 2 * (H_w @ C_w.T)
        return dists.argmin(axis=1)


class BufferZCAWhitenedRouter:
    """
    Buffer-based ZCA: accumulate embeddings until buffer >= buffer_size,
    then fit ZCA once. All centroids re-whitened with that fixed ZCA.

    This matches srt_router.py fix: fit ZCA once when enough data is available
    (zca_buffer_size), then keep it fixed.
    """
    def __init__(self, shrink_factor=0.1, zca_buffer_size=800):
        self.shrink_factor = shrink_factor
        self.zca_buffer_size = zca_buffer_size
        self.train_embs_seen = []
        self._buffer_embs = []
        self.mu_global = None
        self.W_zca = None
        self._zca_fitted = False
        self.centroids_whitened = []

    def _fit_zca_from_buffer(self):
        if len(self._buffer_embs) == 0:
            return
        self.mu_global, self.W_zca = fit_zca(self._buffer_embs, self.shrink_factor)
        self._zca_fitted = True

        # Re-whiten all existing centroids
        self.centroids_whitened = []
        for _, emb in self.train_embs_seen:
            mu = emb.mean(axis=0)
            mu_w = apply_whitening(mu, self.mu_global, self.W_zca)
            self.centroids_whitened.append(mu_w)

    def add_task(self, embs, task_name=None):
        self.train_embs_seen.append((task_name, embs.copy()))
        self._buffer_embs.append(embs.copy())

        if not self._zca_fitted and len(self._buffer_embs) > 0:
            total_n = sum(e.shape[0] for e in self._buffer_embs)
            if total_n >= self.zca_buffer_size:
                self._fit_zca_from_buffer()

        # Create whitened centroid for new task
        if self._zca_fitted:
            mu = embs.mean(axis=0)
            mu_w = apply_whitening(mu, self.mu_global, self.W_zca)
            self.centroids_whitened.append(mu_w)
        else:
            # ZCA not yet fitted — store raw centroid (will be whitened later if buffer fills)
            # Use raw centroid for now (first few tasks)
            mu_raw = embs.mean(axis=0)
            self.centroids_whitened.append(mu_raw)  # raw, not whitened

    def route(self, h_batch):
        if not self.centroids_whitened:
            return np.zeros(h_batch.shape[0], dtype=np.int64)

        if self._zca_fitted:
            # Use whitened centroids
            C_w = np.stack(self.centroids_whitened)
            H_w = apply_whitening(h_batch, self.mu_global, self.W_zca)
            H_sq = np.sum(H_w ** 2, axis=1, keepdims=True)
            C_sq = np.sum(C_w ** 2, axis=1).T
            dists = H_sq + C_sq - 2 * (H_w @ C_w.T)
            return dists.argmin(axis=1)
        else:
            # ZCA not fitted yet — fall back to raw L2
            C = np.stack(self.centroids_whitened)
            H_sq = np.sum(h_batch ** 2, axis=1, keepdims=True)
            C_sq = np.sum(C ** 2, axis=1).T
            dists = H_sq + C_sq - 2 * (h_batch @ C.T)
            return dists.argmin(axis=1)


class RLSRouter:
    """Recursive Least Squares router (Woodbury matrix identity)."""
    def __init__(self, d_model, expansion_dim=2048, lam=0.1, seed=42):
        self.d_model = d_model
        self.E = expansion_dim
        self.lam = lam
        rng = np.random.RandomState(seed)
        self.W_phi = (rng.randn(d_model, expansion_dim) / np.sqrt(d_model)).astype(np.float64)
        self.b_phi = (rng.randn(expansion_dim) * 0.01).astype(np.float64)
        self.R = np.eye(expansion_dim, dtype=np.float64) / lam
        self.Q = np.zeros((expansion_dim, 0), dtype=np.float64)
        self.W_r = np.zeros((expansion_dim, 0), dtype=np.float64)
        self.num_tasks = 0

    def _expand(self, h):
        return np.maximum(0, h @ self.W_phi + self.b_phi)

    def add_task(self, embs):
        H = self._expand(embs.astype(np.float64))
        N = H.shape[0]
        chunk = 512
        R = self.R.copy()
        for start in range(0, N, chunk):
            Hc = H[start:min(start + chunk, N)]
            RH = R @ Hc.T
            S = np.eye(Hc.shape[0]) + Hc @ RH
            try:
                S_inv_HcR = np.linalg.solve(S, Hc @ R)
                R = R - RH @ S_inv_HcR
            except np.linalg.LinAlgError:
                pass
        R = (R + R.T) * 0.5
        R += 1e-6 * np.eye(self.E)
        self.R = R
        tid = self.num_tasks
        extra = np.zeros((self.E, 1), dtype=np.float64)
        extra[:, 0] = H.T @ np.ones(N)
        self.Q = np.hstack([self.Q, extra])
        self.W_r = self.R @ self.Q
        self.num_tasks += 1

    def route(self, h_batch):
        H = self._expand(h_batch.astype(np.float64))
        logits = H @ self.W_r
        return logits.argmax(axis=1)


# ─── Incremental evaluation ─────────────────────────────────────────────────

def run_incremental_eval(train_embs_dict, test_embs_dict, task_order, args):
    """
    Core evaluation loop: simulates continual learning.

    At each step t (0-indexed):
      1. Add task t to all routers (router sees only train embeddings of task t)
      2. Evaluate routing accuracy on all test samples of tasks 0..t
         (router must correctly assign each test sample to its true task)

    Metrics: macro accuracy = avg(per_task_accuracy) over all seen tasks
    Corresponds to: mean(scores[t][:t+1]) for score matrix row t.
    """
    # Filter to tasks that exist
    ordered_found = [t for t in task_order if t in train_embs_dict and t in test_embs_dict]
    n_tasks = len(ordered_found)
    print(f"Tasks with train+test: {n_tasks}/{len(task_order)}")

    if n_tasks == 0:
        print("ERROR: No tasks found.")
        return {}

    # Build router instances
    routers = OrderedDict()
    routers["NearestCentroid"] = NearestCentroidRouter()
    routers["CosineNearestCentroid"] = CosineNearestCentroidRouter()
    routers["PSR"] = PSRRouter(k=args.subspace_k)
    routers["GlobalZCA_Shrink0.1"] = GlobalZCAWhitenedRouter(shrink_factor=0.1)
    routers["GlobalZCA_Shrink0.5"] = GlobalZCAWhitenedRouter(shrink_factor=0.5)
    routers["GlobalZCA_Shrink0.9"] = GlobalZCAWhitenedRouter(shrink_factor=0.9)
    routers["IncrementalZCA_Shrink0.1"] = IncrementalShrinkageWhitenedRouter(shrink_factor=0.1)
    routers["IncrementalZCA_Shrink0.5"] = IncrementalShrinkageWhitenedRouter(shrink_factor=0.5)
    routers["IncrementalZCA_Shrink0.9"] = IncrementalShrinkageWhitenedRouter(shrink_factor=0.9)
    routers["BufferZCA_800"] = BufferZCAWhitenedRouter(zca_buffer_size=800)
    routers["BufferZCA_1600"] = BufferZCAWhitenedRouter(zca_buffer_size=1600)
    routers["BufferZCA_2400"] = BufferZCAWhitenedRouter(zca_buffer_size=2400)
    routers["RLS_Woodbury"] = RLSRouter(
        d_model=next(iter(train_embs_dict.values())).shape[1],
        expansion_dim=args.rls_expansion, lam=args.rls_lambda)

    # Results: per router → list of step results
    all_results = {name: [] for name in routers}

    for t_idx, task_name in enumerate(ordered_found):
        print(f"\n  [{t_idx+1}/{n_tasks}] Task: {task_name}")

        # ── Add task t to all routers ──────────────────────────────
        embs_train = train_embs_dict[task_name]
        for name, router in routers.items():
            if "RLS" in name:
                router.add_task(embs_train)
            elif "GlobalZCA" in name or "BufferZCA" in name:
                router.add_task(embs_train, task_name)
            else:
                router.add_task(embs_train)

        # ── Evaluate on all seen tasks ─────────────────────────────
        seen_tasks = ordered_found[:t_idx + 1]

        for router_name, router in routers.items():
            per_task_acc = []
            for j, seen_task in enumerate(seen_tasks):
                embs_test = test_embs_dict[seen_task]
                preds = router.route(embs_test)
                true_idx = j  # j-th seen task → index j
                correct = int((preds == true_idx).sum())
                total = embs_test.shape[0]
                acc = correct / max(total, 1)
                per_task_acc.append(acc)

            macro_acc = sum(per_task_acc) / len(per_task_acc) if per_task_acc else 0.0
            row_str = " | ".join([f"{a*100:5.1f}%" for a in per_task_acc])

            all_results[router_name].append({
                "step": t_idx + 1,
                "n_seen_tasks": len(seen_tasks),
                "accuracy": macro_acc,
                "per_task": {t: a for t, a in zip(seen_tasks, per_task_acc)},
            })
            print(f"    {router_name:30s} macro_acc={macro_acc*100:6.2f}%  Row: [{row_str}]")

    return all_results


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Continual Learning Routing Evaluation (corrected)")
    parser.add_argument("--emb_dir", required=True, help="Path to extracted embeddings")
    parser.add_argument("--benchmark", required=True, choices=["SuperNI", "Long_Sequence"])
    parser.add_argument("--out_dir", default="results_cl")
    parser.add_argument("--subspace_k", type=int, default=8)
    parser.add_argument("--max_train_per_task", type=int, default=None,
                        help="Cap training samples per task (default: use all)")
    parser.add_argument("--rls_expansion", type=int, default=2048)
    parser.add_argument("--rls_lambda", type=float, default=0.1)
    parser.add_argument("--task_order", type=str, default=None,
                        help="Comma-separated task names (overrides default)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    backbone = Path(args.emb_dir).name

    if args.task_order:
        task_order = [t.strip() for t in args.task_order.split(",") if t.strip()]
    else:
        task_order = BENCHMARK_ORDER.get(args.benchmark, [])

    tag = f"{backbone}_{args.benchmark}"
    out_path = out_dir / f"cl_routing_{tag}.json"

    if out_path.exists() and not args.force:
        print(f"[SKIP] {out_path} already exists. Use --force to re-run.")
        return

    print(f"=== CL Routing Evaluation [{tag}] ===")
    print(f"    Benchmark: {args.benchmark}")
    print(f"    Backbone:  {backbone}")
    print(f"    Tasks:     {len(task_order)}")
    print(f"    Max/train: {args.max_train_per_task or 'all'}")
    print(f"    PSR k:     {args.subspace_k}")
    print()

    # Load data
    train_embs = load_all(args.emb_dir, args.benchmark, task_order, "train", args.max_train_per_task)
    test_embs = load_all(args.emb_dir, args.benchmark, task_order, "test", None)

    # Run evaluation
    results = run_incremental_eval(train_embs, test_embs, task_order, args)

    # Print summary
    print(f"\n{'='*80}")
    print(f"  Final Routing Accuracy (all {len(task_order)} tasks)")
    print(f"{'='*80}")
    print(f"  {'Method':30s}  {'Final Acc':>10s}  {'Step-by-step Acc':>40s}")
    print(f"  {'-'*82}")

    final_report = {}
    for name, steps in results.items():
        if not steps:
            continue
        final = steps[-1]
        final_accs = final["accuracy"]

        # Also compute average across all steps (progression)
        avg_acc = sum(s["accuracy"] for s in steps) / len(steps)

        # Per-step accuracy string
        step_str = "  ".join([f"T{i+1}:{s['accuracy']*100:.0f}%" for i, s in enumerate(steps)])

        print(f"  {name:30s}  {final_accs*100:8.2f}%  ({avg_acc*100:.2f}% avg)  {step_str}")
        final_report[name] = {
            "final_accuracy": final_accs,
            "avg_accuracy": avg_acc,
            "step_accuracies": [s["accuracy"] for s in steps],
        }

    # Save results
    report = {
        "backbone": backbone,
        "benchmark": args.benchmark,
        "n_tasks": len(task_order),
        "max_train_per_task": args.max_train_per_task,
        "subspace_k": args.subspace_k,
        "results": final_report,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n✓ Saved {out_path}")


if __name__ == "__main__":
    main()