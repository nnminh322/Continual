#!/usr/bin/env python3
"""
Contribution 2 (GALA) — Empirical Validation of Core Hypotheses.

Uses the same embeddings as C1 (routing_analysis) to validate GALA's
theoretical claims about LoRA training geometry.

Phases:
  Phase 1 (TARA): Task geometric complexity varies → fixed rank is suboptimal
  Phase 2 (GGI):  Generalized EVP > PCA > Random for init subspace
  Phase 3 (SGR):  Soft Grassmannian penalty vs hard projection
  Phase 4 (BNG):  Activation preconditioning effect

Usage:
  python validate_gala.py --emb_dir embeddings/T5EncoderModel --benchmark Long_Sequence
  python validate_gala.py --emb_dir embeddings/LlamaForCausalLM --benchmark SuperNI
  python validate_gala.py --emb_dir embeddings/T5EncoderModel --benchmark Long_Sequence --phase 1
  python validate_gala.py --emb_dir embeddings/T5EncoderModel --benchmark Long_Sequence --phase 1 --whiten

Flags:
  --phase N       Run only phase N (1-4). Default: all.
  --whiten        Apply ZCA whitening before analysis.
  --device auto   GPU acceleration (auto/cuda/cpu).
  --rank_list     Ranks to sweep for TARA (default: 2,4,8,16,32,64).
  --n_trials      Number of random trials for init comparison (default: 50).
  --output_dir    Where to save results (default: results/).
"""
from __future__ import annotations
import argparse, json, os, sys, time, warnings
from pathlib import Path
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ═══════════════════════════════════════════════════════════════════════
# Device helpers (from C1 codebase)
# ═══════════════════════════════════════════════════════════════════════

_USE_GPU = False
_DEVICE = None

def _resolve_device(device_str: str) -> str:
    if device_str == "auto":
        if HAS_TORCH and torch.cuda.is_available():
            try:
                _t = torch.zeros(8, device="cuda") + 1
                del _t
                torch.cuda.synchronize()
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"[GPU] {gpu_name}  {vram_gb:.1f} GB VRAM — using CUDA")
                return "cuda"
            except Exception:
                pass
        if HAS_TORCH and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("[GPU] Apple MPS device — using MPS")
            return "mps"
        print("[Device] CPU mode")
        return "cpu"
    return device_str


def _to_torch(arr: np.ndarray) -> "torch.Tensor":
    return torch.as_tensor(np.ascontiguousarray(arr), dtype=torch.float32, device=_DEVICE)


def _cov(X: "torch.Tensor") -> "torch.Tensor":
    N = X.shape[0]
    Xc = X - X.mean(0)
    return (Xc.T @ Xc) / max(N - 1, 1)


def _eigh(C: "torch.Tensor"):
    eigvals, eigvecs = torch.linalg.eigh(C)
    idx = torch.argsort(eigvals, descending=True)
    return eigvals[idx], eigvecs[:, idx]


def _par(eigvals):
    """Participation ratio from eigenvalues (tensor or ndarray)."""
    if HAS_TORCH and isinstance(eigvals, torch.Tensor):
        eigvals = eigvals.clamp(min=0)
        s = eigvals.sum()
        if s < 1e-15:
            return 0.0
        return float((s ** 2 / (eigvals ** 2).sum()).item())
    else:
        if HAS_TORCH and isinstance(eigvals, torch.Tensor):
            eigvals = eigvals.cpu().numpy()
        eigvals = np.maximum(np.asarray(eigvals), 0)
        s = eigvals.sum()
        if s < 1e-15:
            return 0.0
        return float(s ** 2 / (eigvals ** 2).sum())


# ═══════════════════════════════════════════════════════════════════════
# Data loading (same format as C1)
# ═══════════════════════════════════════════════════════════════════════

BENCHMARKS = {
    "Long_Sequence": [
        "yelp", "amazon", "mnli", "cb", "copa", "qqp", "rte",
        "imdb", "sst2", "dbpedia", "agnews", "yahoo", "multirc", "boolq", "wic",
    ],
    "SuperNI": [
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
    ],
}

LONG_SEQ_CLUSTERS = {
    "sentiment": ["yelp", "amazon", "imdb", "sst2"],
    "NLI": ["mnli", "cb", "rte"],
    "topic": ["dbpedia", "agnews", "yahoo"],
    "RC": ["multirc", "boolq"],
    "misc": ["copa", "qqp", "wic"],
}


def load_split(emb_dir: str, benchmark: str, task: str, split: str = "train"):
    p = Path(emb_dir) / benchmark / task / f"{split}.npz"
    if not p.exists():
        return None, None
    data = np.load(str(p), allow_pickle=True)
    return data["embeddings"].astype(np.float64), data["labels"]


def load_all_tasks(emb_dir, benchmark, tasks, split="train"):
    out = OrderedDict()
    for t in tasks:
        embs, labels = load_split(emb_dir, benchmark, t, split)
        if embs is not None:
            out[t] = {"embeddings": embs, "labels": labels}
    return out


def compute_whitening(task_data: dict) -> Tuple[np.ndarray, np.ndarray]:
    all_embs = np.vstack([d["embeddings"] for d in task_data.values()])
    mu = all_embs.mean(0)
    cov = np.cov(all_embs, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-8)
    W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return mu, W


def apply_whitening(task_data: dict, mu: np.ndarray, W: np.ndarray) -> dict:
    out = {}
    for t, d in task_data.items():
        embs_w = (d["embeddings"] - mu) @ W.T
        out[t] = {"embeddings": embs_w, "labels": d["labels"]}
    return out


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: TARA — Task-Adaptive Rank Allocation Validation
# ═══════════════════════════════════════════════════════════════════════

def phase1_tara(task_data: dict, rank_list: List[int] = None) -> dict:
    """
    H1: Tasks have different geometric complexity → fixed rank is suboptimal.
    H6: Different PaR variants give different effective ranks.

    For each task, compute:
    1. Full eigenvalue spectrum of activation covariance
    2. PaR (activation-based)
    3. Simulated "gradient PaR" (task-residual: Σ_task - Σ_pool proxy)
    4. Rank sweep: what fraction of variance captured at each rank?
    5. Recommended rank from TARA vs oracle (90%/95%/99% variance)
    """
    if rank_list is None:
        rank_list = [2, 4, 8, 16, 32, 64]

    print("\n" + "=" * 70)
    print("PHASE 1: TARA — Task-Adaptive Rank Allocation")
    print("=" * 70)

    tasks = list(task_data.keys())
    d = task_data[tasks[0]]["embeddings"].shape[1]

    # Compute pooled covariance
    all_embs = np.vstack([td["embeddings"] for td in task_data.values()])
    if _USE_GPU:
        all_t = _to_torch(all_embs)
        cov_pool = _cov(all_t)
        eigvals_pool, _ = _eigh(cov_pool)
        del all_t
    else:
        cov_pool_np = np.cov(all_embs, rowvar=False)
        eigvals_pool_np = np.flip(np.sort(np.linalg.eigvalsh(cov_pool_np)))

    results = {"tasks": {}, "rank_list": rank_list, "d": d}

    for task in tasks:
        embs = task_data[task]["embeddings"]
        n = embs.shape[0]

        if _USE_GPU:
            X = _to_torch(embs)
            cov_task = _cov(X)
            eigvals_task, eigvecs_task = _eigh(cov_task)

            # Task-residual covariance (proxy for gradient covariance)
            # Intuition: directions where task differs from pool ≈ where LoRA needs to act
            cov_residual = cov_task - cov_pool
            eigvals_res, _ = _eigh(cov_residual)
            # Only positive eigenvalues matter (task has MORE variance than pool)
            eigvals_res_pos = eigvals_res.clamp(min=0)

            par_activation = _par(eigvals_task)
            par_residual = _par(eigvals_res_pos)

            # Rank sweep: fraction of task-residual variance captured
            total_res = eigvals_res_pos.sum().item()
            total_act = eigvals_task.sum().item()
            rank_sweep = {}
            for r in rank_list:
                if r > d:
                    continue
                frac_act = float((eigvals_task[:r].sum() / max(total_act, 1e-15)).item())
                frac_res = float((eigvals_res_pos[:r].sum() / max(total_res, 1e-15)).item()) if total_res > 1e-15 else 1.0
                rank_sweep[r] = {
                    "frac_activation_var": round(frac_act, 4),
                    "frac_residual_var": round(frac_res, 4),
                }

            # Recommended rank: smallest r capturing 90% of residual variance
            cumvar_res = torch.cumsum(eigvals_res_pos, 0)
            if total_res > 1e-15:
                cumfrac = cumvar_res / total_res
                r_90 = int((torch.searchsorted(cumfrac, 0.90)).item()) + 1
                r_95 = int((torch.searchsorted(cumfrac, 0.95)).item()) + 1
                r_99 = int((torch.searchsorted(cumfrac, 0.99)).item()) + 1
            else:
                r_90 = r_95 = r_99 = 2

            # Full spectrum stats
            spectrum = eigvals_task[:min(64, d)].cpu().tolist()
            spectrum_res = eigvals_res_pos[:min(64, d)].cpu().tolist()

            del X, cov_task, eigvals_task, eigvecs_task, cov_residual, eigvals_res
        else:
            cov_task_np = np.cov(embs, rowvar=False)
            eigvals_task_np = np.flip(np.sort(np.linalg.eigvalsh(cov_task_np)))

            cov_residual_np = cov_task_np - cov_pool_np
            eigvals_res_np = np.flip(np.sort(np.linalg.eigvalsh(cov_residual_np)))
            eigvals_res_pos_np = np.maximum(eigvals_res_np, 0)

            par_activation = _par(eigvals_task_np)
            par_residual = _par(eigvals_res_pos_np)

            total_res = eigvals_res_pos_np.sum()
            total_act = eigvals_task_np.sum()
            rank_sweep = {}
            for r in rank_list:
                if r > d:
                    continue
                frac_act = float(eigvals_task_np[:r].sum() / max(total_act, 1e-15))
                frac_res = float(eigvals_res_pos_np[:r].sum() / max(total_res, 1e-15)) if total_res > 1e-15 else 1.0
                rank_sweep[r] = {
                    "frac_activation_var": round(frac_act, 4),
                    "frac_residual_var": round(frac_res, 4),
                }

            if total_res > 1e-15:
                cumfrac = np.cumsum(eigvals_res_pos_np) / total_res
                r_90 = int(np.searchsorted(cumfrac, 0.90)) + 1
                r_95 = int(np.searchsorted(cumfrac, 0.95)) + 1
                r_99 = int(np.searchsorted(cumfrac, 0.99)) + 1
            else:
                r_90 = r_95 = r_99 = 2

            spectrum = eigvals_task_np[:min(64, d)].tolist()
            spectrum_res = eigvals_res_pos_np[:min(64, d)].tolist()

        task_result = {
            "n_samples": n,
            "par_activation": round(par_activation, 2),
            "par_residual": round(par_residual, 2),
            "tara_recommended_rank_90": r_90,
            "tara_recommended_rank_95": r_95,
            "tara_recommended_rank_99": r_99,
            "rank_sweep": rank_sweep,
            "spectrum_top64": spectrum,
            "spectrum_residual_top64": spectrum_res,
        }
        results["tasks"][task] = task_result

        print(f"  {task:40s}  PaR_act={par_activation:5.1f}  PaR_res={par_residual:5.1f}  "
              f"TARA_r90={r_90:3d}  TARA_r95={r_95:3d}  TARA_r99={r_99:3d}")

    # Summary statistics
    act_pars = [results["tasks"][t]["par_activation"] for t in tasks]
    res_pars = [results["tasks"][t]["par_residual"] for t in tasks]
    r90s = [results["tasks"][t]["tara_recommended_rank_90"] for t in tasks]

    results["summary"] = {
        "par_activation_range": [round(min(act_pars), 1), round(max(act_pars), 1)],
        "par_activation_mean": round(np.mean(act_pars), 1),
        "par_residual_range": [round(min(res_pars), 1), round(max(res_pars), 1)],
        "par_residual_mean": round(np.mean(res_pars), 1),
        "tara_r90_range": [min(r90s), max(r90s)],
        "hypothesis_H1_supported": max(act_pars) / max(min(act_pars), 0.1) > 2.0,
        "hypothesis_H6_supported": abs(np.mean(act_pars) - np.mean(res_pars)) > 2.0,
    }

    print(f"\n  [H1] Task complexity varies? PaR range = {results['summary']['par_activation_range']} "
          f"→ {'YES ✓' if results['summary']['hypothesis_H1_supported'] else 'NO ✗'}")
    print(f"  [H6] PaR_activation ≠ PaR_residual? means {np.mean(act_pars):.1f} vs {np.mean(res_pars):.1f} "
          f"→ {'YES ✓' if results['summary']['hypothesis_H6_supported'] else 'NO ✗'}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: GGI — Generalized EVP vs PCA vs Random Init
# ═══════════════════════════════════════════════════════════════════════

def phase2_ggi(task_data: dict, n_trials: int = 50, rank: int = 8) -> dict:
    """
    H2: Generalized EVP > PCA > Random for init subspace quality.

    Since we don't have actual gradients, we simulate:
    - "Gradient covariance" ≈ per-class centroid dispersion (between-class scatter)
    - "Activation covariance" = standard covariance

    Generalized EVP: maximize between-class / within-class ratio
    PCA: maximize total variance
    Random: baseline

    Metrics:
    - Fisher discriminant ratio (DA criterion)
    - Subspace "task-specificity" = between-class var / total var in subspace
    - Alignment with class-discriminative directions
    """
    print("\n" + "=" * 70)
    print("PHASE 2: GGI — Initialization Strategy Comparison")
    print("=" * 70)

    tasks = list(task_data.keys())
    d = task_data[tasks[0]]["embeddings"].shape[1]
    results = {"tasks": {}, "rank": rank, "n_trials": n_trials, "d": d}

    for task in tasks:
        embs = task_data[task]["embeddings"]
        labels = task_data[task]["labels"]
        n = embs.shape[0]

        if labels is None:
            print(f"  {task}: SKIP (no labels)")
            continue

        # Get unique classes
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        if n_classes < 2:
            print(f"  {task}: SKIP ({n_classes} classes)")
            continue

        if _USE_GPU:
            X = _to_torch(embs)
            mu_global = X.mean(0)

            # Within-class scatter (Σ_W) and between-class scatter (Σ_B)
            Sw = torch.zeros(d, d, device=_DEVICE)
            Sb = torch.zeros(d, d, device=_DEVICE)

            for label in unique_labels:
                mask = (labels == label) if isinstance(labels[0], (int, np.integer)) else (labels == label)
                Xc = X[mask]
                nc = Xc.shape[0]
                if nc < 2:
                    continue
                mu_c = Xc.mean(0)
                diff = Xc - mu_c
                Sw += diff.T @ diff
                mu_diff = (mu_c - mu_global).unsqueeze(1)
                Sb += nc * (mu_diff @ mu_diff.T)

            Sw /= n
            Sb /= n
            St = _cov(X)  # Total scatter

            # Regularize Sw for numerical stability
            reg = 1e-6 * torch.trace(Sw) / d
            Sw_reg = Sw + reg * torch.eye(d, device=_DEVICE)

            # === Strategy 1: GGI (Generalized EVP) ===
            # Solve Sb v = λ Sw v → maximize between/within ratio
            try:
                gev = torch.linalg.eigh(torch.linalg.solve(Sw_reg, Sb))
                eigvals_gev, eigvecs_gev = gev
                idx = torch.argsort(eigvals_gev, descending=True)
                V_ggi = eigvecs_gev[:, idx[:rank]]  # (d, r)
            except Exception:
                # Fallback: standard EVP on Sb
                eigvals_sb, eigvecs_sb = _eigh(Sb)
                V_ggi = eigvecs_sb[:, :rank]

            # === Strategy 2: PCA (standard top-r eigenvectors) ===
            eigvals_st, eigvecs_st = _eigh(St)
            V_pca = eigvecs_st[:, :rank]

            # === Strategy 3: Random (Kaiming-like) ===
            fisher_ratios_random = []
            specificity_random = []
            for _ in range(n_trials):
                V_rand = torch.randn(d, rank, device=_DEVICE)
                V_rand, _ = torch.linalg.qr(V_rand)

                # Project scatter matrices
                Sb_proj = V_rand.T @ Sb @ V_rand
                Sw_proj = V_rand.T @ Sw_reg @ V_rand
                St_proj = V_rand.T @ St @ V_rand

                fr = float((torch.trace(Sb_proj) / torch.trace(Sw_proj).clamp(min=1e-15)).item())
                sp = float((torch.trace(Sb_proj) / torch.trace(St_proj).clamp(min=1e-15)).item())
                fisher_ratios_random.append(fr)
                specificity_random.append(sp)

            # Evaluate GGI and PCA
            def evaluate_subspace(V, name):
                Sb_p = V.T @ Sb @ V
                Sw_p = V.T @ Sw_reg @ V
                St_p = V.T @ St @ V
                fr = float((torch.trace(Sb_p) / torch.trace(Sw_p).clamp(min=1e-15)).item())
                sp = float((torch.trace(Sb_p) / torch.trace(St_p).clamp(min=1e-15)).item())
                # Effective rank of projected scatter
                eigvals_proj, _ = _eigh(St_p)
                er = _par(eigvals_proj)
                return {"fisher_ratio": round(fr, 4), "specificity": round(sp, 4),
                        "effective_rank": round(er, 2)}

            ggi_metrics = evaluate_subspace(V_ggi, "GGI")
            pca_metrics = evaluate_subspace(V_pca, "PCA")

            # Overlap between strategies
            overlap_ggi_pca = float(torch.linalg.norm(V_ggi.T @ V_pca, "fro").item() ** 2 / rank)

            del X, Sw, Sb, St, Sw_reg
        else:
            # CPU path
            mu_global = embs.mean(0)
            Sw = np.zeros((d, d))
            Sb = np.zeros((d, d))

            for label in unique_labels:
                mask = labels == label
                Xc = embs[mask]
                nc = Xc.shape[0]
                if nc < 2:
                    continue
                mu_c = Xc.mean(0)
                diff = Xc - mu_c
                Sw += diff.T @ diff
                mu_diff = (mu_c - mu_global).reshape(-1, 1)
                Sb += nc * (mu_diff @ mu_diff.T)

            Sw /= n
            Sb /= n
            St = np.cov(embs, rowvar=False)
            reg = 1e-6 * np.trace(Sw) / d
            Sw_reg = Sw + reg * np.eye(d)

            try:
                Sw_inv_Sb = np.linalg.solve(Sw_reg, Sb)
                eigvals_gev, eigvecs_gev = np.linalg.eigh(Sw_inv_Sb)
                idx = np.argsort(eigvals_gev)[::-1]
                V_ggi = eigvecs_gev[:, idx[:rank]]
            except Exception:
                eigvals_sb, eigvecs_sb = np.linalg.eigh(Sb)
                idx = np.argsort(eigvals_sb)[::-1]
                V_ggi = eigvecs_sb[:, idx[:rank]]

            eigvals_st, eigvecs_st = np.linalg.eigh(St)
            idx = np.argsort(eigvals_st)[::-1]
            V_pca = eigvecs_st[:, idx[:rank]]

            fisher_ratios_random = []
            specificity_random = []
            rng = np.random.RandomState(42)
            for _ in range(n_trials):
                V_rand = rng.randn(d, rank)
                V_rand, _ = np.linalg.qr(V_rand)
                Sb_proj = V_rand.T @ Sb @ V_rand
                Sw_proj = V_rand.T @ Sw_reg @ V_rand
                St_proj = V_rand.T @ St @ V_rand
                fr = float(np.trace(Sb_proj) / max(np.trace(Sw_proj), 1e-15))
                sp = float(np.trace(Sb_proj) / max(np.trace(St_proj), 1e-15))
                fisher_ratios_random.append(fr)
                specificity_random.append(sp)

            def evaluate_subspace(V, name):
                Sb_p = V.T @ Sb @ V
                Sw_p = V.T @ Sw_reg @ V
                St_p = V.T @ St @ V
                fr = float(np.trace(Sb_p) / max(np.trace(Sw_p), 1e-15))
                sp = float(np.trace(Sb_p) / max(np.trace(St_p), 1e-15))
                eigvals_proj = np.flip(np.sort(np.linalg.eigvalsh(St_p)))
                er = _par(eigvals_proj)
                return {"fisher_ratio": round(fr, 4), "specificity": round(sp, 4),
                        "effective_rank": round(er, 2)}

            ggi_metrics = evaluate_subspace(V_ggi, "GGI")
            pca_metrics = evaluate_subspace(V_pca, "PCA")
            overlap_ggi_pca = float(np.linalg.norm(V_ggi.T @ V_pca, "fro") ** 2 / rank)

        random_metrics = {
            "fisher_ratio_mean": round(np.mean(fisher_ratios_random), 4),
            "fisher_ratio_std": round(np.std(fisher_ratios_random), 4),
            "specificity_mean": round(np.mean(specificity_random), 4),
            "specificity_std": round(np.std(specificity_random), 4),
        }

        task_result = {
            "n_samples": n,
            "n_classes": n_classes,
            "GGI": ggi_metrics,
            "PCA": pca_metrics,
            "Random": random_metrics,
            "overlap_GGI_PCA": round(overlap_ggi_pca, 4),
            "GGI_vs_PCA_fisher_ratio": round(ggi_metrics["fisher_ratio"] / max(pca_metrics["fisher_ratio"], 1e-15), 2),
            "GGI_vs_Random_fisher_ratio": round(ggi_metrics["fisher_ratio"] / max(random_metrics["fisher_ratio_mean"], 1e-15), 2),
        }
        results["tasks"][task] = task_result

        print(f"  {task:40s}  GGI_FR={ggi_metrics['fisher_ratio']:.3f}  "
              f"PCA_FR={pca_metrics['fisher_ratio']:.3f}  "
              f"Rand_FR={random_metrics['fisher_ratio_mean']:.3f}±{random_metrics['fisher_ratio_std']:.3f}  "
              f"GGI/PCA={task_result['GGI_vs_PCA_fisher_ratio']:.1f}x  "
              f"GGI/Rand={task_result['GGI_vs_Random_fisher_ratio']:.1f}x")

    # Summary
    valid_tasks = [t for t in tasks if t in results["tasks"]]
    if valid_tasks:
        ggi_frs = [results["tasks"][t]["GGI"]["fisher_ratio"] for t in valid_tasks]
        pca_frs = [results["tasks"][t]["PCA"]["fisher_ratio"] for t in valid_tasks]
        rand_frs = [results["tasks"][t]["Random"]["fisher_ratio_mean"] for t in valid_tasks]
        ratios_gp = [results["tasks"][t]["GGI_vs_PCA_fisher_ratio"] for t in valid_tasks]
        ratios_gr = [results["tasks"][t]["GGI_vs_Random_fisher_ratio"] for t in valid_tasks]

        results["summary"] = {
            "GGI_fisher_ratio_mean": round(np.mean(ggi_frs), 4),
            "PCA_fisher_ratio_mean": round(np.mean(pca_frs), 4),
            "Random_fisher_ratio_mean": round(np.mean(rand_frs), 4),
            "GGI_vs_PCA_ratio_mean": round(np.mean(ratios_gp), 2),
            "GGI_vs_Random_ratio_mean": round(np.mean(ratios_gr), 2),
            "hypothesis_H2_GGI_gt_PCA": np.mean(ggi_frs) > np.mean(pca_frs),
            "hypothesis_H2_GGI_gt_Random": np.mean(ggi_frs) > np.mean(rand_frs),
            "n_tasks_GGI_wins_PCA": sum(1 for r in ratios_gp if r > 1.0),
            "n_tasks_total": len(valid_tasks),
        }

        print(f"\n  [H2] GGI > PCA? mean FR {np.mean(ggi_frs):.4f} vs {np.mean(pca_frs):.4f} "
              f"→ {'YES ✓' if results['summary']['hypothesis_H2_GGI_gt_PCA'] else 'NO ✗'} "
              f"({results['summary']['n_tasks_GGI_wins_PCA']}/{len(valid_tasks)} tasks)")
        print(f"  [H2] GGI > Random? mean FR {np.mean(ggi_frs):.4f} vs {np.mean(rand_frs):.4f} "
              f"→ {'YES ✓' if results['summary']['hypothesis_H2_GGI_gt_Random'] else 'NO ✗'}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: SGR — Soft vs Hard Orthogonal Projection
# ═══════════════════════════════════════════════════════════════════════

def phase3_sgr(task_data: dict, rank: int = 8) -> dict:
    """
    H3: Hard projection is lossy; Soft Grassmannian penalty is better.
    H7: Same-domain tasks have high subspace overlap.

    Simulate CL sequence: tasks arrive one by one.
    For each new task:
    - Measure SSE (Shared Subspace Exclusion)
    - Compare: (a) Hard projection, (b) Soft relaxation (β sweep)
    - Measure: info retained, subspace overlap, effective rank after projection

    Uses chordal distance = ||V_t^T V_s||_F^2 as overlap metric.
    """
    print("\n" + "=" * 70)
    print("PHASE 3: SGR — Hard vs Soft Projection Analysis")
    print("=" * 70)

    tasks = list(task_data.keys())
    d = task_data[tasks[0]]["embeddings"].shape[1]
    beta_list = [0.0, 0.3, 0.5, 0.7, 0.85, 1.0]  # 1.0 = hard (InfLoRA), 0.0 = no projection

    results = {
        "tasks": {},
        "rank": rank,
        "d": d,
        "beta_list": beta_list,
        "pairwise_overlap": {},
        "cl_sequence": [],
    }

    # Compute per-task subspaces (top-r eigenvectors)
    task_subspaces = {}  # task → V (d, r)
    task_eigenvalues = {}

    for task in tasks:
        embs = task_data[task]["embeddings"]
        if _USE_GPU:
            X = _to_torch(embs)
            cov = _cov(X)
            eigvals, eigvecs = _eigh(cov)
            V = eigvecs[:, :rank]  # (d, r)
            task_subspaces[task] = V
            task_eigenvalues[task] = eigvals
            del X, cov
        else:
            cov = np.cov(embs, rowvar=False)
            eigvals = np.flip(np.sort(np.linalg.eigvalsh(cov)))
            eigvecs_all = np.linalg.eigh(cov)[1]
            eigvecs_all = eigvecs_all[:, ::-1]
            V = eigvecs_all[:, :rank]
            task_subspaces[task] = V
            task_eigenvalues[task] = eigvals

    # H7: Pairwise subspace overlap matrix
    print("\n  --- Pairwise Grassmannian Overlap (||V_t^T V_s||_F^2 / r) ---")
    overlap_matrix = np.zeros((len(tasks), len(tasks)))
    for i, ti in enumerate(tasks):
        for j, tj in enumerate(tasks):
            if i == j:
                overlap_matrix[i, j] = 1.0
                continue
            Vi = task_subspaces[ti]
            Vj = task_subspaces[tj]
            if _USE_GPU:
                ov = float((torch.linalg.norm(Vi.T @ Vj, "fro").item() ** 2) / rank)
            else:
                ov = float(np.linalg.norm(Vi.T @ Vj, "fro") ** 2 / rank)
            overlap_matrix[i, j] = ov

    results["pairwise_overlap"] = {
        "tasks": tasks,
        "matrix": overlap_matrix.tolist(),
    }

    # Check domain clusters for H7
    if "Long_Sequence" in str(tasks[:3]):
        cluster_overlaps = {}
        for cname, ctasks in LONG_SEQ_CLUSTERS.items():
            present = [t for t in ctasks if t in task_subspaces]
            if len(present) < 2:
                continue
            ovs = []
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    idx_i = tasks.index(present[i])
                    idx_j = tasks.index(present[j])
                    ovs.append(overlap_matrix[idx_i, idx_j])
            cluster_overlaps[cname] = {
                "mean_overlap": round(np.mean(ovs), 4),
                "tasks": present,
            }
        results["cluster_overlaps"] = cluster_overlaps
        print(f"\n  Same-domain overlap:")
        for cname, info in cluster_overlaps.items():
            print(f"    {cname:12s}: mean_overlap = {info['mean_overlap']:.4f} ({info['tasks']})")

    # Simulate CL sequence: tasks arrive one by one
    print(f"\n  --- CL Sequence Simulation (rank={rank}) ---")
    protected_subspace = None  # Will grow as tasks arrive

    for t_idx, task in enumerate(tasks):
        embs = task_data[task]["embeddings"]
        V_task = task_subspaces[task]

        if t_idx == 0:
            # First task: no projection needed
            results["cl_sequence"].append({
                "task": task,
                "task_idx": t_idx,
                "sse": 0.0,
                "beta_sweep": {str(b): {"info_retained": 1.0, "overlap_with_prev": 0.0} for b in beta_list},
            })
            if _USE_GPU:
                protected_subspace = V_task.clone()
            else:
                protected_subspace = V_task.copy()
            print(f"  Task {t_idx:2d} ({task:30s}): First task, no projection")
            continue

        # Compute SSE: how much of task's variance lies in protected subspace
        if _USE_GPU:
            X = _to_torch(embs)
            cov_task = _cov(X)
            # P_old = V_prev @ V_prev^T (projection matrix)
            P_old = protected_subspace @ protected_subspace.T
            sse = float((torch.trace(P_old @ cov_task) / torch.trace(cov_task).clamp(min=1e-15)).item())

            # Beta sweep: project V_task with varying strength
            beta_results = {}
            for beta in beta_list:
                if beta == 0.0:
                    V_proj = V_task
                else:
                    # V_proj = V_task - β * P_old @ V_task
                    V_proj = V_task - beta * (P_old @ V_task)
                    # Re-orthogonalize
                    V_proj, _ = torch.linalg.qr(V_proj)

                # Info retained: how much task variance captured after projection
                info_retained = float((torch.trace(V_proj.T @ cov_task @ V_proj) /
                                       torch.trace(V_task.T @ cov_task @ V_task).clamp(min=1e-15)).item())

                # Overlap with protected subspace
                overlap = float((torch.linalg.norm(V_proj.T @ protected_subspace, "fro").item() ** 2)
                                / rank)

                # Effective rank of projected subspace
                projected_cov = V_proj.T @ cov_task @ V_proj
                ev_proj, _ = _eigh(projected_cov)
                eff_rank = _par(ev_proj)

                beta_results[str(beta)] = {
                    "info_retained": round(info_retained, 4),
                    "overlap_with_prev": round(overlap, 4),
                    "effective_rank": round(eff_rank, 2),
                }

            # Accumulate protected subspace (simplified: stack and truncate)
            combined = torch.cat([protected_subspace, V_task], dim=1)
            U_comb, S_comb, _ = torch.linalg.svd(combined, full_matrices=False)
            max_dims = min(rank * (t_idx + 1), d)
            protected_subspace = U_comb[:, :max_dims]

            del X, cov_task, P_old
        else:
            cov_task = np.cov(embs, rowvar=False)
            P_old = protected_subspace @ protected_subspace.T
            sse = float(np.trace(P_old @ cov_task) / max(np.trace(cov_task), 1e-15))

            beta_results = {}
            for beta in beta_list:
                if beta == 0.0:
                    V_proj = V_task
                else:
                    V_proj = V_task - beta * (P_old @ V_task)
                    V_proj, _ = np.linalg.qr(V_proj)

                info_retained = float(np.trace(V_proj.T @ cov_task @ V_proj) /
                                      max(np.trace(V_task.T @ cov_task @ V_task), 1e-15))
                overlap_val = float(np.linalg.norm(V_proj.T @ protected_subspace, "fro") ** 2 / rank)

                projected_cov = V_proj.T @ cov_task @ V_proj
                ev_proj = np.flip(np.sort(np.linalg.eigvalsh(projected_cov)))
                eff_rank = _par(ev_proj)

                beta_results[str(beta)] = {
                    "info_retained": round(info_retained, 4),
                    "overlap_with_prev": round(overlap_val, 4),
                    "effective_rank": round(eff_rank, 2),
                }

            combined = np.hstack([protected_subspace, V_task])
            U_comb, S_comb, _ = np.linalg.svd(combined, full_matrices=False)
            max_dims = min(rank * (t_idx + 1), d)
            protected_subspace = U_comb[:, :max_dims]

        results["cl_sequence"].append({
            "task": task,
            "task_idx": t_idx,
            "sse": round(sse, 4),
            "beta_sweep": beta_results,
        })

        hard_info = beta_results["1.0"]["info_retained"]
        soft_info = beta_results["0.5"]["info_retained"]
        print(f"  Task {t_idx:2d} ({task:30s}): SSE={sse:.3f}  "
              f"Hard_info={hard_info:.3f}  Soft(β=0.5)_info={soft_info:.3f}  "
              f"Lost={(1 - hard_info):.1%}")

    # Summary
    sse_vals = [s["sse"] for s in results["cl_sequence"] if s["task_idx"] > 0]
    hard_info_vals = [s["beta_sweep"]["1.0"]["info_retained"] for s in results["cl_sequence"] if s["task_idx"] > 0]
    soft_info_vals = [s["beta_sweep"]["0.5"]["info_retained"] for s in results["cl_sequence"] if s["task_idx"] > 0]

    if sse_vals:
        results["summary"] = {
            "mean_sse": round(np.mean(sse_vals), 4),
            "max_sse": round(max(sse_vals), 4),
            "mean_info_hard": round(np.mean(hard_info_vals), 4),
            "mean_info_soft_05": round(np.mean(soft_info_vals), 4),
            "info_loss_hard_mean": round(1.0 - np.mean(hard_info_vals), 4),
            "info_gain_soft_vs_hard": round(np.mean(soft_info_vals) - np.mean(hard_info_vals), 4),
            "hypothesis_H3_hard_lossy": np.mean(hard_info_vals) < 0.95,
            "hypothesis_H3_soft_better": np.mean(soft_info_vals) > np.mean(hard_info_vals),
        }

        print(f"\n  [H3] Hard projection lossy? mean info_retained={np.mean(hard_info_vals):.4f} "
              f"→ {'YES ✓' if results['summary']['hypothesis_H3_hard_lossy'] else 'NO ✗'} "
              f"(mean SSE={np.mean(sse_vals):.4f})")
        print(f"  [H3] Soft(β=0.5) better? {np.mean(soft_info_vals):.4f} vs {np.mean(hard_info_vals):.4f} "
              f"→ {'YES ✓' if results['summary']['hypothesis_H3_soft_better'] else 'NO ✗'}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# PHASE 4: BNG — Activation Preconditioning Effect
# ═══════════════════════════════════════════════════════════════════════

def phase4_bng(task_data: dict, rank: int = 8) -> dict:
    """
    H4: Activations are anisotropic → preconditioning helps.
    H5: Routing duality — Σ_x^{-1} helps both routing and training.

    Measures:
    1. Condition number κ(Σ_x) per task
    2. Condition number after preconditioning κ(Σ_x^{-1/2} Σ_x Σ_x^{-1/2}) → should be 1
    3. Low-rank approximation quality: κ of preconditioned matrix with k eigenvectors
    4. Gradient amplification factor: how much weak directions get boosted
    5. Cross-task: does the same preconditioning matrix help routing metrics?
    """
    print("\n" + "=" * 70)
    print("PHASE 4: BNG — Activation Preconditioning Analysis")
    print("=" * 70)

    tasks = list(task_data.keys())
    d = task_data[tasks[0]]["embeddings"].shape[1]
    k_list = [4, 8, 16, 32]  # low-rank approximation ranks

    # Compute pooled covariance and its eigendecomposition
    all_embs = np.vstack([td["embeddings"] for td in task_data.values()])
    if _USE_GPU:
        all_t = _to_torch(all_embs)
        cov_pool = _cov(all_t)
        eigvals_pool, eigvecs_pool = _eigh(cov_pool)
        del all_t
    else:
        cov_pool_np = np.cov(all_embs, rowvar=False)
        eigvals_pool_np, eigvecs_pool_np_raw = np.linalg.eigh(cov_pool_np)
        idx = np.argsort(eigvals_pool_np)[::-1]
        eigvals_pool_np = eigvals_pool_np[idx]
        eigvecs_pool_np = eigvecs_pool_np_raw[:, idx]

    results = {"tasks": {}, "rank": rank, "d": d, "k_list": k_list}

    for task in tasks:
        embs = task_data[task]["embeddings"]
        n = embs.shape[0]

        if _USE_GPU:
            X = _to_torch(embs)
            cov_task = _cov(X)
            eigvals_task, eigvecs_task = _eigh(cov_task)

            # Raw condition number
            eigvals_pos = eigvals_task.clamp(min=1e-10)
            kappa_raw = float((eigvals_pos[0] / eigvals_pos[-1]).item())

            # Full preconditioning: Σ^{-1/2} Σ Σ^{-1/2} = I (ideal)
            # Low-rank preconditioning quality
            lowrank_kappas = {}
            for k in k_list:
                if k >= d:
                    continue
                # Low-rank inverse sqrt: V_k Λ_k^{-1/2} V_k^T + λ_bar^{-1/2} (I - V_k V_k^T)
                V_k = eigvecs_task[:, :k]
                lam_k = eigvals_pos[:k]
                lam_bar = eigvals_pos[k:].mean()

                # Construct preconditioned covariance
                # P^{-1/2} Σ P^{-1/2} where P = low-rank approx of Σ
                P_inv_sqrt_diag = torch.cat([1.0 / lam_k.sqrt(), torch.full((d - k,), 1.0 / lam_bar.sqrt(), device=_DEVICE)])
                # In eigvec basis of task: preconditioned eigenvalues = λ_i * p_i^{-1}
                # For top-k: λ_i * λ_i^{-1} = 1
                # For rest: λ_i * λ_bar^{-1}
                precond_eigvals = torch.cat([
                    torch.ones(k, device=_DEVICE),
                    eigvals_pos[k:] / lam_bar
                ])
                kappa_precond = float((precond_eigvals.max() / precond_eigvals.min().clamp(min=1e-10)).item())
                lowrank_kappas[k] = round(kappa_precond, 2)

            # Gradient amplification: ratio of precond eigenvalue / raw eigenvalue
            # For weak directions (bottom 10%): amplification = λ_bar^{-1/2} / λ_i^{-1/2}
            n_weak = max(d // 10, 1)
            weak_eigvals = eigvals_pos[-n_weak:]
            strong_eigvals = eigvals_pos[:n_weak]
            amplification_ratio = float((strong_eigvals.mean() / weak_eigvals.mean()).item())

            # After k=PaR preconditioning: amplification dramatically reduced
            par = _par(eigvals_task)
            k_par = max(2, min(int(round(par)), d - 1))
            if k_par < d:
                lam_bar_par = eigvals_pos[k_par:].mean()
                residual_spread = float((eigvals_pos[k_par:].max() / eigvals_pos[k_par:].min().clamp(min=1e-10)).item())
            else:
                lam_bar_par = eigvals_pos[-1]
                residual_spread = 1.0

            del X, cov_task
        else:
            cov_task = np.cov(embs, rowvar=False)
            eigvals_task = np.flip(np.sort(np.linalg.eigvalsh(cov_task)))
            eigvals_pos = np.maximum(eigvals_task, 1e-10)

            kappa_raw = float(eigvals_pos[0] / eigvals_pos[-1])

            lowrank_kappas = {}
            for k in k_list:
                if k >= d:
                    continue
                lam_bar = eigvals_pos[k:].mean()
                precond_eigvals = np.concatenate([
                    np.ones(k), eigvals_pos[k:] / max(lam_bar, 1e-10)
                ])
                kappa_precond = float(precond_eigvals.max() / max(precond_eigvals.min(), 1e-10))
                lowrank_kappas[k] = round(kappa_precond, 2)

            n_weak = max(d // 10, 1)
            weak_eigvals = eigvals_pos[-n_weak:]
            strong_eigvals = eigvals_pos[:n_weak]
            amplification_ratio = float(strong_eigvals.mean() / max(weak_eigvals.mean(), 1e-15))

            par = _par(eigvals_pos)
            k_par = max(2, min(int(round(par)), d - 1))
            if k_par < d:
                residual_spread = float(eigvals_pos[k_par:].max() / max(eigvals_pos[k_par:].min(), 1e-10))
            else:
                residual_spread = 1.0

        task_result = {
            "n_samples": n,
            "kappa_raw": round(kappa_raw, 1),
            "par": round(par, 2),
            "k_par": k_par,
            "lowrank_kappas": lowrank_kappas,
            "amplification_ratio": round(amplification_ratio, 1),
            "residual_spread_after_par": round(residual_spread, 2),
            "kappa_reduction_k8": round(kappa_raw / max(lowrank_kappas.get(8, kappa_raw), 0.01), 1) if 8 in lowrank_kappas else None,
        }
        results["tasks"][task] = task_result

        print(f"  {task:40s}  κ_raw={kappa_raw:8.1f}  PaR={par:5.1f}  "
              f"κ_precond(k=8)={lowrank_kappas.get(8, 'N/A'):>8}  "
              f"κ_precond(k=PaR)={lowrank_kappas.get(k_par, 'N/A'):>8}  "
              f"amplif={amplification_ratio:.1f}x")

    # H5: Routing-training duality
    # Compare: does per-task Σ_x differ much from pooled Σ_pool?
    print(f"\n  --- Routing-Training Duality (H5) ---")
    duality_results = {}
    for task in tasks:
        embs = task_data[task]["embeddings"]
        if _USE_GPU:
            X = _to_torch(embs)
            cov_task = _cov(X)
            # Cosine similarity between task and pool covariance (as vectors)
            cov_diff_norm = torch.linalg.norm(cov_task - cov_pool, "fro").item()
            cov_pool_norm = torch.linalg.norm(cov_pool, "fro").item()
            relative_diff = cov_diff_norm / max(cov_pool_norm, 1e-15)
            # Trace match
            trace_ratio = float((torch.trace(cov_task) / torch.trace(cov_pool).clamp(min=1e-15)).item())
            del X, cov_task
        else:
            cov_task = np.cov(embs, rowvar=False)
            cov_diff_norm = np.linalg.norm(cov_task - cov_pool_np, "fro")
            cov_pool_norm = np.linalg.norm(cov_pool_np, "fro")
            relative_diff = cov_diff_norm / max(cov_pool_norm, 1e-15)
            trace_ratio = float(np.trace(cov_task) / max(np.trace(cov_pool_np), 1e-15))

        duality_results[task] = {
            "cov_relative_diff": round(relative_diff, 4),
            "trace_ratio": round(trace_ratio, 4),
        }
        print(f"    {task:40s}  ||Σ_task - Σ_pool||/||Σ_pool|| = {relative_diff:.4f}  "
              f"trace_ratio = {trace_ratio:.4f}")

    results["duality_h5"] = duality_results

    # Summary
    kappas = [results["tasks"][t]["kappa_raw"] for t in tasks]
    k8_kappas = [results["tasks"][t]["lowrank_kappas"].get(8, None) for t in tasks]
    k8_kappas = [k for k in k8_kappas if k is not None]
    rel_diffs = [duality_results[t]["cov_relative_diff"] for t in tasks]

    results["summary"] = {
        "kappa_raw_range": [round(min(kappas), 1), round(max(kappas), 1)],
        "kappa_raw_mean": round(np.mean(kappas), 1),
        "kappa_precond_k8_mean": round(np.mean(k8_kappas), 1) if k8_kappas else None,
        "kappa_reduction_factor": round(np.mean(kappas) / max(np.mean(k8_kappas), 0.01), 1) if k8_kappas else None,
        "hypothesis_H4_anisotropic": np.mean(kappas) > 10.0,
        "hypothesis_H4_precond_helps": np.mean(k8_kappas) < np.mean(kappas) * 0.5 if k8_kappas else False,
        "cov_diff_mean": round(np.mean(rel_diffs), 4),
        "hypothesis_H5_duality": np.mean(rel_diffs) < 0.5,
    }

    print(f"\n  [H4] Anisotropic? mean κ={np.mean(kappas):.1f} "
          f"→ {'YES ✓' if results['summary']['hypothesis_H4_anisotropic'] else 'NO ✗'}")
    if k8_kappas:
        print(f"  [H4] Preconditioning helps? κ_raw={np.mean(kappas):.1f} → κ_precond(k=8)={np.mean(k8_kappas):.1f} "
              f"({results['summary']['kappa_reduction_factor']:.1f}x) "
              f"→ {'YES ✓' if results['summary']['hypothesis_H4_precond_helps'] else 'NO ✗'}")
    print(f"  [H5] Routing-training duality? mean ||Σ_task-Σ_pool||/||Σ_pool||={np.mean(rel_diffs):.4f} "
          f"→ {'YES ✓' if results['summary']['hypothesis_H5_duality'] else 'NO ✗'}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    global _USE_GPU, _DEVICE

    parser = argparse.ArgumentParser(description="GALA (C2) Empirical Validation")
    parser.add_argument("--emb_dir", required=True, help="e.g. embeddings/T5EncoderModel")
    parser.add_argument("--benchmark", required=True, choices=list(BENCHMARKS.keys()))
    parser.add_argument("--phase", type=int, default=0, help="Run only phase N (1-4). 0=all.")
    parser.add_argument("--whiten", action="store_true", help="Apply ZCA whitening")
    parser.add_argument("--device", default="auto", help="auto/cuda/cpu/mps")
    parser.add_argument("--rank", type=int, default=8, help="Subspace rank for phases 2-4")
    parser.add_argument("--rank_list", default="2,4,8,16,32,64", help="Ranks for TARA sweep")
    parser.add_argument("--n_trials", type=int, default=50, help="Random trials for GGI comparison")
    parser.add_argument("--output_dir", default="results", help="Output directory")
    args = parser.parse_args()

    device_str = _resolve_device(args.device)
    if HAS_TORCH and device_str != "cpu":
        _DEVICE = torch.device(device_str)
        _USE_GPU = True
    else:
        _DEVICE = None
        _USE_GPU = False

    tasks = BENCHMARKS[args.benchmark]
    backbone = Path(args.emb_dir).name
    rank_list = [int(x) for x in args.rank_list.split(",")]

    print(f"\n{'=' * 70}")
    print(f"GALA (Contribution 2) — Empirical Validation")
    print(f"{'=' * 70}")
    print(f"  Backbone:   {backbone}")
    print(f"  Benchmark:  {args.benchmark}")
    print(f"  Device:     {device_str}")
    print(f"  Whitened:   {args.whiten}")
    print(f"  Rank:       {args.rank}")
    print(f"  Phase:      {'all' if args.phase == 0 else args.phase}")

    # Load data
    print(f"\nLoading embeddings from {args.emb_dir}/{args.benchmark} ...")
    task_data = load_all_tasks(args.emb_dir, args.benchmark, tasks)
    found = list(task_data.keys())
    print(f"  Loaded {len(found)}/{len(tasks)} tasks")
    if not found:
        print("ERROR: No embeddings found. Check --emb_dir path.")
        sys.exit(1)

    d = task_data[found[0]]["embeddings"].shape[1]
    print(f"  d_model = {d}")
    for task in found[:3]:
        print(f"    {task}: {task_data[task]['embeddings'].shape}, labels={task_data[task]['labels'] is not None}")

    # Whitening
    if args.whiten:
        print("\nApplying ZCA whitening ...")
        mu, W = compute_whitening(task_data)
        task_data = apply_whitening(task_data, mu, W)

    # Run phases
    all_results = {
        "backbone": backbone,
        "benchmark": args.benchmark,
        "d_model": d,
        "whitened": args.whiten,
        "rank": args.rank,
        "device": device_str,
        "tasks_found": found,
    }

    t0 = time.time()

    if args.phase in (0, 1):
        all_results["phase1_tara"] = phase1_tara(task_data, rank_list)

    if args.phase in (0, 2):
        all_results["phase2_ggi"] = phase2_ggi(task_data, n_trials=args.n_trials, rank=args.rank)

    if args.phase in (0, 3):
        all_results["phase3_sgr"] = phase3_sgr(task_data, rank=args.rank)

    if args.phase in (0, 4):
        all_results["phase4_bng"] = phase4_bng(task_data, rank=args.rank)

    elapsed = time.time() - t0
    all_results["elapsed_seconds"] = round(elapsed, 1)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    suffix = "_whitened" if args.whiten else ""
    phase_suffix = f"_phase{args.phase}" if args.phase > 0 else ""
    out_path = os.path.join(
        args.output_dir,
        f"gala_{backbone}_{args.benchmark}{suffix}{phase_suffix}.json"
    )
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n{'=' * 70}")
    print(f"Results saved to {out_path}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"{'=' * 70}")

    # Print hypothesis summary
    print(f"\n{'=' * 70}")
    print("HYPOTHESIS SUMMARY")
    print(f"{'=' * 70}")
    for phase_key in ["phase1_tara", "phase2_ggi", "phase3_sgr", "phase4_bng"]:
        if phase_key not in all_results:
            continue
        summary = all_results[phase_key].get("summary", {})
        print(f"\n  {phase_key.upper()}:")
        for k, v in summary.items():
            if k.startswith("hypothesis_"):
                tag = "✓" if v else "✗"
                print(f"    [{tag}] {k}: {v}")


if __name__ == "__main__":
    main()
