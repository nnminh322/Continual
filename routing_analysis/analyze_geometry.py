#!/usr/bin/env python3
"""
Phase A — Geometric EDA of frozen backbone embeddings.

Computes per-task statistics from TRAIN split, evaluates routing on TEST split.
This mirrors the CL pipeline: build task signatures from train data only,
then classify unseen test inputs.

Output: results/ directory with JSON metrics + matplotlib figures.

Usage:
  python analyze_geometry.py --emb_dir embeddings/T5EncoderModel --benchmark Long_Sequence
  python analyze_geometry.py --emb_dir embeddings/LlamaForCausalLM --benchmark SuperNI
"""
from __future__ import annotations
import argparse, json, os, sys, warnings
from pathlib import Path
from collections import OrderedDict

import numpy as np
from numpy.linalg import eigh, svd, norm

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _resolve_device(device_str: str) -> str:
    """Resolve 'auto' → 'cuda' if GPU available, else 'cpu'."""
    if device_str == "auto":
        if HAS_TORCH and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[GPU] {gpu_name}  {vram_gb:.1f} GB VRAM — using CUDA")
            return "cuda"
        print("[GPU] No CUDA device found — using CPU")
        return "cpu"
    return device_str


# ═══════════════════════════════════════════════════════════════════════
# Global whitening (addresses anisotropy concern from review)
# ═══════════════════════════════════════════════════════════════════════

def compute_whitening(task_embs: dict[str, np.ndarray], device: str = "cpu"):
    """Compute global mean and ZCA whitening matrix.  Uses GPU when device='cuda*'."""
    if "cuda" in device and HAS_TORCH:
        dev = torch.device(device)
        chunks = [torch.tensor(v, dtype=torch.float32, device=dev) for v in task_embs.values()]
        all_t  = torch.cat(chunks, dim=0)
        mu_t   = all_t.mean(0)
        Xc     = all_t - mu_t
        cov_t  = (Xc.T @ Xc) / (all_t.shape[0] - 1)
        ev_t, evec_t = torch.linalg.eigh(cov_t)
        ev_t   = torch.clamp(ev_t, min=1e-8)
        W_t    = evec_t @ torch.diag(1.0 / torch.sqrt(ev_t)) @ evec_t.T
        return mu_t.cpu().numpy(), W_t.cpu().numpy()
    # --- CPU path ---
    all_embs = np.vstack(list(task_embs.values()))
    mu_global = all_embs.mean(0)
    cov_global = np.cov(all_embs, rowvar=False)
    eigvals, eigvecs = eigh(cov_global)
    # ZCA whitening: W = U diag(1/sqrt(lam)) U^T
    eigvals = np.maximum(eigvals, 1e-8)
    W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return mu_global, W


def apply_whitening(task_embs: dict[str, np.ndarray], mu_global, W):
    """Center and whiten all task embeddings."""
    return {t: (embs - mu_global) @ W.T for t, embs in task_embs.items()}

# ── Benchmark → task list mapping ──
BENCHMARKS = {
    "Long_Sequence": [
        "yelp","amazon","mnli","cb","copa","qqp","rte",
        "imdb","sst2","dbpedia","agnews","yahoo","multirc","boolq","wic",
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

# Domain clusters for Long_Sequence analysis
LONG_SEQ_CLUSTERS = {
    "sentiment": ["yelp","amazon","imdb","sst2"],
    "NLI":       ["mnli","cb","rte"],
    "topic":     ["dbpedia","agnews","yahoo"],
    "RC":        ["multirc","boolq"],
    "misc":      ["copa","qqp","wic"],
}


# ═══════════════════════════════════════════════════════════════════════
# Loaders
# ═══════════════════════════════════════════════════════════════════════

def load_split(emb_dir: str, benchmark: str, task: str, split: str):
    """Load embeddings + labels from .npz.  Returns (embs, labels) or (None, None)."""
    p = Path(emb_dir) / benchmark / task / f"{split}.npz"
    if not p.exists():
        return None, None
    data = np.load(str(p), allow_pickle=True)
    return data["embeddings"].astype(np.float64), data["labels"]


def load_all_tasks(emb_dir, benchmark, tasks, split="train"):
    """Load one split for every task. Returns OrderedDict[task] → embs."""
    out = OrderedDict()
    for t in tasks:
        embs, _ = load_split(emb_dir, benchmark, t, split)
        if embs is not None:
            out[t] = embs
    return out


# ═══════════════════════════════════════════════════════════════════════
# A1 — Effective dimensionality
# ═══════════════════════════════════════════════════════════════════════

def effective_dim(embs: np.ndarray, threshold=0.95):
    """Return EVR-k*, participation ratio, effective rank."""
    cov = np.cov(embs, rowvar=False)                     # (d, d)
    eigvals = np.flip(np.sort(np.linalg.eigvalsh(cov)))   # descending
    eigvals = np.maximum(eigvals, 0)                       # numerical safety
    total = eigvals.sum()
    if total < 1e-12:
        return {"evr_k": 0, "participation_ratio": 0.0, "effective_rank": 0.0}
    cumvar = np.cumsum(eigvals) / total
    evr_k = int(np.searchsorted(cumvar, threshold)) + 1
    pr = total**2 / (eigvals**2).sum()
    # effective rank via entropy
    p = eigvals / total
    p = p[p > 1e-15]
    erank = float(np.exp(-np.sum(p * np.log(p))))
    return {"evr_k95": int(evr_k), "participation_ratio": float(pr),
            "effective_rank": float(erank), "top5_eigvals": eigvals[:5].tolist(),
            "d": int(embs.shape[1]), "n": int(embs.shape[0])}


# ═══════════════════════════════════════════════════════════════════════
# A2 — Gaussianity (marginal kurtosis on top PCs)
# ═══════════════════════════════════════════════════════════════════════

def gaussianity_check(embs: np.ndarray, n_components=10):
    """Marginal excess kurtosis on top PCs.  Gaussian → ~0."""
    from scipy.stats import kurtosis as sp_kurtosis
    cov = np.cov(embs, rowvar=False)
    eigvals, eigvecs = eigh(cov)
    idx = np.argsort(eigvals)[::-1][:n_components]
    V = eigvecs[:, idx]  # (d, k)
    scores = (embs - embs.mean(0)) @ V  # (N, k)
    kurt = [float(sp_kurtosis(scores[:, i], fisher=True)) for i in range(scores.shape[1])]
    return {"excess_kurtosis_top_pcs": kurt,
            "mean_abs_kurtosis": float(np.mean(np.abs(kurt)))}


# ═══════════════════════════════════════════════════════════════════════
# A2b — Multi-modality test (GMM vs single Gaussian)
# ═══════════════════════════════════════════════════════════════════════

def multimodality_test(embs: np.ndarray, max_components=5):
    """Fit GMM with 1..max_components, compare BIC. Multi-modal if best_k > 1."""
    from sklearn.mixture import GaussianMixture
    n = embs.shape[0]
    # Project to top PCs first (GMM in full d is expensive and unreliable)
    n_pcs = min(30, embs.shape[1], n - 1)
    cov = np.cov(embs, rowvar=False)
    eigvals, eigvecs = eigh(cov)
    idx = np.argsort(eigvals)[::-1][:n_pcs]
    X_pca = (embs - embs.mean(0)) @ eigvecs[:, idx]

    bics = {}
    for k in range(1, max_components + 1):
        if n < k * (n_pcs + 1):  # not enough data
            break
        gmm = GaussianMixture(n_components=k, covariance_type="diag",
                               random_state=42, max_iter=100)
        gmm.fit(X_pca)
        bics[k] = float(gmm.bic(X_pca))
    best_k = min(bics, key=bics.get) if bics else 1
    return {"bic_per_k": bics, "best_k": best_k,
            "multi_modal": best_k > 1, "n_pcs_used": n_pcs}


# ═══════════════════════════════════════════════════════════════════════
# A3 — Centroid distances
# ═══════════════════════════════════════════════════════════════════════

def centroid_distances(task_embs: dict[str, np.ndarray]):
    """Pairwise centroid distances: cosine, L2, normalized-L2."""
    tasks = list(task_embs.keys())
    centroids = {t: embs.mean(0) for t, embs in task_embs.items()}
    n = len(tasks)
    cos_mat = np.zeros((n, n))
    l2_mat  = np.zeros((n, n))
    for i in range(n):
        ci = centroids[tasks[i]]
        ni = ci / (norm(ci) + 1e-12)
        for j in range(i+1, n):
            cj = centroids[tasks[j]]
            nj = cj / (norm(cj) + 1e-12)
            cos_mat[i, j] = cos_mat[j, i] = float(1.0 - ni @ nj)
            l2_mat[i, j]  = l2_mat[j, i]  = float(norm(ci - cj))
    return {"tasks": tasks, "cosine_dist": cos_mat.tolist(), "l2_dist": l2_mat.tolist()}


# ═══════════════════════════════════════════════════════════════════════
# A4 — Principal angles on Grassmannian
# ═══════════════════════════════════════════════════════════════════════

def principal_angles(V1: np.ndarray, V2: np.ndarray):
    """Principal angles between span(V1) and span(V2).  V_i: (d, k)."""
    M = V1.T @ V2  # (k1, k2)
    svals = np.clip(svd(M, compute_uv=False), 0.0, 1.0)
    angles = np.arccos(np.clip(svals, -1, 1))
    return angles  # radians, ascending-ish


def subspace_distances(task_embs: dict[str, np.ndarray], k=8):
    """Pairwise Grassmann geodesic, chordal, projection distances."""
    tasks = list(task_embs.keys())
    # top-k eigenvectors per task
    bases = {}
    for t, embs in task_embs.items():
        cov = np.cov(embs, rowvar=False)
        eigvals, eigvecs = eigh(cov)
        idx = np.argsort(eigvals)[::-1][:k]
        bases[t] = eigvecs[:, idx]  # (d, k)

    n = len(tasks)
    geodesic  = np.zeros((n, n))
    chordal   = np.zeros((n, n))
    proj_dist = np.zeros((n, n))
    fro_overlap = np.zeros((n, n))
    for i in range(n):
        Vi = bases[tasks[i]]
        for j in range(i+1, n):
            Vj = bases[tasks[j]]
            angles = principal_angles(Vi, Vj)
            geodesic[i,j] = geodesic[j,i] = float(norm(angles))
            chordal[i,j]  = chordal[j,i]  = float(np.sqrt(k - np.sum(np.cos(angles)**2)))
            Pi = Vi @ Vi.T; Pj = Vj @ Vj.T
            proj_dist[i,j] = proj_dist[j,i] = float(norm(Pi - Pj, 'fro'))
            fro_overlap[i,j] = fro_overlap[j,i] = float(np.sum(np.cos(angles)**2))  # δ_ij

    return {"tasks": tasks, "k": k,
            "geodesic": geodesic.tolist(), "chordal": chordal.tolist(),
            "projection": proj_dist.tolist(), "frobenius_overlap": fro_overlap.tolist()}


# ═══════════════════════════════════════════════════════════════════════
# A5 — Anisotropy
# ═══════════════════════════════════════════════════════════════════════

def anisotropy_metrics(embs: np.ndarray):
    cov = np.cov(embs, rowvar=False)
    eigvals = np.flip(np.sort(np.linalg.eigvalsh(cov)))
    eigvals = np.maximum(eigvals, 0)
    if eigvals[0] < 1e-12:
        return {"anisotropy_ratio": 0, "isotropy_score": 0}
    return {
        "anisotropy_ratio": float(eigvals[0] / max(eigvals[-1], 1e-15)),
        "isotropy_score":   float(eigvals[-1] / eigvals[0]),
        "condition_number": float(eigvals[0] / max(eigvals[min(len(eigvals)-1, 99)], 1e-15)),
    }


# ═══════════════════════════════════════════════════════════════════════
# A7 — Few-shot stability
# ═══════════════════════════════════════════════════════════════════════

def fewshot_stability(embs: np.ndarray, k=8, n_samples_list=(50, 100, 200, 500),
                      n_repeats=5, rng=None):
    """Measure how stable top-k eigenvectors are under subsampling."""
    if rng is None:
        rng = np.random.default_rng(42)
    N = embs.shape[0]
    # reference from full dataset
    cov_full = np.cov(embs, rowvar=False)
    eigvals_full, eigvecs_full = eigh(cov_full)
    idx = np.argsort(eigvals_full)[::-1][:k]
    V_full = eigvecs_full[:, idx]

    results = {}
    for ns in n_samples_list:
        if ns >= N:
            continue
        dists = []
        for _ in range(n_repeats):
            sel = rng.choice(N, size=ns, replace=False)
            cov_sub = np.cov(embs[sel], rowvar=False)
            ev, evec = eigh(cov_sub)
            V_sub = evec[:, np.argsort(ev)[::-1][:k]]
            # projection distance
            d = float(norm(V_full @ V_full.T - V_sub @ V_sub.T, 'fro'))
            dists.append(d)
        results[str(ns)] = {"mean_proj_dist": float(np.mean(dists)),
                            "std_proj_dist":  float(np.std(dists))}
    return results


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase A — Geometric EDA")
    parser.add_argument("--emb_dir", required=True,
                        help="e.g. embeddings/T5EncoderModel")
    parser.add_argument("--benchmark", required=True, choices=list(BENCHMARKS.keys()))
    parser.add_argument("--subspace_k", type=int, default=8,
                        help="Rank for subspace analysis (default 8 = LoRA rank)")
    parser.add_argument("--whiten", action="store_true",
                        help="Apply global ZCA whitening before analysis (addresses anisotropy)")
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--device", default="auto",
                        help="Device: cpu | cuda | cuda:0 | auto (default: auto)")
    args = parser.parse_args()
    args.device = _resolve_device(args.device)

    tasks = BENCHMARKS[args.benchmark]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    backbone = Path(args.emb_dir).name  # e.g. T5EncoderModel
    tag = f"{backbone}_{args.benchmark}" + ("_whitened" if args.whiten else "")

    print(f"=== Phase A: Geometric EDA  [{tag}] ===")
    if args.whiten:
        print("  [whitening enabled — global ZCA applied]")
    print()

    # Load TRAIN embeddings for building signatures
    task_embs = load_all_tasks(args.emb_dir, args.benchmark, tasks, "train")
    found = list(task_embs.keys())
    print(f"Loaded {len(found)}/{len(tasks)} tasks from {args.emb_dir}/{args.benchmark}")
    if not found:
        print("ERROR: No embeddings found. Check --emb_dir path."); sys.exit(1)
    d = next(iter(task_embs.values())).shape[1]
    print(f"d_model = {d}\n")

    # Optional global whitening
    if args.whiten:
        mu_g, W = compute_whitening(task_embs, device=args.device)
        task_embs = apply_whitening(task_embs, mu_g, W)
        print("Applied ZCA whitening to remove global anisotropy\n")

    report = {"backbone": backbone, "benchmark": args.benchmark,
              "d_model": d, "tasks_found": found, "subspace_k": args.subspace_k,
              "whitened": args.whiten}

    # ── A1: Effective dimensionality ──
    print("─── A1: Effective dimensionality ───")
    dim_info = {}
    for t, embs in task_embs.items():
        info = effective_dim(embs)
        dim_info[t] = info
        print(f"  {t:50s}  n={info['n']:5d}  EVR-k95={info['evr_k95']:3d}"
              f"  PR={info['participation_ratio']:.1f}  erank={info['effective_rank']:.1f}")
    report["A1_effective_dim"] = dim_info

    # ── A2: Gaussianity ──
    print("\n─── A2: Gaussianity (excess kurtosis on top PCs) ───")
    gauss_info = {}
    for t, embs in task_embs.items():
        info = gaussianity_check(embs)
        gauss_info[t] = info
        print(f"  {t:50s}  mean|kurt|={info['mean_abs_kurtosis']:.3f}"
              f"  top5={[f'{k:.2f}' for k in info['excess_kurtosis_top_pcs'][:5]]}")
    report["A2_gaussianity"] = gauss_info

    # ── A2b: Multi-modality (GMM BIC) ──
    print("\n─── A2b: Multi-modality test (GMM, BIC) ───")
    multi_info = {}
    for t, embs in task_embs.items():
        info = multimodality_test(embs)
        multi_info[t] = info
        flag = " ← MULTI-MODAL" if info["multi_modal"] else ""
        print(f"  {t:50s}  best_k={info['best_k']}{flag}")
    report["A2b_multimodality"] = multi_info

    # ── A3: Centroid distances ──
    print("\n─── A3: Centroid distances ───")
    cd = centroid_distances(task_embs)
    report["A3_centroid_distances"] = cd
    # Print top 10 closest pairs (cosine)
    cos = np.array(cd["cosine_dist"])
    pairs = []
    for i in range(len(found)):
        for j in range(i+1, len(found)):
            pairs.append((cos[i,j], found[i], found[j]))
    pairs.sort()
    print("  Top 10 closest (cosine distance):")
    for d_val, a, b in pairs[:10]:
        print(f"    {a:30s} ↔ {b:30s}  cos_dist={d_val:.4f}")

    # ── A4: Subspace (Grassmannian) analysis ──
    print(f"\n─── A4: Subspace analysis (k={args.subspace_k}) ───")
    sd = subspace_distances(task_embs, k=args.subspace_k)
    report["A4_subspace_distances"] = sd
    geo = np.array(sd["geodesic"])
    overlap = np.array(sd["frobenius_overlap"])
    pairs_g = []
    for i in range(len(found)):
        for j in range(i+1, len(found)):
            pairs_g.append((overlap[i,j], found[i], found[j], geo[i,j]))
    pairs_g.sort(reverse=True)
    print("  Top 10 highest subspace overlap (δ_ij = ||Vi'Vj||_F^2):")
    for ov, a, b, g in pairs_g[:10]:
        print(f"    {a:30s} ↔ {b:30s}  δ={ov:.4f}  geodesic={g:.4f}")
    print(f"  Grassmann capacity bound T_max ≤ {d}/({args.subspace_k}*(1-{overlap.max():.3f}))"
          f" = {d / (args.subspace_k * (1 - overlap.max() + 1e-9)):.0f}")

    # ── A5: Anisotropy ──
    print("\n─── A5: Anisotropy ───")
    aniso_info = {}
    for t, embs in task_embs.items():
        info = anisotropy_metrics(embs)
        aniso_info[t] = info
        print(f"  {t:50s}  aniso_ratio={info['anisotropy_ratio']:.1f}"
              f"  isotropy={info['isotropy_score']:.6f}")
    report["A5_anisotropy"] = aniso_info

    # ── A7: Few-shot stability ──
    print(f"\n─── A7: Few-shot stability (k={args.subspace_k}) ───")
    fewshot_info = {}
    for t, embs in task_embs.items():
        info = fewshot_stability(embs, k=args.subspace_k)
        fewshot_info[t] = info
        summary = "  ".join([f"n={n}: {v['mean_proj_dist']:.3f}±{v['std_proj_dist']:.3f}"
                             for n, v in info.items()])
        print(f"  {t:40s}  {summary}")
    report["A7_fewshot_stability"] = fewshot_info

    # ── Save ──
    out_path = out_dir / f"geometry_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n✓ Saved {out_path}")


if __name__ == "__main__":
    main()
