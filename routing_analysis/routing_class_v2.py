#!/usr/bin/env python3
"""
Continual Learning Routing Evaluation — Theoretically Grounded + CUDA-accelerated.

Design principles:
  1. ZCA whitening: fit ONCE from accumulated embeddings, then frozen.
     → WhitenedNearestCentroid: ZCA + L2
     → WhitenedCosine: ZCA + cosine
     No shrinkage. Simple. The whitening transform is fit once when buffer ≥ min_samples.
  2. All routers support CUDA via PyTorch (eigendecomposition on GPU).
  3. Adaptive metric via Participation Ratio (SRT Thm 6).

Mathematical foundation:
  - ZCA: W_zca = V @ diag(1/√λ) @ Vᵀ such that (h-μ)ᵀWᵀW(h-μ) is isotropic.
  - After ZCA: routing by L2 in whitened space ≡ routing by Mahalanobis in raw space.
  - Fit ZCA from accumulated pool when n_pool ≥ min_samples.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from collections import OrderedDict
from pathlib import Path
from math import log

import numpy as np

# ─── CUDA support ──────────────────────────────────────────────────────────────
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
    DEVICE = "cuda" if HAS_CUDA else "cpu"
    if HAS_CUDA:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU] {gpu_name}  {vram_gb:.1f} GB — using CUDA")
    else:
        print("[CPU] No CUDA available — using CPU")
except ImportError:
    HAS_CUDA = False
    DEVICE = "cpu"
    print("[CPU] PyTorch not available")


# ─── Shared constants ─────────────────────────────────────────────────────────

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


# ══════════════════════════════════════════════════════════════════════════════
# ZCA WHITENING  (torch — runs on GPU when available)
# ══════════════════════════════════════════════════════════════════════════════

def fit_zca_torch(embs_list, device="cuda", rcond=1e-6):
    """
    Fit ZCA whitening from pooled embeddings (torch, GPU-accelerated).

    W_zca = V @ diag(1/√λ) @ Vᵀ
    No shrinkage. Simple oracle-free approach.

    Args:
        embs_list: list of (N, d) numpy arrays
        device: "cuda" or "cpu"
        rcond: eigenvalue threshold for numerical stability

    Returns:
        mu: (d,) mean vector (numpy)
        W: (d, d) whitening matrix (numpy)
    """
    dev = torch.device(device)

    # Stack all embeddings
    tensors = [torch.from_numpy(e.astype(np.float32)) for e in embs_list if e is not None]
    all_t = torch.cat(tensors, dim=0).to(dev)  # (N_total, d)

    # Mean and centered data
    mu_t = all_t.mean(dim=0)                    # (d,)
    N = all_t.shape[0]

    # Sample covariance: Σ = XcᵀXc / (N-1)
    Xc = all_t - mu_t                           # (N, d)
    cov_t = (Xc.T @ Xc) / max(N - 1, 1)       # (d, d)

    del all_t, Xc, tensors
    if device == "cuda":
        torch.cuda.empty_cache()

    # Eigendecomposition on GPU
    eigvals, eigvecs = torch.linalg.eigh(cov_t)  # ascending
    eigvals = torch.clamp(eigvals, min=rcond * eigvals.abs().max().item())

    # Sort descending
    idx = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # ZCA: W = V @ diag(1/√λ) @ Vᵀ
    W_t = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals)) @ eigvecs.T

    return mu_t.cpu().numpy(), W_t.cpu().numpy(), N


def apply_whitening_torch(h, mu, W, device="cuda"):
    """
    Apply ZCA whitening: h_w = (h - mu) @ W.T
    Uses GPU when available.
    """
    if h.ndim == 1:
        h = h.reshape(1, -1)

    dev = torch.device(device)

    if isinstance(h, np.ndarray):
        h_t = torch.from_numpy(h.astype(np.float32)).to(dev)
    else:
        h_t = h.to(dev)

    if isinstance(mu, np.ndarray):
        mu_t = torch.from_numpy(mu.astype(np.float32)).to(dev)
    else:
        mu_t = mu.to(dev)

    if isinstance(W, np.ndarray):
        W_t = torch.from_numpy(W.astype(np.float32)).to(dev)
    else:
        W_t = W.to(dev)

    h_w = (h_t - mu_t) @ W_t.T
    return h_w.cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_split(emb_dir, benchmark, task, split):
    p = Path(emb_dir) / benchmark / task / f"{split}.npz"
    if not p.exists():
        return None
    data = np.load(str(p), allow_pickle=True)
    return data["embeddings"].astype(np.float64)


def load_all(emb_dir, benchmark, tasks, split):
    """Load all tasks. No caps."""
    out = OrderedDict()
    for t in tasks:
        embs = load_split(emb_dir, benchmark, t, split)
        if embs is not None:
            out[t] = embs
    return out


# ══════════════════════════════════════════════════════════════════════════════
# PARTICIPATION RATIO  (torch, GPU-accelerated)
# ══════════════════════════════════════════════════════════════════════════════

def participation_ratio_torch(cov_t, device="cuda"):
    """
    PaR = (Σλ)² / Σλ² ∈ [1, d].
    Uses torch eigendecomposition on GPU.
    """
    dev = torch.device(device)
    if isinstance(cov_t, np.ndarray):
        cov_t = torch.from_numpy(cov_t.astype(np.float32)).to(dev)
    eigvals = torch.linalg.eigvalsh(cov_t)
    eigvals = torch.clamp(eigvals, min=1e-10)
    par = (eigvals.sum() ** 2) / ((eigvals ** 2).sum())
    return par.cpu().item()


# ══════════════════════════════════════════════════════════════════════════════
# INCREMENTAL POOLED STATISTICS  (Welford-Hart, torch)
# ══════════════════════════════════════════════════════════════════════════════

def welford_pooled_update(mu_old_t, cov_old_t, n_old, mu_new_t, cov_new_t, n_new):
    """
    Welford-Hart pooled update (torch).
    Returns: (mu_pool, cov_pool, n_pool)
    """
    total = n_old + n_new
    mu_pool = (n_old * mu_old_t + n_new * mu_new_t) / total
    delta = mu_new_t - mu_old_t
    C = (n_old * n_new / total) * torch.outer(delta, delta)
    cov_pool = ((n_old - 1) * cov_old_t + (n_new - 1) * cov_new_t + C) / max(total - 1, 1)
    return mu_pool, cov_pool, total


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════════════════

class NearestCentroidRouter:
    """
    Baseline: L2 distance to raw centroid.
    Optimal when Σ ≈ σ²I (isotropic case).
    """
    def __init__(self, device="cpu"):
        self.device = device
        self.centroids = []

    def add_task(self, embs):
        self.centroids.append(embs.mean(axis=0))

    def route(self, h_batch):
        C = np.stack(self.centroids)
        H_sq = np.sum(h_batch ** 2, axis=1, keepdims=True)
        C_sq = np.sum(C ** 2, axis=1)
        dists = H_sq + C_sq[None, :] - 2 * (h_batch @ C.T)
        return dists.argmin(axis=1)


class CosineNearestCentroidRouter:
    """
    Baseline: cosine similarity to raw centroid.
    Scale-invariant but not whitening-aware.
    """
    def __init__(self, device="cpu"):
        self.device = device
        self.centroids = []

    def add_task(self, embs):
        mu = embs.mean(0)
        self.centroids.append(mu / (np.linalg.norm(mu) + 1e-12))

    def route(self, h_batch):
        C = np.stack(self.centroids)
        h_norm = h_batch / (np.linalg.norm(h_batch, axis=1, keepdims=True) + 1e-12)
        sims = h_norm @ C.T
        return sims.argmax(axis=1)


class WhitenedNearestCentroidRouter:
    """
    ZCA whitening + L2 distance (equivalent to Pooled Mahalanobis in raw space).

    Protocol:
      1. Accumulate embeddings in buffer.
      2. When buffer has ≥ min_samples: fit ZCA ONCE (all accumulated embeddings).
      3. Freeze ZCA. Transform all centroids to whitened space.
      4. For new tasks: compute whitened centroid directly (no re-fitting ZCA).
      5. Route by L2 in whitened space.

    This is the SIMPLE whitening approach (no shrinkage, no re-fitting).
    Matches SRT Thm 4: ZCAWhitened + L2 ≡ Pooled Mahalanobis.

    After ZCA fitting:
      - mu_w = (mu_raw - mu_global) @ W_zca.T
      - Route: d_t(h) = ||h_w - mu_w_t||²
             = (h - mu_t)ᵀWᵀW(h - mu_t)
             = (h - mu_t)ᵀΣ_pool⁻¹(h - mu_t)  [when W = Σ⁻¹/²]
    """
    def __init__(self, min_samples=800, device="cuda"):
        self.min_samples = min_samples
        self.device = device
        self.dev = torch.device(device)

        # Raw centroids (accumulated)
        self.raw_centroids = []  # list of (d,) numpy arrays
        self._emb_buffer = []   # list of (n_i, d) numpy arrays
        self.n_seen = 0

        # ZCA state (fitted once)
        self.mu_global = None   # (d,) numpy
        self.W_zca = None       # (d, d) numpy
        self.zca_fitted = False

        # Whitened centroids
        self.centroids_whitened = []  # list of (d,) numpy

    def _fit_zca(self):
        """Fit ZCA ONCE from accumulated buffer. Then freeze."""
        if self.zca_fitted:
            return
        all_embs = [e for e in self._emb_buffer if e is not None and e.shape[0] > 0]
        if len(all_embs) == 0:
            return
        mu, W, N = fit_zca_torch(all_embs, device=self.device)
        self.mu_global = mu
        self.W_zca = W
        self.zca_fitted = True

        # Re-whiten all existing centroids
        self.centroids_whitened = []
        for mu_raw in self.raw_centroids:
            mu_w = apply_whitening_torch(mu_raw, mu, W, device=self.device)
            if mu_w.ndim > 1:
                mu_w = mu_w.squeeze(0)
            self.centroids_whitened.append(mu_w.astype(np.float64))

        print(f"    [WhitenedL2] ZCA fitted: N={N}, zca_fitted=True")

    def add_task(self, embs, task_name=None):
        n_t, d = embs.shape
        mu_raw = embs.mean(axis=0)

        self.raw_centroids.append(mu_raw)
        self._emb_buffer.append(embs.copy())
        self.n_seen += n_t

        # Fit ZCA when buffer is large enough (only once!)
        if not self.zca_fitted and self.n_seen >= self.min_samples:
            self._fit_zca()

        # Add whitened centroid
        if self.zca_fitted:
            mu_w = apply_whitening_torch(mu_raw, self.mu_global, self.W_zca, device=self.device)
            if mu_w.ndim > 1:
                mu_w = mu_w.squeeze(0)
            self.centroids_whitened.append(mu_w.astype(np.float64))
        else:
            # ZCA not yet fitted — use raw centroid (first few tasks)
            self.centroids_whitened.append(mu_raw.astype(np.float64))

    def route(self, h_batch):
        if len(self.centroids_whitened) == 0:
            return np.zeros(h_batch.shape[0], dtype=np.int64)

        if self.zca_fitted:
            # Use whitened centroids + whitened query
            C_w = np.stack(self.centroids_whitened)
            H_w = apply_whitening_torch(h_batch, self.mu_global, self.W_zca, device=self.device)
            if H_w.ndim == 1:
                H_w = H_w.reshape(1, -1)
            H_w = H_w.astype(np.float64)

            H_sq = np.sum(H_w ** 2, axis=1, keepdims=True)
            C_sq = np.sum(C_w ** 2, axis=1)
            dists = H_sq + C_sq[None, :] - 2 * (H_w @ C_w.T)
        else:
            # Fallback: raw L2
            C = np.stack(self.centroids_whitened)
            H_sq = np.sum(h_batch ** 2, axis=1, keepdims=True)
            C_sq = np.sum(C ** 2, axis=1)
            dists = H_sq + C_sq[None, :] - 2 * (h_batch @ C.T)

        return dists.argmin(axis=1)


class WhitenedCosineRouter:
    """
    ZCA whitening + cosine similarity.

    Protocol:
      1. Accumulate embeddings in buffer.
      2. Fit ZCA ONCE when buffer ≥ min_samples.
      3. Transform centroids to whitened space.
      4. Route by cosine similarity in whitened space.

    Cosine in whitened space measures angular alignment in the
    Mahalanobis-normalized metric: cos(h_w, μ_w).
    This is invariant to both scale AND the anisotropy captured by ZCA.
    """
    def __init__(self, min_samples=800, device="cuda"):
        self.min_samples = min_samples
        self.device = device
        self.dev = torch.device(device)

        self.raw_centroids = []
        self._emb_buffer = []
        self.n_seen = 0

        self.mu_global = None
        self.W_zca = None
        self.zca_fitted = False
        self.centroids_whitened = []

    def _fit_zca(self):
        if self.zca_fitted:
            return
        all_embs = [e for e in self._emb_buffer if e is not None and e.shape[0] > 0]
        if len(all_embs) == 0:
            return
        mu, W, N = fit_zca_torch(all_embs, device=self.device)
        self.mu_global = mu
        self.W_zca = W
        self.zca_fitted = True

        self.centroids_whitened = []
        for mu_raw in self.raw_centroids:
            mu_w = apply_whitening_torch(mu_raw, mu, W, device=self.device)
            if mu_w.ndim > 1:
                mu_w = mu_w.squeeze(0)
            # Normalize for cosine
            norm = np.linalg.norm(mu_w) + 1e-12
            self.centroids_whitened.append((mu_w / norm).astype(np.float64))

        print(f"    [WhitenedCosine] ZCA fitted: N={N}, zca_fitted=True")

    def add_task(self, embs, task_name=None):
        n_t, d = embs.shape
        mu_raw = embs.mean(axis=0)

        self.raw_centroids.append(mu_raw)
        self._emb_buffer.append(embs.copy())
        self.n_seen += n_t

        if not self.zca_fitted and self.n_seen >= self.min_samples:
            self._fit_zca()

        if self.zca_fitted:
            mu_w = apply_whitening_torch(mu_raw, self.mu_global, self.W_zca, device=self.device)
            if mu_w.ndim > 1:
                mu_w = mu_w.squeeze(0)
            norm = np.linalg.norm(mu_w) + 1e-12
            self.centroids_whitened.append((mu_w / norm).astype(np.float64))
        else:
            # Fallback: raw cosine
            norm = np.linalg.norm(mu_raw) + 1e-12
            self.centroids_whitened.append((mu_raw / norm).astype(np.float64))

    def route(self, h_batch):
        if len(self.centroids_whitened) == 0:
            return np.zeros(h_batch.shape[0], dtype=np.int64)

        if self.zca_fitted:
            C = np.stack(self.centroids_whitened)
            H_w = apply_whitening_torch(h_batch, self.mu_global, self.W_zca, device=self.device)
            if H_w.ndim == 1:
                H_w = H_w.reshape(1, -1)
            H_w = H_w.astype(np.float64)
            # Cosine: normalize query, compute dot product
            H_norm = H_w / (np.linalg.norm(H_w, axis=1, keepdims=True) + 1e-12)
            sims = H_norm @ C.T
        else:
            # Fallback: raw cosine
            C = np.stack(self.centroids_whitened)
            H_norm = h_batch / (np.linalg.norm(h_batch, axis=1, keepdims=True) + 1e-12)
            sims = H_norm @ C.T

        return sims.argmax(axis=1)


class AdaptivePooledMahalanobisRouter:
    """
    SRT Thm 6: Adaptive metric via Participation Ratio.

    Computes PaR of Σ_pool at each step:
      - If PaR/d > threshold: use L2 (isotropic)
      - Else: use Pooled Mahalanobis (anisotropic — whitening critical)

    Threshold is theoretically grounded: PaR/d → 1 when isotropic,
    PaR/d → 0 when low-rank. The threshold distinguishes these regimes.
    Uses Welford-Hart for incremental pooled covariance (GPU when available).
    """
    def __init__(self, par_threshold=0.9, device="cuda"):
        self.par_threshold = par_threshold
        self.device = device
        self.dev = torch.device(device)

        self.centroids = []   # raw μ_t
        self.n_tasks_list = []
        self.mu_pool_t = None  # torch
        self.Sigma_pool_t = None  # torch
        self.n_pool = 0

        self.Sinv = None  # torch (shrunk inverse)
        self.current_metric = 'l2'
        self.par = 0.0

    def add_task(self, embs, task_name=None):
        n_t, d = embs.shape

        # Compute sufficient statistics (torch)
        X = torch.from_numpy(embs.astype(np.float32)).to(self.dev)
        mu_t_t = X.mean(dim=0)
        Xc = X - mu_t_t
        Sigma_t_t = (Xc.T @ Xc) / max(n_t - 1, 1)

        del X, Xc
        if self.device == "cuda":
            torch.cuda.empty_cache()

        self.centroids.append(mu_t_t.cpu().numpy())
        self.n_tasks_list.append(n_t)

        # Welford-Hart pooled update
        if self.n_pool == 0:
            self.mu_pool_t = mu_t_t.clone()
            self.Sigma_pool_t = Sigma_t_t.clone()
            self.n_pool = n_t
        else:
            mu_old = self.mu_pool_t
            cov_old = self.Sigma_pool_t
            self.mu_pool_t, self.Sigma_pool_t, self.n_pool = welford_pooled_update(
                mu_old, cov_old, self.n_pool, mu_t_t, Sigma_t_t, n_t)

        del mu_t_t, Sigma_t_t

        # Compute PaR
        self.par = participation_ratio_torch(self.Sigma_pool_t, device=self.device)
        par_ratio = self.par / d

        # Adaptive metric selection
        if par_ratio > self.par_threshold:
            self.current_metric = 'l2'
            self.Sinv = None
        else:
            self.current_metric = 'mahalanobis'
            # Compute pseudo-inverse on GPU
            eigvals, eigvecs = torch.linalg.eigh(self.Sigma_pool_t)
            eigvals = torch.clamp(eigvals, min=1e-6 * eigvals.abs().max().item())
            idx = torch.argsort(eigvals, descending=True)
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
            self.Sinv = eigvecs @ torch.diag(1.0 / eigvals) @ eigvecs.T

    def route(self, h_batch):
        if len(self.centroids) == 0:
            return np.zeros(h_batch.shape[0], dtype=np.int64)

        n_t = len(self.centroids)

        if self.current_metric == 'l2':
            C = np.stack(self.centroids)
            H_sq = np.sum(h_batch ** 2, axis=1, keepdims=True)
            C_sq = np.sum(C ** 2, axis=1)
            dists = H_sq + C_sq[None, :] - 2 * (h_batch @ C.T)
        else:
            # Mahalanobis on GPU
            dev = self.dev
            H = torch.from_numpy(h_batch.astype(np.float32)).to(dev)
            C = np.stack(self.centroids)
            n_sample = H.shape[0]
            dists = np.zeros((n_sample, n_t), dtype=np.float64)

            Sinv = self.Sinv
            for i, mu_t_np in enumerate(C):
                mu_t_t = torch.from_numpy(mu_t_np.astype(np.float32)).to(dev)
                diff = H - mu_t_t
                # d² = diff @ Sinv @ diff.T  (all torch tensors)
                diff_Sinv = diff @ Sinv
                dists[:, i] = (diff * diff_Sinv).sum(dim=1).cpu().numpy()

            del H

        return dists.argmin(axis=1)


class RLSRouter:
    """
    Recursive Least Squares router (Woodbury identity).
    Learns a linear projection to separate tasks.
    Included as a strong baseline from the literature.
    """
    def __init__(self, d_model, expansion_dim=2048, lam=0.1, seed=42, device="cuda"):
        self.d_model = d_model
        self.E = min(expansion_dim, d_model * 2)
        self.lam = lam
        self.device = device
        self.dev = torch.device(device)

        rng = np.random.RandomState(seed)
        self.W_phi_np = (rng.randn(d_model, self.E) / np.sqrt(d_model)).astype(np.float32)
        self.b_phi_np = (rng.randn(self.E) * 0.01).astype(np.float32)
        self.R_np = np.eye(self.E, dtype=np.float64) / lam
        self.Q_np = np.zeros((self.E, 0), dtype=np.float64)
        self.W_r_np = np.zeros((self.E, 0), dtype=np.float64)
        self.num_tasks = 0

    def _expand(self, X_np):
        return np.maximum(0, X_np @ self.W_phi_np + self.b_phi_np)

    def add_task(self, embs, task_name=None):
        H = self._expand(embs.astype(np.float64))
        N = H.shape[0]
        R = self.R_np.copy()
        chunk = 512
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
        self.R_np = R
        extra = np.zeros((self.E, 1), dtype=np.float64)
        extra[:, 0] = H.T @ np.ones(N)
        self.Q_np = np.hstack([self.Q_np, extra])
        self.W_r_np = self.R_np @ self.Q_np
        self.num_tasks += 1

    def route(self, h_batch):
        H = self._expand(h_batch.astype(np.float64))
        logits = H @ self.W_r_np
        return logits.argmax(axis=1)


class PSRRouter:
    """
    Probabilistic Subspace Routing (SRT Thm 6, anisotropic case).
    Uses eigendecomposition of individual task covariance.
    k principal components for subspace identification.
    """
    def __init__(self, k=8, device="cuda"):
        self.k = k
        self.device = device
        self.dev = torch.device(device)
        self.sigs = []

    def add_task(self, embs, task_name=None):
        n, d = embs.shape
        X = torch.from_numpy(embs.astype(np.float32)).to(self.dev)
        mu_t = X.mean(dim=0).cpu().numpy()
        Xc = X - mu_t
        cov_t = (Xc.T @ Xc) / max(n - 1, 1)
        del X, Xc

        # Eigendecomposition on GPU
        eigvals, eigvecs = torch.linalg.eigh(cov_t)
        eigvals = eigvals.cpu().numpy()
        eigvecs = eigvecs.cpu().numpy()
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        k_eff = min(self.k, d)
        V = eigvecs[:, :k_eff]
        lam = np.maximum(eigvals[:k_eff], 1e-12)
        sigma2 = max(eigvals[k_eff:].mean() if k_eff < d else 1e-12, 1e-12)
        self.sigs.append((mu_t, V, lam, sigma2, d))

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


# ══════════════════════════════════════════════════════════════════════════════
# INCREMENTAL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def run_incremental_eval(train_embs_dict, test_embs_dict, task_order, args):
    """
    CL evaluation loop.

    Protocol:
      1. Add task t to all routers (only train embeddings of task t seen).
      2. Evaluate routing accuracy on all test samples of tasks 0..t.

    Metrics: macro accuracy = mean(accuracy per seen task)
    """
    ordered_found = [t for t in task_order if t in train_embs_dict and t in test_embs_dict]
    n_tasks = len(ordered_found)
    d = next(iter(train_embs_dict.values())).shape[1]
    print(f"Tasks: {n_tasks}/{len(task_order)}, d={d}, device={DEVICE}")

    if n_tasks == 0:
        return {}

    # Router registry
    routers = OrderedDict()

    # ── No-whitening baselines ───────────────────────────────────────────────
    routers["NearestCentroid"] = NearestCentroidRouter(device=DEVICE)
    routers["CosineNearestCentroid"] = CosineNearestCentroidRouter(device=DEVICE)

    # ── Whitened routers (ZCA fit once from buffer, frozen) ───────────────────
    for min_s in [400, 800, 1600]:
        routers[f"WhitenedL2_min{min_s}"] = WhitenedNearestCentroidRouter(
            min_samples=min_s, device=DEVICE)
        routers[f"WhitenedCosine_min{min_s}"] = WhitenedCosineRouter(
            min_samples=min_s, device=DEVICE)

    # ── Adaptive router ───────────────────────────────────────────────────────
    routers["AdaptivePar_0.9"] = AdaptivePooledMahalanobisRouter(
        par_threshold=0.9, device=DEVICE)
    routers["AdaptivePar_0.7"] = AdaptivePooledMahalanobisRouter(
        par_threshold=0.7, device=DEVICE)
    routers["AdaptivePar_0.5"] = AdaptivePooledMahalanobisRouter(
        par_threshold=0.5, device=DEVICE)

    # ── PSR ──────────────────────────────────────────────────────────────────
    routers["PSR_k8"] = PSRRouter(k=8, device=DEVICE)

    # ── RLS ───────────────────────────────────────────────────────────────────
    routers["RLS_Woodbury"] = RLSRouter(
        d_model=d, expansion_dim=min(2048, d * 2), lam=0.1, device=DEVICE)

    all_results = {name: [] for name in routers}

    for t_idx, task_name in enumerate(ordered_found):
        embs_train = train_embs_dict[task_name]
        n_t, d_t = embs_train.shape
        n_pool = sum(train_embs_dict[t].shape[0] for t in ordered_found[:t_idx+1])
        print(f"\n  [{t_idx+1}/{n_tasks}] {task_name}  (n={n_t}, n/d={n_t/d_t:.4f}, pool={n_pool})")

        # Add task to all routers
        for name, router in routers.items():
            router.add_task(embs_train, task_name)

        # Log diagnostics
        for name, router in routers.items():
            if hasattr(router, 'zca_fitted'):
                zca_state = f"ZCA={'fit' if router.zca_fitted else 'wait'}"
            elif hasattr(router, 'n_pool') and router.n_pool > 0:
                par_str = f"PaR={router.par:.1f}/{d}" if hasattr(router, 'par') else ""
                zca_state = f"pool={router.n_pool} {par_str}"
            elif hasattr(router, 'num_tasks'):
                zca_state = f"tasks={router.num_tasks}"
            else:
                zca_state = ""
            print(f"    {name:30s}  {zca_state}")

        # Evaluate on seen tasks
        seen_tasks = ordered_found[:t_idx + 1]

        for router_name, router in routers.items():
            per_task_acc = []
            for j, seen_task in enumerate(seen_tasks):
                embs_test = test_embs_dict[seen_task]
                preds = router.route(embs_test)
                true_idx = j
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
            print(f"    RESULT {router_name:30s} macro={macro_acc*100:6.2f}%  [{row_str}]")

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CL Routing — Whitened L2 + Whitened Cosine + Adaptive + CUDA")
    parser.add_argument("--emb_dir", required=True)
    parser.add_argument("--benchmark", required=True, choices=["SuperNI", "Long_Sequence"])
    parser.add_argument("--out_dir", default="results_cl_v2")
    parser.add_argument("--task_order", type=str, default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    # Override device if CUDA not available
    global DEVICE, HAS_CUDA
    if args.device == "cuda" and not HAS_CUDA:
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        DEVICE = "cpu"
    elif args.device == "cpu":
        DEVICE = "cpu"

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    backbone = Path(args.emb_dir).name

    if args.task_order:
        task_order = [t.strip() for t in args.task_order.split(",") if t.strip()]
    else:
        task_order = BENCHMARK_ORDER.get(args.benchmark, [])

    tag = f"{backbone}_{args.benchmark}"
    out_path = out_dir / f"cl_routing_{tag}.json"

    if out_path.exists() and not args.force:
        print(f"[SKIP] {out_path} exists. Use --force.")
        return

    print(f"=== CL Routing Evaluation [{tag}] ===")
    print(f"    Backbone: {backbone}")
    print(f"    Benchmark: {args.benchmark}")
    print(f"    Tasks: {len(task_order)}")
    print(f"    Device: {DEVICE}")
    print(f"    NOTE: Uses ALL available train samples (no caps)")
    print()

    train_embs = load_all(args.emb_dir, args.benchmark, task_order, "train")
    test_embs = load_all(args.emb_dir, args.benchmark, task_order, "test")

    t0 = time.time()
    results = run_incremental_eval(train_embs, test_embs, task_order, args)
    elapsed = time.time() - t0

    # Summary table
    print(f"\n{'='*90}")
    print(f"  Final Routing Accuracy ({len(task_order)} tasks, {elapsed:.1f}s, {DEVICE})")
    print(f"{'='*90}")
    print(f"  {'Method':40s}  {'Final':>7s}  {'Avg':>7s}  Steps")
    print(f"  {'-'*108}")

    final_report = {}
    for name, steps in results.items():
        if not steps:
            continue
        final = steps[-1]
        avg = sum(s["accuracy"] for s in steps) / len(steps)
        step_str = "  ".join([f"T{i+1}:{s['accuracy']*100:.0f}%" for i, s in enumerate(steps)])
        print(f"  {name:40s}  {final['accuracy']*100:6.2f}%  {avg*100:6.2f}%  {step_str}")
        final_report[name] = {
            "final_accuracy": float(final["accuracy"]),
            "avg_accuracy": float(avg),
            "step_accuracies": [float(s["accuracy"]) for s in steps],
        }

    report = {
        "backbone": backbone,
        "benchmark": args.benchmark,
        "n_tasks": len(task_order),
        "device": DEVICE,
        "elapsed_seconds": float(elapsed),
        "routers": {
            "WhitenedL2_minN": "ZCA fit once when pool≥N, frozen. L2 in whitened space.",
            "WhitenedCosine_minN": "ZCA fit once when pool≥N, frozen. Cosine in whitened space.",
            "AdaptivePar_θ": "SRT Thm6: PaR metric selection. PaR/d>θ→L2 else→Mahalanobis.",
            "NearestCentroid": "Baseline: raw L2.",
            "CosineNearestCentroid": "Baseline: raw cosine.",
            "PSR_k8": "Probabilistic Subspace Routing.",
            "RLS_Woodbury": "Recursive Least Squares.",
        },
        "results": final_report,
    }

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n✓ Saved {out_path}")


if __name__ == "__main__":
    main()