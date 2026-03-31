#!/usr/bin/env python3
"""
Phase F — Learned Routing Simulators (GPM / RLS / Baselines).

Simulates ROOT/GainLoRA's learned GPM routing and SpecRoute's RLS routing
on pre-extracted embeddings, incrementally (task-by-task).

Architecture reproduced from ROOT source:
  - trans_input MLP:  Linear(d→h, no bias) → SiLU → Linear(h→d, no bias) → SiLU
  - prompt_key:       (1, d) per task, init from top eigvec of output covariance
  - cal_attention:    |sigmoid(4 * cos_sim) * 2 - 1|
  - GPM protection:   SVD-based subspace extraction + null-space projection

Usage:
  python simulate_gpm_routing.py \\
    --emb_dir embeddings/T5EncoderModel \\
    --benchmark Long_Sequence \\
    --mlp_hidden_dim 100 \\
    --transthreshold 0.995

Hyperparameters matching ROOT defaults:
  --mlp_hidden_dim  100      (from gen_script: --mlp_hidden_dim 100)
  --transthreshold  0.995    (from gen_script: --transthreshold 0.995)
  --chunk           1        (number of chunks for GPM; 1 = full dimension)
  --lr              1e-3     (proxy training LR)
  --epochs          30       (proxy training epochs per task)
"""
from __future__ import annotations
import argparse, json, math, sys, warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np

# ─── Try torch; graceful fallback message ───
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ═══════════════════════════════════════════════════════════════════════
# Benchmark definitions (shared across experiment scripts)
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_split(emb_dir, benchmark, task, split):
    p = Path(emb_dir) / benchmark / task / f"{split}.npz"
    if not p.exists():
        return None
    data = np.load(str(p), allow_pickle=True)
    return data["embeddings"].astype(np.float32)


def load_all(emb_dir, benchmark, tasks, split):
    out = OrderedDict()
    for t in tasks:
        embs = load_split(emb_dir, benchmark, t, split)
        if embs is not None:
            out[t] = embs
    return out


# ═══════════════════════════════════════════════════════════════════════
# 1. GPM Routing (ROOT/GainLoRA faithful reproduction)
# ═══════════════════════════════════════════════════════════════════════

class TransInputMLP(nn.Module):
    """Single-task trans_input: Linear(d→h) → SiLU → Linear(h→d) → SiLU.

    Matches ROOT's `nn.Sequential(Linear, SiLU, Linear, SiLU)` with bias=False.
    """
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.linear2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        """x: (B, d) → (B, d)"""
        return self.act(self.linear2(self.act(self.linear1(x))))


class FrozenTransInput(nn.Module):
    """Multi-task frozen trans_input — stores (n_tasks, h, d) and (n_tasks, d, h) weights.

    Matches ROOT's `Trans_input` class with batched matmul.
    """
    def __init__(self, input_weights, output_weights):
        """
        input_weights:  list of (h, d) tensors, one per frozen task
        output_weights: list of (d, h) tensors
        """
        super().__init__()
        # Stack into (n_tasks, h, d) and (n_tasks, d, h)
        self.register_buffer('W_in', torch.stack(input_weights, dim=0))    # (T, h, d)
        self.register_buffer('W_out', torch.stack(output_weights, dim=0))  # (T, d, h)
        self.act = nn.SiLU()

    def forward(self, x):
        """x: (B, d) → (B, T, d)"""
        # x: (B, d) → (B, 1, 1, d)
        x = x.unsqueeze(1).unsqueeze(2)           # (B, 1, 1, d)
        mid = torch.matmul(x, self.W_in.permute(0, 2, 1))  # (B, T, 1, h)
        mid = self.act(mid)
        out = torch.matmul(mid, self.W_out.permute(0, 2, 1))  # (B, T, 1, d)
        out = self.act(out)
        return out.squeeze(2)  # (B, T, d)


def cal_attention(prompt_keys, x_features):
    """ROOT's cal_attention: |sigmoid(4 * cos_sim) * 2 - 1|.

    Args:
        prompt_keys: (B, T, d) or (T, d) — prompt keys per task
        x_features:  (B, T, d) — transformed features per task
    Returns:
        weights: (B, T) — non-negative routing weights
    """
    if prompt_keys.dim() == 2:
        prompt_keys = prompt_keys.unsqueeze(0).expand(x_features.shape[0], -1, -1)
    # L2 normalize
    x_norm = x_features / (x_features.norm(dim=-1, keepdim=True) + 1e-12)
    pk_norm = prompt_keys / (prompt_keys.norm(dim=-1, keepdim=True) + 1e-12)
    # Cosine similarity per task
    cos_sim = (x_norm * pk_norm).sum(dim=-1)  # (B, T)
    weights = torch.abs(torch.sigmoid(cos_sim * 4) * 2 - 1)
    return weights


def gpm_svd_threshold(cov_matrix, threshold):
    """Extract GPM bases from covariance matrix via SVD + cumulative energy threshold.

    Returns U[:, :r] where r is smallest such that cumsum(S²)/sum(S²) >= threshold.
    Returns None if r=0 (skip update).
    """
    U, S, _ = np.linalg.svd(cov_matrix, full_matrices=False)
    sval_sq = S ** 2
    sval_total = sval_sq.sum()
    if sval_total < 1e-12:
        return None
    sval_ratio = sval_sq / sval_total
    cumsum = np.cumsum(sval_ratio)
    r = int(np.searchsorted(cumsum, threshold)) + 1
    r = min(r, len(S))
    if r == 0:
        return None
    return U[:, :r]


def gpm_update_bases(old_bases, new_cov, threshold):
    """Update GPM bases incrementally (Eq-8, Eq-9 from GPM paper).

    For task t>1:
    1. Project new covariance into null-space of existing bases
    2. SVD projected covariance
    3. Append new bases if variance threshold not met

    Args:
        old_bases: (d, r_old) or None
        new_cov:   (d, d) covariance matrix for new task
        threshold: cumulative energy threshold
    Returns:
        updated_bases: (d, r_new) or None
    """
    if old_bases is None:
        # First task: just extract from full covariance
        return gpm_svd_threshold(new_cov, threshold)

    d = new_cov.shape[0]
    # Projection matrix P_old = U @ U^T
    P_old = old_bases @ old_bases.T  # (d, d)

    # Full SVD for threshold computation
    U_full, S_full, _ = np.linalg.svd(new_cov, full_matrices=False)
    sval_total = (S_full ** 2).sum()

    # Project into null-space: C_hat = (I - P_old) @ C @ (I - P_old)
    Q = np.eye(d) - P_old
    C_hat = Q @ new_cov  # simplified: project rows only (matches ROOT source)

    U_hat, S_hat, _ = np.linalg.svd(C_hat, full_matrices=False)
    sval_hat = (S_hat ** 2).sum()

    # Check how much variance is already explained
    accumulated = (sval_total - sval_hat) / max(sval_total, 1e-12)

    r = 0
    sval_ratio = (S_hat ** 2) / max(sval_total, 1e-12)
    for ii in range(len(sval_ratio)):
        if accumulated < threshold:
            accumulated += sval_ratio[ii]
            r += 1
        else:
            break

    if r == 0:
        # Skip update — existing bases sufficient
        return old_bases

    # Concatenate old + new bases
    new_dirs = U_hat[:, :r]
    updated = np.hstack([old_bases, new_dirs])
    # Clip if exceeds dimension
    if updated.shape[1] > d:
        updated = updated[:, :d]
    return updated


def init_prompt_key_from_cov(output_cov, d_model, old_routing_bases=None):
    """Initialize prompt_key from output covariance (ROOT's method).

    Task 1: top eigenvector of output covariance.
    Task t>1: top eigenvector of random matrix projected into null-space.
    """
    if old_routing_bases is None:
        # Task 1: data-informed init
        U, S, _ = np.linalg.svd(output_cov, full_matrices=False)
        pk = U[:, 0]  # top eigenvector
    else:
        # Task t>1: null-space init
        d = output_cov.shape[0]
        P_old = old_routing_bases @ old_routing_bases.T
        R = np.random.randn(d, d).astype(np.float32)
        R_proj = R - P_old @ R
        U, S, _ = np.linalg.svd(R_proj, full_matrices=False)
        pk = U[:, 0]

    # Normalize (ROOT divides by sqrt(chunk) then scales to pre_norm)
    pk = pk / (np.linalg.norm(pk) + 1e-12)
    return pk.astype(np.float32)


class GPMRouter:
    """Full GPM routing simulator, faithful to ROOT/GainLoRA.

    Incrementally learns trans_input MLP + prompt_key per task,
    then applies GPM protection between tasks.

    Parameters:
        d_model:         embedding dimension
        mlp_hidden_dim:  trans_input MLP hidden dim (ROOT default: 100)
        transthreshold:  GPM energy threshold (ROOT default: 0.995)
        lr:              proxy training learning rate
        epochs:          proxy training epochs per task
        device:          'cpu' or 'cuda'
    """

    def __init__(self, d_model, mlp_hidden_dim=100, transthreshold=0.995,
                 lr=1e-3, epochs=30, batch_size=256, device='cpu'):
        self.d_model = d_model
        self.mlp_hidden_dim = mlp_hidden_dim
        self.transthreshold = transthreshold
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

        # State accumulated across tasks
        self.prompt_keys = []              # list of np.ndarray (d,)
        self.frozen_W_in = []              # list of np.ndarray (h, d)
        self.frozen_W_out = []             # list of np.ndarray (d, h)

        # GPM bases for 3 stages of trans_input:
        # stage 1 = input covariance, stage 2 = medium, stage 3 = output
        self.gpm_bases_input = None        # (d, r) or None
        self.gpm_bases_medium = None       # (h, r) or None
        self.gpm_bases_output = None       # (d, r) or None

    def _adaptive_threshold(self, task_idx, total_tasks=15):
        """Linearly interpolate threshold like ROOT:
        threshold_t = (1 - base) * t / T + base
        """
        return (1.0 - self.transthreshold) * task_idx / total_tasks + self.transthreshold

    def add_task(self, task_embs, task_idx, total_tasks=15):
        """Learn routing for a new task and apply GPM protection.

        Args:
            task_embs: np.ndarray (N, d) — training embeddings for this task
            task_idx:  0-indexed task number
            total_tasks: total expected tasks (for threshold scheduling)
        """
        d = self.d_model
        h = self.mlp_hidden_dim
        threshold = self._adaptive_threshold(task_idx, total_tasks)

        # ── Step 1: Initialize trans_input MLP ──
        mlp = TransInputMLP(d, h).to(self.device)

        # If task > 0: project initial MLP weights into GPM null-space
        if task_idx > 0:
            with torch.no_grad():
                if self.gpm_bases_input is not None:
                    P = torch.from_numpy(
                        self.gpm_bases_input @ self.gpm_bases_input.T
                    ).float().to(self.device)
                    # Project linear1 weight rows into null-space
                    mlp.linear1.weight.data -= mlp.linear1.weight.data @ P

                if self.gpm_bases_output is not None:
                    P = torch.from_numpy(
                        self.gpm_bases_output @ self.gpm_bases_output.T
                    ).float().to(self.device)
                    # linear2.weight has shape (d, h); P is (d, d)
                    # We must project the OUTPUT dimension: P @ W (d,d) @ (d,h) -> (d,h)
                    mlp.linear2.weight.data -= P @ mlp.linear2.weight.data

        # ── Step 2: Initialize prompt_key ──
        # Compute output covariance from MLP applied to task embeddings
        X_t = torch.from_numpy(task_embs).float().to(self.device)
        with torch.no_grad():
            out_features = mlp(X_t).cpu().numpy()
        out_cov = out_features.T @ out_features / max(out_features.shape[0], 1)

        pk_np = init_prompt_key_from_cov(out_cov, d, self.gpm_bases_output)
        prompt_key = nn.Parameter(torch.from_numpy(pk_np).float().to(self.device))

        # ── Step 3: Train trans_input + prompt_key via routing proxy loss ──
        if task_idx > 0:
            self._train_routing(mlp, prompt_key, task_embs, task_idx)

        # ── Step 4: Collect covariance and update GPM ──
        X_t = torch.from_numpy(task_embs).float().to(self.device)
        with torch.no_grad():
            medium = mlp.act(mlp.linear1(X_t)).cpu().numpy()   # (N, h)
            output = mlp(X_t).cpu().numpy()                    # (N, d)
            inp = task_embs                                     # (N, d)

        # Covariance matrices
        cov_input = inp.T @ inp / max(inp.shape[0], 1)
        cov_medium = medium.T @ medium / max(medium.shape[0], 1)
        cov_output = output.T @ output / max(output.shape[0], 1)

        self.gpm_bases_input = gpm_update_bases(
            self.gpm_bases_input, cov_input, threshold)
        self.gpm_bases_medium = gpm_update_bases(
            self.gpm_bases_medium, cov_medium, threshold)
        self.gpm_bases_output = gpm_update_bases(
            self.gpm_bases_output, cov_output, threshold)

        # ── Step 5: Snapshot ──
        self.prompt_keys.append(prompt_key.detach().cpu().numpy())
        self.frozen_W_in.append(mlp.linear1.weight.detach().cpu().numpy())   # (h, d)
        self.frozen_W_out.append(mlp.linear2.weight.detach().cpu().numpy())  # (d, h)

    def _train_routing(self, mlp, prompt_key, task_embs, task_idx):
        """Train current trans_input + prompt_key to correctly route.

        Proxy objective: cross-entropy over [current_task, prev_tasks].
        Current task's features come from trainable mlp + prompt_key.
        Previous tasks' features come from frozen snapshots.
        """
        n_prev = len(self.prompt_keys)
        n_total = n_prev + 1  # current is index 0

        # Build frozen components
        frozen_pks = torch.tensor(
            np.stack(self.prompt_keys), dtype=torch.float32, device=self.device
        )  # (n_prev, d)
        frozen_trans = FrozenTransInput(
            [torch.tensor(w, dtype=torch.float32) for w in self.frozen_W_in],
            [torch.tensor(w, dtype=torch.float32) for w in self.frozen_W_out],
        ).to(self.device)
        frozen_trans.eval()

        optimizer = torch.optim.Adam(
            list(mlp.parameters()) + [prompt_key], lr=self.lr
        )

        X = torch.from_numpy(task_embs).float().to(self.device)
        N = X.shape[0]
        # Target: current task = index 0 (highest weight)
        target = torch.zeros(N, dtype=torch.long, device=self.device)

        for epoch in range(self.epochs):
            perm = torch.randperm(N, device=self.device)
            total_loss = 0.0
            for start in range(0, N, self.batch_size):
                idx = perm[start:start + self.batch_size]
                x_batch = X[idx]

                # Current task features
                x_cur = mlp(x_batch)  # (B, d)

                # Previous tasks' features
                with torch.no_grad():
                    x_prev = frozen_trans(x_batch)  # (B, n_prev, d)

                # Stack: (B, n_total, d) — current first, then previous
                all_x = torch.cat([x_cur.unsqueeze(1), x_prev], dim=1)

                # Stack prompt keys: current first
                all_pk = torch.cat([prompt_key.unsqueeze(0), frozen_pks], dim=0)
                all_pk = all_pk.unsqueeze(0).expand(x_batch.shape[0], -1, -1)

                # Compute routing weights via cal_attention
                weights = cal_attention(all_pk, all_x)  # (B, n_total)

                # Cross-entropy loss: want index 0 to be max
                loss = F.cross_entropy(weights * 10, target[idx[:x_batch.shape[0]]])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

    def route(self, h_batch):
        """Route a batch of embeddings to task indices.

        Args:
            h_batch: np.ndarray (B, d)
        Returns:
            predicted task indices: np.ndarray (B,) — 0-indexed into self.prompt_keys
        """
        n_tasks = len(self.prompt_keys)
        if n_tasks == 0:
            raise ValueError("No tasks added yet")
        if n_tasks == 1:
            return np.zeros(h_batch.shape[0], dtype=np.int64)

        X = torch.from_numpy(h_batch).float().to(self.device)

        # Build all prompt keys
        all_pk = torch.tensor(
            np.stack(self.prompt_keys), dtype=torch.float32, device=self.device
        )  # (T, d)

        # Build all trans_input features
        frozen_trans = FrozenTransInput(
            [torch.tensor(w, dtype=torch.float32) for w in self.frozen_W_in],
            [torch.tensor(w, dtype=torch.float32) for w in self.frozen_W_out],
        ).to(self.device)
        frozen_trans.eval()

        with torch.no_grad():
            all_x = frozen_trans(X)  # (B, T, d)
            weights = cal_attention(all_pk, all_x)  # (B, T)

        return weights.argmax(dim=1).cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════
# 2. RLS Routing (SpecRoute — Woodbury incremental)
# ═══════════════════════════════════════════════════════════════════════

class RLSRouter:
    """Offline RLS router matching SpecRoute's Woodbury implementation.

    Random feature expansion + incremental analytical ridge regression.
    """
    def __init__(self, d_model, expansion_dim=2048, lam=0.1, seed=42):
        self.d_model = d_model
        self.E = expansion_dim
        self.lam = lam

        rng = np.random.RandomState(seed)
        self.W_phi = rng.randn(d_model, expansion_dim).astype(np.float32) / np.sqrt(d_model)
        self.b_phi = rng.randn(expansion_dim).astype(np.float32) * 0.01

        self.R = np.eye(expansion_dim, dtype=np.float64) / lam
        self.Q = np.zeros((expansion_dim, 0), dtype=np.float64)
        self.W_r = np.zeros((expansion_dim, 0), dtype=np.float64)
        self.num_tasks = 0

    def _expand(self, h):
        """h: (N, d) → (N, E) via frozen random features."""
        return np.maximum(0, h @ self.W_phi + self.b_phi)  # ReLU

    def add_task(self, task_embs):
        """Woodbury update for new task."""
        H = self._expand(task_embs).astype(np.float64)
        N = H.shape[0]

        # Woodbury: R_new = R - R H^T (I + H R H^T)^{-1} H R
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

        # Extend Q
        tid = self.num_tasks
        extra = np.zeros((self.E, 1), dtype=np.float64)
        extra[:, 0] = H.T @ np.ones(N)
        self.Q = np.hstack([self.Q, extra])
        self.W_r = self.R @ self.Q
        self.num_tasks += 1

    def route(self, h_batch):
        """h_batch: (N, d) → predicted task indices (N,)."""
        H = self._expand(h_batch).astype(np.float64)
        logits = H @ self.W_r
        return logits.argmax(axis=1)


# ═══════════════════════════════════════════════════════════════════════
# 3. Non-parametric baselines
# ═══════════════════════════════════════════════════════════════════════

class NearestCentroidRouter:
    """L2 nearest centroid (simplest possible baseline)."""
    def __init__(self):
        self.centroids = []

    def add_task(self, embs):
        self.centroids.append(embs.mean(0))

    def route(self, h_batch):
        C = np.stack(self.centroids)  # (T, d)
        dists = np.linalg.norm(h_batch[:, None, :] - C[None, :, :], axis=2)
        return dists.argmin(axis=1)


class CosineNearestCentroidRouter:
    """Cosine nearest centroid."""
    def __init__(self):
        self.centroids = []

    def add_task(self, embs):
        mu = embs.mean(0)
        self.centroids.append(mu / (np.linalg.norm(mu) + 1e-12))

    def route(self, h_batch):
        C = np.stack(self.centroids)
        h_norm = h_batch / (np.linalg.norm(h_batch, axis=1, keepdims=True) + 1e-12)
        sims = h_norm @ C.T  # (N, T)
        return sims.argmax(axis=1)


class PSRRouter:
    """PPCA-based PSR routing (non-parametric, from analyze_geometry)."""
    def __init__(self, k=8):
        self.k = k
        self.sigs = []

    def add_task(self, embs):
        n, d = embs.shape
        mu = embs.mean(0)
        cov = np.cov(embs, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        k = min(self.k, d)
        V = eigvecs[:, :k]
        lam = np.maximum(eigvals[:k], 1e-12)
        sigma2 = max(eigvals[k:].mean() if k < d else 1e-12, 1e-12)
        self.sigs.append((mu, V, lam, sigma2, d))

    def _psr_dist(self, h, sig):
        mu, V, lam, s2, d = sig
        delta = h - mu
        proj = V.T @ delta
        weights = lam / (s2 * (lam + s2))
        dist = np.sum(weights * proj**2)
        dist += np.sum(delta**2) / s2
        dist += np.sum(np.log(lam + s2)) + (d - len(lam)) * np.log(s2)
        return dist

    def route(self, h_batch):
        preds = []
        for h in h_batch:
            dists = [self._psr_dist(h, sig) for sig in self.sigs]
            preds.append(np.argmin(dists))
        return np.array(preds, dtype=np.int64)


# ═══════════════════════════════════════════════════════════════════════
# Incremental Evaluation
# ═══════════════════════════════════════════════════════════════════════

def run_incremental_comparison(train_embs, test_embs, tasks, args):
    """Run all routing methods incrementally, task by task.

    Returns dict: method → list of {step, n_tasks, accuracy, per_task}.
    """
    d = next(iter(train_embs.values())).shape[1]

    # Initialize routers
    routers = OrderedDict()
    routers["NearestCentroid"] = NearestCentroidRouter()
    routers["CosineNearestCentroid"] = CosineNearestCentroidRouter()
    routers["PSR"] = PSRRouter(k=args.subspace_k)

    routers["RLS_Woodbury"] = RLSRouter(
        d, expansion_dim=args.rls_expansion, lam=args.rls_lambda)

    if HAS_TORCH:
        routers["GPM_ROOT"] = GPMRouter(
            d, mlp_hidden_dim=args.mlp_hidden_dim,
            transthreshold=args.transthreshold,
            lr=args.lr, epochs=args.epochs,
            batch_size=args.batch_size, device=args.device)
    else:
        print("WARNING: torch not available, skipping GPM_ROOT routing.")

    total_tasks = len(tasks)
    all_results = {name: [] for name in routers}

    for t_idx, task_name in enumerate(tasks):
        if task_name not in train_embs:
            continue

        print(f"\n  [{t_idx+1}/{total_tasks}] Adding task: {task_name}")

        # Add task to all routers
        for name, router in routers.items():
            if name == "GPM_ROOT":
                router.add_task(train_embs[task_name], t_idx, total_tasks)
            elif name == "RLS_Woodbury":
                router.add_task(train_embs[task_name])
            else:
                router.add_task(train_embs[task_name])

        # Evaluate on all seen tasks so far
        seen_tasks = [tasks[i] for i in range(t_idx + 1) if tasks[i] in test_embs]
        if not seen_tasks:
            continue

        task2idx = {t: i for i, t in enumerate(
            [tasks[j] for j in range(t_idx + 1) if tasks[j] in train_embs]
        )}

        for name, router in routers.items():
            correct, total = 0, 0
            per_task = {}
            for true_task in seen_tasks:
                if true_task not in test_embs or true_task not in task2idx:
                    continue
                embs_test = test_embs[true_task]
                preds = router.route(embs_test)
                true_idx = task2idx[true_task]
                c = int((preds == true_idx).sum())
                per_task[true_task] = c / max(embs_test.shape[0], 1)
                correct += c
                total += embs_test.shape[0]

            acc = correct / max(total, 1)
            all_results[name].append({
                "step": t_idx + 1,
                "n_tasks": len(task2idx),
                "accuracy": acc,
                "per_task": per_task,
            })
            print(f"    {name:25s}  acc={acc*100:.2f}% ({len(task2idx)} tasks)")

    return all_results


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Phase F — Learned Routing Comparison (GPM vs RLS vs baselines)")
    parser.add_argument("--emb_dir", required=True,
                        help="Path to backbone embedding dir")
    parser.add_argument("--benchmark", required=True, choices=list(BENCHMARKS.keys()))
    parser.add_argument("--subspace_k", type=int, default=8,
                        help="PSR subspace rank")
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--whiten", action="store_true",
                        help="Apply global ZCA whitening")

    # GPM (ROOT) parameters
    gpm = parser.add_argument_group("GPM/ROOT routing parameters")
    gpm.add_argument("--mlp_hidden_dim", type=int, default=100,
                     help="trans_input MLP hidden dim (ROOT default: 100)")
    gpm.add_argument("--transthreshold", type=float, default=0.995,
                     help="GPM energy threshold for routing subspace (ROOT default: 0.995)")
    gpm.add_argument("--lr", type=float, default=1e-3,
                     help="Proxy training learning rate")
    gpm.add_argument("--epochs", type=int, default=30,
                     help="Proxy training epochs per task")
    gpm.add_argument("--batch_size", type=int, default=256,
                     help="Training batch size")
    gpm.add_argument("--device", default="cpu",
                     help="Device for GPM training (cpu/cuda)")

    # RLS parameters
    rls = parser.add_argument_group("RLS routing parameters")
    rls.add_argument("--rls_expansion", type=int, default=2048,
                     help="Random feature expansion dim (SpecRoute default: 2048)")
    rls.add_argument("--rls_lambda", type=float, default=0.1,
                     help="Ridge regularization (SpecRoute default: 0.1)")

    args = parser.parse_args()

    tasks = BENCHMARKS[args.benchmark]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    backbone = Path(args.emb_dir).name
    tag = f"{backbone}_{args.benchmark}" + ("_whitened" if args.whiten else "")

    print(f"=== Phase F: Learned Routing Comparison  [{tag}] ===")
    print(f"    GPM: mlp_hidden={args.mlp_hidden_dim}, transthreshold={args.transthreshold}, "
          f"lr={args.lr}, epochs={args.epochs}")
    print(f"    RLS: expansion={args.rls_expansion}, lambda={args.rls_lambda}")
    print(f"    PSR: k={args.subspace_k}\n")

    train_embs = load_all(args.emb_dir, args.benchmark, tasks, "train")
    test_embs = load_all(args.emb_dir, args.benchmark, tasks, "test")
    found = sorted(set(train_embs) & set(test_embs))
    if not found:
        print("ERROR: No tasks found."); sys.exit(1)
    print(f"Tasks found: {len(found)}/{len(tasks)}")

    train_embs = OrderedDict((t, train_embs[t]) for t in found)
    test_embs = OrderedDict((t, test_embs[t]) for t in found)

    if args.whiten:
        from compare_routing import compute_whitening, apply_whitening
        mu_g, W = compute_whitening(train_embs)
        train_embs = apply_whitening(train_embs, mu_g, W)
        test_embs = apply_whitening(test_embs, mu_g, W)
        # Convert back to float32
        train_embs = OrderedDict((t, e.astype(np.float32)) for t, e in train_embs.items())
        test_embs = OrderedDict((t, e.astype(np.float32)) for t, e in test_embs.items())
        print("Applied ZCA whitening\n")

    results = run_incremental_comparison(train_embs, test_embs, found, args)

    # ── Summary table ──
    print(f"\n{'='*70}")
    print(f"  Final Routing Accuracy (all {len(found)} tasks)")
    print(f"{'='*70}")
    print(f"  {'Method':25s}  {'Accuracy':>10s}")
    print(f"  {'-'*37}")
    final_accs = {}
    for name, steps in results.items():
        if steps:
            final = steps[-1]
            final_accs[name] = final["accuracy"]
            print(f"  {name:25s}  {final['accuracy']*100:>8.2f}%")

    # ── Save ──
    report = {
        "backbone": backbone, "benchmark": args.benchmark,
        "tasks": found, "d_model": next(iter(train_embs.values())).shape[1],
        "params": {
            "mlp_hidden_dim": args.mlp_hidden_dim,
            "transthreshold": args.transthreshold,
            "lr": args.lr, "epochs": args.epochs,
            "rls_expansion": args.rls_expansion,
            "rls_lambda": args.rls_lambda,
            "subspace_k": args.subspace_k,
        },
        "results": {name: steps for name, steps in results.items()},
        "final_accuracy": final_accs,
    }
    out_path = out_dir / f"learned_routing_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n✓ Saved {out_path}")


if __name__ == "__main__":
    main()
