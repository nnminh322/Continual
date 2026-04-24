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


def _resolve_device(device_str: str) -> str:
    """Resolve 'auto' → 'cuda' if GPU available, else 'cpu'.

    Executes a tiny kernel to catch arch-mismatch errors
    (cudaErrorNoKernelImageForDevice) before committing to CUDA.
    Falls back to CPU on any failure.
    """
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
            except Exception as e:
                print(f"[GPU] CUDA reported available but kernel launch failed "
                      f"({type(e).__name__}: {e}) — falling back to CPU")
                return "cpu"
        print("[GPU] No CUDA device found — using CPU")
        return "cpu"
    return device_str
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
#
# Verified against root_gainlora/src/:
#   T5:    t5_gainlora_inflora.py   — SiLU after each Linear
#   Llama: llama_gainlora_inflora.py — SiLU only at the end
#   Trainer: cl_trainer_gainlora_inflora.py — GPM SVD, gradient projection
#   Scripts: gen_script_*  — actual hyperparameters
# ═══════════════════════════════════════════════════════════════════════

class TransInputMLP(nn.Module):
    """Single-task trans_input MLP (bias=False).

    T5 (ROOT t5_gainlora_inflora.py L1051):
        Linear(d→h) → SiLU → Linear(h→d) → SiLU
    Llama (ROOT llama_gainlora_inflora.py L743):
        Linear(d→h) → Linear(h→d) → SiLU   (one SiLU at end)

    Init: kaiming_uniform_(a=√3) matching ROOT's Trans_input class.
    """
    def __init__(self, d_model, hidden_dim, backbone='t5'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.linear2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.act = nn.SiLU()
        self.backbone = backbone
        # ROOT Trans_input: kaiming_uniform_(a=√3)
        nn.init.kaiming_uniform_(self.linear1.weight, a=math.sqrt(3))
        nn.init.kaiming_uniform_(self.linear2.weight, a=math.sqrt(3))

    def forward(self, x):
        """x: (B, d) → (B, d)"""
        if self.backbone == 'llama':
            return self.act(self.linear2(self.linear1(x)))
        return self.act(self.linear2(self.act(self.linear1(x))))

    def forward_medium(self, x):
        """Return intermediate hidden features for GPM covariance."""
        if self.backbone == 'llama':
            return self.linear1(x)                 # no activation
        return self.act(self.linear1(x))           # SiLU


class FrozenTransInput(nn.Module):
    """Multi-task frozen trans_input (vectorized via batched einsum).

    Backbone-aware: T5 has SiLU after each matmul, Llama only at end.
    """
    def __init__(self, input_weights, output_weights, backbone='t5'):
        super().__init__()
        self.register_buffer('W_in', torch.stack(input_weights, dim=0))    # (T, h, d)
        self.register_buffer('W_out', torch.stack(output_weights, dim=0))  # (T, d, h)
        self.act = nn.SiLU()
        self.backbone = backbone

    def forward(self, x):
        """x: (B, d) → (B, T, d)  —  fully vectorized, no Python loop."""
        # mid = x @ W_in^T for all T tasks simultaneously
        mid = torch.einsum('bd,thd->bth', x, self.W_in)    # (B, T, h)
        if self.backbone != 'llama':
            mid = self.act(mid)                              # T5: SiLU after first linear
        out = self.act(torch.einsum('bth,tdh->btd', mid, self.W_out))  # (B, T, d)
        return out


def cal_attention(prompt_keys, x_features):
    """ROOT's cal_attention: |sigmoid(4 * cos_sim) * 2 - 1|.

    Matches ROOT t5_gainlora_inflora.py L1188-L1217.
    """
    if prompt_keys.dim() == 2:
        prompt_keys = prompt_keys.unsqueeze(0).expand(x_features.shape[0], -1, -1)
    x_norm = x_features / (x_features.norm(dim=-1, keepdim=True) + 1e-12)
    pk_norm = prompt_keys / (prompt_keys.norm(dim=-1, keepdim=True) + 1e-12)
    cos_sim = (x_norm * pk_norm).sum(dim=-1)  # (B, T)
    return torch.abs(torch.sigmoid(cos_sim * 4) * 2 - 1)


# ── GPU-accelerated GPM functions (torch.linalg.svd) ────────────────

def _gpm_svd_extract(cov, threshold):
    """First-task GPM basis extraction on GPU.

    Args:
        cov: (d, d) torch tensor on device
        threshold: cumulative energy threshold
    Returns:
        (d, r) basis tensor or None
    """
    U, S, _ = torch.linalg.svd(cov, full_matrices=False)
    sval_sq = S ** 2
    sval_total = sval_sq.sum()
    if sval_total.item() < 1e-12:
        return None
    cumsum = torch.cumsum(sval_sq / sval_total, dim=0)
    r = int((cumsum < threshold).sum().item()) + 1
    r = min(max(r, 1), S.shape[0])
    return U[:, :r]


def _gpm_update(old_bases, new_cov, threshold):
    """Incremental GPM base update on GPU.

    Matches ROOT cl_trainer_gainlora_inflora.py L406-L470 (Eq-8, Eq-9).
    """
    if old_bases is None:
        return _gpm_svd_extract(new_cov, threshold)

    # Full SVD for total variance
    _, S_full, _ = torch.linalg.svd(new_cov, full_matrices=False)
    sval_total = (S_full ** 2).sum()

    # Project into null-space (Eq-8): act_hat = cov - U U^T cov
    act_hat = new_cov - old_bases @ (old_bases.T @ new_cov)

    U_hat, S_hat, _ = torch.linalg.svd(act_hat, full_matrices=False)
    sval_hat = (S_hat ** 2).sum()

    # Greedy accumulation (Eq-9)
    accumulated = (sval_total - sval_hat) / max(sval_total.item(), 1e-12)
    sval_ratio = (S_hat ** 2) / max(sval_total.item(), 1e-12)

    r = 0
    for ii in range(sval_ratio.shape[0]):
        if accumulated < threshold:
            accumulated += sval_ratio[ii].item()
            r += 1
        else:
            break

    if r == 0:
        return old_bases

    updated = torch.cat([old_bases, U_hat[:, :r]], dim=1)
    d = updated.shape[0]
    if updated.shape[1] > d:
        updated = updated[:, :d]
    return updated


class GPMRouter:
    """Full GPM routing simulator, 100% faithful to ROOT/GainLoRA.

    Verified against root_gainlora/src/ for:
      - Backbone-specific architecture (T5 vs Llama)
      - Kaiming init (a=√3)
      - Per-chunk GPM bases (chunk=1 for T5, chunk=4 for Llama)
      - Gradient projection after every optimizer step with norm preservation
      - Prompt_key null-space initialization per chunk
      - All SVD/matrix ops on GPU via torch.linalg

    Hyperparameters from ROOT gen_scripts:
      T5:    mlp_hidden_dim=100, chunk=1,  transthreshold=0.995
      Llama: mlp_hidden_dim=50,  chunk=4,  transthreshold=0.995
    """

    def __init__(self, d_model, mlp_hidden_dim=100, transthreshold=0.995,
                 lr=1e-3, epochs=30, batch_size=256,
                 chunk=1, backbone='t5', device='cpu'):
        self.d_model = d_model
        self.mlp_hidden_dim = mlp_hidden_dim
        self.transthreshold = transthreshold
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.chunk = chunk
        self.step = d_model // chunk   # dim per chunk
        self.backbone = backbone
        self.device = device

        # Frozen snapshots (CPU numpy for storage, loaded to GPU on demand)
        self.prompt_keys = []              # list of np.ndarray (d,)
        self.frozen_W_in = []              # list of np.ndarray (h, d)
        self.frozen_W_out = []             # list of np.ndarray (d, h)

        # Cached GPU tensors (rebuilt when task count changes)
        self._cached_frozen_trans = None   # FrozenTransInput on device
        self._cached_frozen_pks = None     # (T, d) tensor on device
        self._cached_n_tasks = 0           # number of tasks when cache was built

        # GPM bases — per-chunk for input/output, single for medium
        # All stored as torch tensors on device
        self.gpm_bases_input = {}          # {chunk_idx: (step, r) tensor}
        self.gpm_bases_medium = None       # (h, r) tensor or None
        self.gpm_bases_output = {}         # {chunk_idx: (step, r) tensor}

    def _adaptive_threshold(self, task_idx, total_tasks=15):
        """ROOT cl_trainer_gainlora_inflora.py L340:
        threshold_t = (1 - base) * t / T + base
        """
        return (1.0 - self.transthreshold) * task_idx / total_tasks + self.transthreshold

    def add_task(self, task_embs, task_idx, total_tasks=15):
        """Learn routing for a new task and apply GPM protection.

        All computation on GPU. Per-chunk GPM matching ROOT exactly.
        """
        d = self.d_model
        h = self.mlp_hidden_dim
        C = self.chunk
        step = self.step
        threshold = self._adaptive_threshold(task_idx, total_tasks)
        dev = self.device

        X_gpu = torch.from_numpy(task_embs).float().to(dev)
        N = X_gpu.shape[0]

        # ── Step 1: Initialize trans_input MLP (kaiming a=√3) ──
        mlp = TransInputMLP(d, h, backbone=self.backbone).to(dev)

        # Project initial weights into GPM null-space (per-chunk)
        if task_idx > 0:
            with torch.no_grad():
                # linear1.weight (h, d): project INPUT columns per-chunk
                for ci in range(C):
                    s, e = ci * step, (ci + 1) * step
                    bases = self.gpm_bases_input.get(ci)
                    if bases is not None:
                        P = bases @ bases.T  # (step, step)
                        mlp.linear1.weight.data[:, s:e] -= (
                            mlp.linear1.weight.data[:, s:e] @ P)

                # linear2.weight (d, h): project MEDIUM columns (no chunk)
                if self.gpm_bases_medium is not None:
                    P = self.gpm_bases_medium @ self.gpm_bases_medium.T  # (h, h)
                    mlp.linear2.weight.data -= mlp.linear2.weight.data @ P

        # ── Step 2: Initialize prompt_key (per-chunk, ROOT's method) ──
        with torch.no_grad():
            out_feat = mlp(X_gpu)  # (N, d)

        pk = torch.zeros(d, device=dev)
        if task_idx == 0:
            # Task 0: top eigvec of output covariance per-chunk
            # ROOT cl_trainer_gainlora_inflora.py L218-L240
            with torch.no_grad():
                for ci in range(C):
                    s, e = ci * step, (ci + 1) * step
                    cov_c = (out_feat[:, s:e].T @ out_feat[:, s:e]).div_(max(N, 1))
                    U, _, _ = torch.linalg.svd(cov_c, full_matrices=False)
                    pk[s:e] = U[:, 0]
        else:
            # Task ≥1: top eigvec of random matrix projected to null-space
            # ROOT cl_trainer_gainlora_inflora.py L247-L260
            with torch.no_grad():
                for ci in range(C):
                    s, e = ci * step, (ci + 1) * step
                    R = torch.randn(step, step, device=dev)
                    bases = self.gpm_bases_output.get(ci)
                    if bases is not None:
                        P = bases @ bases.T
                        R = R - P @ R
                    U, _, _ = torch.linalg.svd(R, full_matrices=False)
                    pk[s:e] = U[:, 0]

        del out_feat
        # ROOT normalizes: /= √chunk then *= pre_norm.
        # Since cal_attention L2-normalizes anyway, unit-norm suffices.
        pk = pk / (pk.norm() + 1e-12)
        prompt_key = nn.Parameter(pk.to(dev))

        # ── Step 3: Train routing with gradient projection ──
        if task_idx > 0:
            self._train_routing(mlp, prompt_key, X_gpu, task_idx)

        # ── Step 4: Collect covariance per-chunk on GPU & update GPM ──
        with torch.no_grad():
            medium = mlp.forward_medium(X_gpu)  # (N, h)
            output = mlp(X_gpu)                 # (N, d)

            # Medium: single (h, h) covariance (not chunked — ROOT's design)
            cov_med = (medium.T @ medium).div_(max(N, 1))
            self.gpm_bases_medium = _gpm_update(
                self.gpm_bases_medium, cov_med, threshold)

            # Input & output: per-chunk (step, step) covariance
            for ci in range(C):
                s, e = ci * step, (ci + 1) * step
                cov_inp = (X_gpu[:, s:e].T @ X_gpu[:, s:e]).div_(max(N, 1))
                cov_out = (output[:, s:e].T @ output[:, s:e]).div_(max(N, 1))
                self.gpm_bases_input[ci] = _gpm_update(
                    self.gpm_bases_input.get(ci), cov_inp, threshold)
                self.gpm_bases_output[ci] = _gpm_update(
                    self.gpm_bases_output.get(ci), cov_out, threshold)

        del medium, output

        # ── Step 5: Snapshot ──
        self.prompt_keys.append(prompt_key.detach().cpu().numpy())
        self.frozen_W_in.append(mlp.linear1.weight.detach().cpu().numpy())
        self.frozen_W_out.append(mlp.linear2.weight.detach().cpu().numpy())

        del mlp, prompt_key, X_gpu
        if 'cuda' in dev:
            torch.cuda.empty_cache()

    # ── Precompute projection matrices for gradient projection ──
    def _build_proj_matrices(self):
        """Build P = U U^T for each GPM stage. Returns dict of GPU tensors."""
        C, step = self.chunk, self.step
        proj = {'input': {}, 'medium': None, 'output': {}}
        for ci in range(C):
            b = self.gpm_bases_input.get(ci)
            if b is not None:
                proj['input'][ci] = b @ b.T
            b = self.gpm_bases_output.get(ci)
            if b is not None:
                proj['output'][ci] = b @ b.T
        if self.gpm_bases_medium is not None:
            proj['medium'] = self.gpm_bases_medium @ self.gpm_bases_medium.T
        return proj

    def _get_frozen_trans_and_pks(self):
        """Return cached FrozenTransInput and prompt_keys on GPU.

        Rebuilds only when the task count changes (avoids repeated numpy→GPU).
        """
        n = len(self.prompt_keys)
        if n == self._cached_n_tasks and self._cached_frozen_trans is not None:
            return self._cached_frozen_trans, self._cached_frozen_pks

        frozen_pks = torch.tensor(
            np.stack(self.prompt_keys), dtype=torch.float32, device=self.device)
        frozen_trans = FrozenTransInput(
            [torch.tensor(w, dtype=torch.float32) for w in self.frozen_W_in],
            [torch.tensor(w, dtype=torch.float32) for w in self.frozen_W_out],
            backbone=self.backbone,
        ).to(self.device)
        frozen_trans.eval()

        self._cached_frozen_trans = frozen_trans
        self._cached_frozen_pks = frozen_pks
        self._cached_n_tasks = n
        return frozen_trans, frozen_pks

    def _train_routing(self, mlp, prompt_key, X_gpu, task_idx):
        """Train trans_input + prompt_key with ROOT-style gradient projection.

        ROOT cl_trainer_gainlora_inflora.py L1336-L1436:
          After every optimizer.step():
            1. W_new = W_new - (W_new - W_old) @ P   (per-chunk for input/output)
            2. Restore row norms (norm-preserving projection)
        """
        C, step = self.chunk, self.step

        frozen_trans, frozen_pks = self._get_frozen_trans_and_pks()

        optimizer = torch.optim.Adam(
            list(mlp.parameters()) + [prompt_key], lr=self.lr)

        # Precompute projection matrices (stays on GPU, reused every step)
        proj = self._build_proj_matrices()

        N = X_gpu.shape[0]
        bs = self.batch_size

        for epoch in range(self.epochs):
            perm = torch.randperm(N, device=self.device)
            for start in range(0, N, bs):
                idx = perm[start:start + bs]
                x_batch = X_gpu[idx]

                # ── Save old weights (for gradient projection) ──
                old_W1 = mlp.linear1.weight.data.clone()
                old_W2 = mlp.linear2.weight.data.clone()
                old_pk = prompt_key.data.clone()

                # ── Forward + loss ──
                x_cur = mlp(x_batch)
                with torch.no_grad():
                    x_prev = frozen_trans(x_batch)

                all_x = torch.cat([x_cur.unsqueeze(1), x_prev], dim=1)
                all_pk = torch.cat([prompt_key.unsqueeze(0), frozen_pks], dim=0)
                all_pk = all_pk.unsqueeze(0).expand(x_batch.shape[0], -1, -1)

                weights = cal_attention(all_pk, all_x)
                target = torch.zeros(x_batch.shape[0], dtype=torch.long,
                                     device=self.device)
                loss = F.cross_entropy(weights * 10, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # ── Gradient projection (ROOT's norm-preserving null-space) ──
                with torch.no_grad():
                    W1 = mlp.linear1.weight.data
                    W2 = mlp.linear2.weight.data
                    pk = prompt_key.data

                    # Save post-step norms
                    W1_norm = W1.norm(dim=1, keepdim=True)
                    W2_norm = W2.norm(dim=1, keepdim=True)
                    pk_norm = pk.norm()

                    # linear1 (h, d): input bases, per-chunk
                    for ci in range(C):
                        P = proj['input'].get(ci)
                        if P is not None:
                            s, e = ci * step, (ci + 1) * step
                            W1[:, s:e] -= (W1[:, s:e] - old_W1[:, s:e]) @ P

                    # linear2 (d, h): medium bases, full
                    P_med = proj['medium']
                    if P_med is not None:
                        W2.copy_(W2 - (W2 - old_W2) @ P_med)

                    # prompt_key (d,): output bases, per-chunk
                    for ci in range(C):
                        P = proj['output'].get(ci)
                        if P is not None:
                            s, e = ci * step, (ci + 1) * step
                            pk[s:e] -= (pk[s:e] - old_pk[s:e]) @ P

                    # Restore norms (ROOT's norm-preserving step)
                    mlp.linear1.weight.data = W1 * W1_norm / (W1.norm(dim=1, keepdim=True) + 1e-12)
                    mlp.linear2.weight.data = W2 * W2_norm / (W2.norm(dim=1, keepdim=True) + 1e-12)
                    prompt_key.data = pk * (pk_norm / (pk.norm() + 1e-12))

    def route(self, h_batch):
        """Route embeddings to task indices (argmax of cal_attention weights)."""
        n_tasks = len(self.prompt_keys)
        if n_tasks == 0:
            raise ValueError("No tasks added yet")
        if n_tasks == 1:
            return np.zeros(h_batch.shape[0], dtype=np.int64)

        X = torch.from_numpy(h_batch).float().to(self.device)
        frozen_trans, all_pk = self._get_frozen_trans_and_pks()

        cs = 2048 if 'cuda' in self.device else 256
        all_preds = []
        with torch.no_grad():
            for i in range(0, X.shape[0], cs):
                xc = X[i:i + cs]
                all_x = frozen_trans(xc)
                w = cal_attention(all_pk, all_x)
                all_preds.append(w.argmax(dim=1).cpu())

        del X
        return torch.cat(all_preds).numpy()


# ═══════════════════════════════════════════════════════════════════════
# 2. RLS Routing (SpecRoute — Woodbury incremental)
# ═══════════════════════════════════════════════════════════════════════

class RLSRouter:
    """Offline RLS router matching SpecRoute's Woodbury implementation.

    Random feature expansion + incremental analytical ridge regression.
    Supports GPU acceleration when device='cuda'.
    """
    def __init__(self, d_model, expansion_dim=2048, lam=0.1, seed=42, device='cpu'):
        self.d_model = d_model
        self.E = expansion_dim
        self.lam = lam
        self.device = device
        self.use_gpu = 'cuda' in device and HAS_TORCH

        rng = np.random.RandomState(seed)
        W_phi_np = (rng.randn(d_model, expansion_dim) / np.sqrt(d_model)).astype(np.float32)
        b_phi_np = (rng.randn(expansion_dim) * 0.01).astype(np.float32)

        if self.use_gpu:
            dev = torch.device(device)
            self.W_phi = torch.tensor(W_phi_np, dtype=torch.float32, device=dev)
            self.b_phi = torch.tensor(b_phi_np, dtype=torch.float32, device=dev)
            self.R = torch.eye(expansion_dim, dtype=torch.float64, device=dev) / lam
            self.Q = torch.zeros(expansion_dim, 0, dtype=torch.float64, device=dev)
            self.W_r = torch.zeros(expansion_dim, 0, dtype=torch.float64, device=dev)
        else:
            self.W_phi = W_phi_np
            self.b_phi = b_phi_np
            self.R = np.eye(expansion_dim, dtype=np.float64) / lam
            self.Q = np.zeros((expansion_dim, 0), dtype=np.float64)
            self.W_r = np.zeros((expansion_dim, 0), dtype=np.float64)
        self.num_tasks = 0

    def _expand(self, h):
        """h: (N, d) → (N, E) via frozen random features."""
        if self.use_gpu:
            if not isinstance(h, torch.Tensor):
                h = torch.tensor(h, dtype=torch.float32, device=torch.device(self.device))
            return torch.relu(h @ self.W_phi + self.b_phi).to(torch.float64)
        return np.maximum(0, h @ self.W_phi + self.b_phi)

    def add_task(self, task_embs):
        """Woodbury update for new task."""
        if self.use_gpu:
            dev = torch.device(self.device)
            if isinstance(task_embs, np.ndarray):
                task_embs_t = torch.tensor(task_embs, dtype=torch.float32, device=dev)
            else:
                task_embs_t = task_embs.to(dev)
            H = self._expand(task_embs_t)
            N = H.shape[0]
            chunk = 512
            R = self.R.clone()
            for start in range(0, N, chunk):
                Hc = H[start:min(start + chunk, N)]
                RH = R @ Hc.T
                S = torch.eye(Hc.shape[0], dtype=torch.float64, device=dev) + Hc @ RH
                try:
                    S_inv_HcR = torch.linalg.solve(S, Hc @ R)
                    R = R - RH @ S_inv_HcR
                except Exception:
                    pass
            R = (R + R.T) * 0.5
            R += 1e-6 * torch.eye(self.E, dtype=torch.float64, device=dev)
            self.R = R
            extra = torch.zeros(self.E, 1, dtype=torch.float64, device=dev)
            extra[:, 0] = H.T @ torch.ones(N, dtype=torch.float64, device=dev)
            self.Q = torch.cat([self.Q, extra], dim=1)
            self.W_r = self.R @ self.Q
            self.num_tasks += 1
        else:
            H = self._expand(task_embs).astype(np.float64)
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
        """h_batch: (N, d) → predicted task indices (N,)."""
        if self.use_gpu:
            dev = torch.device(self.device)
            if isinstance(h_batch, np.ndarray):
                h_batch = torch.tensor(h_batch, dtype=torch.float32, device=dev)
            H = self._expand(h_batch)
            logits = H @ self.W_r
            return logits.argmax(dim=1).cpu().numpy()
        H = self._expand(h_batch).astype(np.float64)
        logits = H @ self.W_r
        return logits.argmax(axis=1)


# ═══════════════════════════════════════════════════════════════════════
# 3. Non-parametric baselines
# ═══════════════════════════════════════════════════════════════════════

class NearestCentroidRouter:
    """L2 nearest centroid (simplest possible baseline). GPU-accelerated."""
    def __init__(self, device='cpu'):
        self.centroids = []
        self.device = device
        self.use_gpu = 'cuda' in device and HAS_TORCH

    def add_task(self, embs):
        if self.use_gpu:
            dev = torch.device(self.device)
            X = torch.tensor(embs, dtype=torch.float32, device=dev)
            self.centroids.append(X.mean(0))
            del X
        else:
            self.centroids.append(embs.mean(0))

    def route(self, h_batch):
        if self.use_gpu:
            dev = torch.device(self.device)
            C = torch.stack(self.centroids)  # (T, d) — already on GPU
            if isinstance(h_batch, np.ndarray):
                H = torch.tensor(h_batch, dtype=torch.float32, device=dev)
            else:
                H = h_batch.to(dev)
            # L2 via expansion: ||h-c||^2 = ||h||^2 + ||c||^2 - 2*h@c^T
            H_sq = (H ** 2).sum(1, keepdim=True)  # (N, 1)
            C_sq = (C ** 2).sum(1, keepdim=True).T  # (1, T)
            dists = H_sq + C_sq - 2 * (H @ C.T)
            return dists.argmin(dim=1).cpu().numpy()
        C = np.stack(self.centroids)  # (T, d)
        # Avoid (N, T, d) intermediate — use expansion
        H_sq = np.sum(h_batch ** 2, axis=1, keepdims=True)
        C_sq = np.sum(C ** 2, axis=1, keepdims=True).T
        dists = H_sq + C_sq - 2 * (h_batch @ C.T)
        return dists.argmin(axis=1)


class CosineNearestCentroidRouter:
    """Cosine nearest centroid. GPU-accelerated."""
    def __init__(self, device='cpu'):
        self.centroids = []
        self.device = device
        self.use_gpu = 'cuda' in device and HAS_TORCH

    def add_task(self, embs):
        if self.use_gpu:
            dev = torch.device(self.device)
            X = torch.tensor(embs, dtype=torch.float32, device=dev)
            mu = X.mean(0)
            self.centroids.append(mu / (mu.norm() + 1e-12))
            del X
        else:
            mu = embs.mean(0)
            self.centroids.append(mu / (np.linalg.norm(mu) + 1e-12))

    def route(self, h_batch):
        if self.use_gpu:
            dev = torch.device(self.device)
            C = torch.stack(self.centroids)  # (T, d)
            if isinstance(h_batch, np.ndarray):
                H = torch.tensor(h_batch, dtype=torch.float32, device=dev)
            else:
                H = h_batch.to(dev)
            H_norm = H / (H.norm(dim=1, keepdim=True) + 1e-12)
            sims = H_norm @ C.T
            return sims.argmax(dim=1).cpu().numpy()
        C = np.stack(self.centroids)
        h_norm = h_batch / (np.linalg.norm(h_batch, axis=1, keepdims=True) + 1e-12)
        sims = h_norm @ C.T  # (N, T)
        return sims.argmax(axis=1)

class BuggyFitOnceWhitenedRouter:
    """
    [A/B Test - Bản Lỗi] Mô phỏng chính xác code hiện tại:
    Fit ZCA DUY NHẤT ở Task 1. Dùng ZCA cũ rích này để biến đổi cho mọi Task sau.
    Không lưu Raw Centroid, không Re-whiten.
    """
    def __init__(self, device='cpu'):
        self.signatures = []
        self.mu_g = None
        self.W_zca = None
        self.zca_fitted = False
        self.device = device
        self.use_gpu = 'cuda' in device and HAS_TORCH

    def add_task(self, embs):
        if self.use_gpu:
            dev = torch.device(self.device)
            X = torch.tensor(embs, dtype=torch.float32, device=dev) if isinstance(embs, np.ndarray) else embs.to(dev)
            
            if not self.zca_fitted:
                self.mu_g = X.mean(0)
                Xc = X - self.mu_g
                cov = (Xc.T @ Xc) / max(X.shape[0] - 1, 1)
                ev, evec = torch.linalg.eigh(cov)
                ev = torch.clamp(ev, min=1e-8)
                self.W_zca = evec @ torch.diag(1.0 / torch.sqrt(ev)) @ evec.T
                self.zca_fitted = True
            
            # Extract raw centroid and project using the STALE ZCA
            mu_raw = X.mean(0)
            self.signatures.append((mu_raw - self.mu_g) @ self.W_zca)
        else:
            if not self.zca_fitted:
                self.mu_g = embs.mean(0)
                cov = np.cov(embs, rowvar=False)
                ev, evec = np.linalg.eigh(cov)
                ev = np.maximum(ev, 1e-8)
                self.W_zca = evec @ np.diag(1.0 / np.sqrt(ev)) @ evec.T
                self.zca_fitted = True
                
            mu_raw = embs.mean(0)
            self.signatures.append((mu_raw - self.mu_g) @ self.W_zca)

    def route(self, h_batch):
        if not self.signatures: return np.zeros(h_batch.shape[0], dtype=np.int64)
        if self.use_gpu:
            dev = torch.device(self.device)
            H = torch.tensor(h_batch, dtype=torch.float32, device=dev) if isinstance(h_batch, np.ndarray) else h_batch.to(dev)
            H_w = (H - self.mu_g) @ self.W_zca
            C_w = torch.stack(self.signatures)
            H_sq = (H_w ** 2).sum(1, keepdim=True)
            C_sq = (C_w ** 2).sum(1, keepdim=True).T
            dists = H_sq + C_sq - 2 * (H_w @ C_w.T)
            return dists.argmin(dim=1).cpu().numpy()
        else:
            H_w = (h_batch - self.mu_g) @ self.W_zca
            C_w = np.stack(self.signatures)
            H_sq = np.sum(H_w ** 2, axis=1, keepdims=True)
            C_sq = np.sum(C_w ** 2, axis=1, keepdims=True).T
            dists = H_sq + C_sq - 2 * (H_w @ C_w.T)
            return dists.argmin(axis=1)

class IncrementalWhitenedRouter:
    """
    [A/B Test - Bản Đúng] Mô phỏng kịch bản chuẩn:
    Lưu Raw Centroid. Re-fit ZCA sau mỗi Task.
    Re-whiten toàn bộ Signatures cũ bằng ZCA mới nhất.
    """
    def __init__(self, device='cpu'):
        self.raw_centroids = []
        self.seen_embs = []
        self.mu_g = None
        self.W_zca = None
        self.signatures = []
        self.device = device
        self.use_gpu = 'cuda' in device and HAS_TORCH

    def add_task(self, embs):
        if self.use_gpu:
            dev = torch.device(self.device)
            X = torch.tensor(embs, dtype=torch.float32, device=dev) if isinstance(embs, np.ndarray) else embs.to(dev)
            
            # 1. Lưu Raw Centroid
            self.raw_centroids.append(X.mean(0))
            
            # 2. Cập nhật dữ liệu để tính ZCA (Pool)
            self.seen_embs.append(X)
            all_embs = torch.cat(self.seen_embs, dim=0)
            
            # 3. Tính hệ quy chiếu mới (New ZCA)
            self.mu_g = all_embs.mean(0)
            all_embs_c = all_embs - self.mu_g
            cov = (all_embs_c.T @ all_embs_c) / max(all_embs.shape[0] - 1, 1)
            ev, evec = torch.linalg.eigh(cov)
            ev = torch.clamp(ev, min=1e-8)
            self.W_zca = evec @ torch.diag(1.0 / torch.sqrt(ev)) @ evec.T
            
            # 4. RE-WHITEN toàn bộ Signatures
            self.signatures = [(mu_r - self.mu_g) @ self.W_zca for mu_r in self.raw_centroids]
        else:
            self.raw_centroids.append(embs.mean(0))
            self.seen_embs.append(embs)
            all_embs = np.vstack(self.seen_embs)
            
            self.mu_g = all_embs.mean(0)
            cov = np.cov(all_embs, rowvar=False)
            ev, evec = np.linalg.eigh(cov)
            ev = np.maximum(ev, 1e-8)
            self.W_zca = evec @ np.diag(1.0 / np.sqrt(ev)) @ evec.T
            
            self.signatures = [(mu_r - self.mu_g) @ self.W_zca for mu_r in self.raw_centroids]

    def route(self, h_batch):
        if not self.signatures: return np.zeros(h_batch.shape[0], dtype=np.int64)
        if self.use_gpu:
            dev = torch.device(self.device)
            H = torch.tensor(h_batch, dtype=torch.float32, device=dev) if isinstance(h_batch, np.ndarray) else h_batch.to(dev)
            H_w = (H - self.mu_g) @ self.W_zca
            C_w = torch.stack(self.signatures)
            H_sq = (H_w ** 2).sum(1, keepdim=True)
            C_sq = (C_w ** 2).sum(1, keepdim=True).T
            dists = H_sq + C_sq - 2 * (H_w @ C_w.T)
            return dists.argmin(dim=1).cpu().numpy()
        else:
            H_w = (h_batch - self.mu_g) @ self.W_zca
            C_w = np.stack(self.signatures)
            H_sq = np.sum(H_w ** 2, axis=1, keepdims=True)
            C_sq = np.sum(C_w ** 2, axis=1, keepdims=True).T
            dists = H_sq + C_sq - 2 * (H_w @ C_w.T)
            return dists.argmin(axis=1)

class PSRRouter:
    """PPCA-based PSR routing (non-parametric, from analyze_geometry).
    
    Supports GPU acceleration when device='cuda'.
    """
    def __init__(self, k=8, device='cpu'):
        self.k = k
        self.sigs = []
        self.device = device
        self.use_gpu = 'cuda' in device and HAS_TORCH

    def add_task(self, embs):
        n, d = embs.shape
        if self.use_gpu:
            dev = torch.device(self.device)
            X = torch.tensor(embs, dtype=torch.float32, device=dev)
            mu = X.mean(0)
            Xc = X - mu
            cov_t = (Xc.T @ Xc) / max(n - 1, 1)
            ev, evec = torch.linalg.eigh(cov_t)
            idx = torch.argsort(ev, descending=True)
            ev = ev[idx]; evec = evec[:, idx]
            k = min(self.k, d)
            V = evec[:, :k]
            lam = torch.clamp(ev[:k], min=1e-12)
            sigma2 = float(max(ev[k:].mean().item(), 1e-12)) if k < d else 1e-12
            self.sigs.append((mu, V, lam, sigma2, d))
            del X, Xc, cov_t, ev, evec
        else:
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

    def route(self, h_batch):
        """Vectorized PSR routing — all tasks evaluated simultaneously."""
        T = len(self.sigs)
        if T == 0 or T == 1:
            return np.zeros(h_batch.shape[0], dtype=np.int64)

        if self.use_gpu:
            dev = torch.device(self.device)
            d = self.sigs[0][4]
            k = min(self.k, d)
            C   = torch.stack([s[0] for s in self.sigs])  # (T, d) — already on GPU
            V   = torch.stack([s[1] for s in self.sigs])  # (T, d, k)
            lam = torch.stack([s[2] for s in self.sigs])  # (T, k)
            s2  = torch.tensor([s[3] for s in self.sigs], dtype=torch.float32, device=dev)
            W_psr = lam / (s2[:, None] * (lam + s2[:, None]))
            pen = torch.log(lam + s2[:, None]).sum(1) + (d - k) * torch.log(s2)
            if isinstance(h_batch, np.ndarray):
                H = torch.tensor(h_batch, dtype=torch.float32, device=dev)
            else:
                H = h_batch.to(dev)
            # Avoid (N,T,d) intermediate: iso = ||H-C||^2 / s2
            H_sq = (H ** 2).sum(1, keepdim=True)                            # (N, 1)
            C_sq = (C ** 2).sum(1)                                           # (T,)
            l2   = H_sq + C_sq.unsqueeze(0) - 2 * (H @ C.T)                 # (N, T)
            iso  = l2 / (s2[None, :] + 1e-12)                               # (N, T)
            # d_proj = (H-C)@V = H@V - C@V  (no (N,T,d) intermediate)
            H_proj = torch.einsum('nd,tdk->ntk', H, V)                      # (N, T, k)
            CV     = torch.einsum('td,tdk->tk', C, V)                        # (T, k)
            dp     = H_proj - CV.unsqueeze(0)                                # (N, T, k)
            dists = iso + (W_psr[None, :, :] * dp.pow(2)).sum(-1) + pen[None, :]
            return dists.argmin(dim=1).cpu().numpy().astype(np.int64)
        else:
            d = self.sigs[0][4]
            k = min(self.k, d)
            C   = np.stack([s[0] for s in self.sigs]).astype(np.float32)
            V   = np.stack([s[1] for s in self.sigs]).astype(np.float32)
            lam = np.stack([s[2] for s in self.sigs]).astype(np.float32)
            s2  = np.array([s[3] for s in self.sigs], dtype=np.float32)
            W_psr = lam / (s2[:, None] * (lam + s2[:, None]))
            pen = np.sum(np.log(lam + s2[:, None]), axis=1) + (d - k) * np.log(s2)
            H = h_batch.astype(np.float32)
            # Avoid (N,T,d) intermediate
            H_sq = np.sum(H ** 2, axis=1, keepdims=True)                     # (N, 1)
            C_sq = np.sum(C ** 2, axis=1)                                     # (T,)
            l2   = H_sq + C_sq[None, :] - 2 * (H @ C.T)                      # (N, T)
            iso  = l2 / (s2[None, :] + 1e-12)                                # (N, T)
            H_proj = np.einsum('nd,tdk->ntk', H, V)                          # (N, T, k)
            CV     = np.einsum('td,tdk->tk', C, V)                            # (T, k)
            dp     = H_proj - CV[None, :, :]                                  # (N, T, k)
            dists = iso + np.sum(W_psr[None, :, :] * dp**2, axis=-1) + pen[None, :]
            return np.argmin(dists, axis=1).astype(np.int64)


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
    routers["NearestCentroid"] = NearestCentroidRouter(device=args.device)
    routers["CosineNearestCentroid"] = CosineNearestCentroidRouter(device=args.device)
    routers["PSR"] = PSRRouter(k=args.subspace_k, device=args.device)
    routers["Buggy_FitOnce_Whiten"] = BuggyFitOnceWhitenedRouter(device=args.device)  # <-- THÊM DÒNG NÀY
    routers["Correct_ReWhiten"] = IncrementalWhitenedRouter(device=args.device)       # <-- THÊM DÒNG NÀY
    routers["NearestCentroid"] = NearestCentroidRouter(device=args.device)

    routers["RLS_Woodbury"] = RLSRouter(
        d, expansion_dim=args.rls_expansion, lam=args.rls_lambda,
        device=args.device)

    if HAS_TORCH:
        routers["GPM_ROOT"] = GPMRouter(
            d, mlp_hidden_dim=args.mlp_hidden_dim,
            transthreshold=args.transthreshold,
            lr=args.lr, epochs=args.epochs,
            batch_size=args.batch_size,
            chunk=args.chunk, backbone=args.backbone_type,
            device=args.device)
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
    gpm.add_argument("--mlp_hidden_dim", type=int, default=None,
                     help="trans_input MLP hidden dim (auto: T5=100, Llama=50)")
    gpm.add_argument("--transthreshold", type=float, default=0.995,
                     help="GPM energy threshold (ROOT default: 0.995)")
    gpm.add_argument("--chunk", type=int, default=None,
                     help="GPM chunking factor (auto: T5=1, Llama=4)")
    gpm.add_argument("--backbone_type", default="auto",
                     choices=["t5", "llama", "auto"],
                     help="Backbone type for architecture selection (default: auto-detect)")
    gpm.add_argument("--lr", type=float, default=1e-3,
                     help="Proxy training learning rate")
    gpm.add_argument("--epochs", type=int, default=30,
                     help="Proxy training epochs per task")
    gpm.add_argument("--batch_size", type=int, default=256,
                     help="Training batch size")
    gpm.add_argument("--device", default="auto",
                     help="Device for GPM training: cpu | cuda | cuda:0 | auto (default: auto)")
    gpm.add_argument("--force", action="store_true",
                     help="Force re-run even if output already exists")

    # RLS parameters
    rls = parser.add_argument_group("RLS routing parameters")
    rls.add_argument("--rls_expansion", type=int, default=2048,
                     help="Random feature expansion dim (SpecRoute default: 2048)")
    rls.add_argument("--rls_lambda", type=float, default=0.1,
                     help="Ridge regularization (SpecRoute default: 0.1)")

    args = parser.parse_args()
    args.device = _resolve_device(args.device)

    # ── Auto-detect backbone type from directory name ──
    backbone = Path(args.emb_dir).name
    if args.backbone_type == 'auto':
        args.backbone_type = 'llama' if 'llama' in backbone.lower() else 't5'
    is_llama = (args.backbone_type == 'llama')

    # ── Auto-set ROOT defaults based on backbone ──
    if args.mlp_hidden_dim is None:
        args.mlp_hidden_dim = 50 if is_llama else 100
    if args.chunk is None:
        args.chunk = 4 if is_llama else 1

    tasks = BENCHMARKS[args.benchmark]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{backbone}_{args.benchmark}" + ("_whitened" if args.whiten else "")

    # ── Skip if already done ──
    out_path = out_dir / f"learned_routing_{tag}.json"
    if out_path.exists() and not args.force:
        print(f"[SKIP] Phase F: {out_path} already exists. Use --force to re-run.")
        return

    print(f"=== Phase F: Learned Routing Comparison  [{tag}] ===")
    print(f"    GPM: backbone={args.backbone_type}, mlp_hidden={args.mlp_hidden_dim}, "
          f"chunk={args.chunk}, transthreshold={args.transthreshold}, "
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
        mu_g, W = compute_whitening(train_embs, device=args.device)
        train_embs = apply_whitening(train_embs, mu_g, W, device=args.device)
        test_embs = apply_whitening(test_embs, mu_g, W, device=args.device)
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
