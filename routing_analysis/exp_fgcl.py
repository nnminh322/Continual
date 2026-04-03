#!/usr/bin/env python3
"""
exp_fgcl.py — FGCL Experiment Framework
========================================
Usage:
  python exp_fgcl.py                                    # ALL: all phases × all models × all benchmarks
  python exp_fgcl.py --phase T4                         # T4, all models, all benchmarks
  python exp_fgcl.py --model test_embeddings/TestBackbone/   # all phases, this model, all benchmarks
  python exp_fgcl.py --benchmark SuperNI                # all phases, all models, this benchmark
  python exp_fgcl.py --model test_embeddings/TestBackbone/ --benchmark SuperNI  # specific

Embedding path convention:
  {emb_root}/{model_name}/{benchmark}/{task}/train.npz
  e.g. test_embeddings/TestBackbone/Long_Sequence/sst2/train.npz

Methods:
  standard_lora  — Plain LoRA (baseline)
  gainlora       — GainLoRA root port (routing + GPM + KL)
  inflora        — InfLoRA (GPM only)
  fgcl_fsr       — FGCL: LoRA + Fisher Subspace Regularization
  fgcl_kfng      — FGCL: LoRA + FSR + Kronecker-Factored Fisher NG
  fgcl_taa       — FGCL: LoRA + FSR + Task Arithmetic Alignment
  fgcl_sgr       — FGCL: LoRA + Soft Grassmannian Regularization (GALA baseline)

Phases:
  T1  FSR vs GPM isolation (2 tasks: sst2 → imdb)
  T2  KF-FNG convergence   (1 task: sst2)
  T3  TAA vs SGR ablation  (3 tasks: sst2 → imdb → yelp)
  T4  Full comparison      (5 tasks: sst2 → imdb → yelp → amazon → agnews)
"""

import os, sys, json, time, math, copy, pickle, argparse, gc, warnings
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Experiment Design ────────────────────────────────────────────────────
# "One run must be statistically conclusive."
# Params chosen: enough statistical power, minimal redundancy.

ALL_PHASES = ["T1", "T2", "T3", "T4"]
ALL_PHASE_METHODS = {
    "T1": ["standard_lora", "inflora", "fgcl_fsr"],
    "T2": ["standard_lora", "fgcl_kfng"],
    "T3": ["standard_lora", "fgcl_fsr", "fgcl_taa", "fgcl_sgr"],
    "T4": ["standard_lora", "gainlora", "inflora", "fgcl_fsr", "fgcl_kfng", "fgcl_taa"],
}
PHASE_DESCS = {
    "T1": "FSR vs GPM isolation (6 tasks)",
    "T2": "KF-FNG convergence (1 task)",
    "T3": "TAA vs SGR ablation (8 tasks)",
    "T4": "Full comparison (all tasks)",
}
# Task lists are DISCOVERED at runtime from the actual NPZ files available
# for each (model_dir, benchmark) combo — no hardcoded names.
PHASE_TASK_COUNTS = {
    "T1": 6,
    "T2": 1,
    "T3": 8,
    "T4": None,   # use all available tasks
}

# Default training hyperparams (sufficient for all phases)
DEFAULT_LORA_RANK = 8
DEFAULT_LORA_ALPHA = 32
DEFAULT_LR = 1e-3
DEFAULT_EPOCHS = {"T1": 20, "T2": 30, "T3": 20, "T4": 20}
DEFAULT_BATCH = 32
DEFAULT_GPM_STEPS = 100
DEFAULT_LAMBDA_FSR = 0.1
DEFAULT_LAMBDA_TAA = 0.05
DEFAULT_LAMBDA_SGR = 0.1


# ═══════════════════════════════════════════════════════════════════════
# LoRA — Shared
# ═══════════════════════════════════════════════════════════════════════

class LoRALinear(nn.Module):
    """LoRA: A=Kaiming init, B=zero init → ΔW=0 at step 0."""

    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: int = 32):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.in_features = in_features
        self.out_features = out_features
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.lora_A.T @ self.lora_B.T * self.scale

    @property
    def delta_W(self) -> torch.Tensor:
        return self.lora_B @ self.lora_A * self.scale


# ═══════════════════════════════════════════════════════════════════════
# GPM — Gradient Projection Memory (embedding-level)
# ═══════════════════════════════════════════════════════════════════════

class GPM:
    """GPM: accumulate activation covariances → SVD bases → project gradients."""

    def __init__(self, threshold: float = 0.99):
        self.threshold = threshold
        self.bases: Dict[str, torch.Tensor] = {}
        self.cov: Dict[str, torch.Tensor] = {}
        self.counts: Dict[str, int] = {}

    def accumulate(self, name: str, h: torch.Tensor):
        """Accumulate outer product of activation tensor."""
        if h.dim() == 1:
            h = h.unsqueeze(0)
        if h.dim() > 2:
            h = h.reshape(h.shape[0], -1)
        N, d = h.shape
        self.cov[name] = self.cov.get(name, torch.zeros(d, d, device=h.device)) + h.T @ h
        self.counts[name] = self.counts.get(name, 0) + N

    def compute_bases(self, min_var=0.99):
        self.bases.clear()
        for name, cov in self.cov.items():
            try:
                U, S, _ = torch.linalg.svd(cov.float(), full_matrices=False)
                cumvar = torch.cumsum(S * S, dim=0)
                total = cumvar[-1]
                k = (cumvar >= total * min_var).nonzero(as_tuple=True)[0][0].item() + 1
                k = min(k, U.shape[1])
                self.bases[name] = U[:, :k]
            except Exception:
                self.bases[name] = torch.empty(0)
        self.cov.clear()
        self.counts.clear()

    def project(self, named_params) -> Dict[str, torch.Tensor]:
        result = {}
        for name, param in named_params:
            if param.grad is None:
                continue
            g = param.grad.clone()
            if name in self.bases and self.bases[name].numel() > 0:
                U = self.bases[name].to(g)
                g = g - U @ (U.T @ g.reshape(-1, 1)).squeeze(-1)
            result[name] = g
        return result

    def state_dict(self) -> Dict:
        return {"bases": self.bases, "cov": self.cov, "counts": self.counts}

    def load_state(self, s: Dict):
        self.bases = s.get("bases", {})
        self.cov = s.get("cov", {})
        self.counts = s.get("counts", {})


# ═══════════════════════════════════════════════════════════════════════
# FSR — Fisher Subspace Regularization
# ═══════════════════════════════════════════════════════════════════════

class FSR:
    """L_FSR = λ · ||P_{<t} · ∇L_t||²"""

    def __init__(self, lam: float = 0.1):
        self.lam = lam
        self.bases: Dict[str, torch.Tensor] = {}

    def compute_fisher(self, named_params, grad_dict) -> Dict[str, torch.Tensor]:
        fisher = {}
        for name, param in named_params:
            if name not in grad_dict or grad_dict[name] is None:
                continue
            g = grad_dict[name]
            if g.numel() == 0:
                continue
            g_f = g.reshape(-1, 1)
            fisher[name] = g_f @ g_f.T
        return fisher

    def update_subspace(self, fisher_mats, threshold=0.99):
        for name, fm in fisher_mats.items():
            try:
                U, S, _ = torch.linalg.svd(fm.float(), full_matrices=False)
                cumvar = torch.cumsum(S * S, dim=0)
                k = (cumvar >= cumvar[-1] * threshold).nonzero(as_tuple=True)[0][0].item() + 1
                k = min(k, U.shape[1])
                basis = U[:, :k]
                if name in self.bases:
                    comb = torch.cat([self.bases[name], basis], dim=1)
                    Q, _ = torch.linalg.qr(comb)
                    cov2 = Q.T @ Q
                    U2, S2, _ = torch.linalg.svd(cov2.float(), full_matrices=False)
                    c2 = torch.cumsum(S2 * S2, dim=0)
                    k2 = (c2 >= c2[-1] * threshold).nonzero(as_tuple=True)[0][0].item() + 1
                    k2 = min(k2, Q.shape[1])
                    self.bases[name] = (Q @ U2)[:, :k2]
                else:
                    self.bases[name] = basis
            except Exception:
                pass

    def loss(self, named_params, grad_dict) -> Tuple[torch.Tensor, Dict]:
        if not self.bases:
            return torch.tensor(0.0), {}
        norms, total = {}, 0.0
        for name, param in named_params:
            if name in grad_dict and grad_dict[name] is not None:
                n = (grad_dict[name] ** 2).sum().item()
                norms[name] = n
                total += n
        total += 1e-8
        total_l, comp = 0.0, {}
        for name, param in named_params:
            if name not in grad_dict or grad_dict[name] is None or name not in self.bases:
                comp[name] = 0.0
                continue
            g = grad_dict[name]
            U = self.bases[name].to(g)
            g_f = g.reshape(-1, 1)
            proj_sq = (U.T @ g_f).pow(2).sum().item()
            c = self.lam * (norms.get(name, 0.0) / total) * proj_sq
            total_l += c
            comp[name] = c
        return torch.tensor(total_l, device="cpu"), comp

    def state_dict(self) -> Dict:
        return {"bases": self.bases}

    def load_state(self, s: Dict):
        self.bases = s.get("bases", {})


# ═══════════════════════════════════════════════════════════════════════
# KFFNG — Kronecker-Factored Fisher Natural Gradient
# ═══════════════════════════════════════════════════════════════════════

class KFFNGOptimizer(torch.optim.Optimizer):
    """KF-FNG: F = G ⊗ Σ_x (exact for LoRA)."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), k_f=16, f_ema=0.99, eps=1e-8):
        defaults = dict(lr=lr, betas=betas, k_f=k_f, f_ema=f_ema, eps=eps)
        super().__init__(params, defaults)
        self._fs: Dict[int, Dict] = {}

    @torch.no_grad()
    def update_fisher(self, named_params, grad_dict):
        for name, param in named_params:
            if name not in grad_dict or grad_dict[name] is None:
                continue
            pid = id(param)
            if pid not in self._fs:
                self._fs[pid] = {"ema": None, "cnt": 0}
            g = grad_dict[name].detach()
            outer = (g.reshape(-1, 1) @ g.reshape(1, -1))
            s = self._fs[pid]
            if s["ema"] is None:
                s["ema"] = outer.clone()
            else:
                s["ema"] = self.defaults["f_ema"] * s["ema"] + (1 - self.defaults["f_ema"]) * outer
            s["cnt"] += 1
            if s["cnt"] % 50 == 0 and s["ema"] is not None:
                try:
                    ev, ec = torch.linalg.eigh(s["ema"].float())
                    ev = ev.flip(0)
                    ec = ec.flip(1)
                    k = min(self.defaults["k_f"], ev.shape[0])
                    s["V"] = ec[:, :k]
                    s["lam"] = ev[:k].clamp(min=1e-8)
                except Exception:
                    pass

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, b1, b2, eps = group["lr"], group["betas"][0], group["betas"][1], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                pid = id(p)
                state = self.state[p]
                if not state:
                    state.update(step=0, exp_avg=torch.zeros_like(p), exp_avg_sq=torch.zeros_like(p))
                state["step"] += 1
                g = p.grad
                state["exp_avg"].mul_(b1).add_(g, alpha=1 - b1)
                state["exp_avg_sq"].mul_(b2).addcmul_(g, g, value=1 - b2)
                bias1 = 1 - b1 ** state["step"]
                bias2 = 1 - b2 ** state["step"]
                denom = (state["exp_avg_sq"].sqrt() / math.sqrt(bias2)).add_(eps)
                adam_up = state["exp_avg"] / denom * (lr / bias1)

                # NG correction
                fs = self._fs.get(pid, {})
                if fs.get("V") is not None and fs.get("lam") is not None:
                    try:
                        V, lam = fs["V"].to(p), fs["lam"].to(p)
                        inv_sqrt = (1.0 / lam).sqrt()
                        ng = (V * inv_sqrt.unsqueeze(0)) @ (V.T @ g.reshape(-1, 1))
                        ng_up = ng.reshape_as(g) * (lr / bias1)
                        update = 0.5 * adam_up + 0.5 * ng_up
                    except Exception:
                        update = adam_up
                else:
                    update = adam_up
                p.data.add_(update, alpha=-1)
        return loss


# ═══════════════════════════════════════════════════════════════════════
# TAA — Task Arithmetic Alignment
# ═══════════════════════════════════════════════════════════════════════

class TAA:
    """L_TAA = μ · Σ_{s<t} w_s · ⟨∇L_t, τ_s⟩²"""

    def __init__(self, lam=0.05):
        self.lam = lam
        self.task_vectors: Dict[int, Dict[str, torch.Tensor]] = {}

    def register(self, task_id: int, lora_modules: Dict[str, LoRALinear]):
        self.task_vectors[task_id] = {
            name: lm.delta_W.detach().clone()
            for name, lm in lora_modules.items()
        }

    def loss(self, lora_modules: Dict[str, LoRALinear],
             grad_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        if not self.task_vectors:
            return torch.tensor(0.0), {}
        mags, total = {}, 0.0
        for tid, tv in self.task_vectors.items():
            mag = sum((v ** 2).sum().item() for v in tv.values())
            mags[tid] = math.sqrt(mag + 1e-8)
            total += mag
        total += 1e-8
        total_l, comp = 0.0, {}
        for name, lm in lora_modules.items():
            if name not in grad_dict or grad_dict[name] is None:
                comp[name] = 0.0
                continue
            g = grad_dict[name].detach().reshape(-1)
            for tid, tv in self.task_vectors.items():
                if name not in tv:
                    continue
                tau = tv[name].reshape(-1)
                mn = min(len(g), len(tau))
                dot_sq = (g[:mn] * tau[:mn]).sum().item() ** 2
                c = self.lam * (mags[tid] ** 2 / total) * dot_sq
                total_l += c
            comp[name] = total_l
        return torch.tensor(total_l, device="cpu"), comp

    def state_dict(self) -> Dict:
        return {"task_vectors": self.task_vectors}

    def load_state(self, s: Dict):
        self.task_vectors = s.get("task_vectors", {})


# ═══════════════════════════════════════════════════════════════════════
# SGR — Soft Grassmannian Regularization (GALA baseline)
# ═══════════════════════════════════════════════════════════════════════

class SGR:
    """L_SGR = λ₁ · Σ_{s<t} ||V_t^T V_s||_F^2"""

    def __init__(self, lam=0.1):
        self.lam = lam
        self.prev_subspaces: Dict[str, torch.Tensor] = {}

    def register(self, name: str, A: torch.Tensor):
        try:
            U, _, _ = torch.linalg.svd(A.float(), full_matrices=False)
            k = min(A.shape[0], A.shape[1])
            self.prev_subspaces[name] = U[:, :k]
        except Exception:
            pass

    def loss(self, named_params) -> torch.Tensor:
        if not self.prev_subspaces:
            return torch.tensor(0.0)
        total = 0.0
        for name, param in named_params:
            if "lora_A" not in name or param.dim() != 2:
                continue
            try:
                U, _, _ = torch.linalg.svd(param.float(), full_matrices=False)
                k = min(param.shape[0], param.shape[1])
                U_new = U[:, :k]
                if name in self.prev_subspaces:
                    total += torch.norm(U_new.T @ self.prev_subspaces[name], "fro") ** 2
            except Exception:
                pass
        return self.lam * total


# ═══════════════════════════════════════════════════════════════════════
# EMBEDDING LOADER (pre-extracted .npz files)
# ═══════════════════════════════════════════════════════════════════════

def load_npz(emb_dir: Path, benchmark: str, task: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """Load embeddings and labels from .npz file.

    Returns: (embeddings float32, labels object array, d_model)
    """
    path = emb_dir / benchmark / task / "train.npz"
    if not path.exists():
        raise FileNotFoundError(f"Embedding not found: {path}")
    data = np.load(str(path), allow_pickle=True)
    emb = data["embeddings"].astype(np.float32)
    lbl = data["labels"]
    return emb, lbl, emb.shape[1]


def get_num_classes(labels: np.ndarray) -> int:
    """Infer number of classes from label array."""
    unique = np.unique(labels)
    return len(unique)


def encode_labels(labels: np.ndarray) -> np.ndarray:
    """Map string labels to integer class indices."""
    label_map = {v: i for i, v in enumerate(np.unique(labels))}
    return np.array([label_map[l] for l in labels])


# ═══════════════════════════════════════════════════════════════════════
# EMBEDDING DATASET & MODEL
# ═══════════════════════════════════════════════════════════════════════

class EmbDataset(Dataset):
    """Embedding dataset for CL training."""

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.X = torch.from_numpy(embeddings)
        self.y = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return {"x": self.X[i], "y": self.y[i]}


class EmbLoRAClassifier(nn.Module):
    """
    Linear classifier with LoRA on top of frozen embeddings.
    Architecture: frozen_embed → (frozen linear) → LoRA classifier head → output
    Since embeddings are pre-extracted, we only LoRA-ize the classifier head.
    """

    def __init__(self, d_model: int, num_classes: int, rank: int = 8, alpha: int = 32):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        # Frozen linear baseline
        self.fc = nn.Linear(d_model, num_classes, bias=True)
        for p in self.fc.parameters():
            p.requires_grad = False
        # LoRA on the classifier
        self.lora_fc = LoRALinear(d_model, num_classes, rank=rank, alpha=alpha)
        # Collect LoRA params
        self.lora_names = ["lora_fc.lora_A", "lora_fc.lora_B"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x) + self.lora_fc(x)

    def lora_modules(self) -> Dict[str, LoRALinear]:
        return {"lora_fc": self.lora_fc}

    def named_lora_params(self):
        return [(n, p) for n, p in self.named_parameters() if "lora_" in n]


# ═══════════════════════════════════════════════════════════════════════
# EMBEDDING TRAINER (fast, pre-extracted)
# ═══════════════════════════════════════════════════════════════════════

class EmbTrainer:
    """
    Fast trainer using pre-extracted embeddings.
    Loads from test_embeddings/TestBackbone/{benchmark}/{task}/train.npz
    Trains LoRA-ized linear classifier on frozen embeddings.
    Time: ~2-5 min per task (vs 20-40 min with full model)
    """

    def __init__(self, emb_dir: Path, benchmark: str, method: str, tasks: List[str],
                 rank=8, alpha=32, lr=1e-3, epochs=10,
                 batch_size=64, lambda_fsr=0.1, lambda_taa=0.05, lambda_sgr=0.1,
                 gpm_threshold=0.99, gpm_steps=200,
                 output_dir: Path = Path("results_fgcl")):
        self.emb_dir = Path(emb_dir)
        self.benchmark = benchmark
        self.method = method
        self.tasks = tasks
        self.rank = rank
        self.alpha = alpha
        self.lr = lr
        self.epochs = epochs
        self.bs = batch_size
        self.lam_fsr = lambda_fsr
        self.lam_taa = lambda_taa
        self.lam_sgr = lambda_sgr
        self.gpm_thresh = gpm_threshold
        self.gpm_steps = gpm_steps
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # CL components
        self.gpm = GPM(threshold=gpm_threshold)
        self.fsr = FSR(lam=lambda_fsr)
        self.taa = TAA(lam=lambda_taa)
        self.sgr = SGR(lam=lambda_sgr)

        # Load embeddings to determine d_model and num_classes
        self.embs: Dict[str, np.ndarray] = {}
        self.lbls: Dict[str, np.ndarray] = {}
        self.num_classes: Dict[str, int] = {}
        self.d_model = None
        for task in tasks:
            emb, lbl, d = load_npz(self.emb_dir, benchmark, task)
            self.embs[task] = emb
            self.lbls[task] = encode_labels(lbl)
            self.num_classes[task] = get_num_classes(lbl)
            if self.d_model is None:
                self.d_model = d

        self.ckpts: List[Path] = []

    def load_task_data(self, task: str) -> Tuple[np.ndarray, np.ndarray]:
        return self.embs[task], self.lbls[task]

    def train_task(self, task_id: int, task: str) -> Dict:
        log.info(f"\n  [Task {task_id+1}/{len(self.tasks)}] {task} ({self.method})")

        # Load embeddings
        emb, lbl = self.load_task_data(task)
        n_cls = self.num_classes[task]
        log.info(f"    {len(emb)} samples, d={self.d_model}, {n_cls} classes")

        # Build model
        model = EmbLoRAClassifier(self.d_model, n_cls, rank=self.rank, alpha=self.alpha)
        model.to(DEVICE)

        # Load previous checkpoints
        for prev_id in range(task_id):
            prev_dir = self.ckpts[prev_id]
            with open(prev_dir / "lora_weights.pt", "rb") as f:
                prev_state = pickle.load(f)
            for name, sd in prev_state.items():
                if hasattr(model, name):
                    getattr(model, name).lora_A.data.copy_(sd["lora_A"].to(DEVICE))
                    getattr(model, name).lora_B.data.copy_(sd["lora_B"].to(DEVICE))
                    getattr(model, name).lora_A.requires_grad_(False)
                    getattr(model, name).lora_B.requires_grad_(False)

            if self.method in ("gainlora", "inflora"):
                gp = prev_dir / "gpm_state.pt"
                if gp.exists():
                    self.gpm.load_state(torch.load(gp, map_location="cpu"))
            if self.method.startswith("fgcl_") and self.method != "fgcl_sgr":
                fs = prev_dir / "fsr_state.pt"
                if fs.exists():
                    self.fsr.load_state(torch.load(fs, map_location="cpu"))
            if self.method == "fgcl_taa":
                tv = prev_dir / "task_vectors.pt"
                if tv.exists():
                    self.taa.load_state({"task_vectors": torch.load(tv, map_location="cpu")})
            if self.method == "fgcl_sgr":
                sg = prev_dir / "sgr_state.pt"
                if sg.exists():
                    self.sgr.prev_subspaces = torch.load(sg, map_location="cpu").get("subspaces", {})

        # GPM accumulation (on embedding activations)
        if self.method in ("gainlora", "inflora") and self.gpm_steps > 0:
            self._accumulate_gpm(model, task)

        # Optimizer
        named_lora = model.named_lora_params()
        if self.method == "fgcl_kfng":
            kf_opt = KFFNGOptimizer([p for _, p in named_lora], lr=self.lr)
            optimizer = kf_opt
        else:
            optimizer = torch.optim.AdamW([p for _, p in named_lora], lr=self.lr)

        # Data
        ds = EmbDataset(emb, lbl)
        loader = DataLoader(ds, batch_size=self.bs, shuffle=True)

        # Training
        model.train()
        epoch_losses = []

        for epoch in range(self.epochs):
            el, ns = 0.0, 0
            for batch in loader:
                x = batch["x"].to(DEVICE)
                y = batch["y"].to(DEVICE)

                optimizer.zero_grad()

                logits = model(x)
                loss_ce = F.cross_entropy(logits, y)

                # ── Build total loss with ALL components ─────────────────────────
                total_loss = loss_ce

                # SGR: pure Python computation, add as Python scalar
                if self.method == "fgcl_sgr":
                    sl = self.sgr.loss(model.named_lora_params())
                    total_loss = total_loss + sl.to(DEVICE)

                # FSR: differentiable projector loss (computed before backward)
                if self.method.startswith("fgcl_") and self.method not in ("fgcl_sgr", "fgcl_kfng") and self.fsr.bases:
                    named_p = model.named_lora_params()
                    fsr_l = self._fsr_differentiable_loss(named_p)
                    total_loss = total_loss + fsr_l

                # TAA: differentiable gradient-alignment loss
                if self.method == "fgcl_taa" and self.taa.task_vectors:
                    named_p = model.named_lora_params()
                    taa_l = self._taa_differentiable_loss(named_p, model.lora_modules())
                    total_loss = total_loss + taa_l

                # ── Single backward pass ─────────────────────────────────────────
                total_loss.backward()

                # ── Capture grad_dict for GPM / KF-FNG ──────────────────────────
                grad_dict = {n: p.grad.detach().clone()
                             for n, p in model.named_lora_params() if p.grad is not None}

                # GPM projection: modify grads in-place
                if self.method in ("gainlora", "inflora") and self.gpm.bases:
                    proj = self.gpm.project(model.named_lora_params())
                    for n, p in model.named_lora_params():
                        if n in proj and p.grad is not None:
                            p.grad.copy_(proj[n])

                clip_grad_norm_(model.parameters(), max_norm=1.0)

                # KF-FNG: update Fisher then step
                if self.method == "fgcl_kfng":
                    kf_opt.update_fisher(model.named_lora_params(), grad_dict)
                    kf_opt.step()
                else:
                    optimizer.step()

                el += loss_ce.item()
                ns += 1

            avg = el / max(ns, 1)
            epoch_losses.append(avg)
            if (epoch + 1) % 2 == 0 or epoch == self.epochs - 1:
                log.info(f"    Epoch {epoch+1} loss: {avg:.4f}")

        model.eval()

        # Post-task accumulation
        self._post_accumulate(model, task_id, task)

        # Save checkpoint
        self._save_ckpt(model, task_id, task)

        # Eval
        eval_res = self._eval_task(model, task_id)
        for tname, acc in eval_res.items():
            log.info(f"    Eval [{tname}]: {acc:.4f}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

        return {"task_id": task_id, "task_name": task,
                "epoch_losses": epoch_losses, "eval": eval_res}

    def _fsr_differentiable_loss(self, named_params) -> torch.Tensor:
        """
        Differentiable FSR loss:
          L_FSR = λ · ||P_{<t} · ∇L_t||² / ||∇L_t||²
        Both projector and norm are computed in the graph → single backward.
        """
        if not self.fsr.bases:
            return torch.tensor(0.0, device=DEVICE)

        total_sq = torch.tensor(0.0, device=DEVICE)
        total_norm = torch.tensor(0.0, device=DEVICE)
        for name, param in named_params:
            if not hasattr(param, 'grad') or param.grad is None:
                continue
            g = param.grad.reshape(-1)
            U = self.fsr.bases.get(name)
            if U is None or U.numel() == 0:
                continue
            U = U.to(g)
            proj_sq = (U.T @ g).pow(2).sum()
            total_sq += proj_sq
            total_norm += g.pow(2).sum()

        total_norm = total_norm.detach() + 1e-8  # prevent double-backprop
        return (self.lam_fsr * total_sq / total_norm.detach())

    def _taa_differentiable_loss(self, named_params, lora_modules) -> torch.Tensor:
        """
        Differentiable TAA loss:
          L_TAA = μ · Σ_{s<t} w_s · ⟨∇L_t, τ_s⟩² / ||∇L_t||²
        Uses grad from current backward pass (must be called AFTER .backward()).
        """
        if not self.taa.task_vectors:
            return torch.tensor(0.0, device=DEVICE)

        # Collect current grads
        grads = {}
        for name, param in named_params:
            if hasattr(param, 'grad') and param.grad is not None:
                grads[name] = param.grad.reshape(-1)

        if not grads:
            return torch.tensor(0.0, device=DEVICE)

        # Importance weights
        mags = {}
        total_mag = 0.0
        for tid, tv in self.taa.task_vectors.items():
            mag = sum((v ** 2).sum().item() for v in tv.values())
            mags[tid] = math.sqrt(mag + 1e-8)
            total_mag += mag
        total_mag += 1e-8

        total_align = 0.0
        for name, g in grads.items():
            for tid, tv in self.taa.task_vectors.items():
                if name not in tv:
                    continue
                tau = tv[name].reshape(-1).to(g)
                mn = min(len(g), len(tau))
                dot_sq = (g[:mn] * tau[:mn]).sum().pow(2)
                w = (mags[tid] ** 2 / total_mag)
                total_align += w * dot_sq

        g_norm = sum(g.pow(2).sum() for g in grads.values()).detach() + 1e-8
        return (self.lam_taa * total_align / g_norm.detach())

    def _accumulate_gpm(self, model: EmbLoRAClassifier, task: str):
        """Accumulate activation covariances for GPM."""
        emb, lbl = self.load_task_data(task)
        n = min(len(emb), self.gpm_steps * self.bs)
        ds = EmbDataset(emb[:n], lbl[:n])
        loader = DataLoader(ds, batch_size=self.bs, shuffle=False)
        model.eval()
        steps = 0
        with torch.no_grad():
            for batch in loader:
                if steps >= self.gpm_steps:
                    break
                x = batch["x"].to(DEVICE)
                _ = model.fc(x) + model.lora_fc(x)
                # Accumulate input activations for GPM
                self.gpm.accumulate("fc.weight", x)
                steps += 1
        self.gpm.compute_bases(min_var=0.99)
        log.info(f"    GPM: {len(self.gpm.bases)} bases")

    def _post_accumulate(self, model: EmbLoRAClassifier, task_id: int, task: str):
        # FSR
        if self.method.startswith("fgcl_") and self.method != "fgcl_sgr":
            emb, lbl = self.load_task_data(task)
            ds = EmbDataset(emb[:500], lbl[:500])
            loader = DataLoader(ds, batch_size=self.bs)
            fisher_acc: Dict[str, torch.Tensor] = {}
            for batch in loader:
                model.zero_grad()
                x, y = batch["x"].to(DEVICE), batch["y"].to(DEVICE)
                F.cross_entropy(model(x), y).backward()
                gd = {n: p.grad.detach().clone()
                      for n, p in model.named_lora_params() if p.grad is not None}
                if gd:
                    fm = self.fsr.compute_fisher(model.named_lora_params(), gd)
                    for n, fv in fm.items():
                        fisher_acc[n] = fisher_acc.get(n, torch.zeros_like(fv)) + fv
                model.zero_grad()
            for n in fisher_acc:
                fisher_acc[n] = fisher_acc[n] / max(len(loader), 1)
            if fisher_acc:
                self.fsr.update_subspace(fisher_acc)
            log.info(f"    FSR: {len(self.fsr.bases)} bases")

        # TAA
        if self.method == "fgcl_taa":
            self.taa.register(task_id, model.lora_modules())
            log.info(f"    TAA: {len(self.taa.task_vectors)} task vectors")

        # SGR
        if self.method == "fgcl_sgr":
            for name, lm in model.lora_modules().items():
                self.sgr.register(f"{name}.lora_A", lm.lora_A.data)
            log.info(f"    SGR: {len(self.sgr.prev_subspaces)} subspaces")

    def _save_ckpt(self, model: EmbLoRAClassifier, task_id: int, task: str):
        d = self.output_dir / f"task_{task_id:02d}_{task}"
        d.mkdir(parents=True, exist_ok=True)
        state = {n: {"lora_A": getattr(model, n).lora_A.cpu(),
                     "lora_B": getattr(model, n).lora_B.cpu()}
                 for n in model.lora_names if hasattr(model, n)}
        with open(d / "lora_weights.pt", "wb") as f:
            pickle.dump(state, f)
        if self.method in ("gainlora", "inflora"):
            torch.save(self.gpm.state_dict(), d / "gpm_state.pt")
        if self.fsr.bases:
            torch.save(self.fsr.state_dict(), d / "fsr_state.pt")
        if self.taa.task_vectors:
            torch.save(self.taa.task_vectors, d / "task_vectors.pt")
        if self.sgr.prev_subspaces:
            torch.save({"subspaces": self.sgr.prev_subspaces}, d / "sgr_state.pt")
        self.ckpts.append(d)

    def _eval_task(self, model: EmbLoRAClassifier, task_id: int) -> Dict[str, float]:
        results = {}
        for tid in range(task_id + 1):
            tname = self.tasks[tid]
            emb, lbl = self.load_task_data(tname)
            ds = EmbDataset(emb, lbl)
            loader = DataLoader(ds, batch_size=self.bs)
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for batch in loader:
                    x, y = batch["x"].to(DEVICE), batch["y"].to(DEVICE)
                    preds = model(x).argmax(dim=-1)
                    correct += (preds == y).sum().item()
                    total += len(y)
            results[tname] = correct / max(total, 1)
        return results

    def run(self) -> Dict:
        log.info(f"\n{'='*60}")
        log.info(f"  METHOD: {self.method}")
        log.info(f"  TASKS:  {' → '.join(self.tasks)}")
        log.info(f"  EPOCHS: {self.epochs}  |  RANK: {self.rank}")
        log.info(f"{'='*60}")
        t0 = time.time()
        all_results = []
        for tid, tname in enumerate(self.tasks):
            res = self.train_task(tid, tname)
            all_results.append(res)
        elapsed = time.time() - t0

        # Metrics
        n = len(all_results)
        mat = np.zeros((n, n))
        for i, res in enumerate(all_results):
            for j, (tname, acc) in enumerate(res["eval"].items()):
                mat[i, j] = acc
        ap = float(np.mean(mat[-1, :]))
        ft = float(np.mean([mat[t, t] for t in range(n)]))
        bwt_scores = [mat[-1, t] - mat[t, t] for t in range(n - 1)]
        bwt = float(np.mean(bwt_scores)) if bwt_scores else 0.0

        log.info(f"\n{'='*60}")
        log.info(f"  RESULTS ({self.method})")
        log.info(f"  AP  = {ap:.4f}")
        log.info(f"  FT  = {ft:.4f}")
        log.info(f"  BWT = {bwt:.4f}")
        log.info(f"  TIME = {elapsed:.0f}s ({elapsed/60:.1f}m)")
        log.info(f"{'='*60}")

        return {
            "method": self.method, "tasks": self.tasks,
            "epochs": self.epochs, "rank": self.rank,
            "elapsed_seconds": elapsed,
            "AP": ap, "FT": ft, "BWT": bwt,
            "score_matrix": mat.tolist(),
            "per_task": all_results,
        }


# ═══════════════════════════════════════════════════════════════════════
# PHASE LAUNCHER
# ═══════════════════════════════════════════════════════════════════════

def run_phase(
    phase: str, benchmark: str, model_dir: Path,
    tasks: List[str], output_dir: Path,
) -> Dict[str, Dict]:
    """Run a single phase for one model+benchmark combo."""
    methods = ALL_PHASE_METHODS[phase]
    epochs = DEFAULT_EPOCHS[phase]

    log.info(f"\n{'#'*70}")
    log.info(f"# PHASE {phase}: {PHASE_DESCS[phase]}")
    log.info(f"# MODEL: {model_dir}")
    log.info(f"# BENCHMARK: {benchmark}")
    log.info(f"# TASKS: {' → '.join(tasks)}")
    log.info(f"# METHODS: {methods}")
    log.info(f"# EPOCHS: {epochs}  RANK: {DEFAULT_LORA_RANK}  LR: {DEFAULT_LR}")
    log.info(f"{'#'*70}")

    phase_dir = output_dir / f"phase_{phase}"
    phase_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for method in methods:
        t0 = time.time()
        trainer = EmbTrainer(
            emb_dir=model_dir,
            benchmark=benchmark,
            method=method,
            tasks=tasks,
            output_dir=phase_dir / method,
            rank=DEFAULT_LORA_RANK,
            alpha=DEFAULT_LORA_ALPHA,
            lr=DEFAULT_LR,
            epochs=epochs,
            batch_size=DEFAULT_BATCH,
            lambda_fsr=DEFAULT_LAMBDA_FSR,
            lambda_taa=DEFAULT_LAMBDA_TAA,
            lambda_sgr=DEFAULT_LAMBDA_SGR,
            gpm_threshold=0.99,
            gpm_steps=DEFAULT_GPM_STEPS,
        )
        result = trainer.run()
        result["elapsed_seconds"] = time.time() - t0
        results[method] = result

        with open(phase_dir / method / "result.json", "w") as f:
            json.dump(result, f, indent=2)

    # Summary
    log.info(f"\n{'═'*70}")
    log.info(f"PHASE {phase} RESULTS — {benchmark}")
    log.info(f"  {'Method':<18} {'AP':>8} {'FT':>8} {'BWT':>10} {'Time':>8}")
    log.info(f"  {'─'*60}")
    for m, r in results.items():
        t = r.get("elapsed_seconds", 0)
        log.info(f"  {m:<18} {r['AP']:>8.4f} {r['FT']:>8.4f} "
                f"{r['BWT']:>10.4f}  {t:>6.0f}s")

    combined = {
        "phase": phase, "benchmark": benchmark, "tasks": tasks,
        "methods": methods,
        "description": PHASE_DESCS[phase],
        "results": results,
    }
    with open(phase_dir / "all_results.json", "w") as f:
        json.dump(combined, f, indent=2)
    log.info(f"\n  Saved: {phase_dir / 'all_results.json'}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def discover_models(root: Path = Path("test_embeddings")) -> List[Path]:
    """Auto-discover model dirs: test_embeddings/{model_name}/"""
    if not root.exists():
        return []
    return sorted([d for d in root.iterdir() if d.is_dir()])


def discover_benchmarks(model_dir: Path) -> List[str]:
    """Auto-discover benchmarks under a model dir."""
    return sorted([d.name for d in model_dir.iterdir() if d.is_dir()])


def parse_args():
    p = argparse.ArgumentParser(
        description="FGCL Experiment Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python exp_fgcl.py                                          # ALL
  python exp_fgcl.py --phase T1                               # T1 only
  python exp_fgcl.py --model test_embeddings/TestBackbone/    # specific model
  python exp_fgcl.py --benchmark SuperNI                      # specific benchmark
  python exp_fgcl.py --model test_embeddings/Llama-2-7b-hf/ --benchmark SuperNI
""",
    )
    p.add_argument("--phase", nargs="+", default=None,
                   help="Phase(s) to run: T1 T2 T3 T4. Omit = all.")
    p.add_argument("--model", nargs="+", default=None,
                   help="Path(s) to model embedding dir. Omit = auto-discover all.")
    p.add_argument("--benchmark", nargs="+", default=None,
                   help="Benchmark name(s): Long_Sequence, SuperNI. Omit = auto-discover all.")
    p.add_argument("--output_dir", default="results_fgcl")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Resolve phases ──────────────────────────────────────────────────
    phases = args.phase or ALL_PHASES

    # ── Resolve models ──────────────────────────────────────────────────
    if args.model:
        model_dirs = [Path(m) for m in args.model]
        for md in model_dirs:
            if not md.exists():
                log.error(f"Model dir not found: {md}")
                sys.exit(1)
    else:
        model_dirs = discover_models()
        if not model_dirs:
            log.error("No model dirs found under test_embeddings/. "
                      "Use --model to specify path.")
            sys.exit(1)

    # ── Build run matrix: (phase, model_dir, benchmark, tasks) ──────────
    # Tasks are DISCOVERED from filesystem: {model_dir}/{benchmark}/{task}/train.npz
    runs = []
    for model_dir in model_dirs:
        if args.benchmark:
            benchmarks = args.benchmark
        else:
            benchmarks = discover_benchmarks(model_dir)
            if not benchmarks:
                log.warning(f"No benchmarks found in {model_dir}, skipping.")
                continue

        for bm in benchmarks:
            bm_path = model_dir / bm
            if not bm_path.exists():
                log.warning(f"Benchmark dir not found: {bm_path}, skipping.")
                continue

            # Discover all task dirs under this benchmark
            avail_tasks = sorted([
                d.name for d in bm_path.iterdir()
                if d.is_dir() and (d / "train.npz").exists()
            ])
            if not avail_tasks:
                log.warning(f"No task .npz files found in {bm_path}, skipping.")
                continue

            for phase in phases:
                n = PHASE_TASK_COUNTS[phase]
                tasks = avail_tasks[:n] if n else avail_tasks  # T4=None → all
                runs.append((phase, model_dir, bm, tasks))

    if not runs:
        log.error("No valid (phase, model, benchmark, tasks) combos found.")
        sys.exit(1)

    log.info(f"Run matrix: {len(runs)} combos "
             f"({len(phases)} phases × {len(model_dirs)} models × benchmarks)")
    for phase, md, bm, tasks in runs:
        log.info(f"  {phase} | {md.name} | {bm} | {len(tasks)} tasks")

    output_root = Path(args.output_dir)
    all_results = {}

    for phase, model_dir, benchmark, tasks in runs:
        key = f"{model_dir.name}/{benchmark}/{phase}"
        out_dir = output_root / model_dir.name / benchmark
        results = run_phase(
            phase=phase,
            benchmark=benchmark,
            model_dir=model_dir,
            tasks=tasks,
            output_dir=out_dir,
        )
        all_results[key] = results

    # ── Grand summary ───────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"GRAND SUMMARY — {len(runs)} experiments")
    print(f"{'═'*70}")
    for key, results in all_results.items():
        print(f"\n  {key}")
        print(f"  {'Method':<18} {'AP':>8} {'FT':>8} {'BWT':>10}")
        print(f"  {'─'*50}")
        for m, r in results.items():
            print(f"  {m:<18} {r['AP']:>8.4f} {r['FT']:>8.4f} {r['BWT']:>10.4f}")
    print(f"\n{'═'*70}")
    print(f"Output: {output_root}/")


if __name__ == "__main__":
    main()
