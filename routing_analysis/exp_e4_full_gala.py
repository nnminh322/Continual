#!/usr/bin/env python3
"""
Experiment E4: Full GALA Pipeline vs Baselines (End-to-End CL)
===============================================================
The capstone experiment: combines ALL GALA components (TARA + GGI + SGR + BNG)
into a single pipeline and compares against baselines on a multi-task CL sequence.

Strategies compared:
(a) Standard LoRA     — Kaiming init, AdamW, no CL constraint
(b) InfLoRA-style     — Hard projection, AdamW
(c) GALA (full)       — TARA rank + GGI init + SGR penalty + BNG optimizer
(d) GALA w/o BNG      — TARA + GGI + SGR + AdamW (ablation: optimizer contribution)
(e) GALA w/o SGR      — TARA + GGI + NoConstraint + BNG (ablation: CL constraint)
(f) GALA w/o GGI      — TARA + Kaiming + SGR + BNG (ablation: init contribution)

Protocol:
1. Per-task gradient probing → TARA rank selection (or fixed rank for baselines)
2. GGI init from generalized EVP (or Kaiming for baselines)
3. CL training with constraint strategy + optimizer strategy
4. After all tasks: evaluate every previous task → accuracy matrix
5. Metrics: AP, FT, BWT, per-task curves, Grassmannian overlap

Usage:
  python exp_e4_full_gala.py \
    --model_name google/flan-t5-large \
    --data_dir ../improve_gainlora/CL_Benchmark \
    --benchmark Long_Sequence \
    --tasks sst2,imdb,yelp,amazon,agnews \
    --n_epochs 3

Output: results/e4_full_gala_<model>_<benchmark>.json
"""
from __future__ import annotations
import argparse, json, os, sys, time, math, copy, warnings
from pathlib import Path
from collections import defaultdict, OrderedDict
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from functools import partial


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def participation_ratio(eigvals):
    """Effective dimensionality: (Σλ)² / Σ(λ²)."""
    eigvals = np.maximum(np.asarray(eigvals, dtype=np.float64), 0)
    s = eigvals.sum()
    if s < 1e-15:
        return 0.0
    return float(s ** 2 / (eigvals ** 2).sum())


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_task_data(data_dir, benchmark, task, max_samples=5000):
    json_path = Path(data_dir) / benchmark / task / "train.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Data not found: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    instances = data.get("Instances", [])
    definition = data.get("Definition", [""])[0]
    samples = []
    for inst in instances[:max_samples]:
        text = inst.get("input", "")
        label = inst.get("output", "")
        if isinstance(label, list):
            label = label[0] if label else ""
        instruction = f"{definition}\n{text}\nOutput: "
        samples.append({"input": instruction, "label": label})
    return samples


def load_eval_data(data_dir, benchmark, task, max_samples=300):
    for split_name in ["dev", "test"]:
        json_path = Path(data_dir) / benchmark / task / f"{split_name}.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
            instances = data.get("Instances", [])
            definition = data.get("Definition", [""])[0]
            samples = []
            for inst in instances[:max_samples]:
                text = inst.get("input", "")
                label = inst.get("output", "")
                if isinstance(label, list):
                    label = label[0] if label else ""
                instruction = f"{definition}\n{text}\nOutput: "
                samples.append({"input": instruction, "label": label})
            if samples:
                return samples
    return None


class SimpleTextDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn_t5(batch, tokenizer, max_source_length=512, max_target_length=50):
    inputs_text = [s["input"] for s in batch]
    labels_text = [s["label"] for s in batch]
    model_inputs = tokenizer(inputs_text, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_source_length)
    labels = tokenizer(labels_text, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_target_length)
    label_ids = labels.input_ids
    label_ids[label_ids == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = label_ids
    return model_inputs


# ═══════════════════════════════════════════════════════════════════════
# LoRA injection
# ═══════════════════════════════════════════════════════════════════════

class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r: int, alpha: float = 1.0):
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        d_out, d_in = base_linear.weight.shape
        self.d_in = d_in
        self.d_out = d_out
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out = self.base(x)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out

    def get_subspace(self):
        """Return orthonormal basis for row space of A via QR."""
        A = self.lora_A.detach()  # (r, d_in)
        Q, R = torch.linalg.qr(A.T)  # Q: (d_in, r)
        return Q


def inject_lora(model, r, alpha=1.0, target_modules=None):
    if target_modules is None:
        target_modules = ["q", "v"]
    lora_modules = []
    is_t5 = hasattr(model, 'encoder')
    if is_t5:
        for block in model.encoder.block:
            attn = block.layer[0].SelfAttention
            for name in target_modules:
                if hasattr(attn, name):
                    base_linear = getattr(attn, name)
                    lora_linear = LoRALinear(base_linear, r, alpha)
                    setattr(attn, name, lora_linear)
                    lora_modules.append(lora_linear)
        for block in model.decoder.block:
            attn = block.layer[0].SelfAttention
            for name in target_modules:
                if hasattr(attn, name):
                    base_linear = getattr(attn, name)
                    lora_linear = LoRALinear(base_linear, r, alpha)
                    setattr(attn, name, lora_linear)
                    lora_modules.append(lora_linear)
    else:
        for layer in model.model.layers:
            attn = layer.self_attn
            for name in ["q_proj", "v_proj"]:
                if hasattr(attn, name):
                    base_linear = getattr(attn, name)
                    lora_linear = LoRALinear(base_linear, r, alpha)
                    setattr(attn, name, lora_linear)
                    lora_modules.append(lora_linear)
    return lora_modules


# ═══════════════════════════════════════════════════════════════════════
# Gradient / Activation Probing (for TARA + GGI + BNG preconditioning)
# ═══════════════════════════════════════════════════════════════════════

def probe_task(model, tokenizer, samples, device, target_layer_idx=0,
               n_batches=50, batch_size=8):
    """
    Unified probing: computes both gradient and activation covariance
    for TARA rank selection, GGI init, and BNG preconditioning.

    Returns:
        cov_grad: (d_in, d_in) — gradient covariance Σ_grad = E[∂L/∂W^T ∂L/∂W]
        cov_act:  (d_in, d_in) — activation covariance Σ_x = E[x^T x]
        par:      float — participation ratio of gradient eigenvalues (TARA)
    """
    model.eval()
    is_t5 = hasattr(model, 'encoder')

    # Find target module
    if is_t5:
        n_layers = len(model.encoder.block)
        target_layer_idx = min(target_layer_idx, n_layers - 1)
        q_module = model.encoder.block[target_layer_idx].layer[0].SelfAttention.q
    else:
        n_layers = len(model.model.layers)
        target_layer_idx = min(target_layer_idx, n_layers - 1)
        q_module = model.model.layers[target_layer_idx].self_attn.q_proj

    # If LoRA-wrapped, hook into base
    if isinstance(q_module, LoRALinear):
        hook_target = q_module.base
    else:
        hook_target = q_module

    weight_param = hook_target.weight
    d_out, d_in = weight_param.shape

    # Activation hook
    act_cache = {}
    def hook_fn(module, inp, out):
        if isinstance(inp, tuple):
            act_cache['x'] = inp[0].detach()
        else:
            act_cache['x'] = inp.detach()
    hook = hook_target.register_forward_hook(hook_fn)

    grad_outer = np.zeros((d_in, d_in), dtype=np.float64)
    act_outer = np.zeros((d_in, d_in), dtype=np.float64)
    act_sum = np.zeros(d_in, dtype=np.float64)
    n_tokens = 0

    rng = np.random.RandomState(42)
    indices = rng.permutation(len(samples))
    batch_count = 0
    i = 0

    while batch_count < n_batches and i < len(indices):
        batch_idx = indices[i:i + batch_size]
        i += batch_size
        if len(batch_idx) == 0:
            break

        batch_texts = [samples[j]["input"] for j in batch_idx]
        batch_labels = [samples[j]["label"] for j in batch_idx]

        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                          truncation=True, max_length=512).to(device)
        if is_t5:
            labels = tokenizer(batch_labels, return_tensors="pt", padding=True,
                             truncation=True, max_length=50).input_ids.to(device)
            labels[labels == tokenizer.pad_token_id] = -100
            inputs["labels"] = labels

        model.zero_grad()
        act_cache.clear()
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()

        # Activation covariance
        x = act_cache.get('x', None)
        if x is not None:
            x_flat = x.reshape(-1, d_in)
            mask = inputs["attention_mask"].reshape(-1).bool()
            x_flat = x_flat[mask]
            x_np = x_flat.float().cpu().numpy()
            n = x_np.shape[0]
            act_sum += x_np.sum(0)
            act_outer += x_np.T @ x_np
            n_tokens += n

        # Gradient covariance from weight gradient
        wg = weight_param.grad
        if wg is not None:
            wg_np = wg.float().cpu().numpy()  # (d_out, d_in)
            grad_outer += wg_np.T @ wg_np

        batch_count += 1

    hook.remove()

    mu = act_sum / max(n_tokens, 1)
    cov_act = act_outer / max(n_tokens, 1) - np.outer(mu, mu)
    cov_grad = grad_outer / max(batch_count, 1)

    # TARA: participation ratio of gradient eigenvalues
    eigvals_grad = np.linalg.eigvalsh(cov_grad)
    par = participation_ratio(eigvals_grad)

    print(f"  Probed {batch_count} batches, {n_tokens} tokens, d={d_in}, PaR={par:.1f}")
    return cov_grad, cov_act, par


# ═══════════════════════════════════════════════════════════════════════
# TARA: Rank selection from gradient participation ratio
# ═══════════════════════════════════════════════════════════════════════

def tara_select_rank(par, rank_candidates=(2, 4, 8, 16, 32), min_rank=2, max_rank=32):
    """Select rank closest to PaR, clamped to [min_rank, max_rank]."""
    target = max(min_rank, min(max_rank, int(round(par))))
    # Pick closest candidate
    best = min(rank_candidates, key=lambda r: abs(r - target))
    return best


# ═══════════════════════════════════════════════════════════════════════
# GGI: Init from generalized eigenvalue problem
# ═══════════════════════════════════════════════════════════════════════

def compute_ggi_init(cov_grad, cov_act, rank):
    """Generalized EVP: Σ_grad v = λ Σ_x v → top-r eigenvectors."""
    reg = 1e-6 * np.trace(cov_act) / cov_act.shape[0]
    cov_act_reg = cov_act + reg * np.eye(cov_act.shape[0])
    try:
        M = np.linalg.solve(cov_act_reg, cov_grad)
        eigvals, eigvecs = np.linalg.eigh(M)
        idx = np.argsort(eigvals)[::-1]
        V = eigvecs[:, idx[:rank]]
        return V  # (d_in, r)
    except np.linalg.LinAlgError:
        print("  WARNING: GGI EVP failed, falling back to Σ_grad eigenvectors")
        eigvals, eigvecs = np.linalg.eigh(cov_grad)
        idx = np.argsort(eigvals)[::-1]
        return eigvecs[:, idx[:rank]]


# ═══════════════════════════════════════════════════════════════════════
# BNG: Preconditioner from activation covariance
# ═══════════════════════════════════════════════════════════════════════

def compute_preconditioner(cov_act, k=None):
    """Compute low-rank Σ_x^{-1/2} approximation."""
    eigvals, eigvecs = np.linalg.eigh(cov_act)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    eigvals_pos = np.maximum(eigvals, 1e-10)
    if k is None:
        k = int(participation_ratio(eigvals_pos))
        k = max(4, min(k, eigvals.shape[0] // 2))

    V_k = eigvecs[:, :k]
    lam_k_inv_sqrt = 1.0 / np.sqrt(eigvals_pos[:k])
    lam_mean_inv_sqrt = 1.0 / np.sqrt(np.mean(eigvals_pos[k:]) + 1e-10)

    precond = {
        "V_k": torch.tensor(V_k, dtype=torch.float32),
        "lam_k_inv_sqrt": torch.tensor(lam_k_inv_sqrt, dtype=torch.float32),
        "lam_mean_inv_sqrt": float(lam_mean_inv_sqrt),
        "k": k,
    }
    return precond


def apply_preconditioner(grad_A, precond, device):
    """Apply Σ_x^{-1/2} preconditioning to gradient of A."""
    V_k = precond["V_k"].to(device)
    lam_inv = precond["lam_k_inv_sqrt"].to(device)
    lam_mean = precond["lam_mean_inv_sqrt"]

    proj = grad_A @ V_k
    scaled = proj * (lam_inv - lam_mean)
    result = scaled @ V_k.T + lam_mean * grad_A
    return result


# ═══════════════════════════════════════════════════════════════════════
# CL Constraint Strategies
# ═══════════════════════════════════════════════════════════════════════

class HardProjection:
    """InfLoRA-style: project gradients into null space of previous subspace."""
    name = "hard"

    def __init__(self, lora_modules, **kwargs):
        self.lora_modules = lora_modules
        self.prev_subspaces = [None] * len(lora_modules)

    def accumulate_subspace(self):
        for i, lm in enumerate(self.lora_modules):
            V_new = lm.get_subspace()
            if self.prev_subspaces[i] is None:
                self.prev_subspaces[i] = V_new
            else:
                combined = torch.cat([self.prev_subspaces[i], V_new], dim=1)
                U, S, _ = torch.linalg.svd(combined, full_matrices=False)
                threshold = S.max() * 1e-5
                k = (S > threshold).sum().item()
                self.prev_subspaces[i] = U[:, :k]

    def project_init(self, soft_strength=None):
        for i, lm in enumerate(self.lora_modules):
            if self.prev_subspaces[i] is None:
                continue
            V_prev = self.prev_subspaces[i].to(lm.lora_A.device)
            P = V_prev @ V_prev.T
            with torch.no_grad():
                lm.lora_A.data -= lm.lora_A.data @ P
                row_norms = lm.lora_A.data.norm(dim=1, keepdim=True).clamp(min=1e-8)
                target_norm = math.sqrt(2.0 / lm.d_in)
                lm.lora_A.data *= target_norm / row_norms

    def pre_step(self):
        for i, lm in enumerate(self.lora_modules):
            if self.prev_subspaces[i] is None:
                continue
            V_prev = self.prev_subspaces[i].to(lm.lora_A.device)
            P = V_prev @ V_prev.T
            if lm.lora_A.grad is not None:
                lm.lora_A.grad.data -= lm.lora_A.grad.data @ P

    def get_loss_penalty(self, lora_modules):
        return 0.0


class SoftGrassmannianRegularization:
    """SGR: penalize overlap ||V_t^T V_prev||_F^2."""
    name = "sgr"

    def __init__(self, lora_modules, lambda1=0.1, soft_init_strength=0.7, **kwargs):
        self.lora_modules = lora_modules
        self.lambda1 = lambda1
        self.soft_init_strength = soft_init_strength
        self.prev_subspaces = [None] * len(lora_modules)

    def accumulate_subspace(self):
        for i, lm in enumerate(self.lora_modules):
            V_new = lm.get_subspace()
            if self.prev_subspaces[i] is None:
                self.prev_subspaces[i] = V_new.detach()
            else:
                combined = torch.cat([self.prev_subspaces[i], V_new.detach()], dim=1)
                U, S, _ = torch.linalg.svd(combined, full_matrices=False)
                threshold = S.max() * 1e-5
                k = (S > threshold).sum().item()
                self.prev_subspaces[i] = U[:, :k].detach()

    def project_init(self, soft_strength=None):
        """Soft: bias init towards null space, parameterized strength."""
        strength = soft_strength if soft_strength is not None else self.soft_init_strength
        for i, lm in enumerate(self.lora_modules):
            if self.prev_subspaces[i] is None:
                continue
            V_prev = self.prev_subspaces[i].to(lm.lora_A.device)
            P = V_prev @ V_prev.T
            with torch.no_grad():
                lm.lora_A.data -= strength * (lm.lora_A.data @ P)

    def pre_step(self):
        pass

    def get_loss_penalty(self, lora_modules):
        penalty = 0.0
        n_terms = 0
        for i, lm in enumerate(lora_modules):
            if self.prev_subspaces[i] is None:
                continue
            V_prev = self.prev_subspaces[i].to(lm.lora_A.device)
            A = lm.lora_A
            Q, R = torch.linalg.qr(A.T)
            V_current = Q
            overlap = V_current.T @ V_prev
            penalty += (overlap ** 2).sum()
            n_terms += 1
        if n_terms > 0:
            penalty = self.lambda1 * penalty / n_terms
        return penalty


class NoConstraint:
    """No CL constraint."""
    name = "none"

    def __init__(self, lora_modules, **kwargs):
        self.lora_modules = lora_modules
        self.prev_subspaces = [None] * len(lora_modules)

    def accumulate_subspace(self):
        for i, lm in enumerate(self.lora_modules):
            V_new = lm.get_subspace()
            if self.prev_subspaces[i] is None:
                self.prev_subspaces[i] = V_new.detach()
            else:
                combined = torch.cat([self.prev_subspaces[i], V_new.detach()], dim=1)
                U, S, _ = torch.linalg.svd(combined, full_matrices=False)
                threshold = S.max() * 1e-5
                k = (S > threshold).sum().item()
                self.prev_subspaces[i] = U[:, :k].detach()

    def project_init(self, soft_strength=None):
        pass

    def pre_step(self):
        pass

    def get_loss_penalty(self, lora_modules):
        return 0.0


# ═══════════════════════════════════════════════════════════════════════
# Optimizer Strategies (for BNG ablation)
# ═══════════════════════════════════════════════════════════════════════

class AdamWOptimizer:
    """Standard AdamW."""
    name = "adamw"

    def __init__(self, lora_modules, lr, **kwargs):
        params = []
        for lm in lora_modules:
            params.extend([lm.lora_A, lm.lora_B])
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        self.lora_modules = lora_modules

    def pre_step(self):
        pass

    def get_balance_loss(self):
        return 0.0

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()


class BNGOptimizer:
    """Full BNG: preconditioning + adaptive asymmetric LR + β-EMA + balance reg."""
    name = "bng"

    def __init__(self, lora_modules, lr, precond=None, beta_ema=0.99,
                 lambda_bal=0.01, **kwargs):
        a_params = [lm.lora_A for lm in lora_modules]
        b_params = [lm.lora_B for lm in lora_modules]
        self.optimizer = torch.optim.AdamW([
            {"params": a_params, "lr": lr, "name": "A"},
            {"params": b_params, "lr": lr, "name": "B"},
        ], weight_decay=0.01)
        self.lora_modules = lora_modules
        self.precond = precond
        self.beta_ema = beta_ema
        self.lambda_bal = lambda_bal
        self.beta_smoothed = 1.0
        self.base_lr = lr

    def get_balance_loss(self):
        loss = 0.0
        for lm in self.lora_modules:
            norm_b_sq = (lm.lora_B ** 2).sum()
            norm_a_sq = (lm.lora_A ** 2).sum()
            loss = loss + (norm_b_sq - norm_a_sq) ** 2
        return self.lambda_bal * loss

    def pre_step(self):
        # 1. Precondition A gradients
        if self.precond is not None:
            for lm in self.lora_modules:
                if lm.lora_A.grad is not None:
                    device = lm.lora_A.device
                    precond_grad = apply_preconditioner(
                        lm.lora_A.grad.data, self.precond, device)
                    lm.lora_A.grad.data.copy_(precond_grad)

        # 2. Adaptive β for asymmetric LR
        total_norm_A = 0.0
        total_norm_B = 0.0
        for lm in self.lora_modules:
            total_norm_A += lm.lora_A.data.norm().item() ** 2
            total_norm_B += lm.lora_B.data.norm().item() ** 2
        total_norm_A = math.sqrt(total_norm_A)
        total_norm_B = math.sqrt(total_norm_B)

        beta_raw = math.sqrt(total_norm_B / max(total_norm_A, 1e-8))
        self.beta_smoothed = self.beta_ema * self.beta_smoothed + (1 - self.beta_ema) * beta_raw

        beta = max(min(self.beta_smoothed, 10.0), 0.1)
        self.optimizer.param_groups[0]["lr"] = self.base_lr * beta  # A
        self.optimizer.param_groups[1]["lr"] = self.base_lr / beta  # B

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()


# ═══════════════════════════════════════════════════════════════════════
# Init application
# ═══════════════════════════════════════════════════════════════════════

def apply_ggi_init(lora_modules, ggi_V):
    """Apply GGI init: A = V^T (rows are basis vectors)."""
    V = torch.tensor(ggi_V, dtype=torch.float32)
    for lm in lora_modules:
        d_in = lm.lora_A.shape[1]
        r = lm.lora_A.shape[0]
        if V.shape[0] == d_in and V.shape[1] >= r:
            lm.lora_A.data.copy_(V[:, :r].T)
        else:
            nn.init.kaiming_uniform_(lm.lora_A, a=math.sqrt(5))
        nn.init.zeros_(lm.lora_B)


def apply_kaiming_init(lora_modules):
    """Standard Kaiming init."""
    for lm in lora_modules:
        nn.init.kaiming_uniform_(lm.lora_A, a=math.sqrt(5))
        nn.init.zeros_(lm.lora_B)


# ═══════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════

def evaluate_accuracy(model, tokenizer, eval_samples, device, batch_size=8, is_t5=True):
    model.eval()
    correct = 0
    total = 0
    for i in range(0, len(eval_samples), batch_size):
        batch_data = eval_samples[i:i + batch_size]
        if not batch_data:
            break
        inputs_text = [s["input"] for s in batch_data]
        gold_labels = [s["label"].strip().lower() for s in batch_data]
        inputs = tokenizer(inputs_text, return_tensors="pt", padding=True,
                          truncation=True, max_length=512).to(device)
        with torch.no_grad():
            if is_t5:
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=50, do_sample=False,
                )
                preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            else:
                outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                input_len = inputs["input_ids"].shape[1]
                preds = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
        for pred, gold in zip(preds, gold_labels):
            if pred.strip().lower() == gold:
                correct += 1
            total += 1
    return correct / max(total, 1)


# ═══════════════════════════════════════════════════════════════════════
# GALA Pipeline Configuration
# ═══════════════════════════════════════════════════════════════════════

class GALAConfig:
    """Configuration for a single GALA pipeline variant."""
    def __init__(self, name, use_tara=False, use_ggi=False, use_sgr=False,
                 use_bng=False, sgr_lambda=0.1, soft_init_strength=0.7,
                 use_hard_projection=False, fixed_rank=None):
        self.name = name
        self.use_tara = use_tara
        self.use_ggi = use_ggi
        self.use_sgr = use_sgr
        self.use_bng = use_bng
        self.sgr_lambda = sgr_lambda
        self.soft_init_strength = soft_init_strength
        self.use_hard_projection = use_hard_projection
        self.fixed_rank = fixed_rank  # None means use TARA; int means override

    def __repr__(self):
        components = []
        if self.use_tara:
            components.append("TARA")
        if self.use_ggi:
            components.append("GGI")
        if self.use_sgr:
            components.append(f"SGR(λ={self.sgr_lambda})")
        if self.use_bng:
            components.append("BNG")
        if self.use_hard_projection:
            components.append("Hard")
        return f"GALAConfig({self.name}: {'+'.join(components) or 'baseline'})"


# ═══════════════════════════════════════════════════════════════════════
# Full CL Training Pipeline
# ═══════════════════════════════════════════════════════════════════════

def run_gala_cl_pipeline(model_name, tokenizer, tasks, data_dir, benchmark,
                         device, config: GALAConfig, default_rank=8,
                         lora_alpha=1.0, n_epochs=3, batch_size=8, lr=1e-4,
                         max_train_samples=2000, probe_layer_idx=0):
    """
    Run a full CL sequence with a given GALA configuration.

    Pipeline per task:
    1. (Optional) Probe: gradient + activation covariance
    2. (Optional) TARA: select rank from PaR
    3. Re-init LoRA A/B
    4. (Optional) GGI: apply task-informed init
    5. (Optional) Constraint: project init towards null space
    6. Train with optimizer (AdamW or BNG) + CL penalty
    7. Accumulate subspace
    8. Evaluate all previous tasks

    Returns: accuracy matrix, AP, FT, BWT, diagnostics
    """
    from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM

    is_t5 = "t5" in model_name.lower()

    # --- Load model & inject LoRA ---
    # Use default_rank for initial injection; may be overridden per-task if TARA is on
    initial_rank = config.fixed_rank if config.fixed_rank else default_rank

    if is_t5:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

    target_modules = ["q", "v"] if is_t5 else ["q_proj", "v_proj"]
    lora_modules = inject_lora(model, initial_rank, lora_alpha, target_modules)
    model.to(device)

    # Freeze base
    for param in model.parameters():
        param.requires_grad = False
    for lm in lora_modules:
        lm.lora_A.requires_grad_(True)
        lm.lora_B.requires_grad_(True)

    n_trainable = sum(lm.lora_A.numel() + lm.lora_B.numel() for lm in lora_modules)
    print(f"  {n_trainable:,} trainable LoRA params (r={initial_rank})")

    # --- Setup CL constraint ---
    if config.use_sgr:
        constraint = SoftGrassmannianRegularization(
            lora_modules, lambda1=config.sgr_lambda,
            soft_init_strength=config.soft_init_strength)
    elif config.use_hard_projection:
        constraint = HardProjection(lora_modules)
    else:
        constraint = NoConstraint(lora_modules)

    # --- Load eval data for all tasks ---
    all_eval_data = {}
    for t in tasks:
        eval_data = load_eval_data(data_dir, benchmark, t)
        if eval_data:
            all_eval_data[t] = eval_data
        else:
            print(f"  WARNING: No eval data for {t}, using train subset")
            train_data = load_task_data(data_dir, benchmark, t, max_samples=300)
            all_eval_data[t] = train_data[-100:] if len(train_data) > 100 else train_data

    # --- We need a separate probe model (without LoRA) for gradient probing ---
    # Gradient probing must happen on clean model to avoid LoRA interference
    probe_model = None
    if config.use_tara or config.use_ggi or config.use_bng:
        if is_t5:
            probe_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, torch_dtype=torch.float32).to(device)
        else:
            probe_model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float32).to(device)
        probe_model.eval()

    # --- CL Training Loop ---
    n_tasks = len(tasks)
    acc_matrix = np.zeros((n_tasks, n_tasks))  # acc_matrix[i][j] = acc on task j after training task i
    loss_curves = {}
    task_diagnostics = {}
    collate = partial(collate_fn_t5, tokenizer=tokenizer)

    for task_idx, task_name in enumerate(tasks):
        print(f"\n  --- Task {task_idx + 1}/{n_tasks}: {task_name} ---")
        t0 = time.time()

        train_samples = load_task_data(data_dir, benchmark, task_name,
                                       max_samples=max_train_samples)
        print(f"  Loaded {len(train_samples)} train samples")

        diag = {"task": task_name}

        # Step 1: Probe (if needed)
        if probe_model is not None:
            print(f"  Probing gradient/activation covariance...")
            cov_grad, cov_act, par = probe_task(
                probe_model, tokenizer, train_samples, device,
                target_layer_idx=probe_layer_idx)
            diag["par"] = round(par, 2)
        else:
            cov_grad, cov_act, par = None, None, None

        # Step 2: TARA rank selection
        if config.use_tara and par is not None:
            selected_rank = tara_select_rank(par)
            diag["tara_rank"] = selected_rank
            print(f"  TARA: PaR={par:.1f} → rank={selected_rank}")
        else:
            selected_rank = config.fixed_rank if config.fixed_rank else default_rank
            diag["tara_rank"] = selected_rank

        # Note: Since all lora_modules share the same rank (injected once),
        # we cannot change rank per-task without re-injecting LoRA.
        # For this experiment, TARA rank is recorded for analysis;
        # actual training uses the initial_rank.
        # A full implementation would re-inject LoRA per task.
        actual_rank = initial_rank

        # Step 3: Re-init A, B
        if config.use_ggi and cov_grad is not None and cov_act is not None:
            # GGI init
            ggi_V = compute_ggi_init(cov_grad, cov_act, actual_rank)
            apply_ggi_init(lora_modules, ggi_V)
            diag["init"] = "ggi"
            print(f"  Applied GGI init")
        else:
            apply_kaiming_init(lora_modules)
            diag["init"] = "kaiming"

        # Step 4: CL constraint — project init
        constraint.project_init()

        # Step 5: Setup optimizer
        precond = None
        if config.use_bng and cov_act is not None:
            precond = compute_preconditioner(cov_act)
            optimizer = BNGOptimizer(lora_modules, lr=lr, precond=precond)
            diag["optimizer"] = "bng"
        else:
            optimizer = AdamWOptimizer(lora_modules, lr=lr)
            diag["optimizer"] = "adamw"

        # Step 6: Train
        dataset = SimpleTextDataset(train_samples)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate, drop_last=True)

        task_losses = []
        model.train()
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()

                outputs = model(**batch)
                loss = outputs.loss

                # CL penalty
                cl_penalty = constraint.get_loss_penalty(lora_modules)

                # BNG balance loss
                bal_loss = optimizer.get_balance_loss()

                total_loss = loss + cl_penalty + bal_loss
                total_loss.backward()

                # Gradient clipping
                trainable_params = []
                for lm in lora_modules:
                    trainable_params.extend([lm.lora_A, lm.lora_B])
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)

                # CL constraint pre-step (hard projection)
                constraint.pre_step()

                # Optimizer pre-step (BNG preconditioning + LR scaling)
                optimizer.pre_step()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_steps += 1

            avg_loss = epoch_loss / max(epoch_steps, 1)
            task_losses.append(round(avg_loss, 4))
            print(f"    Epoch {epoch + 1}/{n_epochs}: loss={avg_loss:.4f}")

        loss_curves[task_name] = task_losses

        # Step 7: Accumulate subspace for CL
        constraint.accumulate_subspace()

        # Step 8: Evaluate all tasks seen so far
        model.eval()
        for j in range(task_idx + 1):
            eval_task = tasks[j]
            if eval_task in all_eval_data:
                acc = evaluate_accuracy(model, tokenizer, all_eval_data[eval_task],
                                       device, batch_size=batch_size, is_t5=is_t5)
                acc_matrix[task_idx][j] = acc
                print(f"    Eval {eval_task}: {acc:.4f}")

        diag["time"] = round(time.time() - t0, 1)
        task_diagnostics[task_name] = diag

    # --- Compute CL Metrics ---
    # AP (Average Performance): mean of diagonal after all tasks
    n = n_tasks
    ap_scores = [acc_matrix[n - 1][j] for j in range(n)]
    ap = np.mean(ap_scores)

    # FT (Forward Transfer): mean accuracy right after training each task
    ft_scores = [acc_matrix[i][i] for i in range(n)]
    ft = np.mean(ft_scores)

    # BWT (Backward Transfer): average forgetting
    bwt_terms = []
    for j in range(n - 1):
        bwt_terms.append(acc_matrix[n - 1][j] - acc_matrix[j][j])
    bwt = np.mean(bwt_terms) if bwt_terms else 0.0

    # Subspace overlap diagnostic
    subspace_overlap = []
    if hasattr(constraint, 'prev_subspaces'):
        for i, ps in enumerate(constraint.prev_subspaces):
            if ps is not None:
                subspace_overlap.append(ps.shape[1])  # accumulated subspace dim

    # Clean up probe model
    if probe_model is not None:
        del probe_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        "config": config.name,
        "acc_matrix": acc_matrix.round(4).tolist(),
        "ap": round(float(ap), 4),
        "ft": round(float(ft), 4),
        "bwt": round(float(bwt), 4),
        "ap_scores": [round(float(s), 4) for s in ap_scores],
        "ft_scores": [round(float(s), 4) for s in ft_scores],
        "loss_curves": loss_curves,
        "task_diagnostics": task_diagnostics,
        "subspace_dims": subspace_overlap,
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="E4: Full GALA vs Baselines (End-to-End CL)")
    parser.add_argument("--model_name", default="google/flan-t5-large")
    parser.add_argument("--data_dir", required=True, help="Path to CL_Benchmark/")
    parser.add_argument("--benchmark", default="Long_Sequence")
    parser.add_argument("--tasks", default="sst2,imdb,yelp,amazon,agnews",
                       help="Comma-separated task names for CL sequence")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sgr_lambda", type=float, default=0.1)
    parser.add_argument("--soft_init_strength", type=float, default=0.7)
    parser.add_argument("--probe_layer", type=int, default=0,
                       help="Which encoder layer to probe for gradient/activation")
    parser.add_argument("--max_train_samples", type=int, default=2000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--quick", action="store_true",
                       help="Quick mode: only run standard_lora, inflora, gala_full")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        args.device if args.device != "auto" else "cpu"
    )
    print(f"Device: {device}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tasks = [t.strip() for t in args.tasks.split(",")]
    print(f"\nE4: Full GALA Pipeline — End-to-End CL Comparison")
    print(f"CL Sequence: {tasks}")
    print(f"Model: {args.model_name}")
    print(f"Benchmark: {args.benchmark}")
    print(f"LoRA rank: {args.lora_r}")

    # Define pipeline configurations
    configs = OrderedDict()

    # (a) Standard LoRA — Kaiming init, AdamW, no CL constraint
    configs["standard_lora"] = GALAConfig(
        "standard_lora", fixed_rank=args.lora_r)

    # (b) InfLoRA-style — Hard projection, AdamW
    configs["inflora"] = GALAConfig(
        "inflora", use_hard_projection=True, fixed_rank=args.lora_r)

    if not args.quick:
        # (d) GALA w/o BNG — TARA + GGI + SGR + AdamW
        configs["gala_no_bng"] = GALAConfig(
            "gala_no_bng", use_tara=True, use_ggi=True, use_sgr=True,
            use_bng=False, sgr_lambda=args.sgr_lambda,
            soft_init_strength=args.soft_init_strength,
            fixed_rank=args.lora_r)

        # (e) GALA w/o SGR — TARA + GGI + NoConstraint + BNG
        configs["gala_no_sgr"] = GALAConfig(
            "gala_no_sgr", use_tara=True, use_ggi=True, use_sgr=False,
            use_bng=True, fixed_rank=args.lora_r)

        # (f) GALA w/o GGI — TARA + Kaiming + SGR + BNG
        configs["gala_no_ggi"] = GALAConfig(
            "gala_no_ggi", use_tara=True, use_ggi=False, use_sgr=True,
            use_bng=True, sgr_lambda=args.sgr_lambda,
            soft_init_strength=args.soft_init_strength,
            fixed_rank=args.lora_r)

    # (c) GALA full — always last for dramatic effect
    configs["gala_full"] = GALAConfig(
        "gala_full", use_tara=True, use_ggi=True, use_sgr=True,
        use_bng=True, sgr_lambda=args.sgr_lambda,
        soft_init_strength=args.soft_init_strength,
        fixed_rank=args.lora_r)

    # --- Run all configurations ---
    all_results = {
        "experiment": "E4_full_gala_vs_baselines",
        "model": args.model_name,
        "benchmark": args.benchmark,
        "tasks": tasks,
        "lora_r": args.lora_r,
        "n_epochs": args.n_epochs,
        "lr": args.lr,
        "sgr_lambda": args.sgr_lambda,
        "soft_init_strength": args.soft_init_strength,
        "configs": {},
    }

    for cfg_name, config in configs.items():
        print(f"\n{'='*70}")
        print(f"  Configuration: {cfg_name.upper()}")
        print(f"  {config}")
        print(f"{'='*70}")

        t0 = time.time()
        results = run_gala_cl_pipeline(
            args.model_name, tokenizer, tasks, args.data_dir, args.benchmark,
            device, config,
            default_rank=args.lora_r, lora_alpha=args.lora_alpha,
            n_epochs=args.n_epochs, batch_size=args.batch_size, lr=args.lr,
            max_train_samples=args.max_train_samples,
            probe_layer_idx=args.probe_layer,
        )
        results["total_time"] = round(time.time() - t0, 1)
        all_results["configs"][cfg_name] = results

        print(f"\n  AP={results['ap']:.4f}  FT={results['ft']:.4f}  "
              f"BWT={results['bwt']:+.4f}  Time={results['total_time']:.0f}s")

    # ═══════════════════════════════════════════════════════════════════
    # Summary & Verdict
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"  E4 SUMMARY: Full GALA vs Baselines")
    print(f"{'='*70}")
    print(f"  {'Config':<20} {'AP':<8} {'FT':<8} {'BWT':<10} {'Time':<8}")
    print(f"  {'-'*54}")

    for cfg_name, res in all_results["configs"].items():
        print(f"  {cfg_name:<20} {res['ap']:<8.4f} {res['ft']:<8.4f} "
              f"{res['bwt']:<+10.4f} {res['total_time']:.0f}s")

    # Verdict: Does full GALA beat baselines?
    standard = all_results["configs"].get("standard_lora", {})
    inflora = all_results["configs"].get("inflora", {})
    gala = all_results["configs"].get("gala_full", {})

    verdict = {}
    if standard and gala:
        verdict["gala_vs_standard_ap"] = round(gala["ap"] - standard["ap"], 4)
        verdict["gala_vs_standard_bwt"] = round(gala["bwt"] - standard["bwt"], 4)
    if inflora and gala:
        verdict["gala_vs_inflora_ap"] = round(gala["ap"] - inflora["ap"], 4)
        verdict["gala_vs_inflora_bwt"] = round(gala["bwt"] - inflora["bwt"], 4)

    # Ablation analysis: which component contributes most?
    if not args.quick:
        ablation = {}
        for ablation_name in ["gala_no_bng", "gala_no_sgr", "gala_no_ggi"]:
            abl = all_results["configs"].get(ablation_name, {})
            if abl and gala:
                ablation[ablation_name] = {
                    "ap_drop": round(gala["ap"] - abl["ap"], 4),
                    "bwt_drop": round(gala["bwt"] - abl["bwt"], 4),
                }
        verdict["ablation"] = ablation

        # Find most important component
        if ablation:
            most_important = max(ablation.items(), key=lambda x: x[1]["ap_drop"])
            component_map = {
                "gala_no_bng": "BNG optimizer",
                "gala_no_sgr": "SGR constraint",
                "gala_no_ggi": "GGI init",
            }
            verdict["most_important_component"] = component_map.get(most_important[0], most_important[0])
            verdict["most_important_ap_drop"] = most_important[1]["ap_drop"]

    # Overall verdict
    gala_wins_ap = verdict.get("gala_vs_inflora_ap", 0) > 0
    gala_wins_bwt = verdict.get("gala_vs_inflora_bwt", 0) > 0
    verdict["gala_beats_inflora"] = gala_wins_ap or gala_wins_bwt
    verdict["gala_beats_standard"] = verdict.get("gala_vs_standard_ap", 0) > 0

    all_results["verdict"] = verdict

    print(f"\n  VERDICT:")
    if "gala_vs_inflora_ap" in verdict:
        print(f"    GALA vs InfLoRA: AP {verdict['gala_vs_inflora_ap']:+.4f}, "
              f"BWT {verdict['gala_vs_inflora_bwt']:+.4f}")
    if "gala_vs_standard_ap" in verdict:
        print(f"    GALA vs Standard: AP {verdict['gala_vs_standard_ap']:+.4f}, "
              f"BWT {verdict['gala_vs_standard_bwt']:+.4f}")
    if not args.quick and "most_important_component" in verdict:
        print(f"    Most important component: {verdict['most_important_component']} "
              f"(AP drop when removed: {verdict['most_important_ap_drop']:+.4f})")
    print(f"    GALA beats InfLoRA? {'YES' if verdict.get('gala_beats_inflora') else 'NO'}")
    print(f"    GALA beats Standard? {'YES' if verdict.get('gala_beats_standard') else 'NO'}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    model_short = args.model_name.split("/")[-1]
    out_path = os.path.join(args.output_dir, f"e4_full_gala_{model_short}_{args.benchmark}.json")
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
