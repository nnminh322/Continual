#!/usr/bin/env python3
"""
Experiment E3: BNG (Balanced Natural Gradient) vs AdamW vs LoRA+
=================================================================
Tests whether activation preconditioning + asymmetric LR improves
LoRA training convergence on a single task.

Strategies compared:
(a) AdamW — standard baseline
(b) AdamW + asymmetric LR only (LoRA+ style: η_B = ratio × η_A)
(c) AdamW + Σ_x^{-1/2} preconditioning only
(d) Full BNG: preconditioning + adaptive asymmetric LR + β-EMA

Protocol:
1. Gradient probing: estimate Σ_x for preconditioning matrix
2. Train LoRA with each optimizer strategy
3. Compare: loss curves, convergence speed, final accuracy

Usage:
  python exp_e3_bng_vs_adamw.py \
    --model_name google/flan-t5-large \
    --data_dir ../improve_gainlora/CL_Benchmark \
    --task sst2 --benchmark Long_Sequence \
    --lora_r 8 --n_epochs 5

Output: results/e3_bng_vs_adamw_<model>_<task>.json
"""
from __future__ import annotations
import argparse, json, os, sys, time, math, copy, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from functools import partial


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def participation_ratio(eigvals):
    eigvals = np.maximum(np.asarray(eigvals, dtype=np.float64), 0)
    s = eigvals.sum()
    if s < 1e-15:
        return 0.0
    return float(s ** 2 / (eigvals ** 2).sum())


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_task_data_simple(data_dir, benchmark, task, max_samples=5000):
    json_path = Path(data_dir) / benchmark / task / "train.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Data not found: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    instances = data.get("Instances", [])
    defn_list = data.get("Definition", [])
    definition = defn_list[0] if defn_list else ""
    samples = []
    for inst in instances[:max_samples]:
        text = inst.get("input", "")
        label = inst.get("output", "")
        if isinstance(label, list):
            label = label[0] if label else ""
        instruction = f"{definition}\n{text}\nOutput: "
        samples.append({"input": instruction, "label": label})
    return samples


def load_eval_data(data_dir, benchmark, task, max_samples=500):
    for split_name in ["dev", "test"]:
        json_path = Path(data_dir) / benchmark / task / f"{split_name}.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
            instances = data.get("Instances", [])
            defn_list = data.get("Definition", [])
            definition = defn_list[0] if defn_list else ""
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
        self.r = r
        self.d_in = d_in
        self.d_out = d_out
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out = self.base(x)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out


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
# Activation covariance probing
# ═══════════════════════════════════════════════════════════════════════

def probe_activation_covariance(model, tokenizer, samples, device,
                                 n_batches=50, batch_size=8, target_layer_idx=0):
    """Probe activation covariance at target layer for preconditioning."""
    model.eval()
    is_t5 = hasattr(model, 'encoder')

    if is_t5:
        n_layers = len(model.encoder.block)
        if target_layer_idx >= n_layers:
            target_layer_idx = n_layers - 1
        q_module = model.encoder.block[target_layer_idx].layer[0].SelfAttention.q
    else:
        n_layers = len(model.model.layers)
        if target_layer_idx >= n_layers:
            target_layer_idx = n_layers - 1
        q_module = model.model.layers[target_layer_idx].self_attn.q_proj

    # If LoRA-wrapped, hook into base
    if isinstance(q_module, LoRALinear):
        hook_target = q_module.base
    else:
        hook_target = q_module

    d_in = hook_target.weight.shape[1]
    act_cache = {}

    def hook_fn(module, inp, out):
        if isinstance(inp, tuple):
            act_cache['x'] = inp[0].detach()
        else:
            act_cache['x'] = inp.detach()

    hook = hook_target.register_forward_hook(hook_fn)

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
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                          truncation=True, max_length=512).to(device)

        with torch.no_grad():
            act_cache.clear()
            if is_t5:
                model.encoder(**{k: v for k, v in inputs.items() if k != 'labels'})
            else:
                model(**{k: v for k, v in inputs.items() if k != 'labels'})

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
        batch_count += 1

    hook.remove()

    mu = act_sum / max(n_tokens, 1)
    cov_act = act_outer / max(n_tokens, 1) - np.outer(mu, mu)
    print(f"  Probed {batch_count} batches, {n_tokens} tokens, d={d_in}")
    return cov_act, d_in


def compute_preconditioner(cov_act, k=None):
    """Compute low-rank Σ_x^{-1/2} approximation for preconditioning."""
    eigvals, eigvecs = np.linalg.eigh(cov_act)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    eigvals_pos = np.maximum(eigvals, 1e-10)
    if k is None:
        k = int(participation_ratio(eigvals_pos))
        k = max(4, min(k, eigvals.shape[0] // 2))

    # Low-rank approx: Σ^{-1/2} ≈ V_k diag(λ_k^{-1/2}) V_k^T + λ_mean^{-1/2} (I - V_k V_k^T)
    V_k = eigvecs[:, :k]
    lam_k_inv_sqrt = 1.0 / np.sqrt(eigvals_pos[:k])
    lam_mean_inv_sqrt = 1.0 / np.sqrt(np.mean(eigvals_pos[k:]) + 1e-10)

    # Return as torch tensors for use in training
    precond = {
        "V_k": torch.tensor(V_k, dtype=torch.float32),
        "lam_k_inv_sqrt": torch.tensor(lam_k_inv_sqrt, dtype=torch.float32),
        "lam_mean_inv_sqrt": float(lam_mean_inv_sqrt),
        "k": k,
        "kappa_original": float(eigvals_pos[0] / eigvals_pos[max(k-1, 0)]),
        "kappa_preconditioned": float(max(lam_k_inv_sqrt) / min(lam_k_inv_sqrt)),
    }
    print(f"  Preconditioner: k={k}, κ_orig={precond['kappa_original']:.1f}, "
          f"κ_precond={precond['kappa_preconditioned']:.2f}")
    return precond


def apply_preconditioner(grad_A, precond, device):
    """Apply Σ_x^{-1/2} preconditioning to gradient of A.
    grad_A: (r, d_in)
    Result: grad_A @ Σ_x^{-1/2}
    """
    V_k = precond["V_k"].to(device)           # (d_in, k)
    lam_inv = precond["lam_k_inv_sqrt"].to(device)  # (k,)
    lam_mean = precond["lam_mean_inv_sqrt"]

    # grad_A @ [V_k diag(λ^{-1/2}) V_k^T + λ_mean^{-1/2} (I - V_k V_k^T)]
    # = grad_A @ V_k @ diag(λ^{-1/2} - λ_mean^{-1/2}) @ V_k^T + λ_mean^{-1/2} grad_A
    proj = grad_A @ V_k                       # (r, k)
    scaled = proj * (lam_inv - lam_mean)       # (r, k)
    result = scaled @ V_k.T + lam_mean * grad_A  # (r, d_in)
    return result


# ═══════════════════════════════════════════════════════════════════════
# Optimizer strategies
# ═══════════════════════════════════════════════════════════════════════

class AdamWStrategy:
    """Standard AdamW — baseline."""
    name = "adamw"

    def __init__(self, lora_modules, lr, **kwargs):
        params = []
        for lm in lora_modules:
            params.extend([lm.lora_A, lm.lora_B])
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        self.lora_modules = lora_modules

    def pre_step(self):
        pass

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()


class AsymmetricLRStrategy:
    """LoRA+ style: different LR for A and B (fixed ratio)."""
    name = "asymmetric_lr"

    def __init__(self, lora_modules, lr, lr_ratio=4.0, **kwargs):
        a_params = [lm.lora_A for lm in lora_modules]
        b_params = [lm.lora_B for lm in lora_modules]
        self.optimizer = torch.optim.AdamW([
            {"params": a_params, "lr": lr},
            {"params": b_params, "lr": lr * lr_ratio},
        ], weight_decay=0.01)
        self.lora_modules = lora_modules

    def pre_step(self):
        pass

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()


class PreconditionedStrategy:
    """AdamW + Σ_x^{-1/2} preconditioning on A gradients only."""
    name = "preconditioned"

    def __init__(self, lora_modules, lr, precond=None, **kwargs):
        params = []
        for lm in lora_modules:
            params.extend([lm.lora_A, lm.lora_B])
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        self.lora_modules = lora_modules
        self.precond = precond

    def pre_step(self):
        """Replace A gradients with preconditioned version before optimizer step."""
        if self.precond is None:
            return
        for lm in self.lora_modules:
            if lm.lora_A.grad is not None:
                device = lm.lora_A.device
                precond_grad = apply_preconditioner(lm.lora_A.grad.data, self.precond, device)
                lm.lora_A.grad.data.copy_(precond_grad)

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()


class BNGStrategy:
    """Full BNG: preconditioning + adaptive asymmetric LR + β-EMA + balance reg."""
    name = "bng_full"

    def __init__(self, lora_modules, lr, precond=None, beta_ema=0.99,
                 lambda_bal=0.01, **kwargs):
        # Single optimizer with base LR — we manually scale updates
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
        self.beta_smoothed = 1.0  # EMA of β
        self.base_lr = lr

    def _compute_balance_loss(self):
        """Compute (||B||^2 - ||A||^2)^2 for balance regularization."""
        loss = 0.0
        for lm in self.lora_modules:
            norm_b_sq = (lm.lora_B ** 2).sum()
            norm_a_sq = (lm.lora_A ** 2).sum()
            loss = loss + (norm_b_sq - norm_a_sq) ** 2
        return self.lambda_bal * loss

    def get_balance_loss(self):
        """Public: add to total loss before backward."""
        return self._compute_balance_loss()

    def pre_step(self):
        """Apply preconditioning + adaptive LR."""
        # 1. Precondition A gradients
        if self.precond is not None:
            for lm in self.lora_modules:
                if lm.lora_A.grad is not None:
                    device = lm.lora_A.device
                    precond_grad = apply_preconditioner(
                        lm.lora_A.grad.data, self.precond, device)
                    lm.lora_A.grad.data.copy_(precond_grad)

        # 2. Compute adaptive β and scale LR
        total_norm_A = 0.0
        total_norm_B = 0.0
        for lm in self.lora_modules:
            total_norm_A += lm.lora_A.data.norm().item() ** 2
            total_norm_B += lm.lora_B.data.norm().item() ** 2
        total_norm_A = math.sqrt(total_norm_A)
        total_norm_B = math.sqrt(total_norm_B)

        beta_raw = math.sqrt(total_norm_B / max(total_norm_A, 1e-8))
        self.beta_smoothed = self.beta_ema * self.beta_smoothed + (1 - self.beta_ema) * beta_raw

        # Scale optimizer LR: A gets η*β, B gets η/β
        beta = max(self.beta_smoothed, 0.1)  # clamp to prevent extreme values
        beta = min(beta, 10.0)
        self.optimizer.param_groups[0]["lr"] = self.base_lr * beta       # A
        self.optimizer.param_groups[1]["lr"] = self.base_lr / beta       # B

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()


# ═══════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════

def train_with_strategy(model_name, tokenizer, samples, eval_samples, device,
                        strategy_class, strategy_kwargs, lora_r=8, lora_alpha=1.0,
                        n_epochs=5, batch_size=8, lr=1e-4):
    """Train one task with a given optimizer strategy."""
    from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM

    is_t5 = "t5" in model_name.lower()
    if is_t5:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

    target_modules = ["q", "v"] if is_t5 else ["q_proj", "v_proj"]
    lora_modules = inject_lora(model, lora_r, lora_alpha, target_modules=target_modules)
    model.to(device)

    # Freeze base
    for param in model.parameters():
        param.requires_grad = False
    for lm in lora_modules:
        lm.lora_A.requires_grad_(True)
        lm.lora_B.requires_grad_(True)

    n_trainable = sum(lm.lora_A.numel() + lm.lora_B.numel() for lm in lora_modules)
    print(f"  {n_trainable:,} trainable params")

    strategy = strategy_class(lora_modules, lr=lr, **strategy_kwargs)
    is_bng = isinstance(strategy, BNGStrategy)

    dataset = SimpleTextDataset(samples)
    collate = partial(collate_fn_t5, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                          collate_fn=collate, drop_last=True)

    loss_curve = []
    eval_losses = []
    beta_history = []
    model.train()
    step = 0

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            strategy.zero_grad()

            outputs = model(**batch)
            loss = outputs.loss

            # Add balance loss for BNG
            if is_bng:
                bal_loss = strategy.get_balance_loss()
                total_loss = loss + bal_loss
            else:
                total_loss = loss

            total_loss.backward()

            # Clip gradients
            trainable_params = []
            for lm in lora_modules:
                trainable_params.extend([lm.lora_A, lm.lora_B])
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)

            # Pre-step: preconditioning + LR scaling
            strategy.pre_step()
            strategy.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            epoch_steps += 1
            step += 1

            if step % 50 == 0:
                loss_curve.append({"step": step, "loss": round(loss_val, 4)})
                if is_bng:
                    beta_history.append({"step": step, "beta": round(strategy.beta_smoothed, 4)})

        avg_loss = epoch_loss / max(epoch_steps, 1)
        loss_curve.append({"step": step, "epoch": epoch, "avg_loss": round(avg_loss, 4)})
        print(f"    Epoch {epoch}: loss={avg_loss:.4f}" +
              (f"  β={strategy.beta_smoothed:.3f}" if is_bng else ""))

        # Eval
        if eval_samples:
            model.eval()
            total_eval = 0.0
            n_eval = 0
            for i in range(0, len(eval_samples), batch_size):
                bd = eval_samples[i:i + batch_size]
                if not bd:
                    break
                inp_text = [s["input"] for s in bd]
                lab_text = [s["label"] for s in bd]
                inp = tokenizer(inp_text, return_tensors="pt", padding=True,
                               truncation=True, max_length=512).to(device)
                if is_t5:
                    lab = tokenizer(lab_text, return_tensors="pt", padding=True,
                                  truncation=True, max_length=50).input_ids.to(device)
                    lab[lab == tokenizer.pad_token_id] = -100
                    inp["labels"] = lab
                with torch.no_grad():
                    out = model(**inp)
                    total_eval += out.loss.item()
                n_eval += 1
            eval_loss = total_eval / max(n_eval, 1)
            eval_losses.append({"epoch": epoch, "eval_loss": round(eval_loss, 4)})
            print(f"    Eval loss: {eval_loss:.4f}")
            model.train()

    # Accuracy
    accuracy = None
    if eval_samples:
        model.eval()
        correct = 0
        total = 0
        for i in range(0, len(eval_samples), batch_size):
            bd = eval_samples[i:i + batch_size]
            if not bd:
                break
            inp_text = [s["input"] for s in bd]
            gold = [s["label"].strip().lower() for s in bd]
            inp = tokenizer(inp_text, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(device)
            with torch.no_grad():
                if is_t5:
                    out = model.generate(input_ids=inp["input_ids"],
                                        attention_mask=inp["attention_mask"],
                                        max_new_tokens=50, do_sample=False)
                    preds = tokenizer.batch_decode(out, skip_special_tokens=True)
                else:
                    out = model.generate(**inp, max_new_tokens=50, do_sample=False)
                    input_len = inp["input_ids"].shape[1]
                    preds = tokenizer.batch_decode(out[:, input_len:], skip_special_tokens=True)
            for pred, g in zip(preds, gold):
                if pred.strip().lower() == g:
                    correct += 1
                total += 1
        accuracy = correct / max(total, 1)
        print(f"    Accuracy: {accuracy:.4f}")

    # Effective rank
    eff_ranks = []
    for lm in lora_modules:
        dW = (lm.lora_B @ lm.lora_A).detach().cpu().numpy()
        _, S, _ = np.linalg.svd(dW, full_matrices=False)
        eff_ranks.append(participation_ratio(S ** 2))

    # Norm balance
    norms_A = [lm.lora_A.data.norm().item() for lm in lora_modules]
    norms_B = [lm.lora_B.data.norm().item() for lm in lora_modules]
    mean_ratio = float(np.mean([b / max(a, 1e-8) for a, b in zip(norms_A, norms_B)]))

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        "loss_curve": loss_curve,
        "eval_losses": eval_losses,
        "final_loss": loss_curve[-1].get("avg_loss", loss_curve[-1].get("loss", 0)),
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
        "mean_effective_rank": round(float(np.mean(eff_ranks)), 2),
        "norm_balance_ratio_B_over_A": round(mean_ratio, 3),
        "beta_history": beta_history if beta_history else None,
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="E3: BNG vs AdamW Optimizer Comparison")
    parser.add_argument("--model_name", default="google/flan-t5-large")
    parser.add_argument("--data_dir", required=True, help="Path to CL_Benchmark/")
    parser.add_argument("--task", default="sst2")
    parser.add_argument("--benchmark", default="Long_Sequence")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_probe_batches", type=int, default=50)
    parser.add_argument("--target_layer", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_dir", default="results_con2")
    parser.add_argument("--max_train_samples", type=int, default=5000)
    parser.add_argument("--lr_ratio", type=float, default=4.0,
                       help="B/A LR ratio for asymmetric strategy (LoRA+)")
    parser.add_argument("--beta_ema", type=float, default=0.99,
                       help="EMA coefficient for β smoothing in BNG")
    parser.add_argument("--lambda_bal", type=float, default=0.01,
                       help="Balance regularization weight")
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        args.device if args.device != "auto" else "cpu"
    )
    print(f"Device: {device}")

    is_t5 = "t5" in args.model_name.lower()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    print(f"\nLoading data: {args.benchmark}/{args.task}")
    samples = load_task_data_simple(args.data_dir, args.benchmark, args.task, args.max_train_samples)
    eval_samples = load_eval_data(args.data_dir, args.benchmark, args.task)
    print(f"  Train: {len(samples)}, Eval: {len(eval_samples) if eval_samples else 0}")

    # ---- Phase 0: Probe activation covariance for preconditioning ----
    print(f"\n{'='*60}")
    print(f"Phase 0: Activation Covariance Probing")
    print(f"{'='*60}")

    if is_t5:
        model_probe = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, torch_dtype=torch.float32)
    else:
        model_probe = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32)
    model_probe.to(device)

    t0 = time.time()
    cov_act, d_in = probe_activation_covariance(
        model_probe, tokenizer, samples, device,
        n_batches=args.n_probe_batches, batch_size=args.batch_size,
        target_layer_idx=args.target_layer
    )
    t_probe = time.time() - t0
    print(f"  Probing time: {t_probe:.1f}s")

    precond = compute_preconditioner(cov_act, k=args.lora_r)

    del model_probe
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---- Phase 1: Train with each strategy ----
    strategies = {
        "adamw": {
            "class": AdamWStrategy,
            "kwargs": {},
        },
        "asymmetric_lr": {
            "class": AsymmetricLRStrategy,
            "kwargs": {"lr_ratio": args.lr_ratio},
        },
        "preconditioned": {
            "class": PreconditionedStrategy,
            "kwargs": {"precond": precond},
        },
        "bng_full": {
            "class": BNGStrategy,
            "kwargs": {"precond": precond, "beta_ema": args.beta_ema,
                      "lambda_bal": args.lambda_bal},
        },
    }

    all_results = {
        "experiment": "E3_bng_vs_adamw",
        "model": args.model_name,
        "task": args.task,
        "benchmark": args.benchmark,
        "lora_r": args.lora_r,
        "n_epochs": args.n_epochs,
        "lr": args.lr,
        "probe_time": round(t_probe, 1),
        "preconditioner": {
            "k": precond["k"],
            "kappa_original": round(precond["kappa_original"], 1),
            "kappa_preconditioned": round(precond["kappa_preconditioned"], 2),
        },
        "strategies": {},
    }

    for name, cfg in strategies.items():
        print(f"\n{'='*60}")
        print(f"Strategy: {name.upper()}")
        print(f"{'='*60}")

        t0 = time.time()
        result = train_with_strategy(
            args.model_name, tokenizer, samples, eval_samples, device,
            cfg["class"], cfg["kwargs"],
            lora_r=args.lora_r, lora_alpha=args.lora_alpha,
            n_epochs=args.n_epochs, batch_size=args.batch_size, lr=args.lr
        )
        result["time"] = round(time.time() - t0, 1)
        all_results["strategies"][name] = result

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"SUMMARY: Optimizer Strategy Comparison")
    print(f"{'='*60}")
    print(f"  {'Strategy':<20} {'Loss':<10} {'Eval Loss':<12} {'Acc':<8} "
          f"{'Eff Rank':<10} {'B/A Ratio':<10} {'Time':<8}")
    print(f"  {'-'*78}")

    for name, res in all_results["strategies"].items():
        eval_loss = res["eval_losses"][-1]["eval_loss"] if res["eval_losses"] else "N/A"
        acc = f"{res['accuracy']:.4f}" if res['accuracy'] is not None else "N/A"
        print(f"  {name:<20} {res['final_loss']:<10.4f} {eval_loss:<12} {acc:<8} "
              f"{res['mean_effective_rank']:<10.2f} {res['norm_balance_ratio_B_over_A']:<10.3f} "
              f"{res['time']:.0f}s")

    # Verdict
    adamw = all_results["strategies"]["adamw"]
    bng = all_results["strategies"]["bng_full"]
    precond_res = all_results["strategies"]["preconditioned"]
    asym = all_results["strategies"]["asymmetric_lr"]

    all_results["verdict"] = {
        "bng_vs_adamw_loss": round(bng["final_loss"] - adamw["final_loss"], 4),
        "bng_vs_adamw_acc": (round(bng["accuracy"] - adamw["accuracy"], 4)
                            if bng["accuracy"] and adamw["accuracy"] else None),
        "preconditioned_vs_adamw_loss": round(precond_res["final_loss"] - adamw["final_loss"], 4),
        "asymmetric_vs_adamw_loss": round(asym["final_loss"] - adamw["final_loss"], 4),
        "bng_better_loss": bng["final_loss"] < adamw["final_loss"],
        "bng_better_acc": (bng["accuracy"] > adamw["accuracy"]
                          if bng["accuracy"] and adamw["accuracy"] else None),
        "preconditioning_helps": precond_res["final_loss"] < adamw["final_loss"],
        "asymmetric_lr_helps": asym["final_loss"] < adamw["final_loss"],
        "balance_improved": bng["norm_balance_ratio_B_over_A"] < adamw["norm_balance_ratio_B_over_A"],
    }

    print(f"\n  BNG lower loss than AdamW? "
          f"{'YES ✓' if all_results['verdict']['bng_better_loss'] else 'NO ✗'} "
          f"(Δ={all_results['verdict']['bng_vs_adamw_loss']:+.4f})")
    print(f"  Preconditioning helps? "
          f"{'YES ✓' if all_results['verdict']['preconditioning_helps'] else 'NO ✗'}")
    print(f"  Asymmetric LR helps? "
          f"{'YES ✓' if all_results['verdict']['asymmetric_lr_helps'] else 'NO ✗'}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    model_short = args.model_name.split("/")[-1]
    out_path = os.path.join(args.output_dir, f"e3_bng_vs_adamw_{model_short}_{args.task}.json")
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
