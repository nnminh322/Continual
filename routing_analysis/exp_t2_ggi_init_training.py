#!/usr/bin/env python3
"""
Tier-1 Experiment T2: GGI Init → Actual Training Comparison
=============================================================
Validates whether GGI (Generalized EVP) init leads to better LoRA training
than PCA init and Random (Kaiming) init.

Protocol:
1. Probe: Run K forward-backward passes to estimate Σ_grad and Σ_x
2. Compute 3 init strategies for lora_A:
   (a) Kaiming+zeros (standard) — random A, B=0
   (b) PCA-based — top-r eigenvectors of Σ_x
   (c) GGI-based — top-r generalized eigenvectors (Σ_grad v = λ Σ_x v)
3. Train LoRA on a single task with each init
4. Compare: loss curves, final accuracy, effective rank of ΔW

Usage:
  python exp_t2_ggi_init_training.py \
    --model_name google/flan-t5-large \
    --data_dir ../improve_gainlora/CL_Benchmark \
    --task sst2 --benchmark Long_Sequence \
    --lora_r 8 --n_epochs 3

Output: results/t2_ggi_init_<model>_<task>.json
"""
from __future__ import annotations
import argparse, json, os, sys, time, math, copy, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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
    """Load dev/test data for evaluation."""
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
    def __init__(self, samples, tokenizer, max_source_length=256, max_target_length=50, is_t5=True):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.is_t5 = is_t5

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn_t5(batch, tokenizer, max_source_length=256, max_target_length=50):
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
# LoRA injection (lightweight, standalone)
# ═══════════════════════════════════════════════════════════════════════

class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable low-rank adapter."""
    def __init__(self, base_linear: nn.Linear, r: int, alpha: float = 1.0):
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        d_out, d_in = base_linear.weight.shape
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))
        # Default: Kaiming init for A, zeros for B
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out = self.base(x)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out


def inject_lora(model, r, alpha=1.0, target_modules=None, encoder_only=False):
    """Inject LoRA adapters into target attention projections. Returns list of LoRA modules.
    If encoder_only=True (for T2), only inject into encoder to ensure GGI init covers all modules."""
    if target_modules is None:
        target_modules = ["q", "v"]  # T5-style

    lora_modules = []
    is_t5 = hasattr(model, 'encoder')

    if is_t5:
        for block_idx, block in enumerate(model.encoder.block):
            attn = block.layer[0].SelfAttention
            for name in target_modules:
                if hasattr(attn, name):
                    base_linear = getattr(attn, name)
                    lora_linear = LoRALinear(base_linear, r, alpha)
                    setattr(attn, name, lora_linear)
                    lora_modules.append(lora_linear)

        if not encoder_only:
            for block_idx, block in enumerate(model.decoder.block):
                attn = block.layer[0].SelfAttention
                for name in target_modules:
                    if hasattr(attn, name):
                        base_linear = getattr(attn, name)
                        lora_linear = LoRALinear(base_linear, r, alpha)
                        setattr(attn, name, lora_linear)
                        lora_modules.append(lora_linear)
    else:
        # LLaMA-style
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
# Gradient probing for GGI
# ═══════════════════════════════════════════════════════════════════════

def probe_gradient_and_activation(model, tokenizer, samples, device,
                                   n_batches=50, batch_size=8, target_layer_idx=0):
    """
    Probe to estimate Σ_grad (from weight gradients) and Σ_x (activation cov)
    at a target encoder layer's Q projection.
    Returns: cov_grad (d,d), cov_act (d,d), d
    """
    model.eval()
    is_t5 = hasattr(model, 'encoder')

    # Bounds check
    if is_t5:
        n_layers = len(model.encoder.block)
    else:
        n_layers = len(model.model.layers)
    if target_layer_idx >= n_layers:
        print(f"  WARNING: target_layer={target_layer_idx} >= n_layers={n_layers}, using last layer")
        target_layer_idx = n_layers - 1

    if is_t5:
        block = model.encoder.block[target_layer_idx]
        target = block.layer[0].SelfAttention
        q_module = target.q
        if isinstance(q_module, LoRALinear):
            # For probing, we use the base linear's weight gradient
            hook_target = q_module.base
            weight_param = q_module.base.weight
        else:
            hook_target = q_module
            weight_param = q_module.weight
    else:
        layer = model.model.layers[target_layer_idx]
        q_module = layer.self_attn.q_proj
        if isinstance(q_module, LoRALinear):
            hook_target = q_module.base
            weight_param = q_module.base.weight
        else:
            hook_target = q_module
            weight_param = q_module.weight

    d_in = weight_param.shape[1]

    # Collect activations via hook
    act_cache = {}
    def hook_fn(module, inp, out):
        if isinstance(inp, tuple):
            act_cache['x'] = inp[0].detach()
        else:
            act_cache['x'] = inp.detach()

    hook = hook_target.register_forward_hook(hook_fn)

    act_outer = np.zeros((d_in, d_in), dtype=np.float64)
    act_sum = np.zeros(d_in, dtype=np.float64)
    grad_outer = np.zeros((d_in, d_in), dtype=np.float64)
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
                          truncation=True, max_length=256).to(device)
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

        # Gradient info from weight gradient
        wg = weight_param.grad
        if wg is not None:
            wg_np = wg.float().cpu().numpy()  # (d_out, d_in)
            grad_outer += wg_np.T @ wg_np

        batch_count += 1

    hook.remove()

    mu = act_sum / max(n_tokens, 1)
    cov_act = act_outer / max(n_tokens, 1) - np.outer(mu, mu)
    cov_grad = grad_outer / max(batch_count, 1)

    print(f"  Probed {batch_count} batches, {n_tokens} tokens, d={d_in}")
    return cov_grad, cov_act, d_in


# ═══════════════════════════════════════════════════════════════════════
# Init strategies
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


def compute_pca_init(cov_act, rank):
    """Top-r eigenvectors of activation covariance."""
    eigvals, eigvecs = np.linalg.eigh(cov_act)
    idx = np.argsort(eigvals)[::-1]
    return eigvecs[:, idx[:rank]]  # (d_in, r)


def apply_init_to_lora(lora_modules, init_matrix, init_type, target_layer_idx=0):
    """
    Apply computed init to lora_A of each module.
    init_matrix: (d_in, r) — column vectors form the subspace.
    For lora_A shape (r, d_in): set lora_A = init_matrix.T (rows are basis vectors).
    """
    if init_type == "kaiming":
        # Standard init — already set by default
        for lm in lora_modules:
            nn.init.kaiming_uniform_(lm.lora_A, a=math.sqrt(5))
            nn.init.zeros_(lm.lora_B)
    elif init_type in ("ggi", "pca"):
        # Apply computed subspace init
        V = torch.tensor(init_matrix, dtype=torch.float32)  # (d_in, r)
        for lm in lora_modules:
            d_in = lm.lora_A.shape[1]
            r = lm.lora_A.shape[0]
            # V might have different d_in if layers have different sizes
            if V.shape[0] == d_in and V.shape[1] >= r:
                lm.lora_A.data.copy_(V[:, :r].T)  # (r, d_in)
            else:
                # Fallback: apply to matching layers only, use kaiming otherwise
                nn.init.kaiming_uniform_(lm.lora_A, a=math.sqrt(5))
            # B = small random (not zero, to break symmetry)
            nn.init.zeros_(lm.lora_B)
    elif init_type == "ggi_b_nonzero":
        # GGI init with non-zero B (GALA's recommendation: balanced init)
        V = torch.tensor(init_matrix, dtype=torch.float32)
        for lm in lora_modules:
            d_in = lm.lora_A.shape[1]
            d_out = lm.lora_B.shape[0]
            r = lm.lora_A.shape[0]
            if V.shape[0] == d_in and V.shape[1] >= r:
                lm.lora_A.data.copy_(V[:, :r].T)
            else:
                nn.init.kaiming_uniform_(lm.lora_A, a=math.sqrt(5))
            # Calibrated B init: σ = 1/sqrt(d_out) for balanced ||B|| ≈ ||A||
            nn.init.normal_(lm.lora_B, mean=0, std=1.0 / math.sqrt(d_out))


# ═══════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════

def train_one_task(model, tokenizer, samples, device, lora_modules,
                   n_epochs=3, batch_size=8, lr=1e-4, max_source_length=256,
                   eval_samples=None, log_every=50,
                   grad_accum=4, use_fp16=False):
    """Train LoRA parameters on a single task. Returns loss curve and metrics."""
    model.train()
    model.to(device)

    # Freeze everything except LoRA
    for param in model.parameters():
        param.requires_grad = False
    for lm in lora_modules:
        lm.lora_A.requires_grad_(True)
        lm.lora_B.requires_grad_(True)

    trainable_params = []
    for lm in lora_modules:
        trainable_params.extend([lm.lora_A, lm.lora_B])

    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.3f}%)")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    is_t5 = hasattr(model, 'encoder')

    # Create dataloader
    dataset = SimpleTextDataset(samples, tokenizer)
    from functools import partial
    collate = partial(collate_fn_t5, tokenizer=tokenizer,
                     max_source_length=max_source_length, max_target_length=50)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                          collate_fn=collate, drop_last=True)

    loss_curve = []
    eval_losses = []
    step = 0

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        optimizer.zero_grad()
        for _ai, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            _last = (_ai + 1 == len(dataloader))
            _do_step = ((_ai + 1) % grad_accum == 0) or _last
            _ac = (torch.autocast("cuda", dtype=torch.float16)
                   if use_fp16 and device.type == "cuda"
                   else torch.autocast("cpu", enabled=False))
            with _ac:
                outputs = model(**batch)
                loss = outputs.loss / grad_accum
            loss.backward()
            loss_val = (loss * grad_accum).item()
            if _do_step:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                epoch_steps += 1
                step += 1
                epoch_loss += loss_val
                if step % log_every == 0:
                    loss_curve.append({"step": step, "loss": round(loss_val, 4)})

        avg_loss = epoch_loss / max(epoch_steps, 1)
        loss_curve.append({"step": step, "epoch": epoch, "avg_loss": round(avg_loss, 4)})
        print(f"    Epoch {epoch}: avg_loss={avg_loss:.4f}, steps={epoch_steps}")

        # Eval
        if eval_samples:
            eval_loss = evaluate(model, tokenizer, eval_samples, device, batch_size, is_t5)
            eval_losses.append({"epoch": epoch, "eval_loss": round(eval_loss, 4)})
            print(f"    Eval loss: {eval_loss:.4f}")

    # Compute effective rank of combined ΔW
    delta_norms = []
    effective_ranks = []
    for lm in lora_modules:
        dW = (lm.lora_B @ lm.lora_A).detach().cpu().numpy()  # (d_out, d_in)
        _, S, _ = np.linalg.svd(dW, full_matrices=False)
        delta_norms.append(float(np.linalg.norm(S)))
        effective_ranks.append(participation_ratio(S ** 2))

    return {
        "loss_curve": loss_curve,
        "eval_losses": eval_losses,
        "final_loss": loss_curve[-1].get("avg_loss", loss_curve[-1].get("loss", 0)),
        "delta_w_norms": delta_norms,
        "effective_ranks": effective_ranks,
        "mean_effective_rank": round(np.mean(effective_ranks), 2),
    }


def evaluate(model, tokenizer, eval_samples, device, batch_size=8, is_t5=True):
    """Compute average eval loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for i in range(0, len(eval_samples), batch_size):
        batch_data = eval_samples[i:i + batch_size]
        if not batch_data:
            break

        inputs_text = [s["input"] for s in batch_data]
        labels_text = [s["label"] for s in batch_data]

        inputs = tokenizer(inputs_text, return_tensors="pt", padding=True,
                          truncation=True, max_length=256).to(device)
        if is_t5:
            labels = tokenizer(labels_text, return_tensors="pt", padding=True,
                             truncation=True, max_length=50).input_ids.to(device)
            labels[labels == tokenizer.pad_token_id] = -100
            inputs["labels"] = labels

        with torch.no_grad():
            outputs = model(**inputs)
            total_loss += outputs.loss.item()
        n_batches += 1

    model.train()
    return total_loss / max(n_batches, 1)


def evaluate_accuracy(model, tokenizer, eval_samples, device, batch_size=8, is_t5=True):
    """Generate predictions and compute exact-match accuracy."""
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
                          truncation=True, max_length=256).to(device)

        with torch.no_grad():
            if is_t5:
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=50,
                    do_sample=False,
                )
                preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            else:
                outputs = model.generate(
                    **inputs, max_new_tokens=50, do_sample=False,
                )
                # Decode only the generated part
                input_len = inputs["input_ids"].shape[1]
                preds = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)

        for pred, gold in zip(preds, gold_labels):
            pred_clean = pred.strip().lower()
            if pred_clean == gold:
                correct += 1
            total += 1

    model.train()
    return correct / max(total, 1)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="T2: GGI Init vs PCA vs Random → Training")
    parser.add_argument("--model_name", default="google/flan-t5-large")
    parser.add_argument("--data_dir", required=True, help="Path to CL_Benchmark/")
    parser.add_argument("--task", default="sst2")
    parser.add_argument("--benchmark", default="Long_Sequence")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4,
                       help="Gradient accumulation steps (reduces VRAM, effective_bs = batch_size × grad_accum)")
    parser.add_argument("--fp16", action="store_true",
                       help="Use fp16 mixed precision (recommended for T4/V100)")
    parser.add_argument("--max_length", type=int, default=256,
                       help="Max source sequence length (reduce to 128 for tight VRAM)") 
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_probe_batches", type=int, default=50,
                       help="Batches for gradient probing (Phase 0)")
    parser.add_argument("--target_layer", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_dir", default="results_con2")
    parser.add_argument("--max_train_samples", type=int, default=5000)
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        args.device if args.device != "auto" else "cpu"
    )
    print(f"Device: {device}")

    is_t5 = "t5" in args.model_name.lower()

    # Load data
    print(f"\nLoading data: {args.benchmark}/{args.task}")
    samples = load_task_data_simple(args.data_dir, args.benchmark, args.task, args.max_train_samples)
    eval_samples = load_eval_data(args.data_dir, args.benchmark, args.task, max_samples=500)
    print(f"  Train: {len(samples)}, Eval: {len(eval_samples) if eval_samples else 0}")

    # ---- Phase 0: Gradient probing ----
    print(f"\n{'='*60}")
    print(f"Phase 0: Gradient Probing (n={args.n_probe_batches} batches)")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_t5:
        model_probe = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, torch_dtype=torch.float32)
    else:
        model_probe = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32)
    model_probe.to(device)

    t0 = time.time()
    cov_grad, cov_act, d_in = probe_gradient_and_activation(
        model_probe, tokenizer, samples, device,
        n_batches=args.n_probe_batches, batch_size=args.batch_size,
        target_layer_idx=args.target_layer
    )
    t_probe = time.time() - t0
    print(f"  Probing time: {t_probe:.1f}s")

    # Compute init matrices
    V_ggi = compute_ggi_init(cov_grad, cov_act, args.lora_r)
    V_pca = compute_pca_init(cov_act, args.lora_r)

    # Check quality of inits
    par_grad = participation_ratio(np.maximum(np.linalg.eigvalsh(cov_grad), 0))
    par_act = participation_ratio(np.maximum(np.linalg.eigvalsh(cov_act), 0))
    print(f"  PaR(Σ_grad)={par_grad:.1f}, PaR(Σ_act)={par_act:.1f}")

    # Subspace overlap
    overlap_ggi_pca = float(np.linalg.norm(V_ggi.T @ V_pca, 'fro') ** 2 / args.lora_r)
    print(f"  GGI-PCA overlap: {overlap_ggi_pca:.4f}")

    del model_probe
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---- Train with each init strategy ----
    init_strategies = {
        "kaiming": {"type": "kaiming", "matrix": None},
        "pca": {"type": "pca", "matrix": V_pca},
        "ggi": {"type": "ggi", "matrix": V_ggi},
        "ggi_b_nonzero": {"type": "ggi_b_nonzero", "matrix": V_ggi},
    }

    all_results = {
        "experiment": "T2_ggi_init_training",
        "model": args.model_name,
        "task": args.task,
        "benchmark": args.benchmark,
        "lora_r": args.lora_r,
        "n_epochs": args.n_epochs,
        "lr": args.lr,
        "probe_time": round(t_probe, 1),
        "par_gradient": round(par_grad, 2),
        "par_activation": round(par_act, 2),
        "ggi_pca_overlap": round(overlap_ggi_pca, 4),
        "strategies": {},
    }

    for name, strategy in init_strategies.items():
        print(f"\n{'='*60}")
        print(f"Training with init: {name.upper()}")
        print(f"{'='*60}")

        # Fresh model for each strategy
        if is_t5:
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, torch_dtype=torch.float32)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32)

        lora_modules = inject_lora(model, args.lora_r, args.lora_alpha,
                                   target_modules=["q", "v"] if is_t5 else ["q_proj", "v_proj"],
                                   encoder_only=is_t5)  # encoder-only for T2 purity

        # Apply init
        apply_init_to_lora(lora_modules, strategy["matrix"], strategy["type"],
                          target_layer_idx=args.target_layer)

        model.to(device)

        t0 = time.time()
        train_results = train_one_task(
            model, tokenizer, samples, device, lora_modules,
            n_epochs=args.n_epochs, batch_size=args.batch_size, lr=args.lr,
            eval_samples=eval_samples, log_every=50,
            grad_accum=args.grad_accum, use_fp16=args.fp16,
            max_source_length=args.max_length,
        )
        t_train = time.time() - t0

        # Accuracy evaluation
        accuracy = None
        if eval_samples:
            print(f"  Computing accuracy...")
            accuracy = evaluate_accuracy(model, tokenizer, eval_samples, device,
                                        batch_size=args.batch_size, is_t5=is_t5)
            print(f"  Accuracy: {accuracy:.4f}")

        train_results["train_time"] = round(t_train, 1)
        train_results["accuracy"] = round(accuracy, 4) if accuracy is not None else None
        all_results["strategies"][name] = train_results

        del model, lora_modules
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"SUMMARY: Init Strategy Comparison")
    print(f"{'='*60}")
    print(f"{'Strategy':<20} {'Final Loss':<12} {'Eval Loss':<12} {'Accuracy':<10} {'Eff Rank':<10} {'Time':<8}")
    print("-" * 72)

    for name, res in all_results["strategies"].items():
        final_loss = res["final_loss"]
        eval_loss = res["eval_losses"][-1]["eval_loss"] if res["eval_losses"] else "N/A"
        acc = f"{res['accuracy']:.4f}" if res['accuracy'] is not None else "N/A"
        eff_r = f"{res['mean_effective_rank']:.2f}"
        t = f"{res['train_time']:.0f}s"
        print(f"  {name:<18} {final_loss:<12} {eval_loss:<12} {acc:<10} {eff_r:<10} {t:<8}")

    # Verdict
    strategies = all_results["strategies"]
    if "ggi" in strategies and "kaiming" in strategies:
        ggi_better = (strategies["ggi"]["final_loss"] < strategies["kaiming"]["final_loss"])
        if strategies["ggi"]["accuracy"] is not None and strategies["kaiming"]["accuracy"] is not None:
            ggi_acc_better = strategies["ggi"]["accuracy"] > strategies["kaiming"]["accuracy"]
        else:
            ggi_acc_better = None
        all_results["verdict"] = {
            "ggi_lower_loss_than_kaiming": ggi_better,
            "ggi_higher_accuracy_than_kaiming": ggi_acc_better,
        }
        print(f"\n  GGI lower loss than Kaiming? {'YES ✓' if ggi_better else 'NO ✗'}")
        if ggi_acc_better is not None:
            print(f"  GGI higher accuracy? {'YES ✓' if ggi_acc_better else 'NO ✗'}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    model_short = args.model_name.split("/")[-1]
    out_path = os.path.join(args.output_dir, f"t2_ggi_init_{model_short}_{args.task}.json")
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
