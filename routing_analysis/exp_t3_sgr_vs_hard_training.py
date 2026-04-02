#!/usr/bin/env python3
"""
Tier-1 Experiment T3: SGR (Soft Grassmannian Regularization) vs Hard Projection
================================================================================
Validates whether soft SGR penalty preserves backward transfer better than
InfLoRA's hard orthogonal projection in a CL sequence.

Protocol:
1. Train tasks sequentially (5-task CL sequence)
2. For each new task, apply CL constraint:
   (a) Hard: project A → null space of accumulated subspace (InfLoRA style)
   (b) Soft(λ): add penalty λ₁ ||V_t^T V_prev||_F^2 to loss
   (c) No constraint (upper bound on FT, lower bound on BWT)
3. After all tasks, evaluate every previous task
4. Compare: AP, FT, BWT, per-task accuracy

Usage:
  python exp_t3_sgr_vs_hard_training.py \
    --model_name google/flan-t5-large \
    --data_dir ../improve_gainlora/CL_Benchmark \
    --benchmark Long_Sequence \
    --tasks sst2,imdb,yelp,amazon,agnews \
    --lora_r 8 --n_epochs 3

Output: results/t3_sgr_vs_hard_<model>_<benchmark>.json
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
# Data loading (reuse from T2)
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


def load_eval_data(data_dir, benchmark, task, max_samples=300):
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

    def get_subspace(self):
        """Return the input subspace of A: top-r left singular vectors."""
        A = self.lora_A.detach()  # (r, d_in)
        # Rows of A span the input subspace
        U, S, Vh = torch.linalg.svd(A.T, full_matrices=False)  # A^T: (d_in, r)
        return U  # (d_in, r) — orthonormal columns


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
# CL Constraint Strategies
# ═══════════════════════════════════════════════════════════════════════

class HardProjection:
    """InfLoRA-style: project gradients into null space of previous subspace before each optimizer step."""

    def __init__(self, lora_modules):
        self.lora_modules = lora_modules
        self.prev_subspaces = [None] * len(lora_modules)  # per-module list of (d_in, k) tensors

    def accumulate_subspace(self):
        """After training a task, record current A's subspace."""
        for i, lm in enumerate(self.lora_modules):
            V_new = lm.get_subspace()  # (d_in, r)
            if self.prev_subspaces[i] is None:
                self.prev_subspaces[i] = V_new
            else:
                # Combine and orthonormalize
                combined = torch.cat([self.prev_subspaces[i], V_new], dim=1)
                U, S, _ = torch.linalg.svd(combined, full_matrices=False)
                # Keep directions with non-trivial singular values
                threshold = S.max() * 1e-5
                k = (S > threshold).sum().item()
                self.prev_subspaces[i] = U[:, :k]

    def project_init(self):
        """Project current A into null space of accumulated subspace."""
        for i, lm in enumerate(self.lora_modules):
            if self.prev_subspaces[i] is None:
                continue
            V_prev = self.prev_subspaces[i].to(lm.lora_A.device)  # (d_in, k)
            P = V_prev @ V_prev.T  # projection matrix
            # A_new = A - A @ P (remove component in previous subspace)
            with torch.no_grad():
                lm.lora_A.data -= lm.lora_A.data @ P
                # Re-normalize to preserve scale
                row_norms = lm.lora_A.data.norm(dim=1, keepdim=True).clamp(min=1e-8)
                target_norm = math.sqrt(2.0 / lm.d_in)  # Kaiming-like norm
                lm.lora_A.data *= target_norm / row_norms

    def pre_step(self):
        """Project lora_A gradient into null space BEFORE optimizer step.
        This is how InfLoRA/GPM actually works: constrain the gradient direction,
        not the weight itself."""
        for i, lm in enumerate(self.lora_modules):
            if self.prev_subspaces[i] is None:
                continue
            V_prev = self.prev_subspaces[i].to(lm.lora_A.device)
            P = V_prev @ V_prev.T
            if lm.lora_A.grad is not None:
                # Remove gradient component that lies in previous subspace
                lm.lora_A.grad.data -= lm.lora_A.grad.data @ P

    def get_loss_penalty(self, lora_modules):
        """Hard projection: no loss penalty."""
        return 0.0


class SoftGrassmannianRegularization:
    """
    SGR: add penalty λ₁ Σ ||V_t^T V_prev||_F^2 to loss.
    This penalizes overlap between current subspace and previous subspaces
    without destroying information.
    """

    def __init__(self, lora_modules, lambda1=0.1, soft_init_strength=0.7):
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

    def project_init(self):
        """Soft: bias init towards null space without hard constraint.
        Remove soft_init_strength fraction of the component that overlaps
        with previous subspace, keeping some capacity for shared structure."""
        for i, lm in enumerate(self.lora_modules):
            if self.prev_subspaces[i] is None:
                continue
            V_prev = self.prev_subspaces[i].to(lm.lora_A.device)
            P = V_prev @ V_prev.T
            with torch.no_grad():
                lm.lora_A.data -= self.soft_init_strength * (lm.lora_A.data @ P)

    def pre_step(self):
        """No pre-step gradient projection for SGR — penalty handles it."""
        pass

    def get_loss_penalty(self, lora_modules):
        """Compute SGR penalty: Σ_modules ||V_current^T V_prev||_F^2.
        Uses QR decomposition to get orthonormal V_t from A (matching Prop 3.1).
        Gradient flows through A via the QR factorization chain rule.
        """
        penalty = 0.0
        n_terms = 0
        for i, lm in enumerate(lora_modules):
            if self.prev_subspaces[i] is None:
                continue
            V_prev = self.prev_subspaces[i].to(lm.lora_A.device)
            # Current subspace: QR of A^T gives orthonormal basis for row space of A
            A = lm.lora_A  # (r, d_in)
            # Regularize for numerical stability (εI from Proposition 3.1)
            Q, R = torch.linalg.qr(A.T)  # Q: (d_in, r), R: (r, r)
            V_current = Q  # (d_in, r) — orthonormal, differentiable through QR
            overlap = V_current.T @ V_prev  # (r, k)
            penalty += (overlap ** 2).sum()
            n_terms += 1
        if n_terms > 0:
            penalty = self.lambda1 * penalty / n_terms
        return penalty


class NoConstraint:
    """No CL constraint — upper bound on FT, lower bound on BWT."""

    def __init__(self, lora_modules):
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

    def project_init(self):
        pass

    def pre_step(self):
        pass

    def get_loss_penalty(self, lora_modules):
        return 0.0


# ═══════════════════════════════════════════════════════════════════════
# CL Training Loop
# ═══════════════════════════════════════════════════════════════════════

def train_cl_sequence(model_name, tokenizer, tasks, data_dir, benchmark,
                      device, constraint_strategy_class, constraint_kwargs,
                      lora_r=8, lora_alpha=1.0, n_epochs=3, batch_size=8,
                      lr=1e-4, max_train_samples=2000,
                      grad_accum=4, use_fp16=False, max_source_length=256):
    """
    Train a CL sequence with a given constraint strategy.
    Returns accuracy matrix A[i][j] = accuracy on task j after training task i.
    """
    from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM

    is_t5 = "t5" in model_name.lower()

    # Load fresh model
    if is_t5:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

    lora_modules = inject_lora(model, lora_r, lora_alpha,
                               target_modules=["q", "v"] if is_t5 else ["q_proj", "v_proj"])
    model.to(device)

    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False
    for lm in lora_modules:
        lm.lora_A.requires_grad_(True)
        lm.lora_B.requires_grad_(True)

    constraint = constraint_strategy_class(lora_modules, **constraint_kwargs)

    # Load all eval data upfront
    eval_data = {}
    for task in tasks:
        ed = load_eval_data(data_dir, benchmark, task, max_samples=300)
        if ed:
            eval_data[task] = ed

    # Accuracy matrix: acc[i][j] = accuracy on task j after training up to task i
    n_tasks = len(tasks)
    acc_matrix = np.zeros((n_tasks, n_tasks))
    ft_scores = []  # forward transfer: accuracy right after training each task
    loss_curves = {}

    for t_idx, task in enumerate(tasks):
        print(f"\n  --- Task {t_idx}: {task} ---")

        # Load training data
        try:
            samples = load_task_data_simple(data_dir, benchmark, task, max_train_samples)
        except FileNotFoundError:
            print(f"    SKIP: data not found")
            continue

        # Reset B to zeros, re-init A with Kaiming for new task
        for lm in lora_modules:
            nn.init.kaiming_uniform_(lm.lora_A, a=math.sqrt(5))
            nn.init.zeros_(lm.lora_B)

        # Apply CL constraint on init
        constraint.project_init()

        # Training
        trainable_params = []
        for lm in lora_modules:
            trainable_params.extend([lm.lora_A, lm.lora_B])
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

        dataset = SimpleTextDataset(samples)
        collate = partial(collate_fn_t5, tokenizer=tokenizer,
                         max_source_length=256, max_target_length=50)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate, drop_last=True)

        task_losses = []
        model.train()
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
                    loss = outputs.loss
                    cl_penalty = constraint.get_loss_penalty(lora_modules)
                    if isinstance(cl_penalty, torch.Tensor):
                        total_loss = (loss + cl_penalty) / grad_accum
                    else:
                        total_loss = loss / grad_accum
                total_loss.backward()
                if _do_step:
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    constraint.pre_step()
                    optimizer.step()
                    optimizer.zero_grad()
                    epoch_loss += loss.item()
                    epoch_steps += 1

            avg_loss = epoch_loss / max(epoch_steps, 1)
            task_losses.append(round(avg_loss, 4))
            print(f"    Epoch {epoch}: loss={avg_loss:.4f}")

        loss_curves[task] = task_losses

        # Record subspace for future tasks
        constraint.accumulate_subspace()

        # Evaluate on ALL tasks seen so far
        model.eval()
        for j, eval_task in enumerate(tasks):
            if eval_task in eval_data:
                acc = evaluate_accuracy(model, tokenizer, eval_data[eval_task], device, batch_size, is_t5)
                acc_matrix[t_idx, j] = acc
                if j == t_idx:
                    ft_scores.append(acc)
                    print(f"    FT({eval_task}) = {acc:.4f}")
                elif j < t_idx:
                    print(f"    BWT({eval_task}) = {acc:.4f}")
        model.train()

    # Compute CL metrics
    ap = float(np.mean([acc_matrix[n_tasks - 1, j] for j in range(n_tasks) if acc_matrix[n_tasks - 1, j] > 0]))
    bwt_vals = []
    for j in range(n_tasks - 1):
        if acc_matrix[j, j] > 0:
            bwt_vals.append(acc_matrix[n_tasks - 1, j] - acc_matrix[j, j])
    bwt = float(np.mean(bwt_vals)) if bwt_vals else 0.0
    ft = float(np.mean(ft_scores)) if ft_scores else 0.0

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        "acc_matrix": acc_matrix.tolist(),
        "ap": round(ap, 4),
        "ft": round(ft, 4),
        "bwt": round(bwt, 4),
        "ft_scores": [round(f, 4) for f in ft_scores],
        "loss_curves": loss_curves,
    }


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
                          truncation=True, max_length=getattr(args, "max_length", 256)).to(device)
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
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="T3: SGR vs Hard Projection in CL Sequence")
    parser.add_argument("--model_name", default="google/flan-t5-large")
    parser.add_argument("--data_dir", required=True, help="Path to CL_Benchmark/")
    parser.add_argument("--benchmark", default="Long_Sequence")
    parser.add_argument("--tasks", default="sst2,imdb,yelp,amazon,agnews",
                       help="Comma-separated task names for CL sequence")
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
    parser.add_argument("--sgr_lambda", type=float, default=0.1,
                       help="SGR penalty strength λ₁")
    parser.add_argument("--sgr_lambdas", default="0.01,0.05,0.1,0.5,1.0",
                       help="Lambda values to sweep for SGR")
    parser.add_argument("--soft_init_strength", type=float, default=0.7,
                       help="Fraction of overlap to remove in SGR soft init (0=none, 1=full)")
    parser.add_argument("--max_train_samples", type=int, default=2000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_dir", default="results_con2")
    parser.add_argument("--quick", action="store_true",
                       help="Quick mode: only run hard, soft(0.1), no_constraint")
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
    print(f"\nCL Sequence: {tasks}")
    print(f"Model: {args.model_name}")
    print(f"Benchmark: {args.benchmark}")

    # Define strategies to test
    strategies = OrderedDict()

    # 1. Hard projection (InfLoRA baseline)
    strategies["hard_projection"] = {
        "class": HardProjection,
        "kwargs": {},
    }

    # 2. SGR with lambda sweep
    if args.quick:
        lambdas = [float(args.sgr_lambda)]
    else:
        lambdas = [float(l) for l in args.sgr_lambdas.split(",")]

    for lam in lambdas:
        strategies[f"sgr_lambda{lam}"] = {
            "class": SoftGrassmannianRegularization,
            "kwargs": {"lambda1": lam, "soft_init_strength": args.soft_init_strength},
        }

    # 3. No constraint (upper/lower bound)
    strategies["no_constraint"] = {
        "class": NoConstraint,
        "kwargs": {},
    }

    all_results = {
        "experiment": "T3_sgr_vs_hard_training",
        "model": args.model_name,
        "benchmark": args.benchmark,
        "tasks": tasks,
        "lora_r": args.lora_r,
        "n_epochs": args.n_epochs,
        "lr": args.lr,
        "strategies": {},
    }

    for name, strategy in strategies.items():
        print(f"\n{'='*60}")
        print(f"Strategy: {name.upper()}")
        print(f"{'='*60}")

        t0 = time.time()
        results = train_cl_sequence(
            args.model_name, tokenizer, tasks, args.data_dir, args.benchmark,
            device, strategy["class"], strategy["kwargs"],
            lora_r=args.lora_r, lora_alpha=args.lora_alpha,
            n_epochs=args.n_epochs, batch_size=args.batch_size,
            lr=args.lr, max_train_samples=args.max_train_samples,
            grad_accum=args.grad_accum, use_fp16=args.fp16,
            max_source_length=args.max_length,
        )
        t_total = time.time() - t0
        results["time"] = round(t_total, 1)
        all_results["strategies"][name] = results

        print(f"\n  AP={results['ap']:.4f}  FT={results['ft']:.4f}  BWT={results['bwt']:.4f}  Time={t_total:.0f}s")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"SUMMARY: CL Constraint Strategy Comparison")
    print(f"{'='*60}")
    print(f"{'Strategy':<25} {'AP':<8} {'FT':<8} {'BWT':<8} {'Time':<8}")
    print("-" * 57)

    for name, res in all_results["strategies"].items():
        print(f"  {name:<23} {res['ap']:<8.4f} {res['ft']:<8.4f} {res['bwt']:<+8.4f} {res['time']:.0f}s")

    # Verdict
    hard_res = all_results["strategies"].get("hard_projection", {})
    sgr_results = {k: v for k, v in all_results["strategies"].items() if k.startswith("sgr_")}
    best_sgr = max(sgr_results.items(), key=lambda x: x[1]["ap"]) if sgr_results else (None, None)

    if hard_res and best_sgr[1]:
        sgr_better_ap = best_sgr[1]["ap"] > hard_res["ap"]
        sgr_better_bwt = best_sgr[1]["bwt"] > hard_res["bwt"]
        all_results["verdict"] = {
            "best_sgr_strategy": best_sgr[0],
            "best_sgr_lambda": best_sgr[1].get("lambda", None),
            "sgr_ap_vs_hard": round(best_sgr[1]["ap"] - hard_res["ap"], 4),
            "sgr_bwt_vs_hard": round(best_sgr[1]["bwt"] - hard_res["bwt"], 4),
            "sgr_better_ap": sgr_better_ap,
            "sgr_better_bwt": sgr_better_bwt,
            "H3_supported": sgr_better_ap or sgr_better_bwt,
        }
        print(f"\n  Best SGR: {best_sgr[0]}")
        print(f"    AP improvement over hard: {best_sgr[1]['ap'] - hard_res['ap']:+.4f}")
        print(f"    BWT improvement over hard: {best_sgr[1]['bwt'] - hard_res['bwt']:+.4f}")
        print(f"    H3 (SGR better)? {'YES ✓' if all_results['verdict']['H3_supported'] else 'NO ✗'}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    model_short = args.model_name.split("/")[-1]
    out_path = os.path.join(args.output_dir, f"t3_sgr_vs_hard_{model_short}_{args.benchmark}.json")
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
