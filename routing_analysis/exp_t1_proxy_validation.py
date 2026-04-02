#!/usr/bin/env python3
"""
Tier-1 Experiment T1: Proxy Validation
=======================================
GALA assumes Σ_task - Σ_pool (activation-residual) ≈ proxy for Σ_grad (gradient covariance).
This script validates that assumption by:

1. Loading a pretrained T5/LLaMA model
2. Running K forward-backward passes on a single task
3. Computing ACTUAL gradient covariance Σ_grad = E[g g^T] where g = ∂L/∂x (input grad)
4. Computing proxy: Σ_task - Σ_pool (activation-based)
5. Comparing: PaR(Σ_grad) vs PaR(proxy), subspace alignment, eigenvalue correlation

NOTE on gradient types across experiments:
  - T1 uses ∂L/∂x (input gradient via register_full_backward_hook): Σ_grad = E[∂L/∂x · (∂L/∂x)^T]
    This measures how loss sensitivity is distributed across input directions.
  - T2/E0 use ∂L/∂W (weight gradient): Σ_grad_W = E[(∂L/∂W)^T · ∂L/∂W]
    This measures which weight directions receive the most gradient signal.
  These are related: Σ_grad_W ∝ Σ_token(∂L/∂x · x^T), but probe different quantities.
  Both are valid for subspace analysis; T1 validates the activation-residual proxy,
  while T2/E0 use weight gradients for init computation (GGI/TARA).

Usage:
  python exp_t1_proxy_validation.py \
    --model_name google/flan-t5-large \
    --data_dir ../improve_gainlora/CL_Benchmark \
    --task sst2 --benchmark Long_Sequence \
    --n_batches 100 --batch_size 8

Output: results/t1_proxy_<model>_<task>.json
"""
from __future__ import annotations
import argparse, json, os, sys, time, math, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np

import torch
import torch.nn as nn

# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def participation_ratio(eigvals):
    eigvals = np.maximum(np.asarray(eigvals, dtype=np.float64), 0)
    s = eigvals.sum()
    if s < 1e-15:
        return 0.0
    return float(s ** 2 / (eigvals ** 2).sum())


def subspace_alignment(V1, V2, k=None):
    """Grassmann alignment: ||V1[:,:k]^T V2[:,:k]||_F^2 / k."""
    if k is None:
        k = min(V1.shape[1], V2.shape[1])
    V1k = V1[:, :k]
    V2k = V2[:, :k]
    return float(np.linalg.norm(V1k.T @ V2k, 'fro') ** 2 / k)


def top_k_eigenvectors(C, k):
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    return eigvals[idx], eigvecs[:, idx[:k]]


# ═══════════════════════════════════════════════════════════════════════
# Data loading (lightweight, no HF datasets dependency)
# ═══════════════════════════════════════════════════════════════════════

def load_task_data_simple(data_dir, benchmark, task, max_samples=2000):
    """Load raw text + labels from CL_Benchmark JSON."""
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


# ═══════════════════════════════════════════════════════════════════════
# Gradient covariance collection
# ═══════════════════════════════════════════════════════════════════════

def collect_gradient_covariance(model, tokenizer, samples, device,
                                n_batches=100, batch_size=8,
                                target_layer_idx=0, target_module="SelfAttention"):
    """
    Collect actual gradient covariance at a target attention layer.

    For T5: hooks into encoder.block[target_layer_idx].layer[0].SelfAttention
    Computes Σ_grad = E[g_x g_x^T] where g_x = ∂L/∂x at the attention input,
    using per-token backward hooks for faithful gradient estimation.

    Also collects activation covariance Σ_act = E[x x^T] (centered).

    Returns:
        cov_grad: (d, d) gradient covariance
        cov_act: (d, d) activation covariance
        n_collected: number of tokens collected
    """
    model.eval()  # eval mode but we need gradients for inputs
    model.to(device)

    # Bounds check for layer index
    is_t5 = hasattr(model, 'encoder')
    if is_t5:
        n_layers = len(model.encoder.block)
    else:
        n_layers = len(model.model.layers)
    if target_layer_idx >= n_layers:
        print(f"  WARNING: target_layer={target_layer_idx} >= n_layers={n_layers}, using last layer")
        target_layer_idx = n_layers - 1

    # Find target module
    if is_t5:
        block = model.encoder.block[target_layer_idx]
        target = block.layer[0].SelfAttention
    else:
        target = model.model.layers[target_layer_idx].self_attn

    # Accumulators
    grad_outer_sum = None
    act_outer_sum = None
    act_sum = None
    n_tokens_act = 0
    n_tokens_grad = 0

    # Caches for hooks
    activations_cache = {}
    grad_cache = {}

    # Forward hook: capture activations entering Q projection
    def fwd_hook_fn(module, input, output):
        if isinstance(input, tuple):
            activations_cache['input'] = input[0]
        else:
            activations_cache['input'] = input

    # Backward hook: capture per-token input gradients (∂L/∂x)
    def bwd_hook_fn(module, grad_input, grad_output):
        # grad_input[0] = ∂L/∂x where x is the input to this linear layer
        if grad_input is not None and len(grad_input) > 0 and grad_input[0] is not None:
            grad_cache['grad_input'] = grad_input[0].detach()

    if is_t5:
        hook_target = target.q
    else:
        hook_target = target.q_proj

    fwd_hook = hook_target.register_forward_hook(fwd_hook_fn)
    bwd_hook = hook_target.register_full_backward_hook(bwd_hook_fn)

    rng = np.random.RandomState(42)
    indices = rng.permutation(len(samples))

    batch_count = 0
    i = 0

    while batch_count < n_batches and i < len(indices):
        batch_indices = indices[i:i + batch_size]
        i += batch_size
        if len(batch_indices) == 0:
            break

        batch_texts = [samples[j]["input"] for j in batch_indices]
        batch_labels = [samples[j]["label"] for j in batch_indices]

        # Tokenize
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                          truncation=True, max_length=512).to(device)

        if is_t5:
            labels = tokenizer(batch_labels, return_tensors="pt", padding=True,
                             truncation=True, max_length=50).input_ids.to(device)
            labels[labels == tokenizer.pad_token_id] = -100
            inputs["labels"] = labels
        else:
            # Decoder-only: concatenate input + label
            full_texts = [inp + lab for inp, lab in zip(batch_texts, batch_labels)]
            inputs = tokenizer(full_texts, return_tensors="pt", padding=True,
                             truncation=True, max_length=512).to(device)
            inputs["labels"] = inputs["input_ids"].clone()

        # Forward + backward to get gradients
        model.zero_grad()
        activations_cache.clear()
        grad_cache.clear()

        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()

        # ── Collect activations ──
        x = activations_cache.get('input', None)
        if x is not None:
            x = x.detach()
            x_flat = x.reshape(-1, x.shape[-1])  # (N_tokens, d)
            mask = inputs["attention_mask"].reshape(-1).bool()
            x_flat = x_flat[mask]

            x_np = x_flat.float().cpu().numpy()
            n = x_np.shape[0]
            d = x_np.shape[1]

            if act_outer_sum is None:
                act_outer_sum = np.zeros((d, d), dtype=np.float64)
                act_sum = np.zeros(d, dtype=np.float64)
                grad_outer_sum = np.zeros((d, d), dtype=np.float64)

            act_sum += x_np.sum(0)
            act_outer_sum += x_np.T @ x_np
            n_tokens_act += n

        # ── Collect per-token input gradients (∂L/∂x) ──
        g = grad_cache.get('grad_input', None)
        if g is not None:
            g_flat = g.reshape(-1, g.shape[-1])  # (N_tokens, d)
            mask = inputs["attention_mask"].reshape(-1).bool()
            g_flat = g_flat[mask]

            g_np = g_flat.float().cpu().numpy()
            n_g = g_np.shape[0]

            if grad_outer_sum is None:
                d = g_np.shape[1]
                grad_outer_sum = np.zeros((d, d), dtype=np.float64)

            # Per-token gradient outer product: Σ_grad = E[g g^T]
            grad_outer_sum += g_np.T @ g_np
            n_tokens_grad += n_g
        else:
            # Fallback: use weight gradient outer product (less faithful)
            if is_t5:
                w_grad = target.q.weight.grad
            else:
                w_grad = target.q_proj.weight.grad
            if w_grad is not None:
                wg = w_grad.float().cpu().numpy()  # (out, d)
                if grad_outer_sum is None:
                    d = wg.shape[1]
                    grad_outer_sum = np.zeros((d, d), dtype=np.float64)
                grad_outer_sum += wg.T @ wg
                n_tokens_grad += 1  # count as 1 "sample" for averaging
                if batch_count == 0:
                    print("  WARNING: backward hook did not capture input gradients, "
                          "falling back to weight gradient proxy (less faithful)")

        batch_count += 1

        if batch_count % 20 == 0:
            print(f"  Batch {batch_count}/{n_batches}, tokens(act)={n_tokens_act}, tokens(grad)={n_tokens_grad}")

    fwd_hook.remove()
    bwd_hook.remove()

    # Compute covariances
    if n_tokens_act == 0:
        raise RuntimeError("No activation tokens collected")

    mu = act_sum / n_tokens_act
    cov_act = act_outer_sum / n_tokens_act - np.outer(mu, mu)

    if n_tokens_grad > 0:
        cov_grad = grad_outer_sum / n_tokens_grad
        print(f"  Collected {n_tokens_act} act tokens, {n_tokens_grad} grad tokens over {batch_count} batches, d={cov_act.shape[0]}")
    else:
        cov_grad = np.zeros_like(cov_act)
        print(f"  WARNING: No gradient tokens collected! cov_grad is zero.")

    return cov_grad, cov_act, n_tokens_act


def collect_pooled_covariance(model, tokenizer, data_dir, benchmark, tasks,
                              device, n_batches_per_task=20, batch_size=8,
                              target_layer_idx=0):
    """Collect activation covariance pooled across all tasks (for Σ_pool)."""
    model.eval()
    model.to(device)

    is_t5 = hasattr(model, 'encoder')
    if is_t5:
        block = model.encoder.block[target_layer_idx]
        target = block.layer[0].SelfAttention
        hook_target = target.q
    else:
        target = model.model.layers[target_layer_idx].self_attn
        hook_target = target.q_proj

    activations_cache = {}
    def hook_fn(module, input, output):
        if isinstance(input, tuple):
            activations_cache['input'] = input[0].detach()
        else:
            activations_cache['input'] = input.detach()

    hook = hook_target.register_forward_hook(hook_fn)

    d = None
    act_outer_sum = None
    act_sum = None
    n_tokens = 0

    for task in tasks:
        try:
            samples = load_task_data_simple(data_dir, benchmark, task, max_samples=500)
        except FileNotFoundError:
            continue

        rng = np.random.RandomState(42)
        indices = rng.permutation(len(samples))
        i = 0
        batch_count = 0

        while batch_count < n_batches_per_task and i < len(indices):
            batch_idx = indices[i:i + batch_size]
            i += batch_size
            if len(batch_idx) == 0:
                break

            batch_texts = [samples[j]["input"] for j in batch_idx]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                             truncation=True, max_length=512).to(device)

            with torch.no_grad():
                activations_cache.clear()
                if is_t5:
                    model.encoder(**{k: v for k, v in inputs.items() if k != 'labels'})
                else:
                    model(**{k: v for k, v in inputs.items() if k != 'labels'})

            x = activations_cache.get('input', None)
            if x is None:
                batch_count += 1
                continue

            x_flat = x.reshape(-1, x.shape[-1])
            mask = inputs["attention_mask"].reshape(-1).bool()
            x_flat = x_flat[mask]
            x_np = x_flat.float().cpu().numpy()

            n = x_np.shape[0]
            if d is None:
                d = x_np.shape[1]
                act_outer_sum = np.zeros((d, d), dtype=np.float64)
                act_sum = np.zeros(d, dtype=np.float64)

            act_sum += x_np.sum(0)
            act_outer_sum += x_np.T @ x_np
            n_tokens += n
            batch_count += 1

    hook.remove()

    if n_tokens == 0:
        raise RuntimeError("No tokens collected for pool")

    mu = act_sum / n_tokens
    cov_pool = act_outer_sum / n_tokens - np.outer(mu, mu)
    return cov_pool


# ═══════════════════════════════════════════════════════════════════════
# Main
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


def main():
    parser = argparse.ArgumentParser(description="T1: Validate Σ_residual proxy vs actual Σ_grad")
    parser.add_argument("--model_name", default="google/flan-t5-large",
                       help="HuggingFace model name or path")
    parser.add_argument("--data_dir", required=True,
                       help="Path to CL_Benchmark/ root (containing Long_Sequence/, SuperNI/)")
    parser.add_argument("--task", default="sst2", help="Task to compute gradient covariance for")
    parser.add_argument("--benchmark", default="Long_Sequence", choices=["Long_Sequence", "SuperNI"])
    parser.add_argument("--n_batches", type=int, default=100,
                       help="Number of forward-backward batches for gradient covariance")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--target_layer", type=int, default=0,
                       help="Encoder layer index to probe (0 = first layer)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--ranks", default="4,8,16,32", help="Ranks to compare subspace alignment at")
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    is_t5 = "t5" in args.model_name.lower()
    if is_t5:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, torch_dtype=torch.float32)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32)
    model.to(device)

    # Load task data
    print(f"\nLoading task data: {args.benchmark}/{args.task}")
    samples = load_task_data_simple(args.data_dir, args.benchmark, args.task, max_samples=2000)
    print(f"  {len(samples)} samples loaded")

    # Step 1: Collect actual gradient covariance
    print(f"\n{'='*60}")
    print(f"Step 1: Collecting ACTUAL gradient covariance (Σ_grad)")
    print(f"  Layer: {args.target_layer}, {args.n_batches} batches × {args.batch_size}")
    print(f"{'='*60}")

    t0 = time.time()
    cov_grad, cov_act_task, n_tokens = collect_gradient_covariance(
        model, tokenizer, samples, device,
        n_batches=args.n_batches, batch_size=args.batch_size,
        target_layer_idx=args.target_layer
    )
    t_grad = time.time() - t0
    print(f"  Time: {t_grad:.1f}s")

    # Step 2: Collect pooled activation covariance
    print(f"\n{'='*60}")
    print(f"Step 2: Collecting pooled activation covariance (Σ_pool)")
    print(f"{'='*60}")

    all_tasks = BENCHMARKS[args.benchmark]
    t0 = time.time()
    cov_pool = collect_pooled_covariance(
        model, tokenizer, args.data_dir, args.benchmark, all_tasks,
        device, n_batches_per_task=10, batch_size=args.batch_size,
        target_layer_idx=args.target_layer
    )
    t_pool = time.time() - t0
    print(f"  Time: {t_pool:.1f}s")

    # Step 3: Compute proxy covariance (Σ_task - Σ_pool)
    print(f"\n{'='*60}")
    print(f"Step 3: Comparing Σ_grad vs Σ_residual proxy")
    print(f"{'='*60}")

    cov_residual = cov_act_task - cov_pool
    # Only positive part (task has MORE variance than pool)
    eigvals_res, eigvecs_res = np.linalg.eigh(cov_residual)
    idx_res = np.argsort(eigvals_res)[::-1]
    eigvals_res = eigvals_res[idx_res]
    eigvecs_res = eigvecs_res[:, idx_res]

    eigvals_grad, eigvecs_grad = np.linalg.eigh(cov_grad)
    idx_grad = np.argsort(eigvals_grad)[::-1]
    eigvals_grad = eigvals_grad[idx_grad]
    eigvecs_grad = eigvecs_grad[:, idx_grad]

    eigvals_act, eigvecs_act = np.linalg.eigh(cov_act_task)
    idx_act = np.argsort(eigvals_act)[::-1]
    eigvals_act = eigvals_act[idx_act]
    eigvecs_act = eigvecs_act[:, idx_act]

    # PaR comparison
    par_grad = participation_ratio(np.maximum(eigvals_grad, 0))
    par_residual = participation_ratio(np.maximum(eigvals_res, 0))
    par_activation = participation_ratio(np.maximum(eigvals_act, 0))

    print(f"\n  PaR(Σ_grad)     = {par_grad:.2f}")
    print(f"  PaR(Σ_residual) = {par_residual:.2f}")
    print(f"  PaR(Σ_act)      = {par_activation:.2f}")
    print(f"  PaR ratio (grad/residual) = {par_grad / max(par_residual, 0.01):.2f}")

    # Subspace alignment at various ranks
    ranks = [int(r) for r in args.ranks.split(",")]
    alignments = {}
    for r in ranks:
        if r > eigvecs_grad.shape[1] or r > eigvecs_res.shape[1]:
            continue
        align_grad_res = subspace_alignment(eigvecs_grad, eigvecs_res, r)
        align_grad_pca = subspace_alignment(eigvecs_grad, eigvecs_act, r)
        align_res_pca = subspace_alignment(eigvecs_res, eigvecs_act, r)
        alignments[r] = {
            "grad_vs_residual": round(align_grad_res, 4),
            "grad_vs_pca": round(align_grad_pca, 4),
            "residual_vs_pca": round(align_res_pca, 4),
        }
        print(f"\n  Rank-{r} subspace alignment:")
        print(f"    Σ_grad vs Σ_residual: {align_grad_res:.4f}")
        print(f"    Σ_grad vs Σ_act(PCA): {align_grad_pca:.4f}")
        print(f"    Σ_residual vs Σ_act:  {align_res_pca:.4f}")

    # Eigenvalue correlation
    n_compare = min(64, len(eigvals_grad), len(eigvals_res))
    ev_grad_top = np.maximum(eigvals_grad[:n_compare], 0)
    ev_res_top = np.maximum(eigvals_res[:n_compare], 0)
    ev_act_top = np.maximum(eigvals_act[:n_compare], 0)

    # Log-scale correlation (more meaningful for eigenvalues)
    log_grad = np.log1p(ev_grad_top)
    log_res = np.log1p(ev_res_top)
    log_act = np.log1p(ev_act_top)

    corr_grad_res = float(np.corrcoef(log_grad, log_res)[0, 1]) if np.std(log_res) > 1e-10 else 0.0
    corr_grad_act = float(np.corrcoef(log_grad, log_act)[0, 1]) if np.std(log_act) > 1e-10 else 0.0

    print(f"\n  Eigenvalue correlation (log-scale, top-{n_compare}):")
    print(f"    Σ_grad vs Σ_residual: {corr_grad_res:.4f}")
    print(f"    Σ_grad vs Σ_act:      {corr_grad_act:.4f}")

    # TARA rank comparison
    cumvar_grad = np.cumsum(np.maximum(eigvals_grad, 0))
    total_grad = cumvar_grad[-1] if cumvar_grad[-1] > 1e-15 else 1.0
    r90_grad = int(np.searchsorted(cumvar_grad / total_grad, 0.90)) + 1
    r95_grad = int(np.searchsorted(cumvar_grad / total_grad, 0.95)) + 1

    cumvar_res = np.cumsum(np.maximum(eigvals_res, 0))
    total_res = cumvar_res[-1] if cumvar_res[-1] > 1e-15 else 1.0
    r90_res = int(np.searchsorted(cumvar_res / total_res, 0.90)) + 1
    r95_res = int(np.searchsorted(cumvar_res / total_res, 0.95)) + 1

    print(f"\n  TARA rank recommendation:")
    print(f"    From Σ_grad:     r_90={r90_grad}, r_95={r95_grad}")
    print(f"    From Σ_residual: r_90={r90_res}, r_95={r95_res}")
    print(f"    Difference:      r_90={abs(r90_grad - r90_res)}, r_95={abs(r95_grad - r95_res)}")

    # Verdict
    proxy_valid = (corr_grad_res > 0.7 and
                   all(alignments.get(r, {}).get("grad_vs_residual", 0) > 0.3 for r in [8, 16]))
    print(f"\n  {'='*60}")
    print(f"  VERDICT: Proxy (Σ_task - Σ_pool) is {'VALID ✓' if proxy_valid else 'WEAK ✗'}")
    print(f"  {'='*60}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    model_short = args.model_name.split("/")[-1]
    out_path = os.path.join(args.output_dir,
                           f"t1_proxy_{model_short}_{args.task}_layer{args.target_layer}.json")

    results = {
        "experiment": "T1_proxy_validation",
        "model": args.model_name,
        "task": args.task,
        "benchmark": args.benchmark,
        "target_layer": args.target_layer,
        "n_batches": args.n_batches,
        "batch_size": args.batch_size,
        "n_tokens": n_tokens,
        "time_grad_collection": round(t_grad, 1),
        "time_pool_collection": round(t_pool, 1),
        "par_gradient": round(par_grad, 2),
        "par_residual": round(par_residual, 2),
        "par_activation": round(par_activation, 2),
        "par_ratio_grad_over_residual": round(par_grad / max(par_residual, 0.01), 2),
        "tara_r90_grad": r90_grad,
        "tara_r90_residual": r90_res,
        "tara_r95_grad": r95_grad,
        "tara_r95_residual": r95_res,
        "eigenvalue_correlation_grad_vs_residual": round(corr_grad_res, 4),
        "eigenvalue_correlation_grad_vs_activation": round(corr_grad_act, 4),
        "subspace_alignments": alignments,
        "verdict_proxy_valid": proxy_valid,
        "eigvals_grad_top32": np.maximum(eigvals_grad[:32], 0).tolist(),
        "eigvals_residual_top32": np.maximum(eigvals_res[:32], 0).tolist(),
        "eigvals_activation_top32": np.maximum(eigvals_act[:32], 0).tolist(),
    }

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
