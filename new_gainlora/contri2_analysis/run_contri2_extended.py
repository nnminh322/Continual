#!/usr/bin/env python3
"""
run_contri2_extended.py — CONTRI2 FULL: All 8 Tests for SGWI Validation

Tests 1-3: Original isolated tests (proxy-based)
Tests 4-7: Extended validation tests (NEW)
Test 8:    Full CL pipeline integration (requires HPC)

Usage:
    # Phase 0: Quick proxy validation (Tests 1-3, ~30 min on GPU)
    python run_contri2_extended.py --phase 0

    # Phase 1: Extended validation (Tests 6-7, ~15 min on GPU)
    python run_contri2_extended.py --phase 1

    # Phase 2: Real LoRA validation (Tests 4-5, requires CL checkpoints)
    python run_contri2_extended.py --phase 2 --ckpt-dir /path/to/cl_checkpoints

    # Individual tests
    python run_contri2_extended.py --test 6     # τ sensitivity only
    python run_contri2_extended.py --test 7     # negative transfer only

    # Specific tasks
    python run_contri2_extended.py --phase 0 --tasks cb,rte,mnli
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent
SRC  = ROOT.parent / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

from contri2_utils import (
    TASK_ORDER, BENCHMARK_DIR, RESULTS_DIR,
    load_task_data, extract_frozen_embeddings,
    build_srt_router, build_model, init_lora_weights,
    evaluate_model, train_lora_isolated, eval_zero_shot,
    compute_srt_distances, _compute_temperatures, scipy_softmax,
    RANDOM_INIT, NTI_INIT, SVD_FUSION_INIT, RandomAcc,
)

# Import original tests
from run_contri2 import run_test1, run_test2, run_test3, print_header


# ═════════════════════════════════════════════════════════════════════════════
#  TEST 4 — H1 Direct Validation: d_SRT vs ||ΔW_s - ΔW_t||_F
# ═════════════════════════════════════════════════════════════════════════════

def run_test4(ckpt_dir: str, router_state: dict, task_list: list, model_name: str):
    """
    H1 Direct Validation: Correlation between SRT distance and LoRA weight distance.

    REQUIRES: Real LoRA checkpoints from a completed CL training run.

    For each pair (s, t):
      1. Load LoRA_s (B_s, A_s) and LoRA_t (B_t, A_t)
      2. Compute ΔW_s = B_s·A_s, ΔW_t = B_t·A_t
      3. Compute ||ΔW_s - ΔW_t||_F (Frobenius distance in weight space)
      4. Compute d_SRT(s, t) (whitened L2 in embedding space)
      5. Plot correlation → Expected: strong positive correlation (ρ > 0.5)

    This directly validates the CORE assumption of SGWI.
    """
    print_header("TEST 4 — H1 VALIDATION: d_SRT vs ||ΔW_s - ΔW_t||_F")

    import torch

    if not ckpt_dir or not os.path.exists(ckpt_dir):
        print(f"  ❌ SKIP: checkpoint dir not found: {ckpt_dir}")
        print(f"  Run full CL pipeline first, then provide --ckpt-dir")
        return {"status": "skipped", "reason": "no checkpoints"}

    # Build full router with all tasks
    router = build_srt_router(task_list, router_state)

    # Load all LoRA checkpoints
    lora_weights = {}  # task_name → {layer_name: (A, B)}
    for t_idx, t_name in enumerate(task_list):
        # Try to find checkpoint for this task
        # Convention: ckpt_dir/{task_idx+1}-{task_name}/saved_weights/
        ckpt_path_A = os.path.join(ckpt_dir, f"{t_idx+1}-{t_name}", "saved_weights", "lora_weights_A.pt")
        ckpt_path_B = os.path.join(ckpt_dir, f"{t_idx+1}-{t_name}", "saved_weights", "lora_weights_B.pt")

        if not os.path.exists(ckpt_path_A) or not os.path.exists(ckpt_path_B):
            # Try alternative path convention
            ckpt_path_A = os.path.join(ckpt_dir, t_name, "lora_weights_A.pt")
            ckpt_path_B = os.path.join(ckpt_dir, t_name, "lora_weights_B.pt")

        if os.path.exists(ckpt_path_A) and os.path.exists(ckpt_path_B):
            A = torch.load(ckpt_path_A, map_location="cpu")
            B = torch.load(ckpt_path_B, map_location="cpu")
            lora_weights[t_name] = (A, B)
            print(f"  [{t_idx}] {t_name}: loaded LoRA checkpoint")
        else:
            print(f"  [{t_idx}] {t_name}: ⚠️ checkpoint NOT FOUND")

    if len(lora_weights) < 2:
        print(f"  ❌ Need at least 2 task checkpoints, got {len(lora_weights)}")
        return {"status": "failed", "reason": "insufficient checkpoints"}

    # Compute pairwise distances
    tasks_with_ckpts = [t for t in task_list if t in lora_weights]
    n = len(tasks_with_ckpts)

    srt_dists = []
    lora_dists = []
    pair_labels = []

    for i in range(n):
        for j in range(i + 1, n):
            t_i = tasks_with_ckpts[i]
            t_j = tasks_with_ckpts[j]

            # SRT distance (embedding space)
            sig_i = router.signatures.get(t_i)
            sig_j = router.signatures.get(t_j)
            if sig_i is None or sig_j is None:
                continue
            d_srt = float(np.linalg.norm(sig_i.mu - sig_j.mu))

            # LoRA weight distance (per layer, then average)
            A_i, B_i = lora_weights[t_i]
            A_j, B_j = lora_weights[t_j]

            layer_dists = []
            for key_a in A_i:
                key_b = key_a.replace("lora_A", "lora_B")
                if key_a in A_i and key_a in A_j and key_b in B_i and key_b in B_j:
                    dW_i = B_i[key_b].float() @ A_i[key_a].float()
                    dW_j = B_j[key_b].float() @ A_j[key_a].float()
                    d_lora = torch.norm(dW_i - dW_j, p="fro").item()
                    layer_dists.append(d_lora)

            if layer_dists:
                avg_lora_dist = np.mean(layer_dists)
                srt_dists.append(d_srt)
                lora_dists.append(avg_lora_dist)
                pair_labels.append(f"{t_i}-{t_j}")

    if len(srt_dists) < 3:
        print(f"  ❌ Not enough valid pairs ({len(srt_dists)})")
        return {"status": "failed", "reason": "insufficient pairs"}

    # Compute correlation
    srt_arr = np.array(srt_dists)
    lora_arr = np.array(lora_dists)
    correlation = float(np.corrcoef(srt_arr, lora_arr)[0, 1])

    # Spearman rank correlation (more robust to outliers)
    from scipy.stats import spearmanr
    spearman_rho, spearman_p = spearmanr(srt_arr, lora_arr)

    print(f"\n  ═══ H1 VALIDATION RESULTS ═══")
    print(f"  Pairs analyzed: {len(srt_dists)}")
    print(f"  Pearson ρ:      {correlation:.4f}")
    print(f"  Spearman ρ:     {spearman_rho:.4f} (p={spearman_p:.2e})")
    print(f"  Verdict:        {'✅ H1 SUPPORTED' if correlation > 0.3 else '❌ H1 NOT SUPPORTED'}")

    # Print top-5 most similar and most different pairs
    sorted_pairs = sorted(zip(srt_dists, lora_dists, pair_labels))
    print(f"\n  Top-5 closest pairs (SRT):")
    for d_s, d_l, label in sorted_pairs[:5]:
        print(f"    {label:<25} d_SRT={d_s:.3f}  d_LoRA={d_l:.4f}")
    print(f"\n  Top-5 furthest pairs (SRT):")
    for d_s, d_l, label in sorted_pairs[-5:]:
        print(f"    {label:<25} d_SRT={d_s:.3f}  d_LoRA={d_l:.4f}")

    return {
        "status": "completed",
        "n_pairs": len(srt_dists),
        "pearson_rho": correlation,
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "pairs": [
            {"pair": l, "d_srt": float(s), "d_lora": float(d)}
            for s, d, l in zip(srt_dists, lora_dists, pair_labels)
        ],
    }


# ═════════════════════════════════════════════════════════════════════════════
#  TEST 5 — SGWI with Real LoRA Checkpoints
# ═════════════════════════════════════════════════════════════════════════════

def run_test5(ckpt_dir: str, router_state: dict, task_list: list, model_name: str, cache_dir: str):
    """
    SGWI with REAL LoRA checkpoints (not proxy μ·μᵀ).

    For each task t > 0:
      1. Load real LoRA weights {B_s, A_s} from CL checkpoints
      2. Compute ΔW_init = Σ w_s · B_s·A_s (with real weights)
      3. SVD → rank-r → initialize LoRA_t
      4. Train + evaluate

    Compares: Real SGWI vs Proxy SGWI vs Random
    """
    import torch

    print_header("TEST 5 — SGWI WITH REAL LoRA CHECKPOINTS")

    if not ckpt_dir or not os.path.exists(ckpt_dir):
        print(f"  ❌ SKIP: checkpoint dir not found: {ckpt_dir}")
        return {"status": "skipped", "reason": "no checkpoints"}

    # Load all LoRA checkpoints
    all_lora = {}
    for t_idx, t_name in enumerate(task_list):
        ckpt_A = os.path.join(ckpt_dir, f"{t_idx+1}-{t_name}", "saved_weights", "lora_weights_A.pt")
        ckpt_B = os.path.join(ckpt_dir, f"{t_idx+1}-{t_name}", "saved_weights", "lora_weights_B.pt")
        if not os.path.exists(ckpt_A):
            ckpt_A = os.path.join(ckpt_dir, t_name, "lora_weights_A.pt")
            ckpt_B = os.path.join(ckpt_dir, t_name, "lora_weights_B.pt")
        if os.path.exists(ckpt_A) and os.path.exists(ckpt_B):
            all_lora[t_name] = {
                "A": torch.load(ckpt_A, map_location="cpu"),
                "B": torch.load(ckpt_B, map_location="cpu"),
            }

    results = {}
    # Test on selected tasks (focus on low-data tasks that benefit most)
    test_tasks = ["cb", "rte", "wic", "copa", "amazon"]
    test_tasks = [t for t in test_tasks if t in task_list and t in all_lora]

    for t_name in test_tasks:
        t_idx = task_list.index(t_name)
        if t_idx == 0:
            continue

        print(f"\n  {'─'*60}")
        print(f"  [t={t_idx}] {t_name} — Real SGWI vs Proxy vs Random")

        train_data = load_task_data(t_name, "train")
        test_data  = load_task_data(t_name, "test")
        if not train_data or not test_data:
            continue

        n_few = min(250, len(train_data))
        train_few = train_data[:n_few]

        sig_tasks = task_list[:t_idx]
        router = build_srt_router(sig_tasks, router_state)

        # ── A: Random Init ────────────────────────────────────────────
        print(f"    [A] Random Init...")
        model_r = build_model(model_name)
        init_lora_weights(model_r, RANDOM_INIT, None, t_name, None, task_list)
        curves_r = train_lora_isolated(model_r, train_few, test_data, n_epochs=5,
                                       cache_dir=cache_dir, tag=f"t5_{t_name}_random_real")
        del model_r

        # ── B: Proxy SGWI (μ·μᵀ) ────────────────────────────────────
        print(f"    [B] Proxy SGWI (μ·μᵀ)...")
        model_p = build_model(model_name)
        init_lora_weights(model_p, SVD_FUSION_INIT, router, t_name, None, task_list)
        curves_p = train_lora_isolated(model_p, train_few, test_data, n_epochs=5,
                                       cache_dir=cache_dir, tag=f"t5_{t_name}_proxy_real")
        del model_p

        # ── C: Real SGWI (B_s·A_s) ──────────────────────────────────
        print(f"    [C] Real SGWI (actual LoRA weights)...")
        model_real = build_model(model_name)
        _init_real_sgwi(model_real, router, t_name, task_list, all_lora)
        curves_real = train_lora_isolated(model_real, train_few, test_data, n_epochs=5,
                                          cache_dir=cache_dir, tag=f"t5_{t_name}_real_sgwi")
        del model_real

        acc_r = curves_r["accuracy"][-1] if curves_r["accuracy"] else 0
        acc_p = curves_p["accuracy"][-1] if curves_p["accuracy"] else 0
        acc_real = curves_real["accuracy"][-1] if curves_real["accuracy"] else 0

        print(f"    Results for {t_name}:")
        print(f"      Random:     {acc_r:.2f}%")
        print(f"      Proxy SGWI: {acc_p:.2f}%")
        print(f"      Real SGWI:  {acc_real:.2f}%")
        print(f"      Δ Real-Random: {acc_real - acc_r:+.2f}%")
        print(f"      Δ Real-Proxy:  {acc_real - acc_p:+.2f}%")

        # ── Gap 4 FIX: Proxy Quality Metric ──────────────────────────
        # Compute ||Real_ΔW - Proxy_ΔW||_F / ||Real_ΔW||_F per layer
        proxy_quality = _compute_proxy_quality(router, t_name, task_list, all_lora)
        if proxy_quality is not None:
            print(f"      Proxy Quality: {proxy_quality:.2%} "
                  f"({'✅ Good' if proxy_quality > 0.5 else '⚠️ Weak proxy'})")

        results[t_name] = {
            "random": acc_r, "proxy_sgwi": acc_p, "real_sgwi": acc_real,
            "delta_real_random": acc_real - acc_r,
            "delta_real_proxy": acc_real - acc_p,
            "proxy_quality": proxy_quality,
            "curves": {
                "random": curves_r, "proxy": curves_p, "real": curves_real
            },
        }

    return results


def _compute_proxy_quality(router, t_name, task_list, all_lora, lora_rank=8):
    """
    Gap 4: Compute proxy quality = 1 - ||Real_ΔW - Proxy_ΔW||_F / ||Real_ΔW||_F

    Measures how well μ·μᵀ approximates actual B·A weighted sum.
    Returns float in [0,1] or None if can't compute.
    """
    import torch

    t_idx = task_list.index(t_name) if t_name in task_list else -1
    prior_tasks = [t for t in task_list[:t_idx] if t in all_lora]
    if not prior_tasks:
        return None

    t_sig = router.signatures.get(t_name)
    if t_sig is None:
        return None

    # Compute weights
    dists = []
    for s_name in prior_tasks:
        s_sig = router.signatures.get(s_name)
        if s_sig:
            dists.append((s_name, float(np.linalg.norm(t_sig.mu - s_sig.mu))))
    if not dists:
        return None

    names, d_arr = zip(*dists)
    d_arr = np.array(d_arr)
    tau = _compute_temperatures(router)
    weights = scipy_softmax(-d_arr / max(tau, 1e-8))

    # Compute proxy ΔW (μ·μᵀ based)
    d = t_sig.d
    proxy_dW = np.zeros((d, d), dtype=np.float64)
    for s_name, w in zip(names, weights):
        s_sig = router.signatures.get(s_name)
        if s_sig:
            mu_s = s_sig.mu.astype(np.float64)
            proxy_dW += w * np.outer(mu_s, mu_s)

    # Compute real ΔW (B·A based) — use first layer as representative
    sample_A = all_lora[prior_tasks[0]]["A"]
    layer_keys = [k for k in sample_A.keys() if "lora_A" in k]
    if not layer_keys:
        return None

    key_a = layer_keys[0]
    key_b = key_a.replace("lora_A", "lora_B")

    real_dW = torch.zeros_like(
        all_lora[prior_tasks[0]]["B"][key_b].float() @
        all_lora[prior_tasks[0]]["A"][key_a].float()
    )
    for s_name, w in zip(names, weights):
        if s_name in all_lora:
            A_s = all_lora[s_name]["A"][key_a].float()
            B_s = all_lora[s_name]["B"][key_b].float()
            real_dW += w * (B_s @ A_s)

    real_norm = torch.norm(real_dW, p="fro").item()
    if real_norm < 1e-10:
        return None

    # Proxy is d×d, real is d_out×d_in — truncate proxy to match
    d_out, d_in = real_dW.shape
    proxy_trunc = torch.from_numpy(proxy_dW[:d_out, :d_in].astype(np.float32))
    diff_norm = torch.norm(real_dW - proxy_trunc, p="fro").item()

    quality = 1.0 - (diff_norm / real_norm)
    return max(0.0, quality)


def _init_real_sgwi(model, router, t_name, task_list, all_lora, lora_rank=8):
    """Initialize LoRA using REAL LoRA checkpoints (not proxy)."""
    import torch

    t_idx = task_list.index(t_name) if t_name in task_list else -1
    prior_tasks = [t for t in task_list[:t_idx] if t in all_lora]
    if not prior_tasks:
        print(f"    [REAL-SGWI] No prior tasks with checkpoints → RANDOM")
        for m in model.modules():
            if hasattr(m, 'lora_A') and hasattr(m, 'lora_B'):
                torch.nn.init.kaiming_uniform_(m.lora_A, a=np.sqrt(5))
                torch.nn.init.zeros_(m.lora_B)
        return

    # Compute SRT distances
    t_sig = router.signatures.get(t_name)
    if t_sig is None:
        emb_path = os.path.join(RESULTS_DIR, "cache", f"emb_{t_name}.npy")
        if os.path.exists(emb_path):
            data = np.load(emb_path)
            t_sig = router.add_task(task_id=t_name, h_train=data["embeddings"])

    if t_sig is None:
        print(f"    [REAL-SGWI] Cannot compute distances for {t_name} → RANDOM")
        return

    dists = []
    for s_name in prior_tasks:
        s_sig = router.signatures.get(s_name)
        if s_sig:
            dists.append((s_name, float(np.linalg.norm(t_sig.mu - s_sig.mu))))

    if not dists:
        return

    names, d_arr = zip(*dists)
    d_arr = np.array(d_arr)
    tau = _compute_temperatures(router)
    weights = scipy_softmax(-d_arr / max(tau, 1e-8))

    print(f"    [REAL-SGWI] τ={tau:.3f}, top weights:")
    for n, w in sorted(zip(names, weights), key=lambda x: -x[1])[:3]:
        print(f"      {n}: w={w:.3f}")

    # For each LoRA layer in the model, compute weighted ΔW from real checkpoints
    # Get a sample layer key from any checkpoint
    sample_A = all_lora[prior_tasks[0]]["A"]
    layer_keys_A = [k for k in sample_A.keys() if "lora_A" in k]

    for key_a in layer_keys_A:
        key_b = key_a.replace("lora_A", "lora_B")
        d_in = sample_A[key_a].shape[1]
        d_out = all_lora[prior_tasks[0]]["B"][key_b].shape[0]

        delta_W = torch.zeros(d_out, d_in, dtype=torch.float32)
        for s_name, w in zip(names, weights):
            if s_name in all_lora:
                A_s = all_lora[s_name]["A"][key_a].float()
                B_s = all_lora[s_name]["B"][key_b].float()
                delta_W += w * (B_s @ A_s)

        # SVD rank-r
        U, S, Vt = torch.linalg.svd(delta_W, full_matrices=False)
        sqrt_S = torch.sqrt(S[:lora_rank])
        B_init = U[:, :lora_rank] * sqrt_S.unsqueeze(0)
        A_init = sqrt_S.unsqueeze(1) * Vt[:lora_rank, :]

        # Find and set the corresponding module
        _set_module_lora(model, key_a, key_b, A_init, B_init)


def _set_module_lora(model, key_a, key_b, A_init, B_init):
    """Set LoRA A,B weights on the module matching the key."""
    import torch
    # Parse key like "encoder.block.0.layer.0.SelfAttention.lora_q.lora_A"
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            # Check if this module's path matches the key
            if any(part in key_a for part in name.split('.')):
                if module.lora_A.shape == A_init.shape and module.lora_B.shape == B_init.shape:
                    module.lora_A.data.copy_(A_init)
                    module.lora_B.data.copy_(B_init)
                    return


# ═════════════════════════════════════════════════════════════════════════════
#  TEST 6 — τ Sensitivity Sweep
# ═════════════════════════════════════════════════════════════════════════════

def run_test6(router_state: dict, task_list: list, model_name: str, cache_dir: str):
    """
    Temperature τ sensitivity analysis.

    For selected tasks, sweep τ ∈ {0 (NTI), 0.1τ_med, 0.5τ_med, τ_med, 2τ_med, 10τ_med, ∞ (uniform)}
    and measure final accuracy after 5 epochs.

    Expected: Plateau around τ_median → robust, no sensitive hyperparameter.
    """
    print_header("TEST 6 — τ SENSITIVITY SWEEP")

    # Build full router to get τ_median
    router_full = build_srt_router(task_list, router_state)
    tau_median = _compute_temperatures(router_full)
    print(f"  τ_median (data-driven) = {tau_median:.4f}")

    # τ values to sweep
    tau_values = {
        "NTI (τ→0)":     1e-8,
        "0.1·τ_med":     0.1 * tau_median,
        "0.5·τ_med":     0.5 * tau_median,
        "τ_median":       tau_median,
        "2·τ_med":       2.0 * tau_median,
        "10·τ_med":      10.0 * tau_median,
        "Uniform (τ→∞)": 1e6,
    }

    # Test on a few representative tasks
    test_tasks_tau = ["cb", "rte", "sst2", "amazon", "wic"]
    test_tasks_tau = [t for t in test_tasks_tau if t in task_list]

    all_results = {}

    for t_name in test_tasks_tau:
        t_idx = task_list.index(t_name)
        if t_idx == 0:
            continue

        print(f"\n  {'─'*60}")
        print(f"  [t={t_idx}] {t_name} — τ sweep")

        train_data = load_task_data(t_name, "train")
        test_data  = load_task_data(t_name, "test")
        if not train_data or not test_data:
            continue

        n_few = min(250, len(train_data))
        train_few = train_data[:n_few]

        sig_tasks = task_list[:t_idx]
        router = build_srt_router(sig_tasks, router_state)

        task_results = {}
        for tau_label, tau_val in tau_values.items():
            print(f"    τ={tau_label} ({tau_val:.6f})...")

            model_t = build_model(model_name)
            # Use SFI with specific temperature
            init_lora_weights(model_t, SVD_FUSION_INIT, router, t_name,
                            None, task_list, temperature=tau_val)

            curves = train_lora_isolated(
                model_t, train_few, test_data, n_epochs=5, lr=3e-4,
                cache_dir=cache_dir, tag=f"t6_{t_name}_tau_{tau_label.replace(' ', '_')}",
            )
            acc = curves["accuracy"][-1] if curves["accuracy"] else 0
            print(f"      → Acc: {acc:.2f}%")

            task_results[tau_label] = {
                "tau": tau_val,
                "accuracy": acc,
                "loss_curve": curves["loss"],
                "acc_curve": curves["accuracy"],
            }
            del model_t

        # Also add random baseline
        model_r = build_model(model_name)
        init_lora_weights(model_r, RANDOM_INIT, None, t_name, None, task_list)
        curves_r = train_lora_isolated(model_r, train_few, test_data, n_epochs=5,
                                       cache_dir=cache_dir, tag=f"t6_{t_name}_random")
        task_results["Random (baseline)"] = {
            "tau": None, "accuracy": curves_r["accuracy"][-1] if curves_r["accuracy"] else 0,
        }
        del model_r

        all_results[t_name] = task_results

        # Print summary
        print(f"\n    Summary for {t_name}:")
        print(f"    {'τ':<20} {'Accuracy':>10}")
        print(f"    {'─'*20} {'─'*10}")
        for label, res in task_results.items():
            print(f"    {label:<20} {res['accuracy']:>9.2f}%")

    return all_results


# ═════════════════════════════════════════════════════════════════════════════
#  TEST 7 — Negative Transfer Detection
# ═════════════════════════════════════════════════════════════════════════════

def run_test7(router_state: dict, task_list: list, model_name: str, cache_dir: str):
    """
    Negative Transfer Detection.

    For selected tasks, deliberately initialize from:
      A. NEAREST task (SRT-guided, expected: best)
      B. FARTHEST task (anti-SRT, expected: worst or ≈ random)
      C. RANDOM init (baseline)

    This validates that SRT distance CORRECTLY identifies beneficial sources
    and that SGWI doesn't cause harm when guided by SRT.
    """
    print_header("TEST 7 — NEGATIVE TRANSFER DETECTION")

    all_results = {}

    # Focus on tasks with clear domain structure
    test_tasks_nt = ["cb", "sst2", "dbpedia", "wic", "rte"]
    test_tasks_nt = [t for t in test_tasks_nt if t in task_list]

    for t_name in test_tasks_nt:
        t_idx = task_list.index(t_name)
        if t_idx < 2:  # Need at least 2 prior tasks
            continue

        print(f"\n  {'─'*60}")
        print(f"  [t={t_idx}] {t_name}")

        train_data = load_task_data(t_name, "train")
        test_data  = load_task_data(t_name, "test")
        if not train_data or not test_data:
            continue

        n_few = min(250, len(train_data))
        train_few = train_data[:n_few]

        # Build router with all prior tasks
        sig_tasks = task_list[:t_idx]
        router = build_srt_router(sig_tasks, router_state)

        # Find nearest and farthest tasks
        t_sig = router.signatures.get(t_name)
        if t_sig is None:
            emb_path = os.path.join(RESULTS_DIR, "cache", f"emb_{t_name}.npy")
            if os.path.exists(emb_path):
                data = np.load(emb_path)
                t_sig = router.add_task(task_id=t_name, h_train=data["embeddings"])
        if t_sig is None:
            continue

        dists = {}
        for s_name in sig_tasks:
            s_sig = router.signatures.get(s_name)
            if s_sig:
                dists[s_name] = float(np.linalg.norm(t_sig.mu - s_sig.mu))

        if len(dists) < 2:
            continue

        nearest = min(dists, key=dists.get)
        farthest = max(dists, key=dists.get)
        print(f"    Nearest:  {nearest} (d={dists[nearest]:.3f})")
        print(f"    Farthest: {farthest} (d={dists[farthest]:.3f})")

        task_results = {}

        # ── A: RANDOM Init ──────────────────────────────────────────
        print(f"    [A] Random Init...")
        model_r = build_model(model_name)
        init_lora_weights(model_r, RANDOM_INIT, None, t_name, None, task_list)
        curves_r = train_lora_isolated(model_r, train_few, test_data, n_epochs=5,
                                       cache_dir=cache_dir, tag=f"t7_{t_name}_random")
        task_results["random"] = curves_r["accuracy"][-1] if curves_r["accuracy"] else 0
        del model_r

        # ── B: NEAREST Task Init (SRT-guided) ───────────────────────
        print(f"    [B] Nearest Init ({nearest})...")
        model_near = build_model(model_name)
        # Use NTI from specific nearest task
        _init_from_specific_task(model_near, router, nearest)
        curves_near = train_lora_isolated(model_near, train_few, test_data, n_epochs=5,
                                          cache_dir=cache_dir, tag=f"t7_{t_name}_nearest_{nearest}")
        task_results["nearest"] = curves_near["accuracy"][-1] if curves_near["accuracy"] else 0
        task_results["nearest_task"] = nearest
        del model_near

        # ── C: FARTHEST Task Init (Anti-SRT) ────────────────────────
        print(f"    [C] Farthest Init ({farthest})...")
        model_far = build_model(model_name)
        _init_from_specific_task(model_far, router, farthest)
        curves_far = train_lora_isolated(model_far, train_few, test_data, n_epochs=5,
                                         cache_dir=cache_dir, tag=f"t7_{t_name}_farthest_{farthest}")
        task_results["farthest"] = curves_far["accuracy"][-1] if curves_far["accuracy"] else 0
        task_results["farthest_task"] = farthest
        del model_far

        # ── D: SFI (full SGWI) ──────────────────────────────────────
        print(f"    [D] SFI (full SGWI)...")
        model_sfi = build_model(model_name)
        init_lora_weights(model_sfi, SVD_FUSION_INIT, router, t_name, None, task_list)
        curves_sfi = train_lora_isolated(model_sfi, train_few, test_data, n_epochs=5,
                                         cache_dir=cache_dir, tag=f"t7_{t_name}_sfi")
        task_results["sfi"] = curves_sfi["accuracy"][-1] if curves_sfi["accuracy"] else 0
        del model_sfi

        all_results[t_name] = task_results

        # Summary
        print(f"\n    Summary for {t_name}:")
        print(f"      Random:    {task_results['random']:.2f}%")
        print(f"      Nearest:   {task_results['nearest']:.2f}% ({nearest})")
        print(f"      Farthest:  {task_results['farthest']:.2f}% ({farthest})")
        print(f"      SFI:       {task_results['sfi']:.2f}%")

        # Validate ordering
        ordering_correct = task_results["nearest"] >= task_results["farthest"]
        srt_helps = task_results["nearest"] >= task_results["random"]
        no_negative = task_results["farthest"] >= task_results["random"] * 0.8  # allow 20% degradation

        print(f"      Nearest ≥ Farthest: {'✅' if ordering_correct else '❌'}")
        print(f"      Nearest ≥ Random:   {'✅' if srt_helps else '❌'}")
        print(f"      Farthest ≥ 0.8×Random: {'✅' if no_negative else '⚠️ NEGATIVE TRANSFER'}")

    return all_results


def _init_from_specific_task(model, router, source_task_name):
    """Initialize LoRA using proxy (μ·μᵀ) from a SPECIFIC source task."""
    sig = router.signatures.get(source_task_name)
    if sig is None:
        print(f"    [INIT] No signature for {source_task_name} → RANDOM")
        for m in model.modules():
            if hasattr(m, 'lora_A') and hasattr(m, 'lora_B'):
                torch.nn.init.kaiming_uniform_(m.lora_A, a=np.sqrt(5))
                torch.nn.init.zeros_(m.lora_B)
        return

    import torch
    mu = sig.mu.astype(np.float64)
    delta_W = np.outer(mu, mu)

    U, S, Vt = np.linalg.svd(delta_W, full_matrices=False)
    r = 8  # lora rank
    sqrt_S = np.sqrt(S[:r])
    B_init = U[:, :r] * sqrt_S[np.newaxis, :]
    A_init = sqrt_S[:, np.newaxis] * Vt[:r, :]

    count = 0
    for m in model.modules():
        if hasattr(m, 'lora_A') and hasattr(m, 'lora_B'):
            m.lora_A.data.copy_(torch.from_numpy(A_init.astype(np.float32)))
            m.lora_B.data.copy_(torch.from_numpy(B_init.astype(np.float32)))
            count += 1
    print(f"    [INIT] From {source_task_name}: set {count} layers")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Contri2 Extended: Full SGWI Validation")
    p.add_argument("--phase", type=int, choices=[0, 1, 2],
                   help="Phase 0: Tests 1-3, Phase 1: Tests 6-7, Phase 2: Tests 4-5")
    p.add_argument("--test", type=int, choices=[1, 2, 3, 4, 5, 6, 7],
                   help="Run only this specific test")
    p.add_argument("--tasks", type=str, default="",
                   help="Comma-separated task names")
    p.add_argument("--model-name", type=str, default="google/flan-t5-small",
                   help="T5 model (small/base for quick tests)")
    p.add_argument("--ckpt-dir", type=str, default=None,
                   help="Path to CL checkpoints (for Tests 4-5)")
    p.add_argument("--cache-dir", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--skip-train", action="store_true")
    return p.parse_args()


def main():
    t0 = time.time()
    args = parse_args()

    model_name = args.model_name
    cache_dir  = args.cache_dir or str(RESULTS_DIR / "cache")
    os.makedirs(cache_dir, exist_ok=True)

    task_list = [t.strip() for t in args.tasks.split(",")] if args.tasks else TASK_ORDER

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║       CONTRI2 EXTENDED — FULL SGWI HYPOTHESIS TESTING        ║
╠══════════════════════════════════════════════════════════════╣
║  Phase     : {args.phase if args.phase is not None else 'all'}
║  Test      : {args.test if args.test else 'all in phase'}
║  Model     : {model_name}
║  Tasks     : {len(task_list)} tasks
║  Ckpt dir  : {args.ckpt_dir or 'N/A'}
║  Cache dir : {cache_dir}
╚══════════════════════════════════════════════════════════════╝
""")

    # ── Extract embeddings ─────────────────────────────────────────────────
    print_header("STEP 0 — EXTRACT FROZEN EMBEDDINGS")
    router_state = {}
    for t_idx, t_name in enumerate(task_list):
        emb_path = os.path.join(cache_dir, f"emb_{t_name}.npy")
        if os.path.exists(emb_path):
            router_state[t_name] = {"emb_path": emb_path}
            print(f"  [{t_idx:2d}] {t_name}: CACHED")
        else:
            print(f"  [{t_idx:2d}] {t_name}: EXTRACTING...")
            train_data = load_task_data(t_name, "train")
            if train_data:
                emb, _ = extract_frozen_embeddings(model_name, train_data,
                                                    max_samples=500, cache_path=emb_path)
                router_state[t_name] = {"emb_path": emb_path}

    # ── Determine which tests to run ───────────────────────────────────────
    results = {}

    run_tests = set()
    if args.test:
        run_tests.add(args.test)
    elif args.phase == 0:
        run_tests = {1, 2, 3}
    elif args.phase == 1:
        run_tests = {6, 7}
    elif args.phase == 2:
        run_tests = {4, 5}
    else:
        run_tests = {1, 2, 3, 6, 7}  # all except 4,5 (need checkpoints)

    # ── Run tests ──────────────────────────────────────────────────────────
    if 1 in run_tests and not args.skip_train:
        results["test1_zeroshot"] = run_test1(router_state, task_list, model_name, cache_dir)

    if 2 in run_tests and not args.skip_train:
        results["test2_fewshot"] = run_test2(router_state, task_list, model_name, cache_dir)

    if 3 in run_tests and not args.skip_train:
        results["test3_ablation"] = run_test3(router_state, task_list, model_name, cache_dir)

    if 4 in run_tests:
        results["test4_h1_validation"] = run_test4(
            args.ckpt_dir, router_state, task_list, model_name)

    if 5 in run_tests:
        results["test5_real_sgwi"] = run_test5(
            args.ckpt_dir, router_state, task_list, model_name, cache_dir)

    if 6 in run_tests and not args.skip_train:
        results["test6_tau_sensitivity"] = run_test6(
            router_state, task_list, model_name, cache_dir)

    if 7 in run_tests and not args.skip_train:
        results["test7_negative_transfer"] = run_test7(
            router_state, task_list, model_name, cache_dir)

    # ── Save results ───────────────────────────────────────────────────────
    output_path = args.output or str(RESULTS_DIR / "contri2_extended_results.json")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    elapsed = time.time() - t0
    print_header("DONE")
    print(f"  Results: {output_path}")
    print(f"  Time:    {elapsed/60:.1f} min")

    # ── Final summary ──────────────────────────────────────────────────────
    if "test7_negative_transfer" in results:
        print(f"\n  ── Test 7 Summary (Negative Transfer) ─────")
        for t_name, res in results["test7_negative_transfer"].items():
            if isinstance(res, dict) and "random" in res:
                print(f"  {t_name:<12} Random={res['random']:.1f}  "
                      f"Near={res['nearest']:.1f}({res.get('nearest_task','?')})  "
                      f"Far={res['farthest']:.1f}({res.get('farthest_task','?')})  "
                      f"SFI={res['sfi']:.1f}")

    if "test6_tau_sensitivity" in results:
        print(f"\n  ── Test 6 Summary (τ Sensitivity) ─────────")
        for t_name, res in results["test6_tau_sensitivity"].items():
            if isinstance(res, dict):
                accs = [(k, v["accuracy"]) for k, v in res.items() if isinstance(v, dict) and "accuracy" in v]
                if accs:
                    best = max(accs, key=lambda x: x[1])
                    worst = min(accs, key=lambda x: x[1])
                    print(f"  {t_name:<12} Best={best[0]}({best[1]:.1f}%)  "
                          f"Worst={worst[0]}({worst[1]:.1f}%)  "
                          f"Range={best[1]-worst[1]:.1f}pp")


if __name__ == "__main__":
    main()
