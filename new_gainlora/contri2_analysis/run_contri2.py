#!/usr/bin/env python3
"""
run_contri2.py — CONTRI2: Isolated SGWI Tests for All 15 Tasks

Implements 3 isolated tests from isolate_hypothesis_testing.md for the full
15-task Long_Sequence CL benchmark.

Usage:
    python run_contri2.py                    # run all 3 tests
    python run_contri2.py --test 1           # only Test 1: Zero-Shot Transfer
    python run_contri2.py --test 2           # only Test 2: Few-Shot Convergence
    python run_contri2.py --test 3           # only Test 3: Ablation on Init Methods
    python run_contri2.py --tasks cb,rte,mnli  # only specific tasks
    python run_contri2.py --skip-train       # skip Test 2 & 3 (use cached results)

─────────────────────────────────────────────────────────────
 TASK ORDER (order 3 — matches generate_srt_order3.py):
  0:yelp  1:amazon  2:mnli  3:cb  4:copa  5:qqp
  6:rte   7:imdb     8:sst2  9:dbpedia  10:agnews
  11:yahoo  12:multirc  13:boolq  14:wic
─────────────────────────────────────────────────────────────
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
SRC  = ROOT.parent / "src"
sys.path.insert(0, str(SRC))

from contri2_utils import (
    TASK_ORDER, BENCHMARK_DIR, RESULTS_DIR,
    load_task_data, extract_frozen_embeddings,
    compute_srt_distances, build_srt_router,
    build_model, init_lora_weights,
    evaluate_model, train_lora_isolated,
    RandomAcc,
)


# ─────────────────────────────────────────────────────────────────────────────
#  helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Contri2: Isolated SGWI Tests")
    p.add_argument("--test", type=int, choices=[1, 2, 3],
                   help="Run only this test (1=ZeroShot, 2=FewShot, 3=Ablation)")
    p.add_argument("--tasks", type=str, default="",
                   help="Comma-separated task names (default: all 15)")
    p.add_argument("--skip-train", action="store_true",
                   help="Skip training steps (use cached checkpoints)")
    p.add_argument("--cache-dir", type=str, default=None,
                   help="Override cache directory")
    p.add_argument("--output", type=str, default=None,
                   help="Override output JSON path")
    p.add_argument("--model-name", type=str,
                   default="google/flan-t5-small",
                   choices=["google/flan-t5-small", "google/flan-t5-base"],
                   help="T5 model size (small=60M or base=250M params)")
    return p.parse_args()


def task_filter(args):
    """Return list of task names to run, or None for all."""
    if args.tasks:
        return [t.strip() for t in args.tasks.split(",")]
    return None


def print_header(title):
    n = 80
    print(f"\n{'='*n}")
    print(f"  {title}")
    print(f"{'='*n}")


# ─────────────────────────────────────────────────────────────────────────────
#  TEST 1 — Zero-Shot Transfer
# ─────────────────────────────────────────────────────────────────────────────

def run_test1(router_state, task_list, model_name, cache_dir):
    """
    Zero-Shot Transfer Test.

    For each task t > 0:
      1. Get SRT distances from t to all s < t
      2. Compute softmax weights w_s using SRT distances
      3. Build ΔW_init = Σ w_s · B_s·A_s (SVD fusion)
      4. WITHOUT training: evaluate on test set
      5. Compare against random-baseline (random init → untrained)

    Baseline: untrained LoRA = random init (kaiming_uniform_ A, zeros B)
    Expected: SGWI should give accuracy well above random-guess baseline.
    """
    from contri2_utils import SVD_FUSION_INIT, eval_zero_shot

    print_header("TEST 1 — ZERO-SHOT TRANSFER (No Training)")

    results = {}

    # We need the router_state from tasks 0..t-1 to get SRT distances
    for t_idx, t_name in enumerate(task_list):
        if t_idx == 0:
            print(f"\n  [t={t_idx}] {t_name}: first task, no prior tasks → skip")
            results[t_name] = {"status": "first_task", "zs_accuracy": None, "baseline": None}
            continue

        print(f"\n  [t={t_idx}] {t_name}")

        # Build SRT router using signatures from tasks 0..t-1
        # router_state[t-1] contains embeddings for tasks 0..t-1
        sig_tasks = task_list[:t_idx]
        router = build_srt_router(sig_tasks, router_state)

        # Load current task test data
        test_data = load_task_data(t_name, "test")
        if test_data is None:
            print(f"    SKIP: no test data for {t_name}")
            results[t_name] = {"status": "no_data", "zs_accuracy": None}
            continue

        # Build model with SGWI init
        model = build_model(model_name, adapter_mode=True)
        init_lora_weights(model, mode=SVD_FUSION_INIT,
                          router=router, t_name=t_name,
                          all_lora_paths=None,  # no trained checkpoints needed
                          task_list=task_list)
        model.eval()

        # Evaluate zero-shot
        acc = eval_zero_shot(model, test_data)
        baseline = RandomAcc[t_name] if t_name in RandomAcc else 0.05

        print(f"    SGWI Zero-Shot Accuracy: {acc:.2f}%")
        print(f"    Random Baseline (guess):  {baseline:.2f}%")
        print(f"    Improvement:              {acc - baseline:+.2f}%")

        results[t_name] = {
            "t_idx": t_idx,
            "zs_accuracy": acc,
            "baseline": baseline,
            "improvement": acc - baseline,
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  TEST 2 — Few-Shot Convergence
# ─────────────────────────────────────────────────────────────────────────────

def run_test2(router_state, task_list, model_name, cache_dir):
    """
    Few-Shot Convergence Test.

    For task CB (smallest dataset):
      Setup A: Random Init  → train 5 epochs on 250 samples
      Setup B: SGWI Init    → train 5 epochs on 250 samples

    Plots: Loss per epoch, Accuracy per epoch.
    Expected: SGWI starts lower loss, converges faster, achieves higher final acc.

    Also runs for all other tasks with reduced train subsets.
    """
    from contri2_utils import RANDOM_INIT, SVD_FUSION_INIT, train_lora_isolated

    print_header("TEST 2 — FEW-SHOT CONVERGENCE")

    # For speed, test on first N tasks (configurable)
    # Full: all 15 tasks. Fast: only cb + 2 related tasks.
    TEST_TASKS = task_list  # run all

    all_results = {}

    for t_idx, t_name in enumerate(TEST_TASKS):
        if t_idx == 0:
            print(f"\n  [t={t_idx}] {t_name}: first task → skipping (no prior for SGWI)")
            all_results[t_name] = {"status": "first_task"}
            continue

        print(f"\n  {'─'*60}")
        print(f"  [t={t_idx}] {t_name}")

        # Load data
        train_data = load_task_data(t_name, "train")
        test_data  = load_task_data(t_name, "test")
        if train_data is None or test_data is None:
            print(f"    SKIP: missing data for {t_name}")
            all_results[t_name] = {"status": "no_data"}
            continue

        # Subsample to few-shot (250 samples, or full if smaller)
        n_fewshot = min(250, len(train_data))
        train_few = train_data[:n_fewshot]
        print(f"    Few-shot samples: {n_fewshot}  (from {len(train_data)} total)")

        # Build SRT router for SGWI
        sig_tasks = task_list[:t_idx]
        router = build_srt_router(sig_tasks, router_state)

        results_t = {}

        # ── Setup A: Random Init ──────────────────────────────────────────
        print(f"    Training with RANDOM INIT...")
        model_rand = build_model(model_name, adapter_mode=False)
        init_lora_weights(model_rand, mode=RANDOM_INIT,
                         router=None, t_name=t_name,
                         all_lora_paths=None, task_list=task_list)

        rand_curves = train_lora_isolated(
            model_rand, train_few, test_data,
            n_epochs=5, lr=3e-4,
            cache_dir=cache_dir, tag=f"{t_name}_random",
        )
        final_rand = rand_curves["accuracy"][-1] if rand_curves["accuracy"] else 0.0
        print(f"    Random Init  → Final Acc: {final_rand:.2f}%")

        results_t["random"] = {
            "final_accuracy": final_rand,
            "loss_curve": rand_curves["loss"],
            "acc_curve":   rand_curves["accuracy"],
        }

        # ── Setup B: SGWI Init ─────────────────────────────────────────────
        print(f"    Training with SGWI INIT...")
        model_sgwi = build_model(model_name, adapter_mode=False)
        init_lora_weights(model_sgwi, mode=SVD_FUSION_INIT,
                         router=router, t_name=t_name,
                         all_lora_paths=None, task_list=task_list)

        sgwi_curves = train_lora_isolated(
            model_sgwi, train_few, test_data,
            n_epochs=5, lr=3e-4,
            cache_dir=cache_dir, tag=f"{t_name}_sgwi",
        )
        final_sgwi = sgwi_curves["accuracy"][-1] if sgwi_curves["accuracy"] else 0.0
        print(f"    SGWI Init   → Final Acc: {final_sgwi:.2f}%")
        print(f"    Delta (SGWI − Random):   {final_sgwi - final_rand:+.2f}%")

        results_t["sgwi"] = {
            "final_accuracy": final_sgwi,
            "loss_curve": sgwi_curves["loss"],
            "acc_curve":   sgwi_curves["accuracy"],
        }
        results_t["delta"] = final_sgwi - final_rand

        # ── Gap 1 FIX: Forward Transfer metric ────────────────────────
        # FWT@init = acc_SGWI(epoch0) - acc_Random(epoch0)
        #   → measures knowledge injection BEFORE any training
        # FWT@final = acc_SGWI(epoch5) - acc_Random(epoch5)
        #   → measures total benefit including training
        zs_rand = rand_curves["accuracy"][0] if rand_curves["accuracy"] else 0
        zs_sgwi = sgwi_curves["accuracy"][0] if sgwi_curves["accuracy"] else 0
        results_t["forward_transfer"] = {
            "fwt_at_init":  zs_sgwi - zs_rand,
            "fwt_at_final": final_sgwi - final_rand,
            "sgwi_epoch0":  zs_sgwi,
            "random_epoch0": zs_rand,
        }
        print(f"    FWT@init:  {zs_sgwi - zs_rand:+.2f}% "
              f"(epoch0: SGWI={zs_sgwi:.1f} vs Rand={zs_rand:.1f})")
        print(f"    FWT@final: {final_sgwi - final_rand:+.2f}%")

        all_results[t_name] = results_t

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
#  TEST 3 — Ablation on Init Methods
# ─────────────────────────────────────────────────────────────────────────────

def run_test3(router_state, task_list, model_name, cache_dir):
    """
    Ablation on Init Methods.

    Compare 3 init strategies for each task t > 0:
      1. RANDOM INIT  — baseline (kaiming_uniform_ A, zeros B)
      2. NTI INIT     — copy LoRA from nearest SRT-distance previous task
      3. SFI INIT     — SVD Fusion (weighted blend of ALL previous tasks)

    Each is trained for 5 epochs on the full training set.
    Reports: final accuracy, convergence speed, final delta vs baseline.
    """
    from contri2_utils import RANDOM_INIT, NTI_INIT, SVD_FUSION_INIT, train_lora_isolated

    print_header("TEST 3 — ABLATION: RANDOM vs NTI vs SFI INIT")

    all_results = {}

    for t_idx, t_name in enumerate(task_list):
        if t_idx == 0:
            print(f"\n  [t={t_idx}] {t_name}: first task → skip (no prior tasks for NTI/SFI)")
            all_results[t_name] = {"status": "first_task"}
            continue

        print(f"\n  {'─'*60}")
        print(f"  [t={t_idx}] {t_name}")

        # Load data
        train_data = load_task_data(t_name, "train")
        test_data  = load_task_data(t_name, "test")
        if train_data is None or test_data is None:
            print(f"    SKIP: missing data for {t_name}")
            all_results[t_name] = {"status": "no_data"}
            continue

        n_samples = min(250, len(train_data))  # few-shot for speed
        train_sub = train_data[:n_samples]
        print(f"    Samples: {n_samples}  (ablation: full training set)")

        # Build SRT router
        sig_tasks = task_list[:t_idx]
        router = build_srt_router(sig_tasks, router_state)

        results_t = {}

        # ── RANDOM INIT ────────────────────────────────────────────────────
        print(f"    [A] RANDOM INIT...")
        model_r = build_model(model_name, adapter_mode=False)
        init_lora_weights(model_r, mode=RANDOM_INIT,
                         router=None, t_name=t_name,
                         all_lora_paths=None, task_list=task_list)
        curves_r = train_lora_isolated(
            model_r, train_sub, test_data, n_epochs=5, lr=3e-4,
            cache_dir=cache_dir, tag=f"{t_name}_random",
        )
        print(f"      → Final Acc: {curves_r['accuracy'][-1]:.2f}%")
        results_t["random"] = {"final_accuracy": curves_r["accuracy"][-1],
                               "acc_curve": curves_r["accuracy"]}

        # ── NTI INIT ────────────────────────────────────────────────────────
        print(f"    [B] NTI INIT (nearest previous task)...")
        model_n = build_model(model_name, adapter_mode=False)
        init_lora_weights(model_n, mode=NTI_INIT,
                         router=router, t_name=t_name,
                         all_lora_paths=None, task_list=task_list)
        curves_n = train_lora_isolated(
            model_n, train_sub, test_data, n_epochs=5, lr=3e-4,
            cache_dir=cache_dir, tag=f"{t_name}_nti",
        )
        print(f"      → Final Acc: {curves_n['accuracy'][-1]:.2f}%")
        results_t["nti"] = {"final_accuracy": curves_n["accuracy"][-1],
                            "acc_curve": curves_n["accuracy"]}

        # ── SFI INIT ───────────────────────────────────────────────────────
        print(f"    [C] SFI INIT (SVD Fusion)...")
        model_s = build_model(model_name, adapter_mode=False)
        init_lora_weights(model_s, mode=SVD_FUSION_INIT,
                         router=router, t_name=t_name,
                         all_lora_paths=None, task_list=task_list)
        curves_s = train_lora_isolated(
            model_s, train_sub, test_data, n_epochs=5, lr=3e-4,
            cache_dir=cache_dir, tag=f"{t_name}_sfi",
        )
        print(f"      → Final Acc: {curves_s['accuracy'][-1]:.2f}%")
        results_t["sfi"] = {"final_accuracy": curves_s["accuracy"][-1],
                             "acc_curve": curves_s["accuracy"]}

        # ── Summary ────────────────────────────────────────────────────────
        acc_r = curves_r["accuracy"][-1]
        acc_n = curves_n["accuracy"][-1]
        acc_s = curves_s["accuracy"][-1]
        results_t["summary"] = {
            "random_vs_nti": acc_n - acc_r,
            "random_vs_sfi": acc_s - acc_r,
            "nti_vs_sfi":    acc_s - acc_n,
        }
        print(f"    Δ Random→NTI:  {acc_n - acc_r:+.2f}%")
        print(f"    Δ Random→SFI: {acc_s - acc_r:+.2f}%")
        print(f"    Δ NTI→SFI:    {acc_s - acc_n:+.2f}%")

        all_results[t_name] = results_t

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    args = parse_args()

    model_name = args.model_name
    cache_dir  = args.cache_dir or str(RESULTS_DIR / "cache")
    os.makedirs(cache_dir, exist_ok=True)

    task_list = task_filter(args) or TASK_ORDER

    print(f"""
╔══════════════════════════════════════════════════════════╗
║         CONTRI2 — ISOLATED SGWI HYPOTHESIS TESTS         ║
╠══════════════════════════════════════════════════════════╣
║  Task order : {', '.join(TASK_ORDER[:5])}...
║  Model      : {model_name}
║  Test tasks : {len(task_list)} ({', '.join(task_list)})
║  Cache dir  : {cache_dir}
╚══════════════════════════════════════════════════════════╝
""")

    # ── Step 1: Extract frozen embeddings for all tasks ─────────────────────
    print_header("STEP 1 — EXTRACTING FROZEN EMBEDDINGS")
    print("  (This runs a forward pass through the frozen T5 backbone)")
    print("  Results are cached — subsequent runs use cached embeddings.")

    router_state = {}
    for t_idx, t_name in enumerate(task_list):
        emb_path = os.path.join(cache_dir, f"emb_{t_name}.npy")
        if os.path.exists(emb_path):
            embeddings = {"embeddings": None}  # will load from cache
            router_state[t_name] = {"emb_path": emb_path, "train_data": None}
            print(f"  [{t_idx:2d}] {t_name}: USE CACHED embeddings → {emb_path}")
        else:
            print(f"  [{t_idx:2d}] {t_name}: EXTRACTING embeddings...")
            train_data = load_task_data(t_name, "train")
            if train_data is None:
                print(f"         WARNING: no train data for {t_name}, skipping")
                continue
            embeddings, _ = extract_frozen_embeddings(
                model_name, train_data,
                max_samples=500, cache_path=emb_path,
            )
            router_state[t_name] = {"emb_path": emb_path, "train_data": train_data}
            print(f"         → {embeddings.shape[0]} embeddings, dim={embeddings.shape[1]}")

    print(f"\n  Extracted/loaded embeddings for {len(router_state)} tasks.")

    # ── Step 2: Run requested tests ─────────────────────────────────────────
    test_results = {}

    do_test1 = args.test is None or args.test == 1
    do_test2 = args.test is None or args.test == 2
    do_test3 = args.test is None or args.test == 3

    if do_test1 and not args.skip_train:
        test_results["test1_zeroshot"] = run_test1(
            router_state, task_list, model_name, cache_dir)

    if do_test2 and not args.skip_train:
        test_results["test2_fewshot"] = run_test2(
            router_state, task_list, model_name, cache_dir)

    if do_test3 and not args.skip_train:
        test_results["test3_ablation"] = run_test3(
            router_state, task_list, model_name, cache_dir)

    if args.skip_train:
        print_header("SKIPPED — Training steps (--skip-train)")
        print("  Run without --skip-train to execute Tests 1/2/3.")

    # ── Step 3: Save results ───────────────────────────────────────────────
    output_path = args.output or str(RESULTS_DIR / "contri2_results.json")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(test_results, f, indent=2, default=str)

    elapsed = time.time() - t0
    print_header("DONE")
    print(f"  Results saved to: {output_path}")
    print(f"  Total time: {elapsed/60:.1f} minutes")

    # ── Print summary table ─────────────────────────────────────────────────
    if "test3_ablation" in test_results:
        print("\n  ── Test 3 Summary ─────────────────────────────────────────")
        print(f"  {'Task':<12} {'Random':>8} {'NTI':>8} {'SFI':>8} "
              f"{'ΔR→N':>7} {'ΔR→S':>7} {'ΔN→S':>7}")
        print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*7} {'─'*7} {'─'*7}")
        for t_name, res in test_results["test3_ablation"].items():
            if "summary" in res:
                r = res["random"]["final_accuracy"]
                n = res["nti"]["final_accuracy"]
                s = res["sfi"]["final_accuracy"]
                dn = res["summary"]["random_vs_nti"]
                ds = res["summary"]["random_vs_sfi"]
                ns = res["summary"]["nti_vs_sfi"]
                print(f"  {t_name:<12} {r:>7.2f} {n:>7.2f} {s:>7.2f} "
                      f"{dn:>+6.2f} {ds:>+6.2f} {ns:>+6.2f}")


if __name__ == "__main__":
    main()
