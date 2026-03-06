#!/usr/bin/env python3
"""
compute_ap_ft.py — Compute AP and FT from training output directories
======================================================================
Usage:
    python compute_ap_ft.py \\
        --output_base logs_and_outputs/ot_sign_order1/outputs \\
        --task_order "task1572_samsum_summary,task363_sst2_polarity_classification,..." \\
        --method_name "OT-SIGN+GainLoRA(InfLoRA)"

AP  = (1/T) * Σ_j R[T-1][j]            — avg score on ALL tasks after training last task
FT  = (1/(T-1)) * Σ_j (R[j][j] - R[T-1][j])  — avg performance DROP from peak

Result matrix R[i][j]:
  R[i][j] = score on task j right after model was trained on task i
  Diagonal R[i][i] = peak performance on task i
  Last row R[T-1][j] = final performance (what we report as AP)

Score file locations (HuggingFace Trainer):
  {output_base}/{i+1}-{task_name}/predict_results.json   → has all task scores
  {output_base}/{i+1}-{task_name}/eval_results.json     → used as fallback peak
"""

import argparse
import json
import os
import re
import sys


# ── Metric key detection ────────────────────────────────────────────────────

def _is_long_benchmark(task_list):
    long_tasks = {'yelp', 'amazon', 'mnli', 'imdb', 'sst2', 'dbpedia', 'agnews',
                  'yahoo', 'multirc', 'boolq', 'wic', 'copa', 'qqp', 'rte', 'cb'}
    return any(t.lower() in long_tasks for t in task_list)


def _get_score(metrics, task_name, is_long):
    """Extract score for `task_name` from a metrics dict."""
    prefix = "exact_match" if is_long else "rougeL"
    candidates = [
        f"predict_{prefix}_for_{task_name}",
        f"eval_{prefix}_for_{task_name}",
        f"predict_{prefix}",
        f"eval_{prefix}",
    ]
    for key in candidates:
        if key in metrics:
            v = float(metrics[key])
            # rougeL from compute_metrics is 0-1 scale → convert to %
            if v <= 1.0 and "rouge" in key.lower():
                v = v * 100.0
            return round(v, 2)

    # broad fallback: any key mentioning the task name
    for k, v in metrics.items():
        if task_name in k and isinstance(v, (int, float)):
            v = float(v)
            if v <= 1.0 and "rouge" in k.lower():
                v = v * 100.0
            return round(v, 2)
    return None


def _load_json_file(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


# ── Result matrix builder ────────────────────────────────────────────────────

def build_result_matrix(output_base, task_order, is_long):
    """
    R[i][j] = score on task j after training on task i.
    R[i][i] = peak performance on task i (from predict_results when trained on i).
    """
    T = len(task_order)
    R = {}

    for i, task_i in enumerate(task_order):
        # Find output directory for this task (numbered 1-based)
        task_dir = _find_task_dir(output_base, i + 1, task_i)
        if task_dir is None:
            print(f"  [WARN] Directory for task {i+1} ({task_i}) not found — skipping", file=sys.stderr)
            continue

        predict_file = os.path.join(task_dir, "predict_results.json")
        metrics = _load_json_file(predict_file)

        if not metrics:
            # Try eval_results.json as fallback
            metrics = _load_json_file(os.path.join(task_dir, "eval_results.json"))

        if not metrics:
            print(f"  [WARN] No metrics found in {task_dir}", file=sys.stderr)
            continue

        R[i] = {}
        # Extract score for EACH task visible at this training step (tasks 0..i)
        for j in range(i + 1):
            score = _get_score(metrics, task_order[j], is_long)
            if score is not None:
                R[i][j] = score

    return R


def _find_task_dir(output_base, task_num, task_name):
    """Locate task output dir robustly: tries exact match, then prefix match."""
    # Try exact name: "1-task1572_samsum_summary"
    candidate = os.path.join(output_base, f"{task_num}-{task_name}")
    if os.path.isdir(candidate):
        return candidate

    # Try prefix match
    if os.path.isdir(output_base):
        prefix = f"{task_num}-"
        matches = [d for d in os.listdir(output_base)
                   if d.startswith(prefix) and os.path.isdir(os.path.join(output_base, d))]
        if matches:
            return os.path.join(output_base, sorted(matches)[0])
    return None


# ── AP / FT computation ───────────────────────────────────────────────────────

def compute_ap_ft(R, task_order):
    """
    Args:
        R: dict {i: {j: score}} — result matrix
        task_order: list of task names
    Returns:
        ap, ft as floats (already rounded to 2 dp)
    """
    T = len(task_order)
    final_row = T - 1

    # AP: average over final row
    final_scores = [R.get(final_row, {}).get(j) for j in range(T)]
    valid_final = [s for s in final_scores if s is not None]
    ap = round(sum(valid_final) / len(valid_final), 2) if valid_final else 0.0

    # FT: mean drop from diagonal to final row (exclude last task by convention)
    drops = []
    for j in range(T - 1):
        diag  = R.get(j, {}).get(j)
        final = R.get(final_row, {}).get(j)
        if diag is not None and final is not None:
            drops.append(diag - final)
    ft = round(sum(drops) / len(drops), 2) if drops else 0.0

    return ap, ft


# ── Output formatter ──────────────────────────────────────────────────────────

def print_comparison_table(R, task_order, method_name, ap, ft):
    T = len(task_order)
    final_row = T - 1
    sep = "─" * 72

    print(f"\n{'═' * 72}")
    print(f"  {method_name}")
    print(f"{'═' * 72}")
    print(f"  {'#':<3} {'Task':<46} {'Peak':>6}  {'Final':>6}  {'Drop':>6}")
    print(sep)

    for j, tname in enumerate(task_order):
        diag  = R.get(j, {}).get(j)
        final = R.get(final_row, {}).get(j)
        drop  = round(diag - final, 2) if diag is not None and final is not None else None
        d_str = f"{diag:.2f}"  if diag  is not None else " ——"
        f_str = f"{final:.2f}" if final is not None else " ——"
        r_str = f"{drop:.2f}"  if drop  is not None else " ——"
        print(f"  {j+1:<3} {tname:<46} {d_str:>6}  {f_str:>6}  {r_str:>6}")

    print(sep)
    print(f"  AP  = {ap:>6.2f}   │   FT = {ft:>5.2f}")
    print(f"{'═' * 72}\n")


def save_results_json(R, task_order, method_name, ap, ft, output_base):
    out = {
        "method": method_name,
        "ap": ap,
        "ft": ft,
        "per_task": {}
    }
    T = len(task_order)
    final_row = T - 1
    for j, tname in enumerate(task_order):
        out["per_task"][tname] = {
            "peak":  R.get(j, {}).get(j),
            "final": R.get(final_row, {}).get(j),
        }

    result_path = os.path.join(output_base, "..", "ap_ft_result.json")
    result_path = os.path.normpath(result_path)
    with open(result_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"[compute_ap_ft] Saved results to {result_path}")
    return result_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compute AP/FT from GainLoRA output directories")
    parser.add_argument("--output_base", required=True,
                        help="Path to the outputs/ directory containing task subdirectories")
    parser.add_argument("--task_order", required=True,
                        help="Comma-separated list of task names in training order")
    parser.add_argument("--method_name", default="OT-SIGN+GainLoRA",
                        help="Label shown in the result table")
    parser.add_argument("--save", action="store_true",
                        help="Also save results to ap_ft_result.json")
    args = parser.parse_args()

    task_order = [t.strip() for t in args.task_order.split(',') if t.strip()]
    is_long = _is_long_benchmark(task_order)

    print(f"[compute_ap_ft] Benchmark type: {'long' if is_long else 'superni'}")
    print(f"[compute_ap_ft] Tasks: {len(task_order)}  |  Output base: {args.output_base}")

    R = build_result_matrix(args.output_base, task_order, is_long)

    if not R:
        print("[compute_ap_ft] ERROR: No results found. Check --output_base path.", file=sys.stderr)
        sys.exit(1)

    ap, ft = compute_ap_ft(R, task_order)
    print_comparison_table(R, task_order, args.method_name, ap, ft)

    if args.save:
        save_results_json(R, task_order, args.method_name, ap, ft, args.output_base)


if __name__ == "__main__":
    main()
