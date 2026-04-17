#!/usr/bin/env python3
"""
Auto-analyze hypothesis testing results and print decisions.
Usage: python ../contri2_analysis/analyze_results.py --phase <1|3|4> --output_base <path>

Designed for C2 Hypothesis Testing Plan:
  Phase 1: 4-arm CB initialization comparison → Q1, Q2, O1
  Phase 3: Dual Fisher λ_emb sweep → Q3, O2
  Phase 4: 5-task end-to-end → Q5
"""
import json
import os
import argparse
import glob
from pathlib import Path


def load_results_json(result_dir):
    """Load all_results.json from a result directory, return full dict or None."""
    results_file = os.path.join(result_dir, 'all_results.json')
    if not os.path.exists(results_file):
        for f in glob.glob(os.path.join(result_dir, '**/all_results.json'), recursive=True):
            results_file = f
            break
    
    if not os.path.exists(results_file):
        return None
    
    with open(results_file) as f:
        return json.load(f)


def extract_score(data, key_priority):
    """Extract first matching score from results dict."""
    if data is None:
        return None
    for key in key_priority:
        if key in data:
            val = data[key]
            if isinstance(val, (int, float)):
                return val * 100 if val <= 1.0 else val
    return None


def load_score(result_dir, key_priority=None):
    """Load best score from all_results.json in a result directory."""
    if key_priority is None:
        key_priority = [
            'predict_exact_match_for_cb',
            'eval_exact_match_for_cb',
            'predict_exact_match',
            'eval_exact_match',
        ]
    data = load_results_json(result_dir)
    return extract_score(data, key_priority)


def load_training_logs(result_dir):
    """Load training logs for convergence analysis."""
    log_file = os.path.join(result_dir, 'trainer_state.json')
    if not os.path.exists(log_file):
        for f in glob.glob(os.path.join(result_dir, '**/trainer_state.json'), recursive=True):
            log_file = f
            break
    
    if not os.path.exists(log_file):
        return None
    
    with open(log_file) as f:
        data = json.load(f)
    
    return data.get('log_history', [])


def analyze_phase1(output_base):
    """Phase 1: 4-arm CB comparison → Q1, Q2, Q4, O1"""
    arms = {
        'A_inflora': f'{output_base}/2-cb_arm_a_inflora',
        'B_sgwi': f'{output_base}/2-cb_arm_b_sgwi',
        'C_sgwi_inflora': f'{output_base}/2-cb_arm_c_sgwi_inflora',
        'D_random': f'{output_base}/2-cb_arm_d_random',
    }
    
    # CB-specific metric is what matters for task 2 performance
    CB_KEYS = ['predict_exact_match_for_cb', 'eval_exact_match_for_cb']
    MNLI_KEYS = ['predict_exact_match_for_mnli', 'eval_exact_match_for_mnli']
    OVERALL_KEYS = ['predict_exact_match', 'eval_exact_match']
    
    cb_scores = {}
    mnli_scores = {}
    overall_scores = {}
    train_losses = {}
    
    for arm_name, arm_path in arms.items():
        data = load_results_json(arm_path)
        if data is not None:
            cb = extract_score(data, CB_KEYS)
            mnli = extract_score(data, MNLI_KEYS)
            ovr = extract_score(data, OVERALL_KEYS)
            if cb is not None:
                cb_scores[arm_name] = cb
            if mnli is not None:
                mnli_scores[arm_name] = mnli
            if ovr is not None:
                overall_scores[arm_name] = ovr
            if 'train_loss' in data:
                train_losses[arm_name] = data['train_loss']
    
    print("=" * 60)
    print("PHASE 1 RESULTS: CB Initialization Comparison")
    print("=" * 60)
    
    if not cb_scores and not overall_scores:
        print("  ERROR: No results found. Check output paths.")
        print(f"  Expected paths: {list(arms.values())}")
        return
    
    # Use CB-specific scores; fall back to overall if not available
    scores = cb_scores if cb_scores else overall_scores
    score_label = "CB" if cb_scores else "Overall"
    
    print(f"\n  {'Arm':25s} | {score_label+' EM':>8s} | {'MNLI EM':>8s} | {'Train Loss':>10s}")
    print(f"  {'-'*25}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")
    for arm in ['A_inflora', 'B_sgwi', 'C_sgwi_inflora', 'D_random']:
        cb = f"{scores.get(arm, 0):.2f}%" if arm in scores else "  N/A  "
        mnli = f"{mnli_scores.get(arm, 0):.2f}%" if arm in mnli_scores else "  N/A  "
        tloss = f"{train_losses.get(arm, 0):.4f}" if arm in train_losses else "   N/A   "
        print(f"  {arm:25s} | {cb:>8s} | {mnli:>8s} | {tloss:>10s}")
    print()
    
    a = scores.get('A_inflora')
    b = scores.get('B_sgwi')
    c = scores.get('C_sgwi_inflora')
    d = scores.get('D_random')
    
    print("DECISIONS:")
    print("-" * 40)
    
    if a is not None and b is not None:
        diff = b - a
        if diff > 1.0:
            print(f"  Q1: ✅ YES — SGWI improves CB by {diff:.2f}pp over InfLoRA baseline")
        elif diff > -1.0:
            print(f"  Q1: ⚠️  MARGINAL — SGWI vs InfLoRA difference = {diff:+.2f}pp (within ±1pp)")
        else:
            print(f"  Q1: ❌ NO — SGWI HURTS CB by {-diff:.2f}pp. Consider STOPPING C2.")
    else:
        print(f"  Q1: ⏳ Cannot determine (missing A or B scores)")
    
    if b is not None and c is not None:
        diff = b - c
        if diff > 1.0:
            print(f"  Q2: ✅ YES — InfLoRA conflicts with SGWI (SGWI-only better by {diff:.2f}pp)")
        elif diff < -1.0:
            print(f"  Q2: ❌ NO — InfLoRA actually HELPS SGWI (combined better by {-diff:.2f}pp)")
        else:
            print(f"  Q2: ❌ NO — No significant conflict (diff = {diff:+.2f}pp)")
    else:
        print(f"  Q2: ⏳ Cannot determine (missing B or C scores)")
    
    if scores:
        best_arm = max(scores, key=scores.get)
        print(f"  O1: Best init mode = {best_arm} ({scores[best_arm]:.2f}%)")
    
    print()
    print("CONVERGENCE ANALYSIS:")
    print("-" * 40)
    for arm_name, arm_path in arms.items():
        logs = load_training_logs(arm_path)
        if logs:
            losses = [(l.get('step', 0), l.get('loss', None)) for l in logs if 'loss' in l]
            if losses:
                first_loss = losses[0][1] if losses else None
                min_loss = min(l[1] for l in losses)
                print(f"  {arm_name:25s}: first_loss={first_loss:.4f}, min_loss={min_loss:.4f}, steps={losses[-1][0]}")
            else:
                print(f"  {arm_name:25s}: no loss data found")
        else:
            print(f"  {arm_name:25s}: no training logs found")
    
    print()
    print("NEXT STEPS:")
    print("-" * 40)
    if a is not None and b is not None and b >= a - 1.0:
        print("  → PROCEED to Phase 3 (Dual Fisher λ_emb sweep)")
        print(f"  → Use init mode: {'B_sgwi' if b >= a else 'A_inflora'}")
    else:
        print("  → STOP. SGWI doesn't improve over baseline. Reconsider C2 direction.")


def analyze_phase3(output_base):
    """Phase 3: λ_emb sweep → Q3, O2"""
    scores = {}
    
    arm_b_score = load_score(f'{output_base}/2-cb_arm_b_sgwi')
    if arm_b_score is not None:
        scores[0.0] = arm_b_score
    
    for l in [0.001, 0.005, 0.01, 0.05]:
        lambda_tag = str(l).replace('.', 'p')
        score = load_score(f'{output_base}/2-cb_lambda_{lambda_tag}')
        if score is not None:
            scores[l] = score
    
    print("=" * 60)
    print("PHASE 3 RESULTS: Dual Fisher λ_emb Sweep")
    print("=" * 60)
    
    if not scores:
        print("  ERROR: No results found.")
        return
    
    for l in sorted(scores.keys()):
        marker = " ← SGWI baseline (no Fisher)" if l == 0 else ""
        print(f"  λ_emb = {l:7.4f}: {scores[l]:.2f}%{marker}")
    print()
    
    baseline = scores.get(0.0, 0)
    nonzero_scores = {k: v for k, v in scores.items() if k > 0}
    
    print("DECISIONS:")
    print("-" * 40)
    
    if nonzero_scores:
        best_lambda = max(nonzero_scores, key=nonzero_scores.get)
        best_score = nonzero_scores[best_lambda]
        
        if best_score > baseline + 0.5:
            print(f"  Q3: ✅ YES — Dual Fisher improves by {best_score - baseline:.2f}pp (λ={best_lambda})")
        elif best_score > baseline:
            print(f"  Q3: ⚠️  MARGINAL — Dual Fisher improves by {best_score - baseline:.2f}pp (< 0.5pp)")
        else:
            print(f"  Q3: ❌ NO — Dual Fisher doesn't add value (best improvement = {best_score - baseline:.2f}pp)")
        
        overall_best = max(scores, key=scores.get)
        print(f"  O2: Best λ_emb = {overall_best} ({scores[overall_best]:.2f}%)")
    else:
        print(f"  Q3: ⏳ No Dual Fisher results yet")
    
    if len(scores) >= 3:
        sorted_lambdas = sorted(scores.keys())
        sorted_scores = [scores[l] for l in sorted_lambdas]
        peak_idx = sorted_scores.index(max(sorted_scores))
        
        print()
        print("TREND ANALYSIS:")
        print("-" * 40)
        if peak_idx == 0:
            print("  → λ=0 is best: regularization HURTS. Dual Fisher not needed.")
            print("  → C2 = SGWI only (simpler, still valid contribution)")
        elif 0 < peak_idx < len(sorted_scores) - 1:
            print(f"  → Inverted U-shape confirmed: optimal λ at interior point ({sorted_lambdas[peak_idx]})")
            print("  → Consistent with theory: too low = no effect, too high = plasticity loss")
        else:
            print(f"  → Monotonically increasing: may need to test higher λ values")
    
    print()
    print("NEXT STEPS:")
    print("-" * 40)
    if nonzero_scores:
        best_overall = max(scores, key=scores.get)
        print(f"  → Best config for Phase 4: SGWI + λ_emb={best_overall}")
        print(f"  → PROCEED to Phase 4 (5-task end-to-end validation)")
    else:
        print(f"  → Run Phase 3 experiments first")


def analyze_phase4(output_base):
    """Phase 4: 5-task end-to-end → Q5"""
    configs = {
        'SRT_baseline': f'{output_base}/phase4_baseline',
        'C2_full': f'{output_base}/phase4_c2',
    }
    
    print("=" * 60)
    print("PHASE 4 RESULTS: 5-Task End-to-End Validation")
    print("=" * 60)
    
    for config_name, config_path in configs.items():
        score_file = os.path.join(config_path, 'cl_metrics.json')
        if os.path.exists(score_file):
            with open(score_file) as f:
                metrics = json.load(f)
            print(f"\n  {config_name}:")
            print(f"    AP:  {metrics.get('Cl', 'N/A'):.2f}%")
            print(f"    BWT: {metrics.get('Bwt', 'N/A'):.2f}%")
            print(f"    FWT: {metrics.get('Fwt', 'N/A'):.2f}%")
            print(f"    Fgt: {metrics.get('Fgt', 'N/A'):.2f}%")
        else:
            print(f"\n  {config_name}: No results found at {score_file}")


def analyze_all(output_base):
    """Run all analyses that have data available."""
    print("\n" + "=" * 60)
    print("C2 HYPOTHESIS TESTING — FULL REPORT")
    print("=" * 60 + "\n")
    
    phase1_exists = any(
        os.path.exists(f'{output_base}/2-cb_arm_{arm}')
        for arm in ['a_inflora', 'b_sgwi', 'c_sgwi_inflora', 'd_random']
    )
    phase3_exists = any(
        os.path.exists(f'{output_base}/2-cb_lambda_{str(l).replace(".", "p")}')
        for l in [0.001, 0.005, 0.01, 0.05]
    )
    phase4_exists = os.path.exists(f'{output_base}/phase4_baseline')
    
    if phase1_exists:
        analyze_phase1(output_base)
        print()
    if phase3_exists:
        analyze_phase3(output_base)
        print()
    if phase4_exists:
        analyze_phase4(output_base)
        print()
    
    if not any([phase1_exists, phase3_exists, phase4_exists]):
        print("No results found. Run hypothesis testing experiments first.")
        print(f"Expected output base: {output_base}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze C2 Hypothesis Testing Results')
    parser.add_argument('--phase', type=int, choices=[1, 3, 4, 0],
                        default=0, help='Phase to analyze (0=all)')
    parser.add_argument('--output_base', type=str,
                        default='logs_and_outputs/c2_hypothesis',
                        help='Base output directory')
    args = parser.parse_args()
    
    if args.phase == 0:
        analyze_all(args.output_base)
    elif args.phase == 1:
        analyze_phase1(args.output_base)
    elif args.phase == 3:
        analyze_phase3(args.output_base)
    elif args.phase == 4:
        analyze_phase4(args.output_base)
