#!/usr/bin/env python3
"""Analyze 6-config ablation results. Reads all_results.json from each config/task."""
import json, os, argparse, glob

CONFIGS = {
    'inflora':       'C1: InfLoRA (null-space+GPM)',
    'random':        'C2: No GPM, A=kaiming(frozen)',
    'full_lora':     'C3: No GPM, A=kaiming(train)',
    'sgwi_freeze_a': 'C5: No GPM, A=SGWI(frozen)',
    'sgwi_train_a':  'C6: No GPM, A=SGWI(train)',
    'sgwi_full':     'C4: No GPM, A=SGWI(train)+B=SGWI',
}

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def get_score(data, task):
    if data is None:
        return None
    for k in [f'predict_exact_match_for_{task}', 'predict_exact_match']:
        if k in data:
            v = data[k]
            return v * 100 if v <= 1.0 else v
    return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', default='logs_and_outputs/ablation_order4')
    args = p.parse_args()

    # Detect tasks from directory structure
    tasks = set()
    for cfg in CONFIGS:
        cfg_dir = os.path.join(args.base, cfg)
        if os.path.isdir(cfg_dir):
            for d in os.listdir(cfg_dir):
                if '-' in d:
                    tasks.add(d.split('-', 1)[1])
    tasks = sorted(tasks)
    last_task = tasks[-1] if tasks else 'cb'

    print("=" * 80)
    print(f"ABLATION RESULTS — {args.base}")
    print(f"Tasks: {tasks}, Eval target: {last_task}")
    print("=" * 80)

    # Collect scores for last task
    results = {}
    for cfg_key, cfg_label in CONFIGS.items():
        cfg_dir = os.path.join(args.base, cfg_key)
        # Find the last task directory
        last_dir = None
        for d in sorted(glob.glob(os.path.join(cfg_dir, f'*-{last_task}'))):
            last_dir = d
        if last_dir is None:
            results[cfg_key] = {'label': cfg_label, 'cb': None, 'train_loss': None, 'all_tasks': {}}
            continue

        data = load_json(os.path.join(last_dir, 'all_results.json'))
        cb = get_score(data, last_task)
        tl = data.get('train_loss') if data else None

        # Also collect all task scores from the last task's predict
        all_task_scores = {}
        if data:
            for t in tasks:
                s = get_score(data, t)
                if s is not None:
                    all_task_scores[t] = s

        results[cfg_key] = {'label': cfg_label, 'cb': cb, 'train_loss': tl, 'all_tasks': all_task_scores}

    # Print table
    print(f"\n{'Config':35s} | {last_task+' EM':>8s} | {'TrainLoss':>10s} | Per-task scores")
    print(f"{'-'*35}-+-{'-'*8}-+-{'-'*10}-+-{'-'*30}")

    for cfg_key in CONFIGS:
        r = results[cfg_key]
        cb_str = f"{r['cb']:.2f}%" if r['cb'] is not None else "  N/A  "
        tl_str = f"{r['train_loss']:.4f}" if r['train_loss'] is not None else "   N/A   "
        per_task = " | ".join(f"{t}={r['all_tasks'].get(t, 0):.1f}" for t in tasks if t in r['all_tasks'])
        print(f"{r['label']:35s} | {cb_str:>8s} | {tl_str:>10s} | {per_task}")

    # Ablation comparisons
    print(f"\nABLATION COMPARISONS (on {last_task}):")
    print("-" * 60)
    pairs = [
        ('inflora', 'random', 'C1→C2: Remove null-space'),
        ('random', 'sgwi_freeze_a', 'C2→C5: SGWI-A vs random-A (frozen)'),
        ('random', 'full_lora', 'C2→C3: Unfreeze A (random)'),
        ('sgwi_freeze_a', 'sgwi_train_a', 'C5→C6: SGWI-A frozen vs trainable'),
        ('full_lora', 'sgwi_train_a', 'C3→C6: SGWI-A vs random-A (trainable)'),
        ('sgwi_train_a', 'sgwi_full', 'C6→C4: Add SGWI-B warm-init'),
    ]
    for a, b, label in pairs:
        sa = results[a]['cb']
        sb = results[b]['cb']
        if sa is not None and sb is not None:
            diff = sb - sa
            marker = "✅" if diff > 1 else ("⚠️" if diff > -1 else "❌")
            print(f"  {marker} {label}: {diff:+.2f}pp ({sa:.2f}→{sb:.2f})")
        else:
            print(f"  ⏳ {label}: missing data")

if __name__ == '__main__':
    main()
