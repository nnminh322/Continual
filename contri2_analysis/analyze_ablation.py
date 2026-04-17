#!/usr/bin/env python3
"""Analyze 6-config ablation results with proper per-task breakdown."""
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
    key = f'predict_exact_match_for_{task}'
    if key in data:
        v = data[key]
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
    
    # Determine task order from directory numbering
    task_order = []
    sample_cfg = list(CONFIGS.keys())[0]
    cfg_dir = os.path.join(args.base, sample_cfg)
    if os.path.isdir(cfg_dir):
        dirs = sorted(os.listdir(cfg_dir))
        for d in dirs:
            if '-' in d:
                task_order.append(d.split('-', 1)[1])
    if not task_order:
        task_order = tasks
    
    first_task = task_order[0] if task_order else tasks[0]
    last_task = task_order[-1] if task_order else tasks[-1]

    print("=" * 95)
    print(f"ABLATION RESULTS — {args.base}")
    print(f"Task sequence: {' → '.join(task_order)}")
    print(f"Eval point: after training task '{last_task}', evaluate ALL tasks")
    print("=" * 95)

    # Collect scores from last task's eval
    results = {}
    for cfg_key, cfg_label in CONFIGS.items():
        cfg_dir = os.path.join(args.base, cfg_key)
        last_dir = None
        for d in sorted(glob.glob(os.path.join(cfg_dir, f'*-{last_task}'))):
            last_dir = d
        
        entry = {'label': cfg_label, 'tasks': {}, 'train_loss': None, 'cl_avg': None}
        
        if last_dir:
            data = load_json(os.path.join(last_dir, 'all_results.json'))
            if data:
                entry['train_loss'] = data.get('train_loss')
                entry['cl_avg'] = get_score(data, 'CL')
                for t in task_order:
                    s = get_score(data, t)
                    if s is not None:
                        entry['tasks'][t] = s
        
        results[cfg_key] = entry

    # ── Table 1: Full per-task breakdown ──
    task_headers = "".join(f" | {t:>7s}" for t in task_order)
    print(f"\n{'Config':35s}{task_headers} | {'CL avg':>7s} | {'Loss':>7s}")
    print("-" * (35 + 11 * len(task_order) + 21))

    for cfg_key in CONFIGS:
        r = results[cfg_key]
        row = f"{r['label']:35s}"
        for t in task_order:
            s = r['tasks'].get(t)
            row += f" | {s:7.2f}" if s is not None else " |    N/A"
        cl = r['cl_avg']
        row += f" | {cl:7.2f}" if cl is not None else " |    N/A"
        tl = r['train_loss']
        row += f" | {tl:7.4f}" if tl is not None else " |    N/A"
        print(row)

    # ── Table 2: Key contrasts ──
    print(f"\n{'='*70}")
    print("KEY ABLATION CONTRASTS")
    print(f"{'='*70}")
    
    pairs = [
        ('inflora', 'random', 'Remove GPM/null-space', 'C1→C2'),
        ('random', 'sgwi_freeze_a', 'SGWI-A vs random-A (frozen)', 'C2→C5'),
        ('random', 'full_lora', 'Unfreeze A (random init)', 'C2→C3'),
        ('sgwi_freeze_a', 'sgwi_train_a', 'Unfreeze A (SGWI init)', 'C5→C6'),
        ('full_lora', 'sgwi_train_a', 'SGWI-A vs random-A (trainable)', 'C3→C6'),
        ('sgwi_train_a', 'sgwi_full', 'Add SGWI-B warm-init', 'C6→C4'),
    ]
    
    for a, b, desc, arrow in pairs:
        ra, rb = results[a], results[b]
        print(f"\n  {arrow}: {desc}")
        for t in task_order:
            sa = ra['tasks'].get(t)
            sb = rb['tasks'].get(t)
            if sa is not None and sb is not None:
                diff = sb - sa
                marker = "✅" if diff > 1 else ("⚠️" if abs(diff) <= 1 else "❌")
                print(f"    {marker} {t:>6s}: {sa:6.2f} → {sb:6.2f}  ({diff:+.2f}pp)")

    # ── Highlight the key finding ──
    print(f"\n{'='*70}")
    print("💡 KEY FINDING")
    print(f"{'='*70}")
    
    c4 = results.get('sgwi_full', {}).get('tasks', {})
    c6 = results.get('sgwi_train_a', {}).get('tasks', {})
    c1 = results.get('inflora', {}).get('tasks', {})
    
    if last_task in c4 and last_task in c6:
        delta = c4[last_task] - c6[last_task]
        print(f"  SGWI-B warm-init (C4 vs C6) on {last_task}: +{delta:.2f}pp")
        print(f"    C6 (no B warm): {c6.get(last_task, 'N/A'):.2f}%")
        print(f"    C4 (with B warm): {c4.get(last_task, 'N/A'):.2f}%")
    
    if first_task in c4 and first_task in c1:
        d_retain = c4[first_task] - c1[first_task]
        print(f"  {first_task} retention (C4 vs C1): {d_retain:+.2f}pp")
        print(f"    C1 (InfLoRA+GPM): {c1.get(first_task, 'N/A'):.2f}%")
        print(f"    C4 (SGWI full):   {c4.get(first_task, 'N/A'):.2f}%")

if __name__ == '__main__':
    main()
