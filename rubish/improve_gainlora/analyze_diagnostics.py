#!/usr/bin/env python3
"""Post-hoc diagnostic analysis for SpecRoute experiments.

Reads init_diagnostics.pt and routing_decisions.pt from each task output
and prints a comprehensive report.

Usage:
    python analyze_diagnostics.py <run_name>
    python analyze_diagnostics.py gen_script_long_order3_t5_specroute

Output:
    Per-task CPI/OAP diagnostics (ρ_l, β_l, SSE, λ_min+, eigenvalue health)
    Per-task routing accuracy (p_e estimate from routing decisions)
"""
import os, sys, torch, json
import numpy as np

def load_diag(task_dir):
    """Load diagnostics from a task output directory."""
    diag_path = os.path.join(task_dir, 'init_diagnostics.pt')
    routing_path = os.path.join(task_dir, 'saved_weights', 'routing_decisions.pt')
    init_diag = torch.load(diag_path, map_location='cpu') if os.path.exists(diag_path) else None
    routing = torch.load(routing_path, map_location='cpu') if os.path.exists(routing_path) else None
    return init_diag, routing


def analyze_init_diag(init_diag, task_name, task_idx):
    """Analyze CPI/OAP initialization diagnostics."""
    if init_diag is None:
        print(f'  [INIT] No init diagnostics (task 1 or missing file)')
        return {}

    all_rho, all_beta, all_sse_b, all_sse_a = [], [], [], []
    all_lmin, all_npos, all_ntotal = [], [], []

    for layer_idx, layer_diag in enumerate(init_diag):
        for chunk_idx, d in layer_diag.items():
            all_rho.append(d['rho_l'])
            all_beta.append(d['beta_l'])
            all_sse_b.append(d['sse_before'])
            all_sse_a.append(d['sse_after'])
            all_lmin.append(d['lambda_min_pos_over_r'])
            all_npos.append(d['n_pos_eigvals'])
            all_ntotal.append(d['n_total_eigvals'])

    summary = {
        'rho_l_mean': np.mean(all_rho),
        'rho_l_max': np.max(all_rho),
        'beta_l_mean': np.mean(all_beta),
        'beta_l_min': np.min(all_beta),
        'sse_before_mean': np.mean(all_sse_b),
        'sse_after_mean': np.mean(all_sse_a),
        'lambda_min_pos_over_r_mean': np.mean(all_lmin),
        'lambda_min_pos_over_r_min': np.min(all_lmin),
        'n_pos_mean': np.mean(all_npos),
        'n_total': all_ntotal[0] if all_ntotal else 0,
        'n_layers': len(init_diag),
    }

    print(f'  [INIT] ρ_l: mean={summary["rho_l_mean"]:.4f} max={summary["rho_l_max"]:.4f}')
    print(f'         β_l: mean={summary["beta_l_mean"]:.4f} min={summary["beta_l_min"]:.4f}')
    print(f'         SSE: {summary["sse_before_mean"]:.4f} → {summary["sse_after_mean"]:.4f} (OAP reduction)')
    print(f'         λ_min+/r: mean={summary["lambda_min_pos_over_r_mean"]:.6f} min={summary["lambda_min_pos_over_r_min"]:.6f} (Thm3 routing margin)')
    print(f'         Eigenvalues: {summary["n_pos_mean"]:.1f}/{summary["n_total"]} positive (avg across layers)')

    # Health checks
    if summary['lambda_min_pos_over_r_min'] < 1e-5:
        print(f'  ⚠  WARNING: λ_min+/r very small → CPI routing margin weak for some layers')
    if summary['n_pos_mean'] < 8:
        print(f'  ⚠  WARNING: Few positive eigenvalues → CPI falling back to Kaiming on some layers')

    return summary


def analyze_routing(routing, task_name, task_idx, n_old_tasks):
    """Analyze routing decisions for p_e estimation."""
    if routing is None:
        print(f'  [ROUTE] No routing data (first task or missing file)')
        return {}

    n_total = len(routing)
    # In SpecRoute, current task is always index 0 at prediction time.
    # But during eval, the model evaluates on the CURRENT task's test set,
    # so correct routing = index 0 (current expert).
    routed_to_current = (routing == 0).float().mean().item()
    p_e = 1.0 - routed_to_current

    summary = {
        'n_samples': n_total,
        'routed_to_current': routed_to_current,
        'p_e': p_e,
        'n_tasks_available': n_old_tasks + 1,
    }

    print(f'  [ROUTE] Routed to current: {routed_to_current:.3f} ({int((routing==0).sum())}/{n_total})')
    print(f'          p_e (routing error): {p_e:.3f} (n_tasks={n_old_tasks+1})')

    # Distribution
    for t in range(n_old_tasks + 1):
        frac = (routing == t).float().mean().item()
        if frac > 0.001:
            label = 'CURRENT' if t == 0 else f'old_{t}'
            print(f'          task_idx={t} ({label}): {frac:.3f}')

    # Health checks
    if p_e > 0.3:
        print(f'  ⚠  WARNING: High routing error (p_e={p_e:.3f}) — forgetting risk')
    if p_e > 0.5:
        print(f'  🚨 CRITICAL: Routing worse than random for 2-class! Method likely failing.')

    return summary


def main():
    if len(sys.argv) < 2:
        print('Usage: python analyze_diagnostics.py <run_name>')
        print('  e.g.: python analyze_diagnostics.py gen_script_long_order3_t5_specroute')
        sys.exit(1)

    run_name = sys.argv[1]
    base_dir = os.path.join('logs_and_outputs', run_name, 'outputs')

    if not os.path.isdir(base_dir):
        print(f'ERROR: {base_dir} not found')
        sys.exit(1)

    # Discover task directories (sorted by task index)
    task_dirs = sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d[0].isdigit()
    ], key=lambda x: int(x.split('-')[0]))

    print('=' * 70)
    print(f'SpecRoute Diagnostic Report: {run_name}')
    print(f'Tasks found: {len(task_dirs)}')
    print('=' * 70)

    all_summaries = []

    for task_dir_name in task_dirs:
        task_dir = os.path.join(base_dir, task_dir_name)
        task_idx = int(task_dir_name.split('-')[0])
        task_name = '-'.join(task_dir_name.split('-')[1:])

        print(f'\n--- Task {task_idx}: {task_name} ---')

        # Try saved_weights subdir first, then task_dir itself
        sw_dir = os.path.join(task_dir, 'saved_weights')
        diag_dir = sw_dir if os.path.isdir(sw_dir) else task_dir

        init_diag_path = os.path.join(diag_dir, 'init_diagnostics.pt')
        init_diag = torch.load(init_diag_path, map_location='cpu') if os.path.exists(init_diag_path) else None

        routing_path = os.path.join(diag_dir, 'routing_decisions.pt')
        routing = torch.load(routing_path, map_location='cpu') if os.path.exists(routing_path) else None

        init_summary = analyze_init_diag(init_diag, task_name, task_idx)
        route_summary = analyze_routing(routing, task_name, task_idx, task_idx - 1)

        all_summaries.append({
            'task_idx': task_idx, 'task_name': task_name,
            'init': init_summary, 'routing': route_summary,
        })

    # Global summary table
    print('\n' + '=' * 70)
    print('SUMMARY TABLE')
    print('=' * 70)
    print(f'{"Task":<20} {"ρ_l":>6} {"β_l":>6} {"SSE→":>10} {"λ+/r":>8} {"p_e":>6} {"n_pos":>6}')
    print('-' * 70)
    for s in all_summaries:
        i = s['init']
        r = s['routing']
        rho = f'{i.get("rho_l_mean", 0):.3f}' if i else '-'
        beta = f'{i.get("beta_l_mean", 0):.3f}' if i else '-'
        sse = f'{i.get("sse_before_mean", 0):.2f}→{i.get("sse_after_mean", 0):.2f}' if i else '-'
        lmin = f'{i.get("lambda_min_pos_over_r_mean", 0):.5f}' if i else '-'
        pe = f'{r.get("p_e", 0):.3f}' if r else '-'
        npos = f'{i.get("n_pos_mean", 0):.0f}' if i else '-'
        print(f'{s["task_name"]:<20} {rho:>6} {beta:>6} {sse:>10} {lmin:>8} {pe:>6} {npos:>6}')

    # Trend analysis
    p_e_values = [s['routing'].get('p_e', None) for s in all_summaries if s['routing']]
    if len(p_e_values) >= 3:
        first_half = np.mean(p_e_values[:len(p_e_values)//2])
        second_half = np.mean(p_e_values[len(p_e_values)//2:])
        print(f'\n[TREND] p_e first half: {first_half:.3f}, second half: {second_half:.3f}')
        if second_half > first_half * 1.5:
            print('  ⚠  p_e increasing significantly — routing degrades with more tasks')
        else:
            print('  ✓  p_e stable across tasks')

    lmin_values = [s['init'].get('lambda_min_pos_over_r_min', None) for s in all_summaries if s['init']]
    if len(lmin_values) >= 3:
        first_half = np.mean(lmin_values[:len(lmin_values)//2])
        second_half = np.mean(lmin_values[len(lmin_values)//2:])
        print(f'[TREND] λ_min+/r first half: {first_half:.6f}, second half: {second_half:.6f}')
        if second_half < first_half * 0.1:
            print('  ⚠  CPI margin collapsing — may need stronger γ or new routing approach')

    print('\n' + '=' * 70)


if __name__ == '__main__':
    main()
