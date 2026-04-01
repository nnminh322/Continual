#!/usr/bin/env python3
"""Extract key metrics from ALL result JSONs for analysis."""
import json, os, sys

DIR = os.path.dirname(os.path.abspath(__file__))

def load(fn):
    fp = os.path.join(DIR, fn)
    if not os.path.exists(fp):
        return None
    with open(fp) as f:
        return json.load(f)

BACKBONES = ['flan-t5-large', 'flan-t5-xl', 'Llama-2-7b-hf']
BENCHMARKS = ['Long_Sequence', 'SuperNI']
SPACES = ['', '_whitened']

def fmt(x, digits=3):
    if x is None or x == '?':
        return '?'
    if isinstance(x, float):
        return f'{x:.{digits}f}'
    return str(x)

# ======= PHASE A =======
def extract_phase_a():
    print('='*80)
    print('PHASE A: GEOMETRY')
    print('='*80)
    for bb in BACKBONES:
        for bench in BENCHMARKS:
            for wh in SPACES:
                fn = f'geometry_{bb}_{bench}{wh}.json'
                d = load(fn)
                if d is None:
                    print(f'\n--- {bb}/{bench}{wh}: MISSING ---')
                    continue
                tag = f'{bb}/{bench}{wh}'
                print(f'\n--- {tag} ---')
                
                # A1
                a1 = d.get('A1_dimensionality', {})
                print(f'  d_model={a1.get("d_model","?")}, n_tasks={a1.get("n_tasks","?")}')
                s = a1.get('summary', {})
                if s:
                    print(f'  mean_EVR95={fmt(s.get("mean_evr95"))}, mean_PR={fmt(s.get("mean_participation_ratio"),1)}, mean_TwoNN={fmt(s.get("mean_twonn_dim"),1)}')
                
                # A2: Gaussianity
                a2 = d.get('A2_gaussianity', {})
                s2 = a2.get('summary', {})
                if s2:
                    print(f'  Gaussianity: mean_sw_p={fmt(s2.get("mean_shapiro_wilk_p"))}, mean_mardia_skew_p={fmt(s2.get("mean_mardia_skew_p"))}, mean_mardia_kurt_p={fmt(s2.get("mean_mardia_kurt_p"))}')
                
                # A3: Subspace overlap
                a3 = d.get('A3_subspace_overlap', {})
                s3 = a3.get('summary', {})
                if s3:
                    print(f'  Overlap: mean_frob={fmt(s3.get("mean_frobenius_overlap"))}, max={fmt(s3.get("max_frobenius_overlap"))}, min={fmt(s3.get("min_frobenius_overlap"))}')
                    print(f'           mean_angle={fmt(s3.get("mean_principal_angle_deg"),1)}deg, std={fmt(s3.get("std_principal_angle_deg"),1)}')

# ======= PHASE B/C =======
def extract_phase_bc():
    print('\n' + '='*80)
    print('PHASE B/C: ROUTING METRICS & CLASSIFIERS')
    print('='*80)
    for bb in BACKBONES:
        for bench in BENCHMARKS:
            for wh in SPACES:
                fn = f'routing_{bb}_{bench}{wh}.json'
                d = load(fn)
                if d is None:
                    print(f'\n--- {bb}/{bench}{wh}: MISSING ---')
                    continue
                tag = f'{bb}/{bench}{wh}'
                print(f'\n--- {tag} ---')
                
                # B: distance metrics
                b = d.get('B_distance_metrics', {})
                if b:
                    # Top-level accuracy summary
                    for metric_name, metric_data in b.items():
                        if isinstance(metric_data, dict) and 'accuracy' in metric_data:
                            print(f'  {metric_name}: acc={fmt(metric_data["accuracy"])}')
                        elif isinstance(metric_data, dict) and 'top1_accuracy' in metric_data:
                            print(f'  {metric_name}: top1={fmt(metric_data["top1_accuracy"])}')
                
                # C: classifiers
                c = d.get('C_classifiers', {})
                if c:
                    for clf_name, clf_data in c.items():
                        if isinstance(clf_data, dict):
                            acc = clf_data.get('accuracy', clf_data.get('test_accuracy', '?'))
                            print(f'  {clf_name}: acc={fmt(acc)}')

# ======= PHASE D =======
def extract_phase_d():
    print('\n' + '='*80)
    print('PHASE D: PSR ABLATION')
    print('='*80)
    for bb in BACKBONES:
        for bench in BENCHMARKS:
            for wh in SPACES:
                fn = f'ablation_{bb}_{bench}{wh}.json'
                d = load(fn)
                if d is None:
                    print(f'\n--- {bb}/{bench}{wh}: MISSING ---')
                    continue
                tag = f'{bb}/{bench}{wh}'
                print(f'\n--- {tag} ---')
                
                # D1: component ablation
                d1 = d.get('D1_component_ablation', {})
                if d1:
                    for comp, data in d1.items():
                        if isinstance(data, dict):
                            acc = data.get('accuracy', data.get('top1_accuracy', '?'))
                            print(f'  D1 {comp}: acc={fmt(acc)}')
                
                # D2: rank sensitivity
                d2 = d.get('D2_rank_sensitivity', {})
                if d2:
                    print(f'  D2 rank sweep:')
                    if isinstance(d2, dict):
                        for k, v in d2.items():
                            if isinstance(v, dict):
                                acc = v.get('accuracy', v.get('top1_accuracy', '?'))
                                print(f'    k={k}: acc={fmt(acc)}')
                            elif isinstance(v, list) and len(v) > 0:
                                # maybe list of rank results
                                print(f'    {k}: {v[:5]}...')
                
                # D3: domain breakdown
                d3 = d.get('D3_domain_breakdown', {})
                if d3:
                    print(f'  D3 domains:')
                    for dom, data in d3.items():
                        if isinstance(data, dict):
                            acc = data.get('accuracy', '?')
                            print(f'    {dom}: acc={fmt(acc)}')

# ======= PHASE E =======
def extract_phase_e():
    print('\n' + '='*80)
    print('PHASE E: THEORY VALIDATION')
    print('='*80)
    for bb in BACKBONES:
        for bench in BENCHMARKS:
            for wh in SPACES:
                fn = f'theory_{bb}_{bench}{wh}.json'
                d = load(fn)
                if d is None:
                    print(f'\n--- {bb}/{bench}{wh}: MISSING ---')
                    continue
                tag = f'{bb}/{bench}{wh}'
                print(f'\n--- {tag} ---')
                
                # E1: KL decomposition
                e1 = d.get('E1_kl_decomposition', {})
                if e1:
                    s = e1.get('summary', e1)
                    print(f'  E1 KL: mean_total={fmt(s.get("mean_kl_total"))}, mean_vs_empirical_corr={fmt(s.get("kl_vs_confusion_correlation", s.get("correlation_kl_vs_confusion")))}')
                
                # E2: Grassmann
                e2 = d.get('E2_grassmann_bound', {})
                if e2:
                    s = e2.get('summary', e2)
                    print(f'  E2 Grassmann: bound_satisfied={s.get("bound_satisfied_ratio", s.get("fraction_satisfied","?"))}')
                
                # E3: RMT
                e3 = d.get('E3_rmt', {})
                if e3:
                    s = e3.get('summary', e3)
                    print(f'  E3 RMT: mean_mp_fit={fmt(s.get("mean_mp_fit_r2", s.get("mean_mp_fit")))}')

# ======= PHASE F =======
def extract_phase_f():
    print('\n' + '='*80)
    print('PHASE F: LEARNED ROUTING SIMULATION')
    print('='*80)
    for bb in BACKBONES:
        for bench in BENCHMARKS:
            for wh in SPACES:
                fn = f'learned_routing_{bb}_{bench}{wh}.json'
                d = load(fn)
                if d is None:
                    print(f'\n--- {bb}/{bench}{wh}: MISSING ---')
                    continue
                tag = f'{bb}/{bench}{wh}'
                print(f'\n--- {tag} ---')
                
                # F: each routing method results
                for method, data in d.items():
                    if isinstance(data, dict):
                        acc = data.get('final_accuracy', data.get('mean_accuracy', data.get('accuracy', '?')))
                        if acc != '?':
                            print(f'  {method}: final_acc={fmt(acc)}')
                        # Check for incremental results
                        inc = data.get('incremental_accuracy', data.get('per_step_accuracy', None))
                        if inc and isinstance(inc, list):
                            print(f'    incremental: {[fmt(x) for x in inc[:5]]}...')
                        elif inc and isinstance(inc, dict):
                            for step, a in list(inc.items())[:3]:
                                print(f'    step {step}: {fmt(a)}')

if __name__ == '__main__':
    extract_phase_a()
    extract_phase_bc()
    extract_phase_d()
    extract_phase_e()
    extract_phase_f()
