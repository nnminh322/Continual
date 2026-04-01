#!/usr/bin/env python3
"""Comprehensive data extraction from ALL 60 result JSONs."""
import json, os, sys
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))

def load(fn):
    fp = os.path.join(DIR, fn)
    if not os.path.exists(fp):
        return None
    with open(fp) as f:
        return json.load(f)

BACKBONES = ['flan-t5-large', 'flan-t5-xl', 'Llama-2-7b-hf']
BB_SHORT = {'flan-t5-large': 'T5L', 'flan-t5-xl': 'T5XL', 'Llama-2-7b-hf': 'LLaMA'}
BENCHMARKS = ['Long_Sequence', 'SuperNI']
BM_SHORT = {'Long_Sequence': 'LS', 'SuperNI': 'SNI'}

def section_header(title):
    print(f"\n{'#'*80}")
    print(f"# {title}")
    print(f"{'#'*80}")

###############################################################################
# PHASE A: GEOMETRY
###############################################################################
def extract_A(bb, bench, wh_suffix):
    tag = f"{BB_SHORT[bb]}/{BM_SHORT[bench]}{wh_suffix}"
    raw = load(f"geometry_{bb}_{bench}.json")
    whi = load(f"geometry_{bb}_{bench}_whitened.json")
    if raw is None:
        print(f"  [{tag}] MISSING")
        return

    d = raw if wh_suffix == '' else whi
    if d is None:
        print(f"  [{tag}] MISSING whitened")
        return

    a1 = d.get('A1_effective_dim', {})
    tasks = d.get('tasks_found', list(a1.keys()))
    d_model = d.get('d_model', '?')

    # A1 aggregates
    evr_vals = [a1[t]['evr_k95'] for t in a1 if isinstance(a1[t], dict)]
    pr_vals = [a1[t]['participation_ratio'] for t in a1 if isinstance(a1[t], dict)]
    er_vals = [a1[t]['effective_rank'] for t in a1 if isinstance(a1[t], dict)]
    print(f"\n  [{tag}] d_model={d_model}, n_tasks={len(tasks)}")
    print(f"    A1 EVR_k95: mean={np.mean(evr_vals):.0f}, min={np.min(evr_vals):.0f}, max={np.max(evr_vals):.0f}")
    print(f"    A1 PaR:     mean={np.mean(pr_vals):.1f}, min={np.min(pr_vals):.1f}, max={np.max(pr_vals):.1f}")
    print(f"    A1 EffRank: mean={np.mean(er_vals):.1f}, min={np.min(er_vals):.1f}, max={np.max(er_vals):.1f}")

    # A2 Gaussianity
    a2 = d.get('A2_gaussianity', {})
    kurt_vals = [a2[t]['mean_abs_kurtosis'] for t in a2 if isinstance(a2[t], dict)]
    if kurt_vals:
        print(f"    A2 |kurtosis|: mean={np.mean(kurt_vals):.3f}, min={np.min(kurt_vals):.3f}, max={np.max(kurt_vals):.3f}")

    # A2b Multimodality
    a2b = d.get('A2b_multimodality', {})
    mm_count = sum(1 for t in a2b if isinstance(a2b[t], dict) and a2b[t].get('multi_modal'))
    total = sum(1 for t in a2b if isinstance(a2b[t], dict))
    if total:
        print(f"    A2b Multimodal: {mm_count}/{total} tasks multimodal")

    # A3 centroid distances
    a3 = d.get('A3_centroid_distances', {})
    if 'cosine_dist' in a3 and 'l2_dist' in a3:
        cos_mat = np.array(a3['cosine_dist'])
        l2_mat = np.array(a3['l2_dist'])
        n = cos_mat.shape[0]
        cos_upper = cos_mat[np.triu_indices(n, k=1)]
        l2_upper = l2_mat[np.triu_indices(n, k=1)]
        print(f"    A3 Cosine dist: mean={np.mean(cos_upper):.4f}, min={np.min(cos_upper):.4f}, max={np.max(cos_upper):.4f}")
        print(f"    A3 L2 dist:     mean={np.mean(l2_upper):.4f}, min={np.min(l2_upper):.4f}, max={np.max(l2_upper):.4f}")

    # A4 subspace distances
    a4 = d.get('A4_subspace_distances', {})
    if 'geodesic' in a4:
        geo_mat = np.array(a4['geodesic'])
        n = geo_mat.shape[0]
        geo_upper = geo_mat[np.triu_indices(n, k=1)]
        print(f"    A4 Geodesic:    mean={np.mean(geo_upper):.3f}, min={np.min(geo_upper):.3f}, max={np.max(geo_upper):.3f}")
    if 'frobenius_overlap' in a4:
        frob_mat = np.array(a4['frobenius_overlap'])
        frob_upper = frob_mat[np.triu_indices(n, k=1)]
        print(f"    A4 Frob overlap:mean={np.mean(frob_upper):.3f}, min={np.min(frob_upper):.3f}, max={np.max(frob_upper):.3f}")

    # A5 anisotropy
    a5 = d.get('A5_anisotropy', {})
    cond_vals = [a5[t]['condition_number'] for t in a5 if isinstance(a5[t], dict)]
    aniso_vals = [a5[t]['anisotropy_ratio'] for t in a5 if isinstance(a5[t], dict)]
    if cond_vals:
        print(f"    A5 Cond#:  mean={np.mean(cond_vals):.1f}, min={np.min(cond_vals):.1f}, max={np.max(cond_vals):.1f}")


###############################################################################
# PHASE B/C: ROUTING
###############################################################################
def extract_BC(bb, bench, wh_suffix):
    tag = f"{BB_SHORT[bb]}/{BM_SHORT[bench]}{wh_suffix}"
    fn = f"routing_{bb}_{bench}{'_whitened' if wh_suffix else ''}.json"
    d = load(fn)
    if d is None:
        print(f"  [{tag}] MISSING")
        return

    # B: distance metrics
    b = d.get('phase_B_distance', {})
    print(f"\n  [{tag}] Phase B distance metrics:")
    for metric in ['L2', 'Cosine', 'NormL2', 'Mahalanobis', 'SpectralAffinity', 'SubspaceResidual', 'WeightedSpectral', 'PSR_full', 'PSR_no_mean', 'PSR_no_subspace', 'PSR_no_penalty']:
        if metric in b:
            acc = b[metric]['accuracy']
            print(f"    {metric:20s}: {acc:.6f} ({acc*100:.2f}%)")

    # C: classifiers
    c = d.get('phase_C_sklearn', {})
    print(f"  [{tag}] Phase C classifiers:")
    for clf in ['LDA', 'RidgeClassifier', 'QDA', 'LinearSVM']:
        if clf in c:
            acc = c[clf]['accuracy']
            print(f"    {clf:20s}: {acc:.6f} ({acc*100:.2f}%)")

    # Top5
    t5 = d.get('top5', [])
    if t5:
        print(f"  [{tag}] Top 5: {[(x[0], f'{x[1]*100:.2f}%') for x in t5]}")


###############################################################################
# PHASE D: ABLATION
###############################################################################
def extract_D(bb, bench, wh_suffix):
    tag = f"{BB_SHORT[bb]}/{BM_SHORT[bench]}{wh_suffix}"
    fn = f"ablation_{bb}_{bench}{'_whitened' if wh_suffix else ''}.json"
    d = load(fn)
    if d is None:
        print(f"  [{tag}] MISSING")
        return

    # D1
    d1 = d.get('D1_ablation', {})
    print(f"\n  [{tag}] D1 component ablation:")
    for comp in ['Centroid_only', 'Subspace_only', 'PSR_light', 'PSR_full', 'PSR_no_penalty']:
        if comp in d1:
            acc = d1[comp]['accuracy']
            print(f"    {comp:20s}: {acc:.6f} ({acc*100:.2f}%)")

    # D2
    d2 = d.get('D2_rank_sweep', {})
    if d2:
        print(f"  [{tag}] D2 rank sweep:")
        for k in ['2', '4', '8', '16', '32', '64']:
            if k in d2:
                acc = d2[k]['accuracy']
                mem = d2[k].get('memory_bytes_per_task', '?')
                print(f"    k={k:3s}: acc={acc:.6f} ({acc*100:.2f}%) mem={mem}")

    # D3
    d3 = d.get('D3_domain', {})
    if d3:
        print(f"  [{tag}] D3 domain breakdown:")
        for dom in d3:
            if isinstance(d3[dom], dict) and 'mean' in d3[dom]:
                print(f"    {dom:12s}: mean={d3[dom]['mean']:.4f} ({d3[dom]['mean']*100:.2f}%) tasks={d3[dom].get('tasks','?')}")

    # D6 incremental
    d6 = d.get('D6_incremental', {})
    if d6:
        print(f"  [{tag}] D6 incremental (final step accuracy):")
        for method in ['PSR', 'RLS_batch', 'RLS_incremental']:
            if method in d6:
                steps = d6[method]
                last_key = str(max(int(k) for k in steps.keys()))
                final_acc = steps[last_key]['accuracy']
                # Also show trajectory
                accs = [steps[str(i)]['accuracy'] for i in range(1, int(last_key)+1)]
                print(f"    {method:20s}: final={final_acc:.6f} ({final_acc*100:.2f}%)")
                print(f"      trajectory: {[f'{a:.3f}' for a in accs]}")


###############################################################################
# PHASE E: THEORY
###############################################################################
def extract_E(bb, bench, wh_suffix):
    tag = f"{BB_SHORT[bb]}/{BM_SHORT[bench]}{wh_suffix}"
    fn = f"theory_{bb}_{bench}{'_whitened' if wh_suffix else ''}.json"
    d = load(fn)
    if d is None:
        print(f"  [{tag}] MISSING")
        return

    # E1
    e1 = d.get('E1_kl_confusion', {})
    if e1:
        print(f"\n  [{tag}] E1 KL-confusion:")
        if 'spearman' in e1:
            print(f"    Spearman rho={e1['spearman']['rho']:.4f}, p={e1['spearman']['pval']:.2e}")
        if 'PSR_accuracy' in e1:
            print(f"    PSR_accuracy={e1['PSR_accuracy']:.6f}")

    # E2
    e2 = d.get('E2_grassmann', {})
    if e2:
        print(f"  [{tag}] E2 Grassmann:")
        print(f"    d={e2.get('d')}, k={e2.get('k')}, T_actual={e2.get('T_actual')}, T_max_bound={e2.get('T_max_bound','?')}")
        print(f"    bound_satisfied={e2.get('bound_satisfied')}")
        print(f"    delta_max={e2.get('delta_max','?')}, delta_mean={e2.get('delta_mean','?')}")
        print(f"    mean_geodesic_nn={e2.get('mean_geodesic_nn','?')}")

    # E3
    e3 = d.get('E3_rmt', {})
    if e3:
        n_signal_list = [e3[t]['n_signal_eigvals'] for t in e3 if isinstance(e3[t], dict)]
        gamma_list = [e3[t]['gamma'] for t in e3 if isinstance(e3[t], dict)]
        evr_signal = [e3[t]['evr_k_signal'] for t in e3 if isinstance(e3[t], dict)]
        shrink_alpha = [e3[t]['oas_shrinkage_alpha'] for t in e3 if isinstance(e3[t], dict)]
        print(f"  [{tag}] E3 RMT:")
        print(f"    n_signal: mean={np.mean(n_signal_list):.0f}, min={np.min(n_signal_list):.0f}, max={np.max(n_signal_list):.0f}")
        print(f"    gamma:    mean={np.mean(gamma_list):.3f}")
        print(f"    EVR_signal: mean={np.mean(evr_signal):.4f}")
        print(f"    OAS_alpha:  mean={np.mean(shrink_alpha):.4f}")

    # E3 shrinkage
    e3s = d.get('E3_shrinkage', {})
    if e3s:
        print(f"  [{tag}] E3 Shrinkage: raw_acc={e3s.get('raw_acc','?')}, shrinkage_acc={e3s.get('shrinkage_acc','?')}, improvement={e3s.get('improvement','?')}")


###############################################################################
# PHASE F: LEARNED ROUTING
###############################################################################
def extract_F(bb, bench, wh_suffix):
    tag = f"{BB_SHORT[bb]}/{BM_SHORT[bench]}{wh_suffix}"
    fn = f"learned_routing_{bb}_{bench}{'_whitened' if wh_suffix else ''}.json"
    d = load(fn)
    if d is None:
        print(f"  [{tag}] MISSING")
        return

    # Final accuracy
    fa = d.get('final_accuracy', {})
    print(f"\n  [{tag}] Phase F final accuracy:")
    for method in ['NearestCentroid', 'CosineNearestCentroid', 'PSR', 'RLS_Woodbury', 'GPM_ROOT']:
        if method in fa:
            acc = fa[method]
            print(f"    {method:25s}: {acc:.6f} ({acc*100:.2f}%)")

    # Incremental trajectory
    results = d.get('results', {})
    print(f"  [{tag}] Phase F incremental trajectories:")
    for method in ['NearestCentroid', 'CosineNearestCentroid', 'PSR', 'RLS_Woodbury', 'GPM_ROOT']:
        if method in results and isinstance(results[method], list):
            accs = [step.get('accuracy', '?') for step in results[method]]
            print(f"    {method:25s}: {[f'{a:.3f}' if isinstance(a, float) else str(a) for a in accs]}")


###############################################################################
# MAIN
###############################################################################
if __name__ == '__main__':
    for bb in BACKBONES:
        dim_map = {'flan-t5-large':1024,'flan-t5-xl':2048,'Llama-2-7b-hf':4096}
        section_header(f"BACKBONE: {bb} (d={dim_map[bb]})")

        # PHASE A
        print("\n" + "="*60)
        print(f"  PHASE A: GEOMETRY ({bb})")
        print("="*60)
        for bench in BENCHMARKS:
            extract_A(bb, bench, '')
            extract_A(bb, bench, '_whitened')

        # PHASE B/C
        print("\n" + "="*60)
        print(f"  PHASE B/C: ROUTING ({bb})")
        print("="*60)
        for bench in BENCHMARKS:
            extract_BC(bb, bench, '')
            extract_BC(bb, bench, '_whitened')

        # PHASE D
        print("\n" + "="*60)
        print(f"  PHASE D: ABLATION ({bb})")
        print("="*60)
        for bench in BENCHMARKS:
            extract_D(bb, bench, '')
            extract_D(bb, bench, '_whitened')

        # PHASE E
        print("\n" + "="*60)
        print(f"  PHASE E: THEORY ({bb})")
        print("="*60)
        for bench in BENCHMARKS:
            extract_E(bb, bench, '')
            extract_E(bb, bench, '_whitened')

        # PHASE F
        print("\n" + "="*60)
        print(f"  PHASE F: LEARNED ROUTING ({bb})")
        print("="*60)
        for bench in BENCHMARKS:
            extract_F(bb, bench, '')
            extract_F(bb, bench, '_whitened')
