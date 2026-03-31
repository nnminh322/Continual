import json, os

base = '/Users/nnminh322/Desktop/personal/Continual/routing_analysis/results'

# === Routing comparison ===
for suffix in ['', '_whitened']:
    fname = f'routing_flan-t5-large_Long_Sequence{suffix}.json'
    path = os.path.join(base, fname)
    if not os.path.exists(path):
        continue
    with open(path) as f:
        d = json.load(f)
    label = "WHITENED" if suffix else "RAW"
    print(f"\n{'='*60}")
    print(f"  ROUTING RESULTS — {label}")
    print(f"{'='*60}")
    
    # Phase B
    if 'phase_B_distance' in d:
        items = sorted(d['phase_B_distance'].items(), key=lambda x: -x[1]['accuracy'])
        print(f"\n  Phase B (Distance-based):")
        for m, r in items:
            print(f"    {m:25s}  {r['accuracy']*100:7.2f}%")
    
    # Phase C
    if 'phase_C_sklearn' in d:
        items = sorted(d['phase_C_sklearn'].items(), key=lambda x: -x[1]['accuracy'])
        print(f"\n  Phase C (Sklearn classifiers):")
        for m, r in items:
            print(f"    {m:25s}  {r['accuracy']*100:7.2f}%")
    
    # Domain breakdown
    for key in ['phase_B_domain_breakdown', 'phase_C_domain_breakdown']:
        if key in d:
            phase = "B" if "B" in key else "C"
            print(f"\n  Domain Breakdown (Phase {phase}):")
            for m, bd in sorted(d[key].items()):
                ic = bd.get('intra_cluster_acc')
                xc = bd.get('inter_cluster_acc')
                ic_s = f"{ic*100:.1f}%" if ic else "N/A"
                xc_s = f"{xc*100:.1f}%" if xc else "N/A"
                print(f"    {m:25s}  intra={ic_s:>8s}  inter={xc_s:>8s}")

    # Top5
    if 'top5' in d:
        print(f"\n  Top 5 Overall:")
        for m, a in d['top5']:
            print(f"    {m:25s}  {a*100:7.2f}%")

# === Ablation ===
for suffix in ['', '_whitened']:
    fname = f'ablation_flan-t5-large_Long_Sequence{suffix}.json'
    path = os.path.join(base, fname)
    if not os.path.exists(path):
        continue
    with open(path) as f:
        d = json.load(f)
    label = "WHITENED" if suffix else "RAW"
    print(f"\n{'='*60}")
    print(f"  ABLATION RESULTS — {label}")
    print(f"{'='*60}")
    
    if 'D1_component_ablation' in d:
        print(f"\n  D1: PSR Component Ablation:")
        for m, r in sorted(d['D1_component_ablation'].items(), key=lambda x: -x[1]['accuracy']):
            print(f"    {m:25s}  {r['accuracy']*100:7.2f}%")
    
    if 'D2_rank_sensitivity' in d:
        print(f"\n  D2: Rank Sensitivity:")
        for k, r in sorted(d['D2_rank_sensitivity'].items(), key=lambda x: int(x[0].split('=')[1])):
            print(f"    {k:15s}  {r['accuracy']*100:7.2f}%")

# === Theory ===
for suffix in ['', '_whitened']:
    fname = f'theory_flan-t5-large_Long_Sequence{suffix}.json'
    path = os.path.join(base, fname)
    if not os.path.exists(path):
        continue
    with open(path) as f:
        d = json.load(f)
    label = "WHITENED" if suffix else "RAW"
    print(f"\n{'='*60}")
    print(f"  THEORY VALIDATION — {label}")
    print(f"{'='*60}")
    
    if 'E1_kl_decomposition' in d:
        print(f"\n  E1: KL Decomposition (sample pairs):")
        for pair_key, decomp in list(d['E1_kl_decomposition'].items())[:5]:
            total = decomp.get('total_kl', decomp.get('KL_total', 0))
            mean_term = decomp.get('mean_term', decomp.get('KL_mean', 0))
            sub_term = decomp.get('subspace_term', decomp.get('KL_subspace', 0))
            spec_term = decomp.get('spectral_term', decomp.get('KL_spectral', 0))
            print(f"    {pair_key:40s}  total={total:.2f}  mean={mean_term:.2f}  sub={sub_term:.2f}  spec={spec_term:.2f}")
    
    if 'E2_grassmann_bound' in d:
        print(f"\n  E2: Grassmann Bound:")
        gb = d['E2_grassmann_bound']
        if isinstance(gb, dict):
            for k, v in list(gb.items())[:3]:
                print(f"    {k}: {v}")

