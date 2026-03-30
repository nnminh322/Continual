#!/usr/bin/env python3
"""
Phase E — Theory Validation.

E1. KL decomposition vs empirical routing confusion
E2. Grassmannian packing bound verification
E3. RMT prediction (Marchenko-Pastur) vs empirical eigenvalue distribution

Usage:
  python validate_theory.py --emb_dir embeddings/T5EncoderModel --benchmark Long_Sequence
  python validate_theory.py --emb_dir embeddings/LlamaForCausalLM --benchmark SuperNI
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from collections import OrderedDict

import numpy as np
from numpy.linalg import eigh, norm


def compute_whitening(task_embs):
    all_embs = np.vstack(list(task_embs.values()))
    mu = all_embs.mean(0)
    cov = np.cov(all_embs, rowvar=False)
    ev, evec = eigh(cov)
    ev = np.maximum(ev, 1e-8)
    W = evec @ np.diag(1.0 / np.sqrt(ev)) @ evec.T
    return mu, W

def apply_whitening(task_embs, mu, W):
    return {t: (e - mu) @ W.T for t, e in task_embs.items()}

# ── shared constants ──
BENCHMARKS = {
    "Long_Sequence": [
        "yelp","amazon","mnli","cb","copa","qqp","rte",
        "imdb","sst2","dbpedia","agnews","yahoo","multirc","boolq","wic",
    ],
    "SuperNI": [
        "task1687_sentiment140_classification",
        "task363_sst2_polarity_classification",
        "task875_emotion_classification",
        "task073_commonsenseqa_answer_generation",
        "task591_sciq_answer_generation",
        "task002_quoref_answer_generation",
        "task1290_xsum_summarization",
        "task1572_samsum_summary",
        "task511_reddit_tifu_long_text_summarization",
        "task181_outcome_extraction",
        "task748_glucose_reverse_cause_event_detection",
        "task1510_evalution_relation_extraction",
        "task639_multi_woz_user_utterance_generation",
        "task1590_diplomacy_text_generation",
        "task1729_personachat_generate_next",
    ],
}


def load_split(emb_dir, benchmark, task, split):
    p = Path(emb_dir) / benchmark / task / f"{split}.npz"
    if not p.exists():
        return None
    return np.load(str(p), allow_pickle=True)["embeddings"].astype(np.float64)

def load_all(emb_dir, benchmark, tasks, split):
    out = OrderedDict()
    for t in tasks:
        e = load_split(emb_dir, benchmark, t, split)
        if e is not None:
            out[t] = e
    return out


class PPCASig:
    def __init__(self, embs, k):
        self.n, self.d = embs.shape
        self.mu = embs.mean(0)
        self.cov = np.cov(embs, rowvar=False)
        ev, evec = eigh(self.cov)
        idx = np.argsort(ev)[::-1]
        self.eigvals = ev[idx]
        self.V = evec[:, idx[:k]]
        self.k = min(k, self.d)
        self.lam = np.maximum(self.eigvals[:self.k], 1e-12)
        self.sigma2 = float(max(self.eigvals[self.k:].mean(), 1e-12)) if self.k < self.d else 1e-12


# ═══════════════════════════════════════════════════════════════════════
# E1 — KL Decomposition vs Routing Confusion
# ═══════════════════════════════════════════════════════════════════════

def compute_kl_ppca(sig_t: PPCASig, sig_s: PPCASig):
    """KL(P_t || P_s) under PPCA models, using full covariance for accuracy."""
    d = sig_t.d
    # Use regularized covariances
    Sigma_t = sig_t.cov + 1e-6 * np.eye(d)
    Sigma_s = sig_s.cov + 1e-6 * np.eye(d)

    # KL = 0.5 * (tr(Σ_s^{-1} Σ_t) + (μ_s-μ_t)^T Σ_s^{-1} (μ_s-μ_t) - d + ln|Σ_s|/|Σ_t|)
    try:
        L_s = np.linalg.cholesky(Sigma_s)
        inv_s = np.linalg.solve(Sigma_s, np.eye(d))
    except np.linalg.LinAlgError:
        inv_s = np.linalg.pinv(Sigma_s)

    delta_mu = sig_s.mu - sig_t.mu

    # Mean displacement term
    D_mu = 0.5 * float(delta_mu @ inv_s @ delta_mu)

    # Distributional shape term
    trace_term = float(np.trace(inv_s @ Sigma_t))
    sign_s, logdet_s = np.linalg.slogdet(Sigma_s)
    sign_t, logdet_t = np.linalg.slogdet(Sigma_t)
    D_sigma = 0.5 * (trace_term - d + logdet_s - logdet_t)

    kl_total = D_mu + D_sigma
    return {"kl_total": float(kl_total), "D_mu": float(D_mu),
            "D_sigma": float(D_sigma)}


def compute_confusion_matrix(sigs, test_embs, tasks):
    """Empirical routing confusion using PSR."""
    task2idx = {t: i for i, t in enumerate(tasks)}
    n = len(tasks)
    cm = np.zeros((n, n))  # cm[i,j] = fraction of task_i routed to task_j
    for i, t in enumerate(tasks):
        if t not in test_embs:
            continue
        embs = test_embs[t]
        counts = np.zeros(n)
        for x in range(embs.shape[0]):
            h = embs[x]
            dists = {s: _psr_dist(h, sigs[s]) for s in tasks}
            pred = min(dists, key=dists.get)
            counts[task2idx[pred]] += 1
        cm[i] = counts / max(embs.shape[0], 1)
    return cm


def _psr_dist(h, sig):
    delta = h - sig.mu
    s2 = sig.sigma2
    dist = float(norm(delta)**2) / s2
    if sig.k > 0:
        proj = sig.V.T @ delta
        w = sig.lam / (s2 * (sig.lam + s2))
        dist += float(np.sum(w * proj**2))
    dist += float(np.sum(np.log(sig.lam + s2))) + (sig.d - sig.k) * np.log(s2)
    return dist


def e1_kl_vs_confusion(train_embs, test_embs, tasks, k):
    """E1: correlate pairwise KL with empirical confusion rates."""
    sigs = {t: PPCASig(e, k) for t, e in train_embs.items()}

    # Pairwise KL
    n = len(tasks)
    kl_mat = np.zeros((n, n))
    dmu_mat = np.zeros((n, n))
    dsig_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            kl = compute_kl_ppca(sigs[tasks[i]], sigs[tasks[j]])
            kl_mat[i, j]   = kl["kl_total"]
            dmu_mat[i, j]  = kl["D_mu"]
            dsig_mat[i, j] = kl["D_sigma"]

    # Confusion matrix
    cm = compute_confusion_matrix(sigs, test_embs, tasks)

    # Correlation: off-diagonal KL vs confusion rate
    pairs_kl, pairs_conf = [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pairs_kl.append(kl_mat[i, j])
            pairs_conf.append(cm[i, j])
    pairs_kl = np.array(pairs_kl)
    pairs_conf = np.array(pairs_conf)

    # Spearman rank correlation (KL should anti-correlate with confusion)
    from scipy.stats import spearmanr
    if len(pairs_kl) > 2:
        corr, pval = spearmanr(pairs_kl, pairs_conf)
    else:
        corr, pval = 0.0, 1.0

    return {
        "kl_matrix": kl_mat.tolist(),
        "D_mu_matrix": dmu_mat.tolist(),
        "D_sigma_matrix": dsig_mat.tolist(),
        "confusion_matrix": cm.tolist(),
        "spearman_kl_vs_confusion": {"rho": float(corr), "pval": float(pval)},
        "tasks": tasks,
        "PSR_accuracy": float(np.mean(np.diag(cm))),
    }


# ═══════════════════════════════════════════════════════════════════════
# E2 — Grassmannian Packing Bound
# ═══════════════════════════════════════════════════════════════════════

def e2_grassmann_bound(train_embs, tasks, k):
    """Verify T_max <= d / (k * (1 - delta_max))."""
    sigs = {t: PPCASig(e, k) for t, e in train_embs.items()}
    n = len(tasks)
    d = next(iter(sigs.values())).d

    # Pairwise subspace overlap δ_ij = ||V_i^T V_j||_F^2
    overlap = np.zeros((n, n))
    for i in range(n):
        Vi = sigs[tasks[i]].V
        for j in range(i+1, n):
            Vj = sigs[tasks[j]].V
            M = Vi.T @ Vj
            ov = float(np.sum(M**2))
            overlap[i, j] = overlap[j, i] = ov

    delta_max = float(np.max(overlap))
    delta_mean = float(np.mean(overlap[np.triu_indices(n, k=1)]))

    # Packing bound
    T_max = d / (k * (1.0 - delta_max + 1e-9))

    # Margin: minimum routing margin from subspace perspective
    # For each task, min geodesic to nearest neighbor
    from numpy.linalg import svd as np_svd
    geodesic_nn = []
    for i in range(n):
        Vi = sigs[tasks[i]].V
        min_geo = float('inf')
        for j in range(n):
            if i == j:
                continue
            Vj = sigs[tasks[j]].V
            sv = np.clip(np_svd(Vi.T @ Vj, compute_uv=False), 0, 1)
            angles = np.arccos(np.clip(sv, -1, 1))
            geo = float(norm(angles))
            min_geo = min(min_geo, geo)
        geodesic_nn.append(min_geo)

    return {
        "d": d, "k": k, "T_actual": n,
        "T_max_bound": float(T_max),
        "delta_max": delta_max, "delta_mean": delta_mean,
        "overlap_matrix": overlap.tolist(),
        "geodesic_to_nearest": {tasks[i]: geodesic_nn[i] for i in range(n)},
        "mean_geodesic_nn": float(np.mean(geodesic_nn)),
        "bound_satisfied": n <= T_max,
    }


# ═══════════════════════════════════════════════════════════════════════
# E3 — RMT / Marchenko-Pastur Prediction
# ═══════════════════════════════════════════════════════════════════════

def marchenko_pastur_edges(gamma, sigma2=1.0):
    """Marchenko-Pastur law edges: lambda_± = sigma² (1 ± sqrt(gamma))²."""
    lam_plus  = sigma2 * (1 + np.sqrt(gamma))**2
    lam_minus = sigma2 * max(1 - np.sqrt(gamma), 0)**2
    return float(lam_minus), float(lam_plus)


def e3_rmt_analysis(train_embs, tasks, k):
    """Compare empirical eigenvalue distributions with RMT predictions."""
    results = {}
    for t in tasks:
        if t not in train_embs:
            continue
        embs = train_embs[t]
        n, d = embs.shape
        gamma = d / n  # aspect ratio

        cov = np.cov(embs, rowvar=False)
        eigvals = np.flip(np.sort(np.linalg.eigvalsh(cov)))
        eigvals = np.maximum(eigvals, 0)

        # Estimate noise floor (sigma2) from bottom eigenvalues
        sigma2_est = float(np.median(eigvals[k:]))  # median of "noise" eigenvalues

        # MP edges
        lam_minus, lam_plus = marchenko_pastur_edges(gamma, sigma2_est)

        # How many eigenvalues exceed MP upper edge? → signal eigenvalues
        n_signal = int(np.sum(eigvals > lam_plus))

        # Shrinkage effect: compare raw vs Ledoit-Wolf
        # Oracle Approximating Shrinkage (OAS) estimator
        trace_cov = np.trace(cov)
        trace_cov2 = np.sum(cov**2)
        mu_hat = trace_cov / d
        rho_num = (1 - 2/d) * trace_cov2 + trace_cov**2
        rho_den = (n + 1 - 2/d) * (trace_cov2 - trace_cov**2 / d)
        alpha_oas = max(0.0, min(1.0, rho_num / max(rho_den, 1e-12)))

        results[t] = {
            "n": n, "d": d, "gamma": float(gamma),
            "sigma2_est": float(sigma2_est),
            "mp_lower": lam_minus, "mp_upper": lam_plus,
            "n_signal_eigvals": n_signal,
            "top5_eigvals": eigvals[:5].tolist(),
            "evr_k_signal": float(eigvals[:n_signal].sum() / max(eigvals.sum(), 1e-12)),
            "oas_shrinkage_alpha": float(alpha_oas),
            "eigenvalue_inflation_ratio": float(eigvals[0] / max(sigma2_est, 1e-12)),
        }

    return results


def e3_shrinkage_routing(train_embs, test_embs, tasks, k):
    """Compare routing accuracy with/without LW shrinkage on covariance."""
    # Raw PSR
    sigs_raw = {t: PPCASig(e, k) for t, e in train_embs.items()}

    # PSR with shrinkage: re-estimate eigenvalues after shrinkage
    sigs_shrunk = {}
    for t, embs in train_embs.items():
        n, d = embs.shape
        cov = np.cov(embs, rowvar=False)
        # OAS shrinkage
        trace_cov = np.trace(cov)
        trace_cov2 = np.sum(cov**2)
        mu_hat = trace_cov / d
        rho_num = (1 - 2/d) * trace_cov2 + trace_cov**2
        rho_den = (n + 1 - 2/d) * (trace_cov2 - trace_cov**2 / d)
        alpha = max(0.0, min(1.0, rho_num / max(rho_den, 1e-12)))
        cov_s = (1 - alpha) * cov + alpha * np.eye(d) * mu_hat

        sig = PPCASig.__new__(PPCASig)
        sig.n, sig.d = n, d
        sig.mu = embs.mean(0)
        sig.cov = cov_s
        ev, evec = eigh(cov_s)
        idx = np.argsort(ev)[::-1]
        sig.eigvals = ev[idx]
        sig.V = evec[:, idx[:k]]
        sig.k = min(k, d)
        sig.lam = np.maximum(sig.eigvals[:sig.k], 1e-12)
        sig.sigma2 = float(max(sig.eigvals[sig.k:].mean(), 1e-12))
        sigs_shrunk[t] = sig

    fn_raw = lambda h: min(sigs_raw, key=lambda t: _psr_dist(h, sigs_raw[t]))
    fn_shr = lambda h: min(sigs_shrunk, key=lambda t: _psr_dist(h, sigs_shrunk[t]))

    accs_raw, accs_shr = {}, {}
    for t in tasks:
        if t not in test_embs:
            continue
        embs = test_embs[t]
        c_raw = sum(1 for i in range(embs.shape[0]) if fn_raw(embs[i]) == t)
        c_shr = sum(1 for i in range(embs.shape[0]) if fn_shr(embs[i]) == t)
        accs_raw[t] = c_raw / max(embs.shape[0], 1)
        accs_shr[t] = c_shr / max(embs.shape[0], 1)

    return {
        "raw_mean_acc": float(np.mean(list(accs_raw.values()))),
        "shrinkage_mean_acc": float(np.mean(list(accs_shr.values()))),
        "raw_per_task": accs_raw,
        "shrinkage_per_task": accs_shr,
        "improvement": float(np.mean(list(accs_shr.values())) - np.mean(list(accs_raw.values()))),
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase E — Theory Validation")
    parser.add_argument("--emb_dir", required=True)
    parser.add_argument("--benchmark", required=True, choices=list(BENCHMARKS.keys()))
    parser.add_argument("--subspace_k", type=int, default=8)
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--whiten", action="store_true",
                        help="Apply global ZCA whitening")
    args = parser.parse_args()

    tasks = BENCHMARKS[args.benchmark]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    backbone = Path(args.emb_dir).name
    tag = f"{backbone}_{args.benchmark}" + ("_whitened" if args.whiten else "")

    print(f"=== Phase E: Theory Validation  [{tag}]  k={args.subspace_k} ===\n")

    train_embs = load_all(args.emb_dir, args.benchmark, tasks, "train")
    test_embs  = load_all(args.emb_dir, args.benchmark, tasks, "test")
    found = sorted(set(train_embs) & set(test_embs))
    if not found:
        print("ERROR: No tasks found."); sys.exit(1)
    train_embs = OrderedDict((t, train_embs[t]) for t in found)
    test_embs  = OrderedDict((t, test_embs[t])  for t in found)

    if args.whiten:
        mu_g, W = compute_whitening(train_embs)
        train_embs = apply_whitening(train_embs, mu_g, W)
        test_embs  = apply_whitening(test_embs, mu_g, W)
        print("Applied ZCA whitening\n")

    report = {"backbone": backbone, "benchmark": args.benchmark,
              "k": args.subspace_k, "tasks": found}

    # ── E1: KL vs Confusion ──
    print("─── E1: KL Decomposition vs Routing Confusion ───")
    e1 = e1_kl_vs_confusion(train_embs, test_embs, found, args.subspace_k)
    report["E1_kl_confusion"] = {
        "spearman": e1["spearman_kl_vs_confusion"],
        "PSR_accuracy": e1["PSR_accuracy"],
    }
    rho = e1["spearman_kl_vs_confusion"]["rho"]
    p = e1["spearman_kl_vs_confusion"]["pval"]
    print(f"  Spearman(KL, confusion): rho={rho:.4f}  p={p:.2e}")
    print(f"  PSR routing accuracy:    {e1['PSR_accuracy']:.2%}")
    print(f"  Expected: rho < 0 (higher KL → less confusion)")

    # Save full matrices
    np.save(str(out_dir / f"kl_matrix_{tag}.npy"), np.array(e1["kl_matrix"]))
    np.save(str(out_dir / f"confusion_psr_{tag}.npy"), np.array(e1["confusion_matrix"]))
    np.save(str(out_dir / f"D_mu_matrix_{tag}.npy"), np.array(e1["D_mu_matrix"]))
    np.save(str(out_dir / f"D_sigma_matrix_{tag}.npy"), np.array(e1["D_sigma_matrix"]))

    # ── E2: Grassmann bound ──
    print(f"\n─── E2: Grassmannian Packing Bound (k={args.subspace_k}) ───")
    e2 = e2_grassmann_bound(train_embs, found, args.subspace_k)
    report["E2_grassmann"] = {k: v for k, v in e2.items() if k != "overlap_matrix"}
    print(f"  T_actual = {e2['T_actual']},  T_max_bound = {e2['T_max_bound']:.1f}")
    print(f"  δ_max = {e2['delta_max']:.4f},  δ_mean = {e2['delta_mean']:.4f}")
    print(f"  Bound satisfied: {e2['bound_satisfied']}")
    print(f"  Mean geodesic to nearest neighbor: {e2['mean_geodesic_nn']:.4f}")
    np.save(str(out_dir / f"grassmann_overlap_{tag}.npy"), np.array(e2["overlap_matrix"]))

    # ── E3: RMT analysis ──
    print(f"\n─── E3: RMT / Marchenko-Pastur Analysis ───")
    rmt = e3_rmt_analysis(train_embs, found, args.subspace_k)
    for t, info in rmt.items():
        print(f"  {t:40s}  γ={info['gamma']:.2f}  #signal={info['n_signal_eigvals']:3d}"
              f"  λ₁/σ²={info['eigenvalue_inflation_ratio']:.1f}"
              f"  α_OAS={info['oas_shrinkage_alpha']:.3f}")
    report["E3_rmt"] = rmt

    # Shrinkage routing comparison
    print(f"\n─── E3b: Shrinkage vs Raw routing ───")
    shr = e3_shrinkage_routing(train_embs, test_embs, found, args.subspace_k)
    print(f"  Raw PSR mean acc:      {shr['raw_mean_acc']:.2%}")
    print(f"  Shrinkage PSR mean acc:{shr['shrinkage_mean_acc']:.2%}")
    print(f"  Improvement:           {shr['improvement']:+.2%}")
    report["E3_shrinkage"] = {
        "raw_acc": shr["raw_mean_acc"],
        "shrinkage_acc": shr["shrinkage_mean_acc"],
        "improvement": shr["improvement"],
    }

    # ── Save ──
    out_path = out_dir / f"theory_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n✓ Saved {out_path}")


if __name__ == "__main__":
    main()
