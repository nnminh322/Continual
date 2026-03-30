#!/usr/bin/env python3
"""
Phase B+C — Distance Metric & Routing Algorithm Comparison.

Builds task signatures from TRAIN split, routes TEST split, reports accuracy.

Usage:
  python compare_routing.py --emb_dir embeddings/T5EncoderModel --benchmark Long_Sequence
  python compare_routing.py --emb_dir embeddings/LlamaForCausalLM --benchmark SuperNI
"""
from __future__ import annotations
import argparse, json, os, sys, warnings
from pathlib import Path
from collections import OrderedDict

import numpy as np
from numpy.linalg import eigh, svd, norm
# scipy.linalg.sqrtm removed — BW distance dropped per review


# ═══ Global whitening ═══

def compute_whitening(task_embs: dict[str, np.ndarray]):
    all_embs = np.vstack(list(task_embs.values()))
    mu_global = all_embs.mean(0)
    cov_global = np.cov(all_embs, rowvar=False)
    eigvals, eigvecs = eigh(cov_global)
    eigvals = np.maximum(eigvals, 1e-8)
    W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return mu_global, W

def apply_whitening(task_embs, mu_global, W):
    return {t: (embs - mu_global) @ W.T for t, embs in task_embs.items()}

# ── Benchmark → task list mapping (shared with analyze_geometry.py) ──
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

LONG_SEQ_CLUSTERS = {
    "sentiment": ["yelp","amazon","imdb","sst2"],
    "NLI":       ["mnli","cb","rte"],
    "topic":     ["dbpedia","agnews","yahoo"],
    "RC":        ["multirc","boolq"],
    "misc":      ["copa","qqp","wic"],
}


# ═══════════════════════════════════════════════════════════════════════
# Loaders  (same as analyze_geometry.py)
# ═══════════════════════════════════════════════════════════════════════

def load_split(emb_dir, benchmark, task, split):
    p = Path(emb_dir) / benchmark / task / f"{split}.npz"
    if not p.exists():
        return None, None
    data = np.load(str(p), allow_pickle=True)
    return data["embeddings"].astype(np.float64), data["labels"]


def load_all_tasks(emb_dir, benchmark, tasks, split="train"):
    out = OrderedDict()
    for t in tasks:
        embs, _ = load_split(emb_dir, benchmark, t, split)
        if embs is not None:
            out[t] = embs
    return out


# ═══════════════════════════════════════════════════════════════════════
# Task Signatures  (built from TRAIN split only)
# ═══════════════════════════════════════════════════════════════════════

class TaskSignature:
    """PPCA signature for one task:  (mu, V, Lambda, sigma2)."""
    def __init__(self, embs: np.ndarray, k: int):
        self.n, self.d = embs.shape
        self.mu = embs.mean(0)                              # (d,)
        cov = np.cov(embs, rowvar=False)                    # (d, d)
        eigvals, eigvecs = eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        self.k = min(k, self.d)
        self.V = eigvecs[:, :self.k]                        # (d, k)
        self.lam = np.maximum(eigvals[:self.k], 1e-12)      # (k,)
        # sigma2 = average of remaining eigenvalues
        remaining = eigvals[self.k:]
        self.sigma2 = float(max(remaining.mean(), 1e-12))
        self.cov = cov
        self.eigvals = eigvals

    def shrunk_cov(self, alpha=None):
        """Ledoit-Wolf style shrink: alpha*diag + (1-alpha)*cov."""
        if alpha is None:
            # Oracle approximation shrinkage (OAS)
            n, d = self.n, self.d
            trace_cov = np.trace(self.cov)
            trace_cov2 = np.sum(self.cov ** 2)
            mu_hat = trace_cov / d
            rho_num = (1 - 2/d) * trace_cov2 + trace_cov**2
            rho_den = (n + 1 - 2/d) * (trace_cov2 - trace_cov**2 / d)
            alpha = max(0.0, min(1.0, rho_num / max(rho_den, 1e-12)))
        return (1 - alpha) * self.cov + alpha * np.eye(self.d) * np.trace(self.cov) / self.d


def build_signatures(task_embs: dict[str, np.ndarray], k: int):
    """Build TaskSignature for each task from train embeddings."""
    return {t: TaskSignature(embs, k) for t, embs in task_embs.items()}


# ═══════════════════════════════════════════════════════════════════════
# Phase B — Distance Metrics  (point → task routing)
# ═══════════════════════════════════════════════════════════════════════

def route_l2(h: np.ndarray, sigs: dict[str, TaskSignature]):
    return min(sigs, key=lambda t: norm(h - sigs[t].mu))

def route_cosine(h: np.ndarray, sigs: dict[str, TaskSignature]):
    hn = h / (norm(h) + 1e-12)
    return min(sigs, key=lambda t: 1.0 - hn @ (sigs[t].mu / (norm(sigs[t].mu) + 1e-12)))

def route_norm_l2(h: np.ndarray, sigs: dict[str, TaskSignature]):
    hn = h / (norm(h) + 1e-12)
    return min(sigs, key=lambda t: norm(hn - sigs[t].mu / (norm(sigs[t].mu) + 1e-12)))

def route_mahalanobis_pooled(h: np.ndarray, sigs: dict[str, TaskSignature], inv_pool):
    """Mahalanobis with pooled covariance."""
    def dist(t):
        delta = h - sigs[t].mu
        return float(delta @ inv_pool @ delta)
    return min(sigs, key=dist)

def route_spectral_affinity(h: np.ndarray, sigs: dict[str, TaskSignature]):
    """SpecRoute-style: largest projection onto principal subspace."""
    h_norm2 = float(norm(h)**2) + 1e-12
    return max(sigs, key=lambda t: float(norm(sigs[t].V.T @ h)**2) / h_norm2)

def route_subspace_residual(h: np.ndarray, sigs: dict[str, TaskSignature]):
    """Smallest residual after subspace projection."""
    return min(sigs, key=lambda t: norm(h - sigs[t].V @ (sigs[t].V.T @ h)))

def route_weighted_spectral(h: np.ndarray, sigs: dict[str, TaskSignature]):
    """Spectral affinity weighted by eigenvalues."""
    def score(t):
        proj = sigs[t].V.T @ h  # (k,)
        return float(np.sum(sigs[t].lam * proj**2) / (sigs[t].lam.sum() + 1e-12))
    return max(sigs, key=score)


# ── PSR variants ──

def _psr_distance(h, sig: TaskSignature, use_mean=True, use_subspace=True,
                  use_penalty=True):
    """PSR routing distance (lower = better match)."""
    delta = h - sig.mu if use_mean else h
    k, s2 = sig.k, sig.sigma2
    d = sig.d

    dist = 0.0
    if use_subspace and k > 0:
        proj = sig.V.T @ delta  # (k,)
        weights = sig.lam / (s2 * (sig.lam + s2))
        dist += float(np.sum(weights * proj**2))
    dist += float(norm(delta)**2) / s2  # isotropic term

    if use_penalty:
        dist += float(np.sum(np.log(sig.lam + s2))) + (d - k) * np.log(s2)
    return dist


def route_psr_full(h, sigs):
    return min(sigs, key=lambda t: _psr_distance(h, sigs[t], True, True, True))

def route_psr_no_mean(h, sigs):
    return min(sigs, key=lambda t: _psr_distance(h, sigs[t], False, True, True))

def route_psr_no_subspace(h, sigs):
    return min(sigs, key=lambda t: _psr_distance(h, sigs[t], True, False, True))

def route_psr_no_penalty(h, sigs):
    return min(sigs, key=lambda t: _psr_distance(h, sigs[t], True, True, False))


# NOTE: Bures-Wasserstein for point-vs-distribution degenerates to
# L2-to-centroid + task-dependent constant tr(Sigma_t). Removed per review.


# ═══════════════════════════════════════════════════════════════════════
# Phase C — Routing Algorithms  (sklearn-based classifiers)
# ═══════════════════════════════════════════════════════════════════════

def _build_Xy(task_embs_train: dict[str, np.ndarray]):
    """Concatenate all tasks into X (N_total, d), y (N_total,) with integer labels."""
    tasks = list(task_embs_train.keys())
    Xs, ys = [], []
    for i, t in enumerate(tasks):
        Xs.append(task_embs_train[t])
        ys.append(np.full(task_embs_train[t].shape[0], i, dtype=np.int64))
    return np.vstack(Xs), np.concatenate(ys), tasks


def train_sklearn_classifiers(X_train, y_train, tasks):
    """Train routing-relevant classifiers. Returns dict[name → model].

    Kept: LDA (shared cov = PSR k=d, Sigma_t=Sigma), QDA (full per-task cov),
    LinearSVM (discriminative baseline), RidgeClassifier (≈ RLS from V11).
    Removed per review: kNN, RF, XGBoost, LogReg — not theory-motivated.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import RidgeClassifier

    models = {}
    models["LDA"] = LinearDiscriminantAnalysis()
    models["QDA"] = QuadraticDiscriminantAnalysis(reg_param=1e-2)
    models["LinearSVM"] = LinearSVC(max_iter=5000, dual="auto")
    models["RidgeClassifier"] = RidgeClassifier(alpha=1.0)

    trained = {}
    for name, clf in models.items():
        try:
            clf.fit(X_train, y_train)
            trained[name] = clf
        except Exception as e:
            print(f"  WARN: {name} failed to fit: {e}")
    return trained


# ═══════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════

def evaluate_distance_methods(sigs, task_embs_test, tasks, inv_pool=None):
    """Route each test sample using distance-based methods. Returns accuracy dict."""
    methods = {
        "L2":               lambda h: route_l2(h, sigs),
        "Cosine":           lambda h: route_cosine(h, sigs),
        "NormL2":           lambda h: route_norm_l2(h, sigs),
        "SpectralAffinity": lambda h: route_spectral_affinity(h, sigs),
        "SubspaceResidual": lambda h: route_subspace_residual(h, sigs),
        "WeightedSpectral": lambda h: route_weighted_spectral(h, sigs),
        "PSR_full":         lambda h: route_psr_full(h, sigs),
        "PSR_no_mean":      lambda h: route_psr_no_mean(h, sigs),
        "PSR_no_subspace":  lambda h: route_psr_no_subspace(h, sigs),
        "PSR_no_penalty":   lambda h: route_psr_no_penalty(h, sigs),
    }
    if inv_pool is not None:
        methods["Mahalanobis"] = lambda h: route_mahalanobis_pooled(h, sigs, inv_pool)

    results = {m: {"correct": 0, "total": 0, "per_task": {}} for m in methods}
    confusion = {m: np.zeros((len(tasks), len(tasks)), dtype=np.int64) for m in methods}
    task2idx = {t: i for i, t in enumerate(tasks)}

    for true_task in tasks:
        if true_task not in task_embs_test:
            continue
        embs = task_embs_test[true_task]
        true_idx = task2idx[true_task]
        for m_name, route_fn in methods.items():
            correct = 0
            for i in range(embs.shape[0]):
                pred_task = route_fn(embs[i])
                pred_idx = task2idx[pred_task]
                confusion[m_name][true_idx, pred_idx] += 1
                if pred_task == true_task:
                    correct += 1
            results[m_name]["per_task"][true_task] = correct / max(embs.shape[0], 1)
            results[m_name]["correct"] += correct
            results[m_name]["total"]   += embs.shape[0]

    for m in results:
        results[m]["accuracy"] = results[m]["correct"] / max(results[m]["total"], 1)
    return results, confusion


def evaluate_sklearn_methods(trained_models, task_embs_test, tasks):
    """Route each test sample using sklearn classifiers."""
    task2idx = {t: i for i, t in enumerate(tasks)}
    results = {}
    confusion = {}
    X_tests, y_tests = [], []
    for t in tasks:
        if t not in task_embs_test:
            continue
        embs = task_embs_test[t]
        X_tests.append(embs)
        y_tests.append(np.full(embs.shape[0], task2idx[t], dtype=np.int64))
    if not X_tests:
        return {}, {}
    X_test = np.vstack(X_tests)
    y_test = np.concatenate(y_tests)

    for name, clf in trained_models.items():
        preds = clf.predict(X_test)
        acc = float((preds == y_test).mean())
        cm = np.zeros((len(tasks), len(tasks)), dtype=np.int64)
        for true, pred in zip(y_test, preds):
            cm[true, pred] += 1
        # per-task accuracy
        per_task = {}
        for t in tasks:
            idx = task2idx[t]
            mask = y_test == idx
            if mask.sum() > 0:
                per_task[t] = float((preds[mask] == idx).sum() / mask.sum())
        results[name] = {"accuracy": acc, "per_task": per_task,
                         "correct": int((preds == y_test).sum()),
                         "total": int(len(y_test))}
        confusion[name] = cm

    return results, confusion


# ═══════════════════════════════════════════════════════════════════════
# Domain cluster analysis
# ═══════════════════════════════════════════════════════════════════════

def domain_breakdown(results: dict, tasks: list, benchmark: str):
    """Split accuracy into same-domain (intra-cluster) vs cross-domain."""
    if benchmark != "Long_Sequence":
        return {}
    cluster_map = {}
    for cluster, members in LONG_SEQ_CLUSTERS.items():
        for m in members:
            cluster_map[m] = cluster

    breakdown = {}
    for method, res in results.items():
        pt = res.get("per_task", {})
        intra, inter = [], []
        for t in tasks:
            if t not in pt:
                continue
            c = cluster_map.get(t)
            cluster_size = sum(1 for m in tasks if cluster_map.get(m) == c and m in pt)
            if cluster_size > 1:
                intra.append(pt[t])
            else:
                inter.append(pt[t])
        breakdown[method] = {
            "intra_cluster_acc": float(np.mean(intra)) if intra else None,
            "inter_cluster_acc": float(np.mean(inter)) if inter else None,
        }
    return breakdown


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase B+C — Routing Comparison")
    parser.add_argument("--emb_dir", required=True)
    parser.add_argument("--benchmark", required=True, choices=list(BENCHMARKS.keys()))
    parser.add_argument("--subspace_k", type=int, default=8)
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--whiten", action="store_true",
                        help="Apply global ZCA whitening before routing")
    parser.add_argument("--skip_sklearn", action="store_true",
                        help="Skip sklearn classifiers (Phase C)")
    args = parser.parse_args()

    tasks = BENCHMARKS[args.benchmark]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    backbone = Path(args.emb_dir).name
    tag = f"{backbone}_{args.benchmark}" + ("_whitened" if args.whiten else "")

    print(f"=== Phase B+C: Routing Comparison  [{tag}]  k={args.subspace_k} ===\n")

    # Load train → build signatures, load test → evaluate
    train_embs = load_all_tasks(args.emb_dir, args.benchmark, tasks, "train")
    test_embs  = load_all_tasks(args.emb_dir, args.benchmark, tasks, "test")
    found = sorted(set(train_embs.keys()) & set(test_embs.keys()))
    print(f"Tasks with both train & test: {len(found)}/{len(tasks)}")
    if not found:
        print("ERROR: No tasks with both train+test."); sys.exit(1)

    # Filter to found
    train_embs = OrderedDict((t, train_embs[t]) for t in found)
    test_embs  = OrderedDict((t, test_embs[t])  for t in found)

    # Optional whitening (fit on train, apply to both)
    if args.whiten:
        mu_g, W = compute_whitening(train_embs)
        train_embs = apply_whitening(train_embs, mu_g, W)
        test_embs  = apply_whitening(test_embs, mu_g, W)
        print("Applied ZCA whitening\n")

    # Build signatures
    sigs = build_signatures(train_embs, args.subspace_k)

    # Pooled (inverse) covariance for Mahalanobis
    d = next(iter(train_embs.values())).shape[1]
    cov_pool = np.zeros((d, d))
    n_total = 0
    for embs in train_embs.values():
        cov_pool += np.cov(embs, rowvar=False) * (embs.shape[0] - 1)
        n_total += embs.shape[0] - 1
    cov_pool /= max(n_total, 1)
    # Regularize before inversion
    cov_pool += 1e-4 * np.eye(d)
    try:
        inv_pool = np.linalg.inv(cov_pool)
    except np.linalg.LinAlgError:
        inv_pool = None
        print("WARN: pooled covariance singular, skipping Mahalanobis.")

    report = {"backbone": backbone, "benchmark": args.benchmark,
              "k": args.subspace_k, "d_model": d, "tasks": found}

    # ── Phase B: Distance methods ──
    print("\n─── Phase B: Distance-based routing ───")
    dist_results, dist_confusion = evaluate_distance_methods(
        sigs, test_embs, found, inv_pool)

    print(f"\n{'Method':25s} {'Accuracy':>10s}")
    print("-" * 37)
    for m in sorted(dist_results, key=lambda x: -dist_results[x]["accuracy"]):
        acc = dist_results[m]["accuracy"]
        print(f"  {m:23s}  {acc*100:7.2f}%")
    report["phase_B_distance"] = {
        m: {"accuracy": r["accuracy"], "per_task": r["per_task"]}
        for m, r in dist_results.items()
    }

    # ── Domain breakdown ──
    bd = domain_breakdown(dist_results, found, args.benchmark)
    if bd:
        report["phase_B_domain_breakdown"] = bd
        print(f"\n{'Method':25s} {'Intra-cluster':>14s} {'Inter-cluster':>14s}")
        print("-" * 55)
        for m in sorted(bd, key=lambda x: -(dist_results[x]["accuracy"])):
            ic = bd[m]["intra_cluster_acc"]
            xc = bd[m]["inter_cluster_acc"]
            ic_s = f"{ic*100:.2f}%" if ic is not None else "N/A"
            xc_s = f"{xc*100:.2f}%" if xc is not None else "N/A"
            print(f"  {m:23s}  {ic_s:>12s}  {xc_s:>12s}")

    # ── Phase C: sklearn classifiers ──
    if not args.skip_sklearn:
        print("\n─── Phase C: Sklearn classifiers ───")
        X_train, y_train, _ = _build_Xy(train_embs)
        print(f"Training on {X_train.shape[0]} samples, {len(found)} classes, d={d}")
        trained = train_sklearn_classifiers(X_train, y_train, found)
        sk_results, sk_confusion = evaluate_sklearn_methods(trained, test_embs, found)

        print(f"\n{'Method':25s} {'Accuracy':>10s}")
        print("-" * 37)
        for m in sorted(sk_results, key=lambda x: -sk_results[x]["accuracy"]):
            acc = sk_results[m]["accuracy"]
            print(f"  {m:23s}  {acc*100:7.2f}%")
        report["phase_C_sklearn"] = {
            m: {"accuracy": r["accuracy"], "per_task": r["per_task"]}
            for m, r in sk_results.items()
        }

        # domain breakdown for sklearn too
        bd_sk = domain_breakdown(sk_results, found, args.benchmark)
        if bd_sk:
            report["phase_C_domain_breakdown"] = bd_sk

        # Save confusion matrices
        for m, cm in {**dist_confusion, **sk_confusion}.items():
            np.save(str(out_dir / f"confusion_{tag}_{m}.npy"), cm)
    else:
        for m, cm in dist_confusion.items():
            np.save(str(out_dir / f"confusion_{tag}_{m}.npy"), cm)

    # ── Summary: top 5 ──
    all_results = {**dist_results}
    if not args.skip_sklearn:
        all_results.update(sk_results)
    ranked = sorted(all_results.items(), key=lambda x: -x[1]["accuracy"])
    print(f"\n══ Top 5 Methods ({tag}) ══")
    for i, (m, r) in enumerate(ranked[:5], 1):
        print(f"  #{i}  {m:25s}  {r['accuracy']*100:.2f}%")

    report["top5"] = [(m, r["accuracy"]) for m, r in ranked[:5]]

    out_path = out_dir / f"routing_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n✓ Saved {out_path}")


if __name__ == "__main__":
    main()
