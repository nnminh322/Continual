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

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _resolve_device(device_str: str) -> str:
    """Resolve 'auto' → 'cuda' if GPU available, else 'cpu'.

    Executes a tiny kernel to catch arch-mismatch errors
    (cudaErrorNoKernelImageForDevice) before committing to CUDA.
    Falls back to CPU on any failure.
    """
    if device_str == "auto":
        if HAS_TORCH and torch.cuda.is_available():
            try:
                _t = torch.zeros(8, device="cuda") + 1
                del _t
                torch.cuda.synchronize()
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"[GPU] {gpu_name}  {vram_gb:.1f} GB VRAM — using CUDA")
                return "cuda"
            except Exception as e:
                print(f"[GPU] CUDA reported available but kernel launch failed "
                      f"({type(e).__name__}: {e}) — falling back to CPU")
                return "cpu"
        print("[GPU] No CUDA device found — using CPU")
        return "cpu"
    return device_str


def compute_whitening(task_embs, device: str = "cpu"):
    if "cuda" in device and HAS_TORCH:
        dev = torch.device(device)
        chunks = [torch.tensor(v, dtype=torch.float32, device=dev) for v in task_embs.values()]
        all_t  = torch.cat(chunks, dim=0)
        mu_t   = all_t.mean(0)
        Xc     = all_t - mu_t
        cov_t  = (Xc.T @ Xc) / (all_t.shape[0] - 1)
        ev_t, evec_t = torch.linalg.eigh(cov_t)
        ev_t   = torch.clamp(ev_t, min=1e-8)
        W_t    = evec_t @ torch.diag(1.0 / torch.sqrt(ev_t)) @ evec_t.T
        return mu_t.cpu().numpy(), W_t.cpu().numpy()
    all_embs = np.vstack(list(task_embs.values()))
    mu = all_embs.mean(0)
    cov = np.cov(all_embs, rowvar=False)
    ev, evec = eigh(cov)
    ev = np.maximum(ev, 1e-8)
    W = evec @ np.diag(1.0 / np.sqrt(ev)) @ evec.T
    return mu, W

def apply_whitening(task_embs, mu, W, device='cpu'):
    if 'cuda' in device and HAS_TORCH:
        dev = torch.device(device)
        mu_t = torch.tensor(mu, dtype=torch.float32, device=dev)
        W_t  = torch.tensor(W, dtype=torch.float32, device=dev)
        out = {}
        for t, e in task_embs.items():
            X = torch.tensor(e, dtype=torch.float32, device=dev)
            out[t] = ((X - mu_t) @ W_t.T).cpu().numpy()
            del X
        return out
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
    def __init__(self, embs, k, device='cpu'):
        self.n, self.d = embs.shape
        if 'cuda' in device and HAS_TORCH:
            dev = torch.device(device)
            X = torch.tensor(embs, dtype=torch.float32, device=dev)
            self.mu = X.mean(0).cpu().numpy()
            Xc = X - X.mean(0)
            cov_t = (Xc.T @ Xc) / max(self.n - 1, 1)
            ev_t, evec_t = torch.linalg.eigh(cov_t)
            idx = torch.argsort(ev_t, descending=True)
            ev_t = ev_t[idx]; evec_t = evec_t[:, idx]
            self.cov = cov_t.cpu().numpy()
            self.eigvals = ev_t.cpu().numpy()
            self.V = evec_t[:, :min(k, self.d)].cpu().numpy()
            self.k = min(k, self.d)
            self.lam = np.maximum(self.eigvals[:self.k], 1e-12)
            self.sigma2 = float(max(self.eigvals[self.k:].mean(), 1e-12)) if self.k < self.d else 1e-12
            del X, Xc, cov_t, ev_t, evec_t
        else:
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


def compute_kl_batch_gpu(sigs, tasks, device='cpu'):
    """Batch-compute all pairwise KL divergences on GPU.

    Returns (kl_mat, dmu_mat, dsig_mat) each of shape (n, n).
    ~50-100x faster than calling compute_kl_ppca in a double loop for d=4096.
    """
    n = len(tasks)
    d = sigs[tasks[0]].d
    use_gpu = 'cuda' in device and HAS_TORCH

    if not use_gpu:
        # Fallback to CPU loop
        kl_mat  = np.zeros((n, n))
        dmu_mat = np.zeros((n, n))
        dsig_mat= np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                kl = compute_kl_ppca(sigs[tasks[i]], sigs[tasks[j]])
                kl_mat[i, j]  = kl["kl_total"]
                dmu_mat[i, j] = kl["D_mu"]
                dsig_mat[i, j]= kl["D_sigma"]
        return kl_mat, dmu_mat, dsig_mat

    dev = torch.device(device)
    # Stack all covariances and means on GPU once
    Covs = torch.zeros(n, d, d, dtype=torch.float64, device=dev)
    Mus  = torch.zeros(n, d, dtype=torch.float64, device=dev)
    for i, t in enumerate(tasks):
        Covs[i] = torch.tensor(sigs[t].cov, dtype=torch.float64, device=dev)
        Mus[i]  = torch.tensor(sigs[t].mu, dtype=torch.float64, device=dev)
    # Adaptive regularization: scale by trace so rank-deficient covs become PD
    eye_d = torch.eye(d, dtype=torch.float64, device=dev)
    traces = torch.diagonal(Covs, dim1=-2, dim2=-1).sum(-1)          # (n,)
    reg = (1e-6 * traces / d).clamp(min=1e-10)                       # per-task scale
    Covs = Covs + reg[:, None, None] * eye_d.unsqueeze(0)

    # Precompute Cholesky and log-determinants for all tasks
    for attempt, bump in enumerate([0, 1e-4, 1e-3, 1e-2]):
        try:
            if bump > 0:
                Covs = Covs + bump * eye_d.unsqueeze(0)
            L_all = torch.linalg.cholesky(Covs)  # (n, d, d)
            break
        except torch.linalg.LinAlgError:
            if attempt == 3:
                raise

    # log|Σ_i| = 2 * sum(log(diag(L_i)))
    logdets = 2.0 * torch.log(torch.diagonal(L_all, dim1=-2, dim2=-1)).sum(-1)  # (n,)

    # Precompute Σ_s^{-1} for all s via Cholesky solve
    # inv_all[s] = Σ_s^{-1} = solve(L_s @ L_s^T, I)
    eye_batch = eye_d.unsqueeze(0).expand(n, -1, -1)
    inv_all = torch.cholesky_solve(eye_batch, L_all)  # (n, d, d)

    # Precompute traces: tr(Σ_s^{-1} @ Σ_t) for all (s,t) pairs
    # tr(A @ B) = sum(A * B^T) = sum(A * B) for symmetric matrices
    # inv_all: (n, d, d), Covs: (n, d, d)
    # trace_mat[i,j] = tr(inv_all[j] @ Covs[i])
    # = sum over (a,b) of inv_all[j,a,b] * Covs[i,b,a]
    # = sum over (a,b) of inv_all[j,a,b] * Covs[i,a,b]  (symmetric)
    # Vectorize: reshape to (n, d*d) and do matmul
    inv_flat = inv_all.reshape(n, d*d)  # (n, d*d)
    cov_flat = Covs.reshape(n, d*d)     # (n, d*d)
    trace_mat = cov_flat @ inv_flat.T   # (n_t, n_s) = trace_mat[i,j] = tr(inv[j] @ Cov[i])

    # Compute D_mu for all pairs: D_mu[i,j] = 0.5 * (mu_j - mu_i)^T @ inv[j] @ (mu_j - mu_i)
    # delta_mu[i,j] = mu_j - mu_i  → shape (n, n, d)
    delta = Mus.unsqueeze(0) - Mus.unsqueeze(1)  # (n_i, n_j, d): delta[i,j] = mu[j] - mu[i]
    # quad_form[i,j] = delta[i,j]^T @ inv[j] @ delta[i,j]
    # = (delta[i,j] @ inv[j]) . delta[i,j]
    # inv_delta[i,j] = inv[j] @ delta[i,j]
    inv_delta = torch.einsum('jab,ijb->ija', inv_all, delta)  # (n, n, d)
    dmu_mat_t = 0.5 * (delta * inv_delta).sum(-1)  # (n, n)

    # D_sigma[i,j] = 0.5 * (tr(inv[j] @ Cov[i]) - d + logdet[j] - logdet[i])
    dsig_mat_t = 0.5 * (trace_mat - d + logdets.unsqueeze(0) - logdets.unsqueeze(1))

    kl_mat_t = dmu_mat_t + dsig_mat_t

    # Zero diagonal
    mask = torch.eye(n, dtype=torch.bool, device=dev)
    kl_mat_t[mask] = 0
    dmu_mat_t[mask] = 0
    dsig_mat_t[mask] = 0

    kl_mat  = kl_mat_t.cpu().numpy()
    dmu_mat = dmu_mat_t.cpu().numpy()
    dsig_mat= dsig_mat_t.cpu().numpy()

    del Covs, Mus, L_all, inv_all, inv_flat, cov_flat, trace_mat, delta, inv_delta
    del kl_mat_t, dmu_mat_t, dsig_mat_t, eye_d, eye_batch, logdets, mask

    return kl_mat, dmu_mat, dsig_mat


def compute_confusion_matrix(sigs, test_embs, tasks, device='cpu'):
    """Empirical routing confusion using PSR (vectorized batch version)."""
    T = len(tasks)
    task2idx = {t: i for i, t in enumerate(tasks)}
    task_list = [t for t in tasks if t in sigs and t in test_embs]

    # Pre-compute stacked signatures
    d  = sigs[task_list[0]].d
    k  = sigs[task_list[0]].k
    C  = np.stack([sigs[t].mu  for t in task_list]).astype(np.float32)   # (T, d)
    V  = np.stack([sigs[t].V   for t in task_list]).astype(np.float32)   # (T, d, k)
    lam= np.stack([sigs[t].lam for t in task_list]).astype(np.float32)   # (T, k)
    s2 = np.array([sigs[t].sigma2 for t in task_list], dtype=np.float32) # (T,)
    W_psr = lam / (s2[:, None] * (lam + s2[:, None]))                    # (T, k)
    pen   = np.sum(np.log(lam + s2[:, None]), axis=1) + (d - k) * np.log(s2)  # (T,)

    use_gpu = "cuda" in device and HAS_TORCH
    if use_gpu:
        dev   = torch.device(device)
        C_t   = torch.tensor(C,     device=dev)
        V_t   = torch.tensor(V,     device=dev)
        W_t   = torch.tensor(W_psr, device=dev)
        pen_t = torch.tensor(pen,   device=dev)
        s2_t  = torch.tensor(s2,    device=dev)
        # Precompute (avoids (N,T,d) intermediates)
        CV_t  = torch.einsum('td,tdk->tk', C_t, V_t)   # (T, k)
        C_sq_t = (C_t ** 2).sum(1)                      # (T,)
    else:
        CV  = np.einsum('td,tdk->tk', C, V)              # (T, k)
        C_sq_np = np.sum(C ** 2, axis=1)                  # (T,)

    cm = np.zeros((T, T))
    for i, t in enumerate(task_list):
        embs = test_embs[t]
        H_np = embs.astype(np.float32)
        N    = H_np.shape[0]
        if use_gpu:
            H    = torch.tensor(H_np, device=dev)
            H_sq = (H ** 2).sum(1, keepdim=True)                    # (N, 1)
            l2   = H_sq + C_sq_t.unsqueeze(0) - 2 * (H @ C_t.T)    # (N, T)
            iso  = l2 / (s2_t.unsqueeze(0) + 1e-12)                 # (N, T)
            H_proj = torch.einsum('nd,tdk->ntk', H, V_t)            # (N, T, k)
            dp   = H_proj - CV_t.unsqueeze(0)                        # (N, T, k)
            dists = iso + (W_t.unsqueeze(0) * dp.pow(2)).sum(-1) + pen_t.unsqueeze(0)
            preds = dists.argmin(dim=1).cpu().numpy()
            del H, H_sq, l2, iso, H_proj, dp, dists
        else:
            H   = H_np.astype(np.float64)
            H_sq = np.sum(H ** 2, axis=1, keepdims=True)            # (N, 1)
            l2   = H_sq + C_sq_np[None, :] - 2 * (H @ C.T)          # (N, T)
            iso  = l2 / (s2[None, :] + 1e-12)
            H_proj = np.einsum('nd,tdk->ntk', H, V)                  # (N, T, k)
            dp   = H_proj - CV[None, :, :]                           # (N, T, k)
            dists = iso + np.sum(W_psr[None, :, :] * dp**2, axis=-1) + pen[None, :]
            preds = np.argmin(dists, axis=1)
        counts = np.zeros(T)
        np.add.at(counts, preds, 1)
        cm[i] = counts / max(N, 1)
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


def e1_kl_vs_confusion(train_embs, test_embs, tasks, k, device='cpu'):
    """E1: correlate pairwise KL with empirical confusion rates."""
    sigs = {t: PPCASig(e, k, device=device) for t, e in train_embs.items()}

    # Pairwise KL — batch GPU or CPU loop
    n = len(tasks)
    kl_mat, dmu_mat, dsig_mat = compute_kl_batch_gpu(sigs, tasks, device=device)

    # Confusion matrix
    cm = compute_confusion_matrix(sigs, test_embs, tasks, device=device)

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

def e2_grassmann_bound(train_embs, tasks, k, device='cpu'):
    """Verify T_max <= d / (k * (1 - delta_max))."""
    sigs = {t: PPCASig(e, k, device=device) for t, e in train_embs.items()}
    n = len(tasks)
    d = next(iter(sigs.values())).d
    use_gpu = 'cuda' in device and HAS_TORCH

    if use_gpu:
        dev = torch.device(device)
        # Stack all V matrices on GPU: (n, d, k)
        V_all = torch.stack([torch.tensor(sigs[tasks[i]].V, dtype=torch.float32, device=dev)
                             for i in range(n)])  # (n, d, k)
        # Pairwise overlap: δ_ij = ||V_i^T V_j||_F^2
        # V_i^T V_j = (k, d) @ (d, k) = (k, k) per pair
        # Batch: VtV[i,j] = V_all[i].T @ V_all[j]
        # overlap[i,j] = ||VtV[i,j]||_F^2
        VtV = torch.einsum('idk,jdk->ijk', V_all, V_all)  # remap: idK,jdK→ijK... no
        # Actually: V_all is (n, d, k), V_i^T is (k, d)
        # V_i^T @ V_j = (k, d) @ (d, k) = (k, k)
        # Using einsum: 'ikd,jld->ijkl' no, simpler:
        # VtV_{i,j} shape (k, k) = V_all[i].T @ V_all[j]
        # = einsum('da,db->ab', V_all[i], V_all[j]) 
        # For batch: einsum('ida,jda->ija', ...) but V is (n,d,k)
        # V_all[i,:,a] is column a of V_i → V_i^T[a,:] = V_all[i,:,a]
        # V_i^T @ V_j = sum_d V_all[i,d,a] * V_all[j,d,b] → einsum('ida,jdb->ijab', V_all, V_all)
        VtV = torch.einsum('ida,jdb->ijab', V_all, V_all)  # (n, n, k, k)
        overlap_t = (VtV ** 2).sum(dim=(-2, -1))  # (n, n)
        # Zero diagonal
        overlap_t.fill_diagonal_(0)
        overlap = overlap_t.cpu().numpy()

        # Geodesic distances via SVD on GPU
        # sv[i,j] = svdvals(V_i^T @ V_j) = svdvals(VtV[i,j])
        # VtV already computed as (n, n, k, k)
        geodesic_nn = []
        for i in range(n):
            min_geo = float('inf')
            for j in range(n):
                if i == j:
                    continue
                sv = torch.linalg.svdvals(VtV[i, j])
                sv = torch.clamp(sv, 0, 1)
                angles = torch.arccos(torch.clamp(sv, -1, 1))
                geo = float(torch.linalg.norm(angles))
                min_geo = min(min_geo, geo)
            geodesic_nn.append(min_geo)
        del V_all, VtV, overlap_t
    else:
        # CPU path
        overlap = np.zeros((n, n))
        for i in range(n):
            Vi = sigs[tasks[i]].V
            for j in range(i+1, n):
                Vj = sigs[tasks[j]].V
                M = Vi.T @ Vj
                ov = float(np.sum(M**2))
                overlap[i, j] = overlap[j, i] = ov

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


def e3_rmt_analysis(train_embs, tasks, k, device='cpu'):
    """Compare empirical eigenvalue distributions with RMT predictions."""
    use_gpu = 'cuda' in device and HAS_TORCH
    results = {}
    for t in tasks:
        if t not in train_embs:
            continue
        embs = train_embs[t]
        n, d = embs.shape
        gamma = d / n  # aspect ratio

        if use_gpu:
            dev = torch.device(device)
            X = torch.tensor(embs, dtype=torch.float32, device=dev)
            Xc = X - X.mean(0)
            cov_t = (Xc.T @ Xc) / max(n - 1, 1)
            eigvals_t = torch.linalg.eigvalsh(cov_t)
            eigvals_t = eigvals_t.flip(0).clamp(min=0)
            eigvals = eigvals_t.cpu().numpy()
            cov = cov_t.cpu().numpy()
            del X, Xc, cov_t, eigvals_t
        else:
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


def e3_shrinkage_routing(train_embs, test_embs, tasks, k, device='cpu'):
    """Compare routing accuracy with/without LW shrinkage on covariance."""
    use_gpu = 'cuda' in device and HAS_TORCH
    # Raw PSR
    sigs_raw = {t: PPCASig(e, k, device=device) for t, e in train_embs.items()}

    # PSR with shrinkage: re-estimate eigenvalues after shrinkage
    sigs_shrunk = {}
    for t, embs in train_embs.items():
        n, d = embs.shape

        if use_gpu:
            dev = torch.device(device)
            X = torch.tensor(embs, dtype=torch.float32, device=dev)
            Xc = X - X.mean(0)
            cov_t = (Xc.T @ Xc) / max(n - 1, 1)
            # OAS shrinkage on GPU
            trace_cov = torch.trace(cov_t)
            trace_cov2 = (cov_t ** 2).sum()
            mu_hat = trace_cov / d
            rho_num = (1 - 2/d) * trace_cov2 + trace_cov**2
            rho_den = (n + 1 - 2/d) * (trace_cov2 - trace_cov**2 / d)
            alpha = max(0.0, min(1.0, float(rho_num / max(rho_den, 1e-12))))
            cov_s = (1 - alpha) * cov_t + alpha * torch.eye(d, device=dev) * mu_hat
            ev_t, evec_t = torch.linalg.eigh(cov_s)
            idx = torch.argsort(ev_t, descending=True)
            ev_t = ev_t[idx]; evec_t = evec_t[:, idx]

            sig = PPCASig.__new__(PPCASig)
            sig.n, sig.d = n, d
            sig.mu = X.mean(0).cpu().numpy()
            sig.cov = cov_s.cpu().numpy()
            sig.eigvals = ev_t.cpu().numpy()
            sig.V = evec_t[:, :min(k, d)].cpu().numpy()
            sig.k = min(k, d)
            sig.lam = np.maximum(sig.eigvals[:sig.k], 1e-12)
            sig.sigma2 = float(max(sig.eigvals[sig.k:].mean(), 1e-12))
            sigs_shrunk[t] = sig
            del X, Xc, cov_t, cov_s, ev_t, evec_t
        else:
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

    # Vectorized evaluation using compute_confusion_matrix logic
    # We reuse the batch PSR infrastructure from compare_routing
    # Build per-task accuracy faster via vectorized PSR
    task_list = [t for t in tasks if t in test_embs and t in sigs_raw]

    def _vec_acc(sigs_dict, tl, te, dev):
        """Inline vectorized PSR accuracy for a given sigs dict."""
        from collections import OrderedDict
        _te = OrderedDict((t, te[t]) for t in tl if t in te)
        _tr_dummy = {}  # not needed here
        d  = sigs_dict[tl[0]].d
        k_ = sigs_dict[tl[0]].k
        C  = np.stack([sigs_dict[t].mu  for t in tl]).astype(np.float32)
        V  = np.stack([sigs_dict[t].V   for t in tl]).astype(np.float32)
        lam= np.stack([sigs_dict[t].lam for t in tl]).astype(np.float32)
        s2 = np.array([sigs_dict[t].sigma2 for t in tl], dtype=np.float32)
        W_ = lam / (s2[:, None] * (lam + s2[:, None]))
        pen= np.sum(np.log(lam + s2[:, None]), axis=1) + (d - k_) * np.log(s2)
        use_gpu = "cuda" in dev and HAS_TORCH
        if use_gpu:
            _dev = torch.device(dev)
            C_t = torch.tensor(C, device=_dev); V_t = torch.tensor(V, device=_dev)
            W_t = torch.tensor(W_, device=_dev); pen_t = torch.tensor(pen, device=_dev)
            s2_t= torch.tensor(s2, device=_dev)
            CV_t = torch.einsum('td,tdk->tk', C_t, V_t)        # (T, k)
            C_sq_t = (C_t ** 2).sum(1)                          # (T,)
        else:
            CV_np = np.einsum('td,tdk->tk', C, V)               # (T, k)
            C_sq_np = np.sum(C ** 2, axis=1)                     # (T,)
        per_task = {}
        for i, t in enumerate(tl):
            if t not in te:
                continue
            H_np = te[t].astype(np.float32)
            N = H_np.shape[0]
            if use_gpu:
                H_ = torch.tensor(H_np, device=_dev)
                H_sq = (H_ ** 2).sum(1, keepdim=True)              # (N, 1)
                l2 = H_sq + C_sq_t.unsqueeze(0) - 2 * (H_ @ C_t.T) # (N, T)
                iso= l2 / (s2_t.unsqueeze(0) + 1e-12)
                H_proj = torch.einsum('nd,tdk->ntk', H_, V_t)
                dp = H_proj - CV_t.unsqueeze(0)
                dists_ = iso + (W_t.unsqueeze(0) * dp.pow(2)).sum(-1) + pen_t.unsqueeze(0)
                preds_ = dists_.argmin(dim=1).cpu().numpy()
                del H_, H_sq, l2, iso, H_proj, dp, dists_
            else:
                H_ = H_np.astype(np.float64)
                H_sq = np.sum(H_ ** 2, axis=1, keepdims=True)      # (N, 1)
                l2 = H_sq + C_sq_np[None, :] - 2 * (H_ @ C.T)      # (N, T)
                iso= l2 / (s2[None, :] + 1e-12)
                H_proj = np.einsum('nd,tdk->ntk', H_, V)
                dp = H_proj - CV_np[None, :, :]
                dists_ = iso + np.sum(W_[None, :, :] * dp**2, axis=-1) + pen[None, :]
                preds_ = np.argmin(dists_, axis=1)
            per_task[t] = float((preds_ == i).sum()) / max(N, 1)
        return per_task

    accs_raw = _vec_acc(sigs_raw,    task_list, test_embs, device)
    accs_shr = _vec_acc(sigs_shrunk, task_list, test_embs, device)

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
    parser.add_argument("--device", default="auto",
                        help="Device: cpu | cuda | cuda:0 | auto (default: auto)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if output already exists")
    args = parser.parse_args()
    args.device = _resolve_device(args.device)

    tasks = BENCHMARKS[args.benchmark]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    backbone = Path(args.emb_dir).name
    tag = f"{backbone}_{args.benchmark}" + ("_whitened" if args.whiten else "")

    # ── Skip if already done ──
    out_path = out_dir / f"theory_{tag}.json"
    if out_path.exists() and not args.force:
        print(f"[SKIP] Phase E: {out_path} already exists. Use --force to re-run.")
        return

    print(f"=== Phase E: Theory Validation  [{tag}]  k={args.subspace_k} ===\n")

    train_embs = load_all(args.emb_dir, args.benchmark, tasks, "train")
    test_embs  = load_all(args.emb_dir, args.benchmark, tasks, "test")
    found = sorted(set(train_embs) & set(test_embs))
    if not found:
        print("ERROR: No tasks found."); sys.exit(1)
    train_embs = OrderedDict((t, train_embs[t]) for t in found)
    test_embs  = OrderedDict((t, test_embs[t])  for t in found)

    if args.whiten:
        mu_g, W = compute_whitening(train_embs, device=args.device)
        train_embs = apply_whitening(train_embs, mu_g, W, device=args.device)
        test_embs  = apply_whitening(test_embs, mu_g, W, device=args.device)
        print("Applied ZCA whitening\n")

    report = {"backbone": backbone, "benchmark": args.benchmark,
              "k": args.subspace_k, "tasks": found}

    # ── E1: KL vs Confusion ──
    print("─── E1: KL Decomposition vs Routing Confusion ───")
    e1 = e1_kl_vs_confusion(train_embs, test_embs, found, args.subspace_k, device=args.device)
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
    e2 = e2_grassmann_bound(train_embs, found, args.subspace_k, device=args.device)
    report["E2_grassmann"] = {k: v for k, v in e2.items() if k != "overlap_matrix"}
    print(f"  T_actual = {e2['T_actual']},  T_max_bound = {e2['T_max_bound']:.1f}")
    print(f"  δ_max = {e2['delta_max']:.4f},  δ_mean = {e2['delta_mean']:.4f}")
    print(f"  Bound satisfied: {e2['bound_satisfied']}")
    print(f"  Mean geodesic to nearest neighbor: {e2['mean_geodesic_nn']:.4f}")
    np.save(str(out_dir / f"grassmann_overlap_{tag}.npy"), np.array(e2["overlap_matrix"]))

    # ── E3: RMT analysis ──
    print(f"\n─── E3: RMT / Marchenko-Pastur Analysis ───")
    rmt = e3_rmt_analysis(train_embs, found, args.subspace_k, device=args.device)
    for t, info in rmt.items():
        print(f"  {t:40s}  γ={info['gamma']:.2f}  #signal={info['n_signal_eigvals']:3d}"
              f"  λ₁/σ²={info['eigenvalue_inflation_ratio']:.1f}"
              f"  α_OAS={info['oas_shrinkage_alpha']:.3f}")
    report["E3_rmt"] = rmt

    # Shrinkage routing comparison
    print(f"\n─── E3b: Shrinkage vs Raw routing ───")
    shr = e3_shrinkage_routing(train_embs, test_embs, found, args.subspace_k, device=args.device)
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
