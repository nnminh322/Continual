#!/usr/bin/env python3
"""
Phase D — PSR Ablation & Comparative Analysis.

D1. PSR component ablation (mean / subspace / spectrum / penalty)
D2. Rank sensitivity sweep (k = 2,4,8,16,32,64)
D3. Same-domain vs cross-domain breakdown
D4. T5 vs LLaMA comparison (runs both backbones if available)
D6. Incremental update simulation (add tasks one by one)

Usage:
  python ablation_psr.py --emb_dir embeddings/T5EncoderModel --benchmark Long_Sequence
  python ablation_psr.py --emb_dir embeddings/LlamaForCausalLM --benchmark SuperNI
  python ablation_psr.py --emb_dir embeddings --benchmark Long_Sequence --compare_backbones
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from collections import OrderedDict

import numpy as np
from numpy.linalg import norm, eigh

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
        ev_t = torch.clamp(ev_t, min=1e-8)
        W_t = evec_t @ torch.diag(1.0 / torch.sqrt(ev_t)) @ evec_t.T
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

LONG_SEQ_CLUSTERS = {
    "sentiment": ["yelp","amazon","imdb","sst2"],
    "NLI":       ["mnli","cb","rte"],
    "topic":     ["dbpedia","agnews","yahoo"],
    "RC":        ["multirc","boolq"],
    "misc":      ["copa","qqp","wic"],
}

# SuperNI approximate clusters (by task type)
SUPERNI_CLUSTERS = {
    "sentiment": ["task1687_sentiment140_classification",
                  "task363_sst2_polarity_classification",
                  "task875_emotion_classification"],
    "QA":        ["task073_commonsenseqa_answer_generation",
                  "task591_sciq_answer_generation",
                  "task002_quoref_answer_generation"],
    "summarization": ["task1290_xsum_summarization",
                      "task1572_samsum_summary",
                      "task511_reddit_tifu_long_text_summarization"],
    "extraction":    ["task181_outcome_extraction",
                      "task748_glucose_reverse_cause_event_detection",
                      "task1510_evalution_relation_extraction"],
    "dialogue":      ["task639_multi_woz_user_utterance_generation",
                      "task1590_diplomacy_text_generation",
                      "task1729_personachat_generate_next"],
}


# ═══════════════════════════════════════════════════════════════════════
# Loader + Signature (duplicated for standalone usage)
# ═══════════════════════════════════════════════════════════════════════

def load_split(emb_dir, benchmark, task, split):
    p = Path(emb_dir) / benchmark / task / f"{split}.npz"
    if not p.exists():
        return None
    data = np.load(str(p), allow_pickle=True)
    return data["embeddings"].astype(np.float64)

def load_all(emb_dir, benchmark, tasks, split):
    out = OrderedDict()
    for t in tasks:
        e = load_split(emb_dir, benchmark, t, split)
        if e is not None:
            out[t] = e
    return out


class PPCASig:
    """Lightweight PPCA signature."""
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
            self.k = min(k, self.d)
            self.V = evec_t[:, :self.k].cpu().numpy()
            self.lam = torch.clamp(ev_t[:self.k], min=1e-12).cpu().numpy()
            self.sigma2 = float(max(ev_t[self.k:].cpu().numpy().mean(), 1e-12)) if self.k < self.d else 1e-12
            del X, Xc, cov_t, ev_t, evec_t
        else:
            self.mu = embs.mean(0)
            cov = np.cov(embs, rowvar=False)
            ev, evec = eigh(cov)
            idx = np.argsort(ev)[::-1]
            ev = ev[idx]; evec = evec[:, idx]
            self.k = min(k, self.d)
            self.V = evec[:, :self.k]
            self.lam = np.maximum(ev[:self.k], 1e-12)
            self.sigma2 = float(max(ev[self.k:].mean(), 1e-12)) if self.k < self.d else 1e-12


def psr_dist(h, sig, use_mean=True, use_subspace=True, use_penalty=True):
    delta = h - sig.mu if use_mean else h
    s2, k, d_dim = sig.sigma2, sig.k, sig.d
    dist = float(norm(delta)**2) / s2
    if use_subspace and k > 0:
        proj = sig.V.T @ delta
        w = sig.lam / (s2 * (sig.lam + s2))
        dist += float(np.sum(w * proj**2))
    if use_penalty:
        dist += float(np.sum(np.log(sig.lam + s2))) + (d_dim - k) * np.log(s2)
    return dist


def route_accuracy(sigs, test_embs, tasks, route_fn):
    """Compute overall + per-task routing accuracy."""
    correct, total = 0, 0
    per_task = {}
    for t in tasks:
        if t not in test_embs:
            continue
        embs = test_embs[t]
        c = sum(1 for i in range(embs.shape[0]) if route_fn(embs[i]) == t)
        per_task[t] = c / max(embs.shape[0], 1)
        correct += c; total += embs.shape[0]
    return correct / max(total, 1), per_task


def route_accuracy_vec(sigs, test_embs, tasks, device='cpu',
                       use_mean=True, use_subspace=True, use_penalty=True):
    """Vectorized PSR routing accuracy.  ~100-500x faster than route_accuracy.

    Processes all tasks simultaneously via matrix operations (GPU or CPU).
    """
    task_list = [t for t in tasks if t in sigs and t in test_embs]
    if not task_list:
        return 0.0, {}
    T = len(task_list)
    d = sigs[task_list[0]].d
    k = sigs[task_list[0]].k

    C   = np.stack([sigs[t].mu  for t in task_list]).astype(np.float32)  # (T, d)
    V   = np.stack([sigs[t].V   for t in task_list]).astype(np.float32)  # (T, d, k)
    lam = np.stack([sigs[t].lam for t in task_list]).astype(np.float32)  # (T, k)
    s2  = np.array([sigs[t].sigma2 for t in task_list], dtype=np.float32)# (T,)
    W_psr = lam / (s2[:, None] * (lam + s2[:, None]))                    # (T, k)
    pen   = (np.sum(np.log(lam + s2[:, None]), axis=1)
             + (d - k) * np.log(s2))                                     # (T,)

    use_gpu = "cuda" in device and HAS_TORCH
    if use_gpu:
        dev   = torch.device(device)
        C_t   = torch.tensor(C,     device=dev)
        V_t   = torch.tensor(V,     device=dev)
        W_t   = torch.tensor(W_psr, device=dev)
        pen_t = torch.tensor(pen,   device=dev)
        s2_t  = torch.tensor(s2,    device=dev)

    correct, total = 0, 0
    per_task = {}

    # Precompute centroid projections (avoids (N,T,d) intermediate)
    if use_gpu:
        CV_t = torch.einsum('td,tdk->tk', C_t, V_t)   # (T, k)
        C_sq_t = (C_t ** 2).sum(1)                     # (T,)
    else:
        CV = np.einsum('td,tdk->tk', C, V)             # (T, k)
        C_sq = np.sum(C ** 2, axis=1)                   # (T,)

    for i, t_true in enumerate(task_list):
        H_np = test_embs[t_true].astype(np.float32)  # (N, d)
        N    = H_np.shape[0]
        if use_gpu:
            H = torch.tensor(H_np, device=dev)                     # (N, d)
            H_sq = (H ** 2).sum(1, keepdim=True)                   # (N, 1)
            if use_mean:
                # iso = ||H-C||^2 / s2, via expansion (no (N,T,d) intermediate)
                l2 = H_sq + C_sq_t.unsqueeze(0) - 2 * (H @ C_t.T)  # (N, T)
                iso = l2 / (s2_t.unsqueeze(0) + 1e-12)             # (N, T)
            else:
                iso = H_sq / (s2_t.unsqueeze(0) + 1e-12)           # (N, T)
            dists = iso.clone()
            if use_subspace:
                H_proj = torch.einsum('nd,tdk->ntk', H, V_t)       # (N, T, k)
                if use_mean:
                    dp = H_proj - CV_t.unsqueeze(0)                 # (N, T, k)
                else:
                    dp = H_proj
                dists = dists + (W_t.unsqueeze(0) * dp.pow(2)).sum(-1)
            if use_penalty:
                dists = dists + pen_t.unsqueeze(0)
            preds = dists.argmin(dim=1).cpu().numpy()
            del H, H_sq, iso, dists
        else:
            H  = H_np.astype(np.float64)
            H_sq = np.sum(H ** 2, axis=1, keepdims=True)           # (N, 1)
            if use_mean:
                l2 = H_sq + C_sq[None, :] - 2 * (H @ C.T)          # (N, T)
                iso = l2 / (s2[None, :] + 1e-12)                   # (N, T)
            else:
                iso = H_sq / (s2[None, :] + 1e-12)                 # (N, T)
            dists = iso.copy()
            if use_subspace:
                H_proj = np.einsum('nd,tdk->ntk', H, V)             # (N, T, k)
                if use_mean:
                    dp = H_proj - CV[None, :, :]                    # (N, T, k)
                else:
                    dp = H_proj
                dists += np.sum(W_psr[None, :, :] * dp**2, axis=-1)
            if use_penalty:
                dists += pen[None, :]
            preds = np.argmin(dists, axis=1)
        c = int((preds == i).sum())
        per_task[t_true] = c / max(N, 1)
        correct += c
        total   += N
    return correct / max(total, 1), per_task


# ═══════════════════════════════════════════════════════════════════════
# D1 — PSR Component Ablation
# ═══════════════════════════════════════════════════════════════════════

def ablation_components(train_embs, test_embs, tasks, k=8, device='cpu'):
    """Test PSR with each component enabled/disabled."""
    sigs = {t: PPCASig(e, k, device=device) for t, e in train_embs.items()}
    configs = {
        "Centroid_only":   dict(use_mean=True,  use_subspace=False, use_penalty=False),
        "Subspace_only":   dict(use_mean=False, use_subspace=True,  use_penalty=False),
        "PSR_light":       dict(use_mean=True,  use_subspace=True,  use_penalty=False),
        "PSR_full":        dict(use_mean=True,  use_subspace=True,  use_penalty=True),
        "PSR_no_penalty":  dict(use_mean=True,  use_subspace=True,  use_penalty=False),
    }
    results = {}
    for name, kw in configs.items():
        acc, pt = route_accuracy_vec(sigs, test_embs, tasks, device=device, **kw)
        results[name] = {"accuracy": acc, "per_task": pt}
    return results


# ═══════════════════════════════════════════════════════════════════════
# D2 — Rank sensitivity
# ═══════════════════════════════════════════════════════════════════════

def rank_sweep(train_embs, test_embs, tasks, ks=(2, 4, 8, 16, 32, 64), device='cpu'):
    results = {}
    for k in ks:
        sigs = {t: PPCASig(e, k, device=device) for t, e in train_embs.items()}
        acc, pt = route_accuracy_vec(sigs, test_embs, tasks, device=device)
        mem_per_task = next(iter(sigs.values())).d * k * 4 + k * 4 + next(iter(sigs.values())).d * 4 + 4  # bytes
        results[k] = {"accuracy": acc, "memory_bytes_per_task": mem_per_task}
    return results


# ═══════════════════════════════════════════════════════════════════════
# D3 — Same-domain vs cross-domain
# ═══════════════════════════════════════════════════════════════════════

def domain_analysis(per_task_acc, benchmark):
    clusters = LONG_SEQ_CLUSTERS if benchmark == "Long_Sequence" else SUPERNI_CLUSTERS
    cluster_map = {}
    for c, members in clusters.items():
        for m in members:
            cluster_map[m] = c

    domain_acc = {}
    for c, members in clusters.items():
        accs = [per_task_acc[t] for t in members if t in per_task_acc]
        if accs:
            domain_acc[c] = {"mean": float(np.mean(accs)), "tasks": len(accs),
                             "per_task": {t: per_task_acc[t] for t in members if t in per_task_acc}}
    return domain_acc


# ═══════════════════════════════════════════════════════════════════════
# D6 — Incremental simulation
# ═══════════════════════════════════════════════════════════════════════

def incremental_simulation(train_embs, test_embs, tasks, k=8, device='cpu'):
    """Simulate adding tasks one by one. At step t, only tasks[0:t+1] are available.
    Compares PSR (incremental) vs RLS (incremental Woodbury) vs RLS (batch=upper bound).

    When device='cuda', all RLS matrix ops (Woodbury, ridge solve) run on GPU.
    """
    use_gpu = 'cuda' in device and HAS_TORCH

    psr_results = {}
    rls_inc_results = {}
    rls_batch_results = {}

    alpha_rls = 1.0
    d = next(iter(train_embs.values())).shape[1]

    if use_gpu:
        dev = torch.device(device)
        R_inv_t = None
        XtY_accum_t = None
        # Pre-convert all embeddings to GPU tensors
        train_tensors = {t: torch.tensor(e, dtype=torch.float64, device=dev) for t, e in train_embs.items()}
        test_tensors  = {t: torch.tensor(e, dtype=torch.float64, device=dev) for t, e in test_embs.items()}
        eye_d = torch.eye(d, dtype=torch.float64, device=dev)

        for t_idx in range(1, len(tasks) + 1):
            available = [tasks[i] for i in range(t_idx) if tasks[i] in train_embs]
            if not available:
                continue
            task2idx = {t: i for i, t in enumerate(available)}
            n_cls = len(available)

            # ── PSR (incremental) ──
            sigs = {t: PPCASig(train_embs[t], k, device=device) for t in available}
            test_avail = OrderedDict((t, test_embs[t]) for t in available if t in test_embs)
            acc_psr, _ = route_accuracy_vec(sigs, test_avail, available, device=device)
            psr_results[t_idx] = {"n_tasks": n_cls, "accuracy": acc_psr}

            # ── RLS batch (GPU ridge solve) ──
            Xs_t = torch.cat([train_tensors[t] for t in available], dim=0)
            y_np = np.concatenate([np.full(train_embs[t].shape[0], task2idx[t], dtype=np.int64) for t in available])
            # One-hot Y
            Y_t = torch.zeros(Xs_t.shape[0], n_cls, dtype=torch.float64, device=dev)
            for i, t in enumerate(available):
                start = sum(train_embs[available[j]].shape[0] for j in range(i))
                end = start + train_embs[t].shape[0]
                Y_t[start:end, i] = 1.0
            # W = (X^TX + αI)^{-1} X^T Y
            XtX = Xs_t.T @ Xs_t + alpha_rls * eye_d
            XtY = Xs_t.T @ Y_t
            W_batch = torch.linalg.solve(XtX, XtY)  # (d, n_cls)

            correct_b, total_b = 0, 0
            for t in available:
                if t not in test_tensors:
                    continue
                scores = test_tensors[t] @ W_batch
                preds = scores.argmax(dim=1)
                correct_b += int((preds == task2idx[t]).sum().item())
                total_b += test_tensors[t].shape[0]
            rls_batch_results[t_idx] = {"n_tasks": n_cls,
                                         "accuracy": correct_b / max(total_b, 1)}

            # ── RLS incremental (Woodbury on GPU) ──
            new_task = available[-1]
            X_new = train_tensors[new_task]
            n_new = X_new.shape[0]

            if R_inv_t is None:
                R_inv_t = torch.linalg.inv(X_new.T @ X_new + alpha_rls * eye_d)
                XtY_accum_t = torch.zeros(d, n_cls, dtype=torch.float64, device=dev)
                XtY_accum_t[:, task2idx[new_task]] = X_new.T @ torch.ones(n_new, dtype=torch.float64, device=dev)
            else:
                if n_cls > XtY_accum_t.shape[1]:
                    pad = torch.zeros(d, n_cls - XtY_accum_t.shape[1], dtype=torch.float64, device=dev)
                    XtY_accum_t = torch.cat([XtY_accum_t, pad], dim=1)
                U = X_new.T  # (d, n_new)
                V = X_new     # (n_new, d)
                RU = R_inv_t @ U
                S = torch.eye(n_new, dtype=torch.float64, device=dev) + V @ RU
                try:
                    S_inv_VR = torch.linalg.solve(S, V @ R_inv_t)
                    R_inv_t = R_inv_t - RU @ S_inv_VR
                except Exception:
                    Xs_so_far = torch.cat([train_tensors[t] for t in available], dim=0)
                    R_inv_t = torch.linalg.inv(Xs_so_far.T @ Xs_so_far + alpha_rls * eye_d)
                XtY_accum_t[:, task2idx[new_task]] = X_new.T @ torch.ones(n_new, dtype=torch.float64, device=dev)

            W_inc = R_inv_t @ XtY_accum_t[:, :n_cls]

            correct_i, total_i = 0, 0
            for t in available:
                if t not in test_tensors:
                    continue
                scores = test_tensors[t] @ W_inc
                preds = scores.argmax(dim=1)
                correct_i += int((preds == task2idx[t]).sum().item())
                total_i += test_tensors[t].shape[0]
            rls_inc_results[t_idx] = {"n_tasks": n_cls,
                                       "accuracy": correct_i / max(total_i, 1)}

        del train_tensors, test_tensors
    else:
        # CPU path (original)
        from sklearn.linear_model import RidgeClassifier
        R_inv = None
        XtY_accum = None

        for t_idx in range(1, len(tasks) + 1):
            available = [tasks[i] for i in range(t_idx) if tasks[i] in train_embs]
            if not available:
                continue
            task2idx = {t: i for i, t in enumerate(available)}
            n_cls = len(available)

            # ── PSR ──
            sigs = {t: PPCASig(train_embs[t], k, device=device) for t in available}
            test_avail = OrderedDict((t, test_embs[t]) for t in available if t in test_embs)
            acc_psr, _ = route_accuracy_vec(sigs, test_avail, available, device=device)
            psr_results[t_idx] = {"n_tasks": n_cls, "accuracy": acc_psr}

            # ── RLS batch (sklearn) ──
            Xs, ys = [], []
            for t in available:
                Xs.append(train_embs[t])
                ys.append(np.full(train_embs[t].shape[0], task2idx[t], dtype=np.int64))
            X_all = np.vstack(Xs)
            y_all = np.concatenate(ys)
            ridge = RidgeClassifier(alpha=alpha_rls)
            ridge.fit(X_all, y_all)
            correct_b, total_b = 0, 0
            for t in available:
                if t not in test_embs:
                    continue
                preds = ridge.predict(test_embs[t])
                correct_b += int((preds == task2idx[t]).sum())
                total_b += test_embs[t].shape[0]
            rls_batch_results[t_idx] = {"n_tasks": n_cls,
                                         "accuracy": correct_b / max(total_b, 1)}

            # ── RLS incremental (Woodbury) ──
            new_task = available[-1]
            X_new = train_embs[new_task]
            n_new = X_new.shape[0]
            if R_inv is None:
                R_inv = np.linalg.inv(X_new.T @ X_new + alpha_rls * np.eye(d))
                XtY_accum = np.zeros((d, n_cls))
                XtY_accum[:, task2idx[new_task]] = X_new.T @ np.ones(n_new)
            else:
                if n_cls > XtY_accum.shape[1]:
                    pad = np.zeros((d, n_cls - XtY_accum.shape[1]))
                    XtY_accum = np.concatenate([XtY_accum, pad], axis=1)
                U = X_new.T
                V = X_new
                RU = R_inv @ U
                S = np.eye(n_new) + V @ RU
                try:
                    S_inv = np.linalg.inv(S)
                    R_inv = R_inv - RU @ S_inv @ (V @ R_inv)
                except np.linalg.LinAlgError:
                    Xs_so_far = np.vstack([train_embs[t] for t in available])
                    R_inv = np.linalg.inv(Xs_so_far.T @ Xs_so_far + alpha_rls * np.eye(d))
                XtY_accum[:, task2idx[new_task]] = X_new.T @ np.ones(n_new)
            W_inc = R_inv @ XtY_accum[:, :n_cls]

            correct_i, total_i = 0, 0
            for t in available:
                if t not in test_embs:
                    continue
                scores = test_embs[t] @ W_inc
                preds = np.argmax(scores, axis=1)
                correct_i += int((preds == task2idx[t]).sum())
                total_i += test_embs[t].shape[0]
            rls_inc_results[t_idx] = {"n_tasks": n_cls,
                                       "accuracy": correct_i / max(total_i, 1)}

    return {
        "PSR": {str(k): v for k, v in psr_results.items()},
        "RLS_batch": {str(k): v for k, v in rls_batch_results.items()},
        "RLS_incremental": {str(k): v for k, v in rls_inc_results.items()},
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase D — PSR Ablation")
    parser.add_argument("--emb_dir", required=True)
    parser.add_argument("--benchmark", required=True, choices=list(BENCHMARKS.keys()))
    parser.add_argument("--subspace_k", type=int, default=8)
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--whiten", action="store_true",
                        help="Apply global ZCA whitening")
    parser.add_argument("--compare_backbones", action="store_true",
                        help="If --emb_dir points to parent, compare all backbone subdirs")
    parser.add_argument("--device", default="auto",
                        help="Device: cpu | cuda | cuda:0 | auto (default: auto)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if output already exists")
    args = parser.parse_args()
    args.device = _resolve_device(args.device)

    tasks = BENCHMARKS[args.benchmark]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ── Handle multi-backbone comparison ──
    if args.compare_backbones:
        emb_root = Path(args.emb_dir)
        backbone_dirs = sorted([d for d in emb_root.iterdir() if d.is_dir()
                                and (d / args.benchmark).exists()])
        if len(backbone_dirs) < 2:
            print(f"Need >=2 backbone dirs under {emb_root}, found {len(backbone_dirs)}")
            sys.exit(1)
        compare = {}
        for bd in backbone_dirs:
            print(f"\n{'='*60}\n  Backbone: {bd.name}\n{'='*60}")
            tr = load_all(str(bd), args.benchmark, tasks, "train")
            te = load_all(str(bd), args.benchmark, tasks, "test")
            found = sorted(set(tr) & set(te))
            if not found:
                continue
            tr = OrderedDict((t, tr[t]) for t in found)
            te = OrderedDict((t, te[t]) for t in found)
            sigs = {t: PPCASig(e, args.subspace_k, device=args.device) for t, e in tr.items()}
            acc, pt = route_accuracy_vec(sigs, te, found, device=args.device)
            compare[bd.name] = {"accuracy": acc, "d_model": next(iter(tr.values())).shape[1],
                                "n_tasks": len(found), "per_task": pt}
            print(f"  PSR accuracy: {acc*100:.2f}%  (d={compare[bd.name]['d_model']}, {len(found)} tasks)")
        out_path = out_dir / f"backbone_compare_{args.benchmark}.json"
        with open(out_path, "w") as f:
            json.dump(compare, f, indent=2, default=str)
        print(f"\n✓ Saved {out_path}")
        return

    # ── Single backbone ──
    backbone = Path(args.emb_dir).name
    tag = f"{backbone}_{args.benchmark}" + ("_whitened" if args.whiten else "")

    # ── Skip if already done ──
    out_path_check = out_dir / f"ablation_{tag}.json"
    if out_path_check.exists() and not args.force:
        print(f"[SKIP] Phase D: {out_path_check} already exists. Use --force to re-run.")
        return

    print(f"=== Phase D: PSR Ablation  [{tag}]  k={args.subspace_k} ===\n")

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

    # ── D1: Component ablation ──
    print("─── D1: Component Ablation ───")
    abl = ablation_components(train_embs, test_embs, found, args.subspace_k, device=args.device)
    for name, r in sorted(abl.items(), key=lambda x: -x[1]["accuracy"]):
        print(f"  {name:20s}  {r['accuracy']*100:.2f}%")
    report["D1_ablation"] = {n: {"accuracy": r["accuracy"]} for n, r in abl.items()}

    # ── D2: Rank sweep ──
    print("\n─── D2: Rank sensitivity ───")
    rs = rank_sweep(train_embs, test_embs, found, device=args.device)
    for k_val, r in sorted(rs.items()):
        print(f"  k={k_val:3d}  acc={r['accuracy']*100:.2f}%  mem={r['memory_bytes_per_task']/1024:.1f}KB/task")
    report["D2_rank_sweep"] = {str(k): r for k, r in rs.items()}

    # ── D3: Domain breakdown ──
    print("\n─── D3: Domain breakdown (PSR_full) ───")
    psr_pt = abl["PSR_full"]["per_task"]
    da = domain_analysis(psr_pt, args.benchmark)
    for domain, info in da.items():
        print(f"  {domain:15s}  mean_acc={info['mean']:.2%}  ({info['tasks']} tasks)")
    report["D3_domain"] = da

    # ── D6: Incremental simulation (PSR vs RLS-incremental vs RLS-batch) ──
    print("\n─── D6: Incremental simulation ───")
    inc = incremental_simulation(train_embs, test_embs, found, args.subspace_k, device=args.device)
    print(f"  {'Step':>5s}  {'#Tasks':>6s}  {'PSR':>8s}  {'RLS_inc':>8s}  {'RLS_bat':>8s}")
    for step in sorted(inc["PSR"].keys(), key=int):
        psr_a = inc["PSR"].get(step, {}).get("accuracy", 0)
        rls_i = inc["RLS_incremental"].get(step, {}).get("accuracy", 0)
        rls_b = inc["RLS_batch"].get(step, {}).get("accuracy", 0)
        n_t = inc["PSR"].get(step, {}).get("n_tasks", 0)
        print(f"  {step:>5s}  {n_t:>6d}  {psr_a*100:>7.2f}%  {rls_i*100:>7.2f}%  {rls_b*100:>7.2f}%")
    report["D6_incremental"] = inc

    out_path = out_dir / f"ablation_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n✓ Saved {out_path}")


if __name__ == "__main__":
    main()
