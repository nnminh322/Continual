#!/usr/bin/env python3
"""
Phase F — Learned Routing Simulators (GPM / RLS / Baselines).
Mô phỏng chính xác sự sụp đổ của ZCA Khóa cứng (Fit-Once) trên không gian LLaMA.
"""
from __future__ import annotations
import argparse, json, math, sys, warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _resolve_device(device_str: str) -> str:
    if device_str == "auto":
        if HAS_TORCH and torch.cuda.is_available():
            try:
                _t = torch.zeros(8, device="cuda") + 1
                del _t
                torch.cuda.synchronize()
                gpu_name = torch.cuda.get_device_name(0)
                return "cuda"
            except Exception as e:
                return "cpu"
        return "cpu"
    return device_str

# HARDCODE CHẾT CỨNG ORDER CỦA ANH ĐỂ TRÁNH TRƯỜNG HỢP CHẠY LỆCH TASK 1
DEFAULT_TASK_ORDER = [
    "task1572_samsum_summary",
    "task363_sst2_polarity_classification",
    "task1290_xsum_summarization",
    "task181_outcome_extraction",
    "task002_quoref_answer_generation",
    "task1510_evalution_relation_extraction",
    "task639_multi_woz_user_utterance_generation",
    "task1729_personachat_generate_next",
    "task073_commonsenseqa_answer_generation",
    "task1590_diplomacy_text_generation",
    "task748_glucose_reverse_cause_event_detection",
    "task511_reddit_tifu_long_text_summarization",
    "task591_sciq_answer_generation",
    "task1687_sentiment140_classification",
    "task875_emotion_classification"
]

BENCHMARKS = {
    "SuperNI": DEFAULT_TASK_ORDER
}

def load_split(emb_dir, benchmark, task, split):
    p = Path(emb_dir) / benchmark / task / f"{split}.npz"
    if not p.exists():
        return None
    data = np.load(str(p), allow_pickle=True)
    return data["embeddings"].astype(np.float32)

def load_all(emb_dir, benchmark, tasks, split):
    out = OrderedDict()
    for t in tasks:
        embs = load_split(emb_dir, benchmark, t, split)
        if embs is not None:
            out[t] = embs
    return out

class TransInputMLP(nn.Module):
    def __init__(self, d_model, hidden_dim, backbone='t5'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.linear2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.act = nn.SiLU()
        self.backbone = backbone
        nn.init.kaiming_uniform_(self.linear1.weight, a=math.sqrt(3))
        nn.init.kaiming_uniform_(self.linear2.weight, a=math.sqrt(3))

    def forward(self, x):
        if self.backbone == 'llama':
            return self.act(self.linear2(self.linear1(x)))
        return self.act(self.linear2(self.act(self.linear1(x))))

    def forward_medium(self, x):
        if self.backbone == 'llama':
            return self.linear1(x)
        return self.act(self.linear1(x))


class FrozenTransInput(nn.Module):
    def __init__(self, input_weights, output_weights, backbone='t5'):
        super().__init__()
        self.register_buffer('W_in', torch.stack(input_weights, dim=0))
        self.register_buffer('W_out', torch.stack(output_weights, dim=0))
        self.act = nn.SiLU()
        self.backbone = backbone

    def forward(self, x):
        mid = torch.einsum('bd,thd->bth', x, self.W_in)
        if self.backbone != 'llama':
            mid = self.act(mid)
        out = self.act(torch.einsum('bth,tdh->btd', mid, self.W_out))
        return out

def cal_attention(prompt_keys, x_features):
    if prompt_keys.dim() == 2:
        prompt_keys = prompt_keys.unsqueeze(0).expand(x_features.shape[0], -1, -1)
    x_norm = x_features / (x_features.norm(dim=-1, keepdim=True) + 1e-12)
    pk_norm = prompt_keys / (prompt_keys.norm(dim=-1, keepdim=True) + 1e-12)
    cos_sim = (x_norm * pk_norm).sum(dim=-1)
    return torch.abs(torch.sigmoid(cos_sim * 4) * 2 - 1)

def _gpm_svd_extract(cov, threshold):
    U, S, _ = torch.linalg.svd(cov, full_matrices=False)
    sval_sq = S ** 2
    sval_total = sval_sq.sum()
    if sval_total.item() < 1e-12:
        return None
    cumsum = torch.cumsum(sval_sq / sval_total, dim=0)
    r = int((cumsum < threshold).sum().item()) + 1
    r = min(max(r, 1), S.shape[0])
    return U[:, :r]

def _gpm_update(old_bases, new_cov, threshold):
    if old_bases is None:
        return _gpm_svd_extract(new_cov, threshold)
    _, S_full, _ = torch.linalg.svd(new_cov, full_matrices=False)
    sval_total = (S_full ** 2).sum()
    act_hat = new_cov - old_bases @ (old_bases.T @ new_cov)
    U_hat, S_hat, _ = torch.linalg.svd(act_hat, full_matrices=False)
    sval_hat = (S_hat ** 2).sum()
    accumulated = (sval_total - sval_hat) / max(sval_total.item(), 1e-12)
    sval_ratio = (S_hat ** 2) / max(sval_total.item(), 1e-12)
    r = 0
    for ii in range(sval_ratio.shape[0]):
        if accumulated < threshold:
            accumulated += sval_ratio[ii].item()
            r += 1
        else:
            break
    if r == 0:
        return old_bases
    updated = torch.cat([old_bases, U_hat[:, :r]], dim=1)
    d = updated.shape[0]
    if updated.shape[1] > d:
        updated = updated[:, :d]
    return updated

class GPMRouter:
    def __init__(self, d_model, mlp_hidden_dim=100, transthreshold=0.995,
                 lr=1e-3, epochs=30, batch_size=256,
                 chunk=1, backbone='t5', device='cpu'):
        self.d_model = d_model
        self.mlp_hidden_dim = mlp_hidden_dim
        self.transthreshold = transthreshold
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.chunk = chunk
        self.step = d_model // chunk
        self.backbone = backbone
        self.device = device
        self.prompt_keys = []
        self.frozen_W_in = []
        self.frozen_W_out = []
        self._cached_frozen_trans = None
        self._cached_frozen_pks = None
        self._cached_n_tasks = 0
        self.gpm_bases_input = {}
        self.gpm_bases_medium = None
        self.gpm_bases_output = {}

    def _adaptive_threshold(self, task_idx, total_tasks=15):
        return (1.0 - self.transthreshold) * task_idx / total_tasks + self.transthreshold

    def add_task(self, task_embs, task_idx, total_tasks=15):
        d = self.d_model
        h = self.mlp_hidden_dim
        C = self.chunk
        step = self.step
        threshold = self._adaptive_threshold(task_idx, total_tasks)
        dev = self.device

        X_gpu = torch.from_numpy(task_embs).float().to(dev)
        N = X_gpu.shape[0]
        mlp = TransInputMLP(d, h, backbone=self.backbone).to(dev)

        if task_idx > 0:
            with torch.no_grad():
                for ci in range(C):
                    s, e = ci * step, (ci + 1) * step
                    bases = self.gpm_bases_input.get(ci)
                    if bases is not None:
                        P = bases @ bases.T
                        mlp.linear1.weight.data[:, s:e] -= (
                            mlp.linear1.weight.data[:, s:e] @ P)
                if self.gpm_bases_medium is not None:
                    P = self.gpm_bases_medium @ self.gpm_bases_medium.T
                    mlp.linear2.weight.data -= mlp.linear2.weight.data @ P

        with torch.no_grad():
            out_feat = mlp(X_gpu)

        pk = torch.zeros(d, device=dev)
        if task_idx == 0:
            with torch.no_grad():
                for ci in range(C):
                    s, e = ci * step, (ci + 1) * step
                    cov_c = (out_feat[:, s:e].T @ out_feat[:, s:e]).div_(max(N, 1))
                    U, _, _ = torch.linalg.svd(cov_c, full_matrices=False)
                    pk[s:e] = U[:, 0]
        else:
            with torch.no_grad():
                for ci in range(C):
                    s, e = ci * step, (ci + 1) * step
                    R = torch.randn(step, step, device=dev)
                    bases = self.gpm_bases_output.get(ci)
                    if bases is not None:
                        P = bases @ bases.T
                        R = R - P @ R
                    U, _, _ = torch.linalg.svd(R, full_matrices=False)
                    pk[s:e] = U[:, 0]

        del out_feat
        pk = pk / (pk.norm() + 1e-12)
        prompt_key = nn.Parameter(pk.to(dev))

        if task_idx > 0:
            self._train_routing(mlp, prompt_key, X_gpu, task_idx)

        with torch.no_grad():
            medium = mlp.forward_medium(X_gpu)
            output = mlp(X_gpu)
            cov_med = (medium.T @ medium).div_(max(N, 1))
            self.gpm_bases_medium = _gpm_update(self.gpm_bases_medium, cov_med, threshold)
            for ci in range(C):
                s, e = ci * step, (ci + 1) * step
                cov_inp = (X_gpu[:, s:e].T @ X_gpu[:, s:e]).div_(max(N, 1))
                cov_out = (output[:, s:e].T @ output[:, s:e]).div_(max(N, 1))
                self.gpm_bases_input[ci] = _gpm_update(self.gpm_bases_input.get(ci), cov_inp, threshold)
                self.gpm_bases_output[ci] = _gpm_update(self.gpm_bases_output.get(ci), cov_out, threshold)

        del medium, output
        self.prompt_keys.append(prompt_key.detach().cpu().numpy())
        self.frozen_W_in.append(mlp.linear1.weight.detach().cpu().numpy())
        self.frozen_W_out.append(mlp.linear2.weight.detach().cpu().numpy())
        del mlp, prompt_key, X_gpu
        if 'cuda' in dev:
            torch.cuda.empty_cache()

    def _build_proj_matrices(self):
        C, step = self.chunk, self.step
        proj = {'input': {}, 'medium': None, 'output': {}}
        for ci in range(C):
            b = self.gpm_bases_input.get(ci)
            if b is not None:
                proj['input'][ci] = b @ b.T
            b = self.gpm_bases_output.get(ci)
            if b is not None:
                proj['output'][ci] = b @ b.T
        if self.gpm_bases_medium is not None:
            proj['medium'] = self.gpm_bases_medium @ self.gpm_bases_medium.T
        return proj

    def _get_frozen_trans_and_pks(self):
        n = len(self.prompt_keys)
        if n == self._cached_n_tasks and self._cached_frozen_trans is not None:
            return self._cached_frozen_trans, self._cached_frozen_pks
        frozen_pks = torch.tensor(np.stack(self.prompt_keys), dtype=torch.float32, device=self.device)
        frozen_trans = FrozenTransInput(
            [torch.tensor(w, dtype=torch.float32) for w in self.frozen_W_in],
            [torch.tensor(w, dtype=torch.float32) for w in self.frozen_W_out],
            backbone=self.backbone,
        ).to(self.device)
        frozen_trans.eval()
        self._cached_frozen_trans = frozen_trans
        self._cached_frozen_pks = frozen_pks
        self._cached_n_tasks = n
        return frozen_trans, frozen_pks

    def _train_routing(self, mlp, prompt_key, X_gpu, task_idx):
        C, step = self.chunk, self.step
        frozen_trans, frozen_pks = self._get_frozen_trans_and_pks()
        optimizer = torch.optim.Adam(list(mlp.parameters()) + [prompt_key], lr=self.lr)
        proj = self._build_proj_matrices()
        N = X_gpu.shape[0]
        bs = self.batch_size
        for epoch in range(self.epochs):
            perm = torch.randperm(N, device=self.device)
            for start in range(0, N, bs):
                idx = perm[start:start + bs]
                x_batch = X_gpu[idx]
                old_W1 = mlp.linear1.weight.data.clone()
                old_W2 = mlp.linear2.weight.data.clone()
                old_pk = prompt_key.data.clone()
                x_cur = mlp(x_batch)
                with torch.no_grad():
                    x_prev = frozen_trans(x_batch)
                all_x = torch.cat([x_cur.unsqueeze(1), x_prev], dim=1)
                all_pk = torch.cat([prompt_key.unsqueeze(0), frozen_pks], dim=0)
                all_pk = all_pk.unsqueeze(0).expand(x_batch.shape[0], -1, -1)
                weights = cal_attention(all_pk, all_x)
                target = torch.zeros(x_batch.shape[0], dtype=torch.long, device=self.device)
                loss = F.cross_entropy(weights * 10, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    W1 = mlp.linear1.weight.data
                    W2 = mlp.linear2.weight.data
                    pk = prompt_key.data
                    W1_norm = W1.norm(dim=1, keepdim=True)
                    W2_norm = W2.norm(dim=1, keepdim=True)
                    pk_norm = pk.norm()
                    for ci in range(C):
                        P = proj['input'].get(ci)
                        if P is not None:
                            s, e = ci * step, (ci + 1) * step
                            W1[:, s:e] -= (W1[:, s:e] - old_W1[:, s:e]) @ P
                    P_med = proj['medium']
                    if P_med is not None:
                        W2.copy_(W2 - (W2 - old_W2) @ P_med)
                    for ci in range(C):
                        P = proj['output'].get(ci)
                        if P is not None:
                            s, e = ci * step, (ci + 1) * step
                            pk[s:e] -= (pk[s:e] - old_pk[s:e]) @ P
                    mlp.linear1.weight.data = W1 * W1_norm / (W1.norm(dim=1, keepdim=True) + 1e-12)
                    mlp.linear2.weight.data = W2 * W2_norm / (W2.norm(dim=1, keepdim=True) + 1e-12)
                    prompt_key.data = pk * (pk_norm / (pk.norm() + 1e-12))

    def route(self, h_batch):
        n_tasks = len(self.prompt_keys)
        if n_tasks == 0:
            raise ValueError("No tasks added yet")
        if n_tasks == 1:
            return np.zeros(h_batch.shape[0], dtype=np.int64)
        X = torch.from_numpy(h_batch).float().to(self.device)
        frozen_trans, all_pk = self._get_frozen_trans_and_pks()
        cs = 2048 if 'cuda' in self.device else 256
        all_preds = []
        with torch.no_grad():
            for i in range(0, X.shape[0], cs):
                xc = X[i:i + cs]
                all_x = frozen_trans(xc)
                w = cal_attention(all_pk, all_x)
                all_preds.append(w.argmax(dim=1).cpu())
        del X
        return torch.cat(all_preds).numpy()


class RLSRouter:
    def __init__(self, d_model, expansion_dim=2048, lam=0.1, seed=42, device='cpu'):
        self.d_model = d_model
        self.E = expansion_dim
        self.lam = lam
        self.device = device
        self.use_gpu = 'cuda' in device and HAS_TORCH
        rng = np.random.RandomState(seed)
        W_phi_np = (rng.randn(d_model, expansion_dim) / np.sqrt(d_model)).astype(np.float32)
        b_phi_np = (rng.randn(expansion_dim) * 0.01).astype(np.float32)
        if self.use_gpu:
            dev = torch.device(device)
            self.W_phi = torch.tensor(W_phi_np, dtype=torch.float32, device=dev)
            self.b_phi = torch.tensor(b_phi_np, dtype=torch.float32, device=dev)
            self.R = torch.eye(expansion_dim, dtype=torch.float64, device=dev) / lam
            self.Q = torch.zeros(expansion_dim, 0, dtype=torch.float64, device=dev)
            self.W_r = torch.zeros(expansion_dim, 0, dtype=torch.float64, device=dev)
        else:
            self.W_phi = W_phi_np
            self.b_phi = b_phi_np
            self.R = np.eye(expansion_dim, dtype=np.float64) / lam
            self.Q = np.zeros((expansion_dim, 0), dtype=np.float64)
            self.W_r = np.zeros((expansion_dim, 0), dtype=np.float64)
        self.num_tasks = 0

    def _expand(self, h):
        if self.use_gpu:
            if not isinstance(h, torch.Tensor):
                h = torch.tensor(h, dtype=torch.float32, device=torch.device(self.device))
            return torch.relu(h @ self.W_phi + self.b_phi).to(torch.float64)
        return np.maximum(0, h @ self.W_phi + self.b_phi)

    def add_task(self, task_embs):
        if self.use_gpu:
            dev = torch.device(self.device)
            if isinstance(task_embs, np.ndarray):
                task_embs_t = torch.tensor(task_embs, dtype=torch.float32, device=dev)
            else:
                task_embs_t = task_embs.to(dev)
            H = self._expand(task_embs_t)
            N = H.shape[0]
            chunk = 512
            R = self.R.clone()
            for start in range(0, N, chunk):
                Hc = H[start:min(start + chunk, N)]
                RH = R @ Hc.T
                S = torch.eye(Hc.shape[0], dtype=torch.float64, device=dev) + Hc @ RH
                try:
                    S_inv_HcR = torch.linalg.solve(S, Hc @ R)
                    R = R - RH @ S_inv_HcR
                except Exception:
                    pass
            R = (R + R.T) * 0.5
            R += 1e-6 * torch.eye(self.E, dtype=torch.float64, device=dev)
            self.R = R
            extra = torch.zeros(self.E, 1, dtype=torch.float64, device=dev)
            extra[:, 0] = H.T @ torch.ones(N, dtype=torch.float64, device=dev)
            self.Q = torch.cat([self.Q, extra], dim=1)
            self.W_r = self.R @ self.Q
            self.num_tasks += 1
        else:
            H = self._expand(task_embs).astype(np.float64)
            N = H.shape[0]
            chunk = 512
            R = self.R.copy()
            for start in range(0, N, chunk):
                Hc = H[start:min(start + chunk, N)]
                RH = R @ Hc.T
                S = np.eye(Hc.shape[0]) + Hc @ RH
                try:
                    S_inv_HcR = np.linalg.solve(S, Hc @ R)
                    R = R - RH @ S_inv_HcR
                except np.linalg.LinAlgError:
                    pass
            R = (R + R.T) * 0.5
            R += 1e-6 * np.eye(self.E)
            self.R = R
            tid = self.num_tasks
            extra = np.zeros((self.E, 1), dtype=np.float64)
            extra[:, 0] = H.T @ np.ones(N)
            self.Q = np.hstack([self.Q, extra])
            self.W_r = self.R @ self.Q
            self.num_tasks += 1

    def route(self, h_batch):
        if self.use_gpu:
            dev = torch.device(self.device)
            if isinstance(h_batch, np.ndarray):
                h_batch = torch.tensor(h_batch, dtype=torch.float32, device=dev)
            H = self._expand(h_batch)
            logits = H @ self.W_r
            return logits.argmax(dim=1).cpu().numpy()
        H = self._expand(h_batch).astype(np.float64)
        logits = H @ self.W_r
        return logits.argmax(axis=1)


class NearestCentroidRouter:
    def __init__(self, device='cpu'):
        self.centroids = []
        self.device = device
        self.use_gpu = 'cuda' in device and HAS_TORCH

    def add_task(self, embs):
        if self.use_gpu:
            dev = torch.device(self.device)
            X = torch.tensor(embs, dtype=torch.float32, device=dev)
            self.centroids.append(X.mean(0))
            del X
        else:
            self.centroids.append(embs.mean(0))

    def route(self, h_batch):
        if self.use_gpu:
            dev = torch.device(self.device)
            C = torch.stack(self.centroids)
            if isinstance(h_batch, np.ndarray):
                H = torch.tensor(h_batch, dtype=torch.float32, device=dev)
            else:
                H = h_batch.to(dev)
            H_sq = (H ** 2).sum(1, keepdim=True)
            C_sq = (C ** 2).sum(1, keepdim=True).T
            dists = H_sq + C_sq - 2 * (H @ C.T)
            return dists.argmin(dim=1).cpu().numpy()
        C = np.stack(self.centroids)
        H_sq = np.sum(h_batch ** 2, axis=1, keepdims=True)
        C_sq = np.sum(C ** 2, axis=1, keepdims=True).T
        dists = H_sq + C_sq - 2 * (h_batch @ C.T)
        return dists.argmin(axis=1)


class CosineNearestCentroidRouter:
    def __init__(self, device='cpu'):
        self.centroids = []
        self.device = device
        self.use_gpu = 'cuda' in device and HAS_TORCH

    def add_task(self, embs):
        if self.use_gpu:
            dev = torch.device(self.device)
            X = torch.tensor(embs, dtype=torch.float32, device=dev)
            mu = X.mean(0)
            self.centroids.append(mu / (mu.norm() + 1e-12))
            del X
        else:
            mu = embs.mean(0)
            self.centroids.append(mu / (np.linalg.norm(mu) + 1e-12))

    def route(self, h_batch):
        if self.use_gpu:
            dev = torch.device(self.device)
            C = torch.stack(self.centroids)
            if isinstance(h_batch, np.ndarray):
                H = torch.tensor(h_batch, dtype=torch.float32, device=dev)
            else:
                H = h_batch.to(dev)
            H_norm = H / (H.norm(dim=1, keepdim=True) + 1e-12)
            sims = H_norm @ C.T
            return sims.argmax(dim=1).cpu().numpy()
        C = np.stack(self.centroids)
        h_norm = h_batch / (np.linalg.norm(h_batch, axis=1, keepdims=True) + 1e-12)
        sims = h_norm @ C.T
        return sims.argmax(axis=1)


class PSRRouter:
    def __init__(self, k=8, device='cpu'):
        self.k = k
        self.sigs = []
        self.device = device
        self.use_gpu = 'cuda' in device and HAS_TORCH

    def add_task(self, embs):
        n, d = embs.shape
        if self.use_gpu:
            dev = torch.device(self.device)
            X = torch.tensor(embs, dtype=torch.float32, device=dev)
            mu = X.mean(0)
            Xc = X - mu
            cov_t = (Xc.T @ Xc) / max(n - 1, 1)
            ev, evec = torch.linalg.eigh(cov_t)
            idx = torch.argsort(ev, descending=True)
            ev = ev[idx]; evec = evec[:, idx]
            k = min(self.k, d)
            V = evec[:, :k]
            lam = torch.clamp(ev[:k], min=1e-12)
            sigma2 = float(max(ev[k:].mean().item(), 1e-12)) if k < d else 1e-12
            self.sigs.append((mu, V, lam, sigma2, d))
            del X, Xc, cov_t, ev, evec
        else:
            mu = embs.mean(0)
            cov = np.cov(embs, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov)
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
            k = min(self.k, d)
            V = eigvecs[:, :k]
            lam = np.maximum(eigvals[:k], 1e-12)
            sigma2 = max(eigvals[k:].mean() if k < d else 1e-12, 1e-12)
            self.sigs.append((mu, V, lam, sigma2, d))

    def route(self, h_batch):
        T = len(self.sigs)
        if T == 0 or T == 1:
            return np.zeros(h_batch.shape[0], dtype=np.int64)

        if self.use_gpu:
            dev = torch.device(self.device)
            d = self.sigs[0][4]
            k = min(self.k, d)
            C   = torch.stack([s[0] for s in self.sigs])
            V   = torch.stack([s[1] for s in self.sigs])
            lam = torch.stack([s[2] for s in self.sigs])
            s2  = torch.tensor([s[3] for s in self.sigs], dtype=torch.float32, device=dev)
            W_psr = lam / (s2[:, None] * (lam + s2[:, None]))
            pen = torch.log(lam + s2[:, None]).sum(1) + (d - k) * torch.log(s2)
            if isinstance(h_batch, np.ndarray):
                H = torch.tensor(h_batch, dtype=torch.float32, device=dev)
            else:
                H = h_batch.to(dev)
            H_sq = (H ** 2).sum(1, keepdim=True)
            C_sq = (C ** 2).sum(1)
            l2   = H_sq + C_sq.unsqueeze(0) - 2 * (H @ C.T)
            iso  = l2 / (s2[None, :] + 1e-12)
            H_proj = torch.einsum('nd,tdk->ntk', H, V)
            CV     = torch.einsum('td,tdk->tk', C, V)
            dp     = H_proj - CV.unsqueeze(0)
            dists = iso + (W_psr[None, :, :] * dp.pow(2)).sum(-1) + pen[None, :]
            return dists.argmin(dim=1).cpu().numpy().astype(np.int64)
        else:
            d = self.sigs[0][4]
            k = min(self.k, d)
            C   = np.stack([s[0] for s in self.sigs]).astype(np.float32)
            V   = np.stack([s[1] for s in self.sigs]).astype(np.float32)
            lam = np.stack([s[2] for s in self.sigs]).astype(np.float32)
            s2  = np.array([s[3] for s in self.sigs], dtype=np.float32)
            W_psr = lam / (s2[:, None] * (lam + s2[:, None]))
            pen = np.sum(np.log(lam + s2[:, None]), axis=1) + (d - k) * np.log(s2)
            H = h_batch.astype(np.float32)
            H_sq = np.sum(H ** 2, axis=1, keepdims=True)
            C_sq = np.sum(C ** 2, axis=1)
            l2   = H_sq + C_sq[None, :] - 2 * (H @ C.T)
            iso  = l2 / (s2[None, :] + 1e-12)
            H_proj = np.einsum('nd,tdk->ntk', H, V)
            CV     = np.einsum('td,tdk->tk', C, V)
            dp     = H_proj - CV[None, :, :]
            dists = iso + np.sum(W_psr[None, :, :] * dp**2, axis=-1) + pen[None, :]
            return np.argmin(dists, axis=1).astype(np.int64)


# --- [A/B TEST ROUTERS] ---

class BuggyFitOnceWhitenedRouter:
    """
    Mô phỏng ĐÚNG SỰ THẬT TÀN KHỐC:
    1. ZCA fit 1 lần ở Task 1 (N=160, d=4096 => Singular).
    2. W_zca_1 chứa các chiều khuếch đại nhiễu khổng lồ (1e4).
    3. Tâm Task 1 (mu_raw_1) bị whiten sẽ nằm chính xác ở gốc tọa độ (0,0..0).
    4. Lúc Eval Task 1, dữ liệu test của Task 1 (khác với train) đi qua W_zca_1 
       sẽ bị văng ra xa do nhiễu ở các chiều null-space. Nó không còn nằm ở (0,0) nữa.
    5. Nó bị các Signature của Task 2, Task 3 (vốn bị văng lung tung) "hút" mất.
    """
    def __init__(self, device='cpu'):
        self.signatures = []
        self.mu_g = None
        self.W_zca = None
        self.zca_fitted = False
        self.device = device
        self.use_gpu = 'cuda' in device and HAS_TORCH

    def add_task(self, embs):
        X = embs.detach().cpu().numpy() if isinstance(embs, torch.Tensor) else np.array(embs)
        
        if not self.zca_fitted:
            # 1. Fit ZCA ở Task 1 (samsum)
            self.mu_g = X.mean(axis=0)
            cov = np.cov(X, rowvar=False, ddof=1)
            
            # Mô phỏng sự tàn khốc của LLaMA (4096 chiều):
            # Các eigenvalue nhỏ sẽ bị ép về 1e-8, tạo ra multiplier = 1/sqrt(1e-8) = 10000
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 1e-8)
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
            self.W_zca = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            self.zca_fitted = True
            
        # 2. Tính Signature (Biến đổi bằng cái W_zca rác rưởi này)
        mu_raw = X.mean(axis=0)
        signature = (mu_raw - self.mu_g) @ self.W_zca.T
        self.signatures.append(signature)

    def route(self, h_batch):
        if not self.signatures: return np.zeros(h_batch.shape[0], dtype=np.int64)
        H = h_batch.detach().cpu().numpy() if isinstance(h_batch, torch.Tensor) else np.array(h_batch)
        
        # 3. Eval: Dữ liệu bị bóp méo bởi W_zca rác
        H_w = (H - self.mu_g) @ self.W_zca.T
        C_w = np.stack(self.signatures)
        
        # 4. Tính L2 (Nearest Centroid)
        H_sq = np.sum(H_w ** 2, axis=1, keepdims=True)
        C_sq = np.sum(C_w ** 2, axis=1, keepdims=True).T
        dists = H_sq + C_sq - 2 * (H_w @ C_w.T)
        
        # In ra khoảng cách để debug: Anh sẽ thấy Dists nổ tung hàng triệu
        return dists.argmin(axis=1)


class IncrementalWhitenedRouter:
    def __init__(self, device='cpu'):
        self.raw_centroids = []
        self.seen_embs = []
        self.mu_g = None
        self.W_zca = None
        self.signatures = []
        self.device = device

    def add_task(self, embs):
        X = embs.detach().cpu().numpy() if isinstance(embs, torch.Tensor) else np.array(embs)
        self.raw_centroids.append(X.mean(axis=0))
        self.seen_embs.append(X)
        all_embs = np.vstack(self.seen_embs)
        
        self.mu_g = all_embs.mean(axis=0)
        cov = np.cov(all_embs, rowvar=False, ddof=1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-8)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        self.W_zca = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        
        self.signatures = [(mu_r - self.mu_g) @ self.W_zca.T for mu_r in self.raw_centroids]

    def route(self, h_batch):
        if not self.signatures: return np.zeros(h_batch.shape[0], dtype=np.int64)
        H = h_batch.detach().cpu().numpy() if isinstance(h_batch, torch.Tensor) else np.array(h_batch)
        H_w = (H - self.mu_g) @ self.W_zca.T
        C_w = np.stack(self.signatures)
        H_sq = np.sum(H_w ** 2, axis=1, keepdims=True)
        C_sq = np.sum(C_w ** 2, axis=1, keepdims=True).T
        dists = H_sq + C_sq - 2 * (H_w @ C_w.T)
        return dists.argmin(axis=1)


class ShrinkageWhitenedRouter:
    def __init__(self, shrink_factor=0.1, device='cpu'):
        self.raw_centroids = []
        self.seen_embs = []
        self.mu_g = None
        self.W_zca = None
        self.signatures = []
        self.shrink_factor = shrink_factor
        self.device = device

    def add_task(self, embs):
        X = embs.detach().cpu().numpy() if isinstance(embs, torch.Tensor) else np.array(embs)
        self.raw_centroids.append(X.mean(axis=0))
        self.seen_embs.append(X)
        all_embs = np.vstack(self.seen_embs)
        
        self.mu_g = all_embs.mean(axis=0)
        cov = np.cov(all_embs, rowvar=False, ddof=1)
        
        d = cov.shape[0]
        target = (np.trace(cov) / d) * np.eye(d)
        cov = (1 - self.shrink_factor) * cov + self.shrink_factor * target
        
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-8)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        self.W_zca = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        
        self.signatures = [(mu_r - self.mu_g) @ self.W_zca.T for mu_r in self.raw_centroids]

    def route(self, h_batch):
        if not self.signatures: return np.zeros(h_batch.shape[0], dtype=np.int64)
        H = h_batch.detach().cpu().numpy() if isinstance(h_batch, torch.Tensor) else np.array(h_batch)
        H_w = (H - self.mu_g) @ self.W_zca.T
        C_w = np.stack(self.signatures)
        H_sq = np.sum(H_w ** 2, axis=1, keepdims=True)
        C_sq = np.sum(C_w ** 2, axis=1, keepdims=True).T
        dists = H_sq + C_sq - 2 * (H_w @ C_w.T)
        return dists.argmin(axis=1)


# ═══════════════════════════════════════════════════════════════════════
# Incremental Evaluation
# ═══════════════════════════════════════════════════════════════════════

def run_incremental_comparison(train_embs, test_embs, tasks, args):
    d = next(iter(train_embs.values())).shape[1]

    routers = OrderedDict()
    routers["NearestCentroid"] = NearestCentroidRouter(device=args.device)
    routers["CosineNearestCentroid"] = CosineNearestCentroidRouter(device=args.device)
    routers["PSR"] = PSRRouter(k=args.subspace_k, device=args.device)
    routers["Buggy_FitOnce_Whiten"] = BuggyFitOnceWhitenedRouter(device=args.device)  
    routers["Incremental_ReWhiten"] = IncrementalWhitenedRouter(device=args.device)       
    routers["Shrinkage_ReWhiten"] = ShrinkageWhitenedRouter(shrink_factor=0.1, device=args.device)
    routers["RLS_Woodbury"] = RLSRouter(
        d, expansion_dim=args.rls_expansion, lam=args.rls_lambda,
        device=args.device)

    if HAS_TORCH:
        routers["GPM_ROOT"] = GPMRouter(
            d, mlp_hidden_dim=args.mlp_hidden_dim,
            transthreshold=args.transthreshold,
            lr=args.lr, epochs=args.epochs,
            batch_size=args.batch_size,
            chunk=args.chunk, backbone=args.backbone_type,
            device=args.device)

    total_tasks = len(tasks)
    all_results = {name: [] for name in routers}

    added_tasks = []

    for t_idx, task_name in enumerate(tasks):
        if task_name not in train_embs:
            continue

        print(f"\n  [{t_idx+1}/{total_tasks}] Adding task: {task_name}")
        added_tasks.append(task_name)

        for name, router in routers.items():
            if name == "GPM_ROOT":
                router.add_task(train_embs[task_name], t_idx, total_tasks)
            elif name == "RLS_Woodbury":
                router.add_task(train_embs[task_name])
            else:
                router.add_task(train_embs[task_name])

        seen_tasks = [t for t in added_tasks if t in test_embs]
        if not seen_tasks:
            continue

        for name, router in routers.items():
            per_task_acc = []
            for true_task in seen_tasks:
                embs_test = test_embs[true_task]
                
                preds = router.route(embs_test)
                true_idx = added_tasks.index(true_task)
                
                correct = int((preds == true_idx).sum())
                total = embs_test.shape[0]
                acc = correct / max(total, 1)
                
                per_task_acc.append(acc)

            macro_acc = sum(per_task_acc) / len(per_task_acc) if per_task_acc else 0.0
            row_str = " | ".join([f"{a*100:5.1f}%" for a in per_task_acc])
            
            all_results[name].append({
                "step": t_idx + 1,
                "n_tasks": len(seen_tasks),
                "accuracy": macro_acc,
                "per_task": {t: a for t, a in zip(seen_tasks, per_task_acc)},
            })
            print(f"    {name:25s}  macro_acc={macro_acc*100:6.2f}%   Row: [{row_str}]")

    return all_results

# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dir", required=True)
    parser.add_argument("--benchmark", required=True, choices=list(BENCHMARKS.keys()))
    parser.add_argument("--subspace_k", type=int, default=8)
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--whiten", action="store_true")

    gpm = parser.add_argument_group("GPM/ROOT routing parameters")
    gpm.add_argument("--mlp_hidden_dim", type=int, default=None)
    gpm.add_argument("--transthreshold", type=float, default=0.995)
    gpm.add_argument("--chunk", type=int, default=None)
    gpm.add_argument("--backbone_type", default="auto", choices=["t5", "llama", "auto"])
    gpm.add_argument("--lr", type=float, default=1e-3)
    gpm.add_argument("--epochs", type=int, default=30)
    gpm.add_argument("--batch_size", type=int, default=256)
    gpm.add_argument("--device", default="auto")
    gpm.add_argument("--force", action="store_true")

    rls = parser.add_argument_group("RLS routing parameters")
    rls.add_argument("--rls_expansion", type=int, default=2048)
    rls.add_argument("--rls_lambda", type=float, default=0.1)

    parser.add_argument("--task_order", type=str, default=None)

    args = parser.parse_args()
    args.device = _resolve_device(args.device)

    backbone = Path(args.emb_dir).name
    if args.backbone_type == 'auto':
        args.backbone_type = 'llama' if 'llama' in backbone.lower() else 't5'
    is_llama = (args.backbone_type == 'llama')

    if args.mlp_hidden_dim is None:
        args.mlp_hidden_dim = 50 if is_llama else 100
    if args.chunk is None:
        args.chunk = 4 if is_llama else 1

    # TỰ ĐỘNG ÉP TASK ORDER CỦA ANH NẾU KHÔNG TRUYỀN VÀO TỪ TERMINAL
    if args.task_order:
        tasks = [t.strip() for t in args.task_order.split(",") if t.strip()]
    else:
        tasks = DEFAULT_TASK_ORDER
        
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{backbone}_{args.benchmark}" + ("_whitened" if args.whiten else "")
    if args.task_order:
        tag += "_custom_order"

    out_path = out_dir / f"learned_routing_{tag}.json"
    if out_path.exists() and not args.force:
        print(f"[SKIP] Phase F: {out_path} already exists. Use --force to re-run.")
        return

    print(f"=== Phase F: Learned Routing Comparison  [{tag}] ===")
    print(f"    Task Order: {tasks}\n")

    train_embs = load_all(args.emb_dir, args.benchmark, tasks, "train")
    test_embs = load_all(args.emb_dir, args.benchmark, tasks, "test")
    found = sorted(set(train_embs) & set(test_embs))
    if not found:
        print("ERROR: No tasks found."); sys.exit(1)
    
    ordered_found = [t for t in tasks if t in found]

    print(f"Tasks found: {len(ordered_found)}/{len(tasks)}")

    # 1. ÉP N=160 (CHÍNH XÁC NHƯ TRONG LOG THỰC TẾ CỦA ANH)
    # 2. CHỈ TRÍCH XUẤT 160 SAMPLES ĐẦU TIÊN TỪ train_embs.
    train_embs = OrderedDict((t, train_embs[t][:160]) for t in ordered_found)
    test_embs = OrderedDict((t, test_embs[t]) for t in ordered_found)

    if args.whiten:
        from compare_routing import compute_whitening, apply_whitening
        mu_g, W = compute_whitening(train_embs, device=args.device)
        train_embs = apply_whitening(train_embs, mu_g, W, device=args.device)
        test_embs = apply_whitening(test_embs, mu_g, W, device=args.device)
        train_embs = OrderedDict((t, e.astype(np.float32)) for t, e in train_embs.items())
        test_embs = OrderedDict((t, e.astype(np.float32)) for t, e in test_embs.items())
        print("Applied ZCA whitening\n")

    results = run_incremental_comparison(train_embs, test_embs, ordered_found, args)

    print(f"\n{'='*70}")
    print(f"  Final Routing Accuracy (all {len(ordered_found)} tasks)")
    print(f"{'='*70}")
    print(f"  {'Method':25s}  {'Accuracy':>10s}")
    print(f"  {'-'*37}")
    final_accs = {}
    for name, steps in results.items():
        if steps:
            final = steps[-1]
            final_accs[name] = final["accuracy"]
            print(f"  {name:25s}  {final['accuracy']*100:>8.2f}%")

if __name__ == "__main__":
    main()