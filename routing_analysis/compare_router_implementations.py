#!/usr/bin/env python3
"""
Compare ShrinkageWhitenedRouter (routing_class.py) vs SRTRouter (srt_router.py)
using IDENTICAL inputs and IDENTICAL parameters.

This test should reveal the exact difference.
"""
import numpy as np
from numpy.linalg import eigh

np.random.seed(42)

# ============================================================
# IMPLEMENTATION 1: ShrinkageWhitenedRouter (routing_class.py)
# ============================================================
class RefShrinkageRouter:
    """Exact copy of routing_class.py ShrinkageWhitenedRouter."""
    def __init__(self, shrink_factor=0.1):
        self.raw_centroids = []
        self.seen_embs = []
        self.mu_g = None
        self.W_zca = None
        self.signatures = []
        self.shrink_factor = shrink_factor

    def add_task(self, embs):
        X = np.array(embs, dtype=np.float64)
        self.raw_centroids.append(X.mean(axis=0))
        self.seen_embs.append(X)
        all_embs = np.vstack(self.seen_embs)

        self.mu_g = all_embs.mean(axis=0)
        cov = np.cov(all_embs, rowvar=False, ddof=1)

        d = cov.shape[0]
        target = (np.trace(cov) / d) * np.eye(d)
        cov = (1 - self.shrink_factor) * cov + self.shrink_factor * target

        eigvals, eigvecs = eigh(cov)
        eigvals = np.maximum(eigvals, 1e-8)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        self.W_zca = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        self.signatures = [(mu_r - self.mu_g) @ self.W_zca.T for mu_r in self.raw_centroids]

    def route(self, h_batch):
        if not self.signatures:
            return np.zeros(h_batch.shape[0], dtype=np.int64)
        H = np.array(h_batch, dtype=np.float64)
        H_w = (H - self.mu_g) @ self.W_zca.T
        C_w = np.stack(self.signatures)
        H_sq = np.sum(H_w ** 2, axis=1, keepdims=True)
        C_sq = np.sum(C_w ** 2, axis=1, keepdims=True).T
        dists = H_sq + C_sq - 2 * (H_w @ C_w.T)
        return dists.argmin(axis=1)


# ============================================================
# IMPLEMENTATION 2: SRTRouter hard mode (srt_router.py)
# ============================================================
def update_pooled_cov(mu_old, cov_old, n_old, mu_new, cov_new, n_new):
    total = n_old + n_new
    mu_pool = (n_old * mu_old + n_new * mu_new) / total
    delta = mu_new - mu_old
    C = (n_old * n_new / total) * np.outer(delta, delta)
    cov_pool = ((n_old - 1) * cov_old + (n_new - 1) * cov_new + C) / (total - 1)
    return mu_pool, cov_pool, total

class SRTRouter:
    """Exact copy of srt_router.py hard mode."""
    def __init__(self, shrink_factor=0.1):
        self.signatures = {}  # dict instead of list
        self._mu_pool = None
        self._Sigma_pool = None
        self._n_pool = 0
        self._mu_global = None
        self._W_zca = None
        self.shrink_factor = shrink_factor

    def _update_pooled(self, mu_t, Sigma_t, n_t):
        if self._n_pool == 0:
            self._mu_pool = mu_t.copy()
            self._Sigma_pool = Sigma_t.copy()
            self._n_pool = n_t
        else:
            self._mu_pool, self._Sigma_pool, self._n_pool = update_pooled_cov(
                self._mu_pool, self._Sigma_pool, self._n_pool, mu_t, Sigma_t, n_t)

    def add_task(self, task_id, h_train):
        h_train = np.array(h_train, dtype=np.float64)
        n_t, d = h_train.shape
        mu_t = h_train.mean(axis=0)
        Sigma_t = np.cov(h_train, rowvar=False, ddof=1)
        self._update_pooled(mu_t, Sigma_t, n_t)

        # Shrink pooled covariance
        trace = np.trace(self._Sigma_pool)
        target = (trace / d) * np.eye(d)
        cov_shrunk = (1 - self.shrink_factor) * self._Sigma_pool + self.shrink_factor * target

        # Refit ZCA
        self._mu_global = self._mu_pool.copy()
        eigvals, eigvecs = eigh(cov_shrunk)
        eigvals = np.maximum(eigvals, 1e-8)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        self._W_zca = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        # Create signature
        mu_raw = mu_t.copy()
        Sigma_raw = Sigma_t.copy()
        mu_whitened = (mu_t - self._mu_global) @ self._W_zca.T
        Sigma_whitened = self._W_zca @ Sigma_raw @ self._W_zca.T

        self.signatures[task_id] = {
            'mu': mu_whitened,
            'mu_raw': mu_raw,
            'Sigma_raw': Sigma_raw,
        }

    def route(self, h):
        if h.ndim == 1:
            h = h.reshape(1, -1)
        h = np.array(h, dtype=np.float64)
        h_w = (h - self._mu_global) @ self._W_zca.T
        task_list = sorted(self.signatures.keys())
        dists = np.zeros((h.shape[0], len(task_list)))
        for i, t_id in enumerate(task_list):
            sig = self.signatures[t_id]
            diff = h_w - sig['mu']
            dists[:, i] = np.sqrt(np.einsum('nd,nd->n', diff, diff))
        return dists.argmin(axis=1)


# ============================================================
# TEST WITH SYNTHETIC DATA (realistic LLaMA embedding geometry)
# ============================================================
print("=" * 70)
print("TEST 1: Synthetic data — 2 tasks, 160 samples each, d=4096")
print("=" * 70)

d = 4096
n1 = 160
n2 = 160

# Task 1: samsum-like embeddings
np.random.seed(42)
mu1_true = np.random.randn(d) * 0.5
X1 = mu1_true + np.random.randn(n1, d) * 0.5

# Task 2: sst2-like embeddings (different mean)
np.random.seed(123)
mu2_true = np.random.randn(d) * 0.5 + 2.0
X2 = mu2_true + np.random.randn(n2, d) * 0.5

# Test embeddings (same distribution as train)
np.random.seed(999)
Xt1 = mu1_true + np.random.randn(n1, d) * 0.5
Xt2 = mu2_true + np.random.randn(n2, d) * 0.5

print(f"\nTask 1 train: {X1.shape}, centroid norm = {np.linalg.norm(X1.mean(0)):.4f}")
print(f"Task 2 train: {X2.shape}, centroid norm = {np.linalg.norm(X2.mean(0)):.4f}")
print(f"Raw centroid distance: {np.linalg.norm(X1.mean(0) - X2.mean(0)):.4f}")

ref = RefShrinkageRouter(shrink_factor=0.1)
srt = SRTRouter(shrink_factor=0.1)

ref.add_task(X1)
srt.add_task('task1', X1)

ref.add_task(X2)
srt.add_task('task2', X2)

# Compare mu_g and W_zca
print(f"\nmu_g diff: {np.max(np.abs(ref.mu_g - srt._mu_global)):.10e}")
print(f"W_zca diff: {np.max(np.abs(ref.W_zca - srt._W_zca)):.10e}")
print(f"W_zca same shape: {ref.W_zca.shape == srt._W_zca.shape}")

# Compare whitened signatures
print(f"\nSignature 1:")
print(f"  Ref norm:    {np.linalg.norm(ref.signatures[0]):.6f}")
print(f"  SRT norm:   {np.linalg.norm(srt.signatures['task1']['mu']):.6f}")
print(f"  Diff:       {np.max(np.abs(ref.signatures[0] - srt.signatures['task1']['mu'])):.10e}")

print(f"\nSignature 2:")
print(f"  Ref norm:    {np.linalg.norm(ref.signatures[1]):.6f}")
print(f"  SRT norm:   {np.linalg.norm(srt.signatures['task2']['mu']):.6f}")
print(f"  Diff:       {np.max(np.abs(ref.signatures[1] - srt.signatures['task2']['mu'])):.10e}")

# Route test embeddings
pred_ref = ref.route(np.vstack([Xt1, Xt2]))
pred_srt = srt.route(np.vstack([Xt1, Xt2]))
print(f"\nRouting predictions (first 5):")
print(f"  Ref: {pred_ref[:5]}")
print(f"  SRT: {pred_srt[:5]}")
print(f"  Match: {np.array_equal(pred_ref, pred_srt)}")

# Compute routing accuracy
correct_ref = (pred_ref[:n1] == 0).sum() + (pred_ref[n1:] == 1).sum()
correct_srt = (pred_srt[:n1] == 0).sum() + (pred_srt[n1:] == 1).sum()
print(f"\nRef accuracy: {correct_ref}/{2*n1} = {correct_ref/(2*n1)*100:.1f}%")
print(f"SRT accuracy: {correct_srt}/{2*n1} = {correct_srt/(2*n1)*100:.1f}%")

# Distance comparison
dists_ref = []
dists_srt = []
for i in range(min(10, n1)):
    h = Xt1[i:i+1]
    h_w_ref = (h - ref.mu_g) @ ref.W_zca.T
    h_w_srt = (h - srt._mu_global) @ srt._W_zca.T
    C_w_ref = np.stack(ref.signatures)
    C_w_srt_task1 = srt.signatures['task1']['mu']
    C_w_srt_task2 = srt.signatures['task2']['mu']
    d_ref = np.sqrt(np.sum((h_w_ref - C_w_ref[0])**2))
    d_srt = np.sqrt(np.sum((h_w_srt - C_w_srt_task1)**2))
    print(f"\n  Sample from Task1:")
    print(f"    Ref dist to task1: {d_ref:.6f}")
    print(f"    SRT dist to task1: {d_srt:.6f}")
    print(f"    h_w_ref norm: {np.linalg.norm(h_w_ref):.6f}")
    print(f"    h_w_srt norm: {np.linalg.norm(h_w_srt):.6f}")
    print(f"    h_w diff: {np.max(np.abs(h_w_ref - h_w_srt)):.10e}")

print()
print("=" * 70)
print("TEST 2: With SAME random data (n/d=0.04) — 15 tasks")
print("=" * 70)

# Create 15 tasks with distinct centroids
n_tasks = 15
n_per_task = 160
all_tasks = []
all_test = []
centroids = []

for t in range(n_tasks):
    np.random.seed(t * 1000)
    # Each task has a random centroid with different magnitude
    mu_t = np.random.randn(d) * (1.0 + t * 0.1)
    X_t = mu_t + np.random.randn(n_per_task, d) * 0.3
    X_test_t = mu_t + np.random.randn(n_per_task, d) * 0.3
    all_tasks.append(X_t)
    all_test.append(X_test_t)
    centroids.append(mu_t)

# Run both routers
ref15 = RefShrinkageRouter(shrink_factor=0.1)
srt15 = SRTRouter(shrink_factor=0.1)

for t in range(n_tasks):
    ref15.add_task(all_tasks[t])
    srt15.add_task(f'task{t}', all_tasks[t])

# Check mu_g and W_zca match
print(f"\nmu_g diff: {np.max(np.abs(ref15.mu_g - srt15._mu_global)):.10e}")
print(f"W_zca diff: {np.max(np.abs(ref15.W_zca - srt15._W_zca)):.10e}")

# Check whitened centroids match
all_match = True
for t in range(n_tasks):
    ref_mu = ref15.signatures[t]
    srt_mu = srt15.signatures[f'task{t}']['mu']
    diff = np.max(np.abs(ref_mu - srt_mu))
    if diff > 1e-10:
        print(f"  Task {t}: sig DIFF = {diff:.10e}")
        all_match = False
print(f"All whitened centroids match: {all_match}")

# Route all test embeddings
all_test_embs = np.vstack(all_test)
true_labels = np.repeat(range(n_tasks), n_per_task)

pred_ref15 = ref15.route(all_test_embs)
pred_srt15 = srt15.route(all_test_embs)

acc_ref15 = (pred_ref15 == true_labels).mean() * 100
acc_srt15 = (pred_srt15 == true_labels).mean() * 100

print(f"\nRef accuracy (15 tasks): {acc_ref15:.2f}%")
print(f"SRT accuracy (15 tasks): {acc_srt15:.2f}%")
print(f"Predictions match: {np.array_equal(pred_ref15, pred_srt15)}")

# Check whitened centroid norms
print(f"\nWhitened centroid norms:")
for t in range(n_tasks):
    ref_norm = np.linalg.norm(ref15.signatures[t])
    srt_norm = np.linalg.norm(srt15.signatures[f'task{t}']['mu'])
    print(f"  Task {t:2d}: ref={ref_norm:.4f}  srt={srt_norm:.4f}  diff={abs(ref_norm-srt_norm):.2e}")
