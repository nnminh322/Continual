#!/usr/bin/env python3
"""
Debug: Xác nhận root cause của vấn đề n/d = 0.04.

So sánh hai kịch bản:
1. Incremental ZCA (fit lần đầu với n/d = 0.04) → như srt_router.py hiện tại
2. Global ZCA (fit một lần trên tất cả tasks, n/d = 0.59) → như reference với --whiten

Hypothesis: Incremental ZCA với n/d = 0.04 tạo ra whitened space gần như
isotropic (ZCA ≈ identity), dẫn đến khoảng cách gần như bằng nhau cho tất cả centroids.
"""
import numpy as np
from numpy.linalg import eigh

# Simulate embedding geometry: 15 tasks, 160 samples/task, d=4096
np.random.seed(42)
n_tasks = 15
n_samples_per_task = 160
d = 4096

print(f"d={d}, n/task={n_samples_per_task}, n_total={n_tasks*n_samples_per_task}")
print(f"n/d per task = {n_samples_per_task/d:.4f}")
print(f"n/d total    = {n_tasks*n_samples_per_task/d:.4f}")
print()

# Create realistic task embeddings: each task has a distinct direction
# Simulate: h = task_direction * task_scale + noise
task_directions = []
for t in range(n_tasks):
    # Each task has a dominant direction (spread across embedding space)
    # Real embeddings have structure: ~50-200 "effective" dimensions
    dir_t = np.random.randn(d)
    dir_t = dir_t / np.linalg.norm(dir_t)

    # Create 160 embeddings around the task centroid
    centroid = dir_t * (5.0 + t * 0.3)  # task centroids at different radii
    noise = np.random.randn(n_samples_per_task, d) * 0.5
    embs = centroid + noise
    task_directions.append(embs)

print("=" * 70)
print("SCENARIO 1: Incremental ZCA (fit lần đầu với n=160, n/d=0.04)")
print("=" * 70)

# Task 1: fit ZCA from just task 1 (n=160)
embs_t1 = task_directions[0]
mu1 = embs_t1.mean(0)
Sigma1 = np.cov(embs_t1, rowvar=False, ddof=1)

# Shrink pooled (only task 1)
trace = np.trace(Sigma1)
target = (trace / d) * np.eye(d)
for shrink_factor in [0.1, 0.5, 0.9]:
    cov_shrunk = (1 - shrink_factor) * Sigma1 + shrink_factor * target
    eigvals, eigvecs = eigh(cov_shrunk)
    eigvals = np.maximum(eigvals, 1e-8)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    W_zca = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    print(f"\nShrink factor = {shrink_factor}:")
    print(f"  Eigenvalue range: [{eigvals.min():.6f}, {eigvals.max():.6f}]")
    print(f"  Eigenvalue ratio (max/min): {eigvals.max()/eigvals.min():.2e}")
    print(f"  Whitened space condition number: {eigvals.max()/eigvals.min():.2e}")

    # Whiten all task 1 centroids
    mu1_w = (mu1 - mu1) @ W_zca.T
    print(f"  Whitened centroid norm: {np.linalg.norm(mu1_w):.6f}")

    # Simulate adding task 2
    embs_t2 = task_directions[1]
    mu2 = embs_t2.mean(0)
    Sigma2 = np.cov(embs_t2, rowvar=False, ddof=1)

    # Pooled: task1 + task2 (n=320)
    n_pool = 320
    mu_pool = (mu1 * 160 + mu2 * 160) / 320
    delta = mu2 - mu1
    C = (160 * 160 / 320) * np.outer(delta, delta)
    cov_pool = (
        159 * Sigma1 + 159 * Sigma2 + C
    ) / 319

    cov_shrunk_p = (1 - shrink_factor) * cov_pool + shrink_factor * target
    eigvals_p, eigvecs_p = eigh(cov_shrunk_p)
    eigvals_p = np.maximum(eigvals_p, 1e-8)
    idx_p = np.argsort(eigvals_p)[::-1]
    eigvals_p = eigvals_p[idx_p]
    eigvecs_p = eigvecs_p[:, idx_p]
    W_zca_p = eigvecs_p @ np.diag(1.0 / np.sqrt(eigvals_p)) @ eigvecs_p.T

    print(f"  After task 2 (n=320, n/d={320/d:.4f}):")
    print(f"    Eigenvalue range: [{eigvals_p.min():.6f}, {eigvals_p.max():.6f}]")
    print(f"    Condition number: {eigvals_p.max()/eigvals_p.min():.2e}")

    # Whiten both centroids
    mu1_w2 = (mu1 - mu_pool) @ W_zca_p.T
    mu2_w2 = (mu2 - mu_pool) @ W_zca_p.T
    d12 = np.linalg.norm(mu1_w2 - mu2_w2)
    print(f"    L2 dist(whitened_centroid1, whitened_centroid2): {d12:.6f}")

    # Also show L2 in raw space
    d12_raw = np.linalg.norm(mu1 - mu2)
    print(f"    L2 dist(raw_centroid1, raw_centroid2): {d12_raw:.6f}")

    # For each whitened centroid, also show its norm (distance from origin)
    print(f"    ||mu1_whitened|| = {np.linalg.norm(mu1_w2):.6f}")
    print(f"    ||mu2_whitened|| = {np.linalg.norm(mu2_w2):.6f}")

print()
print("=" * 70)
print("SCENARIO 2: Global ZCA (fit một lần trên tất cả 15 tasks, n=2400, n/d=0.59)")
print("=" * 70)

all_embs = np.vstack(task_directions)
print(f"  Total samples: {all_embs.shape[0]}, dims: {all_embs.shape[1]}")

mu_global = all_embs.mean(0)
cov_global = np.cov(all_embs, rowvar=False, ddof=1)

for shrink_factor in [0.1, 0.5, 0.9]:
    trace_g = np.trace(cov_global)
    target_g = (trace_g / d) * np.eye(d)
    cov_shrunk_g = (1 - shrink_factor) * cov_global + shrink_factor * target_g
    eigvals_g, eigvecs_g = eigh(cov_shrunk_g)
    eigvals_g = np.maximum(eigvals_g, 1e-8)
    idx_g = np.argsort(eigvals_g)[::-1]
    eigvals_g = eigvals_g[idx_g]
    eigvecs_g = eigvecs_g[:, idx_g]
    W_zca_g = eigvecs_g @ np.diag(1.0 / np.sqrt(eigvals_g)) @ eigvecs_g.T

    print(f"\nShrink factor = {shrink_factor}:")
    print(f"  Eigenvalue range: [{eigvals_g.min():.6f}, {eigvals_g.max():.6f}]")
    print(f"  Condition number: {eigvals_g.max()/eigvals_g.min():.2e}")

    # Whiten all centroids
    for t_idx in range(n_tasks):
        mu_t = task_directions[t_idx].mean(0)
        mu_t_w = (mu_t - mu_global) @ W_zca_g.T
        print(f"    Task {t_idx}: ||mu_w|| = {np.linalg.norm(mu_t_w):.6f}")

    # Compute pairwise distances
    centroids_w = []
    for t_idx in range(n_tasks):
        mu_t = task_directions[t_idx].mean(0)
        mu_t_w = (mu_t - mu_global) @ W_zca_g.T
        centroids_w.append(mu_t_w)
    centroids_w = np.stack(centroids_w)

    # L2 distance matrix
    dists = np.sqrt(np.sum((centroids_w[:, None, :] - centroids_w[None, :, :]) ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_dists = dists.min(axis=1)
    print(f"    Min pairwise distance: {min_dists.min():.6f} (task {min_dists.argmin()})")
    print(f"    Max pairwise distance: {min_dists.max():.6f} (task {min_dists.argmax()})")
    print(f"    Mean min distance: {min_dists.mean():.6f}")

print()
print("=" * 70)
print("DIAGNOSIS: Nguồn gốc vấn đề n/d = 0.04")
print("=" * 70)
print("""
Với n/d = 0.04 (160 samples / 4096 dims):
  - Sample covariance có eigenvalues trải rộng từ gần 0 đến rất lớn
  - Ngay cả với shrinkage=0.1, các eigenvalue nhỏ vẫn bị ảnh hưởng nặng bởi noise
  - ZCA whitening khuếch đại các hướng noise → whitened space ≈ isotropic
  - Tất cả centroids sau whitening đều gần origin → khoảng cách gần bằng nhau

Với n/d = 0.59 (2400 samples / 4096 dims):
  - Pooled covariance có eigenvalues ổn định hơn
  - ZCA whitening tạo ra whitened space với discriminative structure
  - Khoảng cách giữa các centroids khác biệt rõ ràng

ROOT CAUSE: 97% reference đạt được nhờ GLOBAL ZCA (--whiten flag) với n/d=0.59.
srt_router.py hiện tại dùng INCREMENTAL ZCA bắt đầu từ n/d=0.04 → uninformative.

FIX: Tăng shrink_factor lên 0.5-0.9 (eponymous shrinkage) hoặc dùng global ZCA fit-once.
""")
