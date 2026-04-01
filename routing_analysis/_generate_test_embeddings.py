#!/usr/bin/env python3
"""
Generate synthetic embeddings for testing validate_gala.py locally.
Creates NPZ files mimicking the C1 embeddings format.

Usage:
  python3 _generate_test_embeddings.py
  # Then:
  python3 validate_gala.py --emb_dir test_embeddings/TestBackbone --benchmark Long_Sequence --device cpu
"""
import os, sys
import numpy as np

EMB_DIR = "test_embeddings/TestBackbone"
BENCHMARK = "Long_Sequence"
D = 128  # small dimension for fast testing
N_PER_TASK = 200  # samples per task

TASKS = [
    "yelp", "amazon", "mnli", "cb", "copa", "qqp", "rte",
    "imdb", "sst2", "dbpedia", "agnews", "yahoo", "multirc", "boolq", "wic",
]

# Task clusters with similar distributions (for testing domain overlap)
CLUSTER_PARAMS = {
    "sentiment": {"mean_shift": np.array([1, 0, 0, 0] + [0] * (D - 4)), "n_classes": 2},
    "NLI": {"mean_shift": np.array([0, 1, 0, 0] + [0] * (D - 4)), "n_classes": 3},
    "topic": {"mean_shift": np.array([0, 0, 1, 0] + [0] * (D - 4)), "n_classes": 4},
    "RC": {"mean_shift": np.array([0, 0, 0, 1] + [0] * (D - 4)), "n_classes": 2},
    "misc": {"mean_shift": np.array([0.5, 0.5, 0.5, 0.5] + [0] * (D - 4)), "n_classes": 2},
}

TASK_CLUSTER = {
    "yelp": "sentiment", "amazon": "sentiment", "imdb": "sentiment", "sst2": "sentiment",
    "mnli": "NLI", "cb": "NLI", "rte": "NLI",
    "dbpedia": "topic", "agnews": "topic", "yahoo": "topic",
    "multirc": "RC", "boolq": "RC",
    "copa": "misc", "qqp": "misc", "wic": "misc",
}

rng = np.random.RandomState(42)

# Generate anisotropic global covariance (mimics real backbone)
# Eigenvalues decay like 1/i^alpha, floored to avoid singularity
eigvals_global = 1.0 / np.arange(1, D + 1).astype(np.float64) ** 0.8
eigvals_global = np.maximum(eigvals_global, 1e-4)
eigvals_global = eigvals_global / eigvals_global.sum() * D  # normalize
# Random rotation
Q, _ = np.linalg.qr(rng.randn(D, D))
cov_global = Q @ np.diag(eigvals_global) @ Q.T

# Cholesky for sampling
L_global = np.linalg.cholesky(cov_global)

for task in TASKS:
    cluster = TASK_CLUSTER[task]
    params = CLUSTER_PARAMS[cluster]

    # Task-specific shift
    shift = params["mean_shift"] * (2.0 + rng.randn() * 0.5)
    # Add small task-specific perturbation
    shift += rng.randn(D) * 0.1

    n_classes = params["n_classes"]
    n_per_class = N_PER_TASK // n_classes

    embs_list = []
    labels_list = []
    for c in range(n_classes):
        # Class-specific sub-shift
        class_shift = shift + rng.randn(D) * 0.3
        # Sample from anisotropic distribution
        z = rng.randn(n_per_class, D)
        embs_c = z @ L_global.T + class_shift
        embs_list.append(embs_c)
        labels_list.extend([str(c)] * n_per_class)

    embs = np.vstack(embs_list)
    # Replace any NaN/Inf from numerical issues
    embs = np.nan_to_num(embs, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float64)
    labels = np.array(labels_list)

    # Shuffle
    perm = rng.permutation(len(labels))
    embs = embs[perm]
    labels = labels[perm]

    # Save
    out_dir = os.path.join(EMB_DIR, BENCHMARK, task)
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "train.npz"), embeddings=embs, labels=labels)
    print(f"  {task}: {embs.shape}, {n_classes} classes")

print(f"\nGenerated test embeddings in {EMB_DIR}/{BENCHMARK}/")
print(f"Run: python3 validate_gala.py --emb_dir {EMB_DIR} --benchmark {BENCHMARK} --device cpu")
