"""
Routing accuracy and confidence metrics for SRT hypothesis testing.
"""
from typing import Dict, List, Tuple

import numpy as np


def compute_routing_accuracy(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    """
    Compute routing accuracy.

    Args:
        predictions: (N,) predicted task indices.
        ground_truth: (N,) true task indices.

    Returns:
        Fraction of correct predictions in [0, 1].
    """
    return float((predictions == ground_truth).mean())


def compute_confidence(
    dists: np.ndarray,
) -> np.ndarray:
    """
    Compute routing confidence from distance matrix.

    confidence = (d_2nd - d_1st) / d_1st

    Higher confidence → larger margin between best and second-best prediction.
    """
    sorted_dists = np.sort(dists, axis=1)
    d_first = sorted_dists[:, 0]
    d_second = sorted_dists[:, 1]
    confidence = np.where(
        d_first > 1e-8,
        (d_second - d_first) / d_first,
        0.0,
    )
    return confidence


def compute_per_task_accuracy(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    task_names: List[str],
) -> Dict[str, float]:
    """
    Compute accuracy per individual task.

    Returns:
        Dict mapping task_name -> accuracy.
    """
    results = {}
    for i, name in enumerate(task_names):
        mask = ground_truth == i
        if mask.sum() > 0:
            results[name] = float((predictions[mask] == ground_truth[mask]).mean())
        else:
            results[name] = 0.0
    return results


def compute_macro_accuracy(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    task_names: List[str],
) -> Tuple[float, Dict[str, float]]:
    """
    Compute macro accuracy (mean of per-task accuracies).

    Returns:
        (macro_accuracy, per_task_accuracies dict).
    """
    per_task = compute_per_task_accuracy(predictions, ground_truth, task_names)
    macro = float(np.mean(list(per_task.values())))
    return macro, per_task


def incremental_routing_evaluation(
    router,
    task_order: List[str],
    task_embeddings: Dict[str, np.ndarray],
    step: int,
) -> Dict:
    """
    Evaluate routing accuracy at a given CL step.

    Args:
        router: PooledMahalanobisRouter instance.
        task_order: List of task names in order.
        task_embeddings: Dict mapping task_name -> (N, d) embeddings.
        step: Current step (0-indexed). Evaluate on tasks[0..step].

    Returns:
        Dict with macro_acc, per_task_acc, n_seen, task_names.
    """
    seen_tasks = task_order[: step + 1]
    all_preds, all_gt = [], []

    for t_idx, task_name in enumerate(seen_tasks):
        embs = task_embeddings[task_name]
        preds = router.route(embs)
        gts = np.full(embs.shape[0], t_idx, dtype=np.int64)
        all_preds.append(preds)
        all_gt.append(gts)

    all_preds = np.concatenate(all_preds)
    all_gt = np.concatenate(all_gt)

    macro_acc, per_task_acc = compute_macro_accuracy(
        all_preds, all_gt, seen_tasks
    )

    return {
        "step": step + 1,
        "n_seen_tasks": len(seen_tasks),
        "macro_accuracy": macro_acc,
        "per_task_accuracy": per_task_acc,
        "total_samples": len(all_preds),
    }


def baseline_cosine_routing(
    centroids: List[np.ndarray],
    h_batch: np.ndarray,
) -> np.ndarray:
    """
    Baseline: cosine similarity to centroids.

    Args:
        centroids: List of (d,) centroid vectors.
        h_batch: (B, d) query embeddings.

    Returns:
        (B,) predicted task indices.
    """
    C = np.stack(centroids)  # (n_tasks, d)
    h_norm = h_batch / (np.linalg.norm(h_batch, axis=1, keepdims=True) + 1e-12)
    c_norm = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    sims = h_norm @ c_norm.T  # (B, n_tasks)
    return sims.argmax(axis=1).astype(np.int64)


def baseline_l2_routing(
    centroids: List[np.ndarray],
    h_batch: np.ndarray,
) -> np.ndarray:
    """
    Baseline: L2 distance to centroids.

    Args:
        centroids: List of (d,) centroid vectors.
        h_batch: (B, d) query embeddings.

    Returns:
        (B,) predicted task indices.
    """
    C = np.stack(centroids)  # (n_tasks, d)
    H_sq = np.sum(h_batch ** 2, axis=1, keepdims=True)
    C_sq = np.sum(C ** 2, axis=1)
    dists = H_sq + C_sq[None, :] - 2 * (h_batch @ C.T)
    return dists.argmin(axis=1).astype(np.int64)
