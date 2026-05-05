"""
SRT Router V1: ZCA Whitening + L2 Distance.

Based on "Whitening Sentence Representations" (Huang et al., ACL 2021).

Pipeline:
  add_task(t, embeddings):
    1. μ_t = mean(embeddings), Σ_t = cov(embeddings)
    2. Welford-Hart pooled update → μ_pool, Σ_pool
    3. Eigendecomposition: Σ_pool = V Λ V^T
    4. ZCA Whitening: W_zca = V @ Λ^{-1/2} @ V^T
    5. μ_global = mean of all task centroids
    6. Whitened centroids: μ_t^w = (μ_t - μ_global) @ W_zca.T
    7. Store μ_t, Σ_t, μ_t^w

  route(h):
    h_w = (h - μ_global) @ W_zca.T
    d_t = ||h_w - μ_t^w||_2  (L2 distance to each whitened centroid)
    return argmin_t d_t

This matches the user's spec:
  - Task Signature: μ_t + Σ_t from CLS token embeddings
  - SGWI: Mahalanobis distance between centroids (use Σ_pool⁻¹)
  - Inference: ZCA Whitening → L2 distance → hard route
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch


def _participation_ratio(eigvals: np.ndarray) -> float:
    """PaR = (Σλ)² / Σλ² ∈ [1, d]."""
    lam = np.maximum(eigvals, 1e-10)
    return float((lam.sum() ** 2) / (lam ** 2).sum())


def welford_pooled_update(
    mu_old: np.ndarray,
    cov_old: np.ndarray,
    n_old: int,
    mu_new: np.ndarray,
    cov_new: np.ndarray,
    n_new: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Welford-Hart compact update for pooled mean and covariance."""
    total = n_old + n_new
    if total <= 1:
        return mu_old.copy(), cov_old.copy(), n_old

    mu_pool = (n_old * mu_old + n_new * mu_new) / total
    delta = mu_new - mu_old
    cross = (n_old * n_new / total) * np.outer(delta, delta)
    cov_pool = (
        (n_old - 1) * cov_old
        + (n_new - 1) * cov_new
        + cross
    ) / max(total - 1, 1)
    return mu_pool, cov_pool, total


class TaskSignature:
    """Statistical signature for one task: μ_t, Σ_t, whitened μ_t^w."""

    def __init__(
        self,
        task_id: Union[int, str],
        mu: np.ndarray,
        Sigma: np.ndarray,
        n: int,
        mu_whitened: Optional[np.ndarray] = None,
        par: float = 0.0,
    ):
        self.task_id = task_id
        self.mu = mu.astype(np.float64)        # (D,) raw centroid
        self.Sigma = Sigma.astype(np.float64)  # (D, D) task covariance
        self.n = int(n)
        self.mu_whitened = mu_whitened.astype(np.float64) if mu_whitened is not None else None  # (D,) ZCA-whitened centroid
        self.par = float(par)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "mu": self.mu,
            "Sigma": self.Sigma,
            "n": self.n,
            "mu_whitened": self.mu_whitened,
            "par": self.par,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TaskSignature":
        return cls(
            task_id=data["task_id"],
            mu=data["mu"],
            Sigma=data["Sigma"],
            n=data["n"],
            mu_whitened=data.get("mu_whitened"),
            par=data.get("par", 0.0),
        )


class SRT_Router:
    """
    ZCA Whitening Router (V1).

    Key idea:
      - Each task stores its raw centroid μ_t and covariance Σ_t
      - ALL tasks share the same ZCA whitening matrix W_zca (computed from pooled Σ)
      - Routing: whitened L2 distance d(h, t) = ||h_w - μ_t^w||₂

    Pipeline:
      add_task(t, embeddings):
        1. μ_t = mean(embeddings), Σ_t = cov(embeddings)
        2. Welford-Hart update: μ_pool, Σ_pool += task t
        3. Eigendecomposition of Σ_pool → W_zca
        4. μ_global = mean(μ_s for all s)
        5. μ_t^w = (μ_t - μ_global) @ W_zca.T
        6. Store μ_t in centroids list

      route(h):
        h_w = (h - μ_global) @ W_zca.T
        d_t = ||h_w - μ_t^w||₂ for all tasks t
        Return argmin across tasks

    Zero-rehearsal: only μ_t, Σ_t, W_zca, μ_global stored — no raw data.
    """

    def __init__(
        self,
        embed_dim: int,
        device=None,
    ):
        self.embed_dim = embed_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Per-task centroids (raw)
        self.centroids: List[np.ndarray] = []
        self.n_tasks_list: List[int] = []

        # Pooled statistics (numpy)
        self._mu_pool: Optional[np.ndarray] = None
        self._Sigma_pool: Optional[np.ndarray] = None
        self._n_pool: int = 0

        # ZCA whitening matrix (recomputed after each add_task)
        self._W_zca: Optional[np.ndarray] = None    # (D, D)
        self._eigvals: Optional[np.ndarray] = None
        self._eigvecs: Optional[np.ndarray] = None

        # Global centroid (computed from all task centroids)
        self._mu_global: Optional[np.ndarray] = None  # (D,)

        # Task signatures by ID
        self._signatures_by_id: Dict[Union[int, str], TaskSignature] = {}

    def _compute_zca_from_pooled(self):
        """
        Compute ZCA whitening matrix from current pooled covariance.
        W_zca = V @ Λ^{-1/2} @ V^T
        """
        if self._Sigma_pool is None:
            return

        d = self.embed_dim
        eigvals, eigvecs = np.linalg.eigh(self._Sigma_pool)
        eigvals = np.maximum(eigvals, 1e-8 * np.abs(eigvals).max())
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # ZCA: W_zca = V @ Λ^{-1/2} @ V^T
        self._eigvals = eigvals
        self._eigvecs = eigvecs
        self._W_zca = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    def _compute_global_centroid(self):
        """Compute mean of all task centroids."""
        if not self.centroids:
            self._mu_global = None
            return
        self._mu_global = np.mean(self.centroids, axis=0)

    def add_task(self, task_id: Union[int, str], embeddings: np.ndarray) -> TaskSignature:
        """
        Add task with its CLS token embeddings. Computes centroid μ_t, covariance Σ_t,
        updates pooled stats, computes ZCA whitening matrix, and stores whitened centroid.

        Args:
            task_id: integer or string task ID
            embeddings: (n_t, D) — CLS token embeddings, NOT mean-pooled
        """
        n_t, d = embeddings.shape
        assert d == self.embed_dim, f"dim {d} != {self.embed_dim}"

        # ── Task centroid and covariance ──────────────────────────────
        mu_t = embeddings.mean(axis=0)  # (D,)
        self.centroids.append(mu_t.astype(np.float64))
        self.n_tasks_list.append(n_t)

        Sigma_t = np.cov(embeddings, rowvar=False, ddof=1)
        if np.isnan(Sigma_t).any() or np.isinf(Sigma_t).any():
            Sigma_t = np.eye(d, dtype=np.float64)

        # ── Welford-Hart pooled update ───────────────────────────────
        if self._n_pool == 0:
            self._mu_pool = mu_t.copy()
            self._Sigma_pool = Sigma_t.copy()
            self._n_pool = n_t
        else:
            self._mu_pool, self._Sigma_pool, self._n_pool = welford_pooled_update(
                self._mu_pool, self._Sigma_pool, self._n_pool,
                mu_t, Sigma_t, n_t,
            )

        # ── Eigendecomposition → ZCA whitening matrix ──────────────────
        self._compute_zca_from_pooled()

        # ── Compute global centroid ───────────────────────────────────
        self._compute_global_centroid()

        # ── Compute whitened centroid for this task ──────────────────
        if self._W_zca is not None and self._mu_global is not None:
            mu_w = (mu_t - self._mu_global) @ self._W_zca.T
        else:
            mu_w = None

        # ── Compute PaR ──────────────────────────────────────────────
        par = _participation_ratio(self._eigvals) if self._eigvals is not None else 0.0

        sig = TaskSignature(
            task_id=task_id,
            mu=mu_t.astype(np.float64),
            Sigma=Sigma_t.astype(np.float64),
            n=n_t,
            mu_whitened=mu_w,
            par=par,
        )
        self._signatures_by_id[task_id] = sig
        return sig

    def route(self, h: np.ndarray) -> np.ndarray:
        """
        Route embeddings to nearest task via ZCA Whitened L2 Distance.

        d(h, t) = ||h_w - μ_t^w||_2

        where:
          h_w = (h - μ_global) @ W_zca.T
          μ_t^w = (μ_t - μ_global) @ W_zca.T

        Args:
            h: (n, D) or (D,) — CLS token embeddings from frozen ViT

        Returns:
            task_indices: (n,) — predicted task ID for each embedding
        """
        if h.ndim == 1:
            h = h.reshape(1, -1)

        n_samples, d = h.shape
        if d != self.embed_dim:
            raise ValueError(f"Embedding dim {d} != {self.embed_dim}")
        if self._W_zca is None or self._mu_global is None:
            raise RuntimeError("SRT Router: no tasks registered or ZCA matrix not computed.")

        T = len(self.centroids)
        if T == 0:
            raise RuntimeError("SRT Router: no tasks registered.")

        # ── ZCA whitening of input: h_w = (h - μ_global) @ W_zca.T ───
        h_centered = h - self._mu_global  # (n, D)
        h_w = h_centered @ self._W_zca.T  # (n, D)

        # ── L2 distance to each whitened centroid ────────────────────
        dists = np.zeros((n_samples, T), dtype=np.float64)
        for idx, sig in enumerate(self._signatures_by_id.values()):
            if sig.mu_whitened is None:
                # Recompute if not stored
                mu_w = (sig.mu - self._mu_global) @ self._W_zca.T
            else:
                mu_w = sig.mu_whitened
            diff = h_w - mu_w  # (n, D)
            dists[:, idx] = np.linalg.norm(diff, axis=1)  # (n,)

        # argmin across tasks
        nearest_idx = np.argmin(dists, axis=1)

        # Map to task IDs
        task_ids_ordered = list(self._signatures_by_id.keys())
        nearest_task = np.array([task_ids_ordered[i] for i in nearest_idx])

        # Debug output every 1000 calls
        if not hasattr(self, "_route_debug_count"):
            self._route_debug_count = 0
        self._route_debug_count += 1
        if self._route_debug_count % 1000 == 0 and n_samples > 0:
            print(f"[SRT-ROUTE-ZCA] n_tasks={T}, h_w[0][:5]={h_w[0][:5].tolist()}, "
                  f"dists[0]={dists[0].tolist()}, argmin={nearest_idx[0]}, "
                  f"pred={nearest_task[0]}, mu_global[:3]={self._mu_global[:3].tolist()}")

        return nearest_task

    def route_debug(self, h: np.ndarray) -> dict:
        """Route with debug info: distances, confidence ratio, etc."""
        if h.ndim == 1:
            h = h.reshape(1, -1)

        n_samples, d = h.shape
        T = len(self.centroids)

        # ZCA whitening
        h_centered = h - self._mu_global
        h_w = h_centered @ self._W_zca.T

        dists = np.zeros((n_samples, T), dtype=np.float64)
        for idx, sig in enumerate(self._signatures_by_id.values()):
            mu_w = sig.mu_whitened if sig.mu_whitened is not None else (
                (sig.mu - self._mu_global) @ self._W_zca.T
            )
            diff = h_w - mu_w
            dists[:, idx] = np.linalg.norm(diff, axis=1)

        sorted_idx = np.argsort(dists, axis=1)
        task_ids_ordered = list(self._signatures_by_id.keys())
        best_idx = sorted_idx[:, 0]
        best_d = dists[np.arange(n_samples), best_idx]

        if T >= 2:
            second_idx = sorted_idx[:, 1]
            second_d = dists[np.arange(n_samples), second_idx]
            second_task = np.array([task_ids_ordered[i] for i in second_idx])
            with np.errstate(divide="ignore", invalid="ignore"):
                conf = (second_d - best_d) / (best_d + 1e-10)
                conf = np.where(np.isfinite(conf), conf, 999.0)
        else:
            second_d = np.full(n_samples, np.inf)
            second_task = np.full(n_samples, -1, dtype=object)
            conf = np.full(n_samples, 999.0)

        return {
            "task_ids": np.array([task_ids_ordered[i] for i in best_idx]),
            "dists": dists,
            "best_dist": best_d,
            "second_task": second_task,
            "second_dist": second_d,
            "confidence_ratio": conf,
            "n_tasks": T,
            "task_id_list": task_ids_ordered,
        }

    @property
    def n_tasks(self) -> int:
        return len(self.centroids)

    @property
    def signatures(self) -> Dict[Union[int, str], TaskSignature]:
        return self._signatures_by_id

    def summary(self) -> dict:
        task_ids = list(self._signatures_by_id.keys())
        par = _participation_ratio(self._eigvals) if self._eigvals is not None else 0.0
        return {
            "n_tasks": len(task_ids),
            "task_ids": task_ids,
            "avg_par": par,
            "n_pool": self._n_pool,
            "par": par,
            "par_d_ratio": par / self.embed_dim if self._n_pool > 0 else 0.0,
        }

    # ── SGWI helper: Mahalanobis distance for SGWI weights ─────────

    def mahalanobis_distance(self, mu_a: np.ndarray, mu_b: np.ndarray) -> float:
        """
        Compute Mahalanobis distance between two centroids using pooled Σ⁻¹.
        Used by SGWI to compute blend weights.
        d(a,b) = (μ_a - μ_b)ᵀ Σ_pool⁻¹ (μ_a - μ_b)

        Returns scalar Mahalanobis distance.
        """
        if self._W_zca is None:
            return float(np.linalg.norm(mu_a - mu_b))
        # Σ⁻¹ = W_zca @ W_zca.T
        Sinv = self._W_zca @ self._W_zca.T
        diff = mu_a - mu_b
        return float((diff @ Sinv @ diff))

    def mahalanobis_distance_to_all(self, mu: np.ndarray) -> Dict[Union[int, str], float]:
        """Compute Mahalanobis distance from mu to all registered task centroids."""
        dists = {}
        for task_id, sig in self._signatures_by_id.items():
            dists[task_id] = self.mahalanobis_distance(mu, sig.mu)
        return dists

    # ── Save / Load ───────────────────────────────────────────────────

    def save(self, path: str):
        """Save all signatures to disk."""
        sigs_data = {k: v.to_dict() for k, v in self._signatures_by_id.items()}
        np.savez_compressed(
            path,
            signatures=sigs_data,
            mu_pool=self._mu_pool if self._mu_pool is not None else np.array([]),
            Sigma_pool=self._Sigma_pool if self._Sigma_pool is not None else np.array([]),
            W_zca=self._W_zca if self._W_zca is not None else np.array([]),
            mu_global=self._mu_global if self._mu_global is not None else np.array([]),
            n_pool=np.array([self._n_pool]),
            eigvals=self._eigvals if self._eigvals is not None else np.array([]),
            eigvecs=self._eigvecs if self._eigvecs is not None else np.array([]),
            task_ids=list(self._signatures_by_id.keys()),
        )

    def load(self, path: str):
        """Load signatures from disk."""
        data = np.load(path, allow_pickle=True)

        mu_p = data["mu_pool"]
        self._mu_pool = mu_p if mu_p.size > 0 else None
        sigma_p = data["Sigma_pool"]
        self._Sigma_pool = sigma_p if sigma_p.size > 0 else None

        wzca_p = data.get("W_zca", np.array([]))
        self._W_zca = wzca_p if wzca_p.size > 0 else None

        mu_g = data.get("mu_global", np.array([]))
        self._mu_global = mu_g if mu_g.size > 0 else None

        self._n_pool = int(data.get("n_pool", [0])[0])
        self._eigvals = data.get("eigvals", np.array([]))
        if self._eigvals.size == 0:
            self._eigvals = None
        self._eigvecs = data.get("eigvecs", np.array([]))
        if self._eigvecs.size == 0:
            self._eigvecs = None

        sigs_dict = data["signatures"].item()
        self._signatures_by_id = {}
        self.centroids = []
        self.n_tasks_list = []
        for task_id, sig_dict in sigs_dict.items():
            sig = TaskSignature.from_dict(sig_dict)
            self._signatures_by_id[task_id] = sig
            self.centroids.append(sig.mu)
            self.n_tasks_list.append(sig.n)