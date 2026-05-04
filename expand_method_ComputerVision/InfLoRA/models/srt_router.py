"""
SRT Router V2: Pooled Mahalanobis with ridge shrinkage.
Ported from code_srt_sgwi_v1/new_llama_gainlora/src/srt_router_v2.py.

Simpler than V1 (no ZCA whitening):
  - Pooled Mahalanobis: d(h, t) = (h - μ_t)ᵀ Σ_pool⁻¹ (h - μ_t) for ALL tasks
  - Ridge shrinkage on Σ_pool each time a new task is added
  - Welford-Hart pooled update for μ and Σ
  - Zero-rehearsal compliant: only sufficient statistics.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch


def _participation_ratio(eigvals: np.ndarray) -> float:
    """PaR = (Σλ)² / Σλ² ∈ [1, d]."""
    lam = np.maximum(eigvals, 1e-10)
    return float((lam.sum() ** 2) / (lam ** 2).sum())


def _shrink_ridge(cov: np.ndarray, n: int, d: int) -> Tuple[np.ndarray, float]:
    """Ridge (Ledoit-Wolf-like) shrinkage: δ = d/(n+d+1)."""
    delta = d / (n + d + 1e-10)
    tr_s = np.trace(cov)
    target = (tr_s / d) * np.eye(d)
    return (1 - delta) * cov + delta * target, float(delta)


_SHRINK_METHODS = {
    "ridge": _shrink_ridge,
    "none": lambda cov, n, d: (cov, 0.0),
}


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
    """Statistical signature for one task: μ_t, Σ_pool, n_t, etc."""

    def __init__(
        self,
        task_id: Union[int, str],
        mu: np.ndarray,
        Sigma: np.ndarray,
        n: int,
        shrinkage: str = "ridge",
        delta: float = 0.0,
        Sinv: Optional[np.ndarray] = None,
        par: float = 0.0,
    ):
        self.task_id = task_id
        self.mu = mu.astype(np.float64)
        self.Sigma = Sigma.astype(np.float64)
        self.n = int(n)
        self.shrinkage = shrinkage
        self.delta = float(delta)
        self.Sinv = Sinv.astype(np.float64) if Sinv is not None else None
        self.par = float(par)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "mu": self.mu,
            "Sigma": self.Sigma,
            "n": self.n,
            "shrinkage": self.shrinkage,
            "delta": self.delta,
            "Sinv": self.Sinv,
            "par": self.par,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TaskSignature":
        return cls(
            task_id=data["task_id"],
            mu=data["mu"],
            Sigma=data["Sigma"],
            n=data["n"],
            shrinkage=data.get("shrinkage", "ridge"),
            delta=data.get("delta", 0.0),
            Sinv=data.get("Sinv"),
            par=data.get("par", 0.0),
        )


class SRT_Router:
    """
    Pooled Mahalanobis Router (V2).

    Key idea:
      - Each task stores its centroid μ_t
      - ALL tasks use the SAME pooled covariance Σ_pool⁻¹ for routing
      - Routing: d(h, t) = (h - μ_t)ᵀ Σ_pool⁻¹ (h - μ_t)

    Pipeline:
      add_task(t, embeddings):
        1. μ_t = mean(embeddings), Σ_t = cov(embeddings)
        2. Welford-Hart update: μ_pool, Σ_pool += task t
        3. Ridge shrinkage on Σ_pool → Σ_pool_shrunk
        4. Eigendecomposition → Σ_pool⁻¹
        5. Store μ_t in centroids list

      route(h):
        For each sample h[i]: compute d(h[i], t) for all t using pooled Σ⁻¹
        Return argmin across tasks

    Zero-rehearsal: only μ_t, Σ_pool, n_pool stored — no raw data.
    """

    def __init__(
        self,
        embed_dim: int,
        shrinkage: str = "ridge",
        device=None,
    ):
        assert shrinkage in _SHRINK_METHODS, f"Unknown shrinkage: {shrinkage}"
        self.embed_dim = embed_dim
        self.shrinkage = shrinkage
        self.shrink_fn = _SHRINK_METHODS[shrinkage]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Per-task centroids
        self.centroids: List[np.ndarray] = []
        self.n_tasks_list: List[int] = []

        # Pooled statistics (numpy)
        self._mu_pool: Optional[np.ndarray] = None
        self._Sigma_pool: Optional[np.ndarray] = None
        self._n_pool: int = 0

        # Shrunk pooled precision (recomputed after each add_task)
        self._Sinv: Optional[np.ndarray] = None
        self._delta: float = 0.0
        self._eigvals: Optional[np.ndarray] = None

        # Task signatures by ID
        self._signatures_by_id: Dict[Union[int, str], TaskSignature] = {}

    def add_task(self, task_id: Union[int, str], embeddings: np.ndarray) -> TaskSignature:
        """
        Add task with its embeddings. Computes centroid μ_t, updates pooled stats,
        applies ridge shrinkage, and recomputes Σ_pool⁻¹.

        Args:
            task_id: integer or string task ID
            embeddings: (n_t, D) — full embeddings, NOT just the mean
        """
        n_t, d = embeddings.shape
        assert d == self.embed_dim, f"dim {d} != {self.embed_dim}"

        # ── Task centroid (mean over samples) ───────────────────────────
        mu_t = embeddings.mean(axis=0)  # (D,)
        self.centroids.append(mu_t.astype(np.float64))
        self.n_tasks_list.append(n_t)

        # ── Task covariance (for Welford update) ───────────────────────
        Sigma_t = np.cov(embeddings, rowvar=False, ddof=1)
        if np.isnan(Sigma_t).any() or np.isinf(Sigma_t).any():
            Sigma_t = np.eye(d, dtype=np.float64)

        # ── Welford-Hart pooled update ────────────────────────────────
        if self._n_pool == 0:
            self._mu_pool = mu_t.copy()
            self._Sigma_pool = Sigma_t.copy()
            self._n_pool = n_t
        else:
            self._mu_pool, self._Sigma_pool, self._n_pool = welford_pooled_update(
                self._mu_pool, self._Sigma_pool, self._n_pool,
                mu_t, Sigma_t, n_t,
            )

        # ── Ridge shrinkage on pooled covariance ───────────────────────
        Sigma_shrunk, self._delta = self.shrink_fn(self._Sigma_pool, self._n_pool, d)

        # ── Eigendecomposition → precision matrix Σ⁻¹ ─────────────────
        eigvals, eigvecs = np.linalg.eigh(Sigma_shrunk)
        eigvals = np.maximum(eigvals, 1e-6 * np.abs(eigvals).max())
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        self._Sinv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
        self._eigvals = eigvals

        # ── Compute PaR ───────────────────────────────────────────────
        par = _participation_ratio(eigvals)

        sig = TaskSignature(
            task_id=task_id,
            mu=mu_t.astype(np.float64),
            Sigma=self._Sigma_pool.copy().astype(np.float64),
            n=n_t,
            shrinkage=self.shrinkage,
            delta=self._delta,
            Sinv=self._Sinv.copy().astype(np.float64),
            par=par,
        )
        self._signatures_by_id[task_id] = sig
        return sig

    def route(self, h: np.ndarray) -> np.ndarray:
        """
        Route embeddings to nearest task via Pooled Mahalanobis.

        d(h, t) = (h - μ_t)ᵀ Σ_pool⁻¹ (h - μ_t)  for ALL tasks t.

        Args:
            h: (n, D) or (D,) — mean-pooled frozen backbone embeddings

        Returns:
            task_indices: (n,) — predicted task ID for each embedding
        """
        if h.ndim == 1:
            h = h.reshape(1, -1)

        n_samples, d = h.shape
        if d != self.embed_dim:
            raise ValueError(f"Embedding dim {d} != {self.embed_dim}")
        if self._Sinv is None:
            raise RuntimeError("SRT Router: no tasks registered or Σ⁻¹ not computed.")

        T = len(self.centroids)

        # Compute Mahalanobis distance to each centroid using pooled Σ⁻¹
        # d_t = ||Σ⁻¹/² (h - μ_t)||² = (h - μ_t)ᵀ Σ⁻¹ (h - μ_t)
        dists = np.zeros((n_samples, T), dtype=np.float64)
        Sinv = self._Sinv  # (D, D)

        for idx, mu_t in enumerate(self.centroids):
            diff = h - mu_t  # (n, D)
            # (n, D) @ (D, D) @ (D, n) = (n, n) — too slow
            # Instead: (n, D) @ (D, D) → (n, D), then element-wise mul and sum
            diff_Sinv = diff @ Sinv  # (n, D)
            dists[:, idx] = (diff * diff_Sinv).sum(axis=1)  # (n,)

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
            print(f"[SRT-ROUTE] n_tasks={T}, sample_0 dists={dists[0].tolist()}, "
                  f"argmin={nearest_idx[0]}, pred={nearest_task[0]}, shrink_delta={self._delta:.4f}")

        return nearest_task

    def route_debug(self, h: np.ndarray) -> dict:
        """Route with debug info: distances, confidence ratio, etc."""
        if h.ndim == 1:
            h = h.reshape(1, -1)

        n_samples, d = h.shape
        T = len(self.centroids)

        dists = np.zeros((n_samples, T), dtype=np.float64)
        Sinv = self._Sinv
        for idx, mu_t in enumerate(self.centroids):
            diff = h - mu_t
            diff_Sinv = diff @ Sinv
            dists[:, idx] = (diff * diff_Sinv).sum(axis=1)

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

    @property
    def _Sigma_pool_np(self) -> Optional[np.ndarray]:
        return self._Sigma_pool

    def summary(self) -> dict:
        task_ids = list(self._signatures_by_id.keys())
        par = _participation_ratio(self._eigvals) if self._eigvals is not None else 0.0
        return {
            "n_tasks": len(task_ids),
            "task_ids": task_ids,
            "metrics": [self.shrinkage] * len(task_ids),
            "avg_par": par,
            "shrinkage": self.shrinkage,
            "delta": self._delta,
            "n_pool": self._n_pool,
            "par": par,
            "par_d_ratio": par / self.embed_dim if self._n_pool > 0 else 0.0,
        }

    # ── Save / Load ───────────────────────────────────────────────────

    def save(self, path: str):
        """Save all signatures to disk."""
        sigs_data = {k: v.to_dict() for k, v in self._signatures_by_id.items()}
        np.savez_compressed(
            path,
            signatures=sigs_data,
            mu_pool=self._mu_pool if self._mu_pool is not None else np.array([]),
            Sigma_pool=self._Sigma_pool if self._Sigma_pool is not None else np.array([]),
            Sinv=self._Sinv if self._Sinv is not None else np.array([]),
            n_pool=np.array([self._n_pool]),
            shrinkage=self.shrinkage,
            delta=np.array([self._delta]),
            task_ids=list(self._signatures_by_id.keys()),
            eigvals=self._eigvals if self._eigvals is not None else np.array([]),
        )

    def load(self, path: str):
        """Load signatures from disk."""
        data = np.load(path, allow_pickle=True)
        self.shrinkage = str(data.get("shrinkage", "ridge"))
        self.shrink_fn = _SHRINK_METHODS.get(self.shrinkage, _shrink_ridge)
        self._delta = float(data.get("delta", [0.0])[0])
        self._n_pool = int(data.get("n_pool", [0])[0])

        mu_p = data["mu_pool"]
        self._mu_pool = mu_p if mu_p.size > 0 else None
        sigma_p = data["Sigma_pool"]
        self._Sigma_pool = sigma_p if sigma_p.size > 0 else None
        sinv_p = data.get("Sinv", np.array([]))
        self._Sinv = sinv_p if sinv_p.size > 0 else None
        self._eigvals = data.get("eigvals", np.array([]))
        if self._eigvals.size == 0:
            self._eigvals = None

        sigs_dict = data["signatures"].item()
        self._signatures_by_id = {}
        self.centroids = []
        self.n_tasks_list = []
        for task_id, sig_dict in sigs_dict.items():
            sig = TaskSignature.from_dict(sig_dict)
            self._signatures_by_id[task_id] = sig
            self.centroids.append(sig.mu)
            self.n_tasks_list.append(sig.n)
