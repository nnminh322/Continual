"""
SRT Router: Statistical Routing Theory with Pooled Mahalanobis Distance.

Core principle (SRT Theorem 4):
  d_t(h) = (h - μ_t)ᵀ Σ_pool⁻¹ (h - μ_t)

  PooledMahalanobis is Bayes-optimal for discriminating between tasks whose
  embeddings follow shared-covariance Gaussians.

Shrinkage (critical for n/d << 1):
  Ridge: δ* = d / (n + d) — analytical optimal for n≈d regime in CL.
  No data-dependent estimation needed. No ZCA. No buffer waiting.

Zero-rehearsal compliant: only sufficient statistics (Σ_pool, μ_pool, n_pool).
Zero-drift: no learnable parameters.

GPU-accelerated via torch.linalg.eigh on CUDA.
"""

from __future__ import annotations

import math
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union


# ─────────────────────────────────────────────────────────────────────────────
#  CUDA SUPPORT
# ─────────────────────────────────────────────────────────────────────────────

HAS_CUDA = torch.cuda.is_available()
DEVICE_DEFAULT = "cuda" if HAS_CUDA else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
#  SHRINKAGE FUNCTIONS  (analytical, no CV needed)
# ─────────────────────────────────────────────────────────────────────────────

def _shrink_ridge(cov: torch.Tensor, n: int, d: int) -> Tuple[torch.Tensor, float]:
    """
    Analytical Ridge shrinkage: δ* = d / (n + d).

    Best for the n≈d regime in continual learning. Derived from
    Marchenko-Pastur + Claude Opus 4.6 MSE under Frobenius norm.

    δ* converges to:
      n/d → 0  → δ* → 1  (dominantly regularized → near isotropic)
      n/d → 1  → δ* → 0.5
      n/d → ∞  → δ* → 0  (pure sample covariance)
    """
    delta = d / (n + d + 1e-10)
    tr_S = torch.trace(cov).item()
    target = (tr_S / d) * torch.eye(d, device=cov.device, dtype=cov.dtype)
    return (1 - delta) * cov + delta * target, float(delta)


def _shrink_oas(cov: torch.Tensor, n: int, d: int) -> Tuple[torch.Tensor, float]:
    """Oracle-Approximating Shrinkage (Chen et al., 2010). Closed-form."""
    tr_S = torch.trace(cov).item()
    tr_S2 = (cov ** 2).sum().item()
    rho_hat = tr_S2 / (tr_S ** 2 + 1e-10)
    delta = min(1.0, max(0.0, rho_hat / (n + 1 - 2 / d)))
    target = (tr_S / d) * torch.eye(d, device=cov.device, dtype=cov.dtype)
    return (1 - delta) * cov + delta * target, float(delta)


def _shrink_lw(cov: torch.Tensor, n: int, d: int) -> Tuple[torch.Tensor, float]:
    """Ledoit-Wolf Oracle Linear Shrinkage (2004). Analytical δ."""
    tr_S = torch.trace(cov).item()
    sum_sq = (cov ** 2).sum().item()
    diag_sq = (torch.diag(cov) ** 2).sum().item()
    denom = n * (diag_sq - (tr_S / d) ** 2)
    if abs(denom) < 1e-10:
        delta = 1.0
    else:
        numerator = sum_sq - tr_S ** 2 / d
        delta = max(0.0, min(1.0, numerator / denom))
    target = (tr_S / d) * torch.eye(d, device=cov.device, dtype=cov.dtype)
    return (1 - delta) * cov + delta * target, float(delta)


def _shrink_none(cov: torch.Tensor, n: int, d: int) -> Tuple[torch.Tensor, float]:
    """No shrinkage (raw sample covariance)."""
    return cov, 0.0


_SHRINK_METHODS = {
    "ridge": _shrink_ridge,
    "oas":   _shrink_oas,
    "lw":    _shrink_lw,
    "none":  _shrink_none,
}


# ─────────────────────────────────────────────────────────────────────────────
#  WELFORD-HART INCREMENTAL POOLED UPDATE
# ─────────────────────────────────────────────────────────────────────────────

def welford_pooled_update(
    mu_old: torch.Tensor,
    cov_old: torch.Tensor,
    n_old: int,
    mu_new: torch.Tensor,
    cov_new: torch.Tensor,
    n_new: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Welford-Hart compact update for pooled mean and covariance.

    Maintains running Σ_pool across tasks without storing raw data.
    Zero-rehearsal compliant.
    """
    total = n_old + n_new
    if total <= 1:
        return mu_old, cov_old, n_old

    # Align all tensors to mu_new's device before arithmetic
    target_device = mu_new.device
    if mu_old.device != target_device:
        mu_old = mu_old.to(target_device)
        cov_old = cov_old.to(target_device)
    if cov_new.device != target_device:
        cov_new = cov_new.to(target_device)

    mu_pool = (n_old * mu_old + n_new * mu_new) / total
    delta = mu_new - mu_old
    C = (n_old * n_new / total) * torch.outer(delta, delta)
    cov_pool = (
        (n_old - 1) * cov_old
        + (n_new - 1) * cov_new
        + C
    ) / max(total - 1, 1)

    return mu_pool, cov_pool, total


# ─────────────────────────────────────────────────────────────────────────────
#  POOLED MAHALANOBIS ROUTER
# ─────────────────────────────────────────────────────────────────────────────

class PooledMahalanobisRouter:
    """
    SRT Theorem 4: Pooled Mahalanobis Distance with analytical shrinkage.

    Routing metric:
      d_t(h) = (h - μ_t)ᵀ Σ_pool⁻¹ (h - μ_t)

    Pipeline per task:
      1. Extract embeddings h ~ N(μ_t, Σ_t) from frozen backbone
      2. Compute sufficient stats: μ_t, Σ_t, n_t
      3. Welford-Hart update: Σ_pool ← merge(Σ_pool, Σ_t)
      4. Ridge shrink Σ_pool: Σ_shrunk = (1-δ*)Σ_pool + δ*·(tr/ d)·I
      5. Compute Σ_shrunk⁻¹ via torch.linalg.eigh on GPU
      6. Store {μ_t, Σ_shrunk⁻¹} as task signature

    Inference:
      For each test embedding h_test:
        For each task t: d_t = (h_test - μ_t)ᵀ Σ_pool⁻¹ (h_test - μ_t)
        Return argmin d_t

    Args:
        shrinkage: 'ridge' | 'oas' | 'lw' | 'none' (default: 'ridge')
        device: 'cuda' | 'cpu' (default: auto-detect)
    """

    def __init__(
        self,
        shrinkage: str = "ridge",
        device: Optional[str] = None,
    ):
        assert shrinkage in _SHRINK_METHODS, f"Unknown shrinkage: {shrinkage}"
        self.shrinkage = shrinkage
        self.shrink_fn = _SHRINK_METHODS[shrinkage]
        self.device = device or DEVICE_DEFAULT
        self.dev = torch.device(self.device)

        # Per-task centroids (numpy, for routing)
        self.centroids: List[np.ndarray] = []
        self.n_tasks_list: List[int] = []
        # Signatures keyed by task_id
        self._signatures_by_id: Dict[Union[int, str], TaskSignature] = {}

        # Pooled sufficient statistics (torch, GPU when available)
        self._mu_pool_t: Optional[torch.Tensor] = None
        self._Sigma_pool_t: Optional[torch.Tensor] = None
        self._n_pool: int = 0

        # Shrunk inverse of Σ_pool (torch, GPU) — recomputed after each task
        self._Sigma_inv_t: Optional[torch.Tensor] = None
        self._delta: float = 0.0

        # Cached eigenvalues for summary
        self._eigvals: Optional[np.ndarray] = None

    # ── Add task ────────────────────────────────────────────────────────────

    def add_task(
        self,
        task_id: Union[int, str],
        h_train: np.ndarray,
    ) -> TaskSignature:
        """
        Register a new task's statistical signature.

        Args:
            task_id: task identifier (string or int)
            h_train: (n_t, d) embeddings from frozen backbone (numpy)

        Returns:
            TaskSignature for the added task
        """
        n_t, d = h_train.shape

        # Move to GPU for all computations
        X = torch.from_numpy(h_train.astype(np.float32)).to(self.dev)
        mu_t_t = X.mean(dim=0)
        Xc = X - mu_t_t
        Sigma_t_t = (Xc.T @ Xc) / max(n_t - 1, 1)
        del X, Xc
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # Store centroid (numpy, CPU)
        mu_t_np = mu_t_t.cpu().numpy()
        self.centroids.append(mu_t_np)
        self.n_tasks_list.append(n_t)

        # ── Welford-Hart pooled update ────────────────────────────────────
        if self._n_pool == 0:
            self._mu_pool_t = mu_t_t.clone()
            self._Sigma_pool_t = Sigma_t_t.clone()
            self._n_pool = n_t
        else:
            self._mu_pool_t, self._Sigma_pool_t, self._n_pool = welford_pooled_update(
                self._mu_pool_t,
                self._Sigma_pool_t,
                self._n_pool,
                mu_t_t,
                Sigma_t_t,
                n_t,
            )

        del mu_t_t, Sigma_t_t

        # ── Ridge shrinkage ───────────────────────────────────────────────
        Sigma_shrunk_t, self._delta = self.shrink_fn(
            self._Sigma_pool_t, self._n_pool, d
        )

        # ── Eigendecomposition → inverse ───────────────────────────────────
        eigvals, eigvecs = torch.linalg.eigh(Sigma_shrunk_t)
        eigvals = torch.clamp(
            eigvals,
            min=1e-6 * eigvals.abs().max().item()
        )
        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        self._Sigma_inv_t = eigvecs @ torch.diag(1.0 / eigvals) @ eigvecs.T

        # Cache eigenvalues (for PaR)
        self._eigvals = eigvals.cpu().numpy()

        sig = TaskSignature(
            task_id=task_id,
            mu=mu_t_np,
            Sigma=self._Sigma_pool_t.cpu().numpy(),
            n=n_t,
            shrinkage=self.shrinkage,
            delta=self._delta,
            Sinv=self._Sigma_inv_t.cpu().numpy(),
            par=self._participation_ratio(),
        )
        self._signatures_by_id[task_id] = sig
        return sig

    def _participation_ratio(self) -> float:
        """PaR = (Σλ)² / Σλ² ∈ [1, d]. Indicates effective dimensionality."""
        if self._eigvals is None:
            return 0.0
        lam = np.maximum(self._eigvals, 1e-10)
        return float((lam.sum() ** 2) / (lam ** 2).sum())

    # ── Route ─────────────────────────────────────────────────────────────

    def route(self, h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Route embeddings h to the nearest stored task.

        Args:
            h: (n, d) or (d,) — embeddings from frozen backbone

        Returns:
            (task_ids, dists): predicted task ID and distance array for each sample
        """
        if h.ndim == 1:
            h = h.reshape(1, -1)

        n_sample, d_h = h.shape
        T = len(self.centroids)

        if T == 0:
            raise RuntimeError("PooledMahalanobisRouter: no tasks registered.")

        # Move to GPU
        H = torch.from_numpy(h.astype(np.float32)).to(self.dev)
        Sinv = self._Sigma_inv_t
        dists = np.zeros((n_sample, T), dtype=np.float64)

        for i, mu_t_np in enumerate(self.centroids):
            mu_t_t = torch.from_numpy(mu_t_np.astype(np.float32)).to(self.dev)
            diff = H - mu_t_t
            diff_Sinv = diff @ Sinv
            dists[:, i] = (diff * diff_Sinv).sum(dim=1).cpu().numpy()

        del H
        nearest_idx = np.argmin(dists, axis=1)
        task_ids_ordered = list(self._signatures_by_id.keys())
        nearest_task = np.array([task_ids_ordered[i] for i in nearest_idx])
        return nearest_task, dists

    def route_debug(self, h: np.ndarray) -> dict:
        """
        Route with full debug info: all distances, top-2 predictions, confidence.

        Returns:
            dict with keys: task_ids, dists, nearest_task, confidence_ratio,
                            second_task, n_tasks, task_id_list
        """
        if h.ndim == 1:
            h = h.reshape(1, -1)

        n_sample, d_h = h.shape
        T = len(self.centroids)

        if T == 0:
            raise RuntimeError("PooledMahalanobisRouter: no tasks registered.")

        # Move to GPU
        H = torch.from_numpy(h.astype(np.float32)).to(self.dev)
        Sinv = self._Sigma_inv_t
        dists = np.zeros((n_sample, T), dtype=np.float64)

        for i, mu_t_np in enumerate(self.centroids):
            mu_t_t = torch.from_numpy(mu_t_np.astype(np.float32)).to(self.dev)
            diff = H - mu_t_t
            diff_Sinv = diff @ Sinv
            dists[:, i] = (diff * diff_Sinv).sum(dim=1).cpu().numpy()

        del H
        nearest_idx = np.argmin(dists, axis=1)
        task_ids_ordered = list(self._signatures_by_id.keys())
        nearest_task = np.array([task_ids_ordered[i] for i in nearest_idx])

        # Confidence: ratio of 2nd-best distance to best distance
        sorted_idx = np.argsort(dists, axis=1)
        best_idx = sorted_idx[:, 0]
        best_d = dists[np.arange(n_sample), best_idx]

        if T >= 2:
            second_idx = sorted_idx[:, 1]
            second_d = dists[np.arange(n_sample), second_idx]
            second_task = np.array([task_ids_ordered[i] for i in second_idx])
            with np.errstate(divide='ignore', invalid='ignore'):
                conf = (second_d - best_d) / (best_d + 1e-10)
                conf = np.where(np.isfinite(conf), conf, 999.0)
        else:
            # Only 1 task: no second-best exists
            second_d = np.full_like(best_d, np.inf)
            second_task = np.full(n_sample, -1, dtype=nearest_task.dtype)
            conf = np.full(n_sample, 999.0)

        return {
            "task_ids": nearest_task,
            "dists": dists,
            "nearest_task": nearest_task,
            "second_task": second_task,
            "best_dist": best_d,
            "second_dist": second_d,
            "confidence_ratio": conf,
            "n_tasks": T,
            "task_id_list": task_ids_ordered,
        }

    @property
    def signatures(self) -> Dict[Union[int, str], TaskSignature]:
        """Dict of all task signatures, keyed by task_id."""
        return self._signatures_by_id

    def summary(self) -> dict:
        """Return routing statistics."""
        task_ids = list(self._signatures_by_id.keys())
        return {
            "n_tasks": len(task_ids),
            "task_ids": task_ids,
            "metrics": [self.shrinkage] * len(task_ids),
            "avg_par": self._participation_ratio(),
            "shrinkage": self.shrinkage,
            "delta": self._delta,
            "n_pool": self._n_pool,
            "par": self._participation_ratio(),
            "par_d_ratio": self._participation_ratio() / 4096 if self._n_pool > 0 else 0.0,
        }

    def save(self, path: str):
        """Save router state to disk (zero-rehearsal compliant)."""
        sigs_data = {k: v.to_dict() for k, v in self._signatures_by_id.items()}

        Sigma_pool_np = (
            self._Sigma_pool_t.cpu().numpy()
            if self._Sigma_pool_t is not None
            else np.array([])
        )
        mu_pool_np = (
            self._mu_pool_t.cpu().numpy()
            if self._mu_pool_t is not None
            else np.array([])
        )
        Sinv_np = (
            self._Sigma_inv_t.cpu().numpy()
            if self._Sigma_inv_t is not None
            else np.array([])
        )

        np.savez_compressed(
            path,
            signatures=sigs_data,
            mu_pool=mu_pool_np,
            Sigma_pool=Sigma_pool_np,
            Sinv_pool=Sinv_np,
            n_pool=np.array([self._n_pool]),
            shrinkage=self.shrinkage,
            delta=np.array([self._delta]),
            task_ids=list(self.signatures.keys()),
        )
        print(f"  [SRT-SAVE] {path}: {len(sigs_data)} tasks, "
              f"n_pool={self._n_pool}, shrinkage={self.shrinkage}, δ={self._delta:.4f}")

    def load(self, path: str):
        """Load router state from disk."""
        data = np.load(path, allow_pickle=True)

        self.shrinkage = str(data.get("shrinkage", "ridge"))
        self._delta = float(data.get("delta", [0.0])[0])
        self._n_pool = int(data.get("n_pool", [0])[0])

        # Restore pooled stats to self.dev
        mu_p = data["mu_pool"]
        self._mu_pool_t = (
            torch.from_numpy(mu_p).float().to(self.dev) if mu_p.size > 0 else None
        )
        sp = data["Sigma_pool"]
        self._Sigma_pool_t = (
            torch.from_numpy(sp).float().to(self.dev) if sp.size > 0 else None
        )
        si = data.get("Sinv_pool", np.array([]))
        self._Sigma_inv_t = (
            torch.from_numpy(si).float().to(self.dev) if si.size > 0 else None
        )

        # Restore centroids and n_tasks_list from signatures
        sigs_dict = data["signatures"].item()
        self._signatures_by_id = {}
        self.centroids = []
        self.n_tasks_list = []
        for tid, sig_dict in sigs_dict.items():
            sig = TaskSignature.from_dict(sig_dict)
            self._signatures_by_id[tid] = sig
            self.centroids.append(sig.mu)
            self.n_tasks_list.append(sig.n)

        print(f"  [SRT-LOAD] {path}: {len(sigs_dict)} tasks, "
              f"n_pool={self._n_pool}, shrinkage={self.shrinkage}, δ={self._delta:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
#  TASK SIGNATURE  (kept for save/load compatibility)
# ─────────────────────────────────────────────────────────────────────────────

class TaskSignature:
    """
    Statistical signature for one task.

    For PooledMahalanobisRouter, all tasks share Σ_pool⁻¹.
    This class stores per-task {μ_t, n_t} and shared routing state.

    Zero-rehearsal compliant: only sufficient statistics, no raw data.
    """

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
        self.n = n
        self.shrinkage = shrinkage
        self.delta = delta
        self.Sinv = Sinv.astype(np.float64) if Sinv is not None else None
        self.par = par

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
    def from_dict(cls, d: dict) -> "TaskSignature":
        return cls(
            task_id=d["task_id"],
            mu=d["mu"],
            Sigma=d["Sigma"],
            n=d["n"],
            shrinkage=d.get("shrinkage", "ridge"),
            delta=d.get("delta", 0.0),
            Sinv=d.get("Sinv"),
            par=d.get("par", 0.0),
        )


# ─────────────────────────────────────────────────────────────────────────────
#  SRTRouter  (legacy wrapper for backward compat)
# ─────────────────────────────────────────────────────────────────────────────

class SRTRouter:
    """
    Legacy wrapper providing backward-compatible interface.

    Internally uses PooledMahalanobisRouter with Ridge shrinkage.
    Preserves save/load format for compatibility with existing checkpoints.

    Args:
        shrinkage: 'ridge' | 'oas' | 'lw' | 'none' (default: 'ridge')
        device: 'cuda' | 'cpu' (default: auto-detect)
    """

    def __init__(
        self,
        shrinkage: str = "ridge",
        device: Optional[str] = None,
    ):
        self._impl = PooledMahalanobisRouter(shrinkage=shrinkage, device=device)

    def add_task(
        self,
        task_id: Union[int, str],
        h_train: np.ndarray,
    ) -> TaskSignature:
        return self._impl.add_task(task_id, h_train)

    def route(self, h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self._impl.route(h)

    def route_debug(self, h: np.ndarray) -> dict:
        return self._impl.route_debug(h)

    def summary(self) -> dict:
        return self._impl.summary()

    def save(self, path: str):
        self._impl.save(path)

    def load(self, path: str):
        self._impl.load(path)

    @property
    def signatures(self) -> Dict[Union[int, str], TaskSignature]:
        return self._impl.signatures
