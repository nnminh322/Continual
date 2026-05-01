"""
SRT Thm 4: Pooled Mahalanobis Router with Welford-Hart incremental update.

d_t(h) = (h - μ_t)^T · Σ_pool^{-1} · (h - μ_t)

GPU-accelerated via torch (CUDA). Falls back to numpy on CPU.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _device() -> str:
    return "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"


# ─── Incremental Pooled Statistics ───────────────────────────────────────────


def welford_pooled_update(
    mu_old: np.ndarray,
    cov_old: np.ndarray,
    n_old: int,
    mu_new: np.ndarray,
    cov_new: np.ndarray,
    n_new: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Welford-Hart pooled update for mean and covariance.
    """
    total = n_old + n_new
    mu_pool = (n_old * mu_old + n_new * mu_new) / total

    delta_mu = mu_new - mu_old
    cross = (n_old * n_new / total) * np.outer(delta_mu, delta_mu)

    cov_pool = (
        (n_old - 1) * cov_old
        + (n_new - 1) * cov_new
        + cross
    ) / max(total - 1, 1)

    return mu_pool, cov_pool, total


def participation_ratio_np(cov: np.ndarray) -> float:
    """Participation Ratio on CPU via numpy."""
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, a_min=1e-10, a_max=None)
    par = (eigvals.sum() ** 2) / ((eigvals ** 2).sum())
    return float(par)


def _participation_ratio_torch(cov: torch.Tensor) -> float:
    """Participation Ratio on GPU via torch."""
    eigvals = torch.linalg.eigvalsh(cov)
    eigvals = torch.clamp(eigvals, min=1e-10)
    par = (eigvals.sum() ** 2) / ((eigvals ** 2).sum())
    return float(par.item())


# ─── Shrinkage Methods ───────────────────────────────────────────────────────


def _shrink_ridge(cov: np.ndarray, n: int) -> Tuple[np.ndarray, float]:
    """Analytical ridge: δ = d/(n+d)."""
    d = cov.shape[0]
    delta = d / (n + d + 1e-10)
    tr_S = np.trace(cov)
    target = (tr_S / d) * np.eye(d)
    return (1 - delta) * cov + delta * target, float(delta)


def _shrink_lw(cov: np.ndarray, n: int) -> Tuple[np.ndarray, float]:
    """Ledoit-Wolf Oracle Linear Shrinkage (2004)."""
    d = cov.shape[0]
    tr_S = float(np.trace(cov))
    sum_sq = float((cov ** 2).sum())
    diag_sq = float((np.diag(cov) ** 2).sum())
    denom = n * (diag_sq - (tr_S / d) ** 2)
    if abs(denom) < 1e-10:
        delta = 1.0
    else:
        numerator = sum_sq - tr_S ** 2 / d
        delta = max(0.0, min(1.0, numerator / denom))
    target = (tr_S / d) * np.eye(d)
    return (1 - delta) * cov + delta * target, float(delta)


_SHRINKAGE_METHODS = {
    "ridge": _shrink_ridge,
    "lw": _shrink_lw,
}


# ─── GPU-accelerated distance core ───────────────────────────────────────────


def _compute_distances_gpu(
    h_batch: torch.Tensor,  # (B, d) on CUDA
    centroids: torch.Tensor,  # (T, d) on CUDA
    Sinv: torch.Tensor,  # (d, d) on CUDA
) -> torch.Tensor:
    """
    Compute Mahalanobis distances on GPU — no Python loop.

    d²_{b,t} = (h_b - μ_t)^T · Sinv · (h_b - μ_t)
                = h_b^T·Sinv·h_b  -  2·h_b^T·Sinv·μ_t  +  μ_t^T·Sinv·μ_t

    Vectorized over all B batches and T tasks.
    Returns (B, T) distance tensor.
    """
    # h^T·Sinv·h  for each batch  →  (B,)
    h_Sinv_h = (h_batch @ Sinv * h_batch).sum(dim=1)

    # h^T·Sinv·μ  for each (b, t)  →  (B, T)
    h_Sinv_mu = h_batch @ Sinv @ centroids.T

    # μ^T·Sinv·μ  for each task  →  (T,)
    mu_Sinv_mu = (centroids @ Sinv * centroids).sum(dim=1)

    # d² = h_Sinv_h[:, None] - 2*h_Sinv_mu + mu_Sinv_mu[None, :]
    dists_sq = h_Sinv_h[:, None] - 2 * h_Sinv_mu + mu_Sinv_mu[None, :]

    return dists_sq


def _eigh_inverse_on_device(
    Sigma: Union[np.ndarray, torch.Tensor],
    device: str,
    dtype: torch.dtype = torch.float32,
) -> Tuple:
    """
    Eigendecomposition-based inverse on specified device.

    Handles both numpy (CPU) and torch (CUDA) inputs.
    Returns (inverse, eigenvalues) on the same device.
    """
    if isinstance(Sigma, np.ndarray):
        # CPU path: numpy
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        max_abs = np.abs(eigvals[-1])
        eigvals = np.clip(eigvals, a_min=1e-6 * max_abs, a_max=None)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        Sinv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
        return Sinv, eigvals

    # GPU path: torch
    if device == "cuda":
        Sigma_t = torch.from_numpy(Sigma).to(dtype).cuda()
    else:
        Sigma_t = torch.from_numpy(Sigma).to(dtype)

    eigvals, eigvecs = torch.linalg.eigh(Sigma_t)
    max_abs = torch.abs(eigvals[-1])
    eigvals = torch.clamp(eigvals, min=1e-6 * max_abs)
    idx = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    Sinv = eigvecs @ torch.diag(1.0 / eigvals) @ eigvecs.T
    return Sinv, eigvals


# ─── Core Router ─────────────────────────────────────────────────────────────


class PooledMahalanobisRouter:
    """
    SRT Theorem 4: Pooled Mahalanobis Distance router.

    Routing decision:
        d_t(h) = (h - μ_t)^T · Σ_pool^{-1} · (h - μ_t)
        t* = argmin_t d_t(h)

    GPU: Uses torch on CUDA for distance computation when device='cuda'.
    CPU: Falls back to numpy. Eigendecomposition always on CPU via numpy.

    Args:
        shrinkage: 'ridge' (default, δ=d/(n+d)) or 'lw' (Ledoit-Wolf 2004).
        device: 'cuda', 'cpu', or 'auto'.
    """

    def __init__(
        self,
        shrinkage: str = "ridge",
        device: str = "auto",
    ):
        if shrinkage not in _SHRINKAGE_METHODS:
            raise ValueError(f"Unknown shrinkage: {shrinkage}. Options: {list(_SHRINKAGE_METHODS.keys())}")

        self.shrinkage = shrinkage
        self.shrink_fn = _SHRINKAGE_METHODS[shrinkage]

        if device == "auto":
            self.device = _device()
        else:
            self.device = device

        # GPU tensors (None on CPU path)
        self._centroids_t: Optional[torch.Tensor] = None
        self._Sinv_t: Optional[torch.Tensor] = None
        self._torch_ready = False

        # CPU/NumPy path
        self.centroids: List[np.ndarray] = []
        self.task_names: List[str] = []
        self.n_tasks_list: List[int] = []

        self.mu_pool: Optional[np.ndarray] = None
        self.Sigma_pool: Optional[np.ndarray] = None
        self.n_pool: int = 0
        self.Sinv: Optional[np.ndarray] = None
        self._delta: float = 0.0
        self._par: float = 0.0

    @property
    def n_tasks(self) -> int:
        return len(self.centroids)

    def add_task(self, embs: np.ndarray, task_name: Optional[str] = None) -> None:
        """
        Add a task's embeddings to the router.

        Args:
            embs: (N, d) numpy array of embeddings.
            task_name: Optional task identifier.
        """
        n_t, d = embs.shape
        mu_t = embs.mean(axis=0)

        # Covariance
        if n_t == 1:
            var_est = float((embs.std(axis=0, ddof=1) ** 2).mean())
            if var_est < 1e-8:
                var_est = 1.0
            Sigma_t = np.eye(d) * var_est
        else:
            Xc = embs - mu_t
            Sigma_t = (Xc.T @ Xc) / max(n_t - 1, 1)

        # Store centroid (numpy always)
        self.centroids.append(mu_t)
        self.task_names.append(task_name or f"task_{self.n_tasks}")
        self.n_tasks_list.append(n_t)

        # Welford-Hart pooled update
        if self.n_pool == 0:
            self.mu_pool = mu_t.copy()
            self.Sigma_pool = Sigma_t.copy()
            self.n_pool = n_t
        else:
            self.mu_pool, self.Sigma_pool, self.n_pool = welford_pooled_update(
                self.mu_pool, self.Sigma_pool, self.n_pool,
                mu_t, Sigma_t, n_t,
            )

        # Shrink + invert (numpy, always CPU)
        Sigma_shrunk, self._delta = self.shrink_fn(self.Sigma_pool, self.n_pool)
        self.Sinv, eigvals = _eigh_inverse_on_device(Sigma_shrunk, "cpu")
        self._par = participation_ratio_np(self.Sigma_pool)

        # Push to GPU tensors
        self._push_to_gpu()

    def _push_to_gpu(self) -> None:
        """Sync numpy centroids/Sinv to GPU tensors."""
        if self.device != "cuda" or not HAS_TORCH:
            return

        C = np.stack(self.centroids)  # (T, d)
        self._centroids_t = torch.from_numpy(C).cuda()
        self._Sinv_t = torch.from_numpy(self.Sinv.astype(np.float32)).cuda()
        self._torch_ready = True

    def route(self, h_batch: np.ndarray) -> np.ndarray:
        """
        Route embeddings to nearest task.

        Args:
            h_batch: (B, d) numpy array.

        Returns:
            (B,) predicted task indices.
        """
        if self.n_tasks == 0:
            return np.zeros(h_batch.shape[0], dtype=np.int64)

        if self._torch_ready and self.device == "cuda":
            # GPU path
            h_t = torch.from_numpy(h_batch.astype(np.float32)).cuda()
            with torch.no_grad():
                dists = _compute_distances_gpu(h_t, self._centroids_t, self._Sinv_t)
            preds = dists.argmin(dim=1).cpu().numpy()
        else:
            # CPU path
            dists = self._compute_distances_cpu(h_batch)
            preds = dists.argmin(axis=1).astype(np.int64)

        return preds

    def _compute_distances_cpu(self, h_batch: np.ndarray) -> np.ndarray:
        """
        CPU path: NumPy Mahalanobis distances.
        Fallback when GPU unavailable.
        """
        Sinv = self.Sinv
        n_tasks = self.n_tasks
        dists = np.zeros((h_batch.shape[0], n_tasks), dtype=np.float64)

        for i, mu_t in enumerate(self.centroids):
            diff = h_batch - mu_t
            diff_Sinv = diff @ Sinv
            dists[:, i] = (diff * diff_Sinv).sum(axis=1)

        return dists

    def route_with_confidence(
        self, h_batch: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Route with confidence score = (d_2nd - d_1st) / d_1st.
        """
        if self.n_tasks == 0:
            return (
                np.zeros(h_batch.shape[0], dtype=np.int64),
                np.zeros(h_batch.shape[0]),
            )

        if self._torch_ready and self.device == "cuda":
            h_t = torch.from_numpy(h_batch.astype(np.float32)).cuda()
            with torch.no_grad():
                dists = _compute_distances_gpu(h_t, self._centroids_t, self._Sinv_t)
            sorted_dists = torch.sort(dists, dim=1).values
            d_first = sorted_dists[:, 0].cpu().numpy()
            d_second = sorted_dists[:, 1].cpu().numpy()
            preds = dists.argmin(dim=1).cpu().numpy()
        else:
            dists = self._compute_distances_cpu(h_batch)
            sorted_idx = np.argsort(dists, axis=1)
            d_first = dists[np.arange(len(dists)), sorted_idx[:, 0]]
            d_second = dists[np.arange(len(dists)), sorted_idx[:, 1]]
            preds = dists.argmin(axis=1).astype(np.int64)

        confidence = np.where(d_first > 1e-8, (d_second - d_first) / d_first, 0.0)
        return preds, confidence

    def get_diagnostics(self) -> Dict:
        """Return router diagnostics."""
        d = self.centroids[0].shape[0] if self.centroids else 0
        return {
            "n_tasks": self.n_tasks,
            "n_pool": self.n_pool,
            "shrinkage_delta": self._delta,
            "shrinkage_method": self.shrinkage,
            "par": self._par,
            "par_ratio": self._par / d if d > 0 else 0.0,
            "dimension": d,
            "device": self.device,
            "gpu_enabled": self._torch_ready,
            "Sinv_condition": float(np.linalg.cond(self.Sinv)) if self.Sinv is not None else None,
            "task_names": list(self.task_names),
        }

    def save(self, path: str) -> None:
        """Save router state to .npz."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "centroids": np.stack(self.centroids) if self.centroids else np.array([]),
            "task_names": json.dumps(self.task_names),
            "n_tasks_list": np.array(self.n_tasks_list),
            "mu_pool": self.mu_pool if self.mu_pool is not None else np.array([]),
            "Sigma_pool": self.Sigma_pool if self.Sigma_pool is not None else np.array([]),
            "n_pool": self.n_pool,
            "Sinv": self.Sinv if self.Sinv is not None else np.array([]),
            "delta": self._delta,
            "shrinkage": self.shrinkage,
            "par": self._par,
            "device": self.device,
        }
        np.savez_compressed(path, **data)

    def load(self, path: str) -> None:
        """Load router state from .npz."""
        data = np.load(Path(path), allow_pickle=True)

        self.shrinkage = str(data["shrinkage"])
        self.shrink_fn = _SHRINKAGE_METHODS.get(self.shrinkage, _shrink_ridge)
        self.device = str(data.get("device", "cpu"))

        centroids = data["centroids"]
        self.centroids = [centroids[i] for i in range(len(centroids))]

        self.task_names = json.loads(str(data["task_names"]))
        self.n_tasks_list = list(data["n_tasks_list"])

        self.mu_pool = None if data["mu_pool"].size == 0 else data["mu_pool"]
        self.Sigma_pool = None if data["Sigma_pool"].size == 0 else data["Sigma_pool"]
        self.n_pool = int(data["n_pool"])
        self.Sinv = None if data["Sinv"].size == 0 else data["Sinv"]
        self._delta = float(data["delta"])
        self._par = float(data["par"])

        self._push_to_gpu()
