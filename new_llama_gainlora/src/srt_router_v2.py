"""
Runtime SRT router ported from routing_analysis/routing_class_v2.py.

The deployed path only needs the pooled Mahalanobis family with ridge shrinkage,
but this module keeps the same shrinkage options and save/load surface so the
runtime and offline analysis share one statistical router implementation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch


HAS_CUDA = torch.cuda.is_available()
DEVICE_DEFAULT = "cuda" if HAS_CUDA else "cpu"
COVARIANCE_MODE = "within_class"


def _covariance_dof(n: int) -> int:
    return max(int(n) - 1, 0)


def _shrink_oas(cov: torch.Tensor, n: int, d: int) -> Tuple[torch.Tensor, float]:
    tr_s = torch.trace(cov).item()
    tr_s2 = (cov ** 2).sum().item()
    rho_hat = tr_s2 / (tr_s ** 2 + 1e-10)
    delta = min(1.0, max(0.0, rho_hat / (n + 1 - 2 / d)))
    target = (tr_s / d) * torch.eye(d, device=cov.device, dtype=cov.dtype)
    return (1 - delta) * cov + delta * target, float(delta)


def _shrink_lw(cov: torch.Tensor, n: int, d: int) -> Tuple[torch.Tensor, float]:
    tr_s = torch.trace(cov).item()
    sum_sq = (cov ** 2).sum().item()
    diag_sq = (torch.diag(cov) ** 2).sum().item()
    denom = n * (diag_sq - (tr_s / d) ** 2)
    if abs(denom) < 1e-10:
        delta = 1.0
    else:
        numerator = sum_sq - tr_s ** 2 / d
        delta = max(0.0, min(1.0, numerator / denom))
    target = (tr_s / d) * torch.eye(d, device=cov.device, dtype=cov.dtype)
    return (1 - delta) * cov + delta * target, float(delta)


def _shrink_ridge(cov: torch.Tensor, n: int, d: int) -> Tuple[torch.Tensor, float]:
    delta = d / (n + d + 1e-10)
    tr_s = torch.trace(cov).item()
    target = (tr_s / d) * torch.eye(d, device=cov.device, dtype=cov.dtype)
    return (1 - delta) * cov + delta * target, float(delta)


def _shrink_none(cov: torch.Tensor, n: int, d: int) -> Tuple[torch.Tensor, float]:
    return cov, 0.0


_SHRINK_METHODS = {
    "oas": _shrink_oas,
    "lw": _shrink_lw,
    "ridge": _shrink_ridge,
    "none": _shrink_none,
}


def welford_pooled_update(
    mu_old: torch.Tensor,
    cov_old: torch.Tensor,
    n_old: int,
    dof_old: int,
    mu_new: torch.Tensor,
    cov_new: torch.Tensor,
    n_new: int,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    total = n_old + n_new
    if total <= 0:
        return mu_old, cov_old, n_old, dof_old

    target_device = mu_new.device
    if mu_old.device != target_device:
        mu_old = mu_old.to(target_device)
        cov_old = cov_old.to(target_device)
    if cov_new.device != target_device:
        cov_new = cov_new.to(target_device)

    mu_pool = (n_old * mu_old + n_new * mu_new) / total
    dof_new = _covariance_dof(n_new)
    total_dof = dof_old + dof_new
    if total_dof <= 0:
        cov_pool = torch.zeros_like(cov_new)
    elif dof_old <= 0:
        cov_pool = cov_new.clone()
    elif dof_new <= 0:
        cov_pool = cov_old.clone()
    else:
        cov_pool = ((dof_old * cov_old) + (dof_new * cov_new)) / total_dof
    return mu_pool, cov_pool, total, total_dof


class TaskSignature:
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


class PooledMahalanobisRouter:
    def __init__(
        self,
        shrinkage: str = "ridge",
        device: Optional[str] = None,
        pca_components: Optional[int] = None,
    ):
        assert shrinkage in _SHRINK_METHODS, f"Unknown shrinkage: {shrinkage}"
        self.shrinkage = shrinkage
        self.shrink_fn = _SHRINK_METHODS[shrinkage]
        self.device = device or DEVICE_DEFAULT
        self.dev = torch.device(self.device)

        self.centroids: List[np.ndarray] = []
        self.n_tasks_list: List[int] = []
        self._signatures_by_id: Dict[Union[int, str], TaskSignature] = {}

        self._mu_pool_t: Optional[torch.Tensor] = None
        self._Sigma_pool_t: Optional[torch.Tensor] = None
        self._n_pool = 0
        self._cov_dof = 0
        self._Sigma_inv_t: Optional[torch.Tensor] = None
        self._delta = 0.0
        self._eigvals: Optional[np.ndarray] = None

        self.pca_components = pca_components
        self._pca_mean: Optional[torch.Tensor] = None
        self._pca_V: Optional[torch.Tensor] = None

    def _fit_pca(self, X: torch.Tensor) -> torch.Tensor:
        d = X.shape[1]
        if self.pca_components is None or self.pca_components >= d:
            return X
        k = min(self.pca_components, d - 1)
        X_centered = X - X.mean(dim=0)
        cov = (X_centered.T @ X_centered) / max(X.shape[0] - 1, 1)
        try:
            _, S, Vt = torch.linalg.svd(cov, full_matrices=False)
        except Exception:
            eigvals, eigvecs = torch.linalg.eigh(cov)
            idx = torch.argsort(eigvals, descending=True)
            S = eigvals[idx]
            Vt = eigvecs[:, idx].T
        V = Vt[:k].T
        self._pca_mean = X.mean(dim=0)
        self._pca_V = V.to(self.dev)
        self._eigvals = S[:k].cpu().numpy()
        return X_centered @ V.to(X.dtype)

    def _apply_pca(self, X: torch.Tensor) -> torch.Tensor:
        if self._pca_V is None:
            return X
        return (X - self._pca_mean.to(X.dtype)) @ self._pca_V.to(X.dtype)

    def _participation_ratio(self) -> float:
        if self._eigvals is None:
            return 0.0
        lam = np.maximum(self._eigvals, 1e-10)
        return float((lam.sum() ** 2) / (lam ** 2).sum())

    def add_task(self, task_id: Union[int, str], h_train: np.ndarray) -> TaskSignature:
        n_t, d_orig = h_train.shape
        X = torch.from_numpy(h_train.astype(np.float32)).to(self.dev)
        if self._pca_V is None and self.pca_components is not None:
            X = self._fit_pca(X)
        elif self._pca_V is not None:
            X = self._apply_pca(X)
        d = X.shape[1] if X.ndim == 2 else d_orig

        mu_t_t = X.mean(dim=0)
        Xc = X - mu_t_t
        Sigma_t_t = (Xc.T @ Xc) / max(n_t - 1, 1)
        del X, Xc

        mu_t_np = mu_t_t.cpu().numpy()
        self.centroids.append(mu_t_np)
        self.n_tasks_list.append(n_t)

        if self._n_pool == 0:
            self._mu_pool_t = mu_t_t.clone()
            self._Sigma_pool_t = Sigma_t_t.clone()
            self._n_pool = n_t
            self._cov_dof = _covariance_dof(n_t)
        else:
            self._mu_pool_t, self._Sigma_pool_t, self._n_pool, self._cov_dof = welford_pooled_update(
                self._mu_pool_t,
                self._Sigma_pool_t,
                self._n_pool,
                self._cov_dof,
                mu_t_t,
                Sigma_t_t,
                n_t,
            )

        del mu_t_t, Sigma_t_t

        Sigma_shrunk_t, self._delta = self.shrink_fn(self._Sigma_pool_t, max(self._cov_dof, 1), d)
        eigvals, eigvecs = torch.linalg.eigh(Sigma_shrunk_t)
        eigvals = torch.clamp(eigvals, min=1e-6 * eigvals.abs().max().item())
        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        self._Sigma_inv_t = eigvecs @ torch.diag(1.0 / eigvals) @ eigvecs.T
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

    def route(self, h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if h.ndim == 1:
            h = h.reshape(1, -1)
        if not self.centroids:
            raise RuntimeError("PooledMahalanobisRouter: no tasks registered.")

        H = torch.from_numpy(h.astype(np.float32)).to(self.dev)
        if self._pca_V is not None:
            H = self._apply_pca(H)

        dists = np.zeros((H.shape[0], len(self.centroids)), dtype=np.float64)
        for idx, mu_t_np in enumerate(self.centroids):
            mu_t_t = torch.from_numpy(mu_t_np.astype(np.float32)).to(self.dev)
            diff = H - mu_t_t
            diff_Sinv = diff @ self._Sigma_inv_t
            dists[:, idx] = (diff * diff_Sinv).sum(dim=1).cpu().numpy()

        nearest_idx = np.argmin(dists, axis=1)
        task_ids_ordered = list(self._signatures_by_id.keys())
        nearest_task = np.array([task_ids_ordered[i] for i in nearest_idx])
        return nearest_task, dists

    def route_debug(self, h: np.ndarray) -> dict:
        task_ids, dists = self.route(h)
        sorted_idx = np.argsort(dists, axis=1)
        task_id_list = list(self._signatures_by_id.keys())
        best_idx = sorted_idx[:, 0]
        best_d = dists[np.arange(dists.shape[0]), best_idx]

        if dists.shape[1] >= 2:
            second_idx = sorted_idx[:, 1]
            second_d = dists[np.arange(dists.shape[0]), second_idx]
            second_task = np.array([task_id_list[i] for i in second_idx])
            with np.errstate(divide="ignore", invalid="ignore"):
                conf = (second_d - best_d) / (best_d + 1e-10)
                conf = np.where(np.isfinite(conf), conf, 999.0)
        else:
            second_d = np.full_like(best_d, np.inf)
            second_task = np.full(dists.shape[0], -1, dtype=task_ids.dtype)
            conf = np.full(dists.shape[0], 999.0)

        return {
            "task_ids": task_ids,
            "dists": dists,
            "nearest_task": task_ids,
            "second_task": second_task,
            "best_dist": best_d,
            "second_dist": second_d,
            "confidence_ratio": conf,
            "n_tasks": len(task_id_list),
            "task_id_list": task_id_list,
        }

    def summary(self) -> dict:
        task_ids = list(self._signatures_by_id.keys())
        par = self._participation_ratio()
        return {
            "n_tasks": len(task_ids),
            "task_ids": task_ids,
            "metrics": [self.shrinkage] * len(task_ids),
            "avg_par": par,
            "shrinkage": self.shrinkage,
            "delta": self._delta,
            "n_pool": self._n_pool,
            "cov_dof": self._cov_dof,
            "covariance_mode": COVARIANCE_MODE,
            "par": par,
            "par_d_ratio": par / 4096 if self._cov_dof > 0 else 0.0,
        }

    def save(self, path: str):
        sigs_data = {key: value.to_dict() for key, value in self._signatures_by_id.items()}
        np.savez_compressed(
            path,
            signatures=sigs_data,
            mu_pool=self._mu_pool_t.cpu().numpy() if self._mu_pool_t is not None else np.array([]),
            Sigma_pool=self._Sigma_pool_t.cpu().numpy() if self._Sigma_pool_t is not None else np.array([]),
            Sinv_pool=self._Sigma_inv_t.cpu().numpy() if self._Sigma_inv_t is not None else np.array([]),
            n_pool=np.array([self._n_pool]),
            cov_dof=np.array([self._cov_dof]),
            shrinkage=self.shrinkage,
            delta=np.array([self._delta]),
            covariance_mode=np.array([COVARIANCE_MODE]),
            task_ids=list(self._signatures_by_id.keys()),
            pca_components=np.array([self.pca_components or -1]),
            pca_mean=self._pca_mean.cpu().numpy() if self._pca_mean is not None else np.array([]),
            pca_V=self._pca_V.cpu().numpy() if self._pca_V is not None else np.array([]),
        )

    def load(self, path: str):
        data = np.load(path, allow_pickle=True)
        self.shrinkage = str(data.get("shrinkage", "ridge"))
        self.shrink_fn = _SHRINK_METHODS[self.shrinkage]
        self._delta = float(data.get("delta", [0.0])[0])
        self._n_pool = int(data.get("n_pool", [0])[0])
        covariance_mode = str(data.get("covariance_mode", np.array(["union_legacy"]))[0])
        if covariance_mode != COVARIANCE_MODE:
            raise ValueError(
                f"Router state at {path} uses covariance_mode={covariance_mode}. "
                "Regenerate router statistics with within-class pooled covariance "
                "to keep runtime and offline verification aligned."
            )
        self._cov_dof = int(data.get("cov_dof", [0])[0])

        mu_p = data["mu_pool"]
        self._mu_pool_t = torch.from_numpy(mu_p).float().to(self.dev) if mu_p.size > 0 else None
        sigma_p = data["Sigma_pool"]
        self._Sigma_pool_t = torch.from_numpy(sigma_p).float().to(self.dev) if sigma_p.size > 0 else None
        sinv_p = data.get("Sinv_pool", np.array([]))
        self._Sigma_inv_t = torch.from_numpy(sinv_p).float().to(self.dev) if sinv_p.size > 0 else None

        sigs_dict = data["signatures"].item()
        self._signatures_by_id = {}
        self.centroids = []
        self.n_tasks_list = []
        for task_id, sig_dict in sigs_dict.items():
            sig = TaskSignature.from_dict(sig_dict)
            self._signatures_by_id[task_id] = sig
            self.centroids.append(sig.mu)
            self.n_tasks_list.append(sig.n)

        pca_arr = data.get("pca_components", np.array([-1]))
        pca_val = int(pca_arr[0]) if pca_arr.size > 0 else -1
        self.pca_components = pca_val if pca_val > 0 else None
        if self.pca_components is not None:
            pca_mean = data.get("pca_mean", np.array([]))
            pca_V = data.get("pca_V", np.array([]))
            if pca_mean.size > 0 and pca_V.size > 0:
                self._pca_mean = torch.from_numpy(pca_mean).float().to(self.dev)
                self._pca_V = torch.from_numpy(pca_V).float().to(self.dev)

    @property
    def signatures(self) -> Dict[Union[int, str], TaskSignature]:
        return self._signatures_by_id


class SRTRouter:
    def __init__(
        self,
        shrinkage: str = "ridge",
        device: Optional[str] = None,
        pca_components: Optional[int] = None,
    ):
        self._impl = PooledMahalanobisRouter(
            shrinkage=shrinkage,
            device=device,
            pca_components=pca_components,
        )

    def add_task(self, task_id: Union[int, str], h_train: np.ndarray) -> TaskSignature:
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