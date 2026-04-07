"""
SRT Router: Statistical Routing Theory implementation.

Two modes:
  --srt_metric_mode hard:
    Whitening + ZCA + L2 (equivalent to Pooled Mahalanobis per Theorem 4).
    Matches routing_analysis experiment: whitening fitted on train, applied always.
    Single fixed metric → argmin is always valid.

  --srt_metric_mode dynamics:
    No whitening. SRM global metric selection on synthetic data.
    Bug fixed: SRM runs AFTER add_task so ALL tasks get SRM-assigned metric.
    → All tasks use same metric → argmin is valid.

Zero-rehearsal compliant: only sufficient statistics, no raw data.
"""

import math
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union


# ─────────────────────────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────────────────────────

def metric_l2(
    h: np.ndarray,    # (n, d) or (d,)
    mu: np.ndarray,
) -> np.ndarray:
    """L2 distance from centroid. Use when isotropic (PaR ≈ d)."""
    if h.ndim == 1:
        h = h.reshape(1, -1)
    diff = h - mu
    return np.sqrt(np.einsum('nd,nd->n', diff, diff))


def metric_mahalanobis(
    h: np.ndarray,
    mu: np.ndarray,
    Sinv: np.ndarray,
) -> np.ndarray:
    """Mahalanobis distance. Use when anisotropic (PaR ≪ d)."""
    if h.ndim == 1:
        h = h.reshape(1, -1)
    diff = h - mu
    return np.einsum('nd,dg,ng->n', diff, Sinv, diff)


def metric_psr(
    h: np.ndarray,
    mu: np.ndarray,
    eigvecs: np.ndarray,   # (d, k)
    eigvals: np.ndarray,   # (k,)
    d_total: int,
) -> np.ndarray:
    """
    Probabilistic Subspace Routing (PSR) metric from C1 §3.3.

    d_PSR(h, t) = ||U_tᵀ(h - μ_t)||² + (r/(d-r)) · ||(I - U_t U_tᵀ)(h - μ_t)||²

    Use when low-rank anisotropic (PaR ≪ d, effective dims low).
    """
    if h.ndim == 1:
        h = h.reshape(1, -1)
    diff = h - mu          # (n, d)

    proj = diff @ eigvecs                       # (n, k)
    in_sub = np.sum(proj ** 2, axis=-1)         # (n,)

    v_norm_sq = np.einsum('nd,nd->n', diff, diff)             # (n,)
    out_sub = v_norm_sq - in_sub                               # (n,)

    k = eigvecs.shape[1]
    scale = k / max(d_total - k, 1)
    return in_sub + scale * out_sub


def pinv_ridge(Sigma: np.ndarray, rcond: float = 1e-6) -> np.ndarray:
    """Stable pseudo-inverse of a symmetric covariance matrix."""
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals = np.maximum(eigvals, rcond * np.max(eigvals))
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T


# ─────────────────────────────────────────────────────────────────────────────
#  ZCA WHITENING
# ─────────────────────────────────────────────────────────────────────────────

def compute_whitening(task_embs: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ZCA whitening transform from pooled embeddings.

    ZCA: W such that (h - μ)ᵀ Wᵀ W (h - μ) is isotropic.

    Fit on ALL task embeddings pooled together (per experiment design
    in compare_routing.py: fit on train, apply to test).

    Returns: (mu_global, W_zca) where W_zca @ (h - mu_global) whitens h.
    """
    all_embs = np.vstack(list(task_embs.values()))
    mu_global = all_embs.mean(0)
    Xc = all_embs - mu_global
    cov_global = np.cov(Xc, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov_global)
    eigvals = np.maximum(eigvals, 1e-8)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    W_zca = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return mu_global, W_zca


def apply_whitening(
    h: np.ndarray,
    mu_global: np.ndarray,
    W_zca: np.ndarray,
) -> np.ndarray:
    """Apply ZCA whitening: h_whitened = (h - mu_global) @ W_zca.T."""
    if h.ndim == 1:
        h = h.reshape(1, -1)
    return (h - mu_global) @ W_zca.T


# ─────────────────────────────────────────────────────────────────────────────
#  PARTICIPATION RATIO
# ─────────────────────────────────────────────────────────────────────────────

def participation_ratio(Sigma: np.ndarray) -> float:
    """PaR = (Σλ)² / Σλ² ∈ [1, d]."""
    eigvals = np.linalg.eigvalsh(Sigma)
    eigvals = np.maximum(eigvals, 1e-10)
    return (eigvals.sum() ** 2) / (eigvals ** 2).sum()


def ledoit_wolf_shrinkage(Sigma: np.ndarray, factor: float = 0.1) -> np.ndarray:
    """
    Ledoit-Wolf shrinkage toward scalar identity.

    Σ_shrunk = (1 - δ) · Σ + δ · λ̄ · I
    """
    d = Sigma.shape[0]
    trace = np.trace(Sigma)
    sigma_mean = trace / d
    target = sigma_mean * np.eye(d)
    return (1 - factor) * Sigma + factor * target


# ─────────────────────────────────────────────────────────────────────────────
#  POOLED COVARIANCE UPDATE  (Welford–Hart)
# ─────────────────────────────────────────────────────────────────────────────

def update_pooled_cov(
    mu_old: np.ndarray,
    cov_old: np.ndarray,
    n_old: int,
    mu_new: np.ndarray,
    cov_new: np.ndarray,
    n_new: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Welford–Hart compact update for pooled mean and covariance.
    """
    total = n_old + n_new
    if total <= 1:
        raise ValueError(f"total={total} must be > 1")

    mu_pool = (n_old * mu_old + n_new * mu_new) / total

    delta = mu_new - mu_old
    C = (n_old * n_new / total) * np.outer(delta, delta)

    cov_pool = (
        (n_old - 1) * cov_old
        + (n_new - 1) * cov_new
        + C
    ) / (total - 1)

    return mu_pool, cov_pool, total


# ─────────────────────────────────────────────────────────────────────────────
#  POOLED SHRINKAGE  (Theorem 5)
# ─────────────────────────────────────────────────────────────────────────────

def pooled_shrinkage_target(
    Sigma_t: np.ndarray,
    Sigma_pool: np.ndarray,
    n_t: int,
    n_pool: int,
    alpha_min: float = 0.01,
    alpha_max: float = 0.99,
) -> Tuple[np.ndarray, float]:
    """
    Compute optimal pooled shrinkage target for task t (Theorem 5 from C1).

    Optimal:  Σ_t* = (1 - α_t*) Σ̂_t + α_t* Σ̂_pool
    where α_t* ≈ n_pool / (n_pool + n_t)
    """
    alpha_opt = n_pool / (n_pool + n_t)
    alpha_opt = max(alpha_min, min(alpha_max, alpha_opt))
    Sigma_shrunk = (1 - alpha_opt) * Sigma_t + alpha_opt * Sigma_pool
    return Sigma_shrunk, alpha_opt


# ─────────────────────────────────────────────────────────────────────────────
#  SRM METRIC SELECTION  (Theorem 7 — dynamics mode only)
# ─────────────────────────────────────────────────────────────────────────────

def srm_metric_selection(
    signatures: Dict[int, 'TaskSignature'],
    n_synthetic: int = 500,
    random_seed: int = 42,
) -> Tuple[Dict[int, str], Dict[str, float]]:
    """
    Structural Risk Minimization (SRM) from C1 Theorem 7.

    Generates synthetic validation points from each task's Gaussian model,
    evaluates each metric globally, selects the metric minimizing routing error.

    CRITICAL FIX: SRM must run AFTER all tasks (including current) are added
    to signatures. Otherwise current task uses PaR fallback → mixed metrics →
    argmin across incomparable distances → misroute.

    Returns: ({task_id: metric}, {metric_name: error_rate})
    """
    rng = np.random.RandomState(random_seed)
    task_list = sorted(signatures.keys())
    T = len(task_list)

    if T < 2:
        return {t: 'l2' for t in task_list}, {}

    # Generate synthetic validation data from each task
    h_val = []
    labels = []
    n_per = n_synthetic // T

    for t_id in task_list:
        sig = signatures[t_id]
        L = np.linalg.cholesky(sig.Sigma + 1e-6 * np.eye(sig.d))
        z = rng.randn(n_per, sig.d)
        h_syn = sig.mu + (L @ z.T).T
        h_val.append(h_syn)
        labels.extend([t_id] * n_per)

    h_val = np.vstack(h_val)
    labels = np.array(labels)

    # Evaluate each metric globally
    errors = {}
    for metric_name in ['l2', 'mahalanobis', 'psr']:
        n_correct = 0
        for i, h_i in enumerate(h_val):
            true_t = labels[i]
            best_t = None
            best_dist = np.inf
            for t_id in task_list:
                sig = signatures[t_id]
                if metric_name == 'l2':
                    d = metric_l2(h_i, sig.mu)
                elif metric_name == 'mahalanobis':
                    d = metric_mahalanobis(h_i, sig.mu, sig.Sinv)
                else:  # psr
                    k = max(1, int(np.sum(sig.eigvals > 1e-6 * sig.eigvals[0])))
                    d = metric_psr(h_i, sig.mu, sig.eigvecs[:, :k], sig.eigvals[:k], sig.d)

                if d < best_dist:
                    best_dist = d
                    best_t = t_id

            if best_t == true_t:
                n_correct += 1

        errors[metric_name] = 1.0 - n_correct / len(labels)

    # Global selection: one metric for ALL tasks
    best_metric = min(errors, key=errors.get)
    print(f"  [SRM] Routing errors: {errors}, selected: {best_metric}")

    # Same metric for all tasks
    return {t_id: best_metric for t_id in task_list}, errors


# ─────────────────────────────────────────────────────────────────────────────
#  TASK SIGNATURE
# ─────────────────────────────────────────────────────────────────────────────

class TaskSignature:
    """
    Statistical signature for one task.
    Stores: μ_t, Σ_t, n_t, eigenvalues, eigenvectors, metric type.
    Zero-rehearsal compliant: only sufficient statistics, no raw data.
    """

    def __init__(
        self,
        task_id: Union[int, str],
        mu: np.ndarray,
        Sigma: np.ndarray,
        n: int,
        metric: str = 'l2',
        alpha: float = 0.0,
        Sigma_raw: Optional[np.ndarray] = None,
        h_train: Optional[np.ndarray] = None,
        mu_raw: Optional[np.ndarray] = None,
    ):
        self.task_id = task_id
        self.mu = mu.astype(np.float64)
        self.d = mu.shape[0]
        self.Sigma_raw = (Sigma_raw if Sigma_raw is not None else Sigma).astype(np.float64)
        self.Sigma = Sigma.astype(np.float64)
        self.n = n
        self.alpha = alpha
        self._h_train = h_train.astype(np.float64) if h_train is not None else None
        self.mu_raw = mu_raw.astype(np.float64) if mu_raw is not None else mu.astype(np.float64).copy()
        self._metric = metric

        # Cached decompositions
        self._eigvals: Optional[np.ndarray] = None
        self._eigvecs: Optional[np.ndarray] = None
        self._Sinv: Optional[np.ndarray] = None
        self._par: Optional[float] = None

    @property
    def eigvals(self) -> np.ndarray:
        if self._eigvals is None:
            self._eigvals, self._eigvecs = np.linalg.eigh(self.Sigma)
            self._eigvals = np.maximum(self._eigvals, 1e-10)
            idx = np.argsort(self._eigvals)[::-1]
            self._eigvals = self._eigvals[idx]
            self._eigvecs = self._eigvecs[:, idx]
        return self._eigvals

    @property
    def eigvecs(self) -> np.ndarray:
        if self._eigvecs is None:
            _ = self.eigvals
        return self._eigvecs

    @property
    def Sinv(self) -> np.ndarray:
        if self._Sinv is None:
            self._Sinv = pinv_ridge(self.Sigma)
        return self._Sinv

    @property
    def par(self) -> float:
        if self._par is None:
            self._par = participation_ratio(self.Sigma)
        return self._par

    @property
    def metric(self) -> str:
        return self._metric

    def reshrink(self, Sigma_pool: np.ndarray, n_pool: int, alpha: float):
        """
        Re-shrink toward pooled covariance using raw Σ as base
        (to avoid compounding shrinkage across re-shrink rounds).
        """
        self.alpha = alpha
        self.Sigma = (1 - alpha) * self.Sigma_raw + alpha * Sigma_pool
        self.Sigma = self.Sigma.astype(np.float64)
        self._eigvals = None
        self._eigvecs = None
        self._Sinv = None
        self._par = None

    def distance(self, h: np.ndarray) -> np.ndarray:
        """
        Compute distance from h to this task's centroid using this task's metric.
        """
        if h.ndim == 1:
            h = h.reshape(1, -1)

        m = self.metric
        if m == 'l2':
            return metric_l2(h, self.mu)
        elif m == 'mahalanobis':
            return metric_mahalanobis(h, self.mu, self.Sinv)
        elif m == 'psr':
            k = max(1, int(np.sum(self.eigvals > 1e-6 * self.eigvals[0])))
            return metric_psr(h, self.mu, self.eigvecs[:, :k], self.eigvals[:k], self.d)
        else:
            # Fallback: PaR-based
            if self.par >= 0.9 * self.d:
                return metric_l2(h, self.mu)
            elif self.par >= 0.3 * self.d:
                return metric_mahalanobis(h, self.mu, self.Sinv)
            else:
                k = max(1, int(np.sum(self.eigvals > 1e-6 * self.eigvals[0])))
                return metric_psr(h, self.mu, self.eigvecs[:, :k], self.eigvals[:k], self.d)

    def to_dict(self) -> dict:
        d = {
            'task_id': self.task_id,
            'mu': self.mu,
            'Sigma': self.Sigma,
            'Sigma_raw': self.Sigma_raw,
            'mu_raw': self.mu_raw,
            'n': self.n,
            'metric': self.metric,
            'alpha': self.alpha,
        }
        # Hard mode needs raw embeddings for ZCA refit on future tasks
        if self._h_train is not None:
            d['h_train'] = self._h_train
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'TaskSignature':
        return cls(
            task_id=d['task_id'],
            mu=d['mu'],
            Sigma=d['Sigma'],
            n=d['n'],
            metric=d.get('metric', 'l2'),
            alpha=d.get('alpha', 0.0),
            Sigma_raw=d.get('Sigma_raw', d['Sigma']),
            h_train=d.get('h_train'),
            mu_raw=d.get('mu_raw'),
        )


# ─────────────────────────────────────────────────────────────────────────────
#  SRT ROUTER
# ─────────────────────────────────────────────────────────────────────────────

class SRTRouter:
    """
    Statistical Routing Theory Router.

    Two modes (set via srt_metric_mode):
      'hard'    : whitening + L2. No SRM. Matches routing_analysis experiment.
                  Whitening fitted on train embeddings, applied always.
                  Single fixed metric (L2 in whitened space = Pooled Mahalanobis).
                  argmin is always valid because all tasks use L2 in same space.

      'dynamics': no whitening. SRM global metric selection on synthetic data.
                  All tasks use same SRM-assigned metric → argmin is valid.
                  Bug fixed: SRM runs AFTER add_task so ALL tasks get metric.

    Zero-drift: no learnable parameters.
    """

    def __init__(
        self,
        srt_metric_mode: str = 'hard',
        use_shrink: bool = True,
        shrink_factor: float = 0.1,
    ):
        """
        Args:
            srt_metric_mode: 'hard' | 'dynamics'
                'hard'     → ZCA whitening + L2 (matches experiment)
                'dynamics' → SRM metric selection (matches contribution_UNIFIED)
            use_shrink: apply Ledoit-Wolf shrinkage to covariance
            shrink_factor: shrinkage intensity
        """
        assert srt_metric_mode in ('hard', 'dynamics'), \
            f"Unknown srt_metric_mode: {srt_metric_mode}"

        self.signatures: Dict[int, TaskSignature] = {}
        self._mu_pool: Optional[np.ndarray] = None
        self._Sigma_pool: Optional[np.ndarray] = None
        self._n_pool: int = 0

        self.srt_metric_mode = srt_metric_mode
        self.use_shrink = use_shrink
        self.shrink_factor = shrink_factor

        # ── ZCA whitening (hard mode) ────────────────────────────────────
        self._mu_global: Optional[np.ndarray] = None
        self._W_zca: Optional[np.ndarray] = None
        self._zca_fitted: bool = False   # True after first ZCA fit; NEVER refit

        # ── SRM state (dynamics mode) ────────────────────────────────────
        self._srm_metrics: Dict[int, str] = {}

    # ── Pooled statistics ───────────────────────────────────────────────

    def _update_pooled(self, mu_t: np.ndarray, Sigma_t: np.ndarray, n_t: int):
        """Update running pooled mean and covariance."""
        if self._n_pool == 0:
            self._mu_pool = mu_t.copy()
            self._Sigma_pool = Sigma_t.copy()
            self._n_pool = n_t
        else:
            self._mu_pool, self._Sigma_pool, self._n_pool = update_pooled_cov(
                self._mu_pool, self._Sigma_pool, self._n_pool,
                mu_t, Sigma_t, n_t,
            )

    def _reshrink_all(self):
        """Re-shrink ALL tasks toward updated pooled covariance (Theorem 5)."""
        if self._n_pool <= 1:
            return
        for t_id, sig in self.signatures.items():
            alpha_opt = self._n_pool / (self._n_pool + sig.n)
            alpha_opt = max(0.01, min(0.99, alpha_opt))
            sig.reshrink(self._Sigma_pool, self._n_pool, alpha_opt)

    # ── Add task ─────────────────────────────────────────────────────

    def add_task(
        self,
        task_id: Union[int, str],
        h_train: np.ndarray,
    ) -> TaskSignature:
        """
        Add a new task's statistical signature.

        HARD MODE — ZCA fit-once (the key fix):
          - Task 1: store raw stats only, ZCA NOT fitted yet
          - Task 2+: fit ZCA ONCE from pooled covariance of ALL seen tasks,
            WHITEN ALL tasks (including previous) with this fixed W.
            NEVER refit W again.
          - This matches the offline experiment (compute_whitening on all train
            embeddings once) and avoids the catastrophic incremental refitting
            bug where W changes every add_task() → centroids non-comparable.

        DYNAMICS MODE — SRM (unchanged logic):
          - Pooled update + re-shrink + SRM per task.

        Args:
            task_id: integer or string task ID
            h_train: (n_t, d) embeddings from FROZEN backbone

        Returns:
            TaskSignature for this task
        """
        n_t, d = h_train.shape

        # ── Sufficient statistics (always in raw space) ────────────────
        mu_t = h_train.mean(axis=0)
        Sigma_t = np.cov(h_train, rowvar=False, ddof=1)

        # Optional pre-whitening LW shrinkage (on raw covariance)
        if self.use_shrink:
            Sigma_t_shrunk = ledoit_wolf_shrinkage(Sigma_t, factor=self.shrink_factor)
        else:
            Sigma_t_shrunk = Sigma_t.copy()

        # ── Pooled statistics update (Welford–Hart) ───────────────────
        self._update_pooled(mu_t, Sigma_t_shrunk, n_t)

        # ── Dynamics mode ────────────────────────────────────────────────
        if self.srt_metric_mode == 'dynamics':
            # Re-shrink previous tasks toward updated pool
            if len(self.signatures) > 0:
                self._reshrink_all()

            sig = TaskSignature(
                task_id, mu_t, Sigma_t, n_t,
                metric='l2',
                Sigma_raw=Sigma_t,
                h_train=h_train,
                mu_raw=mu_t,
            )
            self.signatures[task_id] = sig

            if len(self.signatures) >= 2:
                srm_results, _ = srm_metric_selection(self.signatures)
                self._srm_metrics = srm_results
                for t_id, m in srm_results.items():
                    self.signatures[t_id]._metric = m
            return sig

        # ── Hard mode: ZCA fit-once ───────────────────────────────────
        # Re-shrink previous tasks toward updated pool (raw Σ, pre-whitening)
        if len(self.signatures) > 0:
            self._reshrink_all()

        if not self._zca_fitted:
            # FIRST TIME: fit ZCA from pooled mean + pooled covariance.
            # Uses Σ_pool which already aggregates ALL seen tasks via Welford–Hart.
            # This is the single, FINAL whitening transform — never refitted.
            self._mu_global = self._mu_pool.copy()
            cov_pool = self._Sigma_pool.copy()

            # Eigendecomposition of pooled covariance → ZCA transform
            eigvals, eigvecs = np.linalg.eigh(cov_pool)
            eigvals = np.maximum(eigvals, 1e-8)
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
            self._W_zca = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            self._zca_fitted = True

            # Re-whiten ALL previous tasks (from their raw centroids)
            for sig in self.signatures.values():
                # sig.Sigma was reshrunk-but-unwhitened by _reshrink_all()
                # sig.mu_raw is the unwhitened centroid
                sig.mu = apply_whitening(
                    sig.mu_raw.reshape(1, -1),
                    self._mu_global, self._W_zca,
                ).ravel()
                sig.Sigma = self._W_zca @ sig.Sigma @ self._W_zca.T
                sig._eigvals = None; sig._eigvecs = None
                sig._Sinv = None; sig._par = None
                sig._metric = 'l2'

            n_d = self._n_pool / d
            print(f"  [SRT] ZCA fitted once: n_pool={self._n_pool}, d={d}, n/d={n_d:.2f}")

        # Whiten current task with the FIXED (already-fitted) ZCA
        mu_t_w = apply_whitening(
            mu_t.reshape(1, -1), self._mu_global, self._W_zca,
        ).ravel()
        Sigma_t_w = self._W_zca @ Sigma_t_shrunk @ self._W_zca.T

        # NOTE: NO Ledoit-Wolf here — space is already whitened.
        # Post-whitening LW with lambda-bar-I target would distort isotropy.

        sig = TaskSignature(
            task_id, mu_t_w, Sigma_t_w, n_t,
            metric='l2',
            Sigma_raw=Sigma_t_shrunk,
            h_train=h_train,
            mu_raw=mu_t,
        )
        self.signatures[task_id] = sig
        return sig

    # ── Route ────────────────────────────────────────────────────────

    def route(self, h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Route embeddings h to the nearest stored task.

        Args:
            h: (n, d) or (d,) — embeddings from frozen backbone

        Returns:
            task_ids: (n,) — predicted task ID for each embedding
            dists: (n, T) — distance to each task
        """
        if h.ndim == 1:
            h = h.reshape(1, -1)

        # [hard mode] Apply ZCA whitening to query embeddings
        if self.srt_metric_mode == 'hard' and self._W_zca is not None:
            h = apply_whitening(h, self._mu_global, self._W_zca)

        n = h.shape[0]
        task_list = sorted(self.signatures.keys())
        T = len(task_list)

        if T == 0:
            raise RuntimeError("SRT Router: no tasks registered.")

        dists = np.zeros((n, T), dtype=np.float64)
        for i, t_id in enumerate(task_list):
            dists[:, i] = self.signatures[t_id].distance(h)

        nearest_idx = np.argmin(dists, axis=1)
        nearest_task = np.array(task_list)[nearest_idx]

        # ── DEBUG ────────────────────────────────────────────────
        if not hasattr(self, '_route_debug_count'):
            self._route_debug_count = 0
        self._route_debug_count += 1
        if self._route_debug_count % 1000 == 0 and n > 0:
            print(f"[SRT-ROUTE] mode={self.srt_metric_mode} n_tasks={T} "
                  f"task_list={task_list} whiten={'Yes' if self._W_zca is not None else 'No'}")
            print(f"[SRT-ROUTE] Sample 0: dists={dists[0,:].tolist()} "
                  f"argmin={nearest_idx[0]} pred={nearest_task[0]}")
            if hasattr(self, '_srm_metrics'):
                print(f"[SRT-ROUTE] SRM metrics: {self._srm_metrics}")

        return nearest_task, dists

    # ── Save / Load ───────────────────────────────────────────────────

    def save(self, path: str):
        """Save all signatures to disk."""
        sigs_data = {k: v.to_dict() for k, v in self.signatures.items()}
        np.savez_compressed(
            path,
            signatures=sigs_data,
            mu_pool=self._mu_pool if self._mu_pool is not None else np.array([]),
            Sigma_pool=self._Sigma_pool if self._Sigma_pool is not None else np.array([]),
            n_pool=np.array([self._n_pool]),
            srt_metric_mode=self.srt_metric_mode,
            use_shrink=np.array([self.use_shrink]),
            shrink_factor=np.array([self.shrink_factor]),
            mu_global=self._mu_global if self._mu_global is not None else np.array([]),
            W_zca=self._W_zca if self._W_zca is not None else np.array([]),
            zca_fitted=np.array([self._zca_fitted]),
        )

    def load(self, path: str):
        """Load signatures from disk."""
        data = np.load(path, allow_pickle=True)
        self.srt_metric_mode = str(data.get('srt_metric_mode', 'hard'))

        # Restore pooled statistics (empty array = None sentinel)
        mu_p = data['mu_pool']
        self._mu_pool = mu_p if mu_p.size > 0 else None
        Sigma_p = data['Sigma_pool']
        self._Sigma_pool = Sigma_p if Sigma_p.size > 0 else None
        self._n_pool = int(data['n_pool'][0])

        # Restore shrink settings
        if 'use_shrink' in data:
            self.use_shrink = bool(data['use_shrink'][0])
        if 'shrink_factor' in data:
            self.shrink_factor = float(data['shrink_factor'][0])

        # ZCA whitening (empty array = None sentinel)
        mu_g = data.get('mu_global', np.array([]))
        W_z = data.get('W_zca', np.array([]))
        self._mu_global = mu_g if mu_g.size > 0 else None
        self._W_zca = W_z if W_z.size > 0 else None
        # ZCA was fitted once; restore flag (backward compat: infer from W if not saved)
        if 'zca_fitted' in data:
            self._zca_fitted = bool(data['zca_fitted'][0])
        else:
            self._zca_fitted = (self._mu_global is not None and self._W_zca is not None)

        for k, v in data['signatures'].item().items():
            self.signatures[k] = TaskSignature.from_dict(v)

    # ── Summary ───────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Return routing statistics."""
        task_list = sorted(self.signatures.keys())
        metrics = [self.signatures[t].metric for t in task_list]
        pars = [self.signatures[t].par for t in task_list]
        return {
            'n_tasks': len(self.signatures),
            'task_ids': task_list,
            'metrics': metrics,
            'pars': [f"{p:.1f}" for p in pars],
            'avg_par': float(np.mean(pars)) if pars else 0.0,
            'pool_n': self._n_pool,
            'mode': self.srt_metric_mode,
        }
