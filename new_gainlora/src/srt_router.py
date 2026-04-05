"""
SRT Router: Statistical Routing Theory implementation.

From contribution_UNIFIED.md (C1 / SRT):
  - Per-task signatures: {μ_t, Σ_t}
  - Metric selection: L2 / Mahalanobis / PSR based on anisotropy
  - Routing decision: argmin_t d_SRT(h, {μ_t, Σ_t})
  - Pooled shrinkage (Theorem 5): Σ_s → (1-α_s)Σ_s + α_s Σ_pool
  - SRM metric selection (Theorem 7): incremental validation via synthetic data

Metrics:
  1. L2 (isotropic): d_L2(h, t) = ||h - μ_t||₂
  2. Mahalanobis (anisotropic): d_Mah(h, t) = (h-μ_t)ᵀ Σ_t⁻¹ (h-μ_t)
  3. PSR (low-rank anisotropic): d_PSR(h, t) from C1 §3.3

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
    """L2 distance from centroid. Use when task is isotropic (PaR ≈ d)."""
    if h.ndim == 1:
        h = h.reshape(1, -1)
    diff = h - mu
    return np.sqrt(np.einsum('nd,nd->n', diff, diff))


def metric_mahalanobis(
    h: np.ndarray,
    mu: np.ndarray,
    Sinv: np.ndarray,
) -> np.ndarray:
    """Mahalanobis distance. Use when task is anisotropic (PaR ≪ d)."""
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

    Use when task is low-rank anisotropic (PaR ≪ d, effective dims low).
    """
    if h.ndim == 1:
        h = h.reshape(1, -1)
    diff = h - mu          # (n, d)

    # In-subspace: squared norm in top-k eigenspace
    # FIX: ||Uᵀv||² = sum_k (⟨v, u_k⟩)² = sum over k of (vᵀu_k)²
    # Correct: (diff @ eigvecs)² summed over k → (n, k) → sum over k
    proj = diff @ eigvecs                       # (n, k)
    in_sub = np.sum(proj ** 2, axis=-1)         # (n,)

    # Out-of-subspace: squared norm in orthogonal complement
    # ||v||² - ||Uᵀv||²
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

    where δ = shrink_factor (default 0.1).
    Shrinks eigenvalues toward their mean — stabilizes inverse especially when n < d.

    Args:
        Sigma: (d, d) sample covariance
        factor: shrinkage intensity δ ∈ [0, 1]

    Returns:
        shrunk covariance (d, d)
    """
    d = Sigma.shape[0]
    trace = np.trace(Sigma)
    # Scalar target: mean eigenvalue · I
    sigma_mean = trace / d
    target = sigma_mean * np.eye(d)
    return (1 - factor) * Sigma + factor * target


# ─────────────────────────────────────────────────────────────────────────────
#  POOLLED COVARIANCE UPDATE  (Welford–Hart compact formula)
# ─────────────────────────────────────────────────────────────────────────────

def update_pooled_cov(
    mu_old: np.ndarray,     # (d,)
    cov_old: np.ndarray,     # (d, d)
    n_old: int,
    mu_new: np.ndarray,     # (d,)
    cov_new: np.ndarray,    # (d, d)
    n_new: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Welford–Hart compact update for pooled mean and covariance.

    Correct formula (no overcounting):
      mu_pool = (n_old·mu_old + n_new·mu_new) / (n_old + n_new)
      cov_pool = ((n_old-1)·cov_old + (n_new-1)·cov_new + C) / (n_old + n_new - 1)

    where C = (n_old·n_new)/(n_old+n_new) · (mu_new - mu_old)·(mu_new - mu_old)ᵀ

    Returns: (mu_pool, cov_pool, n_pool)
    """
    total = n_old + n_new
    if total <= 1:
        raise ValueError(f"total={total} must be > 1")

    mu_pool = (n_old * mu_old + n_new * mu_new) / total

    # Between-group correction
    delta = mu_new - mu_old
    C = (n_old * n_new / total) * np.outer(delta, delta)

    # Combined covariance (Bessel correction)
    cov_pool = (
        (n_old - 1) * cov_old
        + (n_new - 1) * cov_new
        + C
    ) / (total - 1)

    return mu_pool, cov_pool, total


# ─────────────────────────────────────────────────────────────────────────────
#  POOLLED SHRINKAGE  (Theorem 5 — the CORRECT shrinkage target)
# ─────────────────────────────────────────────────────────────────────────────

def pooled_shrinkage_target(
    Sigma_t: np.ndarray,      # (d, d) sample covariance of task t
    Sigma_pool: np.ndarray,  # (d, d) pooled covariance
    n_t: int,               # sample count for task t
    n_pool: int,             # total pooled sample count
    alpha_min: float = 0.01,
    alpha_max: float = 0.99,
) -> Tuple[np.ndarray, float]:
    """
    Compute optimal pooled shrinkage target for task t (Theorem 5 from C1).

    Optimal:  Σ_t* = (1 - α_t*) Σ̂_t + α_t* Σ̂_pool

    where α_t* = argmin E[||Σ* - Σ_true||²]
    Analytical solution:
        α_t* = max(α_min, min(α_max, tr((Σ̂_pool - Σ̂_t)·Σ_true) / tr(Σ̂_pool² - Σ̂_t²))

    Asymptotically, this reduces to:
        α_t* = max(α_min, min(α_max, n_pool / (n_pool + n_t)))

    For finite samples, use analytical form with ridge approximation.

    Returns: (shrunk_Sigma_t, optimal_alpha)
    """
    # Analytical optimal alpha (Ledoit–Wolf type for pooled target)
    # α* ≈ n_pool / (n_pool + n_t) for the pooled case
    # (this minimizes mean-squared error when the true distribution is a mixture)
    alpha_opt = n_pool / (n_pool + n_t)
    alpha_opt = max(alpha_min, min(alpha_max, alpha_opt))

    Sigma_shrunk = (1 - alpha_opt) * Sigma_t + alpha_opt * Sigma_pool

    return Sigma_shrunk, alpha_opt


# ─────────────────────────────────────────────────────────────────────────────
#  SRM METRIC SELECTION  (Theorem 7 — the CORRECT algorithm)
# ─────────────────────────────────────────────────────────────────────────────

def srm_metric_selection(
    signatures: Dict[int, 'TaskSignature'],
    Sigma_pool: np.ndarray,
    n_synthetic: int = 500,
    random_seed: int = 42,
) -> Dict[int, str]:
    """
    Structural Risk Minimization (SRM) metric selection from C1 Theorem 7.

    Algorithm:
      1. Generate synthetic validation points from each task's Gaussian model:
         for task t: sample n_synthetic/2 points from N(μ_t, Σ_t)
      2. For each metric {L2, Mahalanobis, PSR}:
         - Route synthetic points
         - Measure routing error on synthetic validation set
      3. Select metric with minimum routing error on synthetic set
      4. OR: select per-task based on that task's PaR

    Practical SRM variant:
      - Generate synthetic data from each stored {μ_t, Σ_t}
      - Evaluate each metric on synthetic data
      - Pick metric minimizing misclassification rate

    Returns: {task_id: 'l2'|'mahalanobis'|'psr'}
    """
    rng = np.random.RandomState(random_seed)
    task_list = sorted(signatures.keys())
    T = len(task_list)

    if T < 2:
        # Not enough tasks for SRM — use PaR-based default
        return {t: signatures[t].metric for t in task_list}

    # Generate synthetic validation data
    h_val = []
    labels = []
    per_task_h = {}
    n_per = n_synthetic // T

    for t_id in task_list:
        sig = signatures[t_id]
        # Sample from N(μ_t, Σ_t)
        L = np.linalg.cholesky(sig.Sigma + 1e-6 * np.eye(sig.d))
        z = rng.randn(n_per, sig.d)
        h_syn = sig.mu + (L @ z.T).T
        per_task_h[t_id] = h_syn
        h_val.append(h_syn)
        labels.extend([t_id] * n_per)

    h_val = np.vstack(h_val)          # (n_synthetic, d)
    labels = np.array(labels)           # (n_synthetic,)

    # Compute routing errors for each metric
    errors = {}
    for metric_name in ['l2', 'mahalanobis', 'psr']:
        n_correct = 0
        for i, h_i in enumerate(h_val):
            true_t = labels[i]

            # Compute distance to each task under this metric
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

    # Select metric minimizing error
    best_metric = min(errors, key=errors.get)
    print(f"  [SRM] Routing errors: {errors}, selected: {best_metric}")

    # Return: same metric for all tasks (SRM global selection)
    # OR: per-task based on each task's PaR
    return {t_id: best_metric for t_id in task_list}


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
        metric: str = 'auto',
        # Pooled shrinkage params (used during re-shrink)
        alpha: float = 0.0,
        Sigma_raw: Optional[np.ndarray] = None,
    ):
        self.task_id = task_id
        self.mu = mu.astype(np.float64)
        self.d = mu.shape[0]
        # Always store the RAW sample covariance separately.
        # All pooled-shrinkage operations must use Sigma_raw as the base
        # to avoid compounding shrinkage across re-shrink rounds.
        self.Sigma_raw = (Sigma_raw if Sigma_raw is not None else Sigma).astype(np.float64)
        # Sigma is the "current" version — may be shrunk, pooled, or raw
        self.Sigma = Sigma.astype(np.float64)
        self.n = n
        self.alpha = alpha

        # Eigendecomposition (computed lazily)
        self._eigvals: Optional[np.ndarray] = None
        self._eigvecs: Optional[np.ndarray] = None
        self._Sinv: Optional[np.ndarray] = None

        # Geometry descriptors
        self._par: Optional[float] = None
        self._metric: str = metric

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
            _ = self.eigvals  # triggers lazy computation
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
        Re-shrink this task's covariance toward pooled covariance (Bug A fix).

        After a new task joins and Σ_pool is updated, ALL previous tasks
        should be re-shrunk using the updated pool (Theorem 5, Step 3).

        Σ̃_t ← (1 - α_t*) · Σ̂_t (raw) + α_t* · Σ̂_pool

        IMPORTANT: always use Sigma_raw (the raw sample covariance) as the base,
        not self.Sigma (which may have already been shrunk in a previous round).
        """
        self.alpha = alpha
        self.Sigma = (1 - alpha) * self.Sigma_raw + alpha * Sigma_pool
        self.Sigma = self.Sigma.astype(np.float64)
        # Invalidate cached decompositions
        self._eigvals = None
        self._eigvecs = None
        self._Sinv = None
        self._par = None

    def distance(self, h: np.ndarray) -> np.ndarray:
        """
        Compute distance from h to this task's centroid using this task's metric.

        Args:
            h: (n, d) or (d,) — embeddings

        Returns:
            distances: (n,) or scalar
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
            # Fallback: auto-select by PaR
            if self.par >= 0.9 * self.d:
                return metric_l2(h, self.mu)
            elif self.par >= 0.3 * self.d:
                return metric_mahalanobis(h, self.mu, self.Sinv)
            else:
                k = max(1, int(np.sum(self.eigvals > 1e-6 * self.eigvals[0])))
                return metric_psr(h, self.mu, self.eigvecs[:, :k], self.eigvals[:k], self.d)

    def to_dict(self) -> dict:
        """Serialize to dict (for saving)."""
        return {
            'task_id': self.task_id,
            'mu': self.mu,
            'Sigma': self.Sigma,
            'Sigma_raw': self.Sigma_raw,
            'n': self.n,
            'metric': self.metric,
            'alpha': self.alpha,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'TaskSignature':
        sig = cls(
            task_id=d['task_id'],
            mu=d['mu'],
            Sigma=d['Sigma'],
            n=d['n'],
            metric=d.get('metric', 'auto'),
            alpha=d.get('alpha', 0.0),
            Sigma_raw=d.get('Sigma_raw', d['Sigma']),
        )
        return sig


# ─────────────────────────────────────────────────────────────────────────────
#  SRT ROUTER
# ─────────────────────────────────────────────────────────────────────────────

class SRTRouter:
    """
    Statistical Routing Theory Router.

    At training time: compute and store {μ_t, Σ_t} for each task.
    At inference time: route input h to nearest task by SRT metric.

    Zero-drift: no learnable parameters → no interference with future tasks.

    Integration:
        trainer.train(task t) → _compute_and_store_signature()
        → model.forward() → _srt_route() → hard task IDs
    """

    def __init__(self, use_srm: bool = False):
        self.signatures: Dict[int, TaskSignature] = {}
        self._mu_pool: Optional[np.ndarray] = None
        self._Sigma_pool: Optional[np.ndarray] = None
        self._n_pool: int = 0
        self.use_srm = use_srm
        self._srm_metrics: Dict[int, str] = {}

    # ── Pooled statistics ───────────────────────────────────────────────

    def _update_pooled(
        self,
        mu_t: np.ndarray,
        Sigma_t: np.ndarray,
        n_t: int,
    ):
        """Update running pooled mean and covariance using Welford-Hart."""
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
        """
        Re-shrink ALL previous tasks toward updated pooled covariance.

        Bug 7 fix: After pool update, re-apply pooled shrinkage to all tasks.
        Theorem 5, Step 3: for each task s < t, recompute α_s* and shrink Σ̃_s.
        """
        if self._n_pool <= 1:
            return

        for t_id, sig in self.signatures.items():
            # Compute optimal α_t* = n_pool / (n_pool + n_t)
            alpha_opt = self._n_pool / (self._n_pool + sig.n)
            alpha_opt = max(0.01, min(0.99, alpha_opt))
            # Re-shrink toward new pool
            sig.reshrink(self._Sigma_pool, self._n_pool, alpha_opt)

    # ── Add task ─────────────────────────────────────────────────────

    def add_task(
        self,
        task_id: Union[int, str],
        h_train: np.ndarray,
        use_shrink: bool = True,
        shrink_factor: float = 0.1,
    ) -> TaskSignature:
        """
        Add a new task's statistical signature.

        Steps (per C1 §3.1):
          1. Compute sufficient statistics μ_t, Σ̂_t from h_train
          2. Update pooled statistics
          3. Re-shrink ALL tasks (Bug 7 fix)
          4. Select metric per SRM (Bug 4 fix)

        Args:
            task_id: integer task ID
            h_train: (n_t, d) embeddings from frozen backbone

        Returns:
            TaskSignature for this task
        """
        n_t, d = h_train.shape

        # Step 1: sufficient statistics
        mu_t = h_train.mean(axis=0)
        Sigma_t = np.cov(h_train, rowvar=False, ddof=1)

        # Step 1b: Ledoit-Wolf shrinkage — ALWAYS when use_shrink=True.
        # BUG E fix: shrinkage is MOST needed when n_t ≤ d (ill-conditioned Σ).
        # When n_t >> d the sample covariance is already well-conditioned.
        if use_shrink:
            Sigma_t = ledoit_wolf_shrinkage(Sigma_t, factor=shrink_factor)

        # Step 2: update pooled
        self._update_pooled(mu_t, Sigma_t, n_t)

        # Step 3: re-shrink ALL tasks (Bug 7 fix)
        if len(self.signatures) > 0:
            self._reshrink_all()

        # Step 4: SRM metric selection (Bug 4 fix)
        # Run SRM once when pool is established (≥3 tasks for meaningful SRM)
        if self.use_srm and len(self.signatures) >= 2:
            srm_results = srm_metric_selection(self.signatures, self._Sigma_pool)
            self._srm_metrics.update(srm_results)
            # Propagate SRM results to existing signatures
            for t_id, m in srm_results.items():
                if t_id in self.signatures:
                    self.signatures[t_id]._metric = m
        elif len(self.signatures) == 0:
            self._srm_metrics[task_id] = 'auto'

        # Determine metric for this task
        metric = self._srm_metrics.get(task_id, 'auto')
        if metric == 'auto':
            # Use PaR-based default
            par = participation_ratio(Sigma_t)
            if par >= 0.9 * d:
                metric = 'l2'
            elif par >= 0.3 * d:
                metric = 'mahalanobis'
            else:
                metric = 'psr'

        sig = TaskSignature(task_id, mu_t, Sigma_t, n_t, metric=metric)
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

        return nearest_task, dists

    # ── Save / Load ───────────────────────────────────────────────────

    def save(self, path: str):
        """Save all signatures to disk (zero-rehearsal compliant)."""
        sigs_data = {k: v.to_dict() for k, v in self.signatures.items()}
        np.savez_compressed(
            path,
            signatures=sigs_data,
            mu_pool=self._mu_pool if self._mu_pool is not None else np.zeros(1),
            Sigma_pool=self._Sigma_pool if self._Sigma_pool is not None else np.zeros((1, 1)),
            n_pool=np.array([self._n_pool]),
            srm_metrics=self._srm_metrics,
        )

    def load(self, path: str):
        """Load signatures from disk."""
        data = np.load(path, allow_pickle=True)
        self._mu_pool = data['mu_pool']
        if self._mu_pool.shape[0] == 1 and self._mu_pool[0] == 0:
            self._mu_pool = None
        self._Sigma_pool = data['Sigma_pool']
        if self._Sigma_pool.shape[0] == 1:
            self._Sigma_pool = None
        self._n_pool = int(data['n_pool'][0])
        self._srm_metrics = dict(data['srm_metrics'].item())

        for k, v in data['signatures'].item().items():
            self.signatures[k] = TaskSignature.from_dict(v)

    # ── Summary ───────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Return routing statistics."""
        task_list = sorted(self.signatures.keys())
        metrics = [self.signatures[t].metric for t in task_list]
        pars = [self.signatures[t].par for t in task_list]
        alphas = [self.signatures[t].alpha for t in task_list]

        return {
            'n_tasks': len(self.signatures),
            'task_ids': task_list,
            'metrics': metrics,
            'pars': [f"{p:.1f}" for p in pars],
            'avg_par': float(np.mean(pars)) if pars else 0.0,
            'alphas': alphas,
            'pool_n': self._n_pool,
        }
