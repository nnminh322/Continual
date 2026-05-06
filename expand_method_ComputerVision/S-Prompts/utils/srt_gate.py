import numpy as np


def safe_cov(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.shape[0] <= 1:
        return np.eye(embeddings.shape[1], dtype=np.float64)
    cov = np.cov(embeddings, rowvar=False, ddof=1)
    if np.ndim(cov) == 0:
        cov = np.eye(embeddings.shape[1], dtype=np.float64)
    if not np.all(np.isfinite(cov)):
        cov = np.eye(embeddings.shape[1], dtype=np.float64)
    return cov.astype(np.float64)


def welford_pooled_update(mu_old, cov_old, n_old, mu_new, cov_new, n_new):
    total = n_old + n_new
    if total <= 1:
        return mu_old.copy(), cov_old.copy(), n_old

    mu_pool = (n_old * mu_old + n_new * mu_new) / total
    delta = mu_new - mu_old
    cross = (n_old * n_new / total) * np.outer(delta, delta)
    cov_pool = ((n_old - 1) * cov_old + (n_new - 1) * cov_new + cross) / max(total - 1, 1)
    return mu_pool, cov_pool, total


def shrink_cov(cov: np.ndarray, n_pool: int, method: str) -> tuple[np.ndarray, float]:
    d = cov.shape[0]
    trace = float(np.trace(cov))
    target = (trace / max(d, 1)) * np.eye(d, dtype=np.float64)

    if method == "ridge":
        delta = d / (n_pool + d + 1e-12)
        return (1.0 - delta) * cov + delta * target, float(delta)

    if method == "oas":
        tr_s2 = float((cov ** 2).sum())
        rho_hat = tr_s2 / (trace ** 2 + 1e-12)
        delta = rho_hat / (n_pool + 1.0 - 2.0 / max(d, 1))
        delta = float(min(1.0, max(0.0, delta)))
        return (1.0 - delta) * cov + delta * target, delta

    if method == "none":
        return cov, 0.0

    raise ValueError("Unknown shrinkage method: {}".format(method))


def batch_mahalanobis_argmin(query: np.ndarray, centroids: np.ndarray, sinv: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    preds = []
    for start in range(0, query.shape[0], batch_size):
        batch = query[start:start + batch_size]
        dists = []
        for mu_t in centroids:
            diff = batch - mu_t[None, :]
            dists.append(np.einsum("bi,ij,bj->b", diff, sinv, diff, optimize=True))
        preds.append(np.argmin(np.stack(dists, axis=1), axis=1))
    return np.concatenate(preds, axis=0)


class PooledMahalanobisGate:
    def __init__(self, shrinkage: str = "ridge"):
        self.shrinkage = shrinkage
        self.centroids = []
        self.mu_pool = None
        self.cov_pool = None
        self.n_pool = 0
        self.sinv = None
        self.delta = 0.0

    def add_task(self, embeddings: np.ndarray):
        mu_t = embeddings.mean(axis=0).astype(np.float64)
        cov_t = safe_cov(embeddings)
        n_t = int(embeddings.shape[0])
        self.centroids.append(mu_t)

        if self.n_pool == 0:
            self.mu_pool = mu_t.copy()
            self.cov_pool = cov_t.copy()
            self.n_pool = n_t
        else:
            self.mu_pool, self.cov_pool, self.n_pool = welford_pooled_update(
                self.mu_pool, self.cov_pool, self.n_pool, mu_t, cov_t, n_t)

        shrunk_cov, self.delta = shrink_cov(self.cov_pool, self.n_pool, self.shrinkage)
        eigvals, eigvecs = np.linalg.eigh(shrunk_cov)
        scale = max(float(np.abs(eigvals).max()), 1e-8)
        eigvals = np.maximum(eigvals, 1e-8 * scale)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        self.sinv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T

    def route(self, query: np.ndarray) -> np.ndarray:
        if not self.centroids:
            raise RuntimeError("PooledMahalanobisGate has no fitted task statistics.")
        return batch_mahalanobis_argmin(query.astype(np.float64, copy=False), np.stack(self.centroids), self.sinv)
