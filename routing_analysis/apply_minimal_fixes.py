#!/usr/bin/env python3
"""Minimal fixes: router PCA + bash PCA flag."""

# ──────────────────────────────────────────────────────────────────────────────
# 1. Add PCA to srt_router.py
# ──────────────────────────────────────────────────────────────────────────────
router_path = "/Users/nhatminhnguyen/Library/Mobile Documents/com~apple~CloudDocs/Thạc sĩ/Continual/new_gainlora/src/srt_router.py"

with open(router_path, "r") as f:
    content = f.read()

# Add PCA class attributes after _delta
old_attrs = '        self._delta: float = 0.0\n\n        # Cached eigenvalues'
new_attrs = '''        self._delta: float = 0.0

        # PCA dimensionality reduction (if pca_components < d)
        self.pca_components: Optional[int] = pca_components
        self._pca_mean: Optional[torch.Tensor] = None
        self._pca_V: Optional[torch.Tensor] = None  # (d, k) principal directions

        # Cached eigenvalues'''
content = content.replace(old_attrs, new_attrs)

# Update PooledMahalanobisRouter.__init__ to accept pca_components
old_init = """    def __init__(
        self,
        shrinkage: str = "ridge",
        device: Optional[str] = None,
    ):
        assert shrinkage in _SHRINK_METHODS, f"Unknown shrinkage: {shrinkage}" """
new_init = """    def __init__(
        self,
        shrinkage: str = "ridge",
        device: Optional[str] = None,
        pca_components: Optional[int] = None,
    ):
        assert shrinkage in _SHRINK_METHODS, f"Unknown shrinkage: {shrinkage}" """
content = content.replace(old_init, new_init)

# Add _fit_pca and _apply_pca methods before add_task
old_add = '    def add_task(\n        self,\n        task_id: Union[int, str],\n        h_train: np.ndarray,\n    ) -> TaskSignature:'
pca_methods = '''    def _fit_pca(self, X: torch.Tensor) -> torch.Tensor:
        """Fit PCA on embeddings X (n, d) → (n, k). Stores _pca_mean + _pca_V."""
        d = X.shape[1]
        if self.pca_components is None or self.pca_components >= d:
            return X
        k = min(self.pca_components, d - 1)
        X_centered = X - X.mean(dim=0)
        cov = (X_centered.T @ X_centered) / max(X.shape[0] - 1, 1)
        try:
            U, S, Vt = torch.linalg.svd(cov, full_matrices=False)
        except Exception:
            eigvals, eigvecs = torch.linalg.eigh(cov)
            idx = torch.argsort(eigvals, descending=True)
            Vt = eigvecs[:, idx].T
        V = Vt[:k].T  # (d, k)
        self._pca_mean = X.mean(dim=0)
        self._pca_V = V.to(self.dev)
        self._eigvals = S[:k].cpu().numpy()
        print(f"  [PCA] d={d} → k={k}  var={float((S[:k].sum() / S.sum()).item()):.4f}")
        return X_centered @ V.to(X.dtype)

    def _apply_pca(self, X: torch.Tensor) -> torch.Tensor:
        """Project (n, d) → (n, k) using fitted PCA."""
        if self._pca_V is None:
            return X
        return (X - self._pca_mean.to(X.dtype)) @ self._pca_V.to(X.dtype)

    def add_task(
        self,
        task_id: Union[int, str],
        h_train: np.ndarray,
    ) -> TaskSignature:'''
content = content.replace(old_add, pca_methods)

# Update add_task body: PCA before stats
old_body = '''        n_t, d = h_train.shape

        # Move to GPU for all computations
        X = torch.from_numpy(h_train.astype(np.float32)).to(self.dev)
        mu_t_t = X.mean(dim=0)
        Xc = X - mu_t_t
        Sigma_t_t = (Xc.T @ Xc) / max(n_t - 1, 1)'''
new_body = '''        n_t, d_orig = h_train.shape

        # Move to GPU + apply PCA
        X = torch.from_numpy(h_train.astype(np.float32)).to(self.dev)
        if self._pca_V is None and self.pca_components is not None:
            X = self._fit_pca(X)
            d = X.shape[1]
        elif self._pca_V is not None:
            X = self._apply_pca(X)
            d = X.shape[1]
        else:
            d = d_orig

        mu_t_t = X.mean(dim=0)
        Xc = X - mu_t_t
        Sigma_t_t = (Xc.T @ Xc) / max(n_t - 1, 1)'''
content = content.replace(old_body, new_body)

# Update route() to apply PCA
old_route_gpu = '        # Move to GPU\n        H = torch.from_numpy(h.astype(np.float32)).to(self.dev)\n        Sinv = self._Sigma_inv_t\n        dists = np.zeros((n_sample, T), dtype=np.float64)\n\n        for i, mu_t_np in enumerate(self.centroids):\n            mu_t_t = torch.from_numpy(mu_t_np.astype(np.float32)).to(self.dev)\n            diff = H - mu_t_t\n            diff_Sinv = diff @ Sinv\n            dists[:, i] = (diff * diff_Sinv).sum(dim=1).cpu().numpy()'
new_route_gpu = '        # Apply PCA if fitted\n        H_raw = torch.from_numpy(h.astype(np.float32)).to(self.dev)\n        if self._pca_V is not None:\n            H = self._apply_pca(H_raw)\n        else:\n            H = H_raw\n        Sinv = self._Sigma_inv_t\n        dists = np.zeros((n_sample, T), dtype=np.float64)\n\n        for i, mu_t_np in enumerate(self.centroids):\n            if self._pca_V is not None:\n                mu_np = mu_t_np.astype(np.float32)\n                if mu_np.ndim == 1:\n                    mu_np = mu_np.reshape(1, -1)\n                mu_t_t = self._apply_pca(torch.from_numpy(mu_np).to(self.dev))\n            else:\n                mu_t_t = torch.from_numpy(mu_t_np.astype(np.float32)).to(self.dev)\n            diff = H - mu_t_t\n            diff_Sinv = diff @ Sinv\n            dists[:, i] = (diff * diff_Sinv).sum(dim=1).cpu().numpy()'
content = content.replace(old_route_gpu, new_route_gpu)

# Update route_debug() GPU section too
old_rdbg_gpu = '''        # Move to GPU
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

        # Confidence: ratio'''
new_rdbg_gpu = '''        # Apply PCA if fitted
        H_raw = torch.from_numpy(h.astype(np.float32)).to(self.dev)
        if self._pca_V is not None:
            H = self._apply_pca(H_raw)
        else:
            H = H_raw
        Sinv = self._Sigma_inv_t
        dists = np.zeros((n_sample, T), dtype=np.float64)

        for i, mu_t_np in enumerate(self.centroids):
            if self._pca_V is not None:
                mu_np = mu_t_np.astype(np.float32)
                if mu_np.ndim == 1:
                    mu_np = mu_np.reshape(1, -1)
                mu_t_t = self._apply_pca(torch.from_numpy(mu_np).to(self.dev))
            else:
                mu_t_t = torch.from_numpy(mu_t_np.astype(np.float32)).to(self.dev)
            diff = H - mu_t_t
            diff_Sinv = diff @ Sinv
            dists[:, i] = (diff * diff_Sinv).sum(dim=1).cpu().numpy()

        del H
        nearest_idx = np.argmin(dists, axis=1)
        task_ids_ordered = list(self._signatures_by_id.keys())
        nearest_task = np.array([task_ids_ordered[i] for i in nearest_idx])

        # Confidence: ratio'''
content = content.replace(old_rdbg_gpu, new_rdbg_gpu)

# Update save to include PCA
old_save = '''            task_ids=list(self.signatures.keys()),
        )
        print(f"  [SRT-SAVE]'''
new_save = '''            task_ids=list(self.signatures.keys()),
            pca_components=np.array([self.pca_components or -1]),
            pca_mean=self._pca_mean.cpu().numpy() if self._pca_mean is not None else np.array([]),
            pca_V=self._pca_V.cpu().numpy() if self._pca_V is not None else np.array([]),
        )
        print(f"  [SRT-SAVE]'''
content = content.replace(old_save, new_save)

# Update load to restore PCA
old_load_end = '''            self.n_tasks_list.append(sig.n)

        print(f"  [SRT-LOAD]'''
new_load_end = '''            self.n_tasks_list.append(sig.n)

        # Restore PCA state
        pca_arr = data.get("pca_components", np.array([-1]))
        self.pca_components = int(pca_arr[0]) if pca_arr.size > 0 else None
        if self.pca_components and self.pca_components > 0:
            pca_mean = data.get("pca_mean", np.array([]))
            pca_V = data.get("pca_V", np.array([]))
            if pca_mean.size > 0 and pca_V.size > 0:
                self._pca_mean = torch.from_numpy(pca_mean).float().to(self.dev)
                self._pca_V = torch.from_numpy(pca_V).float().to(self.dev)

        print(f"  [SRT-LOAD]'''
content = content.replace(old_load_end, new_load_end)

# Add pca_components to SRTRouter wrapper __init__
old_wrapper = '''class SRTRouter:
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
        self._impl = PooledMahalanobisRouter(shrinkage=shrinkage, device=device)'''
new_wrapper = '''class SRTRouter:
    """
    Legacy wrapper providing backward-compatible interface.

    Internally uses PooledMahalanobisRouter with Ridge shrinkage.
    Preserves save/load format for compatibility with existing checkpoints.

    Args:
        shrinkage: 'ridge' | 'oas' | 'lw' | 'none' (default: 'ridge')
        device: 'cuda' | 'cpu' (default: auto-detect)
        pca_components: reduce d→k before Mahalanobis (e.g. 128). Default: None (no PCA)
    """

    def __init__(
        self,
        shrinkage: str = "ridge",
        device: Optional[str] = None,
        pca_components: Optional[int] = None,
    ):
        self._impl = PooledMahalanobisRouter(
            shrinkage=shrinkage, device=device, pca_components=pca_components
        )'''
content = content.replace(old_wrapper, new_wrapper)

with open(router_path, "w") as f:
    f.write(content)
print("✓ srt_router.py: PCA added")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Update bash script: add SRT_PCA_COMPONENTS env var + arg
# ──────────────────────────────────────────────────────────────────────────────
bash_path = "/Users/nhatminhnguyen/Library/Mobile Documents/com~apple~CloudDocs/Thạc sĩ/Continual/new_llama_gainlora/run_superni_order1_llama_cl.sh"

with open(bash_path, "r") as f:
    content = f.read()

# Add env var
if "SRT_PCA_COMPONENTS=" not in content:
    old_env = 'SRT_MAX_EMB_SAMPLES=${SRT_MAX_EMB_SAMPLES:-2000}'
    if old_env in content:
        content = content.replace(
            old_env,
            'SRT_MAX_EMB_SAMPLES=${SRT_MAX_EMB_SAMPLES:-2000}\nSRT_PCA_COMPONENTS=${SRT_PCA_COMPONENTS:-}  # PCA dims (e.g. 128); empty = no PCA'
        )
        print("✓ bash: SRT_PCA_COMPONENTS env var added")
    else:
        print("✗ bash: SRT_MAX_EMB_SAMPLES line not found")

# Add bash arg
if "--srt_pca_components" not in content:
    old_arg = '        --srt_max_emb_samples    ${SRT_MAX_EMB_SAMPLES} \\\\'
    if old_arg in content:
        content = content.replace(
            old_arg,
            '        --srt_max_emb_samples    ${SRT_MAX_EMB_SAMPLES} \\\\\n        --srt_pca_components    ${SRT_PCA_COMPONENTS:-} \\'
        )
        print("✓ bash: --srt_pca_components arg added")
    else:
        print("✗ bash: --srt_max_emb_samples arg not found")

with open(bash_path, "w") as f:
    f.write(content)

print()
print("="*60)
print("Router + bash fixes applied")
print("="*60)
