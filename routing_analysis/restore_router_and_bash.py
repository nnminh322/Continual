#!/usr/bin/env python3
"""Re-apply srt_router.py PCA changes + bash script SRT_PCA_COMPONENTS."""

# ──────────────────────────────────────────────────────────────────────────────
# 1. Update bash script
# ──────────────────────────────────────────────────────────────────────────────
bash_path = "/Users/nhatminhnguyen/Library/Mobile Documents/com~apple~CloudDocs/Thạc sĩ/Continual/new_llama_gainlora/run_superni_order1_llama_cl.sh"

with open(bash_path, "r") as f:
    content = f.read()

# Add SRT_PCA_COMPONENTS env var + bash arg
if "SRT_PCA_COMPONENTS=" not in content:
    old_sample = 'SRT_MAX_EMB_SAMPLES=${SRT_MAX_EMB_SAMPLES:-2000}  # increased'
    if old_sample in content:
        content = content.replace(
            old_sample,
            old_sample + '\nSRT_PCA_COMPONENTS=${SRT_PCA_COMPONENTS:-}  # PCA dims (e.g. 128); empty = no PCA'
        )
        print("✓ bash: SRT_PCA_COMPONENTS env var added")
    else:
        print("✗ bash: SRT_MAX_EMB_SAMPLES line not found")

if "--srt_pca_components" not in content:
    old_args = '        --srt_max_emb_samples    ${SRT_MAX_EMB_SAMPLES} \\\\'
    if old_args in content:
        content = content.replace(
            old_args,
            '        --srt_max_emb_samples    ${SRT_MAX_EMB_SAMPLES} \\\\\n        --srt_pca_components    ${SRT_PCA_COMPONENTS:-} \\'
        )
        print("✓ bash: --srt_pca_components arg added")
    else:
        print("✗ bash: --srt_max_emb_samples arg not found")

with open(bash_path, "w") as f:
    f.write(content)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Add PCA to srt_router.py
# ──────────────────────────────────────────────────────────────────────────────
router_path = "/Users/nhatminhnguyen/Library/Mobile Documents/com~apple~CloudDocs/Thạc sĩ/Continual/new_gainlora/src/srt_router.py"

with open(router_path, "r") as f:
    content = f.read()

if "pca_components" in content:
    print("  srt_router.py: PCA already present, skipping")
else:
    # Add PCA class attributes
    old_attrs = '        # Cached eigenvalues for summary\n        self._eigvals: Optional[np.ndarray] = None'
    new_attrs = '''        # PCA dimensionality reduction (if pca_components < d)
        self.pca_components: Optional[int] = pca_components
        self._pca_mean: Optional[torch.Tensor] = None
        self._pca_V: Optional[torch.Tensor] = None  # (d, pca_dim)

        # Cached eigenvalues for summary
        self._eigvals: Optional[np.ndarray] = None'''
    content = content.replace(old_attrs, new_attrs)

    # Add _fit_pca and _apply_pca methods
    old_add_task = '    def add_task('
    pca_methods = '''    def _fit_pca(self, X: torch.Tensor) -> torch.Tensor:
        """Fit PCA: project (n, d) → (n, pca_dim). Stores _pca_mean + _pca_V."""
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
        """Project X using fitted PCA."""
        if self._pca_V is None:
            return X
        return (X - self._pca_mean.to(X.dtype)) @ self._pca_V.to(X.dtype)

    def add_task('''
    content = content.replace(old_add_task, pca_methods)

    # Update add_task body: PCA on first task
    old_body = '''        n_t, d = h_train.shape

        # Move to GPU for all computations
        X = torch.from_numpy(h_train.astype(np.float32)).to(self.dev)
        mu_t_t = X.mean(dim=0)
        Xc = X - mu_t_t
        Sigma_t_t = (Xc.T @ Xc) / max(n_t - 1, 1)'''
    new_body = '''        n_t, d_orig = h_train.shape

        # Move to GPU for all computations
        X = torch.from_numpy(h_train.astype(np.float32)).to(self.dev)

        # Fit or apply PCA
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
    old_route = '''        # Move to GPU
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

    def route_debug'''
    new_route = '''        # Move to GPU + apply PCA
        H_raw = torch.from_numpy(h.astype(np.float32)).to(self.dev)
        if self._pca_V is not None:
            H = self._apply_pca(H_raw)
        else:
            H = H_raw
        Sinv = self._Sigma_inv_t
        dists = np.zeros((n_sample, T), dtype=np.float64)

        for i, mu_t_np in enumerate(self.centroids):
            # Apply PCA to centroid too
            if self._pca_V is not None:
                mu_np = mu_t_np.astype(np.float32)
                if mu_np.ndim == 1:
                    mu_np = mu_np.reshape(1, -1)
                mu_t_t = torch.from_numpy(mu_np).to(self.dev)
                mu_t_t = self._apply_pca(mu_t_t)
            else:
                mu_t_t = torch.from_numpy(mu_t_np.astype(np.float32)).to(self.dev)
            diff = H - mu_t_t
            diff_Sinv = diff @ Sinv
            dists[:, i] = (diff * diff_Sinv).sum(dim=1).cpu().numpy()

        del H
        nearest_idx = np.argmin(dists, axis=1)
        task_ids_ordered = list(self._signatures_by_id.keys())
        nearest_task = np.array([task_ids_ordered[i] for i in nearest_idx])
        return nearest_task, dists

    def route_debug'''
    content = content.replace(old_route, new_route)

    # Update route_debug() to apply PCA
    old_routedebug = '''        # Move to GPU
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

        # Confidence'''
    new_routedebug = '''        # Move to GPU + apply PCA
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
                mu_t_t = torch.from_numpy(mu_np).to(self.dev)
                mu_t_t = self._apply_pca(mu_t_t)
            else:
                mu_t_t = torch.from_numpy(mu_t_np.astype(np.float32)).to(self.dev)
            diff = H - mu_t_t
            diff_Sinv = diff @ Sinv
            dists[:, i] = (diff * diff_Sinv).sum(dim=1).cpu().numpy()

        del H
        nearest_idx = np.argmin(dists, axis=1)
        task_ids_ordered = list(self._signatures_by_id.keys())
        nearest_task = np.array([task_ids_ordered[i] for i in nearest_idx])

        # Confidence'''
    content = content.replace(old_routedebug, new_routedebug)

    # Update save/load for PCA state
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
    old_wrapper = '''    def __init__(
        self,
        shrinkage: str = "ridge",
        device: Optional[str] = None,
    ):
        self._impl = PooledMahalanobisRouter(shrinkage=shrinkage, device=device)'''
    new_wrapper = '''    def __init__(
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
    print("✓ srt_router.py: PCA support added")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Add pca_components to sgwi_srt_trainer.py
# ──────────────────────────────────────────────────────────────────────────────
trainer_path = "/Users/nhatminhnguyen/Library/Mobile Documents/com~apple~CloudDocs/Thạc sĩ/Continual/new_llama_gainlora/src/sgwi_srt_trainer.py"

with open(trainer_path, "r") as f:
    content = f.read()

# Update __init__ signature
old_sig = '''        srt_shrinkage: str = "ridge",
        srt_max_emb_samples: int = 500,
        srt_load_path: Optional[str] = None,
        srt_skip_forward: bool = False,'''
new_sig = '''        srt_shrinkage: str = "ridge",
        srt_max_emb_samples: int = 2000,
        srt_load_path: Optional[str] = None,
        srt_skip_forward: bool = False,
        srt_pca_components: Optional[int] = None,'''
if old_sig in content:
    content = content.replace(old_sig, new_sig)
    print("✓ sgwi_srt_trainer.py: pca_components param added to __init__")
else:
    print("✗ sgwi_srt_trainer.py: __init__ sig not found")

old_attrs = '''        self.srt_shrinkage = srt_shrinkage
        self.srt_max_emb_samples = srt_max_emb_samples
        self.srt_load_path = srt_load_path
        self.srt_skip_forward = srt_skip_forward'''
new_attrs = '''        self.srt_shrinkage = srt_shrinkage
        self.srt_max_emb_samples = srt_max_emb_samples
        self.srt_load_path = srt_load_path
        self.srt_skip_forward = srt_skip_forward
        self.srt_pca_components = srt_pca_components'''
if old_attrs in content:
    content = content.replace(old_attrs, new_attrs)
    print("✓ sgwi_srt_trainer.py: pca_components stored")
else:
    print("✗ sgwi_srt_trainer.py: attr assignment not found")

old_init_router = '''        self.srt_router = SRTRouter(shrinkage=self.srt_shrinkage)'''
new_init_router = '''        self.srt_router = SRTRouter(
            shrinkage=self.srt_shrinkage,
            pca_components=self.srt_pca_components,
        )'''
if old_init_router in content:
    content = content.replace(old_init_router, new_init_router)
    print("✓ sgwi_srt_trainer.py: SRTRouter with pca_components")
else:
    print("✗ sgwi_srt_trainer.py: SRTRouter init not found")

with open(trainer_path, "w") as f:
    f.write(content)

print()
print("="*60)
print("All router + trainer + bash fixes re-applied")
print("="*60)
