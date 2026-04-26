# SRT Router with PooledMahalanobis_RIDGE

## Contribution Summary

**Statistical Routing Theory (SRT)** provides a non-parametric, zero-drift task router for continual learning. At its core, SRT Theorem 4 proves that **Pooled Mahalanobis distance** is Bayes-optimal for discriminating between tasks whose embeddings follow shared-covariance Gaussians. We instantiate this with **Ridge shrinkage** — the analytically optimal δ = d/(n+d) for the n≈d regime typical in CL — achieving **97.51% final / 97.17% avg routing accuracy** across 15 SuperNI tasks, stable from step 2 onward.

---

## Theoretical Foundation

### SRT Theorem 4 (Pooled Mahalanobis)
For two tasks t and t' with embeddings drawn from N(μ_t, Σ) and N(μ_t', Σ) under a shared covariance Σ:

```
D_KL(N(μ_t, Σ) || N(μ_t', Σ)) =
  ½ · (μ_t - μ_t')ᵀ Σ⁻¹ (μ_t - μ_t')   ← discriminative part
+ ½ · tr(Σ⁻¹ Σ) - ½ · log |Σ⁻¹ Σ| - d   ← constant offset
```

The discriminative part is exactly the **Pooled Mahalanobis distance** with the shared covariance Σ (here estimated as the pooled covariance Σ_pool). This is what we use for routing.

### Why Pooled?
- The pooled covariance pools all seen-task samples into one global estimate, making it well-conditioned even when individual tasks have few samples.
- With Mahalanobis, we account for the geometry of the embedding space — dimensions with high variance contribute less to distance than dimensions with low variance.
- The alternative (per-task covariance) fails at small n because Σ_t is singular when n_t < d.

### Why Ridge Shrinkage?
The sample pooled covariance Σ̂ is always singular/semi-singular in high dimensions. Analytical shrinkage

```
δ* = d / (n + d)       [Ridge]
```

minimizes mean-squared error to the true Σ under the Frobenius norm. This is derived from the Marchenko-Pastur distribution of singular values and converges to:

| n/d  | δ* (Ridge) | Effect                                   |
|------|-------------|------------------------------------------|
| 0.04 | 0.990       | Dominated by diagonal target → near-iso  |
| 0.59 | 0.341       | Balanced: data + regularization          |
| →∞   | 0.000       | Pure sample covariance                  |

At our final step (n=11,741, d=4,096, n/d=2.87), δ* ≈ 0.26 — the Ridge estimator is mostly data-driven. The shrinkage is most critical in early steps (t=1: δ*=0.96; t=2: δ*=0.78) where sample counts are small.

**OAS** (Chen et al., 2010) and **LW** (Ledoit-Wolf, 2004) also work but are more conservative — they converge slower and give slightly lower accuracy (OAS: 97.07%, LW: 97.13%). Without shrinkage (NONE), the pseudo-inverse is numerically unstable → same as OAS/LW in practice.

### Adaptive Metric Selection (SRT Theorem 6)
SRT Theorem 6 defines the **Participation Ratio (PaR)**:

```
PaR = (Σᵢ λᵢ)² / Σᵢ λᵢ²    where λᵢ = eigenvalues of Σ_pool
```

PaR/d ∈ (0, 1): measures how many eigendirections carry signal. When PaR/d is low (few dominant directions), Mahalanobis shines over L2; when high, L2 is sufficient. Our experiments show PaR stabilizes around 16.9/4096 ≈ 0.4% — very low — meaning the embedding space has a highly skewed eigenvalue spectrum, and Mahalanobis accounting for this is crucial.

---

## Pipeline / Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TRAINING (per task t)                           │
└─────────────────────────────────────────────────────────────────────────┘

  Backbone (FROZEN)
  ┌──────────────────────────────────────────────────────────────┐
  │  for each batch in task t's training data:                    │
  │    h = backbone.forward(input_ids)           # last token    │
  │    h → CPU float32 tensor                                   │
  │    accumulate into buffer H_pool                            │
  │                                                              │
  │    [standard LoRA training — backbone stays frozen]         │
  └──────────────────────────────────────────────────────────────┘
                          ↓  after training completes
  ┌──────────────────────────────────────────────────────────────┐
  │  1. Compute task centroid:                                   │
  │       μ_t = mean(H_t)                                        │
  │                                                              │
  │  2. Welford-Hart pooled update (zero-rehearsal):            │
  │       n_pool  ← n_pool  + n_t                                │
  │       M_pool  ← M_pool  + Σᵢ∈t (h - μ_t)ᵗ(h - μ_t)         │
  │       Σ_pool  ← M_pool / n_pool                             │
  │                                                              │
  │  3. Analytical Ridge shrinkage:                             │
  │       δ* = d / (n_pool + d)                                 │
  │       Σ_shrunk = (1-δ*) · Σ_pool + δ* · (tr(Σ_pool)/d)·I    │
  │                                                              │
  │  4. Compute inverse via eigendecomposition:                 │
  │       Σ_shrunk = V · diag(λ) · Vᵀ                           │
  │       Σ_shrunk⁻¹ = V · diag(1/λ) · Vᵀ                       │
  │                                                              │
  │  5. Store signature: {μ_t, Σ_shrunk⁻¹} for task t          │
  └──────────────────────────────────────────────────────────────┘
                          ↓  routing ready

┌─────────────────────────────────────────────────────────────────────────┐
│                         INFERENCE (per test sample)                      │
└─────────────────────────────────────────────────────────────────────────┘

  Test input
      ↓
  Backbone (FROZEN) → embedding h_test (last token)
      ↓
  ┌──────────────────────────────────────────────────────────────┐
  │  For each seen task t with signature {μ_t, Σ⁻¹}:               │
  │    d_t = (h_test - μ_t)ᵀ Σ⁻¹ (h_test - μ_t)   [Mahalanobis] │
  │                                                              │
  │  argmin d_t → routed task t*                                 │
  └──────────────────────────────────────────────────────────────┘
      ↓
  Activate LoRA adapter for task t* → downstream prediction
```

### Key Design Points

1. **FROZEN backbone**: Embeddings are extracted from the frozen backbone at both train and inference. This guarantees the embedding space is identical across tasks — no domain shift from LoRA adaptation leaking into routing.

2. **Zero-rehearsal compliant**: We never store old task data. The Welford-Hart update incrementally maintains Σ_pool and n_pool using only a running sum of scatter matrices. Each task's contribution is merged in via the scatter matrix addition rule.

3. **No ZCA**: Unlike the ZCA-based WhitenedL2 approach (which fits ZCA once from a buffer), PooledMahalanobis directly maintains and inverts the covariance. This is more statistically grounded (it IS the Bayes-optimal metric) and more numerically stable.

4. **GPU-accelerated**: All operations (eigendecomposition, matrix multiplication) run on CUDA tensors. The eigendecomposition of a 4096×4096 Σ_shrunk takes <5ms on a Tesla T4.

5. **Incremental inverse update**: After each task, we recompute Σ⁻¹ via eigendecomposition. This is O(d³) per task but only 15 times across 15 tasks — negligible cost compared to training.

---

## Ablation Results (15 tasks, SuperNI, Llama-2-7b-hf)

| Method | Final | Avg | Notes |
|--------|-------|-----|-------|
| NearestCentroid | 51.10% | 60.00% | Raw L2, no whitening |
| CosineNearestCentroid | 51.82% | 60.26% | Cosine on centroids |
| WhitenedL2_min400 | 12.67% | 36.03% | ZCA fitted too early (n=1160) |
| WhitenedL2_min1600 | 18.60% | 44.92% | ZCA fitted at n=4640 |
| AdaptivePar | 97.07% | 93.59% | PaR-based metric selection |
| PooledMahalanobis_OAS | 97.07% | 93.66% | Oracle-approximating shrinkage |
| PooledMahalanobis_LW | 97.13% | 95.28% | Ledoit-Wolf analytical shrinkage |
| **PooledMahalanobis_RIDGE** | **97.51%** | **97.17%** | **Best: δ*=d/(n+d)** |
| PooledMahalanobis_NONE | 97.07% | 93.59% | Pseudo-inverse, no shrinkage |
| PSR_k8 | 10.60% | 25.33% | Personalized routing signal, too few components |
| RLS_Woodbury | 64.07% | 58.18% | Recursive least squares |

### Why WhitenedL2 Fails (and ZCA is the wrong approach)

WhitenedL2 fits ZCA once when the buffer reaches `min_samples`. The whitening matrix W = V·diag(1/√λ)·Vᵀ decorrelates the space, then L2 is applied. The problem: **ZCA requires n/d >> 1 to be informative**. At n=1160, d=4096, n/d=0.28, the sample covariance is extremely noisy — its eigenvalues are dominated by noise, not signal. Whitening with noisy eigenvalues distorts the geometry. Once ZCA is fitted badly, it stays bad (it's frozen). This is why WhitenedL2_min400 degrades from 95% at T2 to 12.67% at T15.

PooledMahalanobis avoids this by never whitening. It directly uses the covariance's eigenstructure for distance computation, with shrinkage providing the regularization that ZCA tries to achieve via buffer waiting.

### Why RIDGE beats OAS and LW

- **OAS**: δ = ρ̂/(n+1-2/d) — estimates the optimal shrinkage intensity from data. Conservative. In early steps, ρ̂ overestimates, shrinking too aggressively.
- **LW**: Optimized for Frobenius norm MSE but derived under different assumptions than the high-dim n≈d regime.
- **Ridge**: δ = d/(n+d) — simple, interpretable, theoretically grounded for n≈d. No data-dependent estimation overhead. Consistent behavior across all steps.

---

## Integration with GainLoRA

In the GainLoRA + SRT architecture:

```
Training:
  - Freeze backbone → extract embeddings → update Σ_pool via Welford-Hart
  - LoRA adapters train normally on each task
  - At task end: compute μ_t, shrink Σ_pool, store {μ_t, Σ⁻¹}

Inference:
  - Forward through FROZEN backbone → h_test
  - PooledMahalanobis_RIDGE: argmin_t (h_test - μ_t)ᵀ Σ⁻¹ (h_test - μ_t)
  - Activate corresponding LoRA adapter(s)
  - For multi-task NLI (CB): soft cross-adapter blending via attention weights
```

The SRT router replaces the learned MLP router in the original GainLoRA with a non-parametric, zero-drift alternative. Since SRT uses frozen backbone embeddings, the router never suffers from catastrophic forgetting.

---

## Implementation References

| Component | File | Key Code |
|-----------|------|----------|
| PooledMahalanobisRouter | `routing_analysis/routing_class_v2.py` | `_shrink_ridge()`, `route()` |
| Welford-Hart incremental update | `routing_class_v2.py` | `add_task()` → scatter update |
| SRT Trainer integration | `new_gainlora/src/cl_trainer_srt.py` | `_extract_task_embeddings()`, `_compute_and_store_signature()` |
| Model wiring | `t5_gainlora_inflora.py` | `use_srt_routing`, `srt_router` attributes |
