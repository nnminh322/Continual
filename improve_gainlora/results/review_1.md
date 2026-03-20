# Review Round 1 — V5 Prototype Routing

**Reviewer role**: Objective reviewer (code + theory + methodology)  
**Scope**: V5 implementation (t5_specroute.py, run_t5.py), SPECROUTE_IDEA.md theory, experiment design

---

## 1. Code Correctness

### 1.1 Bugs Found and Fixed (DEV Round 1)

| # | Severity | Description | Status |
|---|----------|-------------|--------|
| B1 | **Critical** | Prototype fields (`_current_prototype_sum`, etc.) only initialized under `if not run_single:`. First task (run_single=True) crashes with `AttributeError` when `_update_prototype()` is called in `forward()`. | ✅ Fixed — moved to `if not self.is_decoder:` block |
| B2 | **Critical** | `.squeeze(-1)` in cosine sim changes shape (B,1)→(B,), then division by h_norm (B,1) broadcasts to **(B,B)** instead of (B,1). Routing weights would be completely wrong. | ✅ Fixed — removed `.squeeze(-1)` |
| B3 | **Important** | During training eval, `_current_task_prototype` is None (not finalized) → prototype routing falls back to spectral routing → mid-training eval metrics don't reflect prototype quality. | ✅ Fixed — use running mean as temporary prototype |

### 1.2 Remaining Issues

| # | Severity | Description | Recommendation |
|---|----------|-------------|----------------|
| R1 | **Medium** | `_compute_spectral_entropy_loss()` uses `qr(B.T)` and `qr(A)` — mathematically incorrect QR trick. Should be `qr(B)` and `qr(A.T)` to match `_thin_svd_low_rank()`. Current code computes an *approximation* of true singular values (off by orthogonal rotation factor $W = Q_{B'}^T Q_{A'}$). | Fix: change `qr(B.T) → qr(B)`, `qr(A) → qr(A.T)` in entropy loss. Low priority since regularization is approximate by nature. |
| R2 | **Low** | Prototype accumulation moves data to CPU every forward pass (`.cpu()` in `_update_prototype`). Minimal overhead (~16KB per call) but adds GPU→CPU transfer per batch. | Acceptable. Alternative: accumulate on GPU, move once at finalize. Not worth optimizing unless profiling shows bottleneck. |
| R3 | **Low** | No direct test for prototype routing quality before running full 15-task experiment. | Consider: quick sanity check — after training 2 tasks, verify cos(h_task1, μ_task1) > cos(h_task1, μ_task2) on a few samples. |

### 1.3 Shape Analysis (Verified ✓)

```
_update_prototype:
  h_batch: (B, d_model) → .sum(dim=0) → (d_model,) ✓

compute_spectral_routing (inference, prototype branch):
  h_flat: (B, d_model)
  h_norm: (B, 1)
  proto p: (d_model,) → p.unsqueeze(-1): (d_model, 1)
  matmul(h_flat, p.unsqueeze(-1)): (B, 1)    ← correct after .squeeze(-1) removed
  / h_norm: (B, 1) / (B, 1) = (B, 1)        ← correct
  fits: list of (B, 1) tensors
  cat(fits, dim=1): (B, n_tasks) ✓
  softmax → (B, n_tasks) → unsqueeze(2) → (B, n_tasks, 1) ✓
```

---

## 2. Theoretical Soundness

### 2.1 GPM-Routing Paradox — **Sound ✓**

The paradox is well-formulated and mathematically valid:
- GPM forces $\mathcal{S}_k \perp \mathcal{S}_j$ → same-domain tasks lose access to dominant input directions
- Spectral affinity $\alpha_k(h) = \|V_k^T h\|^2 / \|h\|^2$ depends on LoRA subspace alignment
- For $h \sim P(h|\text{imdb})$ with imdb's A in yelp's null-space: $\alpha_{\text{imdb}}(h) \ll \alpha_{\text{yelp}}(h)$

**Critical validation**: ROOT uses the SAME GPM on LoRA-A and achieves AP=59.70. This confirms the issue is routing (spectral ≠ learned MLP), not orthogonality itself.

### 2.2 Prototype Routing — **Conditionally Sound**

**Strengths:**
1. **Embedding space is GPM-immune** — prototypes live outside LoRA subspace ✓
2. **Zero-replay, drift-free** — frozen embedding table, O(d) storage per task ✓  
3. **Task instruction prefix** provides strong discriminative signal (not discussed in theory but practically important)

**Weaknesses / Assumptions:**
1. **LDA optimality assumption**: Requires Gaussian mixture with shared covariance — real NLP data violates this. Prototype routing is a heuristic, not provably optimal.
2. **Same-vocabulary tasks**: yelp vs amazon (both product reviews) may have very similar $\mu$. Cosine similarity may not discriminate well between these.
3. **Multi-modal tasks**: Tasks like mnli have diverse input distributions (entailment/contradiction/neutral). The mean $\mu_{\text{mnli}}$ averages over modes → poor representative prototype.

### 2.3 Training-Inference Gap

**Concern**: Training uses A-row fit + bias (subspace signal), inference uses prototype cosine (vocabulary signal). These are fundamentally different routing mechanisms measuring different properties.

- During training: B learns under 80% routing weight to current LoRA (forced by adaptive β)
- During inference: prototype router selects expert based on vocabulary distribution
- The expert's B was optimized under forced routing, not under prototype-based routing

**Mitigation**: This gap also exists in ROOT (learned MLP vs frozen MLP at inference). It's inherent to soft-routing CL. The adaptive β ensures sufficient gradient flow regardless of routing mechanism.

---

## 3. Methodology Assessment

### 3.1 Alignment with Theory

| Theory claim | Implementation | Aligned? |
|---|---|---|
| GPM-immune routing | Prototype from frozen embeddings | ✓ |
| Same-domain discrimination | cosine(μ_k, h) captures vocabulary differences | ✓ (conditional on vocabulary divergence) |
| Zero-replay | Running mean, O(d) per task | ✓ |
| Drift-free | Frozen embedding table | ✓ |
| Dual-mode routing | Train=A-row+β, Inference=prototype | ✓ |
| C4 orthogonal to routing | Preconditioning + entropy unchanged | ✓ |

### 3.2 Missing Design Decisions

1. **Temperature calibration**: Cosine similarities are typically in [0.7, 0.99] range. With `attn_temperature=0.01`:
   - cos_diff = 0.03 → score_diff = 3.0 → reasonable routing
   - But if all tasks have cosines ≈ 0.95 ± 0.01: softmax([95, 94.9, 94.8, ...]) ≈ near-uniform
   - **No analysis of expected cosine separation** provided in SPECROUTE_IDEA.md

2. **No hard routing option**: The idea doc mentions $k^* = \arg\max_k \cos(h, \mu_k)$ as an option but it's not implemented. Hard routing might outperform soft routing for clearly separable tasks.

3. **No per-layer prototypes**: Prototypes are computed from input embeddings (layer 0). If later layers' representations are more discriminative, per-layer prototypes could improve routing.

---

## 4. Potential Failure Modes

### 4.1 Prototype Collision
**Scenario**: Tasks with very similar vocabulary distributions (yelp/amazon/imdb all sentiment).  
**Impact**: Routing errors → EM = 0.  
**Likelihood**: Medium for yelp↔amazon, Low for yelp↔imdb (movie vs restaurant vocabulary).  
**Mitigation**: Task instruction prefix in T5 provides additional discrimination.

### 4.2 Temperature Sensitivity
**Scenario**: `attn_temperature=0.01` is the same for spectral (training) and prototype (inference).  
**Impact**: Spectral fits ∈ [0, 0.5], cosine ∈ [0.7, 0.99]. The same temperature maps different score ranges.  
**Mitigation**: Prototype softmax only runs during inference (no gradient coupling). Wrong temperature only causes over-soft/over-hard routing, not gradient issues.

### 4.3 Cold-Start Prototype Quality
**Scenario**: During first few training steps, running mean is computed from very few samples.  
**Impact**: Early eval metrics may be noisy; prototype quality improves over training.  
**Likelihood**: Low risk — eval steps typically happen after enough batches.

### 4.4 Spectral Entropy Bug (Pre-existing)
**Issue**: `_compute_spectral_entropy_loss()` computes `svdvals(R_B' @ R_A'^T)` where the QR decomposition is applied to transposed matrices compared to `_thin_svd_low_rank()`.  
**Mathematical analysis**: Current code computes `svdvals(R_{B^T} @ R_A^T)` ≠ `svdvals(R_B @ R_{A^T})` (the correct singular values of BA).  
**Impact on V5**: The entropy regularizer still approximately encourages rank diversity, but the gradient direction is slightly wrong. This pre-dates V5 and affects V3/V4 equally.

---

## 5. Recommendations

### Priority: High
1. **Fix spectral entropy QR bug** (R1): Change `qr(B.T) → qr(B)` and `qr(A) → qr(A.T)` in `_compute_spectral_entropy_loss()`. This ensures mathematically exact singular values for regularization.

### Priority: Medium  
2. **Add temperature analysis**: After training 2-3 tasks, log the cosine similarity distribution between test samples and all prototypes. Verify discrimination margin. Adjust temperature if needed.
3. **Consider separate `inference_temperature`**: Allow prototype routing to use a different temperature than spectral routing during training.

### Priority: Low
4. **Prototype sanity check**: Before full 15-task run, do a 2-task pilot to verify cos(h, μ_correct) > cos(h, μ_wrong).
5. **Document instruction prefix advantage**: The theory section doesn't mention that CL benchmark tasks have unique instruction prefixes, which strongly benefits prototype routing.

---

## 6. Overall Assessment

**Verdict**: V5 design is **theoretically well-motivated** and **implementation is correct** (after DEV Round 1 fixes). The GPM-Routing Paradox is a genuine insight that explains V3's AP=33.77 failure mode. Prototype routing is a reasonable solution that decouples routing from the LoRA subspace.

**Main risk**: Prototype discrimination quality for same-vocabulary tasks. The approach may struggle with yelp↔amazon but should handle the critical failing tasks (imdb, sst2, wic, cb) well due to vocabulary divergence.

**Expected outcome**: AP(EM) improvement from 33.77 → 40-50 range, driven primarily by recovering the 6 failing tasks. Whether it exceeds ROOT's 59.70 depends on C4's effectiveness + prototype routing quality.

**Pre-existing issue worth fixing**: Spectral entropy QR bug (R1) — affects all versions, easy fix, mathematically correct.
