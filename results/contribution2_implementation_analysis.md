# Contribution 2 (C4): Spectrally-Conditioned LoRA Training — Implementation Analysis

## 1. Summary

C4 addresses **single-task LoRA quality** — the second pillar of continual learning performance (alongside catastrophic forgetting). Even with perfect routing (C1/C3 SpecRoute) and null-space constraints (InfLoRA), each task's LoRA adapter can underperform because:
- **Gradient distortion**: B gradients are distorted by frozen A's non-orthogonal column space
- **Low effective rank**: CE loss alone doesn't encourage full utilization of LoRA's rank budget
- **Suboptimal A initialization**: InfLoRA projects A into null-space, but the resulting A may have poor spectral conditioning

C4 proposes two complementary fixes:
1. **Preconditioned gradient**: $(AA^T + \epsilon I)^{-1/2} \nabla_B$ corrects gradient distortion from frozen A
2. **Spectral entropy regularization**: Maximizes effective rank of $BA$ to fully utilize the rank budget

## 2. Preconditioned Gradient — Mathematical Foundation

### Problem: Gradient Distortion
In standard LoRA, the update $\Delta W = BA$ where A is frozen (InfLoRA constraint). The gradient of loss w.r.t. B is:
$$\nabla_B \mathcal{L} = \nabla_{\Delta W} \mathcal{L} \cdot A^T$$

When A's columns are non-orthogonal (typical after null-space projection), $A^T$ distorts the gradient direction. Directions aligned with A's dominant singular vectors get amplified, while directions aligned with small singular vectors get suppressed.

### Solution: Spectral Preconditioning
Apply $(AA^T + \epsilon I)^{-1/2}$ to B's gradient after backward:
$$\tilde{\nabla}_B = \nabla_B \mathcal{L} \cdot (AA^T + \epsilon I)^{-1/2}$$

This equalizes gradient magnitudes across all directions in A's column space, allowing B to learn uniformly across all rank dimensions.

### Implementation
```python
def precompute_preconditioners(self):
    for lora in [module.lora_q, module.lora_v]:
        A = lora.lora_A.data.float()          # [d_in, r]
        AAt = A.T @ A                          # [r, r]
        AAt += eps * I
        eigvals, eigvecs = torch.linalg.eigh(AAt)
        inv_sqrt = eigvecs @ diag(eigvals^{-0.5}) @ eigvecs^T
        store inv_sqrt for lora_B
```

**Key property**: Computed ONCE after `get_reg_matrix()` projects A into null-space. Since A is frozen during training, the preconditioner is constant — no per-step overhead.

**Compatibility with GPM**: GPM projects A into null-space ONCE before training starts. Preconditioning operates on B's gradients AFTER backward. These are completely independent operations on different parameters at different times.

## 3. Spectral Entropy Regularization — Mathematical Foundation

### Problem: Low Effective Rank
CE loss optimizes for task accuracy but doesn't care about the spectral structure of $BA$. In practice, $BA$ often has very low effective rank — most of the "learning budget" (rank r) is wasted on near-zero singular values.

### Solution: Maximize Spectral Entropy
Define the normalized singular values of $BA$:
$$\hat{\sigma}_i = \frac{\sigma_i(BA)}{\sum_j \sigma_j(BA)}$$

The spectral entropy is:
$$H = -\sum_i \hat{\sigma}_i \log \hat{\sigma}_i$$

Maximum entropy $H_{max} = \log(r)$ occurs when all singular values are equal (full rank utilization).

### Regularization Loss
$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \sum_{\ell} (H_{max} - H_\ell)$$

where the sum is over all LoRA layers $\ell$.

### Efficient QR Trick
Computing SVD of the full $BA$ matrix ($d_{out} \times d_{in}$) is expensive. Instead:
1. $Q_B, R_B = QR(B^T)$ → $R_B$ is $r \times r$
2. $Q_A, R_A = QR(A)$ → $R_A$ is $r \times r$
3. $\hat{\sigma} = \text{svdvals}(R_B \cdot R_A^T)$ → SVD of $r \times r$ matrix

This gives the same singular values as $BA$ but costs $O(r^3)$ instead of $O(d_{out} \cdot d_{in} \cdot r)$.

### Implementation
```python
def _compute_spectral_entropy_loss(self):
    for lora in [module.lora_q, module.lora_v]:
        B = lora.lora_B.float()    # [r, d_out]
        A = lora.lora_A.float()    # [d_in, r]
        _, R_B = torch.linalg.qr(B.T)      # R_B: [r, r]
        _, R_A = torch.linalg.qr(A)         # R_A: [r, r]
        sigma_hat = torch.linalg.svdvals(R_B @ R_A.T)  # [r]
        sigma_hat = sigma_hat / (sigma_hat.sum() + eps)
        ent = -(sigma_hat * log(sigma_hat + eps)).sum()
        loss += (log(r) - ent)
    return loss / count
```

## 4. Pipeline Integration

### Training Pipeline (per task):
```
1. Load model + previous LoRA weights
2. get_reg_matrix()           ← InfLoRA: project A into null-space
3. precompute_preconditioners() ← C4: compute (AA^T+εI)^{-1/2}
4. Training loop:
   a. Forward pass → CE loss
   b. If step >= warmup: compute spectral entropy loss
   c. total_loss = CE + λ * entropy_loss
   d. backward(total_loss)
   e. _apply_preconditioning()   ← C4: modify B gradients
   f. optimizer.step()
5. get_representation()       ← GPM: update subspace bases
6. Save model + GPM bases
```

### Key Integration Points:
- `precompute_preconditioners()` after `get_reg_matrix()` (A is frozen, preconditioner is constant)
- Entropy loss added BEFORE backward (part of computational graph)
- Preconditioning applied AFTER backward (direct gradient modification)
- Warmup ratio prevents entropy regularization from dominating early training

## 5. Synergy with Existing Contributions

| Component | C1 (Spectral Routing) | C3 (Inference Routing) | C4 (LoRA Quality) |
|-----------|----------------------|----------------------|-------------------|
| Target | Task selection | Inference accuracy | Single-task quality |
| Mechanism | SVD signatures | Symmetric routing | Precond + entropy |
| When | Forward pass | Inference time | Training time |
| Interacts with | C4 (better LoRA → better signatures) | C1 (routing probabilities) | C1 (better training → better routing) |

**Virtuous cycle**: C4 improves each LoRA's quality → spectral signatures become more distinctive → C1 routing becomes more accurate → less interference → better continual learning.

## 6. Hyperparameters

| Parameter | Default | Range | Role |
|-----------|---------|-------|------|
| `lambda_entropy` | 0.01 | [0.001, 0.1] | Weight of spectral entropy loss |
| `use_preconditioning` | True | {True, False} | Enable gradient preconditioning |
| `precond_eps` | 1e-6 | [1e-8, 1e-4] | Numerical stability for preconditioner |
| `entropy_warmup_ratio` | 0.1 | [0.0, 0.3] | Fraction of steps before enabling entropy |

## 7. Ablation Plan

| Experiment | Precond | Entropy | Purpose |
|------------|---------|---------|---------|
| V3 (baseline) | ✗ | ✗ | Current best |
| V4a | ✓ | ✗ | Isolate preconditioning effect |
| V4b | ✗ | ✓ | Isolate entropy effect |
| V4 (full) | ✓ | ✓ | Full C4 |
| V4-λ sweep | ✓ | ✓ (λ ∈ {0.001, 0.01, 0.1}) | Sensitivity analysis |

## 8. Risk Assessment

### Low Risk:
- Preconditioning is a well-established technique (natural gradient, K-FAC)
- Spectral entropy is a differentiable, smooth regularizer
- Both components are additive — easy to disable if harmful

### Medium Risk:
- Entropy regularization may conflict with task-specific spectral structure (some tasks may genuinely need low-rank updates)
- Preconditioner may be ill-conditioned if A has very small singular values (mitigated by ε)

### Mitigation:
- Warmup ratio delays entropy regularization until CE loss stabilizes
- ε in preconditioner prevents numerical issues
- Ablation plan isolates each component's effect

## 9. Theoretical Guarantees

### Preconditioned Gradient:
- If A has condition number κ, standard gradient descent on B has convergence rate O(κ²)
- With preconditioning, convergence rate improves to O(1) (condition-number-independent)
- This is equivalent to natural gradient descent in the B parameter space

### Spectral Entropy:
- Maximum entropy ⟺ all singular values equal ⟺ effective rank = r
- This maximizes the "information capacity" of the LoRA adapter
- Connected to matrix information theory: max-entropy distribution on singular values

## 10. Code Changes Summary

### Files Modified:
1. **`cl_trainer_specroute.py`**: +4 init params, +3 methods (`precompute_preconditioners`, `_compute_spectral_entropy_loss`, `_apply_preconditioning`), modified `training_step`
2. **`run_t5.py`**: +4 dataclass fields, updated SpecRoute_Trainer constructor, added `precompute_preconditioners()` call
3. **`gen_script_long_order3_t5_small_specroute_v4.sh`**: New V4 experiment script with C4 args

### Lines of Code: ~80 lines of new logic (excluding comments/docstrings)
### Dependencies: No new dependencies (uses only torch.linalg built-ins)
