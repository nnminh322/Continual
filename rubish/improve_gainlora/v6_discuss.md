# Revised Implementation Plan — After Strict Zero-Replay Constraint

> **Trigger**: User confirmed storing mean embeddings (prototypes) violates zero-replay: *"lưu bất kỳ thứ gì của dữ liệu đều vi phạm"*

---

## I. CONSTRAINT ANALYSIS — What's Available?

Under strict zero-replay, at test time we have **ONLY**:

| Information | Source | Available? |
|------------|--------|-----------|
| Frozen LoRA weights $A_t, B_t$ | Model training artifact | ✅ |
| SVD of $\Delta W_t = B_t A_t$ | Derived from model params | ✅ |
| Current input $h$ | Test sample | ✅ |
| GPM bases | ROOT method (forward on data) | ✅ Already stored |
| **Mean embeddings** $\mu_t$ | **Data statistic** | ❌ **VIOLATES** |
| **Distribution params** | **Data statistic** | ❌ **VIOLATES** |
| **Learned routing params** | Model params | ⚠️ Legal but has forgetting risk |

### Implication

Prototype routing (V5) is **invalid**. Available routing must use only:
$$\text{routing}(h) = f(h; \{A_t, B_t\}_{t=1}^T)$$

This means spectral routing is the **ONLY** parameter-free, zero-replay-compliant routing mechanism.

---

## II. REVISITING THE GPM-ROUTING PARADOX

The paradox: GPM forces $A_{k'} \perp A_k$, so spectral fit favors the first expert that claimed shared input directions.

**But is this paradox insurmountable?** Let me re-examine:

### Severity depends on expert quality

If expert $t$'s LoRA is **well-trained** (captures task-specific information in its 4D subspace), then even though its subspace is orthogonal to earlier experts, the SVD singular values $\sigma_t$ encode **how strongly** the expert responds to different directions. Two same-domain experts with orthogonal $V_t$ can still be discriminated IF:

$$\sigma_t^2 (v_t^T h)^2 \gg \sigma_{t'}^2 (v_{t'}^T h)^2$$

This happens when:
1. Expert $t$ has **large** singular values → strong response in its subspace
2. Input $h$ projects **significantly** onto expert $t$'s orthogonal directions

### C4 directly addresses this

**Preconditioned Gradients** (gradient preconditioning via $(AA^T + \epsilon I)^{-1/2}$):
- Stabilizes training in the constrained (null-space projected) subspace
- Expert learns more effectively **within** its allocated subspace
- → Better singular value spectrum → more discriminative spectral signatures

**Spectral Entropy Regularization** ($\lambda \cdot (H_{max} - H(\hat{\sigma}))$):
- Encourages LoRA to utilize **full rank** (all 8 dimensions, not just 1-2)
- More spread singular values → expert responds to more directions in its subspace
- → Higher projection energy for correct inputs → better routing discrimination

### Hypothesis: C4 is the key to making spectral routing work

The V1-V3 failures may not be purely due to the routing MECHANISM, but also due to **poor expert quality** — LoRA trained in constrained null-space without preconditioning learns poorly, producing weak/degenerate singular values → spectral routing becomes random.

**C4 fixes the expert quality → spectral routing becomes discriminative → performance improves.**

---

## III. PROPOSED APPROACH — Enhanced Spectral Routing + C4

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Routing | Spectral (SVD projection fit) | Only zero-replay-compliant option |
| Training bias | Adaptive β(n) | Handle cold-start, softmax dilution |
| Inference routing | Symmetric SVD (V3) | No bias, all tasks use same formula |
| C4: Preconditioning | **Enabled** | Key to making null-space LoRA learn effectively |
| C4: Spectral Entropy | **Enabled** (λ=0.01) | Full-rank LoRA → more discriminative signatures |
| Protection | GPM on LoRA-A | Unchanged from ROOT |

### Key insight

**V4 crashed due to bugs, not because C4 is wrong.** The trainer code NOW has the fix:
- [precompute_preconditioners()](file:///Users/nnminh322/Desktop/personal/Continual/improve_gainlora/src/cl_trainer_specroute.py#122-139) uses proper `torch.linalg.eigh()` + clamping
- [_compute_spectral_entropy_loss()](file:///Users/nnminh322/Desktop/personal/Continual/improve_gainlora/src/cl_trainer_specroute.py#140-164) uses QR trick (avoids large SVD)
- [_apply_preconditioning()](file:///Users/nnminh322/Desktop/personal/Continual/improve_gainlora/src/cl_trainer_specroute.py#165-178) includes `nan_to_num_()` guard

### What changed from V5 script

The V5 script ALREADY has C4 enabled (`lambda_entropy=0.01`, `use_preconditioning=True`). But V5 also has prototype routing at inference. We need:
1. **Keep C4 enabled** (same as V5)
2. **Disable prototype routing at inference** → fall back to symmetric SVD routing (V3)

---

## IV. IMPLEMENTATION

### [MODIFY] [t5_specroute.py](file:///Users/nnminh322/Desktop/personal/Continual/improve_gainlora/src/t5_specroute.py)

In [compute_spectral_routing()](file:///Users/nnminh322/Desktop/personal/Continual/improve_gainlora/src/t5_specroute.py#231-382), the inference path currently:
1. Tries prototype routing first (cosine similarity to stored prototypes)
2. Falls back to SVD spectral routing if prototypes unavailable

**Change**: Always use SVD spectral routing at inference (skip prototype check).

Specifically: remove or bypass the `_use_proto` path at inference, always go to the SVD-based path (the `else` branch at line 317).

### [MODIFY] Shell Script

Create V6 script: keep C4 enabled, output paths renamed to V6.
- `--lambda_entropy 0.01` ✅ (keep)
- `--use_preconditioning True` ✅ (keep)
- All paths renamed from v5 → v6

### No other code changes needed
- [run_t5.py](file:///Users/nnminh322/Desktop/personal/Continual/improve_gainlora/src/run_t5.py): prototype load code safely handles missing prototype files
- [cl_trainer_specroute.py](file:///Users/nnminh322/Desktop/personal/Continual/improve_gainlora/src/cl_trainer_specroute.py): C4 code already correct

---

## V. VERIFICATION PLAN

### Primary: Run full 15-task experiment
```bash
bash T5_small/gen_script_long_order3_t5_small_specroute_v6.sh <model_path>
python score.py gen_script_long_order3_t5_small_specroute_v6 <output_path>
```

### Diagnostic: Monitor routing quality
- Watch for `[SpecRoute]` log lines showing routing weight distributions
- Check if C4 entropy loss decreases (indicates LoRA using more of its rank)
- Check singular value spectra in `spectral_signatures.pt` files

### Expected outcomes
- **If C4 helps**: AP 45-55 (large improvement from V3's 33.77)
- **If C4 doesn't help**: AP ~33-35 (similar to V3) → spectral routing fundamentally limited under strict zero-replay
- **If C4 hurts** (unstable): Training loss NaN or spike → reduce λ_entropy or disable preconditioning

---

## User Review Required

> [!IMPORTANT]
> **Core question**: Given strict zero-replay, spectral routing is the ONLY zero-replay-compliant routing mechanism. V1-V3 showed it fails. The hypothesis is that C4 (preconditioned training + spectral entropy) will make experts learn better → spectral routing becomes discriminative. This is the last viable axis before we must accept learned routing (GainLoRA-style) as necessary.
>
> **Do you agree with proceeding with C4-enhanced spectral routing (no prototypes)? Or do you prefer a different direction?**
