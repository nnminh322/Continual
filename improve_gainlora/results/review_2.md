# Review Round 2 — V5 Deeper Analysis

**Reviewer role**: Objective reviewer (deeper architectural / edge case / theoretical analysis)  
**Scope**: Building on review_1 findings; focusing on issues review_1 missed

---

## 1. Prototype Discrimination Quality — **Key Risk**

### 1.1 Common-Word Domination Problem

The prototype $\mu_k = \frac{1}{N}\sum_i \bar{h}_i^{(k)}$ averages over ALL tokens including stop words (the, a, is, of, to). These high-frequency tokens have large aggregate weight in the mean, making prototypes for different tasks closer than expected.

**Quantitative estimate**: For a sentence of 80 tokens, ~40 are stop words (50% typical in English). The stop word embeddings are shared across ALL tasks → they push all prototypes toward a common center. The task-discriminative signal comes from:
1. **Content words** (~40 tokens): "movie", "restaurant", "hypothesis", etc.
2. **Instruction prefix** (~15-20 tokens): unique per task in CL benchmarks.

For tasks like yelp vs amazon (both review tasks), content words overlap significantly. Discrimination relies heavily on the instruction prefix.

**Severity**: Medium. The instruction prefix provides a constant, distinctive signal in every sample. But for tasks with very similar instructions (mnli vs rte, both NLI), discrimination may be weak.

### 1.2 Score Range and Temperature

Cosine similarities between task prototypes are likely **very high** (>0.9) because:
- All prototypes share common English embedding structure
- T5 embeddings are not unit-normalized; the common component dominates

With `attn_temperature = 0.01` and cosines in [0.92, 0.97]:
- Score range: [92, 97], max gap ≈ 5
- softmax([97, 95, 93, 92, ...]) → dominant task gets ~50-73% weight
- This is reasonable soft routing

But with cosines in [0.95, 0.96] (near-identical):
- Score range: [95, 96], gap ≈ 1
- softmax([96, 95.8, 95.5, ...]) → nearly uniform → routing fails

**Recommendation**: After first few tasks, LOG the cosine similarity matrix between all prototypes. If max gap < 2 (in temperature-scaled space), consider:
- Lower temperature (harder routing)
- Mean-centering prototypes: $\tilde{\mu}_k = \mu_k - \bar{\mu}$ to remove the common component
- Or TF-IDF weighting (but adds complexity)

---

## 2. Edge Cases Verified ✓

| Case | Handling | Correct? |
|------|----------|----------|
| First task (run_single=True) | Prototype accumulated, no routing, saved after training | ✓ |
| Task 2 with 1 old prototype | _n_expected=2, _protos=[current_running_mean, old_proto]=2 → prototype routing active | ✓ |
| Missing prototype files | _protos < _n_expected → spectral fallback | ✓ |
| All zeros attention_mask | mask_count.clamp(min=1) → avg=0 → routing still works (cos(0, μ)=0 for all) | ✓ |
| Gradient checkpointing | Prototype update at T5Stack.forward() top, not inside checkpointed blocks → runs once | ✓ |
| Decoder | is_decoder=True → routing block skipped, uses encoder's weights | ✓ |
| Mixed precision (fp16/bf16) | Prototype accumulated in float32 on CPU; cast at routing time | ✓ |
| Memory | Single (d_model,) tensor per task; cleared after finalize | ✓ |

---

## 3. Masked Mean Change — Impact Analysis

V5 changed `avg_inputs_embeds` from `.mean()` to `.sum()/mask_count`:
- V3: $h = \frac{1}{L}\sum_i m_i e_i$ (divided by total seq length including padding)
- V5: $h = \frac{\sum_i m_i e_i}{\sum_i m_i}$ (divided by non-padding count)

**Impact on routing**: All scoring functions (A-row fit, spectral fit, cosine sim) are scale-invariant (Rayleigh quotient / cosine). So the magnitude change doesn't affect routing scores.

**Impact on direction**: The direction changes because padding tokens contribute 0 to numerator but inflate denominator in V3. V5 gives the true mean direction. This is more correct.

**Impact on training routing vs V3**: The A-row fit during training uses the same `avg_inputs_embeds`. Since fit ∝ $\|A h\|^2 / \|h\|^2$, both the projection and normalization scale together. Net effect: negligible for same-padding batches, slight direction improvement for mixed-padding batches.

**Conclusion**: The change is beneficial and backward-compatible. ✓

---

## 4. Theoretical Gaps

### 4.1 No Formal Guarantee for Prototype Routing

The SPECROUTE_IDEA.md C2.1 section invokes LDA optimality under Gaussian mixture assumption. However:
- **Real embeddings are NOT Gaussian**: Token embedding means have complex geometry
- **Shared covariance assumption violated**: Different tasks have different variance structures (sentiment has high variance on affect dimensions; factual tasks have high variance on entity dimensions)
- **Bayes risk bound inapplicable**: Without equal covariance, cosine similarity is not Bayes-optimal

**However**: Prototype routing doesn't need to be Bayes-optimal — it just needs to outperform spectral routing (which has a provable failure mode for same-domain tasks). The bar is low given V3's AP=33.77.

### 4.2 No Analysis of Prototype Drift Across Training

The running mean is computed during task k's training. But training modifies lora_B, which affects:
- The training loss landscape → different gradient sizes → different effective contribution per sample?
  
No — the prototype uses frozen embeddings (inputs_embeds), which are independent of lora_B updates. The prototype is truly static w.r.t. model parameters. ✓

### 4.3 Info Leakage from Eval During Training

Bug fix B3 uses running mean as temporary prototype during training eval. The running mean is computed from TRAINING data → it reflects the training distribution. Eval data comes from a held-out split → slightly different distribution.

**Impact**: Minimal. The training and eval distributions for the same task are very close. The running mean converges to the true task mean quickly (after ~50 batches).

---

## 5. Architecture Consistency

### 5.1 Training vs Inference Routing — Consistent?

During training:
- w_cur ≈ 0.8 (adaptive β), w_old ≈ 0.2/N (spectral routing)
- ΔW = w_cur · B_cur · A_cur · h + Σ w_old_j · B_j · A_j · h

During inference:
- w_k = softmax(cos(h, μ_k)/τ) for ALL k
- ΔW = Σ w_k · B_k · A_k · h

The current task's contribution during training (80%) is much higher than during inference (maybe 30-60% depending on prototype discrimination). This means the model is trained with a routing distribution that's different from inference.

**Is this a problem?** In ROOT, the same asymmetry exists (learned routing at training time vs frozen routing at inference). It's standard in CL. The key is that the current task gets sufficient gradient signal during training (80% weight ensures this).

### 5.2 Order of Prototypes vs LoRA Weights

Loading order: `previous_lora_list_sig.reverse()` → oldest first.
- spectral_sigs[0] = oldest task
- task_protos[0] = oldest task  
- previous_lora_weights loaded in same order

In compute_spectral_routing, fits[0] = current task, fits[1:] = old tasks (oldest first).
In prototype routing, _protos[0] = current, _protos[1:] = task_protos (oldest first).

Both match. ✓

---

## 6. Performance Predictions

### 6.1 Tasks That Should Improve (routing fix)
- **imdb, sst2** (EM was 0): Different instruction + vocabulary from yelp/amazon → prototypes should separate → EM > 0
- **wic** (EM was 0): Word-in-context task, very different vocabulary from sentiment → should separate well
- **cb, rte** (EM was 0): NLI tasks, different from sentiment → moderate prototype separation
- **yelp, amazon** (EM ~36): Should maintain or improve with correct routing

### 6.2 Tasks That Might NOT Improve
- **mnli** (EM ~2): Multi-modal distribution (entailment/contradiction/neutral). Prototype averages over modes → weak signal. Also, mnli is trained late (task 4 in order 3) → still OK for prototype separation from earlier tasks.

### 6.3 Overall AP Projection
If imdb/sst2/wic go from 0 → ROOT-level (~50-70 range):
- AP gain ≈ (50+60+55) / 15 / 3 ≈ +11 pts (assuming ~55 avg EM for these 3)
- If cb/rte also recover: additional +5-8 pts
- Net AP: 33.77 + 11 + 6 ≈ **50-51 AP(EM)**
- Compared to ROOT=59.70: still ~9 pts gap (from overall representation quality)

C4 (preconditioning + entropy) could close another 3-5 pts if it improves single-task quality.

---

## 7. New Recommendations

### 7.1 (High) Consider Mean-Centering Prototypes
After all prototypes are collected, subtract the global mean:
$$\tilde{\mu}_k = \mu_k - \frac{1}{T}\sum_j \mu_j$$

This removes the common English embedding component and highlights task-discriminative differences. Implement in `finalize_prototype()` or at routing time.

**Caution**: This requires all prototypes to be available before routing, which conflicts with the streaming CL setup where we only have old prototypes + current. A compromise: subtract the running mean of old prototypes from both old and current prototypes.

### 7.2 (Medium) Log Cosine Similarity Matrix
Add diagnostic logging during final evaluation: compute and print the pairwise cosine similarity matrix between all prototypes. This reveals:
- Which tasks have similar prototypes (collision risk)
- Whether temperature is appropriate (check score gaps)

### 7.3 (Low) Consider Prototype + Spectral Ensemble
Instead of pure prototype routing, combine prototype cosine with spectral fit:
$$s_k(h) = \alpha \cdot \cos(h, \mu_k) + (1-\alpha) \cdot \text{spectral\_fit}(h; V_k, \sigma_k)$$

This handles both same-domain tasks (prototype helps) and orthogonal tasks (spectral helps). But adds a hyperparameter α — violates simplicity principle.

---

## 8. Overall Assessment (Round 2)

**Code quality**: Good after DEV Round 1+2 fixes. No remaining bugs found. All edge cases handled correctly.

**Main risk**: Prototype discrimination quality — specifically for tasks with similar instruction prefixes. The T5 CL benchmark tasks generally have distinctive instructions, so this should work in practice.

**Temperature**: `attn_temperature=0.01` maps cosine scores [0.9, 0.97] to [90, 97] → softmax gives reasonable routing. Acceptable without tuning, but logging recommended.

**Overall confidence**: 75% that V5 significantly improves AP over V3 (33.77). 50% that it reaches ROOT (59.70). The gap would be closed by better single-task quality (C4) and potentially mean-centering (R7.1).
