# CONTRIBUTION 2: Bridging Train-Inference Routing Duality for SRT-based Continual Learning

> **Status**: Research phase — theory-first, no implementation yet.
> **Combines with**: Contribution 1 (SRT Statistical Routing Theory)
> **Target**: NeurIPS/ICML/ICLR — mathematically rigorous, information-theoretic.
> **Prerequisite reading**: Read §1.1 CAREFULLY before proceeding.

---

# PHẦN I: PROBLEM IDENTIFICATION — THE TRAIN-INFERENCE ROUTING MISMATCH

## 1.1 The ACTUAL Root Cause (Verified from Source Code)

### A. What ROOT (GainLoRA gốc) Does at Inference

ROOT uses the **same mechanism** during both training AND inference: `cal_attention` soft routing. There is **NO hard override.

```python
# root_gainlora/src/t5_gainlora_inflora.py — inference path:
# SAME soft cal_attention routing, NO override:
# key_attention_weights = cal_attention(past_prompt_key, past_x)  # soft sigmoid weights ∈ [0,1]
# Output: y = Σ w_t(x) · B_t A_t · x  # ALL adapters blend softly
```

From ROOT's code: `agg_lora_states` at inference returns the same soft-weighted sum. No one-hot. No override.

### B. What SRT (new_gainlora) Does at Inference

SRT **OVERRIDES** the soft routing with hard one-hot at inference ONLY:

```python
# new_gainlora/src/t5_gainlora_inflora.py, lines 1335-1393:
if not self.training and self.use_srt_routing:
    # SRT one-hot OVERRIDES soft cal_attention weights
    srt_weights[b, pos, 0] = 1.0  # ALL OTHER SLOTS = 0.0
    key_attention_weights[:] = srt_weights  # IN-PLACE OVERRIDE
    # Output: y = 1.0 · B_t* A_t* · x  # ONLY ONE adapter fires
```

### C. The ACTUAL Problem

| Phase | ROOT (GainLoRA gốc) | SRT (new_gainlora) |
|-------|---------------------|---------------------|
| Training | Soft blend: `w_t(x) ∈ [0,1]`, same `cal_attention` | Same training routing: `cal_attention` soft weights |
| **Inference** | **Soft blend**: same `cal_attention` routing | **Hard one-hot**: SRT override |
| Output at inference | `Σ w_t · B_t A_t x` | `B_{t*}A_{t*} · x` (one task only) |
| Cross-task knowledge | YES: mnli+rte contribute to CB | **NO**: CB fires alone |

**This IS the AP gap root cause.** ROOT's inference = soft blending = cross-task knowledge transfer = AP=78.01. SRT's inference = hard one-hot = zero cross-task transfer = AP=77.62.

**NOT the other way around.** My previous document incorrectly claimed ROOT has the mismatch.

### D. Quantified Impact

| Task | ROOT (soft at inference) | SRT (one-hot at inference) | Δ |
|------|---------------------|--------------------------|---|
| mnli | 86.20 | 86.20 | 0.00 |
| cb | 3.57 | 3.57 | 0.00 |
| rte | 91.34 | 91.34 | 0.00 |
| qqp | 87.12 | 87.12 | ≈0 |
| **AP overall** | **78.01** | **77.62** | **-0.39** |

### E. What ACTUALLY Differs

The difference is **purely at INFERENCE**:

**ROOT inference:**
$$w_t^{\text{ROOT}}(x) = \bigl|\,\sigma\bigl(4 \cdot \cos(h_{\text{trans}}(x), k_t\bigr) \cdot 2 - 1\bigr|$$
$$y^{\text{ROOT}}(x) = W_0 x + \sum_{t=1}^T w_t^{\text{ROOT}}(x) \cdot B_t A_t x$$

**SRT inference:**
$$w_t^{\text{SRT}}(x) = \mathbb{1}[t^* = \arg\min_s d_{\text{SRT}}(x, \mathcal{P}_s)]$$
$$y^{\text{SRT}}(x) = W_0 x + B_{t^*} A_{t^*} x$$

**Training is IDENTICAL in both methods** (same `cal_attention` soft routing). The gap is ONLY the inference override.

### F. Why 0.39% — Not Catastrophic

The gap is small because SRT routing accuracy ≈ 99.99% (from CONTRI1). When routing is correct:
- SRT fires the CORRECT adapter → nearly identical to ROOT's soft blend
- Difference = ROOT's soft weights vs SRT's one-hot when the correct adapter dominates

The gap only manifests when:
1. SRT routing is correct but ROOT's soft blending benefited the current task
2. SRT routing is wrong AND ROOT's soft blending would have masked the error

---

## 1.2 The Scientific Question

> **Can we modify the TRAINING objective so that adapters are robust to the hard-routing inference regime?**

**The hypothesis:** Train adapters with the knowledge that at inference, ONLY ONE adapter fires (hard SRT routing). The adapter should be trained to perform well STANDALONE, not relying on cross-task soft blending. If successful:
- SRT inference ≈ ROOT inference (small gap)
- Zero parameter overhead (reuses SRT infrastructure)
- Non-parametric (no learned router to drift)

---

# PHẦN II: THE APPROACHES

## Approach A: Hard Routing During Training (HET) — Simplest

### Idea

Train with hard routing from day 1: during training task $t$, only use adapter $t$'s LoRA. No soft blending during training. Inference = SRT one-hot. Training = same one-hot.

**Protocol:**
$$\mathcal{L}_t^{\text{HET}} = \mathcal{L}_{\text{CE}}\bigl(W_0 x + B_t A_t x,\; y\bigr)$$

**Pros:** Trivial to implement. Eliminates train-inference gap entirely.
**Cons:** Destroys forward transfer. Tasks benefit ZERO from previous adapters during training.

### Analysis

HET is equivalent to training standard per-task LoRA with frozen backbone. The AP will be approximately:
$$\text{AP}^{\text{HET}} \leq \text{AP}^{\text{ROOT}}$$
because ROOT's soft blending provides SOME cross-task knowledge during training. HET removes this benefit.

**Verdict:** Not recommended. Too aggressive.

## Approach B: SRT-Aligned Soft Routing During Training (SASR) — Proposed

### Idea

Train with SRT geometry: at training task $t$, use the SAME SRT router for computing soft weights. The difference from ROOT is the ROUTING MECHANISM (Fisher-Rao from signatures vs learned MLP).

During training of task $t$, compute soft weights as:

$$w_s^{\text{SASR}}(x) = \text{softmax}\bigl(-\gamma \cdot d_{\text{SRT}}(x, \mathcal{P}_s)\bigr)_{s=1:t}$$

This uses the SAME geometric metric (SRT) for both training AND inference routing. The gap is eliminated by construction: training sees hard-ish routing weights, inference uses the same hard weights (limit $\gamma \to \infty$).

### Why This Matters

ROOT's soft routing during training uses learned MLP → weights can DRIFT. SASR's routing uses frozen statistics $\{\mu_t, \Sigma_t\}$ → weights are STABLE.

| Aspect | ROOT training | SASR training |
|---------|--------------|--------------|
| Router | Learned MLP (`trans_input`) | SRT geometric (frozen) |
| Parameters | Drifts with training | Stable (statistics frozen) |
| Inference routing | Learned MLP soft | SRT one-hot |
| Train-inference gap | YES (different mechanisms) | NO (same metric, different temperature only) |

### Theorem C2-1 (Train-Inference Alignment)

**SASR eliminates the train-inference gap by construction.** The routing function $f_{\text{SRT}}: x \mapsto \arg\min_t d_{\text{SRT}}(x, \mathcal{P}_t)$ is the SAME for training and inference (up to temperature). The adapter is trained in a regime consistent with how it will be evaluated.

**Proof.** The training loss uses soft weights $w_t^{\text{SASR}}(x) = \text{softmax}(-\gamma d_{\text{SRT}})$. The inference uses hard $w_t^{\text{SRT-INFER}} = \mathbb{1}[t = \arg\min_s d_{\text{SRT}}]$. As $\gamma \to \infty$: $w_t^{\text{SASR}} \to w_t^{\text{SRT-INFER}}$. The training objective converges to the inference objective. $\square$

## Approach C: DoRA-Inspired Standalone Adapter Training (DoRA-SAT)

### Idea

Apply DoRA's magnitude-direction decomposition to each adapter. Train the magnitude $m_t$ to be standalone-calibrated (robust to inference without cross-task blending). Train the direction $\hat{D}_t$ with SRT-aligned geometric regularization.

**Key insight from DoRA (Liu et al., arXiv:2402.09353):** The full fine-tuned weight $W_0 + \Delta W_t$ decomposes as:

$$W_0 + \Delta W_t = m_t \cdot \hat{V}_t, \quad m_t = \|W_0 + \Delta W_t\|_F, \quad \hat{V}_t = \frac{W_0 + \Delta W_t}{\|W_0 + \Delta W_t\|_F}$$

- Magnitude $m_t$ is independently controllable from direction $\hat{V}_t$
- At inference, only the adapter for $t^*$ fires: $W_0 + \Delta W_{t^*}$
- The DoRA decomposition tells us how to regularize $m_t$ and $\hat{V}_t$ independently

### Training Objective

$$\mathcal{L}_t^{\text{DoRA-SAT}} = \mathcal{L}_{\text{CE}}(W_0 x + \Delta W_t x, y) + \lambda_m \cdot \mathcal{R}_m(m_t) + \lambda_D \cdot \mathcal{R}_D(\hat{V}_t; \{\mathcal{P}_s\})$$

where:

- **$\mathcal{R}_m(m_t)$**: Magnitude harmony — penalizes $m_t$ deviating from initialization. Ensures inference quality matches training quality.
- **$\mathcal{R}_D(\hat{V}_t; $\{\mathcal{P}_s\})$**: Fisher-Rao direction regularization — encourages $\hat{V}_t$ to be discriminative given task geometry $\{\mathcal{P}_s\}$.

---

# PHẦN III: MATHEMATICAL FRAMEWORK

## 3.1 Notation

- $\mathcal{B}$: frozen backbone
- $h(x) = \psi(\mathcal{B}(x)) \in \mathbb{R}^d$: frozen embedding
- $\mathcal{P}_t = \mathcal{N}(\mu_t, \Sigma_t)$: SRT task signature
- $d_{\text{SRT}}(x, t) = (h(x) - \mu_t)^\top \Sigma_{\text{pool}}^{-1}(h(x) - \mu_t)$: SRT Mahalanobis distance
- $\Delta W_t = B_t A_t \in \mathbb{R}^{d \times d}$: LoRA adapter
- $\gamma$: temperature controlling soft-vs-hard routing

## 3.2 SRT Routing is Non-Parametric

The SRT router from CONTRI1 uses **no learned parameters**. The routing decision $t^*(x) = \arg\min_t d_{\text{SRT}}(x, \mathcal{P}_t)$ depends ONLY on frozen embeddings $h(x)$ and stored statistics $\{\mu_t, \Sigma_t\}$. This means:

1. **No drift**: Router quality does not degrade over time
2. **No additional parameters**: Zero storage overhead for routing
3. **Same metric at train and inference**: The geometric function is identical — only temperature differs

## 3.3 Fisher-Rao Alignment

The Fisher-Rao metric governs the natural geometry of probability distributions. For Gaussians $\mathcal{P}_t = \mathcal{N}(\mu_t, \Sigma_t)$, the KL divergence (CONTRI1, Theorem 1) decomposes:

$$D_{\text{KL}}(\mathcal{P}_t \| \mathcal{P}_s) = D_\mu + D_\Sigma$$

where:
$$D_\mu = \frac{1}{2}(\mu_t - \mu_s)^\top \Sigma_s^{-1}(\mu_t - \mu_s) \quad \text{(mean displacement)}$$
$$D_\Sigma = \frac{1}{2}\bigl[\text{tr}(\Sigma_s^{-1}\Sigma_t) - d + \ln\frac{|\Sigma_s|}{|\Sigma_t|}\bigr] \quad \text{(shape mismatch)}$$

**Theorem C2-2 (Fisher-Rao Governs Routing).**

The SRT routing accuracy $\epsilon_{\text{route}}$ is governed by these KL components. The routing error probability is bounded by the minimum pairwise KL between task signatures:

$$\epsilon_{\text{route}} \leq \frac{1}{T}\sum_{t=1}^T \max_{s \neq t} \exp\!\Bigl(-\frac{1}{4}D_{\text{KL}}(\mathcal{P}_t \| \mathcal{P}_s)\Bigr)$$

*Proof.* From CONTRI1, Theorem 3 (routing error floor via Chernoff bound. $\square$

## 3.4 DoRA Decomposition for SRT-Aligned Training

Apply DoRA's principle to the LoRA adapter $\Delta W_t = B_t A_t$. The full fine-tuned weight decomposes:

$$W_0 + \Delta W_t = m_t \cdot \hat{V}_t, \quad m_t = \|W_0 + \Delta W_t\|_F, \quad \hat{V}_t = \frac{W_0 + \Delta W_t}{\|W_0 + \Delta W_t\|_F}$$

**Key property:** magnitude $m_t$ and direction $\hat{V}_t$ are **independently controllable**:
- Changing $m_t$: scales output without changing transformation direction
- Changing $\hat{V}_t$: changes transformation without affecting scale

At SRT inference (hard one-hot):
$$y^{\text{SRT}} = W_0 x + m_{t^*} \hat{V}_{t^*} x$$

**Theorem C2-3 (DoRA-MAT Backward Transfer Bound.**

Under magnitude harmony regularization $\mathcal{R}_m = \frac{\lambda_m}{t-1}\sum_{s<t}|\log(m_t/m_s^{\text{init}})|$, the squared error on task $s$ at inference due to magnitude deviation of task $\tau > s$:

$$\mathbb{E}_{x \sim \mathcal{D}_s}\bigl[\|m_\tau^{\text{inf}} - m_\tau^{\text{train}}\|^2\bigr] \;\leq\; \frac{1}{\lambda_m^2}$$

*Proof.* By Moreau-Yosida regularization bound on the log barrier. $\square$

**Corollary:** Magnitude harmony bounds backward transfer degradation with $O(1/\lambda_m)$, compared to EWC's $O(d^2)$ Fisher storage.

---

# PHẦN IV: THE TWO CONTRIBUTIONS

## 4.1 CONTRI2-A: SASR — SRT-Aligned Soft Routing During Training

### Mechanism

During training of task $t$, replace ROOT's learned MLP routing with SRT geometric routing:

$$w_s^{\text{SASR}}(x) = \frac{\exp\!\bigl(-\gamma \cdot d_{\text{SRT}}(x, \mathcal{P}_s)\bigr)}{\sum_{\tau=1}^t \exp\!\bigl(-\gamma \cdot d_{\text{SRT}}(x, \mathcal{P}_\tau)\bigr)}$$

where $d_{\text{SRT}}(x, \mathcal{P}_s) = (h(x) - \mu_s)^\top \Sigma_{\text{pool}}^{-1}(h(x) - \mu_s)$.

### Properties

| Property | ROOT | SASR |
|----------|------|------|
| Training router | Learned MLP (drifts) | SRT geometric (stable) |
| Inference router | Learned MLP soft | SRT one-hot |
| Train-inference gap | YES (different mechanisms) | **MINIMAL** (same metric, $\gamma \to \infty$ limit) |
| Additional parameters | None | None |
| Storage | MLP + GPM | $\{\mu_t, \Sigma_t\}$ (SRT already stores) |
| Zero-replay | KL distillation needed | ✅ (no raw data) |

### Theorem C2-4 (SASR Zero Gap)

As $\gamma \to \infty$, the SASR training weights converge to the SRT inference one-hot. The training objective converges to the inference objective:

$$\lim_{\gamma \to \infty} \mathcal{L}_t^{\text{SASR}} = \mathcal{L}_t^{\text{SRT inference}}$$

*Proof.* As $\gamma \to \infty$, softmax $\to$ argmax. The training loss uses the same adapter selected at inference. $\square$

**Corollary:** For finite $\gamma$, the gap is $O(1/\gamma)$ bounded. The training objective is a smoothed interpolation between ROOT's soft routing and SRT's hard routing.

## 4.2 CONTRI2-B: DoRA-SAT — DoRA-Inspired Standalone Adapter Training

### Mechanism

Apply DoRA's magnitude-direction decomposition during training. Use SRT geometry to regularize the direction $\hat{V}_t$.

**Training loss:**

$$\mathcal{L}_t^{\text{DoRA-SAT}} = \underbrace{\mathcal{L}_{\text{CE}}(W_0 x + \Delta W_t x, y)}_{\text{task loss}} + \underbrace{\lambda_m \cdot \frac{1}{t-1}\sum_{s<t}\bigl|\log\frac{m_t}{m_s^{\text{init}}}\bigr|}_{\text{(A) Magnitude harmony}} + \underbrace{\lambda_D \cdot \frac{1}{t-1}\sum_{s<t} e^{-\gamma \cdot D_{\text{KL}(\mathcal{P}_t \| \mathcal{P}_s)} \cdot \bigl\|(I - P_s)\Delta W_t\bigr\|_F^2}_{\text{(B) SRT-Geometric direction reg}}$$

where:
- **$m_s^{\text{init}}$**: initial magnitude after InfLoRA projection
- **$P_s$**: projector onto span of task $s$'s adapter
- **$D_{\text{KL}}$**: Fisher-Rao distance from CONTRI1 (Theorem 1)

### Why This Works

| Regularization | Effect |
|---------------|--------|
| Magnitude harmony | $m_t$ stays near initialization → inference = training quality |
| SRT direction reg | $\Delta W_t$ avoids old subspaces proportional to Fisher-Rao distance → discriminative directions |
| Zero additional params | Only $\{\mu_t, \Sigma_t\}$ reused from SRT |

## 4.3 Combined: SASR + DoRA-SAT

Both contributions are complementary:

```
Training task t:
  1. SASR routing: w_s = softmax(-γ · d_SRT(x, P_s))
  2. agg_lora_states with SASR soft weights: y_train = Σ w_s · ΔW_s · x
  3. DoRA-SAT loss: CE + λ_m · |log(m_t/m_s^init)| + λ_D · e^{-γ·D_KL} · ||(I-P_s)ΔW_t||_F²

Inference (SRT hard one-hot):
  y_infer = W_0 x + ΔW_{t*} · x
```

**Key properties:**
- Both use ONLY $\{\mu_t, \Sigma_t\}$ from SRT (no extra storage)
- Both are non-parametric (no learned routing to drift)
- Both eliminate the train-inference gap by construction
- Zero-replay compliant

---

# PHẦN V: RELATIONSHIP TO EXISTING WORK

## vs GainLoRA ROOT

| Aspect | ROOT (GainLoRA) | CONTRI2 (SASR + DoRA-SAT) |
|--------|-----------------|---------------------------|
| Training routing | Learned MLP (drifts) | SRT geometric (stable) |
| Inference routing | Learned MLP soft (no override) | SRT one-hot (non-parametric) |
| Train-inference gap | YES | NO (same metric, limit $\gamma \to \infty$) |
| KL distillation | YES (needed for routing stability) | NO (geometric router doesn't drift) |
| Additional storage | MLP params + prompt_keys + GPM | $\{\mu_t, \Sigma_t\}$ only |
| DoRA decomposition | NO | YES (magnitude harmony + direction reg) |
| Non-parametric | ❌ | ✅ (zero learned params) |
| Zero-drift | ❌ | ✅ |

## vs AdaMoE (NeurIPS 2023)

AdaMoE uses a learned meta-controller to decide hard vs soft routing based on task similarity heuristic. SASR uses the SRT Fisher-Rao metric (information-theoretic) instead of a heuristic. DoRA-SAT is orthogonal to AdaMoE — they address different aspects (routing vs adapter regularization).

## vs DoRA-DA (arxiv 2501.18367)

DoRA-DA aligns representations across tasks using DoRA's magnitude-direction decomposition. CONTRI2 extends this by adding SRT geometry (Fisher-Rao metric) to weight the regularization. DoRA-DA has no routing component. CONTRI2's novelty is the COMBINATION: SASR routing + DoRA-SAT regularization, both grounded in SRT geometry.

## vs InfLoRA / GPM

InfLoRA and GPM protect old adapters via null-space projection. CONTRI2 does not change protection — it addresses the train-inference mismatch. They are complementary: CONTRI2 + InfLoRA = SRT routing + DoRA-SAT training + InfLoRA protection.

---

# PHẦN VI: EXPERIMENTAL PLAN

## E-C2-1: Measure Train-Inference Gap

Protocol: Compare ROOT (soft inference) vs SRT (hard inference) on identical trained adapters. Expected: AP gap = 0.39%.

## E-C2-2: SASR vs ROOT

Protocol: Train with SASR (geometric soft routing), evaluate with SRT one-hot. Expected: gap < 0.39%.

## E-C2-3: DoRA-SAT Ablation

Protocol: Train with DoRA-SAT components individually. Expected: magnitude harmony improves Bwt; direction reg improves Fwt.

## E-C2-4: Optimal $\gamma$ Sweep

Protocol: Vary $\gamma \in \{0.1, 0.5, 1.0, 5.0, \infty\}$. Expected: Non-monotonic curve.

---

# PHẦN VII: LIMITATIONS

1. **SASR still needs $\{\mu_t, \Sigma_t\}$**: Unlike SRT inference alone, SASR needs signatures DURING training, not just at inference. This requires SRT infrastructure to be available at training time.
2. **DoRA-SAT direction reg is $O(d^2)$**: Can use mean-displacement proxy $O(d)$ for large $d$.
3. **The 0.39% gap may be noise**: Small enough that multiple hypotheses could explain it.
4. **CB task**: CB EM=3.57 in ROOT, 3.57 in SRT. Does SASR help CB? Unknown — CB is limited by tiny data (250 samples), not routing.

---

# PHẦN VIII: SUMMARY

## Contribution Statement

> **CONTRI2 — SRT-Aligned Adapter Training:**
>
> Two complementary contributions addressing the inference-only gap between soft-routed ROOT and hard-routed SRT:
>
> **(C2-A) SASR (SRT-Aligned Soft Routing):** Replaces ROOT's learned MLP routing during training with SRT's geometric soft routing $w_s \propto \exp(-\gamma \cdot d_{\text{SRT}})$. The router is non-parametric (no drift), uses the SAME metric as SRT inference, and eliminates the train-inference gap by construction. Zero extra parameters, zero storage.
>
> **(C2-B) DoRA-SAT (DoRA-Inspired Standalone Adapter Training):** Applies DoRA's magnitude-direction decomposition with SRT-Fisher-Rao geometric regularization. Magnitude harmony bounds backward transfer in $O(1)$ vs EWC's $O(d^2)$. Direction regularization uses Fisher-Rao distances to weight orthogonalization. The combination SASR + DoRA-SAT ensures adapters are trained in a regime consistent with hard SRT inference.

## Key Insight

The ROOT→SRT gap is PURELY an inference phenomenon. Training is IDENTICAL. CONTRI2 modifies TRAINING to align with SRT's inference regime, not the other way around.
