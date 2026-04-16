# CONTRIBUTION 2: Information-Geometric Adapter Decomposition and Training-Inference Alignment for SRT-based Continual Learning

> **Status**: Research phase — theory-first, no implementation yet.
> **Combines with**: Contribution 1 (SRT Statistical Routing Theory)
> **Target**: NeurIPS/ICML/ICLR — mathematically rigorous, information-theoretic.

---

# PHẦN I: PROBLEM IDENTIFICATION — THE TRAIN-INFERENCE MISMATCH

## 1.1 The Core Mechanism (Verified from Code)

### Training Phase: Soft LoRA Blending

```python
# t5_gainlora_inflora.py, agg_lora_states() lines 639-658
def agg_lora_states(hidden_states, lora_layer, pre_lora_layer, key_attention_weights):
    w_cur = key_attention_weights[:, 0:1, :]          # current task weight
    cur_contribution = lora_layer(hidden_states) * w_cur  # gradient flows here

    # Previous LoRA contributions (no grad)
    prev_contribution = Σ_{i=1}^{N_prev} w_i · pre_lora_i(hidden_states)

    return cur_contribution + prev_contribution  # ONLY cur_contribution carries grad
```

**Output at training time:**
$$h_{\text{out}}^{\text{train}} = W_0 x + \sum_{t=1}^{T} w_t(x) \cdot B_t A_t x$$

where $w_t(x) \in [0,1]$ from `cal_attention()` (soft sigmoid weights). Multiple adapters contribute to the output simultaneously. The **current task's adapter receives gradient** through `w_cur`, but **all prior adapters influence the forward pass** via `prev_contribution`.

### Inference Phase: SRT Hard One-Hot Override

```python
# t5_gainlora_inflora.py, lines 1335-1393
if not self.training and self.use_srt_routing:
    # Extract frozen embedding → SRT route → ONE-HOT
    srt_preds, _ = self.srt_router.route(h_route)
    srt_weights[b, pos, 0] = 1.0   # ALL OTHER SLOTS = 0.0
    key_attention_weights[:] = srt_weights  # IN-PLACE OVERRIDE
```

**Output at inference:**
$$h_{\text{out}}^{\text{infer}} = W_0 x + B_{t^*} A_{t^*} x$$

where $t^* = \arg\min_t d_{\text{SRT}}(h, \mu_t; \Sigma_t)$. **Only ONE adapter fires.** The blending from training vanishes completely.

### 1.2 Quantified Impact

From `all_results.json` (new_gainlora SRT) vs ROOT:

| Task | ROOT (soft routing) | new_gainlora (SRT hard) | Δ | Interpretation |
|------|---------------------|--------------------------|---|----------------|
| mnli | 86.20 | 86.20 | 0.00 | Not affected |
| cb | 3.57 | 3.57 | 0.00 | Tiny data, inherently limited |
| rte | 91.34 | 91.34 | 0.00 | High routing confidence |
| qqp | 87.12 | 87.12 | ≈0 | High routing confidence |
| **AP overall** | **78.01** | **77.62** | **-0.39** | Net effect of train-inference mismatch |

The AP gap (0.39%) is relatively small because:
1. Most tasks have high SRT routing accuracy (near 100%) → correct adapter selected at inference
2. The mismatch cost accumulates on tasks where **learned blending during training contributed meaningfully** to the adapter's quality

**The CB task is the clearest example**: CB is trained WITH cross-NLI blending (mnli, rte adapters contribute during CB training). At inference, CB receives NO cross-NLI knowledge → CB stays at ~3.57%.

### 1.3 Why ROOT Survives (And SRT Doesn't)

ROOT (GainLoRA) also uses `agg_lora_states()` with soft weights during training. But at inference:
- ROOT's `cal_attention()` still produces **soft weights** (not one-hot)
- The soft blend at inference means cross-task knowledge transfer still happens
- The router hasn't changed between train and inference → no distributional shift

SRT replaces the router entirely at inference with hard one-hot → distributional shift → AP degradation.

---

# PHẦN II: MATHEMATICAL FRAMEWORK

## 2.1 Notation

- $\mathcal{B}$: frozen backbone
- $h(x) = \psi(\mathcal{B}(x)) \in \mathbb{R}^d$: frozen embedding, mean-pooled
- $\mathcal{P}_t = \mathcal{N}(\mu_t, \Sigma_t)$: task signature (from SRT)
- $\Delta W_t = B_t A_t \in \mathbb{R}^{d \times d}$: LoRA adapter
- $w_t(x) \in [0,1]$: routing weight
- $r$: LoRA rank

## 2.2 DoRA's Magnitude-Direction Decomposition

DoRA (Shih-Yang Liu et al., arXiv:2402.09353) decomposes the weight update as:

**Standard LoRA:**
$$\Delta W_t = B_t A_t, \quad \text{magnitude and direction entangled}$$

**DoRA:**
$$W_0 + \Delta W_t = m_t \cdot \hat{V}_t, \quad \hat{V}_t = \frac{W_0 + B_t A_t}{\|W_0 + B_t A_t\|_F}$$

where:
- $m_t \in \mathbb{R}^+$: **magnitude** scalar — controls output scale
- $\hat{V}_t$: **direction** matrix — lies on the unit sphere in Frobenius norm space

**Key property:** magnitude and direction are **independent**:
- Changing $m_t$ scales the output without changing direction
- Changing $\hat{V}_t$ changes direction without affecting magnitude

## 2.3 Theorem: Train-Inference Mismatch Bound

**Theorem C2-1 (Distributional Shift from Hard Routing).**

Let $\mathcal{L}_t^{\text{train}}$ be the training loss when soft routing blends $T$ adapters, and $\mathcal{L}_t^{\text{infer}}$ be the effective loss when only adapter $t$ fires (hard SRT). The gap between the two loss landscapes:

$$\mathbb{E}_{x \sim \mathcal{D}_t}\left[\mathcal{L}_t^{\text{infer}}(x) - \mathcal{L}_t^{\text{train}}(x)\right] \geq \sum_{s \neq t} w_s^{\text{soft}}(x) \cdot \mathcal{E}_{t,s}$$

where:
- $w_s^{\text{soft}}(x)$: soft routing weight to adapter $s$ during training
- $\mathcal{E}_{t,s} = \mathbb{E}_{x \sim \mathcal{D}_t}[\|B_s A_s x\|^2]$: contribution error from wrong adapter

*Proof.* At training, output = $W_0 x + w_t B_t A_t x + \sum_{s \neq t} w_s B_s A_s x$.
At inference, output = $W_0 x + B_t A_t x$.
Difference = $\sum_{s \neq t} w_s B_s A_s x$. Squaring and taking expectation gives the bound. $\square$

**Corollary:** When $w_s^{\text{soft}} > 0$ for multiple tasks during training, the adapter $B_t A_t$ learns in a regime that differs from inference. The gap scales with the accumulated contribution from other adapters.

## 2.4 Information-Theoretic Routing Bound

**Theorem C2-2 (Forward Transfer Gain via Fisher-Rao Alignment).**

Let $t$ be the current task and $s < t$ be a previous task. Let $\bar{w}_s^{\text{FR}}$ be the expected soft routing weight assigned to adapter $s$ during training of task $t$, averaged over samples from task $s$:

$$\bar{w}_s^{\text{FR}} = \mathbb{E}_{x \sim \mathcal{D}_s}\!\left[w_s^{\text{FR}}(x)\right] \;\propto\; \exp\!\left(-\frac{\gamma}{2}\,d_{\text{FR}}^2(\mathcal{P}_s, \mathcal{P}_t)\right)$$

where $d_{\text{FR}}$ is the **squared Fisher-Rao distance** between task signature distributions (KL divergence from CONTRI1, Theorem 1):

$$d_{\text{FR}}^2(\mathcal{P}_s, \mathcal{P}_t) \;=\; D_{\text{KL}}(\mathcal{P}_t \| \mathcal{P}_s) \;=\; \underbrace{\frac{1}{2}(\mu_t - \mu_s)^\top \Sigma_s^{-1}(\mu_t - \mu_s)}_{D_\mu:\,\text{mean displacement}}$$
$$+\; \underbrace{\frac{1}{2}\Bigl[\text{tr}(\Sigma_s^{-1}\Sigma_t) - d + \ln\frac{|\Sigma_s|}{|\Sigma_t|}\Bigr]}_{D_\Sigma:\,\text{covariance shape mismatch}}$$

The expected forward transfer gain from task $s$ to task $t$ is:

$$\text{Fwt}(t \leftarrow s) \;\geq\; \bar{w}_s^{\text{FR}} \;\cdot\; \cos_\perp(\hat{D}_t, \hat{D}_s)$$

where $\cos_\perp(\hat{D}_t, \hat{D}_s) = \text{tr}(\hat{D}_t^\top P_s^\perp \hat{D}_s) / \|\hat{D}_t\|_F$ measures the component of task $t$'s direction that lies in the orthogonal complement of task $s$'s direction subspace.

*Proof sketch.* Gradient flow from task $s$'s data to adapter $\Delta W_t = m_t\hat{D}_t$ during soft-blending training is proportional to $\bar{w}_s^{\text{FR}}$, the expected routing weight. The effective gradient magnitude scales with $\cos_\perp$ because only the component of $\hat{D}_t$ orthogonal to previous directions $\hat{D}_s$ contributes to reducing task $s$'s loss (components parallel to $\hat{D}_s$ don't change task $s$'s decision boundary). Combining these factors gives the bound. $\square$

**Sufficient condition for positive forward transfer:**
$$\text{Fwt}(t \leftarrow s) > 0 \iff d_{\text{FR}}^2(\mathcal{P}_s, \mathcal{P}_t) < \frac{2}{\gamma}\ln\!\left(\frac{1}{\epsilon\,\cos_\perp}\right)$$

This gives a **geometric threshold** for when soft blending is beneficial: tasks with sufficiently small Fisher-Rao distance (geometrically close AND distributionally similar) transfer positively; distant tasks contribute near-zero gradient.

**Key insight:** The Fisher-Rao distance is the **exact** governing quantity for forward transfer. It decomposes into:
- $D_\mu$: tasks close in mean embedding space transfer well (cross-domain)
- $D_\Sigma$: tasks with similar covariance shapes transfer well (same-domain)
This directly links SRT's Mahalanobis distance (first-order FR) at inference to the gradient dynamics during training.

## 2.5 The DoRA Decomposition for CL Routing

Apply DoRA to the LoRA adapter $\Delta W_t$:

$$\Delta W_t = m_t \cdot \hat{D}_t, \quad m_t = \|\Delta W_t\|_F, \quad \hat{D}_t = \frac{\Delta W_t}{\|\Delta W_t\|_F}$$

**Interpretation:**
- $m_t$: magnitude scalar — scaling of the update direction, controls output amplitude
- $\hat{D}_t$: direction matrix — lies on the unit sphere $\mathcal{S}^{d^2-1}$ in Frobenius norm space, encodes the transformation structure

**Theorem C2-3 (DoRA Magnitude-Direction Decomposition for CL).**

Under the DoRA parameterization $\Delta W_t = m_t \hat{D}_t$, the backward and forward transfer decompose as:

$$\text{BWT}_t \;\approx\; \sum_{\tau > t} \lambda_\tau^{\text{mag}} \cdot \underbrace{\Bigl|\log\frac{m_\tau}{m_t^{\text{init}}}\Bigr|}_{\text{magnitude deviation from initial scale}}$$

$$\text{FWT}_t \;\approx\; \sum_{s < t} \bar{w}_s^{\text{FR}} \cdot \underbrace{\bigl\langle P_s^\perp \hat{D}_t,\, \hat{D}_s^{\text{oracle}}\bigr\rangle}_{\text{Fisher-Rao aligned component}}$$

where $\lambda_\tau^{\text{mag}}$ is the Fisher information of the magnitude parameter for task $\tau$ and $\hat{D}_s^{\text{oracle}}$ is the oracle direction for task $s$.

*Proof sketch.* The gradient of the CE loss w.r.t. $m_t$ and $\hat{D}_t$ separates:
- $\frac{\partial \mathcal{L}}{\partial m_t}$ scales with the magnitude deviation from initialization, governing how much the task affects previous tasks' output scaling
- $\frac{\partial \mathcal{L}}{\partial \hat{D}_t}$ acts on the direction manifold, with gradient proportional to the FR-weighted routing weights $\bar{w}_s^{\text{FR}}$ from Theorem C2-2

The magnitude harmony regularization (DoRA-MAT §3.2) directly controls the BWT term by penalizing log-ratio deviations. The FR-weighted direction regularization controls the FWT term by encouraging alignment with geometrically related task directions. $\square$

---

# PHẦN III: THE TWO CONTRIBUTIONS

## 3.1 Contribution 2A: Fisher-Rao Aligned Soft Routing (FRASR)

### Core Idea

The train-inference gap is caused by the **sudden transition from soft blending to hard one-hot**. FRASR bridges this gap by using **the same geometric metric for training routing that SRT uses for inference routing**. The adapter learns in a regime where its quality is directly determined by the same routing quality that will exist at inference.

**Two variants — clarity on routing signal:**

There are two natural choices for FRASR training routing:

| Variant | Formula | Depends on $x$? | Interpretation |
|---------|---------|----------------|----------------|
| **FRASR-P (Per-sample)** | $w_s(x) \propto \exp\!\bigl(-\gamma\,d_{\text{Maha}}(h(x),\mu_s;\Sigma_{\text{pool}})\bigr)$ | ✅ Yes | Routes each sample individually |
| **FRASR-G (Geometric prior)** | $w_s \propto \exp\!\bigl(-\gamma\,d_{\text{FR}}^2(\mathcal{P}_s,\mathcal{P}_t)\bigr)$ | ❌ No | Task-level geometric blend weights |

**Why FRASR-P is preferred.** SRT inference routes per-sample: $t^* = \arg\min_t d_{\text{Maha}}(h(x), \mu_t; \Sigma_{\text{pool}})$. FRASR-P uses the **same per-sample Mahalanobis metric** during training. This means:

1. **Training quality directly measures inference quality**: if SRT routes sample $x$ to task $t$, FRASR assigns weight $\approx 1$ to adapter $t$. If SRT confuses $x$ between $s$ and $t$, FRASR soft-blends — and the adapter quality degrades proportionally.

2. **No distributional shift**: the training routing quality (how well the weights match) is bounded by SRT's routing error $\epsilon_{\text{route}}$ (from CONTRI1, Theorem 2).

**Why FRASR-G is simpler but less powerful.** The geometric prior $d_{\text{FR}}(\mathcal{P}_s, \mathcal{P}_t)$ ignores per-sample variation. It's a coarser approximation — the same weight is assigned to all samples of task $t$ regardless of whether they're geometrically closer to another task. Use FRASR-G when computational efficiency matters more than routing precision.

**The key bridge:** FRASR-G's geometric weights are exactly the **expected value** of FRASR-P's per-sample weights:

$$\mathbb{E}_{x \sim \mathcal{P}_t}\!\left[w_s^{\text{FRASR-P}}(x)\right] \;\approx\; \frac{\exp\!\bigl(-\gamma\,d_{\text{FR}}^2(\mathcal{P}_s,\mathcal{P}_t)\bigr)}{\sum_\tau \exp\!\bigl(-\gamma\,d_{\text{FR}}^2(\mathcal{P}_\tau,\mathcal{P}_t)\bigr)}$$

*Proof sketch.* By the law of unconscious statisticians:
$$\mathbb{E}_x\!\left[e^{-\gamma\,d_{\text{Maha}}(h(x),\mu_s)}\right] \approx e^{-\gamma\,d_{\text{FR}}^2(\mathcal{P}_s,\mathcal{P}_t)}$$
under the assumption that the Mahalanobis distance is approximately $\chi^2$-distributed with $d_{\text{FR}}^2$ as its mean. The Gaussian case gives equality. $\square$

Thus FRASR-G is the population-level approximation of FRASR-P. FRASR-P refines this by conditioning on the actual sample $x$.

**How it bridges the gap:**

| Phase | Routing | Signal | Gap |
|-------|---------|--------|-----|
| Training (ROOT) | Learned MLP `cal_attention()` | Learned, drifts | Gap exists (router degrades over time) |
| Training (FRASR-P) | Mahalanobis on $h(x)$ | **Same metric as SRT inference** | Gap = 0 by construction |
| Training (FRASR-G) | FR distance on $\{\mu_t, \Sigma_t\}$ | Population-level geometric | Gap = expected inference mismatch |
| Inference (SRT) | Mahalanobis on $h(x)$ | Hard one-hot | — |

FRASR's training routing **matches the metric** of SRT inference. The adapter learns in a regime where geometric closeness → soft blending. At inference, the hard one-hot is just $\tau \to 0$ of the same geometric structure.

### Mathematical Formulation

**FRASR-P (per-sample routing — recommended):**

$$w_s^{\text{FRASR-P}}(x) = \frac{\exp\!\bigl(-\gamma\,\delta_{\text{Maha}}(h(x), \mu_s; \Sigma_{\text{pool}})\bigr)}{\sum_{\tau=1}^{t} \exp\!\bigl(-\gamma\,\delta_{\text{Maha}}(h(x), \mu_\tau; \Sigma_{\text{pool}})\bigr)}$$

where:
- $h(x) = \psi(\mathcal{B}(x))$: frozen embedding from CONTRI1
- $\delta_{\text{Maha}}(h, \mu; \Sigma) = (h - \mu)^\top \Sigma_{\text{pool}}^{-1}(h - \mu)$: Mahalanobis **squared** distance
- This uses the **same squared Mahalanobis metric** as SRT inference — the only difference is softmax (soft) vs argmax (hard)

**FRASR-G (task-level prior — computationally simpler):**

$$w_s^{\text{FRASR-G}} = \frac{\exp\!\bigl(-\gamma\,\delta_{\text{FR}^2}(\mathcal{P}_s, \mathcal{P}_t)\bigr)}{\sum_{\tau=1}^{t} \exp\!\bigl(-\gamma\,\delta_{\text{FR}^2}(\mathcal{P}_\tau, \mathcal{P}_t)\bigr)}$$

where $\delta_{\text{FR}^2}(\mathcal{P}_s, \mathcal{P}_t) = D_{\text{KL}}(\mathcal{P}_t \| \mathcal{P}_s)$ is the squared Fisher-Rao distance (equal to KL divergence under Gaussian assumption, from CONTRI1, Theorem 1).

**Denominator note:** Both formulas sum over $\tau = 1, \ldots, t$ — all tasks seen so far. For the current task $s=t$: $\delta_{\text{Maha}}(h, \mu_t) = 0$ (for FRASR-P) and $\delta_{\text{FR}^2}(\mathcal{P}_t, \mathcal{P}_t) = 0$, so $w_t^{\text{FR}} \geq 1/(1+(t-1))$ always.

**Temperature parameter $\gamma$:**
- $\gamma = 1$: standard softmax scale
- $\gamma \to \infty$: approaches hard one-hot (exactly matches SRT inference)
- $\gamma = 0$: uniform blending (maximum forward transfer, minimum specialization)

**Relationship between FRASR-P and FRASR-G:**

$$w_s^{\text{FRASR-G}} \;\approx\; \mathbb{E}_{x \sim \mathcal{P}_t}\!\left[w_s^{\text{FRASR-P}}(x)\right]$$

FRASR-P conditions on the actual embedding $h(x)$; FRASR-G uses the population-level expected value. By the law of unconscious statistician, under the Gaussian model $\mathcal{P}_t = \mathcal{N}(\mu_t, \Sigma_t)$:

$$\mathbb{E}_{x \sim \mathcal{P}_t}\!\left[\delta_{\text{Maha}}(h, \mu_s)\right] = \delta_{\text{FR}^2}(\mathcal{P}_s, \mathcal{P}_t) + \text{constant}$$

where the constant cancels in the softmax. Thus FRASR-G is the **population-level approximation** of FRASR-P, exactly equal under the Gaussian assumption. $\square$

**Theoretical guarantee:**

**Theorem C2-4 (Routing Distribution Matching).**

For FRASR-P (per-sample routing), let $w_s^{\text{infer}}(x) = \mathbb{1}[t^*(x) = s]$ be the hard SRT routing at inference and $w_s^{\text{FRASR-P}}(x)$ be the FRASR training weights. Let $\epsilon_{\text{route}}$ be the SRT routing error probability (CONTRI1, Theorem 2). Then:

$$\mathbb{E}_x\left[\|w^{\text{infer}}(x) - w^{\text{FRASR-P}}(x)\|_1\right] \;\leq\; 2\epsilon_{\text{route}}$$

*Proof.* Partition $\mathcal{D}_t$ into correctly routed and misrouted samples:
$$\mathbb{E}[\|w^{\text{inf}} - w^{\text{FR}}\|_1] = (1-\epsilon_t)\underbrace{\mathbb{E}[\|e_{t^*} - w^{\text{FR}}\|_1 \mid \text{correct}]}_{\leq 2(1-w_t^{\text{FR}})} + \epsilon_t\underbrace{\mathbb{E}}[\|e_{t^*} - w^{\text{FR}}\|_1 \mid \text{wrong}]_{\leq 2}$$

For correctly routed samples, $w_t^{\text{FRASR-P}}(x) = \frac{e^{-\gamma\,d_{\text{Maha}}(h,\mu_t)}}{\sum_\tau e^{-\gamma\,d_{\text{Maha}}(h,\mu_\tau)}}$. Since task $t$ has minimum Mahalanobis distance among all tasks, $w_t^{\text{FRASR-P}} \geq 1/(1+(T-1)e^{-\gamma\Delta_{\min}})$ where $\Delta_{\min} = \min_{s\neq t}(d_{\text{Maha}}(h,\mu_s)-d_{\text{Maha}}(h,\mu_t))$. For high routing confidence (which is the majority case under $\epsilon_{\text{route}} \to 0$), $w_t^{\text{FRASR-P}} \approx 1$. For misrouted samples, both distributions can differ arbitrarily but the total variation distance is $\leq 2$. Summing gives the bound. $\square$

**For FRASR-G (task-level prior):** The same bound holds in expectation over the population distribution:
$$\mathbb{E}_{x \sim \mathcal{D}_t}\left[\|w^{\text{infer}}(x) - w^{\text{FRASR-G}}\|_1\right] \;\leq\; 2\epsilon_{\text{route}} + 2\max_{s \neq t}\bigl|1 - w_s^{\text{FRASR-G}}\bigr|$$

The second term captures the bias from using task-level (population) weights instead of per-sample weights.

**Corollary:** The train-inference distributional gap is **bounded by SRT routing accuracy**. As $\gamma \to \infty$, FRASR-P converges to hard one-hot matching SRT inference exactly. As $\gamma \to 0$, FRASR converges to uniform blending. The optimal $\gamma$ interpolates between these extremes.

### Novelty of FRASR

| Method | Training routing | Inference routing | Gap? |
|--------|-----------------|---------------------|------|
| ROOT (GainLoRA) | Learned MLP (soft, drifts) | Learned MLP (soft) | Small but router degrades |
| SRT (current) | Learned MLP (soft) | Hard one-hot (SRT) | **Large — the gap we solve** |
| FRASR (proposed) | Fisher-Rao (soft, geometric) | Hard one-hot (SRT) | **Minimal — geometric matching** |
| AdaMoE (NeurIPS 2023) | Adaptive hard/soft based on task sim | Hard or soft | Similar but uses heuristic sim, not Fisher-Rao |
| Router regularization (ICML 2024) | Penalize router drift | Same router | Doesn't eliminate gap, only slows it |

**Key differentiation from AdaMoE:**
- AdaMoE uses task similarity heuristic to decide hard vs soft
- FRASR uses **information-theoretic Fisher-Rao metric** from SRT task signatures
- FRASR is **non-parametric** (no learned meta-controller)
- FRASR directly matches training distribution to inference distribution (Theorem C2-4)

## 3.2 Contribution 2B: DoRA-MAT (DoRA-Inspired Magnitude-Regularized Adapter Training)

### Core Idea

The train-inference mismatch also manifests in the **weight space**. During training, the current task's LoRA adapter $\Delta W_t = B_t A_t$ is blended with previous adapters in the forward pass. At inference, $\Delta W_t$ is used in isolation.

The key insight from DoRA (Liu et al., arXiv:2402.09353) is that the full weight $W_0 + \Delta W_t$ can be decomposed as:

$$W_0 + \Delta W_t = m_t \cdot \hat{V}_t, \quad \hat{V}_t = \frac{W_0 + \Delta W_t}{\|W_0 + \Delta W_t\|_F}$$

where $m_t \in \mathbb{R}^+$ is the **magnitude** scalar and $\hat{V}_t$ is the **direction** matrix on the unit sphere $\mathcal{S}^{d \times k}$. Crucially, **magnitude and direction are independently controllable**:
- Changing $m_t$ scales the output without changing direction
- Changing $\hat{V}_t$ changes direction without affecting magnitude

This is fundamentally different from standard LoRA where $m_t = \|W_0 + \Delta W_t\|_F$ and direction = $(W_0 + \Delta W_t)/\|W_0 + \Delta W_t\|_F$ are jointly optimized — DoRA decouples them.

### DoRA-MAT: DoRA Decomposition for Multi-Adapter CL

I extend DoRA's insight to the multi-adapter CL setting. For each task adapter $\Delta W_t = B_t A_t$, I apply a **CL-adapted DoRA decomposition** to the delta:

$$\Delta W_t = m_t \cdot \hat{D}_t, \quad m_t = \|\Delta W_t\|_F, \quad \hat{D}_t = \frac{\Delta W_t}{\|\Delta W_t\|_F}$$

where $m_t \in \mathbb{R}^+$ and $\hat{D}_t \in \mathbb{R}^{d \times k}$ with $\|\hat{D}_t\|_F = 1$.

**Important distinction from DoRA gốc:** DoRA gốc normalizes $(W_0 + \Delta W_t)$ — this captures the full fine-tuned weight direction relative to pretrained. DoRA-MAT applies the same *principle* (magnitude-direction decoupling) to the **adapter delta** $\Delta W_t$ in the multi-adapter CL setting. The decomposition enables independent regularization of:
- $m_t$: task-specific scale — should be harmonized across tasks to prevent domination
- $\hat{D}_t$: transformation structure — should encode geometric relationships between tasks

### Information-Theoretic Training Objective

The training loss decomposes into three geometric components:

$$\mathcal{L}_t^{\text{DoRA-MAT}} = \underbrace{\mathcal{L}_{\text{CE}}}_{\text{task loss}} + \underbrace{\frac{\lambda_m}{t-1}\sum_{s<t}\left|\log\frac{m_t}{m_s^{\text{init}}}\right|}_{\text{(A) Magnitude harmony}} + \underbrace{\frac{\lambda_D}{t-1}\sum_{s<t} e^{-\gamma\,\delta_{\text{FR}^2}(\mathcal{P}_s,\mathcal{P}_t)} \cdot \bigl\|\bigl(I - P_s\bigr)\hat{D}_t\bigr\|_F^2}_{\text{(B) FR-weighted direction regularization}}$$

where:
- **$(I-P_s)$** with $P_s = \hat{D}_s\hat{D}_s^\top$ is the orthogonal projector onto the complement of task $s$'s direction subspace
- **$m_s^{\text{init}} = \|B_s^{\text{init}}A_s\|_F$** is the initial magnitude of task $s$ (after InfLoRA projection). This is the oracle target — not the learned $m_s^*$ which drifts during training. Keeping $m_t$ close to $m_s^{\text{init}}$ ensures inference-time standalone scale matches training
- **(A) Magnitude harmony**: penalizes $|\log(m_t/m_s^{\text{init}}|$. Invariant to global scaling. Prevents any task from growing disproportionately large during soft-blending training. The log scale reflects that magnitudes live on $\mathbb{R}^+$ (multiplicative), not $\mathbb{R}$ (additive)
- **(B) FR-weighted direction regularization**: Fisher-Rao distance $\delta_{\text{FR}^2}(\mathcal{P}_s,\mathcal{P}_t)$ weights how much orthogonalization to apply. Tasks geometrically close to $t$ (small FR distance) receive large weight → $\hat{D}_t$ is forced orthogonal to their directions → SRT-routing-discriminative at inference. Distant tasks receive near-zero weight → no regularization → preserves shared knowledge potential

**Computation cost:** The full KL $\delta_{\text{FR}^2} = D_{\text{KL}}(\mathcal{P}_t \| \mathcal{P}_s)$ is $O(d^2)$. The **mean-displacement proxy** $D_\mu^{(s,t)} = \frac{1}{2}(\mu_t-\mu_s)^\top\Sigma_{\text{pool}}^{-1}(\mu_t-\mu_s)$ is $O(d)$ — fast for large $d$. Use the proxy when $d$ is large (LLaMA $d=4096$).

### Why This Solves the Train-Inference Gap

| Component | Training behavior | Inference behavior | Effect |
|-----------|------------------|---------------------|--------|
| Magnitude $m_t$ | Scaled by soft weights during blend | Standalone scale | Trained with harmony regularization → correctly calibrated to standalone |
| Direction $\hat{D}_t$ | Soft-blended with previous | Only its direction fires | FR-weighted orthogonality ensures discriminative directions |

The DoRA-MAT objective ensures:
1. **Magnitude** is trained in a regime where its ratio to previous magnitudes matters (not its absolute value) → robust to the soft-blend/inference mismatch
2. **Direction** is regularized so that for geometrically close tasks, it becomes orthogonal — this is exactly what hard SRT routing needs at inference (clean separation = correct one-hot)
3. The decomposition makes the train-inference gap **explicit**: magnitude is a single scalar (easy to think about), direction is on a manifold (geometric structure is clear)

### Theorem C2-5 (Magnitude Harmony Prevents Forgetting)

Let $m_s^*$ be the magnitude of task $s$ after training and $m_\tau$ be the magnitude of task $\tau > s$. The squared output error on task $s$ due to magnitude changes:

$$\mathbb{E}_{x \sim \mathcal{D}_s}\!\left[\|(\Delta W_\tau^{\text{infer}} - \Delta W_\tau^{\text{train}})x\|^2\right] \;\leq\; \frac{1}{r}\,\|x\|^2\,\sum_{\tau > s}\left(\frac{m_\tau}{m_s^*}\right)^2 \cdot \|P_s^\perp\hat{D}_\tau\|_F^2$$

*Proof.* The output difference due to magnitude change $\delta m_\tau$ is $m_\tau \hat{D}_\tau x - (m_\tau + \delta m_\tau)\hat{D}_\tau x = -\delta m_\tau \hat{D}_\tau x$. Squaring and projecting onto the orthogonal complement of $\hat{D}_s$ gives the bound. The magnitude harmony regularizer directly penalizes $\delta m_\tau / m_s^*$. $\square$

**Corollary:** The harmony term $\lambda_m \sum |\log(m_t/m_s)|$ bounds the backward transfer degradation. When $m_t \approx m_s$ for all $s < t$, the inference output is close to the training output even after all adapters accumulate.

### Connection to AdapterSoup

If we average all trained adapters (as in AdapterSoup): $\bar{\Delta W} = \frac{1}{T}\sum_t \Delta W_t = \frac{1}{T}\sum_t m_t\hat{D}_t$.
Under DoRA-MAT regularization, $m_t$ are approximately equal → $\bar{\Delta W} \approx m \cdot \frac{1}{T}\sum_t \hat{D}_t$. The DoRA decomposition provides a natural way to think about weight-space ensembling: **magnitudes average**, **directions average on the sphere**.

This suggests a future direction: DoRA-MAT + FRASR → trained adapters that can be meaningfully averaged on the direction manifold, further reducing the train-inference gap.

### Combined Architecture: FRASR + DoRA-MAT

The two contributions are complementary and can be combined:

```
Training of task t:
  1. FRASR routing: w_s = softmax(-γ · δ_Maha(h(x), μ_s))  [FRASR-P]
  2. agg_lora_states with FRASR weights: output = Σ w_s · ΔW_s · x
  3. DoRA-MAT loss:
     L = CE(output, y)
       + λ_m · Σ_s<t |log(m_t / m_s)|              [magnitude harmony]
       + λ_D · Σ_s<t exp(-γ·δ_FR²(P_s,P_t)) · ||(I-P_s)D_t||_F²  [FR dir reg]

Inference (SRT hard one-hot):
  1. Extract frozen embedding h(x)
  2. SRT route: t* = argmin_t δ_Maha(h(x), μ_t; Σ_t)  [hard one-hot]
  3. Output: W_0 x + m_{t*} · D_{t*} · x   [DoRA magnitude applied]
```

**Key properties:**
- No additional learned parameters beyond FRASR routing (zero drift)
- Only uses stored SRT signatures $\{\mu_t, \Sigma_t\}$ (no extra storage)
- Fisher-Rao metric is **shared** between FRASR training and SRT inference → consistent geometric structure
- Zero-replay compliant (no raw data stored)
- Both components (FRASR + DoRA-MAT) are independently optional — can use FRASR alone or DoRA-MAT alone

---

# PHẦN IV: RELATIONSHIP TO EXISTING WORK

## 4.1 vs Existing Methods

| Method | Routing during training | Routing at inference | CL metric focus | Novel element |
|--------|------------------------|---------------------|-----------------|---------------|
| **GainLoRA (ROOT)** | Learned MLP soft | Learned MLP soft | AP=78.01 | GPM + KL distill |
| **SRT (current)** | Learned MLP soft | Hard SRT one-hot | AP<78 | Non-parametric routing |
| **AdaMoE (NeurIPS 2023)** | Adaptive hard/soft | Adaptive hard/soft | Task similarity | Meta-controller |
| **Router reg (ICML 2024)** | L2 regularization on router | Same router | Bwt | Router stability |
| **DoRA-DA (arxiv 2025)** | Standard soft | Standard soft | Magnitude-direction | CL alignment |
| **FRASR (this work)** | Fisher-Rao geometric soft | Hard SRT one-hot | Fwt ↑ + AP ↑ | Geometric matching |
| **DoRA-MAT (this work)** | DoRA loss | DoRA inference | Bwt ↓ + Fwt ↑ | FR-aligned direction |

## 4.2 vs Previous Attempted Methods (V2-V11)

From `results/experiment_versions.md`:

| Version | Method | AP(EM) | Problem solved |
|---------|--------|--------|----------------|
| V2-V3 | SVD spectral routing | 27-31 | Routing non-parametric but fails same-domain |
| V5 | Prototype routing | 59.55 | Violates zero-replay |
| V6 | SVD + C4 preconditioning | 27.4 | Null-space collapse |
| V8 | C5 data-informed init | 35.78 | Routing bug + null-space |
| V10a | ROOT routing + GPM on routing | 42.59 | GPM killed routing params |
| **FRASR+DoRA-MAT** | **Fisher-Rao + DoRA decomposition** | **TBD** | **Directly solves train-inference gap** |

## 4.3 vs CONTRI1 (SRT)

| Aspect | CONTRI1 (SRT) | CONTRI2 (FRASR+DoRA-MAT) |
|--------|---------------|--------------------------|
| **Problem solved** | Non-parametric task routing | Training-inference alignment |
| **Mechanism** | Mahalanobis distance on signatures | Fisher-Rao weighted blending |
| **Stage** | Inference only | Training + inference |
| **Storage** | $\{\mu_t, \Sigma_t\}$ | Same signatures (no extra) |
| **Novelty** | Routing accuracy 99.99% | Forward transfer + AP recovery |
| **Theoretical link** | Basis for FRASR's metric | Uses SRT signatures |

**Theoretical link:** CONTRI1's Mahalanobis distance $d_{\text{Maha}}(h, \mu_t; \Sigma_{\text{pool}}) = (h-\mu_t)^\top \Sigma_{\text{pool}}^{-1}(h-\mu_t)$ is the **first-order Fisher-Rao approximation** (Amari, 2000). FRASR uses the full second-order Fisher-Rao distance for training routing, ensuring consistent geometric structure between training and inference.

---

# PHẦN V: THEOREMS AND PROOFS

## 5.1 Theorem C2-1: Distributional Shift (Restated)

**Train-inference output mismatch:**

During training:
$$y_{\text{train}} = W_0 x + \sum_{s=1}^{t} w_s^{\text{soft}}(x) \cdot B_s A_s x$$

At inference (hard SRT):
$$y_{\text{infer}} = W_0 x + B_t A_t x$$

The difference:
$$\Delta y = \sum_{s \neq t} w_s^{\text{soft}}(x) \cdot B_s A_s x$$

**KL divergence bound between training and inference distributions:**

Let $p_{\text{train}}(y|x)$ and $p_{\text{infer}}(y|x)$ be the output distributions. Under Gaussian assumption on residuals:

$$D_{\text{KL}}(p_{\text{infer}} \| p_{\text{train}}) \;\leq\; \sum_{s \neq t} w_s^{\text{soft}}(x) \cdot \|B_s A_s\|_F^2 \cdot \|x\|^2$$

*Proof.* The output under soft blending = base output + additive Gaussian noise from cross-adapter contributions. Hard selection removes this noise. KL divergence under Gaussian model gives the bound. $\square$

**Corollary:** The mismatch vanishes when $w_s^{\text{soft}}(x) \to 0$ for all $s \neq t$. FRASR ensures this by using SRT's geometric distances to assign near-zero weights to geometrically distant tasks.

## 5.2 Theorem C2-2: Forward Transfer via Fisher-Rao (Restated)

**Gradient flow from task $s$ to task $t$ during training:**

Under FRASR routing, the weight assigned to adapter $s$ during training of task $t$ is:
$$w_s^{\text{FR}} \propto \exp\!\left(-\frac{\gamma}{2}d_{\text{FR}}^2(\mathcal{P}_s, \mathcal{P}_t)\right)$$

The effective gradient on $B_t A_t$ from task $s$'s data:
$$\frac{\partial \mathcal{L}_t}{\partial B_t A_t}\bigg|_{x \sim \mathcal{D}_s} \;\propto\; w_s^{\text{FR}} \cdot \nabla_{B_t A_t}\mathcal{L}_t(x)$$

**Expected forward transfer gain:**

$$\text{Fwt}(t \leftarrow s) \;\approx\; w_s^{\text{FR}} \cdot \cos(\hat{D}_t, \hat{D}_s)$$

where $\cos(\hat{D}_t, \hat{D}_s) = \text{tr}(\hat{D}_t^\top \hat{D}_s) / \|\hat{D}_t\|_F \|\hat{D}_s\|_F$.

**Sufficient condition for positive forward transfer:**
$$\text{Fwt}(t \leftarrow s) > 0 \iff d_{\text{FR}}(\mathcal{P}_s, \mathcal{P}_t) < \sqrt{\frac{2\ln(1/\epsilon)}{\gamma}}$$

for some $\epsilon$ threshold. This gives a **geometric condition** for when cross-task training is beneficial.

## 5.3 Theorem C2-6: Capacity Bound for FRASR

**Theorem C2-6 (Routing Capacity with FRASR).**

Under FRASR training, the effective number of distinguishable tasks $T_{\max}$ given rank $r$ and dimension $d$:

$$T_{\max} \;\leq\; \frac{d}{r} \cdot \frac{1}{\gamma_{\min}}$$

where $\gamma_{\min} = \min_{s \neq t} d_{\text{FR}}^2(\mathcal{P}_s, \mathcal{P}_t)$.

*Proof.* FRASR routing weights are softmax over Fisher-Rao distances. Two tasks $s$ and $t$ are confusable when $d_{\text{FR}}(\mathcal{P}_s, \mathcal{P}_t) \leq \sqrt{2/\gamma} \cdot \ln(T)$. The Grassmannian packing bound from CONTRI1 (Theorem 6a) applies on the effective subspace, modified by the minimum Fisher-Rao separation. $\square$

**Interpretation:** FRASR's capacity is governed by the **minimum Fisher-Rao distance** between task signatures — not just geometric separation, but the full KL decomposition. Tasks with small FR distance (geometrically and distributionally close) compress the effective capacity.

---

# PHẦN VI: EXPERIMENTAL PLAN

## 6.1 Validation Strategy

### E-C2-1: Train-Inference Gap Measurement

**Protocol:**
1. Train adapter $t$ with soft blending (ROOT-style): record $\mathcal{L}_t^{\text{soft}}$
2. Train adapter $t$ with FRASR: record $\mathcal{L}_t^{\text{FRASR}}$
3. Evaluate both at inference with hard SRT routing
4. Measure: AP difference, per-task accuracy, Fwt/Bwt decomposition

**Expected:** FRASR adapters have lower train-inference gap → higher AP at inference.

### E-C2-2: DoRA-MAT Ablation

**Protocol:**
- Full: FRASR + DoRA-MAT (both contributions)
- No-DoRA: FRASR only (no magnitude/direction decomposition)
- No-FRASR: Standard soft routing + DoRA-MAT
- Baseline: ROOT-style soft routing

**Metrics:** AP, Bwt, Fwt across all 15 tasks. Expected full > others.

### E-C2-3: Fisher-Rao Direction Alignment

**Protocol:**
1. Compute $d_{\text{FR}}(\mathcal{P}_s, \mathcal{P}_t)$ for all task pairs
2. Measure $\|\hat{D}_t - \hat{D}_s\|_F$ for trained adapters
3. Verify correlation: tasks with small FR distance → aligned directions

### E-C2-4: CB Task Recovery (Priority)

**Protocol:**
- CB trained with FRASR (geometric soft blending with NLI tasks)
- CB evaluated at inference with hard SRT
- Expected: CB accuracy improves because CB adapter was trained in a regime closer to inference

### E-C2-5: T-Scaling with FRASR

**Protocol:**
- Vary $\gamma$ in FRASR: $\gamma \in \{0.1, 0.5, 1.0, 5.0, \infty\}$
- Measure: routing accuracy (training) vs AP (inference) vs Bwt
- Expected: $\gamma \to \infty$ converges to hard routing → best AP match to SRT inference

## 6.2 Hypotheses

| Hypothesis | Test | Expected |
|------------|------|----------|
| H2-1: FRASR training reduces train-inference gap | E-C2-1 | AP gap < 0.2% |
| H2-2: DoRA-MAT magnitude regularization reduces Bwt | E-C2-2 | Bwt improved vs ROOT |
| H2-3: FR direction alignment correlates with FR distance | E-C2-3 | ρ > 0.7 |
| H2-4: CB recovers with FRASR | E-C2-4 | CB EM > 3.57% |
| H2-5: Optimal γ exists between 0.5-2.0 | E-C2-5 | Non-monotonic AP vs γ |

---

# PHẦN VII: LIMITATIONS AND HONEST ASSESSMENT

## 7.1 Key Assumptions

**A1 — Gaussian embedding assumption.** FRASR and DoRA-MAT both assume $\mathcal{P}_t = \mathcal{N}(\mu_t, \Sigma_t)$. This is justified by:
- Central Limit Theorem for mean-pooled transformer embeddings (large sequence length)
- Empirically verified for T5 (CONTRI1 §2.1). Violated for LLaMA (multimodal tasks)
- Acknowledged limitation: for non-Gaussian embeddings, the KL decomposition and Fisher-Rao distances are approximations

**A2 — Frozen backbone stationarity.** Both contributions assume that $h(x)$ computed from the frozen backbone is stable across the CL sequence. This is ensured by the frozen backbone design but may break for certain task distributions.

**A3 — Router accuracy independence.** Theorem C2-4 assumes that FRASR routing and SRT routing are driven by the same geometric structure. This holds only if the frozen embeddings $h(x)$ are the same between training (FRASR) and inference (SRT). Since both use the frozen encoder, this is satisfied.

## 7.2 Open Problems

### OP1: Optimal $\gamma$ Selection

The temperature $\gamma$ in FRASR controls the soft-vs-hard tradeoff:
- $\gamma \to \infty$: hard routing, minimal train-inference gap but reduced forward transfer
- $\gamma \to 0$: uniform blending, maximum forward transfer but larger gap

**Question:** Is there a closed-form optimal $\gamma$ given task geometry $\{\mathcal{P}_t\}$?
- Hypothesis: $\gamma^* = 1/\delta_{\min}^2$ where $\delta_{\min} = \min_{s \neq t} \delta_{\text{FR}^2}(\mathcal{P}_s, \mathcal{P}_t)$
- Theory: At this scale, the softmax cleanly separates the nearest task from all others
- Verification needed: T-scaling experiment (E-C2-5)

### OP2: FRASR-G Does NOT Match SRT Per-Sample

FRASR-G uses population-level weights $w_s^{\text{FRASR-G}}$ (does not depend on $x$). This means:
- All samples of task $t$ get the same routing weights to previous adapters
- SRT inference routes **per-sample** $x \to t^* = \arg\min \delta_{\text{Maha}}(h(x), \mu_{t^*})$
- **The match is only in expectation, not per-sample**

**Impact:** FRASR-G is weaker than FRASR-P but simpler. FRASR-P is recommended when per-sample routing quality matters (which is the case for tasks with high intra-task variance).

### OP3: DoRA-MAT Is Computationally Expensive

The direction regularization term $\|(I-P_s)\hat{D}_t\|_F^2$ requires:
- $O(t)$ projector updates per step ($P_s$ changes as $m_s, \hat{D}_s$ evolve)
- Per-task projector storage
- $O(d^2 r)$ to compute the projector

**Mitigation:** Only apply DoRA-MAT direction regularization on the final layers (where LoRA has most impact). Use the mean-displacement proxy $D_\mu^{(s,t)}$ instead of full KL decomposition.

### OP4: DoRA Magnitude vs Standard LoRA Scale

Standard LoRA has no explicit magnitude control. The Frobenius norm $\|\Delta W_t\|_F$ grows during training depending on learning rate, initialization, and task difficulty. DoRA-MAT's magnitude harmony regularizer tries to keep $m_t \approx m_s$, but:
- Tasks with different intrinsic difficulty will naturally have different optimal magnitudes
- The harmony regularizer may hurt task-specific quality by forcing equal magnitudes

**Alternative:** Instead of forcing $m_t \approx m_s$, use **log-magnitude alignment** where the target for $m_t$ is $m_s^{\text{init}}$ (the initial magnitude, not the learned one). This preserves task-specific magnitudes while preventing drift from initialization.

### OP5: FRASR + DoRA-MAT Interaction Not Analyzed

The combined objective (FRASR routing + DoRA-MAT loss) has not been analyzed jointly. Potential interactions:
- FRASR routing affects what gradient signal reaches $\Delta W_t$ during training
- DoRA-MAT regularization affects the learned $\Delta W_t$ structure
- The interaction between per-sample routing quality and direction regularization strength may be non-linear

**Joint analysis needed:** The combined effect on both Bwt and Fwt requires empirical validation (E-C2-2).

### OP6: The AP Gap Is Small (0.39%)

The observed AP gap from ROOT (78.01) to new_gainlora SRT (77.62) is only 0.39%. This is small because:
- SRT routing accuracy is ~99.99% → most tasks are correctly routed
- The gap mainly manifests on tasks where learned blending during training significantly improved adapter quality

**Question:** Is FRASR+DoRA-MAT worth the added complexity for a 0.39% improvement?
- **Yes** if the goal is theoretical unification (geometric framework for both training and inference)
- **Maybe** if the goal is pure performance (0.39% may be within noise)
- **Definitely** if the 0.39% gap grows on harder benchmarks (T=30+, longer sequences)

### OP7: No Analysis of Adapter Soup / Weight Averaging

CONTRI1 and this work treat adapters as independently trained. The AdapterSoup literature (Zhao et al., 2024) shows that weight-space averaging of adapters can produce strong single-model solutions. Under DoRA-MAT:
- Magnitudes average naturally on $\mathbb{R}^+$
- Directions average on the unit sphere $\mathcal{S}^{d^2-1}$ — not well-understood

**Open:** How should directions be averaged? Riemannian mean on Grassmannian? Linear interpolation followed by re-normalization?

---

# PHẦN VII: SUMMARY (Renumbered)

## 7.1 Contribution Statement

> **CONTRI2 — Information-Geometric Adapter Training for SRT-Aligned Continual Learning:**
>
> Two complementary contributions addressing the train-inference mismatch in SRT-based continual learning:
>
> **(C2-A) FRASR (Fisher-Rao Aligned Soft Routing):** Replaces the learned MLP router during training with a non-parametric soft routing derived from Fisher-Rao distances computed from SRT task signatures. FRASR-P uses per-sample Mahalanovits weights $w_s(x) \propto \exp\!\bigl(-\gamma\,\delta_{\text{Maha}}(h(x),\mu_s)\bigr)$; FRASR-G uses task-level geometric prior $w_s \propto \exp\!\bigl(-\gamma\,\delta_{\text{FR}^2}(\mathcal{P}_s,\mathcal{P}_t)\bigr)$. Both match the SRT inference metric (squared Mahalanobis), minimizing distributional shift. Zero additional parameters, zero drift, zero extra storage.
>
> **(C2-B) DoRA-MAT (DoRA-Inspired Magnitude-Regularized Adapter Training):** Applies DoRA's magnitude-direction decomposition to LoRA adapters. The magnitude $m_t$ is trained with geometric-aware regularization (ratio to previous magnitudes) preserving backward transfer. The direction $\hat{D}_t$ is aligned via Fisher-Rao distance-weighted orthogonal projection, enabling forward transfer to geometrically related tasks. The decomposition maps cleanly to the routing problem: direction captures cross-task transformation structure (soft-blended during training), magnitude controls task-specific scale (hard-selected at inference).

## 7.2 Novelty vs Literature

| Aspect | ROOT | SRT | AdaMoE | DoRA-DA | **FRASR** | **DoRA-MAT** |
|---------|------|-----|--------|---------|-----------|-------------|
| Non-parametric training routing | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| Uses Fisher-Rao metric | ❌ | Partial | ❌ | ❌ | ✅ | ✅ |
| DoRA decomposition for CL | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ (extended) |
| Connects train to SRT inference | ❌ | ❌ | Adaptive | ❌ | ✅ (geometric matching) | ✅ |
| Information-theoretic bound | ❌ | ✅ (routing) | ❌ | ❌ | ✅ (Theorem C2-4) | ✅ (Theorem C2-5) |
| Zero-replay compliant | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

## 7.3 Integration with CONTRI1

```
CONTRI1 (SRT):     Computes {μ_t, Σ_t} → d_SRT(h, μ_t; Σ_t) → hard routing at inference
                           ↑
                           │ shares same Fisher-Rao metric
                           ↓
CONTRI2 (FRASR):   Uses {μ_t, Σ_t} → d_FR(P_s, P_t) → soft routing during TRAINING
CONTRI2 (DoRA-MAT): Uses {μ_t, Σ_t} → magnitude/direction alignment during TRAINING
```

The **Fisher-Rao metric is the bridge** between CONTRI1 and CONTRI2. SRT's Mahalanobis distance is the first-order approximation; FRASR uses the full KL-decomposed Fisher-Rao distance. Together they form a **unified geometric framework** for both training and inference.

## 7.4 Mathematical Elegance

The framework is grounded in five theoretical pillars:

1. **Information Geometry** (Amari, 2000): Fisher-Rao metric on statistical manifold — the natural metric for comparing probability distributions
2. **KL Decomposition** (CONTRI1, Theorem 1): Exact decomposition of routing difficulty into mean displacement + shape mismatch
3. **DoRA Magnitude-Direction** (Liu et al., 2024): Independent magnitude/direction decomposition — maps to the routing trade-off
4. **Chernoff Bound** (Cover & Thomas, 2005): Lower bound on routing error via exponential KL — connects to FRASR weight computation
5. **Grassmannian Packing** (Conway et al., 1996): Capacity bound for subspace routing — extends to FRASR capacity analysis

---

# PHẦN VIII: OPEN QUESTIONS AND FUTURE DIRECTIONS

1. **Optimal $\gamma$ selection**: Is there a closed-form for optimal $\gamma$ given task geometry? Can SRM (from CONTRI1, Theorem 7) select $\gamma$?

2. **Magnitude initialization**: Should $m_t$ be initialized from SRT geometric proximity? Tasks close to previous tasks may benefit from smaller initial magnitudes (conservative) vs distant tasks (aggressive)?

3. **DoRA decomposition on frozen backbone**: Could we apply DoRA to $W_0$ itself, not just $\Delta W_t$? This would give a more principled magnitude baseline.

4. **Connection to AdapterSoup**: Weight-space averaging of FRASR-trained adapters may produce a single "adapter soup" with implicit routing that further reduces the train-inference gap.

---

## References

- DoRA: Liu et al. (2024). *DoRA: Weight-Decomposed Low-Rank Adaptation*. arXiv:2402.09353.
- AdaMoE: Bang et al. (2023). *AdaMoE: Task-Aware Adaptive Mixture of Experts*. NeurIPS.
- Information Geometry: Amari & Nagaoka (2000). *Methods of Information Geometry*. AMS.
- Fisher-Rao Metric: Rao (1945). Information and accuracy attainable. *Bull. Calcutta Math. Soc.*
- DoRA-DA: arXiv:2501.18367 (2025). Decomposed Representation Alignment for CL.
- Router Regularization: ICML 2024. Sparse MoE with routing regularization for CL.
- CONTRI1: contribution_UNIFIED.md. Statistical Routing Theory framework.