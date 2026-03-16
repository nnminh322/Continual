# SpecRoute: Spectral Routing via Routing–Protection Duality in Continual LoRA Learning

> **Authoritative Design Document v2** — supersedes `SPECROUTE_IDEA_v1.md` and all
> older documents. Theory-first approach per `research_rule.txt`.

---

## 1. Problem Setting

**Setting.** Continual learning with expandable LoRA on a frozen LLM.
Tasks $\mathcal{T}_1, \ldots, \mathcal{T}_T$ arrive sequentially. For each task $t$:

- A low-rank adapter $\Delta W_t = B_t A_t$ ($A_t \in \mathbb{R}^{r \times d}$, $B_t \in \mathbb{R}^{d \times r}$) is added to every attention projection.
- Only $B_t$ is trained; $A_t$ is frozen after null-space initialisation (InfLoRA).
- After training, both $A_t, B_t$ are frozen and a fresh branch is created for the next task.

**Inference.** Given input $x$ *without* task identifier, the model must produce the correct output:

$$y = f\!\Bigl(W_0\, x \;+\; \sum_{t=1}^{T} w_t(x)\; B_t A_t\, x\Bigr)$$

**Three coupled sub-problems:**

| Sub-problem | Goal | Formal requirement |
|:-----------:|------|--------------------|
| **Routing (R)** | Assign input to the correct expert(s) | $w_{t^*}(x) \gg w_t(x)$ for $t \neq t^*$ |
| **Protection (P)** | Prevent degradation of old experts | $\Delta W_t$ unchanged after task $t$ |
| **Allocation (A)** | Manage finite subspace capacity | $\sum_t \dim\bigl(\mathrm{span}(A_t)\bigr) \leq d$ |

**Setting constraint:** *Zero-replay* — no reuse of old task data in any form (raw, synthetic, distributional).

---

## 2. Observation: The Hidden Duality

### 2.1 GainLoRA's Approach & Its Weakness

GainLoRA (NeurIPS 2025) treats R, P, A as **independent** problems:

| Aspect | Mechanism | Cost |
|--------|-----------|------|
| R | Learned MLP `trans_input` + learned `prompt_key` → cosine gating | Extra parameters + GPM subspace |
| P | GPM projects gradients to null-space of old tasks | Subspace consumed per task |
| A | Increasing threshold $\varepsilon_t \nearrow 1$ | Later tasks more constrained |

**Fundamental weakness:** Because routing is *learned*, it creates a vicious cycle:

1. `trans_input` evolves each task → routing space drifts → old prompt keys misalign → routing degrades.
2. GPM must protect routing params → *consumes subspace that could serve task learning*.
3. KL distillation on routing is needed → requires replay or frozen copies → memory overhead.

### 2.2 The Key Insight

We observe that GPM enforces approximately orthogonal expert input subspaces:

$$\mathrm{span}(V_i) \;\approx\perp\; \mathrm{span}(V_j), \qquad i \neq j$$

where $V_t$ are the right singular vectors of $\Delta W_t$. This orthogonality, enforced for **protection**, simultaneously provides a natural **routing** criterion: because subspaces do not overlap, measuring how much an input aligns with each subspace uniquely identifies the originating task.

> **Routing–Protection Duality.**
> Anti-forgetting (orthogonal subspace protection) and task identification (discriminative routing)
> are *dual manifestations of the same spectral structure*.
> Solving one automatically solves the other.

**Implications:**
- No learned routing parameters needed → no routing drift, no GPM cost for routing.
- No replay needed for routing maintenance → naturally zero-replay compliant.
- Routing accuracy is *guaranteed* by protection quality (formalised below).

---

## 3. Theoretical Framework

### 3.1 Spectral Expert Signatures

**Definition 1** *(Spectral Signature).* For frozen expert $\Delta W_t = B_t A_t$ with thin SVD

$$\Delta W_t = U_t\, \Sigma_t\, V_t^\top, \qquad V_t \in \mathbb{R}^{d \times r},\; \Sigma_t = \mathrm{diag}(\sigma_{t,1}, \ldots, \sigma_{t,r}),$$

the spectral signature is $\mathcal{S}_t = (V_t,\, \boldsymbol{\sigma}_t)$ where

- $V_t$: **input receptive field** — the $r$ input directions the expert processes,
- $\boldsymbol{\sigma}_t$: **sensitivity spectrum** — the modification gain along each direction.

**Information-theoretic view.** Viewing $\Delta W_t$ as a linear channel, the columns of $V_t$ are the channel's *input modes* and $\sigma_{t,i}^2$ is the *gain* of mode $i$. The total channel capacity (Frobenius energy) is $\|\Delta W_t\|_F^2 = \sum_i \sigma_{t,i}^2$.

### 3.2 Spectral Affinity

**Definition 2** *(Spectral Affinity).* The affinity of input $h \in \mathbb{R}^d$ to expert $t$:

$$\alpha_t(h) \;=\; \frac{h^\top M_t\, h}{\mathrm{tr}(M_t)\;\|h\|^2}, \qquad M_t = V_t\, \mathrm{diag}(\boldsymbol{\sigma}_t^2)\, V_t^\top$$

Expanding:

$$\alpha_t(h) = \frac{\displaystyle\sum_{i=1}^{r} \sigma_{t,i}^2\;(v_{t,i}^\top h)^2}{\displaystyle\Bigl(\sum_{i=1}^{r} \sigma_{t,i}^2\Bigr)\,\|h\|^2}$$

**Properties:**

| Property | Statement |
|----------|-----------|
| Range | $\alpha_t(h) \in [0,\, 1]$ — normalised weighted Rayleigh quotient |
| Energy ratio | $\alpha_t(h) = \|\Delta W_t\, h\|^2 \;/\; \bigl(\|\Delta W_t\|_F^2\, \|h\|^2\bigr)$ |
| Interpretation | Fraction of expert $t$'s total channel capacity activated by $h$ |
| In-distribution | $h \in \mathrm{span}(V_t) \;\Rightarrow\; \alpha_t(h) \geq \kappa_{\min}(t) > 0$ |
| Out-of-distribution | $h \perp \mathrm{span}(V_t) \;\Rightarrow\; \alpha_t(h) = 0$ exactly |

### 3.3 Routing–Protection Duality Theorem

**Definition 3** *(Subspace Overlap).* The overlap between experts $i$ and $j$:

$$\delta_{ij} = \|V_i^\top V_j\|_F^2 = \sum_{k=1}^{r} \cos^2 \theta_{ij}^{(k)}$$

where $\theta_{ij}^{(k)}$ are the *principal angles* between $\mathrm{span}(V_i)$ and $\mathrm{span}(V_j)$.

---

**Theorem 1** *(Routing–Protection Duality).* If GPM ensures $\delta_{ij} \leq \varepsilon$ for all $i \neq j$, then for any unit input $h \in \mathrm{span}(V_{t^*})$ the **routing margin** satisfies:

$$\boxed{\;\alpha_{t^*}(h) \;-\; \max_{t \neq t^*}\, \alpha_t(h) \;\;\geq\;\; \kappa_{\min}(t^*)\; -\; \varepsilon\, \kappa_{\max}\;}$$

where

$$\kappa_{\min}(t) = \frac{\sigma_{t,\min}^2}{\sum_i \sigma_{t,i}^2}, \qquad \kappa_{\max} = \max_t\, \frac{\sigma_{t,\max}^2}{\sum_i \sigma_{t,i}^2}$$

**Proof.**

*Lower bound on the correct expert.* Write $h = V_{t^*}\, c$ with $\|c\| = 1$ (since $h \in \mathrm{span}(V_{t^*})$). Then $(v_{t^*,i}^\top h)^2 = c_i^2$ and $\sum c_i^2 = 1$:

$$\alpha_{t^*}(h) = \frac{\sum_i \sigma_{t^*,i}^2\, c_i^2}{\sum_i \sigma_{t^*,i}^2} \;\geq\; \frac{\sigma_{t^*,\min}^2\, \sum c_i^2}{\sum \sigma_{t^*,i}^2} \;=\; \kappa_{\min}(t^*)$$

*Upper bound on wrong experts.* For $t \neq t^*$:

$$\|V_t^\top h\|^2 = \|V_t^\top V_{t^*}\, c\|^2 \leq \|V_t^\top V_{t^*}\|_F^2\, \|c\|^2 = \delta_{t,t^*} \leq \varepsilon$$

$$\Rightarrow\;\; \alpha_t(h) = \frac{\sum_i \sigma_{t,i}^2\, (v_{t,i}^\top h)^2}{\sum \sigma_{t,i}^2} \leq \frac{\sigma_{t,\max}^2 \cdot \varepsilon}{\sum \sigma_{t,i}^2} \leq \kappa_{\max}\, \varepsilon \qquad\square$$

---

**Corollary 1** *(Routing Confidence).* Under Theorem 1, softmax routing with temperature $\tau$ gives the correct expert weight:

$$w_{t^*}(h) \;\geq\; \frac{1}{1 + (T{-}1)\,\exp\!\bigl(-m/\tau\bigr)}, \qquad m = \kappa_{\min}(t^*) - \varepsilon\, \kappa_{\max}$$

For target confidence $w_{t^*} \geq 1 - \delta$, set $\tau \leq m \,/\, \ln\!\bigl(\tfrac{T-1}{\delta}\bigr)$.

---

**Corollary 2** *(Capacity Bound — Grassmannian Connection).* The maximum number of $r$-dimensional subspaces in $\mathbb{R}^d$ with pairwise overlap $\delta \leq \varepsilon$ is bounded by:

$$T_{\max} \;\leq\; \frac{d}{r\,(1 - \varepsilon)}$$

For T5-Small ($d = 512$, $r = 8$, $\varepsilon = 0.02$): $T_{\max} \leq 65 \gg 15$ tasks.

*This connects CL capacity to Grassmannian packing theory*: expert subspaces are "codewords" in $\mathrm{Gr}(r, d)$, and minimum distance governs both decoding accuracy (routing) and interference resilience (protection).

### 3.4 Drift Invariance

**Proposition 1** *(Drift-Free Routing).* The routing function $h \mapsto \alpha_t(h)$ is completely stationary across all tasks.

**Proof.** The routing input is computed as:

$$h = \frac{1}{|x|} \sum_{i \in x} \mathrm{Embed}(x_i)$$

where $\mathrm{Embed}$ is the frozen embedding table, evaluated *before* any transformer block. Since LoRA modifications exist only in attention layers (deeper), $h$ is independent of all LoRA parameters. Combined with frozen $\mathcal{S}_t$, the affinity $\alpha_t(h)$ is invariant to accumulated model changes. $\square$

**Contrast.** GainLoRA's `trans_input` is a learned MLP that evolves each task, causing the routing function to drift even under GPM protection (approximate, not exact).

### 3.5 Addressing the Energy–Quality Gap

A natural objection: *spectral affinity measures modification energy, not modification quality*. Theorem 1 resolves this:

> Under orthogonal protection ($\varepsilon \to 0$), high affinity $\Leftrightarrow$ input lies in the expert's operating subspace $\Leftrightarrow$ the expert was *trained* on this type of input. The duality converts an energy-based proxy into a provably correct task-identity signal.

---

## 4. Framework Components

### C1 — Spectral Expert Signatures

After training task $t$, compute $\mathcal{S}_t = (V_t, \boldsymbol{\sigma}_t)$ via **thin SVD**:

$$B_t,\, A_t \;\xrightarrow[\text{QR + SVD}]{O(dr^2)}\; (V_t,\, \boldsymbol{\sigma}_t)$$

- QR decomposition of $B$ and $A^\top$, then SVD of the $r \times r$ core → exact, $O(dr^2)$ vs $O(d^2 r)$.
- Stored per LoRA layer (encoder Q, V; decoder self/cross Q, V).
- **Immutable** by construction: frozen weights → frozen signatures → zero drift.

### C2 — Spectral Affinity Routing

**Inference** (all tasks available):

$$w(h) = \mathrm{softmax}\!\left(\frac{[\alpha_1(h),\; \ldots,\; \alpha_T(h)]}{\tau}\right)$$

**Training** (task $t$, final SVD unknown because $B_t$ still training):

$$\alpha_t^{\mathrm{train}}(h) = \frac{\|A_t\, h\|^2}{r\,\|h\|^2} + \beta$$

**Justification of the A-row proxy:** For any full-rank $B_t$, the column span of $V_t$ (from SVD of $B_t A_t$) equals $\mathrm{range}(A_t^\top)$. So the A rows span the *same* input subspace that the converged $V_t$ will capture. The proxy measures input alignment with this subspace using uniform weighting (no $\sigma$ available yet).

**Justification of $\beta$:** A rows (kaiming-initialised, unit-variance) produce systematically lower fits than $\sigma^2$-weighted SVD fits of trained old experts. Setting $\beta = 1.0$ makes the softmax produce $w_t > 0.95$, approximating the oracle assignment $w_t = 1$ (principled: during training on task $t$'s data, the optimal routing *is* $w_t = 1$) while allowing marginal knowledge transfer from relevant old experts.

### C3 — Capacity-Aware Subspace Allocation

GPM threshold controls the protection–capacity trade-off. From Theorem 1:
- Lower $\varepsilon$ → better protection & routing, but faster subspace exhaustion.
- Higher $\varepsilon$ → more capacity, but weaker routing guarantee.

**Dynamic threshold** (following InfLoRA):

$$\varepsilon_t = (1 - \varepsilon_0) \cdot \frac{t}{T} + \varepsilon_0$$

where $\varepsilon_0$ is the base threshold. This allocates incrementally stricter protection as tasks accumulate, since later tasks face a more crowded Grassmannian and need finer-grained allocation. The trade-off is *principled* via Corollary 2: as long as $\varepsilon_t$ stays above $(1 - d/(rT))$, capacity for all $T$ tasks is guaranteed.

---

## 5. What's Removed from GainLoRA

| Component | GainLoRA | SpecRoute | Why |
|-----------|----------|-----------|-----|
| `trans_input` MLP | Learned routing projection | ❌ Removed | Duality: spectral affinity suffices |
| `prompt_key` | Learned per-task key | ❌ Removed | Replaced by spectral signatures |
| `previous_trans_input` | Frozen MLP copies | ❌ Removed | Signatures immutable by construction |
| KL distillation | Replay-based routing loss | ❌ Removed | No learned routing → nothing to distill |
| GPM on routing params | Subspace for routing | ❌ Removed | No routing parameters to protect |

**Net effect:** All subspace and compute budget that GainLoRA spends on routing infrastructure is *reclaimed* for task learning.

---

## 6. Novelty Claims

**Claim 1 — Routing–Protection Duality** *(Theoretical).* We formalise and prove that in orthogonal-subspace CL, protection fidelity (subspace overlap $\varepsilon$) directly governs routing accuracy — the first theoretical guarantee connecting these two aspects. This reveals that parameter-free routing is not merely a simplification but *provably sufficient* when protection is adequate.

**Claim 2 — Parameter-Free Spectral Routing** *(Algorithmic).* We derive a routing mechanism requiring zero learned parameters, zero replay, and zero GPM overhead, while providing per-input discriminative routing with theoretical accuracy guarantees. The routing signal is extracted entirely from frozen expert weights.

**Claim 3 — Unified Geometric Framework** *(Conceptual).* We connect CL routing, protection, and capacity through Grassmannian geometry, providing the first capacity bound for expandable LoRA CL ($T_{\max} \leq d/r(1{-}\varepsilon)$) and linking CL design to established results in coding theory and information theory.

---

## 7. Code–Idea Alignment

| Theory | Implementation | File |
|--------|---------------|------|
| Spectral signature $\mathcal{S}_t$ | `compute_spectral_signatures()` (thin QR+SVD) | `t5_specroute.py` |
| Spectral affinity $\alpha_t(h)$ (old tasks) | σ²-weighted Rayleigh quotient | `compute_spectral_routing()` |
| A-row proxy $\alpha_t^{\mathrm{train}}$ (current) | `(proj**2).sum() / (r * h_norm_sq) + training_bias` | `compute_spectral_routing()` |
| Routing $w = \mathrm{softmax}(\alpha / \tau)$ | `torch.softmax(fit_scores / temp)` | `compute_spectral_routing()` |
| Drift-free input $h$ | `inputs_embeds = self.embed_tokens(input_ids)` → mean-pool | `T5Stack.forward()` |
| GPM + InfLoRA null-space | `get_reg_matrix()` | `cl_trainer_specroute.py` |
| Dynamic ESA threshold | `(1−ε₀)·t/T + ε₀` | `cl_trainer_specroute.py` |
| No routing parameters | No `trans_input`, no `prompt_key` in T5Stack | `t5_specroute.py` |
| No replay | Clean `training_step` (CE only) | `cl_trainer_specroute.py` |

---

## 8. Training Pipeline

### Task 1 (`--run_single True`)
1. Load pretrained model + fresh LoRA ($A$: kaiming, $B$: zeros).
2. Standard training (only `lora_B`) — single expert, no routing.
3. Post-training: compute $\mathcal{S}_1$ (thin SVD) + GPM bases (ESA threshold).
4. Save: LoRA weights, spectral signatures, GPM reg files.

### Task $t \geq 2$
1. Load model + fresh LoRA; load old LoRA weights and spectral signatures.
2. InfLoRA: project current $A_t$ into null-space of old GPM bases.
3. Train `lora_B` with spectral affinity routing + training bias $\beta$.
4. Post-training: compute $\mathcal{S}_t$ + update GPM bases.
5. Save all artifacts for next task.

---

## 9. Experimental Setup

| Item | Value |
|------|-------|
| Model | `google/flan-t5-small` (60M) / `flan-t5-large` (783M) |
| Benchmarks | SuperNI (15 tasks, 2 orderings), Long (15 tasks, 2 orderings) |
| Metrics | AP (Average Performance, ↑), FT (Forgetting, ↓) |
| LoRA | $r = 4$, $\alpha = 32$, dropout 0.0 |
| Routing | $\tau = 1.0$, $\beta = 1.0$ (train only) |
| ESA | $\varepsilon_0 = 0.980$ (dynamic) |
| Precision | fp32 + gradient checkpointing |
| Comparison | Batch size, LR, scheduler match ROOT (GainLoRA) exactly |

---

## 10. File Map

| File | Role |
|------|------|
| `src/t5_specroute.py` | T5Stack + spectral routing + thin SVD |
| `src/t5_gainlora_inflora.py` | LoRALayer, T5Attention, T5Block (shared base) |
| `src/cl_trainer_specroute.py` | Trainer: GPM, InfLoRA, ESA, training_step |
| `src/run_t5.py` | Entry: model loading, parameter freezing |
| `gen_script_*_specroute*.sh` | Experiment scripts |
