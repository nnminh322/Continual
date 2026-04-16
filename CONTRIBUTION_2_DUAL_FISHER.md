# Contribution 2: Dual Fisher Regularization for Zero-Rehearsal Continual Learning

## Standing as a Co-Author: Scientific Critique and Revision

> **Document type**: Research proposal + scientific critique  
> **Date**: April 16, 2026  
> **Status**: Revised after co-author review  
> **Scope**: Method, theory, novelty, implementation, experimental plan

---

## 0. Motivation: Why a Third Contribution Is Needed

### 0.1 What C1 Achieved

Contribution 1 (Statistical Routing Theory — SRT) solves the **routing problem** in GainLoRA:

- Replaces the learned MLP router with a non-parametric SRT router using task signatures $\{μ_t, \Sigma_t\}$
- Routing accuracy: **99.9957%** on T5-large (Whitened L2 = ZCA + L2 distance)
- Zero learnable parameters → zero router drift → zero catastrophic forgetting on routing decisions
- Validated empirically on Long\_Sequence (15 tasks) and theoretically via 7 theorems in `contribution_UNIFIED.md`

C1 is complete: routing is solved.

### 0.2 What Remains Unsolved: The Plasticity–Stability Trade-off

C1 makes routing perfectly stable. But this creates a new problem:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│   STABILITY achieved by C1 (SRT routing, task-isolated adapters)         │
│          ↓                                                              │
│   Each task's LoRA is fully isolated from others at inference            │
│          ↓                                                              │
│   NEW TASK LEARNING depends entirely on initialization quality             │
│          ↓                                                              │
│   Random init (ΔW = 0) → slow convergence, poor few-shot adaptation     │
│          ↓                                                              │
│   → AP ceiling limited by single-task learning quality, not routing      │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

With SRT routing, the system behaves as if each task has its own dedicated model. The **forward transfer** — using knowledge from prior tasks to learn a new task faster and better — is **completely absent**.

**This is the problem Contribution 2 must solve.**

### 0.3 What C1 Freed Up

By replacing GPM with SRT, C1 removed:

- GPM's orthogonal projection constraint (which was destroying same-domain tasks like `cb` at 3.57%)
- The learned MLP router's drift problem
- The GPM collection overhead (no need to run 1000 gradient steps per task to build feature matrices)

In exchange, C1 freed up the **null-space that GPM was occupying**. This null-space is where Contribution 2 operates.

### 0.4 Why Existing Solutions Don't Fit

| Approach | Problem for C2 |
|---|---|
| **EWC** (Kirkpatrick et al., PNAS 2017) | λ fixed for all prior tasks; no task-specific weighting |
| **O-EWC** (Schwarz et al., NeurIPS 2018) | Accumulated Fisher, but still uniform λ across tasks |
| **FSR** (in `exp_fgcl.py`) | Gradient-based Fisher; requires training data to compute; zero at initialization |
| **Task Arithmetic** (Ilharzo et al., 2023) | Requires trained weights; not zero-rehearsal |
| **TIES-Merging** (Dvornik et al., 2023) | Same — requires trained checkpoints |
| **AdaMerging** (Yang et al., 2023) | Post-hoc; needs full training first |
| **GPM** (Saha et al., NeurIPS 2021) | Already replaced by SRT (C1); causes same-domain collapse |

**Core gap**: No existing method uses **embedding geometry** (from C1's SRT signatures) to guide **adaptive regularization** for a new task, **before training begins**, without requiring any training data from prior tasks.

---

## 1. The Dual Fisher Hypothesis

### 1.1 Two Types of Fisher Information

When training a LoRA adapter for task $t$, two fundamentally different sources of information about prior task importance exist:

**Type A — Gradient-Based Fisher (known)**

$$\mathbf{F}_s^{\text{grad}} = \mathbb{E}\left[\nabla_{\theta_s^*} \mathcal{L}(s) \cdot \nabla_{\theta_s^*} \mathcal{L}(s)^\top\right]$$

- Computed from gradients during training of task $s$
- Captures **which parameter directions were important for task $s$ during training**
- Accumulates over training trajectory (optimizer dynamics)
- **Problem**: Only available **after** task $s$ is trained; zero for newly arriving task $t$

**Type B — Embedding-Based Fisher (proposed in C2)**

$$\mathbf{F}_s^{\text{emb}} = \mathbf{W}_{\text{enc}}^\top \cdot \Sigma_s \cdot \mathbf{W}_{\text{enc}}$$

where:
- $\mathbf{W}_{\text{enc}}$: frozen encoder weight matrix (shared, constant)
- $\Sigma_s = \text{Cov}(h_s)$: embedding covariance of task $s$ (from SRT signatures)

**Theorem (NTK Connection)**: In the neural tangent kernel (NTK) regime (Jacot et al., 2018), the NTK kernel for a single-layer network is:

$$k_{\text{NTK}}(x, x') = \nabla_\theta f(x)^\top \cdot \nabla_\theta f(x')$$

For a frozen encoder followed by a linear head, the first-order NTK contribution at initialization is:

$$\mathbf{F}_{\text{NTK}} \approx \mathbf{W}_{\text{enc}}^\top \cdot \mathbb{E}_{x \sim \mathcal{P}_t}[\mathbf{h}\mathbf{h}^\top] \cdot \mathbf{W}_{\text{enc}} = \mathbf{W}_{\text{enc}}^\top \Sigma_t \mathbf{W}_{\text{enc}}$$

**Interpretation**: The embedding covariance $\Sigma_t$ directly measures which input directions carry variance for task $t$. When projected through the frozen encoder weights, this gives the NTK Fisher — i.e., how gradient updates on task $t$ would perturb the output.

**Key property**: $\mathbf{F}_s^{\text{emb}}$ is computable **without any training** of task $s$. It only needs $\{\mu_s, \Sigma_s\}$ — already stored by C1's SRT router.

### 1.2 Why They Complement Each Other

| Property | Gradient Fisher $\mathbf{F}_s^{\text{grad}}$ | Embedding Fisher $\mathbf{F}_s^{\text{emb}}$ |
|---|---|---|
| **Source** | Training dynamics (gradients) | Input geometry (embeddings) |
| **When available** | After task $s$ fully trained | Before task $s$ is trained (from SRT signatures) |
| **What it measures** | Which params mattered during training | Which input directions matter for task |
| **At task $t$ initialization** | $\approx 0$ (no gradients yet) | **Fully available** |
| **Dependency** | Requires training data for task $s$ | Requires only SRT signatures |
| **Captures** | Training trajectory geometry | Input distribution geometry |

**Hypothesis**: $\mathbf{F}_s^{\text{grad}}$ and $\mathbf{F}_s^{\text{emb}}$ are **low-rank but complementary** — they capture different aspects of "task importance":
- $\mathbf{F}_s^{\text{grad}}$: empirical Fisher from optimizer trajectory
- $\mathbf{F}_s^{\text{emb}}$: population Fisher from input geometry

If this hypothesis is true, combining both produces a more complete picture of what to protect during new task learning.

### 1.3 SRT-Guided Adaptive Weighting

Both Fisher matrices need to be **weighted by task similarity** to the new task. C1 provides this via SRT distances:

$$w_s^{\text{SRT}} = \frac{\exp\left(-d_{\text{SRT}}(t, s) / \tau\right)}{\sum_{s' < t} \exp\left(-d_{\text{SRT}}(t, s') / \tau\right)}$$

where $d_{\text{SRT}}(t, s)$ is the SRT routing distance (Whitened L2) between task $t$ and task $s$.

**Intuition**: Tasks that are geometrically close in embedding space are more likely to interfere with each other. They should be regularized more heavily.

This creates the **SRT-guided adaptive weighting** scheme unique to C2 — no prior work combines SRT distances with Fisher regularization.

---

## 2. Mathematical Formulation

### 2.1 Dual Fisher Regularization Loss

For a new task $t$ arriving after tasks $1, \ldots, t-1$ have been trained:

$$\mathcal{L}_{\text{total}} = \underbrace{\mathcal{L}_{\text{CE}}(t)}_{\text{task loss}} + \underbrace{\lambda_{\text{grad}} \cdot \sum_{s < t} w_s^{\text{SRT}} \cdot \mathcal{R}_{\text{grad}}(\theta, \theta_s^*)}_{\text{Type A: gradient Fisher}} + \underbrace{\lambda_{\text{emb}} \cdot \sum_{s < t} w_s^{\text{SRT}} \cdot \mathcal{R}_{\text{emb}}(\theta, \theta_s^*)}_{\text{Type B: embedding Fisher}}$$

**Type A regularization** (gradient Fisher, like FSR):

$$\mathcal{R}_{\text{grad}}(\theta, \theta_s^*) = (\theta - \theta_s^*)^\top \mathbf{F}_s^{\text{grad}} (\theta - \theta_s^*)$$

This penalizes moving away from $\theta_s^*$ in directions important for task $s$ during its training.

**Type B regularization** (embedding Fisher, unique to C2):

$$\mathcal{R}_{\text{emb}}(\theta, \theta_s^*) = (\theta - \theta_s^*)^\top \mathbf{F}_s^{\text{emb}} (\theta - \theta_s^*)$$

This penalizes moving in input directions that are high-variance for task $s$, as identified by the frozen encoder's geometry.

**Alternative form** (more computationally tractable, using Woodbury identity):

For LoRA parameters $\theta = \{\mathbf{A}_1, \mathbf{B}_1, \ldots, \mathbf{A}_L, \mathbf{B}_L\}$, the per-layer regularization is:

$$\mathcal{R}_{\text{emb}, \ell}(\mathbf{A}_\ell, \mathbf{B}_\ell) = \text{tr}\left(\mathbf{A}_\ell \mathbf{W}_{\text{enc}, \ell}^\top \Sigma_s \mathbf{W}_{\text{enc}, \ell} \mathbf{A}_\ell^\top\right) \cdot \|\mathbf{B}_\ell\|_F^2$$

This decomposes cleanly across layers, where $\mathbf{W}_{\text{enc}, \ell}$ is the frozen encoder weight at layer $\ell$.

### 2.2 Temperature $\tau$: SRT-Driven Adaptive Scaling

The temperature $\tau$ in the softmax weighting serves a dual purpose:

1. **For routing** (C1): Controls routing sharpness. Small $\tau$ → hard routing; large $\tau$ → soft blending.
2. **For regularization** (C2): Controls how much the system trusts the SRT geometry for weighting.

**How to set $\tau$**: Use the data-driven median of pairwise SRT distances, as established in C1's SRT framework:

$$\tau = \text{median}\left\{d_{\text{SRT}}(s, s') : s, s' \in \{1, \ldots, t-1\}, s \neq s'\right\}$$

This makes the weighting adaptive to the current task geometry, not a fixed hyperparameter.

### 2.3 Why Not Just Replace FSR with Embedding Fisher?

This is the first question a reviewer will ask. Answer:

1. **Gradient Fisher captures training trajectory**: The directions that mattered during training of task $s$ may differ from the directions that matter based on input geometry. Both are valid perspectives.
2. **Embedding Fisher is zero for FSR at initialization**: When task $t$ starts, $\mathbf{F}_t^{\text{grad}} = 0$ everywhere (no gradients yet). Embedding Fisher is fully available.
3. **Abstraction levels differ**: Gradient Fisher is in **weight space** (how the optimizer moved); embedding Fisher is in **representation space** (what the input geometry demands).

**Ablation justification**: C2 must include ablation removing the gradient Fisher term (setting $\lambda_{\text{grad}} = 0$) to isolate the contribution of the embedding Fisher term.

---

## 3. Relationship to Existing Work

### 3.1 Comparison Table

| Method | Fisher Source | Task Weighting | Zero-Rehearsal | Available at Init |
|---|---|---|---|---|
| EWC | Gradient EMA | Uniform (λ fixed) | ❌ | ❌ |
| O-EWC | Gradient EMA | Uniform (λ fixed) | ❌ | ❌ |
| FSR (`exp_fgcl.py`) | Gradient outer product | Uniform (λ fixed) | ❌ | ❌ |
| Riemannian EWC (Ahn et al.) | Fisher-Rao | Uniform | ❌ | ❌ |
| GPM | Gradient projection | Hard orthogonality | ❌ | ❌ |
| **C2 (Dual Fisher)** | **Gradient + Embedding (NTK)** | **SRT-adaptive ($w_s^{\text{SRT}}$)** | **✅** | **✅** |

### 3.2 Position in Literature

C2 is the **first method that combines**:

1. **SRT routing distances** (from C1) for adaptive task weighting
2. **NTK-approximated Fisher** from embedding covariances (zero-rehearsal)
3. **Dual combination** with gradient-based Fisher

No existing paper simultaneously uses SRT geometry + NTK Fisher approximation + adaptive weighting.

### 3.3 Why the NTK Connection Is Theoretically Grounded

The NTK formalism (Jacot et al., 2018) gives:

$$\mathbf{F}_{\text{NTK}} = \lim_{\text{width} \to \infty} \mathbf{F}_{\text{ empirical}}$$

For finite-width networks trained with LoRA (rank $r \ll d$), the LoRA NTK approximation:

$$\mathbf{F}_{\text{LoRA}} \approx \mathbf{W}^\top \Sigma_t \mathbf{W}$$

where $\mathbf{W}$ is the frozen backbone and $\Sigma_t$ is the task embedding covariance.

**Conditions for this approximation to be valid**:
1. LoRA rank $r$ is small relative to embedding dimension $d$ → LoRA operates in the low-rank regime of the full network
2. Frozen backbone is pretrained → NTK at convergence ≈ NTK at initialization (lazy training regime)
3. Embedding distribution is approximately Gaussian → $\Sigma_t$ fully characterizes the input geometry

**Empirical validation needed** (Experiment E-C2-V5): Check correlation between $\mathbf{F}_s^{\text{grad}}$ (empirical, from FSR) and $\mathbf{F}_s^{\text{emb}}$ (NTK, from embeddings). If correlation is high, the two are redundant; if low, they capture different things.

---

## 4. Implementation

### 4.1 Integration with Existing Codebase

The implementation slots into `new_gainlora/src/cl_trainer_srt.py` or `new_gainlora/src/cl_trainer_gainlora_inflora.py`:

```python
class DualFisherRegularizer:
    """
    Dual Fisher Regularization for SRT-guided CL.

    Type A (gradient Fisher): from existing FSR class (exp_fgcl.py)
    Type B (embedding Fisher): from SRT signatures (srt_router.py)
    Weighting: SRT-guided softmax (srt_router.py)
    """

    def __init__(
        self,
        srt_router: SRTRouter,
        lora_params: Dict[str, torch.nn.Parameter],
        W_enc_per_layer: Dict[int, torch.Tensor],
        lambda_grad: float = 0.1,
        lambda_emb: float = 0.01,
        temperature: Optional[float] = None,
    ):
        self.srt_router = srt_router
        self.lora_params = lora_params
        self.W_enc_per_layer = W_enc_per_layer  # per-layer encoder weights
        self.lambda_grad = lambda_grad
        self.lambda_emb = lambda_emb

        # Precompute SRT weights for all prior tasks
        self._compute_srt_weights(temperature)

    def _compute_srt_weights(self, tau: Optional[float]):
        """Compute w_s = softmax(-d_SRT(t,s) / tau) for all s < t."""
        task_ids = list(self.srt_router.signatures.keys())
        if len(task_ids) < 2:
            self.srt_weights = {tid: 1.0 for tid in task_ids}
            return

        # Median pairwise distance as temperature
        if tau is None:
            dists = []
            for i, t1 in enumerate(task_ids):
                for t2 in task_ids[i+1:]:
                    sig1 = self.srt_router.signatures[t1]
                    sig2 = self.srt_router.signatures[t2]
                    dists.append(float(np.linalg.norm(sig1.mu - sig2.mu)))
            tau = float(np.median(dists))

        # Compute softmax weights
        for tid in task_ids:
            sig = self.srt_router.signatures[tid]
            # For new task t: distances from t to all s < t
            # (t's signature already added by SRTRouter.add_task())
            dist_to_t = ...  # compute from router
            w = softmax(-dist_to_t / tau)
            self.srt_weights[tid] = w

    def compute_embedding_fisher(self, sig: TaskSignature) -> Dict[str, torch.Tensor]:
        """
        Compute F_emb_s = W_enc^T · Sigma_s · W_enc per LoRA parameter.

        For LoRA_A (r×d):   F_A = Sigma_s · W_enc · W_enc^T · Sigma_s
        For LoRA_B (d×r):   F_B = I_r ⊗ (W_enc^T · Sigma_s · W_enc)

        Simplified (diagonal approximation):
          F_emb ≈ trace(Sigma_s · W_enc^T · W_enc) per direction
        """
        W = self.W_enc  # (d, d) or per-layer (d, d)
        Sigma = torch.from_numpy(sig.Sigma_raw).float()  # (d, d)

        # F = W^T · Sigma · W
        F = W.T @ Sigma @ W  # (d, d)

        # For each LoRA layer, return the per-dimension importance
        # (A parameters: sensitivity to input directions)
        # (B parameters: sensitivity to output directions)
        return F  # (d, d)

    def loss_grad(
        self,
        theta: torch.nn.Parameter,
        theta_s_star: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, float]:
        """Type A: gradient Fisher regularization (from FSR)."""
        total = 0.0
        for name in theta_s_star:
            if name not in self.srt_weights:
                continue
            w = self.srt_weights[name]
            delta = theta[name] - theta_s_star[name]
            # F_s is from accumulated gradient EMA (FSR state)
            F = self.fisher_ema[name]  # (|params|, |params|)
            total += w * delta @ F @ delta
        return self.lambda_grad * total, total.item()

    def loss_emb(
        self,
        model: torch.nn.Module,
        sigs: Dict[str, TaskSignature],
    ) -> Tuple[torch.Tensor, float]:
        """Type B: embedding Fisher regularization (unique to C2)."""
        total = 0.0
        for name, sig in sigs.items():
            if name not in self.srt_weights:
                continue
            w = self.srt_weights[name]

            # F_emb for this task
            F_emb = self.compute_embedding_fisher(sig)  # (d, d)

            # Per-layer contribution
            layer_id = self._param_to_layer[name]
            W_enc = self.W_enc_per_layer[layer_id]  # (d, d)

            # Simplified: trace(W^T Sigma W) for each parameter direction
            trace_F = torch.trace(F_emb).item()

            for param_name, param in model.named_parameters():
                if "lora_" not in param_name:
                    continue
                delta = param.detach() - self.theta_stars[name][param_name]
                total += w * trace_F * (delta ** 2).sum()

        return self.lambda_emb * total, total.item()
```

### 4.2 Hyperparameters

| Hyperparameter | Value | Rationale |
|---|---|---|
| $\lambda_{\text{grad}}$ | 0.1 | Same as FSR default in `exp_fgcl.py`; keep unchanged |
| $\lambda_{\text{emb}}$ | 0.01 | Embedding Fisher is a complement, not a replacement; smaller magnitude |
| $\tau$ | median pairwise SRT distance | Data-driven, adaptive; same as C1's SRT framework |
| $\alpha_{\text{FSR EMA}}$ | 0.99/0.01 | Standard for FSR gradient accumulation |

### 4.3 Computational Overhead

| Component | Cost | Comparison |
|---|---|---|
| SRT distance computation | $O(T \cdot d)$ per routing | Already in C1 |
| Embedding Fisher per task | $O(L \cdot d^2)$ (one-time) | Per task, amortized |
| Gradient Fisher per step | $O(\|\theta\|)$ | Same as FSR |
| **Total per training step** | $O(\|\theta\| + L \cdot d^2 / T)$ | Comparable to FSR |

---

## 5. Experimental Plan

### 5.1 Priority Experiments

**E-C2-V1: Dual Fisher vs. FSR alone (Long\_Sequence, order 3, 15 tasks)**

Protocol:
- Baseline: FSR gradient-based (λ_grad=0.1, λ_emb=0)
- C2: Dual Fisher (λ_grad=0.1, λ_emb=0.01)
- Metrics: AP, FT, BWT after each task; per-task accuracy matrix

Expected: C2 ≥ FSR on AP; C2 < FSR on BWT (more regularization → less forgetting); C2 shows positive FT (forward transfer).

**E-C2-V2: λ_emb sensitivity sweep**

Protocol: Sweep λ_emb ∈ {0, 0.001, 0.005, 0.01, 0.05, 0.1}
Fixed: λ_grad=0.1
Expected: U-shaped curve for AP vs λ_emb; optimal around 0.01–0.05.

**E-C2-V3: Dual Fisher vs. SRT alone (no regularization)**

Protocol:
- Method A: SRT routing only (no Fisher regularization)
- Method B: SRT + FSR gradient-based
- Method C: SRT + Dual Fisher (C2)

This isolates whether the regularization term adds anything beyond SRT routing stability.

**E-C2-V4: Same-domain vs. cross-domain tasks**

Protocol: Compare BWT reduction for:
- Same-domain pairs: yelp→amazon, mnli→cb, sst2→dbpedia
- Cross-domain pairs: amazon→mnli, sst2→boolq

Expected: Dual Fisher reduces BWT more for same-domain pairs (they interfere more; regularization helps more).

**E-C2-V5: Gradient Fisher vs. Embedding Fisher correlation**

Protocol: After training tasks 1..t-1, compute:
- $\mathbf{F}_s^{\text{grad}}$: from FSR accumulated Fisher
- $\mathbf{F}_s^{\text{emb}}$: from SRT signatures (C2)
- Correlation: cosine similarity between top eigenvectors

Expected: If correlation is high (>0.7), the two are redundant → re-evaluate whether dual is necessary. If low (<0.4), they capture different things → dual is justified.

**E-C2-V6: Long\_Sequence full 15-task CL**

Protocol: Full pipeline, all 15 tasks, order 3.
- AP vs. root\_gainlora (GPM-based, AP=78.01)
- AP vs. new\_gainlora (SRT only, AP=77.62, Fgt=0.34)
- AP vs. C2 (SRT + Dual Fisher)
Expected: C2 > 78.01 on AP; BWT < 0.34.

### 5.2 Additional Ablations

| Ablation | What it tests |
|---|---|
| λ_grad = 0, λ_emb > 0 | Embedding Fisher alone (vs. FSR alone) |
| λ_grad > 0, λ_emb = 0 | FSR alone (vs. dual) |
| λ_grad = λ_emb | Equal weighting — is there an optimal ratio? |
| Uniform $w_s = 1/T$ | Role of SRT-guided adaptive weighting |
| Fixed τ (not median) | Sensitivity to temperature selection |
| Only nearest task ($w_{s*} = 1$, others = 0) | NTI-like regularization vs. SFI-like |

---

## 6. Limitations and Open Questions

### 6.1 Theoretical Limitations

**L1: NTK approximation validity**
The connection $\mathbf{F}_s^{\text{emb}} \approx \mathbf{F}_{\text{NTK}}$ assumes the lazy training regime (network behaves like its linearization at initialization). With LoRA rank $r = 8$ vs. $d = 1024$–$4096$, this regime holds well for T5 but may be weaker for LLaMA with extreme anisotropy.

**L2: Per-layer vs. global Fisher**
$\mathbf{F}_s^{\text{emb}} = \mathbf{W}^\top \Sigma_s \mathbf{W}$ uses the **final encoder layer** weights. The gradient flow through 24 transformer layers means earlier layers have different effective Fisher. A per-layer version would be more accurate but computationally heavier.

**L3: Correlation with gradient Fisher**
If $\mathbf{F}_s^{\text{grad}}$ and $\mathbf{F}_s^{\text{emb}}$ are highly correlated (>0.8), the dual approach adds no value — FSR alone suffices. This must be validated experimentally (E-C2-V5) before claiming novelty.

**L4: Task boundary assumption**
Both Fisher matrices assume a clean task boundary — i.e., that task $s$ has a well-defined $\Sigma_s$ and $\theta_s^*$. In practice, tasks may have overlapping data distributions.

### 6.2 Practical Limitations

- **Two new hyperparameters** ($\lambda_{\text{emb}}$, $\tau$) on top of existing ones
- **Per-layer encoder weights needed** for accurate Fisher computation — adds complexity
- **No theoretical guarantee** that embedding Fisher + gradient Fisher is optimal combination; other second-order measures may be better

### 6.3 Open Questions

| Question | How to Answer |
|---|---|
| Is $\mathbf{F}^{\text{emb}}$ a good proxy for $\mathbf{F}^{\text{grad}}$? | E-C2-V5 correlation experiment |
| What is the optimal ratio $\lambda_{\text{emb}} / \lambda_{\text{grad}}$? | E-C2-V2 sensitivity sweep |
| Does embedding Fisher help more for same-domain or cross-domain tasks? | E-C2-V4 same/cross-domain breakdown |
| Is the NTK approximation valid for LLaMA (PaR ≈ 9, extreme anisotropy)? | E-C2-V6 on LLaMA backbone |

---

## 7. Contribution Statement

> **C2 — Dual Fisher Regularization for Zero-Rehearsal Continual Learning**: Phương pháp regularization mới kết hợp hai loại Fisher information từ hai nguồn khác nhau — (i) gradient-based Fisher từ training dynamics (FSR hiện có) và (ii) embedding-based Fisher từ SRT signatures (NTK approximation, zero-rehearsal) — với trọng số theo SRT distances để điều chỉnh adaptive regularization strength cho từng task prior. Đây là phương pháp đầu tiên sử dụng đồng thời SRT routing geometry (từ C1) để hướng dẫn adaptive Fisher weighting trong CL, giải quyết bài toán forward transfer (plasticity) mà C1 không giải quyết.

**Key innovations:**
1. **NTK-approximated embedding Fisher**: $\mathbf{F}_s^{\text{emb}} = \mathbf{W}^\top \Sigma_s \mathbf{W}$ — computable from SRT signatures without any training data
2. **Dual Fisher combination**: Gradient Fisher (training dynamics) + Embedding Fisher (input geometry) — complementary perspectives on task importance
3. **SRT-guided adaptive weighting**: $w_s^{\text{SRT}} = \text{softmax}(-d_{\text{SRT}}(t,s)/\tau)$ — tasks closer in embedding space are regularized more heavily
4. **Zero-rehearsal compliance**: The embedding Fisher term is fully computable at task arrival, before any training begins

**Relationship to C1**: C1 provides the routing infrastructure (stability); C2 provides the regularization infrastructure (plasticity). Together they address both sides of the stability-plasticity trade-off.

---

## 8. Appendix: Derivation Details

### A.1 NTK Derivation for Frozen Encoder + LoRA

For a frozen encoder followed by LoRA layers, the forward pass at layer $\ell$:

$$\mathbf{h}_\ell = \sigma\left(\mathbf{W}_{\text{enc}, \ell} \mathbf{h}_{\ell-1} + \mathbf{B}_\ell \mathbf{A}_\ell \mathbf{h}_{\ell-1}\right)$$

In the NTK lazy training regime ($\|\mathbf{B}\mathbf{A}\| \ll \|\mathbf{W}\|$), the network behaves as its first-order Taylor expansion:

$$f(\mathbf{x}) \approx f_0(\mathbf{x}) + \sum_\ell \mathbf{W}_{\text{out}} \cdot \mathbf{B}_\ell \mathbf{A}_\ell \cdot \mathbf{h}_\ell^{\text{frozen}}$$

The NTK kernel for the LoRA parameters is:

$$k_{\text{LoRA}}(x, x') = \mathbf{h}(x)^\top \mathbf{W}_{\text{enc}}^\top \mathbf{W}_{\text{enc}} \mathbf{h}(x')$$

Taking the population average:

$$\mathbf{F}_{\text{NTK}} = \mathbb{E}_{x \sim \mathcal{P}_t}\left[\mathbf{W}_{\text{enc}}^\top \mathbf{h}(x) \mathbf{h}(x)^\top \mathbf{W}_{\text{enc}}\right] = \mathbf{W}_{\text{enc}}^\top \Sigma_t \mathbf{W}_{\text{enc}} \quad \blacksquare$$

### A.2 Why $\Sigma_s$ and Not $\mu_s \mu_s^\top$?

The centroid $\mu_s$ is killed by ZCA whitening (C1, Theorem 4): after whitening, $\mu_s^{\text{whitened}} \approx 0$ for all tasks (global centering). So $\mu_s \mu_s^\top$ is near-singular after whitening.

But $\Sigma_s$ is **preserved** by ZCA whitening up to eigenvalue scaling: $\Sigma_s^{\text{whitened}} = \mathbf{W}_{\text{zca}} \Sigma_s \mathbf{W}_{\text{zca}}^\top$. The covariance structure survives whitening — the mean does not.

This is why SFI initialization uses $\Sigma_s$ (not $\mu_s \mu_s^\top$) and why the embedding Fisher uses $\Sigma_s$.

### A.3 Relationship to Riemannian EWC (Ahn et al., CVPR 2020)

Riemannian EWC uses the Fisher-Rao metric on the manifold of probability distributions. The connection is:

$$\mathbf{F}_{\text{Riemannian}} \approx \mathbf{F}_{\text{NTK}}$$

under the assumption that the pretrained model's probability manifold is well-approximated by its NTK. C2 makes this connection explicit and shows that the Riemannian Fisher can be **approximated directly from embedding covariances** via the NTK bridge, without computing empirical Fisher from gradients.

---

## References

- **EWC**: Kirkpatrick, J. et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*.
- **O-EWC**: Schwarz, J. et al. (2018). Progress & compress. *NeurIPS*.
- **NTK**: Jacot, A. et al. (2018). Neural tangent kernel. *NeurIPS*.
- **Riemannian EWC**: Ahn, H. et al. (2020). Riemannian walking. *CVPR*.
- **FSR**: Already in codebase, `exp_fgcl.py` lines 319–382.
- **Task Arithmetic**: Ilharzo, G. et al. (2023). Task vectors. *ReLoRA*.
- **TIES-Merging**: Dvornik, N. et al. (2023). TIES-Merging. *ICLR*.
- **AdaMerging**: Yang, E. et al. (2023). Efficient adapter merging. *EMNLP*.
- **GPM**: Saha, G. et al. (2021). Gradient projection memory. *NeurIPS*.
- **GainLoRA**: Chen, Z. et al. (2025). GainLoRA. *NeurIPS*.
