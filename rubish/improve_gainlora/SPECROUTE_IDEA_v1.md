# SpecRoute: Spectral Routing for Continual LoRA Learning

> **Consolidated Design Document** — combines and supersedes:
> `proposal_gainlora_upgrade.md`, `C2_analysis_and_revision.md`, `revised_idea_analysis.md`.
> Those files are now obsolete. This document matches the actual implementation.

---

## 1. Motivation & Problem Setting

### 1.1 Setting: Continual Learning with LoRA

Given a sequence of tasks $\mathcal{T}_1, \mathcal{T}_2, \ldots, \mathcal{T}_T$ arriving one at a time, we fine-tune a frozen pretrained LLM by adding low-rank adapters (LoRA) to its attention layers. After training task $t$, LoRA-A is frozen and LoRA-B is reset for the next task. At inference time, the model must correctly handle inputs from **any** previously seen task **without** task identifiers.

**Two core challenges:**
1. **Routing**: Which task's LoRA adapter(s) to activate for a given input?
2. **Forgetting**: How to protect old tasks' learned representations from degradation?

### 1.2 Problems with GainLoRA's Approach

GainLoRA (NeurIPS 2025) uses:
- A **learned MLP** (`trans_input`) to project inputs into a routing space
- A **prompt key** per task for cosine similarity-based routing
- A **GPM (Gradient Projection Memory)** with increasing thresholds to protect subspaces

**Four fundamental problems:**

| # | Problem | Consequence |
|---|---------|-------------|
| 1 | **Routing drift**: `trans_input` MLP evolves each task, so the routing space changes | Old prompt keys computed in $\mathcal{F}_i$ become misaligned with current $\mathcal{F}_t$; routing accuracy degrades |
| 2 | **Learned parameters add overhead**: `trans_input` + `prompt_key` require optimization + GPM cost | Extra memory, compute, and subspace consumed by non-task parameters |
| 3 | **Subspace exhaustion**: Hard orthogonal GPM (InfLoRA) shrinks available capacity monotonically | Task 1 gets full $d_\text{in}$ capacity; later tasks get increasingly constrained (unfair allocation) |
| 4 | **Indirect routing signal**: Cosine similarity in projected space is an indirect proxy for task identity | No guarantee that the routing signal reflects which LoRA subspace actually fits the input |

---

## 2. SpecRoute Framework

SpecRoute replaces GainLoRA's learned routing with **three parameter-free components**:

### 2.1 C1 — Spectral LoRA Signatures

**Idea**: After training task $t$, the frozen LoRA weights $\Delta W_t = B_t A_t$ encode the task's operating subspace. Extract this information via SVD.

**Method**: For each LoRA layer after task $t$ completes:

$$\Delta W_t = B_t A_t = U_t \Sigma_t V_t^\top$$

Store the **spectral signature** $\mathcal{S}_t = \{V_t^{(r)}, \sigma_t^{(r)}\}$ where:
- $V_t^{(r)} \in \mathbb{R}^{r \times d_\text{in}}$: top-$r$ right singular vectors (input directions)
- $\sigma_t^{(r)} \in \mathbb{R}^{r}$: corresponding singular values (importance weights)

**Properties (vs. GainLoRA's prompt key):**
- **Immutable**: extracted from frozen weights → zero drift, zero parameter evolution across tasks
- **Functionally grounded**: V captures the actual input directions that the LoRA processes
- **Multi-resolution**: per-layer signatures capture different levels of representation
- **Zero parameters**: no `trans_input` MLP, no `prompt_key` to train or protect

### 2.2 C2 — Projection-Based Routing

**Idea**: Measure how much of the input's energy falls into each task's LoRA subspace. Route to the best-fitting task(s) via softmax.

**Method**: Given input embedding $h$ (mean-pooled over sequence, from encoder), compute:

**For previous task $t$** (using stored spectral signature):
$$\text{fit}_t(h) = \frac{\sum_{i=1}^{r} \sigma_{t,i}^2 \, (v_{t,i}^\top h)^2}{\left(\sum_{i=1}^{r} \sigma_{t,i}^2\right) \|h\|^2}$$

This is a **weighted Rayleigh quotient**: it measures the fraction of $h$'s energy captured by task $t$'s principal input directions, weighted by their importance $\sigma^2$.

**For the current task** (LoRA-A is known but SVD not yet final):
$$\text{fit}_\text{cur}(h) = \frac{\sum_{i=1}^{r} (a_i^\top h)^2}{r \cdot \|h\|^2}$$

where $a_i$ are the (fixed) rows of the current LoRA-A matrix.

**Routing weights**:
$$w(h) = \text{softmax}\!\left(\frac{[\text{fit}_\text{cur}(h),\, \text{fit}_1(h),\, \ldots,\, \text{fit}_{T-1}(h)]}{\tau}\right)$$

where $\tau$ is a temperature hyperparameter (default 1.0).

**Properties**:
- **Parameter-free**: no learned parameters in the routing mechanism
- **Per-input**: each input gets its own routing weights (no batch-level constraint)
- **Works at batch_size=1**: unlike OT/Sinkhorn which degenerate at small batches
- **Zero overhead on GPM**: no need to protect routing parameters

**Design note**: The original proposal (in `revised_idea_analysis.md`) considered Sinkhorn OT routing. Analysis showed that OT enforces global balance constraints across tasks, which is incorrect for CL: at test time, all inputs may belong to one task. Softmax over projection fits is both simpler and semantically correct.

### 2.3 C3 — Elastic Subspace Allocation (ESA)

**Idea**: Replace InfLoRA's increasing GPM threshold with a **constant threshold** across all tasks.

**Problem with increasing threshold**: In standard GPM, the threshold $\epsilon_t$ increases over tasks (e.g., $\epsilon_1 = 0.97$, $\epsilon_T = 0.998$). This means later tasks have stricter protection, consuming more of the finite subspace. As a result:
- Task 1 gets full $d_\text{in}$ capacity
- Later tasks get severely constrained (can lose >12% capacity)
- This creates **unfair capacity allocation**

**Solution**: Use constant $\epsilon = 0.995$ for all tasks. This ensures:
- Each task's protection level is proportional to its actual activation variance
- Subspace consumption is bounded and predictable
- No unfair advantage to early tasks

**Implementation**: In `get_representation()`, `threshold = self.args.threshold` is constant (passed via `--threshold 0.995`).

---

## 3. Architecture Summary

```
┌──────────────────────────────────────────────────────┐
│                    SpecRoute T5                       │
│                                                       │
│  Encoder:                                             │
│  ┌────────────┐   ┌───────────────────────────────┐  │
│  │ Input IDs  │──▶│ Embedding → mean-pool → h     │  │
│  └────────────┘   └───────────┬───────────────────┘  │
│                               │                       │
│                   ┌───────────▼───────────────────┐  │
│                   │ Spectral Routing:              │  │
│                   │ fit_t(h) for each task         │  │
│                   │ w = softmax(fits / τ)          │  │
│                   └───────────┬───────────────────┘  │
│                               │ weights (B, T, 1)    │
│                   ┌───────────▼───────────────────┐  │
│                   │ Each Block:                    │  │
│                   │ q = W_q·x + Σ w_t·LoRA_t(x)  │  │
│                   │ v = W_v·x + Σ w_t·LoRA_t(x)  │  │
│                   └───────────────────────────────┘  │
│                                                       │
│  Decoder: uses encoder's routing weights              │
├───────────────────────────────────────────────────────┤
│  Post-training:                                       │
│  1. Compute spectral signatures: SVD(B·A) → (V, σ)  │
│  2. Compute GPM bases via ESA (constant threshold)   │
│  3. Save LoRA weights + signatures for next task     │
└──────────────────────────────────────────────────────┘
```

### What's Removed from GainLoRA

| Component | GainLoRA | SpecRoute |
|-----------|----------|-----------|
| `trans_input` (MLP) | Learned projection for routing | ❌ Removed — routing uses spectral fits directly |
| `prompt_key` | Learned per-task key vector | ❌ Removed — replaced by spectral signatures |
| `previous_trans_input` | Frozen snapshots for old-task routing | ❌ Removed — signatures are immutable by construction |
| `memory_replay` (KL loss) | Distillation loss on routing | ❌ Removed — no learned routing to distill |
| Increasing GPM threshold | $\epsilon_t$ grows with $t$ | Constant $\epsilon = 0.995$ (ESA) |

### What's Kept from GainLoRA/InfLoRA

- LoRA structure: separate A (frozen) and B (trained) per task per attention layer
- InfLoRA constraint: project A into null-space of old tasks' GPM bases
- GPM: collect input covariance, SVD-based subspace extraction
- Only `lora_B` is trained; `lora_A` is initialized + projected then frozen

---

## 4. Training Pipeline

### Task 1 (`--run_single True`)
1. Load pretrained model + fresh LoRA (A: kaiming init, B: zeros)
2. Train only `lora_B` (standard LoRA training — no routing needed)
3. After training: compute spectral signatures + GPM bases via ESA
4. Save: `lora_weights_A.pt`, `lora_weights_B.pt`, `spectral_signatures.pt`, GPM reg files

### Task $t$ ($t \geq 2$)
1. Load pretrained model + fresh LoRA
2. Load previous tasks' LoRA weights → `previous_lora_weights_{q,v}`
3. Load spectral signatures → `encoder.spectral_signatures`
4. Project current `lora_A` into null-space of old GPM bases (InfLoRA constraint)
5. Train `lora_B` with spectral routing:
   - Each forward pass: compute routing weights from encoder input embeddings
   - Aggregate LoRA outputs: $\text{output} = \sum_t w_t \cdot \text{LoRA}_t(x)$
6. After training: compute new spectral signatures + update GPM bases
7. Save everything for next task

---

## 5. Code-Idea Alignment

| Concept | Idea Document | Code Location | Matches? |
|---------|---------------|---------------|----------|
| C1: Spectral Signatures | SVD of $B_t A_t$, store $(V^{(r)}, \sigma^{(r)})$ | `compute_spectral_signatures()` | ✅ |
| C2: Routing (prev tasks) | Weighted Rayleigh quotient with $\sigma^2$ | `compute_spectral_routing()` prev loop | ✅ |
| C2: Routing (cur task) | Unweighted fit using A rows | `compute_spectral_routing()` cur loop | ✅ (proxy) |
| C2: Softmax routing | softmax(fits / τ), NOT OT | `torch.softmax(fit_scores / temp)` | ✅ |
| C3: ESA | Constant threshold | `threshold = self.args.threshold` | ✅ |
| InfLoRA constraint | Project A into null-space | `get_reg_matrix()` | ✅ |
| Remove trans_input | No learned routing MLP | Not in T5Stack | ✅ |
| Remove prompt_key | No learned key vectors | Not in T5Stack | ✅ |
| Remove memory_replay | No KL distillation loss | Not in trainer | ✅ |

---

## 6. Novelty Claims

1. **Spectral LoRA signatures for routing** (C1): First to use SVD properties of frozen LoRA weights as per-task identity descriptors. Unlike prompt keys, signatures are immutable and functionally grounded.

2. **Projection-based parameter-free routing** (C2): First parameter-free routing mechanism for CL-LoRA that uses weighted Rayleigh quotient to measure input-subspace alignment. Zero learned parameters, zero GPM overhead for routing.

3. **Elastic Subspace Allocation** (C3): First to identify and address the unfair capacity allocation problem in GPM-based CL. Constant threshold provides bounded, fair subspace distribution.

---

## 7. Experimental Setup

- **Model**: google/flan-t5-large (783M params)
- **Benchmark**: SuperNI, 15 tasks, 2 orderings
- **Metrics**: AP (Average Performance — avg rougeL/accuracy after all tasks, higher=better), FT (Forgetting — avg performance drop on old tasks, lower=better)
- **LoRA config**: r=4, α=32, dropout=0.0
- **Training**: lr=3e-4, constant scheduler, 100 epochs per task, BSZ=32 effective
- **Precision**: fp32 (T5 produces NaN with fp16; use gradient_checkpointing for T4 GPUs)
- **ESA threshold**: 0.995 (constant for all tasks)
- **Routing temperature**: τ=1.0

---

## 8. File Map

| File | Purpose |
|------|---------|
| `src/t5_specroute.py` | Model: T5Stack with spectral routing + T5ForConditionalGeneration |
| `src/t5_gainlora_inflora.py` | Base: LoRALayer, T5Attention, T5Block, T5PreTrainedModel (shared) |
| `src/cl_trainer_specroute.py` | Trainer: GPM, InfLoRA constraints, ESA, optimizer |
| `src/run_t5.py` | Entry point: model loading, parameter freezing, training loop |
| `src/cl_dataset.py` | Dataset: CL benchmark data loader |
| `src/cl_collator.py` | Data collator: tokenization + label masking |
| `gen_script_superni_order1_t5_specroute.sh` | Experiment script: Order 1, 15 tasks |
