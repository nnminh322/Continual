# Phân tích Khoa học: SRT + SGWI + Dual Fisher Framework
## Góc nhìn từ Co-author — Critique + Synthesis + Đề xuất

> **Status**: v3 — Full integrated framework evaluation (SGWI + Dual Fisher + SRT)
> **Source of truth**: new_gainlora/src/run_t5.py, cl_trainer_srt.py, cl_trainer_gainlora_inflora.py, t5_gainlora_inflora.py
> **Date**: April 16, 2026

---

## ⚠️ KEY CORRECTIONS FROM V2

| Claim V2 | Update V3 |
|---|---|
| "C2 không có forward transfer" | **SGWI là genuine forward transfer** — warm init từ similar past tasks |
| "Framing problem: regularization ≠ transfer" | Resolved: SGWI = transfer, Dual Fisher = protection of transferred knowledge |
| "InfLoRA + Dual Fisher double constraint" | **SGWI replaces InfLoRA** — đây là resolution đúng |
| "C2 = smarter regularization" | **C2 = SGWI (transfer) + Dual Fisher (protect)** — hai roles rõ ràng |

---

## PHẦN I: FRAMEWORK MỚI — 4 GIAI ĐOẠN

### Tổng quan kiến trúc

```
ROOT:
  get_reg_matrix() [InfLoRA: lora_A ⊥ U_prev + prompt_key GPM]
  → train() [soft MLP routing cal_attention]
  → inference [soft cal_attention weights]

C1+C2 (Proposed):
  Phase 1: SRT Profiling [μ_t, Σ_t từ frozen backbone]
  Phase 2: SGWI init [warm init lora_A,B từ SRT-weighted past tasks]
  Phase 3: Dual Fisher training [L_CE + SRT-weighted F_emb + F_grad penalty]
  Phase 4: SRT routing inference [hard one-hot Mahalanobis]
```

---

## PHẦN II: ĐÁNH GIÁ KHOA HỌC — TỪNG PHASE

### Phase 2: SGWI — SRT-Guided Warm Initialization

**Idea:**
$$\Delta W_{\text{init}} = \sum_{s < t} w_s^{\text{SRT}} \cdot (B_s A_s)$$
SVD decompose → get rank-r $(A_t^{(0)}, B_t^{(0)})$.

**✅ Đây là genuine forward transfer — Strong contribution**

SGWI giải quyết đúng vấn đề. So sánh:
- Standard LoRA init: $A_t \sim \mathcal{N}(0, \sigma^2)$, $B_t = 0$ → zero prior knowledge
- SGWI init: $\Delta W_t^{(0)} = $ interpolation of similar past tasks → task-relevant starting point

Điều này giải thích "forward transfer" đúng nghĩa: **tác vụ mới khởi đầu từ điểm đã được inform bởi các tác vụ tương đồng**, không phải từ noise.

**✅ Consistency với SRT geometry**

Cùng SRT distances $w_s^{\text{SRT}}$ được dùng cho:
1. Routing tại inference (Phase 4)
2. Warm init (Phase 2)
3. Regularization weighting (Phase 3)

→ Toàn bộ framework dùng **một thước đo nhất quán**: Mahalanobis distance in frozen backbone embedding space.

**✅ Zero-rehearsal compliance**

SGWI chỉ cần $\{B_s, A_s, \mu_s, \Sigma_s\}$ — tất cả available từ past checkpoints, không cần raw data.

---

### 🔴 CRITICAL ISSUE 1: SGWI vs InfLoRA Conflict

**Đây là vấn đề implementation quan trọng nhất.**

Hiện tại trong code, `get_reg_matrix()` làm:
```python
# InfLoRA: project lora_A AWAY from previous activation subspace
lora_A -= lora_A @ U_prev @ U_prev.T
```

Nếu SGWI set $A_t^{(0)}$ = SVD decomposition của weighted average $\sum_s w_s B_s A_s$, thì InfLoRA sẽ **immediately project nó ra khỏi subspace** → SGWI initialization bị phá.

**Resolution**: SGWI **phải thay thế InfLoRA**, không coexist.

Argument: InfLoRA được thiết kế để:
1. Prevent `lora_A` từ interfering với previous task *gradient flows* during training
2. Support routing key quality (prompt_key initialization)

Với C1+C2 framework:
- SRT routing (Phase 4) đã handle task isolation tại inference → không cần (2)
- Dual Fisher (Phase 3) handles soft constraint during training → không cần (1) cho hard init

**Proposed architecture change:**
```
ROOT get_reg_matrix():
    lora_A -= lora_A @ U_prev @ U_prev.T  ← REMOVE
    prompt_key GPM init                    ← REMOVE (not needed for SRT routing)

C1+C2 SGWI:
    ΔW_init = Σ_s w_s^SRT · (B_s A_s)    ← ADD
    A_t^(0), B_t^(0) = SVD(ΔW_init, r)   ← ADD
```

---

### 🟡 CRITICAL ISSUE 2: LoRA Asymmetry — Implementation Detail

Standard LoRA convention: $B$ initialized to $0$ at the START of training, $A$ initialized to $\mathcal{N}$.

Sau khi train task s: $B_s$ có learned weights, $A_s$ có learned weights. Tích $B_s A_s$ là full-rank approximation.

Khi SGWI compute $\Delta W_{\text{init}} = \sum_s w_s (B_s A_s)$ và SVD decompose:
$$\Delta W_{\text{init}} = U \Sigma V^T \implies A_t^{(0)} = \Sigma^{1/2} V^T, \quad B_t^{(0)} = U \Sigma^{1/2}$$

**Điều này valid về mặt toán học** nhưng có 2 implementation considerations:

1. **Numerical stability**: $\Delta W_{\text{init}}$ có thể low-rank (sum of rank-r matrices) → SVD well-conditioned
2. **Scale matching**: lora scaling factor $\alpha/r$ — cần ensure $B_t^{(0)} A_t^{(0)}$ có scale tương tự $\alpha/r$ expected range

**Recommended**: Normalize: $A_t^{(0)} = \sqrt{\alpha/r} \cdot V_{:r}^T$, $B_t^{(0)} = \sqrt{\alpha/r} \cdot U_{:r} \Sigma_{:r}$

---

### Phase 3: Dual Fisher — Đánh giá

**Framing mới từ framework:**
> "Dual Fisher protects transferred knowledge during training"

Đây là framing **đúng và defensible**. SGWI warms init; Dual Fisher prevents gradient updates từ phá hủy warm-started weights.

**"Intra-branch Forgetting" concept:**
- SGWI: $\theta_t^{(0)} \approx \theta_{s^*}^*$ (close to similar task)
- Training: gradients for task t push $\theta_t$ away from $\theta_t^{(0)}$
- Risk: nếu push quá mạnh, lose the transferred knowledge
- Dual Fisher: $\mathcal{R}(\theta_t, \theta_s^*) = (\theta_t - \theta_s^*)^T F_s (\theta_t - \theta_s^*)$ → tethers $\theta_t$ to initialization

**✅ SGWI + Dual Fisher là coherent pair:**
- SGWI: "start here (similar task weights)"
- Dual Fisher: "don't drift too far from here during training"
- SRT weights: "drift less from more similar tasks"

---

### 🟡 CRITICAL ISSUE 3: Dual Fisher + SGWI = Possible Over-Constraint

SGWI warms init $\theta_t^{(0)} \approx \theta_{s^*}^*$.
Dual Fisher penalizes $\|\theta_t - \theta_s^*\|_{F_s}^2$.

Vậy sau SGWI, $\theta_t$ đã gần $\theta_{s^*}^*$ rồi. Dual Fisher có thể:
- **Good scenario**: Soft anchor that lets task t specialize while staying near good starting point
- **Bad scenario**: If $\lambda_{\text{emb}}, \lambda_{\text{grad}}$ too large → plasticity catastrophe (new task cannot learn)

**This is expected/standard in CL literature** (EWC also has this balance). But with SGWI, the optimal $\lambda$ values may need to be smaller (since init is already close).

**Recommendation**: Run ablation $\lambda \in \{0.001, 0.01, 0.05, 0.1\}$ with SGWI init. Expected: optimal $\lambda < 0.05$ (smaller than without SGWI).

---

### 🔴 CRITICAL ISSUE 4: F_grad vs F_emb Correlation — Still Unresolved

Từ analysis trước:
- $F_{\text{grad}} \propto B^T B \otimes \Sigma_{\text{input}}$
- $F_{\text{emb}} = W_{\text{enc}}^T \Sigma_s W_{\text{enc}}$
- Cả hai depend on input covariance → likely correlated

**Tuy nhiên, context mới làm issue này ít critical hơn:**
- Nếu chúng correlated → Dual Fisher dominated by F_emb (có thể drop F_grad)
- Dual Fisher với chỉ F_emb vẫn là valid contribution: zero-rehearsal, NTK-approximated, novel
- F_grad cần backward pass để compute → có computational overhead; F_emb không cần

**Updated recommendation**: Implement F_emb only first. If F_emb alone + SGWI improves AP, add F_grad as ablation.

---

## PHẦN III: ANALYSIS TỔNG THỂ — Đây là paper tốt chưa?

### 3.1 Story Arc (C1 + C2 combined)

**Problem**: Catastrophic forgetting + poor forward transfer in continual learning with expandable LoRA.

**Root cause of forgetting**: Learned routing (cal_attention MLP) drifts as tasks accumulate → old tasks mis-routed → BWT high.

**Root cause of poor transfer**: LoRA initialized randomly; GPM's hard null-space projection prevents knowledge reuse from similar past tasks.

**C1 (SRT)**: Replaces learned routing at inference with non-parametric Mahalanobis routing → routing drift eliminated → BWT ≈ 0.

**C2 (SGWI + Dual Fisher)**: Since SRT routing is task-agnostic and perfect, we can now:
- **SGWI**: Initialize new task LoRA from SRT-weighted combination of past similar task LoRAs → genuine forward transfer
- **Dual Fisher**: Protect transferred initialization with SRT-guided Fisher penalty → prevents intra-branch forgetting during training

**Together**: Complete replacement of GPM (hard projection → soft Fisher), MLP routing (parametric → non-parametric), and random init (blind → SRT-guided). All three tied to same geometric measure (Mahalanobis on frozen backbone embeddings).

### 3.2 Scientific Strengths

| Strength | Why It Matters |
|---|---|
| **Unified geometric framework** | SRT distances used for all 3 roles: routing, warm init, regularization weighting |
| **Zero-rehearsal throughout** | Only {μ_t, Σ_t, B_t, A_t} needed — no raw data |
| **Genuine forward transfer via SGWI** | Addresses real gap in CL literature: expandable LoRA = no reuse of past adapters |
| **Intra-branch forgetting novel concept** | Not in EWC, O-EWC, or existing CL papers with expandable LoRA |
| **NTK-approximated Fisher** | Principled derivation, frozen backbone makes NTK approximation tight |
| **Hard routing at inference = BWT ≈ 0** | Quantifiable, experimentally verified (99.99% routing accuracy) |

### 3.3 Scientific Weaknesses (must address in paper)

| Weakness | Severity | Resolution |
|---|---|---|
| **SGWI + InfLoRA conflict** | 🔴 Critical | Remove InfLoRA from SRT path. Show ablation: InfLoRA hurts with SGWI. |
| **F_grad vs F_emb redundancy** | 🟡 Medium | Run E-C2-V5 correlation. If high correlation, use only F_emb. |
| **SGWI for task 0** | 🟡 Medium | Task 0 has no prior tasks → standard init. Need graceful fallback. |
| **τ temperature sensitivity** | 🟢 Low | Use median heuristic τ = median(d_SRT). Standard practice. |
| **Dual Fisher λ sensitivity with SGWI** | 🟡 Medium | Run ablation. Expected optimal λ smaller than without SGWI. |
| **SVD fusion scale** | 🟢 Low | Normalize ΔW_init before SVD. Implementation detail. |

---

## PHẦN IV: COMPARISON WITH BASELINES

### 4.1 Method Comparison Table

| Method | Routing | Init | Training Protection | Transfer |
|---|---|---|---|---|
| ROOT (GainLoRA+GPM) | Soft MLP (learned) | InfLoRA null-space | GPM gradient projection | None |
| O-EWC | Task-ID given | Random | EWC penalty | None |
| InfLoRA standalone | - | Null-space | InfLoRA | None |
| **C1 only (SRT)** | **Non-param Mahal** | InfLoRA (kept) | GPM (kept) | None |
| **C1+C2 (SRT+SGWI+DF)** | **Non-param Mahal** | **SRT-guided fusion** | **Dual Fisher** | **SGWI from similar tasks** |

### 4.2 Expected Results

| Metric | ROOT | SRT only | SRT+SGWI | SRT+SGWI+DF (full C2) |
|---|---|---|---|---|
| AP | 78.01 | 77.62 | **>78.01** (target) | **>78.01** (target) |
| BWT | Higher | 0.34 | TBD | **≤0.34** |
| FWT | Baseline | Similar | **Higher** | **Higher** |
| Conv. speed | Baseline | Similar | **Faster** (warm init) | Faster + stable |

**Key claim**: SRT+SGWI should recover and surpass ROOT's AP (78.01) because SGWI provides better initialization than InfLoRA's null-space projection. This is the pivotal experiment.

---

## PHẦN V: IMPLEMENTATION PLAN — Revised

### Step 1: Remove InfLoRA from SRT path, Add SGWI

In `cl_trainer_srt.py`, override `get_reg_matrix()`:

```python
class SRT_Trainer(GainLoRA_InfLoRA_Trainer):
    
    def get_reg_matrix(self):
        """Override: Skip InfLoRA, do SGWI instead."""
        if self.cur_task_id == 0:
            # Task 0: no prior tasks, use standard init
            super().get_reg_matrix()  # OK for task 0 (no projection happens)
            return
        
        # SGWI: SRT-guided warm initialization
        srt_weights = self._compute_srt_weights()  # {task_id: float}
        
        # Weighted sum of past LoRA products
        delta_W = {}
        for name, module in self.model.named_modules():
            if hasattr(module, 'lora_A'):
                delta_W[name] = sum(
                    w * (module.lora_B_list[s] @ module.lora_A_list[s])
                    for s, w in srt_weights.items()
                )
        
        # SVD decomposition + initialize current lora_A, lora_B
        for name, module in self.model.named_modules():
            if hasattr(module, 'lora_A') and name in delta_W:
                U, S, Vt = torch.linalg.svd(delta_W[name], full_matrices=False)
                r = module.lora_A.shape[0]  # LoRA rank
                scale = (self.args.lora_alpha / r) ** 0.5
                module.lora_A.data = scale * Vt[:r]
                module.lora_B.data = scale * (U[:, :r] * S[:r])
    
    def _compute_srt_weights(self):
        """Softmax weights based on SRT distances to current task."""
        distances = {}
        current_sig = self.srt_router.compute_signature(self.current_task_embeddings)
        for s, sig in self.srt_router.signatures.items():
            distances[s] = self.srt_router.mahalanobis(current_sig.mu, sig)
        
        # Softmax with temperature τ = median of distances
        tau = torch.tensor(list(distances.values())).median().item()
        weights = {s: math.exp(-d/tau) for s, d in distances.items()}
        Z = sum(weights.values())
        return {s: w/Z for s, w in weights.items()}
```

### Step 2: Add Dual Fisher to training loss

```python
def compute_loss(self, model, inputs, return_outputs=False):
    loss = super().compute_loss(model, inputs, return_outputs)
    
    if self.cur_task_id > 0 and (self.lambda_emb > 0 or self.lambda_grad > 0):
        loss = loss + self._dual_fisher_reg()
    
    return loss

def _dual_fisher_reg(self):
    srt_weights = self._compute_srt_weights()
    total = 0.0
    
    for s, w_s in srt_weights.items():
        sig = self.srt_router.signatures[s]
        
        # F_emb: NTK approximation via SRT covariance
        F_emb_trace = torch.trace(
            self.W_enc.T @ torch.from_numpy(sig.Sigma).float() @ self.W_enc
        )
        
        for name, param in self.model.named_parameters():
            if 'lora_' not in name or name not in self.theta_stars[s]:
                continue
            delta = param - self.theta_stars[s][name].to(param.device)
            
            if self.lambda_emb > 0:
                total += w_s * self.lambda_emb * F_emb_trace * (delta**2).sum()
            
            if self.lambda_grad > 0 and name in self.fisher_grad[s]:
                F_g = self.fisher_grad[s][name].to(param.device)
                total += w_s * self.lambda_grad * (F_g * delta**2).sum()
    
    return total
```

### Step 3: Experiments (in order)

**E1 (Baseline verification)**:
```
ROOT: AP=78.01, BWT=? → replicate
SRT_only: AP=77.62, BWT=0.34 → replicate
```

**E2 (SGWI ablation — critical)**:
```
SRT + SGWI (no Dual Fisher): AP=?, BWT=?
Expected: AP > 77.62 (SGWI improves convergence)
Goal: AP > 78.01 (recover ROOT's advantage)
```

**E3 (Dual Fisher ablation)**:
```
SRT + SGWI + F_emb: λ_emb ∈ {0.001, 0.005, 0.01, 0.05}
SRT + SGWI + F_grad: λ_grad ∈ {0.001, 0.005, 0.01, 0.05}
SRT + SGWI + F_emb + F_grad: best λ_emb + best λ_grad
```

**E4 (E-C2-V5 correlation)**:
```
For each task: cosine_similarity(top-20 eigvecs of F_grad, top-20 eigvecs of F_emb)
Decision: if avg > 0.7 → use only F_emb
```

---

## PHẦN VI: HONEST VERDICT

### Mức độ contribution

**C1 (SRT)**: Strong. Non-parametric routing replacing learned MLP is a principled contribution with theoretical grounding (Mahalanobis = optimal Bayes classifier under Gaussian assumption). BWT ≈ 0 is strong empirical result.

**C2 (SGWI + Dual Fisher)**:
- **SGWI**: Novel, addresses genuine gap (expandable LoRA = no cross-task reuse). Zero-rehearsal. Consistent geometry with SRT. **This is the strongest part of C2.**
- **Dual Fisher**: "Intra-branch forgetting" framing is novel for expandable LoRA setting. F_emb from NTK + SRT signature is principled. SRT-weighted regularization is novel. **Moderate contribution if F_emb complementary to F_grad.**

**Combined**: If E2 shows SGWI recovers AP > 78.01 → **both AP and BWT improve over ROOT** → strong paper. This is the key experiment.

### What the reviewers will ask

1. **"SGWI replaces InfLoRA — does this hurt interference prevention?"**  
   Answer: SRT inference routing (hard one-hot) already ensures task isolation at inference. InfLoRA was needed for GPM routing quality — SRT removes that dependency.

2. **"F_emb vs F_grad — why both? Show they're complementary."**  
   Answer: E-C2-V5 experiment. If correlated, drop F_grad (F_emb alone is novel + zero-rehearsal).

3. **"SGWI with many tasks — does the warm init stay meaningful?"**  
   Answer: SRT routing accuracy increases with tasks (more signatures → better discrimination). SGWI similarly improves (more diverse past tasks → better weighted average).

4. **"What if SRT mis-routes at task 0 (no prior signatures for SGWI)?"**  
   Answer: Task 0 uses standard init (no SGWI). SGWI activates from task 1+.

### Bottom line

**The integrated SGWI + Dual Fisher + SRT framework is significantly stronger than pure Dual Fisher alone.** SGWI resolves the "forward transfer" question properly. The framework has unified geometric coherence (one SRT distance metric for all 3 roles). The key risk is SGWI conflicting with InfLoRA — must remove InfLoRA from SRT path. The key experiment is E2: does SGWI recover AP > 78.01?

---

## APPENDIX: Key Questions Still Open

| Question | When to Resolve | How |
|---|---|---|
| SGWI + InfLoRA conflict → remove InfLoRA? | Before implementation | Design decision: YES, remove for SRT path |
| Does SGWI recover AP > 78.01? | After E2 | Run experiment |
| F_grad vs F_emb correlation | Before full Dual Fisher impl | E-C2-V5 |
| Optimal λ with SGWI (smaller than without?) | After E3 | Ablation sweep |
| Task 0 fallback for SGWI | Implementation | Standard init, graceful |
| τ sensitivity for SRT weighting | After E3 | Median heuristic first |
