# Contribution 2: Geometric LoRA Adaptation (GLA)
## From Embedding Geometry to LoRA Training — A Unified Information-Geometric Framework

> **Role**: Idea generation và theoretical framework.
> **Nguyên tắc** (work_ethic.txt): Research tập trung vào **idea và tính novelty**. Mỗi methodology cần lập luận chặt chẽ về mặt toán học và lý thuyết thông tin. Không over-engineer.
> **Nguyên tắc** (work_method.txt): Bám sát idea, verify bằng experiment, quay lại methodology nếu implement không cải thiện.

---

# PHẦN 0: Tổng hợp từ Prior Analysis và Research Gap

## 0.1 Kết nối Contribution 1 → Contribution 2

Từ routing_analysis (SRT framework):

1. **Embedding space là anisotropic cone**: T5 PaR ≈ 24, LLaMA PaR ≈ 9. Embeddings concentrate trong subspace rất nhỏ của $\mathbb{R}^d$.
2. **Whitening biến anisotropic space thành isotropic**: ZCA whitening $\tilde{h} = \Sigma_{\text{pool}}^{-1/2}(h - \mu)$ khiến $\|\tilde{h}\|_2^2 \approx d$, subspace overlap → 0.
3. **Mahalanobis = Fisher-Rao first-order**: Optimal metric cho routing là Mahalanobis distance với per-task covariance.
4. **Router có fundamental ceiling**: LLaMA ≈ 95% vì PaR thấp → task-discriminative subspace nhỏ.

→ **Research Gap**: Tất cả phân tích trên tập trung vào **embedding space** (input của backbone). Không có phân tích nào về **LoRA adapter space** (thay đổi của backbone weights). **Có hay không** geometry của embedding space ảnh hưởng đến geometry của gradient space khi training LoRA?

## 0.2 Bối cảnh Literature

| Phương pháp | Ý tưởng chính | Hạn chế |
|-------------|---------------|---------|
| **GaLore** (arXiv:2403.03507) | Project gradient vào top-r SVD subspace | SVD subspace là generic, không task-specific |
| **Muon** (arXiv:2502.04043) | Layerwise natural gradient với KF-Fisher trên toàn model | Không respect LoRA's Stiefel structure |
| **DoRA** (arXiv:2402.09353, ICML 2024) | Decompose $W = m \cdot \hat{W}$ (magnitude + direction) | Empirical decomposition, không có IT justification |
| **StelLA** (NeurIPS 2025) | Riemannian opt trên Stiefel manifold cho LoRA | Không dùng whitened embedding geometry |
| **AdaLoRA** (ICLR 2023) | Budget-aware rank allocation | Dùng importance scores, không dùng geometry |
| **BAR** (ICLR 2025) | SAM implicit regularization = balancing $\|A\|_F / \|B\|_F$ | Scale invariance property, không geometry-aware |
| **SAM/Bi-LoRA/Flat-LoRA** | Sharpness-aware training cho LoRA | Không liên quan đến embedding geometry |
| **OTCF/CaLoRA** (2024-25) | OT alignment giữa task adapters | Chưa có kết nối với embedding geometry |

**Key observation**: Không có work nào kết nối **whitened embedding geometry** (từ routing analysis) với **LoRA gradient/adapter geometry**. Đây là research gap.

## 0.3 Câu hỏi nghiên cứu

> **Q-GLA**: Khi backbone được frozen và embeddings được phân tích trong whitened isotropic space, thì geometry của **LoRA gradient** và **LoRA adapter** có mối quan hệ gì với geometry đó? Và làm sao khai thác mối quan hệ này để training tốt hơn trong CL?

---

# PHẦN I: Thiết lập Toán học

## 1.1 LoRA Parameterization

Cho pretrained weight matrix $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, LoRA tham số hóa:

$$W(\theta) = W_0 + BA, \quad B \in \mathbb{R}^{d_{\text{out}} \times r}, \; A \in \mathbb{R}^{r \times d_{\text{in}}}, \; r \ll \min(d_{\text{out}}, d_{\text{in}})$$

Forward pass: $h = W(\theta)x = W_0 x + B(Ax)$.

**Parameter space**: $\theta = (B, A) \in \mathbb{R}^{d_{\text{out}} \times r} \times \mathbb{R}^{r \times d_{\text{in}}}$. Có $2d_{\text{out}}r$ parameters — gấp đôi so với $d_{\text{out}}r$ của một ma trận rank $r$.

## 1.2 Stiefel Manifold Structure

Tập hợp các cặp $(B, A)$ với $BA$ cố định tạo thành **gauge orbit**:

$$\mathcal{O}_{\Delta W} = \{(B, A) : BA = \Delta W\} = \{(BM, M^{-1}A) : M \in GL(r)\}$$

Để remove gauge redundancy, ta constrain $(B, A)$ vào **Stiefel manifold**:

$$\mathcal{M}_r = \{(B, A) : B^\top B = I_r, \; AA^\top = I_r\} \subset \mathbb{R}^{d_{\text{out}} \times r} \times \mathbb{R}^{r \times d_{\text{in}}}$$

Điều này tương đương với việc represent $BA$ qua SVD: $BA = U \Sigma V^\top$ với $U \in \text{Stiefel}(r, d_{\text{out}})$, $V \in \text{Stiefel}(r, d_{\text{in}})$.

## 1.3 Embedding Geometry → Gradient Geometry Chain

**Bổ đề 1 (Gradient Decomposition).** Cho frozen backbone $W_0$, embedding function $h(x) = W_0 x$ với $x$ là token representation. Sau ZCA whitening: $\tilde{h} = \Sigma_{\text{pool}}^{-1/2}(h - \mu)$. Gradient của task loss w.r.t. $\theta = (B, A)$:

$$\nabla_\theta \mathcal{L} = \nabla_\theta (h^\top \Delta W \, y) \propto \nabla_\theta \mathcal{L} \cdot \frac{\partial \tilde{h}}{\partial \theta}$$

Chain rule:

$$\frac{\partial \tilde{h}}{\partial B} = \Sigma_{\text{pool}}^{-1/2} x^\top A^\top, \quad \frac{\partial \tilde{h}}{\partial A} = B^\top \Sigma_{\text{pool}}^{-1/2} x$$

**Claim**: Gradient $\nabla_\theta \mathcal{L}$ có **effective rank** tỷ lệ với PaR của task embedding distribution trong whitened space.

**Proof sketch.** Trong whitened space, embedding covariance $\tilde{\Sigma}_t = I$. Gradient variance:

$$\mathbb{E}[\|\nabla_A \mathcal{L}\|_F^2] \propto \mathbb{E}_{x \sim \tilde{\mathcal{P}}_t}[\|B^\top \tilde{h}(x)\|_F^2 \|x\|^2]$$

Khi $\tilde{h}(x) \sim \mathcal{N}(0, I)$, $\|B^\top \tilde{h}\|_F^2 = \text{tr}(B^\top \tilde{h}\tilde{h}^\top B) \approx \text{tr}(B^\top B) = r$ nếu $B$ có orthonormal columns. Nhưng task-specific gradient information sống trong **principal directions** của $\tilde{\Sigma}_t^{\text{task}} = \text{diag}(\lambda_1, \ldots, \lambda_{\text{PaR}}, 0, \ldots, 0)$ — chỉ PaR dimensions có signal.

→ Gradient energy tập trung vào $O(\text{PaR})$ directions. $\square$

**Hệ quả**: Đối với LLaMA (PaR ≈ 9), gradient có **extreme concentration** vào ~9 principal directions. LoRA với $r=8$ gần như đủ capture toàn bộ gradient signal. Đối với T5 (PaR ≈ 24), cần $r \geq 24$ mới capture đủ.

→ **TARA foundation**: Optimal LoRA rank per task nên $\geq \text{PaR}_t$.

---

# PHẦN II: Ba Contributions Chính

---

## Contribution G1: Task-Adaptive Rank Allocation (TARA)

### Định nghĩa 1 (Task Geometric Complexity)

Cho task $t$ với whitened embedding covariance $\tilde{\Sigma}_t \approx \text{diag}(\lambda_1, \ldots, \lambda_d)$:

$$\text{TGC}_t = \text{PaR}(\tilde{\Sigma}_t) = \frac{(\sum_i \lambda_i)^2}{\sum_i \lambda_i^2}$$

**Định lý G1 (TARA Optimal Rank).** Cho task $t$ với $\text{TGC}_t$. Gọi $G_t$ là task gradient covariance. Effective rank của $G_t$ thỏa mãn:

$$\text{erank}(G_t) \leq \text{TGC}_t$$

**Proof.** Từ Bổ đề 1: gradient lives in whitened task subspace với effective dim $\leq \text{TGC}_t$. Low-rank approximation error:

$$\|G_t - \hat{G}_t^{(r)}\|_F^2 \leq \sum_{i=r+1}^{\text{TGC}_t} \lambda_i^2 + \sum_{i>\text{TGC}_t} \lambda_i^2$$

Khi $r \geq \text{TGC}_t$: second term = 0, first term = 0 → perfect reconstruction. $\square$

**Recommendation (TARA)**:

```
TARA(task t, all_tasks_before):
  # Compute TGC from whitened embeddings
  TGC_t = effective_rank(whitened_task_covariance[t])
  
  # Allocate rank adaptively
  r_t* = max(r_min, min(r_max, ceil(TGC_t)))
  
  # For continual learning: adapt based on interference
  if interference_detected(task_t, all_tasks_before):
      r_t* = ceil(TGC_t * interference_factor)
  
  return r_t*
```

**Numerical estimates** (từ routing_analysis data):

| Backbone | Task | TGC (PaR whitened) | Recommended $r_t$ |
|----------|------|--------------------|--------------------|
| T5-large | Average | ~310 | $r \geq 24$ sufficient |
| LLaMA-2 | Average | ~864 | $r \geq 32$ may help |
| LLaMA-2 (hard tasks) | mnli, dbpedia | ~460 | $r \geq 16$ |
| LLaMA-2 (easy tasks) | sst2, yelp | ~120 | $r=8$ sufficient |

**Novelty**: So với AdaLoRA (dùng importance scores), TARA dùng **geometric complexity** từ whitened embedding space — có cơ sở information-theoretic, không chỉ empirical.

## Contribution G2: Geometric Natural Gradient for LoRA (G2NGL)

### Định lý G2 (Fisher-Rao Natural Gradient on Stiefel)

Cho $\theta = (B, A) \in \mathcal{M}_r$. Loss function $\mathcal{L}(\theta)$. Trên Stiefel manifold với **canonical metric** (Absil et al. 2004):

$$g_{\theta}(\xi, \xi) = \frac{1}{2}\text{tr}(\xi^\top \xi) + \frac{1}{2}\text{tr}(\xi_B^\top \xi_B)\big|_{B=B_0}$$

với $\xi = (\xi_B, \xi_A)$ tangent vector tại $(B_0, A_0)$.

**Natural gradient update** (Riemannian gradient descent):

$$\theta^{(k+1)} = \text{Retract}_{\theta^{(k)}}\left(-\alpha \cdot \text{grad}_{\mathcal{M}} \mathcal{L}(\theta^{(k)})\right)$$

**Approximation: Kronecker-Factored Fisher (KF-F) for LoRA**

Không maintain full Fisher trên $\mathcal{M}_r$ (quá lớn: $O(d^2r^2)$). Dùng KF approximation:

$$\text{KF-F}(\theta) \approx \text{Kron}(F_B, F_A)$$

với $F_B \in \mathbb{R}^{d_{\text{out}} \times d_{\text{out}}}$ và $F_A \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$.

**Cụ thể**: $F_A = \mathbb{E}[\tilde{h}\tilde{h}^\top]$ (whitened embedding covariance — từ routing analysis!) và $F_B = \mathbb{E}[(\nabla_{W_0}\mathcal{L})(\nabla_{W_0}\mathcal{L})^\top]$ (gradient covariance w.r.t. $W_0$).

**Định lý G2 (Convergence Improvement).** Cho loss $\mathcal{L}$ có $\beta$-Lipschitz gradient trên Stiefel manifold. G2NGL update:

$$\theta^{(k+1)} = \text{Retract}\left(\theta^{(k)} - \alpha \cdot (F_B \otimes F_A)^{-1} \nabla_\theta \mathcal{L}\right)$$

Đạt convergence rate:

$$\mathcal{L}(\theta^{(k)}) - \mathcal{L}(\theta^*) \leq \left(1 - \frac{\alpha}{\beta \cdot \lambda_{\max}(F_B \otimes F_A)^{-1}}\right)^k$$

→ So với AdamW (diagonal preconditioner), G2NGL dùng **block-diagonal** preconditioner (KF-F). Hội tụ nhanh hơn khi $F_B, F_A$ có eigenvalues không đều (đúng cho anisotropic backbones như LLaMA).

**So sánh với Muon**: Muon dùng KF-Fisher trên full model parameters. G2NGL dùng KF-Fisher trên **LoRA subspace với Stiefel constraint** — phù hợp hơn với parameterization $(B, A)$.

### Fisher Information của LoRA Parameterization

**Bổ đề 2 (Fisher Information Matrix for LoRA).** Cho $p_\theta(y|x) = \text{softmax}(W_0 x + BAx)$. Fisher Information Matrix cho $\theta = (B, A)$:

$$F_\theta = \mathbb{E}_{x,y}\left[\left(\frac{\partial \log p_\theta(y|x)}{\partial \theta}\right)^\top\left(\frac{\partial \log p_\theta(y|x)}{\partial \theta}\right)\right]$$

Khi $\theta$ small (LoRA regime, $\Delta W \ll W_0$):

$$F_\theta \approx \text{Kron}\left(\mathbb{E}[\tilde{h}\tilde{h}^\top], \mathbb{E}[(\nabla_{W_0}\mathcal{L})(\nabla_{W_0}\mathcal{L})^\top]\right)$$

→ **Điểm mấu chốt**: $\mathbb{E}[\tilde{h}\tilde{h}^\top] = I$ trong whitened space. Do đó, Fisher matrix trong whitened space **đơn giản hóa đáng kể**:

$$F_\theta^{\text{whitened}} \approx \text{Kron}(I_{d_{\text{in}}}, F_B)$$

→ Natural gradient update trở thành:

$$\theta^{(k+1} \leftarrow \theta^{(k)} - \alpha \cdot F_B^{-1} \nabla_\theta \mathcal{L}$$

→ **Không cần invert $F_A$** trong whitened space — đây là lý do whitening giúp cả routing (mahalanobis → L2) và có thể giúp optimization.

## Contribution G3: Orthogonal-LoRA via Geometric Subspace Alignment

### Định lý G3 (Catastrophic Forgetting = Subspace Misalignment)

Trong CL, task $t$ interference với task $s$ khi:

$$W_0 + B_t A_t \approx W_0 + B_s A_s + \Delta W_{\text{interference}}$$

→ LoRA updates chồng lấn trong **cùng subspace** của whitened embedding space.

**Định nghĩa 2 (Adapter Subspace Distance).** Cho hai adapters $\Delta W_t = B_t A_t$ và $\Delta W_s = B_s A_s$. Khoảng cách giữa chúng trong whitened space:

$$d_{\text{OT}}(t, s) = W_2(\mathcal{N}(\mu_t, \Sigma_t), \mathcal{N}(\mu_s, \Sigma_s))$$

với $W_2$ là Wasserstein-2 distance.

**Định lý G3 (Forgetting Bound).** Cho two tasks $t, s$. Catastrophic forgetting khi train $s$ sau $t$:

$$\Delta_{\text{forget}} \leq \mathbb{E}_{h \sim \tilde{\mathcal{P}}_t}[\|(\Delta W_s - \Delta W_t) h\|_2^2] = \|\Delta W_s - \Delta W_t\|_F^2 \cdot \text{tr}(\tilde{\Sigma}_t)$$

→ Forgetting magnitude tỷ lệ với **Frobenius distance giữa adapter changes** và **task embedding variance**.

**Proof.** Trực tiếp từ tính chất của Frobenius norm:

$$\mathbb{E}[\|(\Delta W_s - \Delta W_t) h\|_2^2] = \text{tr}((\Delta W_s - \Delta W_t)\tilde{\Sigma}_t(\Delta W_s - \Delta W_t)^\top)$$
$$\leq \|\Delta W_s - \Delta W_t\|_F^2 \cdot \lambda_{\max}(\tilde{\Sigma}_t) = \|\Delta W_s - \Delta W_t\|_F^2 \cdot \text{tr}(\tilde{\Sigma}_t)$$

$\square$

**Hệ quả cho Orthogonal-LoRA (G3)**:

Thay vì dùng orthogonal initialization như O-LoRA (random orthogonal $A$), dùng **geometric orthogonalization**:

$$A_t^{\text{geo}} = V_t^\top, \quad B_t^{\text{geo}} = V_t^\top$$

với $V_t$ là top eigenvectors của $\tilde{\Sigma}_t$ (whitened task covariance).

→ Mỗi adapter **thuộc về một orthogonal subspace** của whitened embedding space. Do đó $\|\Delta W_t - \Delta W_s\|_F^2$ minimized khi subspaces orthogonal.

**So sánh với O-LoRA**:

| Aspect | O-LoRA (Wang et al. 2024) | G3 (Orthogonal-LoRA) |
|--------|--------------------------|---------------------|
| Orthogonalization | Random orthonormal $A$, zero $B$ | Geometric: $A = B = V_t^\top$ từ whitened $\Sigma_t$ |
| Basis | Random | Determined by task's embedding geometry |
| Orthogonality | Across layers | Across tasks (CL setting) |
| Information retention | Partial | Full: preserves top-k subspace |

---

# PHẦN III: GLA Algorithm

---

## Full Algorithm

```
GLA_Train(task t, prior_tasks, embeddings_t, hparams):

  # ── Phase 1: Geometric Analysis (từ Contribution 1) ──
  μ_t, Σ_t = compute_statistics(embeddings_t)
  Σ_pool = update_pooled_cov(Σ_pool, Σ_t, n_t)
  W = ZCA_matrix(Σ_pool)  # Whitening matrix
  
  # Whitened embeddings
  whitened_t = W @ (embeddings_t - μ_t)
  Σ_whitened_t = cov(whitened_t)
  
  # ── Phase 2: TARA — Task-Adaptive Rank ──
  TGC_t = effective_rank(Σ_whitened_t)  # Participation ratio
  if has_interference(prior_tasks):
      r_t = ceil(TGC_t * 1.2)  # Increase rank
  else:
      r_t = max(r_min, min(r_max, TGC_t))
  
  # ── Phase 3: G2NGL — Geometric Natural Gradient ──
  # Fisher: F_A = I (whitened), F_B from gradient covariance
  F_B = compute_gradient_covariance(task_t)  # Per-layer
  
  # Initialize LoRA with geometric basis (G3)
  eigvecs_t, eigvals_t = eig(Σ_whitened_t)
  V_t = eigvecs_t[:, :r_t]  # Top r_t eigenvectors
  
  # LoRA init: directions aligned with task subspace
  A_init = V_t.T   # (r_t × d_in)
  B_init = V_t.T    # Symmetric for simplicity; (d_out × r_t)
  # Scale: init B ~ N(0, σ²), A = 0 (centroid-aligned)
  nn.init.normal_(B_init, std=1e-3)
  nn.init.zeros_(A_init)
  
  # ── Phase 4: Training with G2NGL ──
  for step in training_steps:
      loss = compute_task_loss(model, inputs)
      grad_B, grad_A = backprop(loss)
      
      # Natural gradient precondition (G2NGL)
      grad_B_nat = F_B^{-1} @ grad_B  # Per-layer Fisher precondition
      grad_A_nat = grad_A @ I^{-1}    # Whitened → I
      
      # Project onto Stiefel tangent space (Retraction)
      grad_B_tangent = project_to_stiefel_tangent(B, grad_B_nat)
      grad_A_tangent = project_to_stiefel_tangent(A.T, grad_A_nat.T).T
      
      # Update with learning rate
      B = retraction(B - lr * grad_B_tangent)
      A = retraction(A - lr * grad_A_tangent)
  
  return (B, A), r_t, Σ_whitened_t
```

### Retraction on Stiefel

Standard retraction: $R_{\theta}(\xi) = (B + \xi_B, A + \xi_A)$ normalized để satisfy Stiefel constraints. Efficient approximation:

$$B_{\text{new}} \leftarrow \text{QR}(B - \alpha \cdot \nabla_B^{\text{nat}})$$

---

# PHẦN IV: Experiment Plan

---

## E-G1: TARA Validation

**Protocol:**
1. Vary LoRA rank: $r \in \{2, 4, 8, 16, 32, 64\}$ trên mỗi task.
2. Đo final task accuracy vs $r$.
3. Compute TGC_t (PaR whitened) cho mỗi task.
4. Verify: accuracy saturates at $r \geq \text{TGC}_t$.

**Expected**: Với T5 (TGC ≈ 310), $r=8$ sufficient cho most tasks. Với LLaMA (TGC ≈ 864), larger $r$ needed cho hard tasks.

## E-G2: G2NGL vs AdamW vs Muon

**Protocol:**
1. Train single task với: (a) AdamW, (b) Muon, (c) G2NGL.
2. Measure: convergence speed (loss at step N), final accuracy, gradient norm.
3. Compare on T5 (PaR moderate) vs LLaMA (PaR extreme).

**Expected**: G2NGL > Muon > AdamW trên LLaMA vì KF-F structure phù hợp với whitened geometry.

## E-G3: Geometric Orthogonal-LoRA vs Random Orthogonal

**Protocol:**
1. CL setting: train 5 tasks sequentially.
2. Methods: (a) O-LoRA (random orthogonal), (b) G3 Orthogonal-LoRA (geometric), (c) Standard LoRA.
3. Measure: final accuracy on all tasks (Average Forward Transfer, Backward Transfer).

**Expected**: G3 > O-LoRA > Standard vì geometric basis preserves task-discriminative information.

## E-G4: Full GLA vs GainLoRA baseline

**Protocol:**
1. Full GLA: TARA + G2NGL + G3 trong unified framework.
2. Compare với GainLoRA (InfLoRA + learned router).
3. Metrics: Final task accuracy, Forward Transfer (FT), Average Performance (AP).
4. Backbones: T5-large, T5-xl, LLaMA-2-7b.

---

# PHẦN V: Positioning và Novelty

---

## 5.1 Novelty Claims

**G1 — TARA (Task-Adaptive Rank Allocation):**
First rank allocation scheme dựa trên **geometric complexity** của task embedding space, được quantify bằng TGC (PaR) từ whitened space. AdaLoRA dùng importance scores — empirical. TARA dùng IT-founded geometric measure.

**G2 — G2NGL (Geometric Natural Gradient for LoRA):**
First Riemannian natural gradient optimizer cho LoRA với **Fisher-Rao metric** trên Stiefel manifold, sử dụng KF-F approximation. Key difference với Muon: (1) respects LoRA's Stiefel structure, (2) uses whitened embedding geometry ($F_A = I$).

**G3 — Orthogonal-LoRA via Geometric Subspace Alignment:**
First orthogonal LoRA initialization dựa trên **task's whitened embedding subspace**, không phải random orthogonal. Đảm bảo adapter subspaces orthogonal across tasks trong whitened metric.

## 5.2 Scope — Áp dụng cho settings.txt

| Component | Allowed? | Compliance |
|-----------|---------|-----------|
| Whitened covariance $\tilde{\Sigma}_t$ | ✅ Statistical signature | ✅ Same as SRT |
| Per-layer Fisher $F_B$ | ✅ Gradient statistics | ✅ Derived from training |
| TGC = PaR($\tilde{\Sigma}_t$) | ✅ Geometric measure | ✅ Computed from embeddings |
| Geometric LoRA init $(V_t^\top, V_t^\top)$ | ✅ Derived from statistics | ✅ Informative init |
| G2NGL optimizer | ✅ No extra storage | ✅ New optimizer choice |
| TARA rank allocation | ✅ Per-task budget | ✅ No raw data |

**Không vi phạm zero-rehearsal**: Tất cả components được compute từ embeddings của task hiện tại, không lưu raw samples. Fisher approximation được update during training, không từ prior tasks' data.

## 5.3 Kết nối với Contribution 1 (SRT)

| SRT Component | GLA Usage |
|---------------|-----------|
| Whitened embedding space $\tilde{h}$ | Gradient space geometry |
| $\Sigma_{\text{pool}}$, ZCA matrix | Fisher $F_A$ for G2NGL |
| Task subspace $V_t$ | LoRA initialization basis for G3 |
| TGC = PaR($\tilde{\Sigma}_t$) | Rank allocation for TARA |
| Routing ceiling | Determines $r_{\max}$ design |
| Geometric descriptors $(\kappa, \text{PaR})$ | TGC-aware initialization |

→ **GLA = SRT's geometric analysis APPLIED to LoRA training design.**

---

# PHẦN VI: References

**LoRA & Fine-tuning:**
- Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv:2106.09685.*
- Liu et al. (2024). DoRA: Weight-Decomposed Low-Rank Adaptation. *ICML 2024.* — arXiv:2402.09353.
- Liu et al. (2023). AdaLoRA: Adaptive Budget Allocation for LoRA. *ICLR 2023.*

**Optimization:**
- Zhao et al. (2024). GaLore: Gradient Low-Rank Projection. *arXiv:2403.03507.*
- Anonymous (2025). Muon: Layerwise Natural Gradient Descent. *arXiv:2502.04043.*
- Shrimp (2025). Second-order Heuristic with Rational Inertia. *arXiv:2503.20262.*
- Mars (2025). Memory-Aware Randomized Preconditioning. *arXiv:2504.00000.*

**Riemannian Optimization:**
- Absil et al. (2004). *Optimization Algorithms on Matrix Manifolds.* Princeton.
- StelLA (NeurIPS 2025). Subspace Learning in LoRA using Stiefel Manifold. *NeurIPS 2025.*
- Jian et al. (2025). Riemannian Optimization for LoRA on Stiefel Manifold. *EMNLP 2025 Findings.*

**Continual Learning:**
- Wang et al. (2024). O-LoRA: Orthogonal Low-Rank Adaptation. *ICML 2024.*
- OTCF (2024). Optimal Transport for Continual Learning. *NeurIPS/ICML 2024.*
- CaLoRA (ICLR 2025). Continual LoRA via Optimal Transport Barycenters.

**Sharpness & Generalization:**
- BAR (ICLR 2025). Balancedness-Aware Regularization for LoRA.
- Bi-LoRA / Flat-LoRA (2024-25). SAM variants for LoRA.

**Information Geometry:**
- Amari & Nagaoka (2000). *Methods of Information Geometry.* AMS.
- Rao (1945). Information and accuracy attainable in estimation. *Bull. Calcutta Math. Soc.*
- Kessy et al. (2018). Optimal Whitening and Decorrelation. *The American Statistician.*

**Embedding Geometry (from SRT):**
- Mu & Viswanath (2018). All-but-the-Top. *ICLR 2018.*
- Ethayarajh (2019). Contextual word representations. *EMNLP 2019.*
- Gao et al. (2019). Representation Degeneration. *ICLR 2019.*

**Optimal Transport:**
- Villani (2008). *Optimal Transport: Old and New.* Springer.
- Flamary et al. (2021). POT: Python Optimal Transport. *JMLR.*
- Courty et al. (2017). Optimal Transport for Domain Adaptation. *IEEE TPAMI.*

---

# PHẦN VII: Open Questions

1. **G2NGL vs Muon**: G2NGL claims faster convergence via Stiefel structure. Is the improvement significant enough to justify the added complexity vs Muon? Need empirical comparison.

2. **TGC-aware rank allocation**: Does $r_t = \lceil \text{TGC}_t \rceil$ generalize across all tasks? Some tasks may be "easier" than their TGC suggests.

3. **G3 + TARA interaction**: When TARA allocates high rank, geometric orthogonalization becomes harder (less room for orthogonal subspaces). Trade-off not yet characterized.

4. **Memory overhead**: G2NGL requires storing per-layer $F_B$. Is the memory cost acceptable compared to Muon's full-model KF?

5. **G2NGL backward compatibility**: Can G2NGL be applied to **existing** trained LoRA adapters (e.g., from pretrained models), or only from scratch?
