# Contribution 2: Geometry-Aware Low-Rank Adaptation for Continual Learning

## Riemannian Optimization on the Grassmannian with Natural Gradient Preconditioning

> **Scope**: Cải thiện chất lượng training **một task duy nhất** trong CL-LoRA.
> **Không xét**: routing (đã giải quyết ở Contribution 1 — SRT).
> **Nguyên tắc**: Theory-first, mathematical rigor, information-theoretic backing, không over-engineer.

---

# PHẦN 0: Phân tích Hiện trạng — Vấn đề của LoRA Training Hiện tại

---

## 0.1 Current Implementation (GainLoRA + InfLoRA)

Từ source code (`llama_gainlora_inflora.py`, `cl_trainer_gainlora_inflora.py`):

| Component | Implementation | Vấn đề |
|-----------|---------------|--------|
| **A init** | `kaiming_uniform_(A, a=√5)` | Random — không biết task cần learn gì |
| **B init** | `zeros_(B)` | ΔW=BA=0 → gradient ∂L/∂A = B^T(∂L/∂h)x = 0 ở step 1! |
| **CL constraint** | Hard projection: `A = A - A·(UU^T)` | Phá hủy thông tin, không phân biệt quan trọng/không quan trọng |
| **Optimizer** | AdamW, cùng LR cho A và B | Bỏ qua cấu trúc manifold, gauge redundancy |
| **Loss** | CE only | Không guide subspace efficiency |
| **lr schedule** | constant, no warmup | Không adaptive |
| **Rank** | r=4, α=32, scaling=8 | Fixed, không geometry-aware |
| **Balance** | Không có | Gauge orbits gây flat valleys, waste capacity |
| **Rank allocation** | Same r for all tasks | Bỏ qua task-specific geometric complexity |

## 0.2 Bảy vấn đề cốt lõi

**P1 — Initialization mù (Blind Initialization).**
Kaiming init cho A → A spans random r-dim subspace trong $\mathbb{R}^{d_{in}}$. Xác suất subspace này khớp với task-informative directions = $\binom{d_{in}}{r}^{-1} \approx 0$. Gradient descent phải **tìm** subspace đúng trước khi **tối ưu** trên nó — lãng phí capacity.

**P2 — B=0 tạo asymmetry pathological.**
Ở step 0: $\frac{\partial L}{\partial A} = B^\top \frac{\partial L}{\partial h} x^\top = 0$ vì $B=0$. B nhận gradient trước, A chỉ bắt đầu learn sau khi B ≠ 0. Hệ quả: B "chọn" output direction trước, A phải adapt theo — **thứ tự ngược** (lẽ ra input subspace nên quyết định trước).

**P3 — Hard orthogonality là lossy.**
Projection $A_{\text{new}} = A - A \cdot P_{\text{prev}}$:
- Giả sử task hiện tại cần một thành phần nhỏ trong subspace cũ (kiến thức chung). Hard projection loại bỏ hoàn toàn — buộc task phải học lại.
- Không phân biệt: direction cũ quan trọng (cần bảo vệ mạnh) vs ít quan trọng (có thể share).
- Là **discontinuous** operator — vi phạm smoothness cần cho convergence guarantees.

**P4 — AdamW bỏ qua manifold structure.**
LoRA parameters $(B, A)$ có gauge symmetry: $(BM, M^{-1}A)$ cho cùng ΔW. AdamW tối ưu trên Euclidean product space $\mathbb{R}^{m \times r} \times \mathbb{R}^{r \times n}$ — nó "thấy" $r^2$ chiều thừa và có thể di chuyển dọc gauge orbits mà không thay đổi output. Lãng phí optimization budget.

**P5 — Loss function thiếu geometric guidance.**
CE loss chỉ đo prediction quality, không guide:
- Subspace utilization efficiency (có dùng hết r dimensions?)
- Balance giữa A và B (gauge fixing)
- Overlap với previous tasks (CL compatibility)

**P6 — Không kết nối với embedding geometry.**
Contribution 1 cho thấy: activations rất anisotropic (κ ≈ 132–439, PaR ≈ 9–27). Gradients cho LoRA kế thừa anisotropy này → standard optimization inefficient trên các chiều yếu nhưng có thể quan trọng.

**P7 — Fixed rank bỏ qua task complexity.**
Mọi task dùng r=4 bất kể geometric complexity khác nhau. Từ C1 data: PaR(LLaMA, whitened) ≈ 9 nhưng PaR(T5, whitened) ≈ 24. Tasks dễ (SST-2, Yelp) có effective dimensionality thấp → r=4 đủ. Tasks khó (MNLI, DBpedia) có effective dimensionality cao → r=4 thiếu. Fixed rank = **one-size-fits-all** — hoặc lãng phí capacity (r quá lớn cho task dễ) hoặc thiếu capacity (r quá nhỏ cho task khó).

---

# PHẦN I: Nền tảng Toán học — Hình học của LoRA

---

## 1.1 Low-Rank Manifold

**Định nghĩa.** Cho $W_0 \in \mathbb{R}^{m \times n}$ (frozen weights), LoRA update:

$$\Delta W = BA, \quad B \in \mathbb{R}^{m \times r}, \; A \in \mathbb{R}^{r \times n}$$

Tập hợp hiệu dụng (effective parameter space):

$$\mathcal{M}_r = \{M \in \mathbb{R}^{m \times n} : \text{rank}(M) = r\}$$

**Fact 1 (Manifold structure).** $\mathcal{M}_r$ là smooth manifold nhúng trong $\mathbb{R}^{m \times n}$, có chiều:

$$\dim(\mathcal{M}_r) = r(m + n - r)$$

*Ref: Helmke & Moore, "Optimization and Dynamical Systems", 1994, §2.5.*

**So sánh**: Product space $(B, A)$ có chiều $r(m+n)$ — thừa $r^2$ chiều (gauge degrees of freedom).

## 1.2 Gauge Invariance

**Proposition 1 (Gauge symmetry).** Ánh xạ $\phi: (B, A) \mapsto BA$ có fiber:

$$\phi^{-1}(\Delta W) = \{(BM, M^{-1}A) : M \in GL(r)\} \cong GL(r)$$

Không gian tham số thực sự là quotient manifold:

$$\mathcal{M}_r \cong (\mathbb{R}^{m \times r}_* \times \mathbb{R}^{r \times n}_*) / GL(r)$$

trong đó subscript $*$ biểu thị full-rank matrices.

*Ref: Absil et al., "Optimization Algorithms on Matrix Manifolds", 2008, Ch. 3.*

**Hệ quả cho optimization:** Hàm loss $L(BA)$ constant trên mỗi gauge orbit → landscape có $r^2$-dimensional flat valleys. AdamW bước trên product space → di chuyển dọc flat valleys mà ΔW không thay đổi. BAR (2024) chứng minh: SAM implicit regularize "balancedness" $\|B\|_F^2 - \|A\|_F^2$, ép về gauge canonical.

## 1.3 SVD Canonical Form

**Proposition 2 (SVD decomposition).** Mỗi $\Delta W \in \mathcal{M}_r$ có unique SVD:

$$\Delta W = U \Sigma V^\top, \quad U \in St(r, m), \; \Sigma = \text{diag}(\sigma_1, \ldots, \sigma_r) \succ 0, \; V \in St(r, n)$$

trong đó $St(r, k) = \{X \in \mathbb{R}^{k \times r} : X^\top X = I_r\}$ là Stiefel manifold.

**Ý nghĩa**: SVD cung cấp canonical representative cho mỗi gauge orbit. Ba thành phần có ý nghĩa hình học rõ ràng:

| Component | Space | Ý nghĩa | CL relevance |
|-----------|-------|---------|--------------|
| $V$ | Stiefel $St(r, n)$ / Grassmannian $Gr(r, n)$ | **Input subspace** — thông tin nào từ input được capture | CL constraint: $V_t$ phải không overlap với $V_{s<t}$ |
| $\Sigma$ | $\mathbb{R}^r_{>0}$ | **Scales** — mức độ quan trọng mỗi direction | Effective rank, capacity utilization |
| $U$ | Stiefel $St(r, m)$ / Grassmannian $Gr(r, m)$ | **Output subspace** — thay đổi representation theo hướng nào | Ít bị CL constrain (different tasks có thể share output space) |

## 1.4 Tangent Space và Riemannian Gradient

**Theorem (Tangent space of $\mathcal{M}_r$).** Tại $\Delta W = U\Sigma V^\top$, tangent space:

$$T_{\Delta W}\mathcal{M}_r = \{U\Lambda V^\top + U_\perp C V^\top + U D V_\perp^\top : \Lambda \in \mathbb{R}^{r \times r}, C \in \mathbb{R}^{(m-r) \times r}, D \in \mathbb{R}^{r \times (n-r)}\}$$

Chiều = $r^2 + r(m-r) + r(n-r) = r(m+n-r)$ ✓.

**Riemannian gradient** (projection của Euclidean gradient):

$$\text{grad}_{\mathcal{M}_r} L = P_{T_{\Delta W}} \nabla_{\Delta W} L$$

trong đó $P_{T_{\Delta W}}(\xi) = U U^\top \xi V V^\top + U_\perp U_\perp^\top \xi V V^\top + U U^\top \xi V_\perp V_\perp^\top$.

*Ref: Vandereycken, "Low-Rank Matrix Completion by Riemannian Optimization", SIAM J. Optim., 2013.*

## 1.5 Grassmannian — Không gian Subspace

**Định nghĩa.** Grassmannian $Gr(r, n)$ là tập hợp tất cả r-dimensional subspaces of $\mathbb{R}^n$:

$$Gr(r, n) = St(r, n) / O(r)$$

**Tại sao quan trọng cho CL-LoRA:** CL constraint trên LoRA thực chất là constraint trên **input subspace** (row space of A, hoặc column space of V). Hai LoRA adapters có overlap khi input subspaces overlap:

$$\text{Overlap}(t, s) = \|V_t^\top V_s\|_F^2$$

Đây chính là **chordal distance squared** trên Grassmannian.

**Khoảng cách trên Grassmannian:**

| Distance | Formula | Tính chất |
|----------|---------|-----------|
| Chordal | $d_c^2 = r - \|V_t^\top V_s\|_F^2$ | Nhúng metric, dễ tính |
| Geodesic | $d_g = \|\theta\|_2$ ($\theta$ = principal angles) | Intrinsic, khó tính |
| Projection | $d_p = \|P_t - P_s\|_F$ ($P = VV^\top$) | = chordal up to constant |

Trong CL: **InfLoRA's hard projection = ép $d_c^2(V_t, V_{1:t-1}) = r$** (perfect orthogonality). Chúng ta đề xuất: **soft penalty trên $d_c^2$** (cho phép overlap nhỏ nếu có lợi).

*Ref: Edelman, Arias, Smith, "The Geometry of Algorithms with Orthogonality Constraints", SIAM J. Matrix Anal. Appl., 1998; Absil et al., 2004.*

## 1.6 Whitened Space: Khi nào $\Sigma_x$ = Input Covariance?

**Clarification quan trọng.** Cần phân biệt:

- **Routing embedding** $h = f_{\text{backbone}}(x_{\text{raw}})$: output cuối cùng của backbone (CLS token, etc.) — cái được whitened trong Contribution 1.
- **LoRA input** $x$: input đến weight matrix $W_Q$, $W_V$ cụ thể — đây là hidden state tại layer đó.

Fisher cho LoRA A sử dụng $\Sigma_x = \mathbb{E}[xx^\top]$ — covariance của **input** $x$ đến layer chứa LoRA, KHÔNG nhất thiết = output embedding.

**Proposition 1.6 (Layer-wise activation structure).**

Cho frozen backbone. Activations tại layer $l$: $x^{(l)} = f_l(x^{(l-1)})$. Input covariance cho LoRA tại layer $l$:

$$\Sigma_x^{(l)} = \mathbb{E}[x^{(l)} x^{(l)\top}]$$

**Trường hợp 1**: LoRA at final layer → $x^{(l)} \approx h$ (routing embedding) → $\Sigma_x \approx \Sigma_{\text{pool}}$ → whitening trực tiếp applicable.

**Trường hợp 2**: LoRA at intermediate layer → $\Sigma_x^{(l)} \neq \Sigma_{\text{pool}}$ nói chung. Tuy nhiên, **anisotropy structure** propagate qua layers: nếu layer $l$ là linear + residual, $\Sigma_x^{(l)}$ kế thừa dominant eigendirections từ input.

**Hệ quả thực tế**: $\Sigma_x^{(l)}$ cần compute per-layer (từ gradient probing data trong GGI, §3.1). Chi phí: amortized — đã có forward passes. Whitening insight từ C1 apply **qualitatively** (anisotropy → preconditioning helps) dù $\Sigma_x^{(l)} \neq \Sigma_{\text{pool}}$ exactly.

**Trường hợp lý tưởng (KF-Fisher simplification):** Nếu ta TRƯỚC hết whitening input tại mỗi layer: $\tilde{x}^{(l)} = (\Sigma_x^{(l)})^{-1/2} x^{(l)}$, thì Fisher cho LoRA A simplifies (adapting GLA's Bổ đề 2):

$$F_A^{(\text{whitened})} \approx (B^\top F_h B) \otimes I_{d_{\text{in}}}$$

→ Kronecker factor A-side trở thành identity → natural gradient cho A chỉ cần invert output-side Fisher $B^\top F_h B$ (kích thước $r \times r$ — rẻ!). Đây là **lý do toán học sâu** tại sao activation preconditioning giúp optimization.

*Insight adapted from GLA (Bổ đề 2), with layer-wise correction.*

---

# PHẦN II: Natural Gradient và Connection với Contribution 1

---

## 2.1 Fisher Information cho LoRA

Cho forward pass: $h = W_0 x + BAx$, loss $\ell(h, y)$.

**Gradient w.r.t. A:**
$$\frac{\partial \ell}{\partial A} = B^\top \frac{\partial \ell}{\partial h} \cdot x^\top \in \mathbb{R}^{r \times n}$$

**Fisher Information Matrix cho A** (expected outer product of gradients):
$$F_A = \mathbb{E}\!\left[\text{vec}\!\left(\frac{\partial \ell}{\partial A}\right) \text{vec}\!\left(\frac{\partial \ell}{\partial A}\right)^\top\right]$$

Dưới Gaussian approximation:
$$F_A \approx \underbrace{(B^\top F_h B)}_{\text{output Fisher}} \;\otimes\; \underbrace{\Sigma_x}_{\text{input covariance}}$$

trong đó $F_h = \mathbb{E}\!\left[\frac{\partial \ell}{\partial h}\frac{\partial \ell}{\partial h}^\top\right]$ và $\Sigma_x = \mathbb{E}[xx^\top]$.

**Natural gradient cho A:**
$$\tilde{g}_A = F_A^{-1} \, \text{vec}(g_A) \;\Leftrightarrow\; \tilde{G}_A = (B^\top F_h B)^{-1} \, G_A \, \Sigma_x^{-1}$$

## 2.2 Kết nối Sâu sắc với Contribution 1

> **Observation cốt lõi:** Natural gradient cho LoRA A có factor $\Sigma_x^{-1}$ — chính xác là **phép biến đổi whitening / Mahalanobis** từ Contribution 1.

| Contribution 1 (SRT) | Contribution 2 (proposed) | Cấu trúc toán chung |
|-----------------------|---------------------------|---------------------|
| Routing metric: $(h-\mu_t)^\top \Sigma_{\text{pool}}^{-1} (h-\mu_t)$ | Natural gradient: $G_A \, \Sigma_x^{-1}$ | $\Sigma_x^{-1}$ preconditioning |
| Whitening = optimal routing preconditioner | Whitening = optimal training preconditioner | ZCA transform |
| Mahalanobis = Fisher-Rao first-order | Natural gradient = Fisher-Rao gradient | Fisher information geometry |
| PaR = effective dimensionality cho routing | PaR = effective rank budget cho training | Participation ratio |

**Theorem 2.1 (Duality Routing–Training).** Dưới Gaussian model cho activations:
- **Routing** tối ưu khi dùng $\Sigma_{\text{pool}}^{-1}$-weighted distance (Contribution 1, Theorem 4)
- **Training** tối ưu khi dùng $\Sigma_x^{-1}$-weighted gradient (natural gradient)
- Khi $\Sigma_{\text{pool}} \approx \Sigma_x$ (frozen backbone, activations ≈ stationary):

$$\text{Routing metric} \propto \text{Training preconditioning}$$

*Proof.* Routing dùng Fisher-Rao distance trên statistical manifold $\{P_\theta\}$ induced bởi biến đổi $h \mapsto \theta(h)$. Training dùng Fisher information matrix of same manifold $\{P_\theta\}$ as preconditioner for gradient. Cả hai xuất phát từ cùng Fisher metric tensor $g_{ij}(\theta) = \mathbb{E}[\partial_i \log p \cdot \partial_j \log p]$; routing dùng nó làm distance, training dùng nó làm gradient rescaling. $\square$

> **Ý nghĩa**: Contribution 1 và 2 chia sẻ cùng nền tảng information-geometric. Cải thiện routing (C1) và cải thiện training (C2) đều đi qua cùng cấu trúc — $\Sigma_x^{-1}$ preconditioning. Đây là **duality**, không phải trùng lặp.

## 2.3 Hệ quả cho Anisotropic Embeddings

Từ Contribution 1: $\kappa(\Sigma_x) \approx 132$–$439$, PaR $\approx 9$–$27$.

**Hệ quả cho LoRA training:**

1. **Gradient anisotropy**: $g_A$ có magnitude tập trung ở top eigendirections của $\Sigma_x$. Standard AdamW (diagonal preconditioner) không sufficient — nó chỉ scale per-parameter, không capture cross-parameter correlations từ $\Sigma_x$.

2. **Rank budget waste**: Với r=4, nếu A aligns với top-4 eigendirections (magnitude lớn nhất), LoRA chỉ điều chỉnh backbone theo directions mà backbone ĐÃ dominant. Task-specific information nằm ở **residual directions** (eigenvalues nhỏ hơn) bị bỏ qua.

3. **Effective rank mismatch**: PaR ≈ 9 cho LLaMA → embedding chỉ có ~9 effective dimensions. Với r=4, LoRA cover ~44% effective space — đủ nếu chọn đúng subspace. Nhưng random init → chọn sai → cần nhiều epochs để converge.

---

# PHẦN III: Framework Đề xuất — Bốn Thành phần

---

## 3.0 Component 0: Task-Adaptive Rank Allocation (TARA)

> *Adapted from GLA framework (con2_cl.md), with corrected numerical estimates and tightened theory.*

### Vấn đề (P7)
Rank r cố định cho mọi task. Nhưng tasks có geometric complexity khác nhau — task đơn giản (sentiment: SST-2) cần ít rank, task phức tạp (NLI: MNLI, multi-class: DBpedia) cần nhiều hơn.

### Định nghĩa (Task Geometric Complexity — TGC)

Cho task $t$ với **input** activations $x$ tại LoRA layer. Compute per-layer activation covariance $\Sigma_x^{(l)}$ và PaR:

$$\text{TGC}_t^{(l)} = \text{PaR}(\Sigma_x^{(l)}) = \frac{(\text{tr}(\Sigma_x^{(l)}))^2}{\text{tr}((\Sigma_x^{(l)})^2)}$$

**Lưu ý quan trọng — Phân biệt PaR raw vs whitened:**

| Space | PaR ý nghĩa | Data từ C1 |
|-------|------------|------------|
| **Raw** activation space | Intrinsic dimensionality of activations | T5: 24, LLaMA: 9 |
| **Whitened** activation space | Dimensionality relative to global structure | Gần $d$ nếu $\Sigma_t \approx \Sigma_{\text{pool}}$ |
| **Task-residual** space: $\Sigma_t - \Sigma_{\text{pool}}$ | Task-SPECIFIC dimensionality | Thường << raw PaR |

Cho TARA, ta cần **task-specific PaR** — effective rank của phần task cần LoRA thay đổi, không phải raw activation dimensionality. Cụ thể:

$$\text{TGC}_t^{\text{eff}} = \text{PaR}\!\left(\Sigma_{\text{grad}}^{(l)}\right) = \frac{(\text{tr}(\Sigma_{\text{grad}}))^2}{\text{tr}(\Sigma_{\text{grad}}^2)}$$

trong đó $\Sigma_{\text{grad}}$ = gradient covariance (từ GGI probing, §3.1). Gradient PaR đo **bao nhiêu independent directions** task gradient sống trong — chính xác là bao nhiêu rank LoRA cần.

### Theorem 3.0 (TARA Rank Bound)

**Claim**: Cho task $t$ với gradient covariance $\Sigma_{\text{grad}}$ có effective rank $\text{TGC}_t^{\text{eff}}$. LoRA rank $r_t$ đủ khi:

$$r_t \geq \text{TGC}_t^{\text{eff}} \quad \Rightarrow \quad \|G_t - \hat{G}_t^{(r_t)}\|_F \leq \epsilon \|G_t\|_F$$

trong đó $\hat{G}_t^{(r)}$ là best rank-$r$ approximation of gradient, và $\epsilon$ controlled by spectral gap.

*Proof.* Eckart-Young-Mirsky theorem: best rank-$r$ approximation error = $\sum_{i>r} \sigma_i^2$. Khi $r \geq \text{TGC}_t^{\text{eff}}$, spectrum đã decay đủ (by PaR definition, eigenvalues beyond PaR contribute < 1/PaR fraction). Formally:

$$\frac{\sum_{i>r} \sigma_i^2}{\sum_i \sigma_i^2} \leq 1 - \frac{r}{\text{TGC}_t^{\text{eff}}} \cdot \frac{1}{r} \sum_{i=1}^r \sigma_i^2 / \bar{\sigma}^2$$

Bound holds when $r \geq \text{TGC}_t^{\text{eff}}$. $\square$

### Practical TARA Algorithm

```
TARA(task t, probing_data, r_min=4, r_max=64):
  # 1. Gradient probing (shared with GGI Phase 1)
  Σ_grad = gradient_covariance(probing_data)  # Already computed in GGI!
  
  # 2. Compute task-specific effective rank
  eigenvalues = eigendecomposition(Σ_grad)
  TGC_eff = participation_ratio(eigenvalues)
  
  # 3. Adaptive rank with CL headroom
  available_dims = d_in - dim(S_prev)         # Subspace budget left
  r_t = max(r_min, min(r_max, ceil(TGC_eff), available_dims))
  
  return r_t
```

### Numerical Estimates (Corrected)

| Backbone | Gradient PaR (estimated) | Recommended $r_t$ | Notes |
|----------|------------------------|--------------------|----|
| T5-large | ~8–20 (varies per task) | $r \in [8, 24]$ | Baseline r=4 may be insufficient for complex NLI tasks |
| LLaMA-2 | ~4–12 (varies per task) | $r \in [4, 16]$ | Low PaR implies r=8 often sufficient |
| Easy tasks (SST-2, Yelp) | ~4–6 | $r=4$ sufficient | Sentiment = low-dimensional |
| Hard tasks (MNLI, DBpedia) | ~12–20 | $r=16$ helps | Multi-class = higher dimensional |

**Lưu ý**: Những con số này là **estimates dựa trên raw PaR từ C1 data** và tỉ lệ gradient/activation. Cần verify bằng experiment (E-TARA, §6.1). GLA's estimates (TGC=310, 864) dùng **whitened full-space PaR** — overestimates needed rank vì không loại phần backbone đã represent tốt.

**Cost**: ZERO extra — TARA reuses gradient covariance từ GGI probing phase.

## 3.1 Component 1: Gradient-Geometry Initialization (GGI)

### Vấn đề
Random init (Kaiming) cho A → random subspace. InfLoRA project vào null space của previous tasks, nhưng vẫn random WITHIN null space.

### Đề xuất

**Bước 1 — Gradient probing.** Trước khi training, chạy $K$ forward-backward passes trên task data (không cập nhật weights) để thu thập gradient covariance:

$$\hat{\Sigma}_{\text{grad}} = \frac{1}{K}\sum_{k=1}^K \text{vec}(\nabla_{W} \ell_k) \cdot \text{vec}(\nabla_{W} \ell_k)^\top$$

Trong thực tế, chỉ cần gradient covariance trên input side:
$$\hat{\Sigma}_{\text{grad}}^{(x)} = \frac{1}{K}\sum_k (x_k x_k^\top) \odot \left(\frac{\partial \ell_k}{\partial h} \frac{\partial \ell_k}{\partial h}^\top \right) \in \mathbb{R}^{n \times n}$$

(Xấp xỉ: dùng outer product of loss-weighted activations — tương tự GaLore/LoRA-GA.)

**Bước 2 — Task-informative subspace via generalized eigenvalue problem.**

Tìm directions có gradient signal cao nhưng backbone activation thấp (task-specific, không phải backbone-dominant):

$$\hat{\Sigma}_{\text{grad}}^{(x)} v = \lambda \, \hat{\Sigma}_x \, v$$

Top-r generalized eigenvectors $\{v_1, \ldots, v_r\}$ tạo thành **task-informative subspace**.

**Ý nghĩa**: Generalized eigenvalue problem tìm directions maximize $\frac{v^\top \Sigma_{\text{grad}} v}{v^\top \Sigma_x v}$ — tỉ số gradient signal / activation magnitude. Directions có tỉ số cao = nơi task cần thay đổi NHIỀU so với mức backbone đã represent → task-specific.

**Bước 3 — CL-constrained initialization.**

Nếu có previous tasks' protected subspace $\mathcal{S}_{\text{prev}} = \text{span}(V_1, \ldots, V_{t-1})$:

$$V_t^{(0)} = \text{top-}r\text{ generalized eigenvectors of } (\hat{\Sigma}_{\text{grad}}^{(x)}, \hat{\Sigma}_x) \text{ restricted to } \mathcal{S}_{\text{prev}}^\perp$$

Cụ thể: project cả $\hat{\Sigma}_{\text{grad}}^{(x)}$ và $\hat{\Sigma}_x$ vào null space:
$$P_\perp = I - V_{\text{prev}} V_{\text{prev}}^\top$$
$$\tilde{\Sigma}_{\text{grad}} = P_\perp \hat{\Sigma}_{\text{grad}}^{(x)} P_\perp, \quad \tilde{\Sigma}_x = P_\perp \hat{\Sigma}_x P_\perp$$

Giải generalized EVP trên projected space.

**Bước 4 — Set A and B.**
$$A^{(0)} = V_t^{(0)\top} \in \mathbb{R}^{r \times n}, \quad B^{(0)} = \sigma_{\text{init}} \cdot U_t^{(0)} \in \mathbb{R}^{m \times r}$$

trong đó $U_t^{(0)}$ = top-r left singular vectors của gradient $\nabla_W \ell$ projected vào subspace tương ứng, $\sigma_{\text{init}} > 0$ nhỏ (không = 0 → fixes P2).

**So sánh init strategies:**

| Method | A init | B init | Subspace quality | CL aware? |
|--------|--------|--------|-----------------|-----------|
| Standard LoRA | Kaiming random | zeros | Random | ❌ |
| InfLoRA | Kaiming → project to null space | zeros | Random in null space | ✅ (hard) |
| LoRA-GA | Gradient-aligned | Gradient-aligned | Task-informed | ❌ |
| GaLore | SVD of gradient | SVD of gradient | Task-informed | ❌ |
| **GGI (proposed)** | Generalized EVP (grad/activation ratio) | Small non-zero | **Task-specific** (not just high-gradient) | ✅ (hard at init, soft during training) |

**Cost**: $K$ forward-backward passes (K ≈ 50–100). Same order as 1 epoch → overhead nhỏ so với 50 epochs training.

*Ref: GaLore — Zhao et al., "GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection", ICML 2024; LoRA-GA — Wang et al., "LoRA-GA: Low-Rank Adaptation with Gradient Approximation", NeurIPS 2024.*

## 3.2 Component 2: Soft Grassmannian Regularization (SGR)

### Vấn đề
InfLoRA's hard projection trên A là discontinuous, lossy, và không phân biệt importance.

### Đề xuất

Thay hard projection bằng **differentiable Grassmannian penalty** trong loss function:

$$\mathcal{L}_{\text{Grass}} = \sum_{s=1}^{t-1} w_s \cdot \|V_t^\top V_s\|_F^2$$

trong đó:
- $V_t \in \mathbb{R}^{n \times r}$: orthonormal basis cho row space of A (via thin QR hoặc SVD)
- $V_s$: protected subspace basis từ task $s$ 
- $w_s$: importance weight cho task $s$ (xem bên dưới)

### Importance Weighting

Không phải tất cả previous subspace dimensions đều quan trọng bằng nhau. Weight dựa trên singular values:

$$w_s = \frac{\text{tr}(\Sigma_s^{(\text{top-}r)})}{\sum_{s'} \text{tr}(\Sigma_{s'}^{(\text{top-}r)})}$$

(Tasks có activation variance lớn hơn được bảo vệ mạnh hơn.)

### Tính chất toán học

**Proposition 3.1 (Gradient of Grassmannian penalty).**

$$\frac{\partial}{\partial A} \|V_t^\top V_s\|_F^2 = 2 (A^\dagger)^\top V_t V_t^\top V_s V_s^\top$$

trong đó $A^\dagger = A^\top(AA^\top)^{-1}$ là right pseudoinverse. Continuous và differentiable khi $A$ full rank.

*Proof.* Dùng chain rule qua QR decomposition: $A = RV_t^\top$ → $V_t = (A^\top R^{-\top})[:, 1:r]$. Xem Absil & Malick, "Projection-like Retractions on Matrix Manifolds", SIAM J. Optim., 2012. $\square$

**Proposition 3.2 (SGR is smooth relaxation of InfLoRA).**
- Khi $\lambda_{\text{Grass}} \to \infty$: SGR converges to exact orthogonality (= InfLoRA's hard projection)
- Khi $\lambda_{\text{Grass}} = 0$: No CL constraint
- Finite $\lambda_{\text{Grass}}$: smooth tradeoff task quality vs CL protection

**Ưu điểm so với hard projection:**

| Aspect | Hard projection (InfLoRA) | SGR (proposed) |
|--------|--------------------------|----------------|
| Continuity | Discontinuous | Smooth (differentiable) |
| Gradient flow | Blocks gradients qua projection | Full gradient flow |
| Flexibility | Binary: orthogonal hoặc không | Continuous: allows small overlap |
| Importance weighting | All directions equal | Weighted by activation variance |
| Convergence | No guarantees (non-smooth) | Standard smooth optimization guarantees |

*Ref: Edelman et al., "The Geometry of Algorithms with Orthogonality Constraints", 1998; Sato & Iwai, "A Riemannian Optimization Approach to the Matrix Singular Value Decomposition", SIAM J. Optim., 2013.*

## 3.3 Component 3: Balanced Natural Gradient (BNG)

### Vấn đề
AdamW trên $(B, A)$: (1) bỏ qua gauge symmetry, (2) bỏ qua activation covariance structure.

### Đề xuất: Ba điều chỉnh

**3.3a — Asymmetric Learning Rates (từ LoRA+ insight, refined).**

Lý thuyết: A xác định input subspace (Grassmannian component), B xác định output projection + scale. Gradient magnitude cho A và B khác nhau:

$$\|\nabla_B L\| = \|(\partial L/\partial h) \cdot x^\top A^\top\| \propto \|A\| \cdot \sigma_x$$
$$\|\nabla_A L\| = \|B^\top (\partial L/\partial h) \cdot x^\top\| \propto \|B\| \cdot \sigma_x$$

Khi $B^{(0)} = 0$: $\|\nabla_A L\|^{(0)} = 0$ (vấn đề P2). Khi $B \neq 0$ nhưng nhỏ: $\|\nabla_A L\| \ll \|\nabla_B L\|$.

**Đề xuất**:
$$\eta_A = \eta \cdot \beta, \quad \eta_B = \eta / \beta, \quad \beta = \sqrt{\|B\|_F / \|A\|_F}$$

Balance factor $\beta$ adaptively equalizes gradient scales. Khi balanced ($\|B\|_F = \|A\|_F$): $\beta = 1$, same LR. Khi unbalanced: tăng LR cho factor nhỏ, giảm cho factor lớn.

**3.3b — Activation-Preconditioned Gradient cho A (KF-Fisher Insight).**

Từ §1.6 (Fisher simplification): khi input được whitening, Fisher cho A trở thành $(B^\top F_h B) \otimes I$. Ngược lại, trong raw space: $F_A \approx (B^\top F_h B) \otimes \Sigma_x$. Do đó, **activation preconditioning = implicit whitening** cho LoRA:

$$A^{(t+1)} = A^{(t)} - \eta_A \cdot \hat{G}_A \cdot \hat{\Sigma}_x^{-1/2}$$

trong đó $\hat{\Sigma}_x^{-1/2}$ là inverse square root của **layer-wise** activation covariance.

**Tại sao $\Sigma_x^{-1/2}$ cần thiết dù đã có AdamW?** AdamW dùng **diagonal** second-moment: $m_t = \beta_2 m_{t-1} + (1-\beta_2)g_t^2$. Nhưng $\Sigma_x$ có **off-diagonal** structure quan trọng (anisotropy). AdamW rescale per-element, không capture cross-element correlations.

**Connection mạnh hơn**: Từ KF-Fisher analysis (GLA's insight, corrected):
- Trong raw space: natural gradient cần invert cả $\Sigma_x$ (size $d_{in} \times d_{in}$) VÀ $B^\top F_h B$ (size $r \times r$).
- Sau preconditioning bằng $\Sigma_x^{-1/2}$: chỉ cần invert $B^\top F_h B$ (kích thước $r \times r$ — trivial!).
- Gain lý thuyết: từ $O(d_{in}^3 + r^3)$ → $O(r^3)$ cho natural gradient step.

**Practical approximation:** Không tính full $\Sigma_x^{-1/2}$ (quá đắt). Thay vào:
1. Compute top-k eigenvectors của $\hat{\Sigma}_x$ (đã có từ GGI, bước 1)
2. Dùng low-rank approximation: $\hat{\Sigma}_x^{-1/2} \approx V_k \Lambda_k^{-1/2} V_k^\top + \bar{\lambda}^{-1/2}(I - V_k V_k^\top)$
3. Cost: $O(nk)$ per step, negligible so với forward pass. $k$ = eigenvectors kept = PaR (từ TARA) — natural choice.

**3.3c — Balance Regularization.**

Từ BAR (2024): thêm regularizer ép gauge balance:

$$\mathcal{L}_{\text{bal}} = (\|B\|_F^2 - \|A\|_F^2)^2$$

Đây là proxy cho SAM mà rẻ hơn ~95% (BAR's result). Ép $(B, A)$ về canonical gauge $\|B\|_F = \|A\|_F$.

**Tại sao balance matters cho BNG?** Balanced gauge ($\|B\|_F = \|A\|_F$) → gradient magnitudes $\|\nabla_A L\| \approx \|\nabla_B L\|$ → adaptive $\beta \approx 1$ → optimizer ổn định. Unbalanced → $\beta$ dao động → learning instability.

*Ref: LoRA+ — Hayou et al., "LoRA+: Efficient Low Rank Adaptation of Large Models", ICML 2024; BAR — Chen et al., "Understanding and Improving Sharpness-Aware Minimization for Scale-Invariant Problems", ICLR 2025.*

---

# PHẦN IV: Total Loss Function và Algorithm

---

## 4.1 Loss Function

$$\boxed{\mathcal{L}_{\text{total}} = \underbrace{\mathcal{L}_{\text{CE}}(W_0 + BA)}_{\text{task performance}} + \underbrace{\lambda_1 \sum_{s<t} w_s \|V_t^\top V_s\|_F^2}_{\text{Grassmannian overlap (SGR)}} + \underbrace{\lambda_2 (\|B\|_F^2 - \|A\|_F^2)^2}_{\text{Balance (BAR insight)}}}$$

**Bốn thành phần, bốn mục tiêu:**

| Term | Mục tiêu | Source |
|------|---------|--------|
| $\mathcal{L}_{\text{CE}}$ | Minimize task loss | Standard |
| $\mathcal{L}_{\text{Grass}}$ | Minimize subspace overlap với previous tasks | Novel cho CL-LoRA |
| $\mathcal{L}_{\text{bal}}$ | Ép gauge balance, stable optimization | Từ BAR, adapted |
| TARA (implicit) | Rank = f(task complexity) | Novel: geometry-driven |

**Hyperparameters:**
- $\lambda_1$: CL protection strength. Nếu $\lambda_1 \to \infty$: recover hard orthogonality. Nếu $\lambda_1 = 0$: no CL protection.
- $\lambda_2$: Balance strength. Nhỏ thôi (BAR shows $\lambda_2 \sim 0.01$ đủ).
- $K$: Gradient probing steps cho GGI ($K \sim 50$–$100$).
- $\beta$ (adaptive LR): computed dynamically, no tuning needed.

## 4.2 Full Algorithm

```
GALA_Train(task t, frozen W_0, previous subspaces {V_s}_{s<t}, activation covariance Σ_x):

  # ═══ Phase 0: Gradient Probing (shared across TARA + GGI) ═══
  Σ_grad = 0; Σ_x_layer = 0
  for k = 1..K:
      x_k, y_k = sample(task_t_data)
      ℓ_k = CE(W_0 · x_k, y_k)
      g_k = ∇_W ℓ_k
      Σ_grad += outer(g_k · x_k^T)  # input-side gradient covariance
      Σ_x_layer += outer(x_k, x_k)  # layer-wise activation covariance
  Σ_grad /= K; Σ_x_layer /= K

  # ═══ Phase 1: TARA — Adaptive Rank ═══
  TGC_eff = participation_ratio(eigenvalues(Σ_grad))
  available_dims = d_in - dim(span(V_1,...,V_{t-1}))
  r_t = max(r_min, min(r_max, ceil(TGC_eff), available_dims))

  # ═══ Phase 2: GGI — Gradient-Geometry Initialization ═══
  # 2a. Project to null space of previous tasks
  P_perp = I - V_prev · V_prev^T    # V_prev = [V_1 | ... | V_{t-1}]
  Σ_grad_proj = P_perp · Σ_grad · P_perp
  Σ_x_proj = P_perp · Σ_x_layer · P_perp

  # 2b. Generalized eigenvalue problem (rank = r_t from TARA!)
  V_t^(0) = top_{r_t}_generalized_eigenvectors(Σ_grad_proj, Σ_x_proj)
  
  # 2c. Initialize A and B
  A = V_t^(0)^T                                    # rows = task-informative directions
  B = σ_init · gradient_aligned_output_vectors()    # small non-zero, not 0!

  # ═══ Phase 3: Compute preconditioning matrix (from probing data) ═══
  eigvecs, eigvals = top_k_eig(Σ_x_layer, k=r_t)   # reuse TARA's rank
  Σ_x_inv_sqrt ≈ eigvecs · diag(eigvals^{-1/2}) · eigvecs^T 
                  + mean(eigvals)^{-1/2} · (I - eigvecs · eigvecs^T)

  # ═══ Phase 4: Training with SGR + BNG ═══
  for epoch = 1..num_epochs:
    for batch (x, y) in task_t_data:
      
      # Forward
      h = W_0 · x + B · A · x
      L_CE = CrossEntropy(h, y)

      # Grassmannian penalty (SGR)
      V_t = orthonormal_basis(row_space(A))  # via QR
      L_Grass = λ₁ · Σ_{s<t} w_s · ||V_t^T · V_s||_F²

      # Balance regularization
      L_bal = λ₂ · (||B||_F² - ||A||_F²)²

      # Total loss
      L = L_CE + L_Grass + L_bal

      # Gradients
      g_A = ∂L/∂A
      g_B = ∂L/∂B

      # Balanced Natural Gradient (BNG)
      β = sqrt(||B||_F / max(||A||_F, ε))
      
      # Precondition g_A by Σ_x^{-1/2} (low-rank approx from Phase 3)
      g_A_precond = g_A · Σ_x_inv_sqrt   # activation-aware

      # Update with asymmetric LR
      A -= η·β · AdamW_step(g_A_precond)
      B -= η/β · AdamW_step(g_B)

  # ═══ Phase 5: Post-training ═══
  # Store V_t for future CL tasks
  V_t = orthonormal_basis(row_space(A))
  
  # Update Σ_x incrementally (if needed for SRT routing)
  Σ_x = (N_old · Σ_x + n_t · Σ_x_new) / (N_old + n_t)

  return (A, B, V_t, r_t)
```

**Key change vs v1**: Phase 0 gradient probing is SHARED across TARA (rank selection) and GGI (subspace init). Zero extra cost for TARA.

## 4.3 Complexity Analysis

| Component | Cost | When | Overhead vs GainLoRA |
|-----------|------|------|---------------------|
| Gradient probing (shared TARA+GGI) | $K \times$ forward-backward | Before training (once per task) | $K/N_{\text{epochs}} \approx 2\%$ |
| TARA eigendecomposition | $O(n^2)$ | Once per task | Negligible |
| GGI generalized EVP | $O(n^2 r)$ | Once per task | Negligible |
| Preconditioning matrix | $O(n^2 k)$ | Once per task | Negligible |
| SGR penalty | $O(r^2 \cdot t)$ per step | During training | Negligible ($r \leq 16, t\leq 15$) |
| QR for $V_t$ | $O(nr^2)$ per step | During training | Negligible |
| $\Sigma_x^{-1/2}$ preconditioning | $O(nk)$ per step (low-rank) | During training | ~5% (amortized by low-rank approx) |
| Balance reg | $O(mr + nr)$ per step | During training | Negligible |

**Total overhead**: ~7% so với standard GainLoRA training. Marginal cost, potential significant gain. TARA adds ZERO extra cost (reuses probing data).

---

# PHẦN V: Phân tích Lý thuyết

---

## 5.1 GGI Optimality

**Theorem 5.1 (GGI finds task-informative subspace).** Cho activations $x \sim \mathcal{P}_t$ có covariance $\Sigma_x$ và gradient covariance $\Sigma_{\text{grad}}$. GGI initialization $V_t^{(0)}$ maximizes:

$$V_t^{(0)} = \arg\max_{V \in Gr(r, n),\, V \perp \mathcal{S}_{\text{prev}}} \frac{\text{tr}(V^\top \Sigma_{\text{grad}} V)}{\text{tr}(V^\top \Sigma_x V)}$$

**Ý nghĩa**: Tỉ số = "gradient signal per unit activation variance". High ratio = direction cần LoRA update nhiều nhưng backbone represent ít → task-specific.

*Proof.* Direct from Rayleigh quotient theory cho generalized eigenvalue problem. Top-r eigenvectors maximize trace ratio (Fukunaga, "Introduction to Statistical Pattern Recognition", 1990, §10.3). Constraint $V \perp \mathcal{S}_{\text{prev}}$ thêm vào bằng projection (orthogonal complement). $\square$

**So sánh:**
- Random init: ratio ≈ 1 (no preference)
- Gradient-only init (LoRA-GA): maximizes $\text{tr}(V^\top \Sigma_{\text{grad}} V)$ — có thể chọn high-gradient directions mà backbone ĐÃ represent tốt (waste rank)
- **GGI**: maximizes RATIO → chọn directions mà LoRA cần thêm vào backbone, không phải directions mà backbone đã mạnh

## 5.2 SGR Convergence

**Theorem 5.2 (SGR penalty is Lipschitz smooth).** $\mathcal{L}_{\text{Grass}}(A) = \sum_s w_s \|V_t(A)^\top V_s\|_F^2$ là $C^1$-smooth khi A full rank, với Lipschitz gradient constant:

$$L_{\text{Lip}} \leq 2 \sum_s w_s \cdot \sigma_{\min}(A)^{-2} \cdot \|V_s\|^2$$

**Hệ quả**: Khi kết hợp với CE loss (assumed Lipschitz smooth), total loss $\mathcal{L}_{\text{total}}$ smooth → SGD/Adam converges to stationary point at rate $O(1/\sqrt{T})$.

**Lưu ý (honest assessment)**: Rate $O(1/\sqrt{T})$ là convergence to **stationary point** ($\min_t \|\nabla L\|^2 \leq O(1/\sqrt{T})$), KHÔNG phải global minimum. LoRA training trên neural network là **non-convex**. Linear convergence rate (exponential) chỉ đúng cho strongly convex problems — không applicable ở đây. Tuy nhiên, smoothness of SGR (vs discontinuity of hard projection) vẫn giúp optimizer tránh oscillation và achieve better practical convergence.

## 5.3 TARA Optimality (Adapted from GLA, corrected)

**Theorem 5.3 (Adaptive Rank Sufficiency).** Cho gradient covariance $\Sigma_{\text{grad}}$ với eigenvalues $\lambda_1 \geq \lambda_2 \geq \ldots \geq 0$ và effective rank $\text{TGC}_{\text{eff}} = \text{PaR}(\Sigma_{\text{grad}})$. Best rank-$r$ approximation error:

$$\frac{\sum_{i>r} \lambda_i^2}{\sum_i \lambda_i^2} \leq 1 - \frac{r}{\text{TGC}_{\text{eff}}}$$

khi $r \leq \text{TGC}_{\text{eff}}$, với bound chặt khi eigenvalues uniform.

*Proof.* PaR = $(\sum \lambda_i)^2 / \sum \lambda_i^2$. Dùng Cauchy-Schwarz và rearrangement inequality. Khi top-$r$ eigenvalues capture fraction $\alpha$ of total variance: $\alpha \geq r / \text{PaR}$ (minimax bound for balanced spectra). Residual $\leq 1 - \alpha \leq 1 - r/\text{PaR}$. $\square$

**Hệ quả**: Khi $r_t = \lceil \text{TGC}_{\text{eff}} \rceil$: residual $\leq 0$ (perfect). Khi $r_t = \text{TGC}_{\text{eff}} / 2$: residual $\leq 50\%$. Khi $r_t = 4$ nhưng $\text{TGC}_{\text{eff}} = 16$: residual $\leq 75\%$ — LoRA chỉ capture 25% gradient signal.

**Connection to GGI**: TARA chọn rank $r_t$ → GGI chọn ĐÚNG $r_t$ directions tốt nhất (maximize gradient/activation ratio). TARA + GGI together: optimal rank AND optimal directions = minimal information loss.

## 5.4 Subspace Utilization Efficiency

**Định nghĩa (Effective rank of LoRA update).**

$$r_{\text{eff}}(\Delta W) = \frac{(\sum_i \sigma_i)^2}{\sum_i \sigma_i^2} = \text{PaR}(\Delta W)$$

trong đó $\sigma_i$ là singular values của $\Delta W = BA$.

**Claim**: GGI initialization → $r_{\text{eff}} \approx r$ (full utilization). Random init → $r_{\text{eff}} < r$ (wasted dimensions).

**Lý do**: GGI chọn directions theo generalized eigenvalue → các directions có gradient ratios tương đương → singular values tương đương → PaR cao. Random init → một số directions có gradient lớn, số khác nhỏ → singular values spread lớn → PaR thấp.

## 5.5 Information-Theoretic Perspective

**Proposition 5.3 (LoRA as Information Bottleneck).**

LoRA update $\Delta W$ tạo ra một information bottleneck với capacity $r \cdot \log\left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)$. Under fixed rank $r$:

$$I(Y; \Delta W \cdot X) \leq r \cdot \log\left(1 + \frac{\sigma_{\max}^2 \cdot \text{Var}(X)}{\sigma_{\text{noise}}^2}\right)$$

**Ý nghĩa**: Capacity bị giới hạn bởi $r$ VÀ condition number. GGI maximizes information throughput bằng cách chọn directions có signal/noise ratio cao nhất → efficiently uses limited capacity.

*Ref: Tishby & Zaslavsky, "Deep Learning and the Information Bottleneck Principle", ITW 2015; Shwartz-Ziv & Tishby, "Opening the Black Box of Deep Neural Networks via Information", 2017.*

---

# PHẦN VI: Experiment Plan

---

## 6.1 Ablation Design

| Experiment | What it tests | Expected result |
|-----------|---------------|-----------------|
| **E0**: TARA rank sweep | Rank vs task accuracy saturation | Accuracy saturates at $r \approx \text{TGC}_{\text{eff}}$ |
| **E1**: GGI init vs random init vs LoRA-GA init | Initialization quality | GGI > LoRA-GA > random (vì ratio-based, task-specific) |
| **E2**: SGR vs hard projection (InfLoRA) | CL protection quality | SGR ≈ hard ở T=15, SGR > hard ở T=30+ (flexibility helps) |
| **E3**: BNG vs AdamW vs LoRA+ | Optimizer quality | BNG > LoRA+ > AdamW (activation preconditioning helps) |
| **E4**: Full GALA vs GainLoRA baseline | End-to-end comparison | GALA > GainLoRA on both AP and FT |
| **E5**: Effective rank tracking | Subspace utilization | GGI → r_eff ≈ r; random → r_eff < r |
| **E6**: λ₁ sweep for SGR | Hard vs soft orthogonality | Optimal λ₁ > 0, < ∞ (sweet spot between flexibility and protection) |
| **E7**: TARA + GGI interaction | Does adaptive rank improve GGI? | TARA+GGI > fixed-rank+GGI (especially for hard tasks) |

### E0 Protocol — TARA Validation
1. Vary LoRA rank: $r \in \{2, 4, 8, 16, 32\}$ per task, fixed init (standard LoRA).
2. Measure final task accuracy vs $r$.
3. Compute $\text{TGC}_{\text{eff}}$ from gradient probing.
4. Verify: accuracy improvement from $r$ to $2r$ becomes negligible when $r \geq \text{TGC}_{\text{eff}}$.
5. Compare TARA's selected rank vs best oracle rank → should match within ±2.

## 6.2 Metrics

| Metric | What it measures | Formula |
|--------|-----------------|---------|
| AP (Average Performance) | Task quality after all tasks trained | $\frac{1}{T}\sum_t \text{acc}_t^{(T)}$ |
| FT (Forward Transfer) | Task-specific training quality | $\text{acc}_t^{(t)}$ (accuracy right after training task $t$) |
| BWT (Backward Transfer) | Forgetting | $\frac{1}{T-1}\sum_t (\text{acc}_t^{(T)} - \text{acc}_t^{(t)})$ |
| $r_{\text{eff}}$ | Subspace utilization | PaR of $\Delta W$ |
| Grassmannian overlap | CL compatibility | $\|V_t^\top V_s\|_F^2$ |

## 6.3 Backbones & Benchmarks

Same as Contribution 1 experiments:
- flan-t5-large (d=1024), flan-t5-xl (d=2048), Llama-2-7b (d=4096)
- Long_Sequence (15 tasks), SuperNI (15 tasks)
- Multiple task orders (order1, order2, order3, order4)

---

# PHẦN VII: Novelty Assessment — Honest

---

## 7.1 What IS Novel

**N0 — Task-Adaptive Rank Allocation (TARA):** Rank allocation dựa trên **gradient PaR** — effective rank của gradient covariance, không phải activation importance scores (AdaLoRA) hay embedding-space PaR. Gradient PaR đo chính xác bao nhiêu independent directions task gradient cần → bao nhiêu rank LoRA cần. Zero extra cost (reuses GGI probing). *Adapted from GLA's TGC concept, corrected numerics.*

**N1 — Gradient-Geometry Initialization (GGI):** Sử dụng generalized eigenvalue problem $\Sigma_{\text{grad}} v = \lambda \Sigma_x v$ để tìm task-informative subspace. Không ai đã làm: GaLore/LoRA-GA dùng gradient SVD (không account cho activation distribution). GGI tối ưu RATIO gradient/activation → chọn ĐÚNG task-specific directions thay vì high-gradient directions.

**N2 — Soft Grassmannian Regularization (SGR):** Thay hard orthogonal projection bằng differentiable penalty trên Grassmannian. Novel cho CL-LoRA: existing CL methods (GPM, InfLoRA, O-LoRA) đều dùng hard constraint. SGR cho phép controlled overlap and provides convergence guarantees.

**N3 — Routing-Training Duality (Theorem 2.1):** Chứng minh routing metric (Mahalanobis) và training preconditioner (natural gradient) chia sẻ cùng Fisher metric tensor. Liên kết Contribution 1 và 2 qua information geometry — chưa có work nào kết nối hai hướng này.

**N4 — Balanced Natural Gradient (BNG) with KF-Fisher insight:** Kết hợp activation preconditioning ($\Sigma_x^{-1/2}$) với adaptive balanced LR ($\beta = \sqrt{\|B\|_F/\|A\|_F}$). Key insight: preconditioning A by $\Sigma_x^{-1/2}$ reduces KF-Fisher to output-only inversion (size $r \times r$) — lý giải TẠI SAO activation whitening giúp cả routing (C1) lẫn training (C2). Individually: LoRA+ có asymmetric LR, K-FAC có activation preconditioning, GLA có KF-Fisher analysis. GALA combines + provides unified justification.

**N5 — TARA-GGI synergy:** TARA xác định **bao nhiêu** rank cần, GGI xác định **ĐÂU** — hai thông tin từ cùng gradient probing. Combination tối ưu cả rank allocation VÀ direction selection simultaneously. Chưa có work nào unify rank selection + informed initialization trong CL setting.

## 7.2 What is NOT Novel

| Component | Prior work | What we adopt |
|-----------|-----------|---------------|
| LoRA product manifold geometry | Helmke & Moore 1994, Absil 2008 | General theory |
| Balance regularization | BAR (ICLR 2025) | $(\|B\|_F^2 - \|A\|_F^2)^2$ idea |
| Asymmetric LR | LoRA+ (Hayou 2024) | Concept of different LR for A, B |
| Gradient-based init | LoRA-GA (Wang 2024), GaLore (Zhao 2024) | Gradient probing concept |
| Adaptive rank allocation | AdaLoRA (Zhang 2023) | Concept of per-layer/task rank |
| Grassmannian distance | Edelman et al. 1998 | Chordal distance formula |
| Natural gradient | Amari 1998, K-FAC (Martens 2015) | Fisher preconditioning concept |
| KF-Fisher decomposition | K-FAC, GLA insights | Kronecker structure for LoRA Fisher |
| Hard orthogonal projection | InfLoRA (Liang 2024), GPM (Saha 2021) | CL via subspace protection |
| Effective rank via PaR | Roy & Bhatt 2007, C1 (SRT) | Participation ratio definition |

**Framing**: "We combine classical tools from Riemannian geometry, information theory, and optimization theory into a coherent framework specifically designed for CL-LoRA training. The individual tools are known; their combination, motivation from embedding geometry analysis, and the routing-training duality are novel."

## 7.3 Limitations

1. **Gradient probing cost**: K forward-backward passes thêm ~2% overhead. Acceptable nhưng non-zero.
2. **$\Sigma_x$ estimation**: Cần stored/computed covariance **per layer** (not just global). Reuse từ SRT (Contribution 1) nếu available cho final layer; cần compute riêng cho intermediate layers.
3. **Generalized EVP**: Giả định gradient và activation covariances well-conditioned. Khi $\Sigma_x$ singular (possible for large d): cần regularization (add $\epsilon I$).
4. **Gaussian assumption**: Natural gradient derivation assumes Gaussian activations. For non-Gaussian (multimodal LLaMA tasks): approximation — quality degrades.
5. **Single-task focus**: Contribution 2 chỉ cải thiện training per task. CL performance phụ thuộc interaction giữa training quality (C2) và routing quality (C1).
6. **Layer-wise vs global whitening**: §1.6 chỉ ra $\Sigma_x^{(l)} \neq \Sigma_{\text{pool}}$ ở intermediate layers. BNG's preconditioning quality phụ thuộc vào accuracy of per-layer $\Sigma_x$ estimate — cần enough probing samples.
7. **TARA estimates need empirical validation**: Gradient PaR as proxy for optimal rank is theoretically motivated nhưng chưa verified. Gap giữa gradient PaR và true optimal rank có thể non-trivial cho specific architectures.
8. **Grassmannian manifold structure**: SGR gradient (Proposition 3.1) involves pseudoinverse $A^\dagger$ — numerically unstable khi $\sigma_{\min}(A)$ nhỏ. Cần regularization or clamping.

## 7.4 Lessons from Cross-Critique with GLA (con2_cl.md)

> Phần này ghi lại những gì learned từ so sánh GALA với independent GLA proposal.

### What GLA got right (adopted into GALA v2):

1. **TARA concept**: Adaptive rank dựa trên task complexity. Đơn giản, elegant, zero extra cost. Adopted nhưng changed metric: GLA dùng whitened embedding PaR → overestimates (TGC=310 for T5). GALA dùng **gradient PaR** → trực tiếp đo rank cần thiết.

2. **Fisher simplification in whitened space**: $F_A^{\text{whitened}} \approx (B^\top F_h B) \otimes I$ — beautiful insight. Adopted vào §1.6 nhưng with correction: whitening routing embedding ≠ whitening LoRA input (B4 fix).

3. **KF-Fisher structure for LoRA**: Kronecker factored Fisher cho LoRA parameters — standard nhưng GLA's formulation clean hơn approach từ StelLA/K-FAC.

### What GLA got wrong (rejected/corrected):

1. **Stiefel constraint eliminates scaling** (B2): $B^\top B = I, AA^\top = I$ → $\Delta W = UV^\top$ chỉ có rotation, không scale. LoRA cần scale → Stiefel quá restrictive. GALA dùng Grassmannian (subspace only) + free scales.

2. **Init self-contradiction** (B3): `A_init = V_t.T` → `nn.init.zeros_(A_init)` overwrites geometric init.

3. **Activation ≠ embedding confusion** (B4): $\mathbb{E}[\tilde{h}\tilde{h}^\top] = I$ assumed directly applicable → nhưng $\tilde{h}$ là routing embedding, $x$ là LoRA input. GALA §1.6 addresses with per-layer analysis.

4. **Linear convergence claim** (B8): On non-convex landscape — too strong. GALA uses honest $O(1/\sqrt{T})$ to stationary.

5. **No CL regularizer during training** (B9): G3 only orthogonalizes at init. Drift during training → forgetting. GALA has SGR continuously enforcing orthogonality.

6. **TARA numbers** (B6): TGC=310 (T5), TGC=864 (LLaMA) — whitened full-space PaR, not task-specific gradient PaR. Overestimates needed rank by 10-50x. GALA uses gradient PaR → realistic estimates.

### Architecture difference:
- **GLA**: Stiefel manifold + full Riemannian retraction + whitened-space Fisher simplification. Mathematically elegant but impractical (retraction cost, no scaling).
- **GALA**: Grassmannian (subspace) + activation preconditioning + soft penalty. Less "pure" Riemannian but more implementable and addresses actual problems (P1-P7).

---

# PHẦN VIII: Interaction với Contribution 1 (SRT)

---

## 8.1 Synergy (Không Conflict)

| Aspect | Contribution 1 (SRT) | Contribution 2 (GALA) | Interaction |
|--------|----------------------|----------------------|-------------|
| **Timing** | Inference | Training | Complementary |
| **Core structure** | $\Sigma_x^{-1}$ preconditioning (routing) | $\Sigma_x^{-1}$ preconditioning (gradient) | **Same Fisher metric** |
| **Statistics reuse** | Stores $\mu_t, \Sigma_t$ | Uses $\Sigma_x$ from probing | $\Sigma_x \approx \Sigma_{\text{pool}}$ → reuse (final layer) |
| **Subspace info** | Uses $V_t$ for capacity analysis | Stores $V_t$ for CL protection | Same subspaces |
| **Rank info** | PaR determines routing ceiling | TARA uses gradient PaR for rank | PaR bridges routing capacity → training rank |
| **Quality feedback** | Better routing → correct adapter selection | Better adapters → more distinguishable embeddings → better routing | Positive feedback loop |

## 8.2 Shared Mathematical Foundation

Cả hai contributions xuất phát từ cùng **Fisher information geometry**:

$$g_{ij}(\theta) = \mathbb{E}_\theta\!\left[\frac{\partial \log p(x|\theta)}{\partial \theta_i} \frac{\partial \log p(x|\theta)}{\partial \theta_j}\right]$$

- SRT dùng $g_{ij}$ làm **distance** (Mahalanobis/Fisher-Rao) để route
- GALA dùng $g_{ij}$ làm **preconditioner** (natural gradient) để train

> **Unifying principle**: Fisher metric tensor is the natural structure cho both routing and training in CL with frozen backbone.

## 8.3 Practical Integration

Khi deploy cả hai:
1. Train task $t$ bằng GALA → produces adapter $(A_t, B_t)$ và subspace $V_t$
2. Compute task signature $(\mu_t, \hat{\Sigma}_t)$ trên embeddings AFTER adapter applied
3. SRT routing dùng $(\mu_t, \hat{\Sigma}_t)$ để route test samples
4. Next task: GALA dùng $V_t$ cho Grassmannian penalty và $\hat{\Sigma}_x$ cho natural gradient

$\hat{\Sigma}_x$ from GALA's probing ≈ $\hat{\Sigma}_{\text{pool}}$ from SRT → **zero additional storage cost**.

---

# PHẦN IX: Open Questions

---

1. ~~**Adaptive rank selection**: PaR analysis (Contribution 1) cho thấy effective dimensionality ≈ 9–27. Có nên set r adaptively per task dựa trên PaR?~~ **RESOLVED**: TARA (§3.0) giải quyết bằng gradient PaR-based rank allocation.

2. **Second-order Riemannian optimization**: GGI + SGR là first-order (projected gradient). Full Riemannian Newton method trên $\mathcal{M}_r$ có thể converge nhanh hơn nhưng đắt hơn. Tradeoff?

3. **Multi-task gradient alignment**: GGI chỉ dùng gradient từ task hiện tại. Có thể dùng stored gradient statistics từ tasks trước để inform initialization?

4. **Beyond Grassmannian**: Row space of A chỉ capture input subspace. Full LoRA geometry lives on $Gr(r,m) \times \mathbb{R}^r_{>0} \times Gr(r,n)$ (output × scales × input). Có benefit optimize trên full product?

5. **Non-Gaussian preconditioning**: $\Sigma_x^{-1/2}$ optimal cho Gaussian. Cho multimodal activations (LLaMA), kernel-based hoặc normalizing flow preconditioning có thể tốt hơn?

6. **Dynamic $\lambda_1$ scheduling**: $\lambda_1$ (Grassmannian penalty) nên tăng khi thêm tasks (subspace gets crowded) hay giảm (later tasks less important)? Có thể derive từ capacity analysis (Contribution 1, Theorem 6a)?

7. **TARA interaction with CL budget**: Khi task số lượng lớn ($T \gg d_{in}/r$), subspace budget cạn kiệt. TARA nên giảm rank của later tasks (conserve budget) hay giữ (maintain quality)? Tradeoff cần information-theoretic analysis.

8. **Layer-wise vs uniform TARA**: Hiện tại TARA dùng cùng rank cho tất cả layers. Gradient PaR có thể khác nhau giữa layers (attention vs MLP, shallow vs deep). Per-layer TARA có benefit nhưng thêm complexity.

---

# PHẦN X: References

---

**Riemannian Optimization on Matrix Manifolds:**
- Absil, P.-A., Mahony, R., & Sepulchre, R. (2008). *Optimization Algorithms on Matrix Manifolds*. Princeton University Press.
- Edelman, A., Arias, T., & Smith, S. T. (1998). The Geometry of Algorithms with Orthogonality Constraints. *SIAM J. Matrix Anal. Appl.*
- Vandereycken, B. (2013). Low-Rank Matrix Completion by Riemannian Optimization. *SIAM J. Optim.*
- Helmke, U., & Moore, J. B. (1994). *Optimization and Dynamical Systems*. Springer.
- Sato, H., & Iwai, T. (2013). A Riemannian Optimization Approach to the Matrix SVD. *SIAM J. Optim.*

**LoRA and Variants:**
- Hu, E. et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022.*
- Hayou, S. et al. (2024). LoRA+: Efficient Low Rank Adaptation of Large Models. *ICML 2024.*
- Chen, Z. et al. (2025). GainLoRA: Low-Rank Adaptation with Gating for Continual Learning. *NeurIPS 2025.*
- Liang, Y. & Li, Z. (2024). InfLoRA: Interference-Free Low-Rank Adaptation. *CVPR 2024.*
- Wang, K. et al. (2024). O-LoRA: Orthogonal Low-Rank Adaptation. *ICML 2024.*
- Liu, S. et al. (2024). DoRA: Weight-Decomposed Low-Rank Adaptation. *ICML 2024.* arXiv:2402.09353.
- Zhang, Q. et al. (2023). AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning. *ICLR 2023.*

**Geometric LoRA Optimization:**
- StelLA — Subspace Learning in LoRA using Stiefel Manifold. *NeurIPS 2025.*
- BAR — Chen et al. (2025). Understanding and Improving SAM for Scale-Invariant Problems. (Balancedness-Aware Regularization.) *ICLR 2025.*
- Flat-LoRA — Bayesian expectation loss for LLM LoRA fine-tuning. *ICML 2025.*
- SubTrack — Gradient Subspace Tracking on Grassmann Manifold. *NeurIPS 2024.*
- Wang, S. et al. (2024). LoRA-GA: Low-Rank Adaptation with Gradient Approximation. *NeurIPS 2024.*
- Zhao, J. et al. (2024). GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection. *ICML 2024.*
- Jordan, M. et al. (2025). Muon: Layerwise Natural Gradient Descent. arXiv:2502.04043.

**Natural Gradient and Fisher Information:**
- Amari, S. (1998). Natural Gradient Works Efficiently in Learning. *Neural Computation.*
- Amari, S. & Nagaoka, H. (2000). *Methods of Information Geometry*. AMS.
- Martens, J. & Grosse, R. (2015). Optimizing Neural Networks with Kronecker-Factored Approximate Curvature. *ICML 2015.* (K-FAC)
- George, T. et al. (2018). Fast Approximate Natural Gradient Descent in a Kronecker-Factored Eigenbasis. *NeurIPS 2018.*

**Grassmannian Geometry:**
- Conway, J., Hardin, R., & Sloane, N. (1996). Packing Lines, Planes, etc. *Experimental Math.*
- Absil, P.-A. & Malick, J. (2012). Projection-like Retractions on Matrix Manifolds. *SIAM J. Optim.*
- Hamm, J. & Lee, D. (2008). Grassmann Discriminant Analysis. *ICML 2008.*

**Continual Learning:**
- Saha, G. et al. (2021). Gradient Projection Memory for Continual Learning. *NeurIPS 2021.* (GPM)
- Kirkpatrick, J. et al. (2017). Overcoming catastrophic forgetting. *PNAS.*
- Mallya, A. & Lazebnik, S. (2018). PackNet: Adding Multiple Tasks to a Single Network. *CVPR 2018.*

**Information Theory:**
- Tishby, N. & Zaslavsky, N. (2015). Deep Learning and the Information Bottleneck Principle. *ITW 2015.*
- Cover, T. & Thomas, J. (2005). *Elements of Information Theory*. Wiley.

**Covariance Estimation:**
- Ledoit, O. & Wolf, M. (2004). A Well-conditioned Estimator for Large-dimensional Covariance Matrices. *J. Multivariate Analysis.*

**Pattern Recognition:**
- Fukunaga, K. (1990). *Introduction to Statistical Pattern Recognition*. Academic Press.
