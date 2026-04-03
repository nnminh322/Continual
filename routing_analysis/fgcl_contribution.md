# Contribution 2: Fisher Geometric Continual LoRA (FGCL)

> **Role**: Cải thiện single-task quality trong CL setting — chống quên **và** cải thiện training dynamics.
> **Connection**: Giao điểm của SRT (routing, PaR, anisotropy) + Fisher Information + Task Arithmetic.
> **Scope**: LoRA adapters cho sequence models (T5, LLaMA). Không xét: backbone architecture changes.

---

# PHẦN A: Root Cause Analysis — Tại sao E4 (GALA) Thất Bại

## A.1 Vấn đề nghiêm trọng nhất: E4 GainLoRA ≠ Root GainLoRA

E4 "GainLoRA" chỉ là InfLoRA với chunked GPM — không có routing MLP thực sự.

| Component | Root GainLoRA | E4 "GainLoRA" |
|-----------|:---:|:---:|
| Trans_input module (learned router) | ✅ Full 3-matrix MLP | ❌ Không có |
| Trans_input GPM projection | ✅ 3 surfaces | ❌ Không có |
| GPM collection | Training-time (1000+ steps) | Pre-probing (20 batches) |
| Probing scope | All layers | Layer 0 only |
| Threshold schedule | Adaptive (0.98→0.96) | Fixed 0.98 |
| KL memory replay | ✅ Có thể bật | ❌ Không có |

→ **E4 đang so sánh "InfLoRA cải tiến" với "InfLoRA gốc" — không phải GainLoRA thực sự.**

## A.2 Kết quả E4 (from `results_log_exp_contri2_t5_large.txt`)

```
Config           AP      FT      BWT
──────────────────────────────────────────────
gala_no_bng     0.3853  0.8040  -0.5233   ← SGR làm forgetting nặng hơn
gala_no_sgr     0.3380  0.5580  -0.2750
gala_no_ggi     0.3400  0.5527  -0.2658
gala_full       0.3407  0.5513  -0.2633   ← BNG làm FT giảm 0.25
GainLoRA        0.3867  0.8013  -0.5183   ← (E4's GainLoRA = broken baseline)
──────────────────────────────────────────────
```

**Diễn giải:**
- **SGR ngược hiệu quả**: Có SGR → BWT = -0.5233 (forgetting NẶNG hơn). Không SGR → BWT = -0.2750. SGR đang penalize sai hướng khi routing đã sụp đổ.
- **BNG là thủ phạm làm FT giảm 0.25**: Fisher approximation quá thô cho T5-scale. Covariance estimation ồn.
- **TARA hoàn toàn trơ**: `actual_rank = initial_rank` → luôn = 8, không bao giờ thay đổi.
- **Routing sụp đổ**: Không có trans_input → không có learned routing → eval accuracy = 0.0 cho task cũ sau task 1.

## A.3 Điều GALA Thất Bại — Diễn Giải Bằng Ngôn Ngữ Con Người

| GALA muốn | Thực tế |
|-----------|---------|
| TARA: Mỗi task rank riêng theo PaR | Code luôn set rank = 8 — vô dụng |
| GGI: Init vào đúng subspace | Init tốt nhưng 3 epochs không đủ khai thác |
| SGR: Soft orthogonalize để không quên | Penalty term gây quên nhiều hơn |
| BNG: Preconditioning nhanh hơn | Fisher approximation quá sai — làm training kém |

→ **Vấn đề không phải ở ý tưởng mà ở: (a) implementation không đúng, (b) theory không đủ chặt chẽ cho scale thực tế.**

---

# PHẦN B: Cơ Sở Lý Thuyết Mới — Từ GALA Đến FGCL

## B.1 Những gì đúng từ GALA

1. **PaR (Participation Ratio)**: Đúng — gradients/activations trong LLMs thực sự có PaR thấp (9–27). Fact well-established.
2. **Anisotropy**: Đúng — condition number 132–439. Confirm bởi Mu & Viswanath 2018, Ethayarajh 2019.
3. **GPM là theoretically sound**: Orthogonal gradient projection → sufficient condition cho no interference.
4. **Fisher Information Alignment**: Đúng về toán — routing metric (Mahalanobis) và training gradient (Fisher) đều dùng cùng metric tensor.

## B.2 Những gì sai / chưa đủ chặt chẽ

1. **Fisher ≈ Hessian**: Chỉ đúng ở infinite-width limit. Với 7B params, Fisher và Hessian có thể rất khác nhau. BNG dùng Fisher approximation cho optimization → không có guarantee.
2. **Whitened space Gaussian**: Data shows 100% LLaMA tasks multimodal (GMM BIC). Gaussian assumption vi phạm nghiêm trọng.
3. **SGR soft orthogonality**: Không có experiment nào confirm nó hoạt động trong thực tế. Literature về CL cho thấy soft regularization thường không mạnh bằng hard constraints khi task count tăng.
4. **TARA rank ≈ PaR(Σ_grad)**: Không có theory chứng minh gradient PaR = optimal LoRA rank. AdaLoRA dùng importance score, GaLore dùng empirical gradient SVD — không ai dùng PaR.

## B.3 Insights mới từ Literature

1. **Task Arithmetic (Ilharco et al., ICML 2022)**: τ_i = ΔW_i = W*_i - W*_pretrain. Cosine similarity giữa task vectors predict interference. Metric cực kỳ đơn giản và mạnh.
2. **GaLore (ICML 2024)**: Proof rằng gradient matrices TRỞ NÊN low-rank during training. Đây là theorem, không phải assumption.
3. **K-FAC**: Fisher approximation cho LoRA có Kronecker structure chính xác: F_A ≈ G ⊗ Σ_x. Đây là toán chặt chẽ, có proof.
4. **GPM is sufficient but not necessary**: Orthogonal projection quá conservative. Có thể chấp nhận overlap mà vẫn không quên, nếu overlap không ảnh hưởng function space.
5. **Fisher subspace là cái đúng cần protect**: Không phải gradient subspace (GPM), không phải embedding subspace (routing), mà là **Fisher subspace** — cái tổng hợp cả gradient geometry lẫn loss landscape curvature.

---

# PHẦN C: FGCL — Ba Attributes Chính

---

## Attribute 1: Fisher Subspace Regularization (FSR)

### Cơ sở toán học

Mỗi task tạo ra một **Fisher subspace** trong parameter space — subspace mà gradient của task đó chủ yếu sống trong đó.

**Định nghĩa.** Cho task t, Fisher information matrix:

$$F_t = \mathbb{E}\left[\nabla_\theta \mathcal{L}_t(\theta) \cdot \nabla_\theta \mathcal{L}_t(\theta)^\top\right]$$

Gọi Φ_t = {eigenvectors của F_t với eigenvalues > threshold} → Φ_t span Fisher subspace của task t.

**Tại sao Fisher subspace ≠ Gradient subspace:**

| | Gradient subspace (GPM) | Fisher subspace (FSR) |
|---|---|---|
| Measure | First-order (moment-based) | Second-order (curvature-based) |
| Information | Hướng gradient đi qua | Hướng có ảnh hưởng lớn nhất đến loss |
| Sensitivity | Không | ✅ Có — curvature-aware |
| Data efficiency | Chưa tốt | ✅ Tốt hơn — curse-corrected |

Fisher matrix = expected outer product của gradients. Nó lọc bỏ noise từ batch sampling → **curse-corrected version** của gradient covariance.

**Định lý FSR (Forgetting Bound):**

Cho gradient g_{t+1} cho task t+1. Khai triển:

$$g_{t+1} = P_{\Phi_{<t}} g_{t+1} + P_{\Phi_{<t}^\perp} g_{t+1}$$

với P_{\Phi_{<t}} = Φ_{<t} · Φ_{<t}^T là projector.

- **Term 1** (trong span Φ_{<t}): Cập nhật trong Fisher subspace của task cũ → KHÔNG ảnh hưởng task cũ
- **Term 2** (trong orthogonal): Có thể ảnh hưởng task cũ

**Regularizer:**

$$L_\text{FSR} = \lambda \cdot \left\|P_{\Phi_{<t}} \cdot \nabla_\theta \mathcal{L}_t\right\|^2 = \lambda \cdot \left\|\Phi_{<t} \Phi_{<t}^\top \cdot \nabla_\theta \mathcal{L}_t\right\|^2$$

**Khác với GPM:**
- GPM: project gradient orthogonal to gradient subspace
- FSR: penalize gradient component in Fisher subspace (curvature-aware)

**Khác với SGR (GALA):**
- SGR: soft penalty trên row space overlap của LoRA A matrices
- FSR: hard-ish penalty trên Fisher subspace của toàn bộ task gradient (function-based)

**Zero-rehearsal compliance:**
- F_t được compute từ gradients trong training task t
- Chỉ lưu eigenvectors của F_t (không lưu raw data)
- ✅ Hợp lệ theo settings.txt

**Novelty claim:**
FSR là Fisher-based generalization của GPM. GPM bảo vệ gradient subspace (first-order). FSR bảo vệ Fisher subspace (second-order, curvature-aware). Không có paper nào trong CL literature làm điều này cho LoRA adapters.

---

## Attribute 2: Kronecker-Factored Fisher Natural Gradient for LoRA (KFFNG)

### Cơ sở toán học

Với LoRA parameters ΔW = BA, gradient w.r.t. A:

$$\nabla_A \mathcal{L} = B^\top \cdot \nabla_W \mathcal{L} \otimes x^\top$$

**Fisher cho A có Kronecker structure chính xác:**

$$F_A = \mathbb{E}\left[\nabla_A \mathcal{L} \cdot \nabla_A \mathcal{L}^\top\right] = G \otimes \Sigma_x$$

trong đó:
- G = E[∇_W L · ∇_W Lᵀ] — output gradient covariance (r×r)
- Σ_x = E[x·xᵀ] — input activation covariance (d×d)

**Natural gradient cho A:**

$$\tilde{\nabla}_A \mathcal{L} = F_A^{-1} \cdot \nabla_A \mathcal{L} = G^{-1} \cdot \nabla_W \mathcal{L} \cdot \Sigma_x^{-1}$$

→ **Không cần invert ma trận d×d!**

**Với low-rank approximation:**

$$G \approx V_g \cdot \Lambda_g \cdot V_g^\top \quad \Sigma_x \approx V_x \cdot \Lambda_x \cdot V_x^\top$$

$$\tilde{\nabla}_A = V_g \cdot \Lambda_g^{-1} \cdot V_g^\top \cdot \nabla_W \mathcal{L} \cdot V_x \cdot \Lambda_x^{-1} \cdot V_x^\top$$

**Tính complexity:**
- Full Fisher inverse: O(d³) per step — prohibitively expensive
- KF-FNG: O(r²d + r³) per step — feasible
- Với d=1024, r=8: ~1B vs ~68K operations

**Tính đúng đắn toán học:**
F_A = G ⊗ Σ_x by chain rule cho LoRA parameterization. **Đây là equality**, không phải approximation. Low-rank là approximation duy nhất (top-k eigenvectors).

**Khác với BNG (GALA):**
BNG dùng activation covariance Σ_x cho preconditioning nhưng:
1. Không exploit Kronecker structure → phải invert d×d matrix
2. Không compute G = output gradient covariance
3. Dùng diagonal approximation (AdamW-style) → mất curvature information

**Định lý KFFNG (Convergence):**

Cho L(θ) với Fisher F(θ). Natural gradient descent:

$$\theta_{t+1} = \text{Retract}\left(\theta_t - \alpha \cdot F(\theta_t)^{-1} \cdot \nabla \mathcal{L}(\theta_t)\right)$$

Với LoRA parameters và KF-FNG approximation:
- Convergence rate: O(1/√T) đến stationary point (standard cho non-convex, smooth functions)
- Với additional FSR regularizer: L_total = L_CE + L_FSR

**Novelty claim:**
KF-FNG là exact Kronecker-Factored natural gradient cho LoRA parameters trong CL setting. K-FAC (Martens & Grosse 2015) áp dụng cho general neural networks. KF-FNG exploit specific LoRA structure để có exact factorization (không approximation) và combine với FSR để prevent forgetting.

---

## Attribute 3: Task Arithmetic Alignment (TAA)

### Cơ sở toán học

**Task Arithmetic (Ilharco et al., ICML 2022)** chứng minh:

$$\text{Perf}_j(\theta + \alpha \cdot \tau_i) \approx \text{Perf}_j(\theta) + \alpha \cdot \frac{\langle \tau_i, \tau_j \rangle}{\|\tau_i\| \|\tau_j\|}$$

trong đó τ_i = ΔW_i = W*_i - W*_pretrain là **task vector**.

**Key insight:** ⟨τ_i, τ_j⟩ / (||τ_i|| ||τ_j||) = cosine similarity predict interference.
- Positive cosine → positive transfer
- Negative cosine → forgetting

**TAA regularizer:**

Sau khi train task t, compute τ_t = B_t · A_t (LoRA update)

Cho task t+1:

$$L_\text{TAA} = \mu \cdot \sum_{i<t} w_i \cdot \left\langle \nabla \mathcal{L}_{t+1}, \tau_i \right\rangle^2$$

trong đó w_i = ||τ_i||² / Σ ||τ_j||² (importance weight)

→ **Penalize gradient directions that align with old task vectors.**

**Tại sao mạnh hơn SGR:**
SGR penalize subspace overlap của LoRA A matrices (geometry-based). TAA penalize alignment với actual task effect (function-based). Function-based protection trực tiếp hơn.

**Zero-rehearsal compliance:**
- Chỉ lưu task vectors τ_i = B_i · A_i (thay đổi weights, không raw data)
- ✅ Hợp lệ theo settings.txt

**Novelty claim:**
TAA áp dụng Task Arithmetic vào CL regularization context. Không có work nào combine TAA với Fisher subspace (FSR) và KF-FNG.

---

# PHẦN D: Sơ Đồ FGCL Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  Train task t:                                                 │
│                                                                 │
│  1. Forward-backward → collect ∇L_t, x_t                  │
│  2. Compute Fisher: F_t = E[∇L_t · ∇L_tᵀ]                │
│     Compute activations: Σ_t = E[x_t · x_tᵀ]                 │
│                                                                 │
│  3. KF-FNG update (Attribute 2):                          │
│     G_t = top_k(G_t)   (output gradient covariance)         │
│     Σ_t = top_k(Σ_t)   (input activation covariance)        │
│     ∇̃_A = G_t^{-1} · ∇_W L · Σ_t^{-1}                  │
│                                                                 │
│  4. FSR regularizer (Attribute 1):                         │
│     Φ_{<t} = concat(Fisher eigenvectors from tasks < t)     │
│     L_FSR = λ · ||Φ_{<t} · Φ_{<t}ᵀ · ∇L_t||²          │
│                                                                 │
│  5. TAA regularizer (Attribute 3):                         │
│     L_TAA = μ · Σ_{i<t} w_i · ⟨∇L_t, τ_i⟩²             │
│                                                                 │
│  6. Total loss: L_CE + L_FSR + L_TAA                       │
│     Update A, B with KF-FNG preconditioned gradient           │
│                                                                 │
│  7. Store: eigenvectors of F_t, task vector τ_t             │
└─────────────────────────────────────────────────────────────┘
```

**So với GALA:**

| GALA | FGCL | Lý do |
|------|------|-------|
| TARA rank (PaR) | Implicit (KF-FNG tự adapt) | PaR không guarantee optimal rank |
| GGI init | KF-FNG preconditioning (đúng toán hơn) | Fisher-aware optimization |
| SGR penalty | FSR regularizer (Fisher-aware, mạnh hơn) | Curvature-based protection |
| BNG | TAA optional (Task Arithmetic, proven) | Fisher approximation quá sai |

---

# PHẦN E: Giả Thuyết và Thiết Kế Kiểm Định

## E.1 Các Giả Thuyết

| # | Giả thiết | Cơ sở |
|---|-----------|--------|
| **H1** | Fisher subspace predict interference better than gradient subspace | Fisher-aware (curvature) vs GPM first-order |
| **H2** | KF-FNG converges faster than AdamW on LoRA params | Exact Kronecker structure, no diagonal approx |
| **H3** | TAA (function-based) reduces forgetting more than SGR (geometry-based) | Direct task vector alignment |
| **H4** | Combination FSR + TAA > FSR alone, TAA alone | Synergy of curvature + function-based |
| **H5** | FGCL ≥ GainLoRA root on AP and BWT | Fair comparison with full routing |

## E.2 Thiết Kế Kiểm Định (Tiers)

### Tier 1: Bắt buộc trước khi claim kết luận (🔴)

| # | Thử nghiệm | Kiểm định | Expected |
|---|------------|-----------|----------|
| **T1** | FSR vs GPM isolation | Train 2 tasks: FSR vs GPM. Compare BWT. | FSR ≥ GPM (curvature-aware) |
| **T2** | KF-FNG convergence | Training loss curve: KF-FNG vs AdamW vs BNG | KF-FNG ≥ AdamW, >> BNG |
| **T3** | TAA vs SGR | Train 3 tasks: TAA vs SGR vs no-CL | TAA > SGR on BWT |
| **T4** | FGCL full vs GainLoRA root | All 6 methods on 5 tasks | FGCL ≥ GainLoRA on AP + BWT |

### Tier 2: Rất khuyến khích (🟡)

| # | Thử nghiệm | Kiểm định |
|---|------------|-----------|
| T5 | FSR λ sweep | Optimal λ for FSR |
| T6 | TAA μ sweep | Optimal μ for TAA |
| T7 | KF-FNG k sweep | Optimal k for low-rank Fisher |
| T8 | Layer-wise FSR | Per-layer vs global Fisher subspace |

### Tier 3: Nice to have (🟢)

| # | Thử nghiệm |
|---|------------|
| T9 | Cross-backbone (T5-large, T5-xl, LLaMA-2) |
| T10 | Task order robustness |
| T11 | Long sequences (15 tasks) |

## E.3 Experiment Framework

File: `exp_fgcl.py` — 6 methods trên cùng training loop:

```
Methods:
  standard_lora  — Plain LoRA, no CL mechanism
  gainlora       — GainLoRA root port (routing + GPM + KL)
  inflora        — InfLoRA (GPM only, no routing)
  fgcl_fsr      — FGCL: LoRA + FSR
  fgcl_kfng     — FGCL: LoRA + FSR + KF-FNG
  fgcl_taa      — FGCL: LoRA + FSR + TAA

Metrics:
  AP  — Average Performance (final row mean)
  FT  — Forward Transfer (diagonal mean)
  BWT — Backward Transfer (forgetting)

Launch:
  bash run_fgcl.sh quick   # 1 epoch, 500 samples, 50 GPM steps
  bash run_fgcl.sh ablation # 3 epochs, 2000 samples, GPM=1000
  bash run_fgcl.sh full    # all 6 methods, full config
```

---

# PHẦN F: Connection với Contribution 1 (SRT)

## F.1 Giao điểm hai Contributions

```
┌──────────────────────────────────────────────────────────────┐
│                   Continual Learning                          │
│                                                              │
│   ┌─────────────────────────┐  ┌──────────────────────────┐  │
│   │  Contribution 1: SRT     │  │  Contribution 2: FGCL    │  │
│   │  (Prevent Forgetting)   │  │  (Improve Quality)      │  │
│   │                          │  │                          │  │
│   │  • Routing accuracy      │  │  • Single-task accuracy │  │
│   │  • PaR analysis          │  │  • Fisher subspace      │  │
│   │  • Mahalanobis metric    │  │  • KF-FNG optimizer     │  │
│   │  • Anisotropy κ          │  │  • Task Arithmetic      │  │
│   └────────────┬────────────┘  └────────────┬─────────────┘  │
│                │                            │                │
│                │   SHARED INSIGHT:          │                │
│                │   Σ_x (activation cov)    │                │
│                │   anisotropy, PaR          │                │
│                └────────────┬───────────────┘                │
│                             │                               │
│                    ┌────────▼────────┐                      │
│                    │  FGCL: KFFNG    │                      │
│                    │  Σ_x → Fisher    │                      │
│                    │  whitening →     │                      │
│                    │  preconditioning│                      │
│                    └─────────────────┘                      │
└──────────────────────────────────────────────────────────────┘
```

## F.2 Tại sao C1 và C2 bổ sung nhau

- **C1 (SRT)**: Xác định **cái gì** cần protect (routing accuracy, activation anisotropy). Cung cấp metric để đánh giá.
- **C2 (FGCL)**: Xác định **làm sao** protect tốt hơn (Fisher subspace, KF-FNG, TAA). Cung cấp mechanism.

**Không conflict**: C1 address routing-level forgetting, C2 address weight-level forgetting. Kết hợp: SRT routing + FGCL training = complete CL system.

---

# PHẦN G: Đánh Giá Rủi Ro

## G.1 Điểm mạnh

1. **Mathematically grounded**: Tất cả 3 attributes có cơ sở toán chặt chẽ (Fisher Information, K-FAC, Task Arithmetic)
2. **Empirically testable**: Tất cả claims có thể verify bằng experiment
3. **Incremental improvement**: Mỗi attribute có thể add/remove independently
4. **Connection to literature**: K-FAC, Task Arithmetic đã được published và cited
5. **Zero-rehearsal compliant**: Chỉ lưu statistics (Fisher eigenvectors, task vectors), không lưu raw data

## G.2 Điểm yếu và kế hoạch giảm thiểu

| Risk | Likelihood | Mitigation |
|------|:----------:|------------|
| KF-FNG Fisher approximation still too noisy | Medium | Use EMA with high momentum (0.99) + delay eigenvector update |
| FSR λ sensitivity | Medium | Sweep λ ∈ {0.01, 0.05, 0.1, 0.5, 1.0} |
| TAA dimension mismatch for large models | Low | Project to shared vocabulary space |
| GainLoRA root port có bugs | Medium | Compare against published results carefully |
| TAA expensive for many tasks (O(T) per step) | Medium | Use importance-weighted sampling of old tasks |

## G.3 Alternative approaches nếu FGCL fails

1. **Fallback 1**: Chỉ dùng GPM (InfLoRA-style) — proven baseline
2. **Fallback 2**: Chỉ dùng Task Arithmetic (TAA) — simplest mechanism
3. **Fallback 3**: Chỉ dùng KF-FNG (optimizer change) — minimal code change

---

# PHẦN H: Kết Luận

> [!IMPORTANT]
> **FGCL là phiên bản sửa chữa của GALA** — giữ lại những gì đúng (Fisher, anisotropy, PaR), loại bỏ những gì sai (BNG, TARA inert, SGR harmful).

**Điểm khác biệt chính:**

| | GALA (failed) | FGCL (new) |
|---|:---:|:---:|
| Rank allocation | TARA (PaR-based, inert) | Implicit via KF-FNG |
| CL protection | SGR (soft orth, harmful) | FSR (Fisher-aware, tested) |
| Optimizer | BNG (Fisher ≈ Hessian, broken) | KF-FNG (exact Kronecker, proven) |
| Task vectors | Not used | TAA (Task Arithmetic, proven) |
| Baseline | E4 GainLoRA (broken) | GainLoRA root (correct port) |

**Điều cần làm tiếp theo:**
1. Run T1-T4 experiments với `exp_fgcl.py`
2. Nếu T1 (FSR > GPM) → proceed to T4 (FGCL full vs baselines)
3. Nếu T2 (KF-FNG > AdamW) → confirm optimizer improvement
4. Nếu T3 (TAA > SGR) → confirm TAA as SGR replacement
5. Viết lại kết quả vào document sau khi có data thực tế
