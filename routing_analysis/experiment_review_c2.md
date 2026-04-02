# Review: GALA (Contribution 2) — Thử nghiệm Tiền đề

> Đánh giá: `validate_gala.py` + `run_gala_validation.sh` + Phần VI (copilot_con_2.md)
> Mục tiêu: Các thử nghiệm hiện tại có đủ bao phủ các giả thiết của GALA không? Chạy hết có đưa ra kết luận chắc chắn không?

---

## 1. Ánh xạ Giả thiết ↔ Thử nghiệm

### Bảng tổng hợp

| # | Giả thiết (Hypothesis) | Thử nghiệm trong `validate_gala.py` | Phủ? | Đủ mạnh? |
|---|----------------------|--------------------------------------|------|----------|
| **H1** | Tasks có geometric complexity khác nhau → fixed rank suboptimal | Phase 1: PaR_act, PaR_res, rank sweep variance capture | ✅ | ⚠️ |
| **H2** | GGI (Gen. EVP) > PCA > Random cho init subspace | Phase 2: Fisher ratio, specificity, 50 random trials | ✅ | ⚠️ |
| **H3** | Hard projection lossy; Soft (SGR) tốt hơn | Phase 3: β sweep, info_retained, CL sequence simulation | ✅ | ⚠️ |
| **H4** | Activations anisotropic → preconditioning helps | Phase 4: κ raw vs κ preconditioned, amplification | ✅ | ✅ |
| **H5** | Routing-Training Duality (Σ_x shared) | Phase 4: cov relative diff task vs pool | ✅ | ⚠️ |
| **H6** | PaR_activation ≠ PaR_gradient (different measures) | Phase 1: so sánh PaR_act vs PaR_residual | ✅ | ✅ |
| **H7** | Same-domain tasks có high subspace overlap | Phase 3: cluster overlaps (sentiment, NLI, topic) | ✅ | ✅ |

### Giả thiết CHƯA được kiểm tra

| # | Giả thiết thiếu | Tại sao quan trọng | Mức độ thiếu |
|---|----------------|---------------------|-------------|
| **H8** | TARA rank ≈ oracle rank | Cần xác nhận r_TARA matching r_oracle (accuracy saturation) — **nhưng không có actual LoRA training trong validate_gala.py** | 🔴 Critical |
| **H9** | SGR convergence tốt hơn hard projection (training dynamics) | Phase 3 đo info_retained (static), không đo training loss curve hay final accuracy | 🔴 Critical |
| **H10** | BNG (preconditioned optimizer) > AdamW (training speed/quality) | Phase 4 chỉ đo κ reduction (lý thuyết), không có actual gradient descent comparison | 🔴 Critical |
| **H11** | Balance regularization cải thiện optimization | Không được test trong validate_gala.py | 🟡 Medium |
| **H12** | B ≠ 0 init cải thiện early training dynamics | Chưa test (cần actual training curves) | 🟡 Medium |
| **H13** | TARA + GGI synergy > TARA alone, GGI alone | E7 trong plan nhưng chưa implement | 🟡 Medium |

---

## 2. Phân tích chi tiết từng Phase

### Phase 1 (TARA) — ⚠️ Cần bổ sung

**Có gì:**
- ✅ PaR per task (activation-based và residual-based)
- ✅ Rank sweep → fraction of variance captured
- ✅ TARA recommended rank (90%, 95%, 99% thresholds)
- ✅ Cross-task comparison (range, mean)

**Thiếu gì:**

> [!CAUTION]
> **Lỗ hổng lớn nhất**: "Gradient PaR" proxy dùng `Σ_task - Σ_pool` (task-residual covariance) thay cho gradient covariance thực sự `Σ_grad = E[∇ℓ · ∇ℓ^T]`. Đây là **approximation chưa được validate**.

1. **Proxy validity**: `Σ_task - Σ_pool` không phải gradient covariance. Gradient phụ thuộc vào **labels + loss function**, không chỉ activation distribution. Cần:
   - Test trên actual gradient covariance (chạy K forward-backward trên actual model)
   - Hoặc chứng minh correlation giữa PaR(Σ_task - Σ_pool) và PaR(Σ_grad)

2. **No accuracy validation**: TARA nói "r_90 đủ" — nhưng chưa verify bằng actual LoRA training ở different ranks. Experiment plan E0 đề cập nhưng **chưa implement**.

3. **Task ordering effect**: PaR_residual phụ thuộc vào Σ_pool. Nếu pool thay đổi theo task order → TARA recommendation thay đổi. Chưa test.

### Phase 2 (GGI) — ⚠️ Cần bổ sung

**Có gì:**
- ✅ Fisher Discriminant Ratio cho GGI vs PCA vs Random
- ✅ Task-specificity metric
- ✅ Effective rank trong projected subspace
- ✅ Overlap GGI ↔ PCA
- ✅ 50 random trials cho statistical significance

**Thiếu gì:**

> [!WARNING]
> **GGI proxy dùng between-class scatter (Σ_B) thay cho gradient covariance.** Đây là Fisher LDA criterion, KHÔNG phải generalized EVP của GALA (Σ_grad v = λ Σ_x v).

1. **Proxy mismatch**: GGI lý thuyết tối ưu `tr(V^T Σ_grad V) / tr(V^T Σ_x V)`. Experiment dùng `tr(V^T Σ_B V) / tr(V^T Σ_W V)`. Hai cái này **không tương đương**:
   - Σ_B = between-class centroid scatter (supervised, class-level)
   - Σ_grad = gradient outer product (supervised, sample-level, loss-weighted)
   - Σ_B chỉ capture class separation. Σ_grad capture cả magnitude of loss per direction.

2. **Thiếu actual training comparison**: GGI Fisher ratio > PCA Fisher ratio ≠ GGI init → better final accuracy. Cần actual LoRA training với 3 init strategies → compare final accuracy.

3. **CL constraint chưa test**: GGI claim "CL-constrained init" (project to null space). Phase 2 chỉ test GGI quality, KHÔNG test CL constraint effect.

### Phase 3 (SGR) — ⚠️ Cần bổ sung

**Có gì:**
- ✅ Pairwise Grassmannian overlap matrix
- ✅ β sweep (0 → 1): info_retained vs overlap tradeoff
- ✅ CL sequence simulation (15 tasks sequential)
- ✅ SSE (Shared Subspace Exclusion) measurement
- ✅ Effective rank after projection
- ✅ Domain cluster analysis (H7)

**Thiếu gì:**

> [!WARNING]
> **Static analysis only!** Phase 3 đo "nếu project thì mất bao nhiêu thông tin" — nhưng KHÔNG biết mất thông tin đó có ảnh hưởng đến accuracy không.

1. **Information ≠ Performance**: Info_retained đo fraction of variance giữ lại. Nhưng variance QUAN TRỌNG cho task có thể nằm trong phần bị loại bỏ (precision/recall vấn đề). Cần: actual training + eval sau hard vs soft projection.

2. **SGR differentiable penalty chưa test**: Phase 3 test `V_task - β * P_old @ V_task` (a linear relaxation of projection). GALA's SGR thực sự dùng continuous gradient `∂/∂A ||V_t^T V_s||_F^2`. Hai cái này KHÔNG giống nhau — linear projection relaxation ≠ gradient-based penalty during training.

3. **Optimal β chưa tìm**: β sweep cho range [0, 0.3, 0.5, 0.7, 0.85, 1.0] — nhưng λ₁ trong SGR loss KHÔNG tương đương β trong linear projection. Relationship chưa established.

4. **Temporal dynamics thiếu**: CL sequence measures SSE growing over time — nhưng GALA's SGR operates DURING training, dynamically adjusting. Static SSE ≠ dynamic SGR behavior.

### Phase 4 (BNG) — ✅ Tốt nhất trong 4 phases

**Có gì:**
- ✅ κ raw per task
- ✅ κ after low-rank preconditioning (k=4,8,16,32)
- ✅ Amplification ratio (strong/weak direction)
- ✅ Residual spread after PaR-based preconditioning
- ✅ Routing-training duality: ||Σ_task - Σ_pool|| / ||Σ_pool||

**Thiếu gì:**

1. **Actual gradient comparison**: κ reduction chứng minh preconditioning CÓ THỂ giúp — nhưng chưa verify rằng preconditioned gradient thực sự converge nhanh hơn. Cần training curves.

2. **Asymmetric LR effect chưa test**: BNG bao gồm `β = sqrt(||B||/||A||)` adaptive LR — không được test ở đây.

---

## 3. Đánh giá Coverage Matrix

```
                    GALA Components
                TARA   GGI    SGR    BNG    BAL    B≠0
Validate    ┌──────────────────────────────────────────┐
Metric      │  ✅     ✅     ✅     ✅     ❌     ❌   │  ← Geometric/statistical
Proxy       │  ⚠️     ⚠️     ⚠️     ✅     N/A    N/A  │  ← Proxy faithful?
Training    │  ❌     ❌     ❌     ❌     ❌     ❌   │  ← Actual LoRA training
Accuracy    │  ❌     ❌     ❌     ❌     ❌     ❌   │  ← Final task accuracy
CL (multi)  │  ❌     ❌     ⚠️     ❌     ❌     ❌   │  ← CL sequence effect
            └──────────────────────────────────────────┘
```

> [!CAUTION]
> **Kết luận rõ ràng: Tất cả thử nghiệm hiện tại ở mức STATIC GEOMETRIC ANALYSIS. Không có actual LoRA training.** Đây là lỗ hổng lớn nhất.

---

## 4. Verdict: Chạy hết có đủ để kết luận chắc chắn không?

### ✅ Có thể kết luận chắc chắn:

1. **H1 (Task complexity varies)**: PaR range đủ cho kết luận ✅
2. **H4 (Anisotropy)**: κ measurement trực tiếp ✅
3. **H6 (PaR variants differ)**: Đo lường trực tiếp ✅
4. **H7 (Domain overlap)**: Chordal distance measurement ✅
5. **H5 (Duality — structural)**: ||Σ_task - Σ_pool|| measurement ✅

### ⚠️ Có thể kết luận ĐỀ NGHỊ (suggestive but not conclusive):

6. **H2 (GGI > PCA > Random)**: Fisher ratio là proxy, không phải actual training result. Kết luận: "GGI subspace chứa nhiều task-discriminative information hơn" — YES. "GGI init dẫn đến better training" — NOT YET PROVEN.
7. **H3 (Soft > Hard)**: Info retained metric suggestive. Kết luận: "Hard projection loại bỏ task-relevant variance" — YES. "SGR thực sự protect cũ mà không hại mới" — NOT YET PROVEN.

### ❌ KHÔNG THỂ kết luận:

8. **TARA rank → better accuracy**: Chưa có training experiments
9. **GGI init → faster convergence**: Chưa có training curves
10. **SGR → better BWT (backward transfer)**: Chưa có multi-task CL eval
11. **BNG → faster training**: Chưa có optimizer comparison
12. **Full GALA > GainLoRA baseline**: Chưa có end-to-end comparison

---

## 5. Khuyến nghị: Thử nghiệm cần bổ sung

### Tier 1: Bắt buộc trước khi claim kết luận (🔴)

| # | Thử nghiệm | Mục đích | Cách làm |
|---|------------|---------|---------|
| **T1** | **Proxy validation** | Verify Σ_task - Σ_pool ≈ Σ_grad | Chạy K=100 forward-backward trên actual model, compute actual Σ_grad, compare PaR(Σ_grad) vs PaR(Σ_residual). Cần actual model weights. |
| **T2** | **GGI init training** | Verify GGI init → better accuracy | Train 1 task (e.g., sst2) with 3 init: (a) Kaiming+zeros (current), (b) PCA-based, (c) GGI-based. Compare loss curves + final accuracy. |
| **T3** | **SGR vs Hard CL sequence** | Verify SGR → better AP/BWT | Train 5 tasks sequentially: (a) Hard projection (InfLoRA-style), (b) SGR (λ₁ sweep). Compare per-task accuracy, AP, BWT. |

### Tier 2: Rất khuyến khích (🟡)

| # | Thử nghiệm | Mục đích |
|---|------------|---------|
| **T4** | TARA rank sweep vs accuracy | Verify accuracy saturation at r ≈ TGC_eff |
| **T5** | BNG vs AdamW training curves | Verify preconditioning speeds convergence |
| **T6** | Balance regularization ablation | Verify L_bal stabilizes training |
| **T7** | B ≠ 0 init early dynamics | Verify early gradient non-zero cho A |
| **T8** | Full GALA vs GainLoRA e2e | End-to-end comparison |

### Tier 3: Nice to have (🟢)

| # | Thử nghiệm |
|---|------------|
| **T9** | Whitened vs raw space GALA |
| **T10** | Layer-wise TARA variation |
| **T11** | λ₁ scheduling strategies |
| **T12** | Task order robustness |

---

## 6. Nhận xét về thiết kế thử nghiệm hiện tại

### Điểm mạnh ✅
1. **Code quality**: `validate_gala.py` viết tốt, có GPU/CPU paths, error handling, JSON output
2. **Reproducible**: Deterministic random seed (42), configurable via CLI
3. **Cross-backbone**: Chạy trên T5-large, T5-xl, LLaMA — coverage tốt
4. **Whitened comparison**: Tự động chạy cả raw và whitened
5. **Hypothesis tagging**: Mỗi phase clearly maps to hypotheses (H1-H7)
6. **Summary output**: Auto-generated hypothesis pass/fail

### Điểm yếu ❌
1. **Tất cả proxy-based**: Không có actual LoRA training → kết luận chỉ ở mức "suggestive"
2. **GGI proxy sai**: Dùng LDA criterion (Σ_B/Σ_W) thay cho GALA's generalized EVP (Σ_grad/Σ_x)
3. **SGR proxy sai**: Dùng linear β-relaxation thay cho continuous gradient penalty
4. **Thiếu statistical tests**: Chỉ có mean comparison, không có p-values hay confidence intervals (trừ Random trong Phase 2)
5. **Thiếu cross-validation**: Không split train/test → potential overfitting trong metric evaluation

---

## 7. Kết luận tổng

> [!IMPORTANT]
> **Các thử nghiệm hiện tại ĐỦ để confirm CÁC TIỀN ĐỀ HÌNH HỌC (geometric preconditions) của GALA**, nhưng **KHÔNG ĐỦ để kết luận GALA thực sự cải thiện LoRA training.**

Cụ thể:
- ✅ "Embedding space có tính chất X, Y, Z" → **có thể kết luận chắc chắn**
- ⚠️ "Vì embedding space có tính chất X → phương pháp P sẽ tốt hơn Q" → **chỉ suggestive, cần training experiments (T1-T3)**
- ❌ "GALA > GainLoRA" → **hoàn toàn chưa có bằng chứng**

**Hành động cần thiết trước khi code GALA chính thức:**
1. Chạy `validate_gala.py` trên tất cả backbone × benchmark → thu thập H1-H7 evidence
2. Thực hiện T1 (proxy validation) → quyết định proxy có đủ tốt không
3. Thực hiện T2 (GGI training) trên 1-2 tasks → proof-of-concept trước khi refactor codebase
4. Nếu T1-T2 positive → proceed to T3 (SGR CL) → rồi mới full implementation
