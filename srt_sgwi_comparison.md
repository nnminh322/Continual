# SRT & SGWI: So sánh Lý thuyết giữa T5 Version và LLaMA Version

> Chỉ phân tích toán học và lý thuyết. Không phân tích code.
> Chỉ xét code path thực sự được gọi khi chạy bash script.

---

## Bảng Tổng Quan

| | **T5 Version** (`code_srt_sgwi_v1`) | **LLaMA Version** (`new_llama_gainlora`) |
|---|---|---|
| **Script entry** | `src/run_t5.py` | `run_llama_gainlora_cl.py` |
| **Trainer** | `SGWI_DualFisher_Trainer` (→ `SRT_Trainer` → `GainLoRATrainer`) | `SRTSGWITrainer` (→ `Trainer`) |
| **Router** | `SRTRouter` (hard mode) | `SRTRouter` / `PooledMahalanobisRouter` |
| **sgwi_mode thực tế** | `sgwi_full` | `sgwi=True` → `sgwi_full` |
| **SRT metric mode** | `hard` (ZCA whitening + L2) | `hard` (Pooled Mahalanobis) |

---

## 1. SRT — Statistical Routing Theory

### 1.1 Embedding Extraction

Cả hai version đều extract embedding từ **frozen pretrained backbone** (không dùng adapted model) qua forward pass trên training data, rồi pool về vector `h ∈ ℝᵈ`.

| | **T5 Version** | **LLaMA Version** |
|---|---|---|
| Backbone layer | `last_encoder_hidden_state` | `last_hidden_state` (LLaMA decoder) |
| Pooling | Mean pooling over non-padding tokens | Last non-padding token |

Sau khi extract được batch embeddings `H_train = {h_i}` cho task `t`, signature được compute từ sufficient statistics:

```
μ_t = mean(H_train)                        ∈ ℝᵈ
Σ_t = cov(H_train, ddof=1)                 ∈ ℝᵈˣᵈ
n_t = |H_train|
```

---

### 1.2 T5 Version: ZCA Whitening + L2

**Lý thuyết cốt lõi (SRT, hard mode):**

Mỗi task `t` có centroid `μ_t` trong không gian embedding gốc. Khoảng cách Mahalanobis với task-specific precision matrix `Σ_t⁻¹` không so sánh được được giữa các tasks (vì mỗi task có `Σ_t⁻¹` khác nhau).

**Giải pháp:** Fit một ZCA whitening transform `W_zca` trên **pooled covariance** của tất cả tasks, rồi biến đổi tất cả centroids về cùng một không gian whitened.

**Bước 1 — Welford–Hart pooled update** (khi thêm task mới `t`):

```
μ_pool^{(new)} = (n_old·μ_pool + n_t·μ_t) / (n_old + n_t)

δ = μ_t - μ_pool
C = (n_old·n_t / (n_old+n_t)) · δδᵀ

Σ_pool^{(new)} = [(n_old-1)·Σ_pool + (n_t-1)·Σ_t + C] / (n_old + n_t - 1)
n_pool^{(new)} = n_old + n_t
```

**Bước 2 — ZCA whitening** (fit ONCE khi có ≥1 task):

```
μ_global = μ_pool

Σ_pool = eigendecompose(Σ_pool)
Σ_pool = V · diag(λ_i) · Vᵀ

W_zca = V · diag(1/√λ_i) · Vᵀ
```

Whitening transform `W_zca` **cố định** sau khi fit, **không bao giờ refit lại** qua các task tiếp theo.

**Bước 3 — Whitened centroids** (cho task `t` đang được thêm):

```
μ_t^w = W_zca · (μ_t - μ_global)         ∈ ℝᵈ
Σ_t^w = W_zca · Σ_t · W_zcaᵀ             ∈ ℝᵈˣᵈ
```

Các task đã có cũng được re-whiten với `W_zca` cố định từ raw centroids của chúng.

**Bước 4 — Optional shrinkage** (trước whitening):

```
Σ_t_shrunk = (1 - δ_LW) · Σ_t + δ_LW · λ̄·I       (Ledoit-Wolf)
δ_LW = shrink_factor = 0.1 (default)
λ̄ = trace(Σ_t) / d
```

**Bước 5 — Routing (inference):**

```
Given new embedding h:
  h^w = W_zca · (h - μ_global)                     [whiten]

  For each task t:
    d(h, t) = ||h^w - μ_t^w||²                     [L2 trong whitened space]
            = (h^w - μ_t^w)ᵀ(h^w - μ_t^w)

  Route to: argmin_t d(h, t)
```

**Equivalence:** Trong whitened space, `Σ = I`, nên Mahalanobis distance `= (h - μ)ᵀΣ⁻¹(h - μ) = ||h^w - μ^w||² = L2`. T5 version dùng L2 trong whitened space ≈ T5 version dùng Mahalanobis distance. Hai cách tính là tương đương về mặt toán.

---

### 1.3 LLaMA Version: Direct Pooled Mahalanobis

**Lý thuyết cốt lõi:**

Thay vì whitening tất cả centroids, LLaMA version giữ nguyên centroids trong không gian gốc và dùng **một global pooled covariance** `Σ_pool` làm precision matrix cho tất cả các task.

**Bước 1 — Welford–Hart pooled update** (tương tự T5):

```
μ_pool, Σ_pool, n_pool ← welford_update(μ_pool, Σ_pool, n_pool, μ_t, Σ_t, n_t)
```

**Bước 2 — Ridge shrinkage** trên `Σ_pool`:

```
δ = d / (n_pool + d + ε)                    [ridge heuristic]
λ̄ = trace(Σ_pool) / d

Σ_pool_shrunk = (1 - δ) · Σ_pool + δ · λ̄ · I

Σ_pool_shrunk = eigendecompose
Σ⁻¹_pool = V · diag(1/λ_i_clamped) · Vᵀ
λ_i_clamped = max(|λ_i|, 1e-6 · max(|λ|))
```

Shrinkage intensity tự động theo tỷ lệ `d/(n_pool+d)`: khi `n_pool` nhỏ → shrinkage mạnh, khi `n_pool` lớn → shrinkage yếu.

**Bước 3 — Optional PCA** (nếu `--srt_pca_components k` được set):

```
X_centered = H - mean(H)                          ∈ ℝⁿˣᵈ
Σ = X_centeredᵀX_centered / (n-1)                ∈ ℝᵈˣᵈ

Σ = eigendecompose(Σ) = V·S·Vᵀ
V_k = V[:, :k]                                    ∈ ℝᵈˣᵏ
H_pca = X_centered · V_k                          ∈ ℝⁿˣᵏ
μ_t_pca = mean(H_pca)
```

PCA giảm chiều từ `d=4096` xuống `k << d` trước khi tính Mahalanobis, ổn định hóa ước lượng covariance khi `n << d`.

**Bước 4 — Routing (inference):**

```
Given new embedding h:

  For each task t:
    d(h, t) = (h - μ_t)ᵀ · Σ⁻¹_pool · (h - μ_t)

  Route to: argmin_t d(h, t)
```

---

### 1.4 So sánh SRT

| | **T5 Version** | **LLaMA Version** |
|---|---|---|
| **Công thức routing** | `d = ||W_zca·(h-μ_global) - μ_t^w||²` | `d = (h-μ_t)ᵀ·Σ⁻¹_pool·(h-μ_t)` |
| **Equivalence** | L2 trong whitened space = Mahalanobis với `Σ=I` | Direct Mahalanobis |
| **Whitening** | ZCA whitening transform `W_zca` fit-once | Không whitening |
| **Precision matrix** | Per-task `Σ_t⁻¹` trong whitened space | Global pooled `Σ⁻¹_pool` |
| **Centroids** | Whitened: `μ_t^w = W_zca·(μ_t - μ_global)` | Raw: `μ_t` trong không gian gốc |
| **Covariance** | Per-task `Σ_t`, whitening sau | Pooled global `Σ_pool` |
| **Shrinkage** | Ledoit-Wolf pre-whitening (factor=0.1) | Ridge shrinkage: `δ=d/(n+d)` |
| **PCA** | Không có | Optional: giảm `d→k` trước Mahalanobis |
| **Covariance update** | Welford–Hart pooled | Welford–Hart pooled |
| **Eigendecomposition** | Trên `Σ_pool` (cho W_zca) | Trên `Σ_pool_shrunk` (cho `Σ⁻¹`) |

**Điểm tương đương:** Cả hai đều tối thiểu hóa Mahalanobis distance. T5 biến đổi không gian trước (whitening) rồi dùng L2; LLaMA giữ nguyên không gian rồi dùng trực tiếp Mahalanobis. Về mặt routing decision: `argmin_t ||h^w - μ_t^w||² = argmin_t (h-μ_t)ᵀΣ⁻¹_pool(h-μ_t)` khi `Σ⁻¹_pool = W_zcaᵀW_zca`. Điều này đúng khi `Σ_pool` được fit tốt và ZCA được fit trên cùng `Σ_pool`.

**Điểm khác biệt:**
- **T5:** whitening tạo isotropic space → L2 đơn giản hơn, nhưng refitting ZCA sau mỗi task có thể gây drift centroids
- **LLaMA:** không có whitening, dùng trực tiếp `Σ⁻¹_pool`. Shrinkage `δ=d/(n+d)` tự động điều chỉnh theo sample size. PCA optional giải quyết vấn đề `n << d`.

---

## 2. SGWI — SRT-Guided Warm Initialization

SGWI chạy **trước khi huấn luyện** task mới `t` (sau khi load LoRA của các task trước). Mục tiêu: khởi tạo `A_t` và `B_t` từ weighted combination của các past adapters.

### 2.1 SGWI Weights Computation

Cả hai version dùng **cùng một công thức** để tính weights cho mỗi past task `s`:

```
Step 1 — Extract current task embeddings:
  h_t = extract(frozen_backbone, current_task_data)        ∈ ℝⁿᵗˣᵈ
  μ_t = mean(h_t)

Step 2 — Mahalanobis distance từ μ_t đến mỗi μ_s:
  d_s = (μ_t - μ_s)ᵀ · Σ⁻¹_pool · (μ_t - μ_s)
```

> **Lưu ý:** T5 version và LLaMA version đều dùng `Σ⁻¹_pool` (pooled covariance inverse) để compute SGWI weights. T5 version có fallback L2 khi `pooled_cov` không tồn tại, nhưng trong thực tế `Σ⁻¹_pool` luôn có.

```
Step 3 — Softmax với median temperature:
  τ = median({d_s}) + ε                          [median heuristic]
  w_s = exp(-d_s / τ) / Σ_s exp(-d_s / τ)
```

Median heuristic tự động chọn temperature scale-appropriate: nếu distances lớn → temperature lớn → weights mềm hơn; nếu distances nhỏ → temperature nhỏ → weights sắc hơn.

**Output:** `w_s ∈ [0,1], Σ_s w_s = 1` cho mỗi past task `s`.

---

### 2.2 SGWI-A: Warm-init lora_A

**Công thức (cả hai version giống hệt):**

```
Step 1 — Weighted direction matrix:
  ΔW = Σ_s w_s · (B_s · A_s)          ∈ ℝ^(out_dim)×^(in_dim)
      = Σ_s w_s · W_s                  [W_s = past LoRA's effective weight change]

Step 2 — SVD decomposition:
  ΔW = U · S · Vᵀ                      (full_matrices=False)
  U ∈ ℝ^(out_dim)×ʳ, S ∈ ℝʳ, Vᵀ ∈ ℝʳ×^(in_dim), r = rank(ΔW)

Step 3 — Extract top-r input directions:
  A_t = √S[:r] ⊗ Vᵀ[:r, :]            ∈ ℝʳ×^(in_dim)
       = diag(√S[:r]) · Vᵀ[:r, :]
```

`A_t` capture những input directions quan trọng nhất (top singular directions) của weighted combination `ΔW`. Điều này định hướng lora_A theo hướng có variance cao nhất trong past adapters.

---

### 2.3 SGWI-B: Warm-init lora_B

**Công thức (cả hai version giống hệt):**

```
Mục tiêu: tìm B_t sao cho B_t · A_t ≈ ΔW

Least-squares solution:
  B_t = ΔW · A_tᵀ · (A_t · A_tᵀ + ε·I)⁻¹    ∈ ℝ^(out_dim)×ʳ
```

Công thức này solve: `min ||B·A - ΔW||_F`. Giải bằng normal equation:
`A·Aᵀ·Bᵀ = ΔWᵀ·A` → `Bᵀ = (A·Aᵀ + εI)⁻¹·A·ΔWᵀ` → `B = ΔW·Aᵀ·(A·Aᵀ + εI)⁻¹`

Ridge term `ε = 1e-4` đảm bảo `A·Aᵀ` khả nghịch ngay cả khi `A` gần singular.

**Output:** `B_t` được warm-initialized để match `ΔW` projection qua `A_t`.

---

### 2.4 So sánh SGWI

| | **T5 Version** | **LLaMA Version** |
|---|---|---|
| **SGWI weights** | Mahalanobis với `Σ⁻¹_pool` | Mahalanobis với `Σ⁻¹_pool` |
| **Fallback weights** | L2 nếu `pooled_cov=None` | Không có fallback (luôn dùng Mahalanobis) |
| **Temperature** | `τ = median(d_s) + ε` | `τ = median(d_s) + ε` |
| **A initialization** | `√S[:r] ⊗ Vᵀ[:r,:]` | `√S[:r] ⊗ Vᵀ[:r,:]` |
| **B initialization** | `ΔW·Aᵀ·(A·Aᵀ+εI)⁻¹` | `ΔW·Aᵀ·(A·Aᵀ+εI)⁻¹` |
| **Công thức toán** | **Hoàn toàn giống nhau** | **Hoàn toàn giống nhau** |

---

## 3. Tổng Kết Toán Học

### 3.1 SRT

| | **T5 Version** | **LLaMA Version** |
|---|---|---|
| Routing distance | `\|\|W_zca·(h-μ_global) - μ_t^w\|\|²` | `(h-μ_t)ᵀ·Σ⁻¹_pool·(h-μ_t)` |
| Core trick | Whitening biến Mahalanobis → L2 | Dùng pooled Σ⁻¹ trực tiếp |
| Covariance usage | Per-task Σ, pooled để fit W_zca | Global pooled Σ |
| Shrinkage target | Scalar `λ̄·I` (Ledoit-Wolf) | Scalar `λ̄·I` (ridge) |
| Shrinkage intensity | Fixed factor `δ=0.1` | Adaptive `δ=d/(n+d)` |

### 3.2 SGWI

| | **T5 Version** | **LLaMA Version** |
|---|---|---|
| SGWI weights | `softmax_τ(-d_s)` với `d_s = Mahalanobis` | `softmax_τ(-d_s)` với `d_s = Mahalanobis` |
| A warm-init | `√S·Vt` từ SVD(ΔW) | `√S·Vt` từ SVD(ΔW) |
| B warm-init | `ΔW·Aᵀ·(A·Aᵀ+εI)⁻¹` | `ΔW·Aᵀ·(A·Aᵀ+εI)⁻¹` |
| **Tất cả công thức** | **Giống hệt nhau** | **Giống hệt nhau** |

### 3.3 Key Takeaways

**SRT:** Cùng mục tiêu (routing tối thiểu Mahalanobis distance). T5 biến đổi không gian trước (ZCA whitening), LLaMA tính trực tiếp với pooled precision. Khác về implementation nhưng cùng lý thuyết cơ bản. LLaMA thêm PCA optional để handle `n << d` regime.

**SGWI:** Cả hai version dùng **hoàn toàn cùng một pipeline toán**: Mahalanobis distance → softmax weights → SVD(A) + least-squares(B). Không có sự khác biệt về công thức. T5 và LLaMA chỉ khác ở cách compute `Σ⁻¹_pool` (ZCA vs direct pooled inverse).
