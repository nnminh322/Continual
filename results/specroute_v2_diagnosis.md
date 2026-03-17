# SpecRoute V2 → V3: Chẩn đoán toàn diện & Kế hoạch khắc phục

## 1. Tổng quan kết quả V2

| Metric | SpecRoute V2 | ROOT (GainLoRA-InfLoRA) | Gap |
|--------|-------------|------------------------|-----|
| AP(EM) | 30.73 | 59.70 | **-28.97** |
| AP(rougeL) | 38.00 | 61.66 | **-23.66** |

### Phân loại thất bại chi tiết

**Nhóm 1: EM = 0 suốt quá trình training (LABEL FORMAT MISMATCH)**
| Task | Pos | Prediction | Ground Truth | Nguyên nhân |
|------|-----|-----------|-------------|-------------|
| imdb | 8 | "positive"/"negative" | "Good"/"Bad" | Misrouted → yelp/amazon LoRA |
| sst2 | 9 | "negative" | "Bad" | Misrouted → yelp/amazon LoRA |
| wic | 15 | "the same meaning"/"different" | "True"/"False" | Misrouted → unknown LoRA |
| cb | 4 | "bedroom"/"yes"/"virtuous"/gibberish | "entailment"/"neutral"/"contradiction" | Misrouted → unrelated LoRA |

**Nhóm 2: Catastrophic Forgetting**
| Task | Best EM | Final EM | Nguyên nhân |
|------|---------|----------|-------------|
| mnli | 31.25 (ep8) | 0.25 | Degenerate → always "neutral" |
| rte | 51.26 (ep4) | 0.36 | Degenerate → always "entailment" |

**Nhóm 3: Hoạt động bình thường**
| Task | SpecRoute | ROOT | Status |
|------|-----------|------|--------|
| copa | **47.00** | 42.00 | ✅ Better (+5.0) |
| multirc | **55.42** | 50.52 | ✅ Better (+4.9) |
| boolq | **61.44** | 60.43 | ✅ Better (+1.0) |
| qqp | **77.03** | 76.96 | ✅ Tie |

---

## 2. Chẩn đoán nguyên nhân gốc (Root Cause Analysis)

### 2.1 BUG CHỦ YẾU: Constant Training Bias Không Scale Theo Số Task

**Công thức hiện tại:**
$$\text{fit}_{\text{cur}} = \frac{1}{L}\sum_\ell \frac{\sum_i (a_i \cdot h)^2}{r \|h\|^2} + \beta \quad (\beta = 1.0 \text{ cố định})$$

**Softmax routing weight cho current task:**
$$w_{\text{cur}} = \frac{e^{(\text{fit}_{\text{cur}})/T}}{e^{(\text{fit}_{\text{cur}})/T} + (n-1) \cdot e^{\text{fit}_{\text{old}}/T}}$$

Với $\beta = 1.0$, $T = 1.0$, fit_raw ≈ 0.12, fit_old ≈ 0.20:

| n_tasks | Training $w_{\text{cur}}$ | Inference $w_{\text{cur}}$ |
|---------|--------------------------|---------------------------|
| 1 | 100% | 100% |
| 2 | 71.5% | 48.0% |
| 5 | 38.5% | 18.8% |
| **8 (imdb)** | **26.4%** | **11.7%** |
| 10 | 21.8% | 9.3% |
| **15 (wic)** | **15.2%** | **6.2%** |

**Hậu quả:**
- Task 8 (imdb): Chỉ 26.4% routing weight khi training → 73.6% gradient signal đi qua LoRA cũ → model không thể học label "Good"/"Bad"
- Task 15 (wic): Chỉ 15.2% routing weight → gần như không học được gì

**So sánh với ROOT:** ROOT dùng sigmoid độc lập cho mỗi task: $w_k = |2\sigma(4\cos(x, \text{key}_k)) - 1|$. Không có zero-sum competition → mỗi task có thể đạt weight ~0.8 bất kể số task.

### 2.2 BUG THỨ HAI: Bất đối xứng A-row fit vs SVD fit (Train-Test Gap)

**Training:** $\text{fit}_{\text{cur}} = \text{A-row fit} + \beta$  
**Inference:** $\text{fit}_{\text{cur}} = \text{A-row fit}$ (no bias)

Hai formula đo fit trên **hai thang đo khác nhau**:
- **A-row fit** (current task): $\frac{\sum_i (a_i \cdot h)^2}{r \|h\|^2}$ — uniform weighting
- **SVD fit** (old tasks): $\frac{\sum_i \sigma_i^2 (v_i \cdot h)^2}{\sum_i \sigma_i^2 \|h\|^2}$ — $\sigma^2$-weighted

Sau null-space projection, A rows bị constrained vào subspace hẹp → A-row fit **hệ thống thấp hơn** SVD fit → old tasks luôn thắng routing tại inference.

### 2.3 BUG THỨ BA: Threshold quá thấp (0.980 vs ROOT 0.995)

- threshold = 0.980 → mỗi task chiếm **nhiều hơn** null-space
- Sau 7 tasks: null-space còn lại cho task 8 rất hẹp
- A_8 rows bị project vào null-space nhỏ → A-row fit cực thấp
- ROOT dùng 0.995 → mỗi task chiếm ít null-space hơn → duy trì capacity cho tasks sau

---

## 3. Phân tích lý thuyết (Theory-Backed)

### 3.1 Softmax Competition Bias (Information Theory)

Softmax + constant bias vi phạm **principle of maximum entropy** khi số task tăng. Với $n$ tasks và fit gần nhau, softmax converge về phân phối uniform $1/n$ (maximum entropy). Constant bias $\beta$ không đủ để chống lại entropy này khi $n$ lớn.

**Adaptive bias derivation:** Muốn current task đạt weight $\alpha$ cố định:
$$\alpha = \frac{e^{(f + \beta)/T}}{e^{(f + \beta)/T} + (n-1)e^{f/T}}$$

Giải cho $\beta$:
$$\boxed{\beta = T \cdot \ln\left(\frac{\alpha(n-1)}{1-\alpha}\right)}$$

Với $\alpha = 0.8$, $T = 1.0$:
- n=2: $\beta$ = 1.39 → w = 80%
- n=8: $\beta$ = 3.33 → w = 80%
- n=15: $\beta$ = 4.03 → w = 80%

**Kết nối paper**: Tương tự "bias correction" trong Adam optimizer — bias phải thay đổi theo thời gian để duy trì tính chất thống kê mong muốn.

### 3.2 Rayleigh Quotient Symmetry (Linear Algebra)

Fit formula hiện tại vi phạm **measurement symmetry**: current task và old tasks dùng metric khác nhau. Trong Grassmannian geometry, khoảng cách giữa hai subspace phải dùng cùng một metric.

**Weighted Rayleigh quotient** (chuẩn cho cả hai):
$$\text{fit}_k(h) = \frac{\sum_{i=1}^r \sigma_{k,i}^2 (v_{k,i} \cdot h)^2}{\sum_{i=1}^r \sigma_{k,i}^2 \cdot \|h\|^2}$$

Tại inference, current task cũng phải dùng SVD-based fit (SVD có sẵn vì B ≠ 0 sau training).

**Kết nối paper**: Principal angle theory (Björck & Golub, 1973) — khoảng cách giữa subspaces phải đo bằng canonical angles, tương đương $\sigma$-weighted Rayleigh quotient.

### 3.3 Null-Space Capacity Bound (GPM Theory)

Từ GPM (Saha et al., 2021): với threshold $\tau$, mỗi task chiếm $\leq r(1-\tau)$ dimensions. Capacity cho $n$ tasks:

$$n_{\max} = \left\lfloor \frac{d}{r(1-\tau)} \right\rfloor$$

Với d=512, r=8:
- $\tau$ = 0.995 → capacity = 512/(8×0.005) = **12,800 tasks** (rất dư)
- $\tau$ = 0.980 → capacity = 512/(8×0.020) = **3,200 tasks** (vẫn dư nhưng aggressive hơn)

Threshold 0.995 bảo vệ nhiều capacity hơn cho tasks sau.

---

## 4. Kế hoạch sửa: SpecRoute V3

### Fix 1: Adaptive Training Bias (Bắt buộc)
```
β = T · ln(α · (n_old) / (1-α))    khi n_old ≥ 1
β = 0                                khi n_old = 0
```
- `n_old = len(self.spectral_signatures)` — tự động từ số signatures đã load
- `α = target_routing_alpha` — config parameter, default 0.8
- Đảm bảo w_cur ≈ 80% bất kể số task

### Fix 2: Symmetric Inference Routing (Bắt buộc)
- **Training**: Giữ A-row fit + adaptive bias (cold-start compatible)
- **Inference**: Tính SVD(B@A) cho current task → dùng SVD fit cho TẤT CẢ tasks
- Method: `prepare_inference_routing()` — gọi 1 lần trước inference
- Loại bỏ hoàn toàn asymmetry A-row vs SVD

### Fix 3: Threshold = 0.995 (Match ROOT)
- Chỉ thay đổi trong shell script
- Giảm null-space consumption per task
- Bảo toàn capacity cho tasks sau

### Không thêm gì khác
- Không thêm KL replay (vi phạm zero-replay settings)
- Không thêm learned routing parameters (mất novelty parameter-free)
- Không thay đổi optimizer/lr/scheduler
- Tôn trọng nguyên tắc: "chỉ cải thiện implement, không over-engineer"

---

## 5. Code Changes Cụ Thể

### File 1: `t5_specroute.py`
1. Thêm `prepare_inference_routing()` method vào T5Stack
2. Sửa `compute_spectral_routing()`:
   - Training: A-row fit + `adaptive_training_bias` (computed from α and n_old)
   - Inference: SVD fit từ `_current_task_svd` (precomputed)
3. Thêm property `adaptive_training_bias`

### File 2: `run_t5.py`
1. Thêm `target_routing_alpha` vào prompt_config
2. Gọi `model.encoder.prepare_inference_routing()` trước inference

### File 3: `gen_script_long_order3_t5_small_specroute_v3.sh`
1. `--threshold 0.995`
2. `--transthreshold 0.995`
3. `--target_routing_alpha 0.8`
4. Output dir: `specroute_v3`
