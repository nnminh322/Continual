# SpecRoute — Báo cáo Thử nghiệm theo Version

> Tracking tất cả versions thử nghiệm, kết quả, phân tích, và cải tiến.  
> Benchmark: Long Sequence Order 3, 15 classification tasks, model T5-Small.

---

## Version 1.0 — Baseline SpecRoute (Kết quả đầu tiên)

### Kịch bản thử nghiệm
- **Model**: T5-Small (d_model=512, 6 encoder + 6 decoder layers)
- **Method**: SpecRoute — spectral routing (SVD of LoRA B@A) thay thế learned routing (trans_input + prompt_key) của GainLoRA
- **So sánh**: ROOT GainLoRA-InfLoRA (original codebase)
- **Hyperparameters**: lora_r=8, lora_alpha=32, lr=3e-4, 10 epochs, threshold=0.995
- **Platform**: Kaggle T4 GPU

### Kết quả

| # | Task | ROOT (Final R_{15,j}) | SpecRoute (Peak R_{j,j}) | Δ |
|---|------|-----------------------|--------------------------|---|
| 1 | yelp | 56.01 | 54.36 | -1.65 |
| 2 | amazon | 52.05 | 50.01 | -2.04 |
| 3 | mnli | 34.07 | 35.50 | +1.43 |
| 4 | cb | 3.57 | 0.00 | -3.57 |
| 5 | copa | 42.00 | 44.00 | +2.00 |
| 6 | qqp | 76.96 | 76.72 | -0.24 |
| 7 | rte | 45.85 | 50.90 | +5.05 |
| 8 | imdb | 89.51 | **0.21** ⚠️ | -89.30 |
| 9 | sst2 | 85.21 | **0.00** ⚠️ | -85.21 |
| 10 | dbpedia | 98.16 | 92.22 | -5.94 |
| 11 | agnews | 88.37 | 68.76 | -19.61 |
| 12 | yahoo | 57.28 | **8.12** ⚠️ | -49.16 |
| 13 | multirc | 50.52 | 54.23 | +3.71 |
| 14 | boolq | 60.43 | 61.13 | +0.70 |
| 15 | wic | 55.49 | **0.00** ⚠️ | -55.49 |
| | **Mean** | **59.70** | **39.74** | **-19.96** |

> ⚠️ **LƯU Ý QUAN TRỌNG**: So sánh KHÔNG công bằng — ROOT dùng R_{15,j} (final, sau tất cả 15 tasks), SpecRoute dùng R_{j,j} (peak, ngay sau train từng task). AP thực của SpecRoute sẽ thấp hơn 39.74.

### Phân tích

**1. Prediction metrics không được lưu**  
- SpecRoute `all_results.json` chỉ chứa training metrics, KHÔNG có `predict_exact_match_for_{task}`
- `task_order.txt` không tồn tại → `score.py` không thể tính AP/FT
- Nguyên nhân: Có thể do experiment được chạy bằng script khác (không phải T5_small/ scripts đã fix `--do_predict`)
- T5-large script generator (`generate_specroute_scripts_v2.py`) vẫn có bug `do_predict=False` cho long benchmarks

**2. Các tasks THẤT BẠI KHÔNG PHẢI do catastrophic forgetting**  

| Task | Train Loss (Root) | Train Loss (SpecRoute) | Ratio | Verdict |
|------|:-:|:-:|:-:|---|
| imdb | 1.41 | **4.15** | 2.9x | Không thể học |
| sst2 | 1.76 | **4.45** | 2.5x | Không thể học |
| yahoo | 1.19 | **3.08** | 2.6x | Không thể học |
| wic | 0.96 | **3.65** | 3.8x | Không thể học |

Training loss cao gấp 2.5-3.8x → model KHÔNG THỂ HỌC ngay từ đầu (inability to learn, NOT catastrophic forgetting).

**3. Nguyên nhân gốc: GPM null-space saturation + thiếu protection mechanisms**

SpecRoute loại bỏ learned routing → đồng thời mất 4/5 cơ chế protection của ROOT:

| Protection Mechanism | ROOT | SpecRoute V1 |
|---------------------|:---:|:---:|
| GPM on LoRA A | ✅ | ✅ |
| KL distillation on routing | ✅ | ❌ |
| Data replay | ❌ (`data_replay_freq=-1`) | ❌ |
| Per-step GPM on routing params | ✅ | ❌ (no routing params) |
| Learned routing adaptation | ✅ | ❌ (by design) |

Khi tasks tương tự (imdb/sst2 vs yelp/amazon — cùng sentiment domain) đến, GPM đã "claim" sentiment-relevant directions → model bị ép vào orthogonal null-space không liên quan → KHÔNG thể học sentiment tasks mới.

ROOT GainLoRA giải quyết vấn đề này nhờ trans_input MLP map input mới vào representation space REUSE kiến thức cũ, kết hợp KL distillation + data replay.

**4. FT (Forgetting) = N/A**  
- Không tính được vì thiếu cross-task prediction metrics

### Cải tiến cho V2

| # | Loại | Nội dung | Tác động |
|---|------|---------|----------|
| 1 | Bug fix | Fix `do_predict=False` → `True` trong generator | Cho phép tính AP/FT đúng |
| 2 | Config | Giảm GPM threshold: 0.995 → 0.980 | Mở rộng null-space cho tasks sau |
| 3 | **Idea change** | Thêm Experience Replay (CE loss trên old task data) | Chống forgetting + hỗ trợ knowledge reuse |

---

## Version 2.0 — SpecRoute V2: Zero-Replay, Cold-Start Fix + Fair Comparison

### Thay đổi về Idea

> **⚠️ V2.0 TRƯỚC ĐÓ ĐÃ BỊ HỦY**: Phiên bản V2 trước đó thêm Experience Replay (CE loss on old data). 
> Điều này **VI PHẠM** ràng buộc zero-replay trong settings.txt:
> *"không được phép sử dụng lại bất kỳ dữ liệu cũ dưới bất kỳ hình thức nào"*
> 
> Hơn nữa, ROOT GainLoRA cũng **KHÔNG** dùng replay (`data_replay_freq=-1` cho TẤT CẢ scripts).
> ROOT đạt AP=59.70 hoàn toàn nhờ: learned routing (trans_input + prompt_key) + GPM on LoRA_A + GPM on routing params.
>
> **V2 Correct**: Fix root causes of V1 failure within zero-replay constraint.

### Root Cause Analysis (V1 Failures)

**Bug 1: Cold-Start — Code không match IDEA doc (Sec 2.2)**
- IDEA doc (Section 2.2) quy định current task routing dùng **A rows trực tiếp**:
  $$\text{fit}_\text{cur}(h) = \frac{\sum_{i=1}^{r} (a_i^\top h)^2}{r \cdot \|h\|^2}$$
- Code V1 dùng **SVD(B@A)** cho current task. Nhưng B=0 tại initialization → SVD trả S=0 → fit≈0 → routing weight≈0 → gradient≈0 → B không thể học (dead loop)
- A rows (kaiming init + null-space projection) luôn non-zero → fit_cur > 0 từ đầu

**Bug 2: Training bias thiếu**
- Ngay khi dùng A rows, fit_cur vẫn thấp hơn systematic so với old tasks (SVD-weighted σ²)
- Old fit ∈ [0,1] (Rayleigh quotient), A-based fit ≤ 1/3 (do A normalized)
- Current task nhận routing weight ~10-12% tại task 8+ → gradient yếu
- Solution: training-time bias β=1.0 cộng vào fit_cur CHỈ khi training. Inference dùng SVD signatures bình thường

**Bug 3: Batch size không fair**
- V1: BSZ=64, GA=1, effective=64
- ROOT: BSZ=8, GA=4, effective=32
- SpecRoute dùng effective BSZ gấp đôi ROOT → so sánh không công bằng

**Bug 4: GPM saturation (threshold=0.995)**
- Sau 7 tasks, null-space bị thu hẹp nghiêm trọng
- Sentiment tasks mới (imdb, sst2) bị ép vào directions orthogonal với yelp/amazon → không học được
- Fix: threshold 0.995→0.980 (already in V1 analysis)

### Kịch bản thử nghiệm
- **Model**: T5-Small (d_model=512)
- **Method**: SpecRoute V2 — A-row routing + training bias + lower threshold
- **Hyperparameters**: 
  - lora_r=8, lora_alpha=32, lr=3e-4, 10 epochs
  - **threshold=0.980** (giảm từ 0.995)
  - **training_bias=1.0** (additive bias cho current task fit khi training)
  - **data_replay_freq=-1** (KHÔNG replay, giống ROOT)
  - BSZ=8, GA=4 trên A100 (effective=32, giống ROOT)
  - BSZ=4, GA=8 trên T4-1gpu; BSZ=2, GA=8 trên T4-2gpu
- **Script**: `T5_small/gen_script_long_order3_t5_small_specroute_v2.sh`

### Code Changes (Actual)

**1. Routing Fix: `t5_specroute.py`**
- Current task: thay SVD(B@A) bằng A-row projection (match IDEA doc Sec 2.2)
  ```python
  # fit_cur(h) = Σ(a_i·h)² / (r·||h||²) — uses A rows directly
  proj = torch.matmul(A.data, h_flat.T)  # (r, N)
  fit = (proj ** 2).sum(dim=0) / (r * h_norm_sq)  # (N,)
  ```
- Training bias: `current_fit = current_fit + self.training_bias` (chỉ khi `model.training`)
- Old tasks: giữ nguyên SVD-based σ-weighted Rayleigh quotient
- Inference: tất cả tasks dùng SVD signatures (current task gets SVD after training)

**2. Replay Removal: `cl_trainer_specroute.py`**
- Xóa `create_memory_replay_generators()` function
- Xóa replay parameters từ `__init__` (data_collator_replay, replay_dataset_dict)
- Xóa replay block từ `training_step()` — chỉ giữ CE loss + gradient diagnostic
- Training step: standard CE → backward → gradient check → return loss

**3. Run entry: `run_t5.py`**
- Thêm `training_bias` vào ModelArguments (default=1.0)
- Pass `training_bias` qua `prompt_config` dict
- Xóa SpecRoute-specific replay loading condition
- Xóa `data_collator_replay`, `replay_dataset_dict` từ SpecRoute_Trainer call

**4. Shell Script: `T5_small/gen_script_long_order3_t5_small_specroute_v2.sh`**
- data_replay_freq: 5 → **-1** (disabled, match ROOT)
- kl_ratio: removed, replaced with **training_bias=1.0**
- BSZ/GA: match ROOT exactly (A100: 8/4, T4-1gpu: 4/8, T4-2gpu: 2/8)
- threshold/transthreshold: 0.980 (kept from previous)

### Kết quả
> *Chưa chạy — cần thực nghiệm*

### Kỳ vọng
- Cold-start fix → tasks sau (8+) nhận routing weight đủ lớn (>30%) → B có thể học
- Threshold 0.980 → sentiment tasks (imdb, sst2) có null-space capacity cho learning
- Training bias β=1.0 → current task dominant trong routing khi training, không ảnh hưởng inference
- Fair BSZ (effective=32 = ROOT) → so sánh AP trực tiếp
- Overall AP: kỳ vọng >50 (V1=39.74), mục tiêu tiếp cận ROOT=59.70

### Nếu kết quả không đạt → V3 Plan
- **V3a**: Adaptive training bias β per-task (higher for later tasks khi null-space nhỏ hơn)
- **V3b**: Adaptive threshold per-layer (layers gần output cần threshold thấp hơn)
- **V3c**: Warm-start: initialize B_new từ weighted combination of old B vectors (zero-replay compliant)

---

## Version 2.1 — Performance Optimization (Thin QR+SVD)

### Vấn đề
SpecRoute V1 dùng full SVD(512×512) per forward pass dù rank(B@A)≤8. Lãng phí compute.

### Tối ưu: Thin QR+SVD (ZERO accuracy loss)

**Áp dụng cho**: `compute_spectral_signatures()` (offline, after training).
**KHÔNG áp dụng cho**: current task routing (V2 dùng A rows → không cần SVD).

**Nguyên lý toán học**: Vì rank(B@A) ≤ r = 8, ta decompose qua 2 QR nhỏ + 1 SVD 8×8:
1. QR(B) → Q_B(512×8), R_B(8×8) — cost O(m·r²)
2. QR(A^T) → Q_A(512×8), R_A(8×8) — cost O(n·r²)  
3. SVD(R_B @ R_A^T) → U_s, S, Vh_s — cost O(r³) = O(512) operations
4. Vt_full = Vh_s @ Q_A^T — cost O(n·r²)

**Nghĩa toán học ĐỒNG NHẤT** — không phải approximation.

**Benchmark (CPU, 512×512 matrix, r=8)**:
- Full SVD: 12.55 ms/call → 150.6 ms per forward (12 calls)
- Thin QR+SVD: 0.067 ms/call → 0.8 ms per forward
- **Speedup: 186×**
- Relative error: ~1e-6 (machine precision)

### Code Changes

**`t5_specroute.py`**:
- Thêm hàm `_thin_svd_low_rank(B, A, device)`: QR decomposition + SVD 8×8 + recover
- `compute_spectral_routing()`: thay `torch.linalg.svd(B@A, ...)` bằng `_thin_svd_low_rank(B, A)`
- `compute_spectral_signatures()`: tương tự

### Tác động

| Component | Trước | Sau |
|-----------|-------|-----|
| SVD per signature compute | ~12.55ms | ~0.067ms |
| Speedup | — | **186×** |
| Accuracy loss | — | 0 (exact, error ~1e-6) |

> V2 không còn dùng SVD per forward cho current task (dùng A rows thẳng).
> Thin QR+SVD chỉ dùng cho `compute_spectral_signatures()` sau khi training xong mỗi task.

### Đề xuất

V2 đã tắt replay (`data_replay_freq=-1`), match ROOT. Runtime ước tính ngang ROOT (~4-5h trên T4).

---

## Changelog

| Date | Version | Change Type | Description |
|------|---------|-------------|-------------|
| 2025-XX-XX | V1.0 | Initial | First experiment — baseline SpecRoute vs ROOT GainLoRA |
| 2025-XX-XX | V2.0 (hủy) | ~~Replay~~ | ~~Thêm experience replay~~ — **BỊ HỦY** do vi phạm zero-replay constraint |
| 2025-XX-XX | V2.0 | Bug fix + Fair | A-row routing (fix cold-start), training bias β=1.0, threshold 0.980, fair BSZ=32 |
| 2025-XX-XX | V2.1 | Perf Optimization | Thin QR+SVD (~186× speedup per SVD, zero accuracy loss) |
