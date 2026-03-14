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
| Data replay | ✅ | ❌ |
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

## Version 2.0 — SpecRoute + Experience Replay (Planned)

### Thay đổi về Idea

> **⚠️ IDEA CHANGE**: Version 2 thêm **Experience Replay (CE loss)** vào SpecRoute.
> 
> SpecRoute V1 claim rằng spectral routing parameter-free đủ để thay thế learned routing. V2 bổ sung rằng:
> - Spectral routing thay thế **routing mechanism** (đúng, giữ nguyên)
> - Nhưng **protection mechanisms** (data replay) là ORTHOGONAL với routing mechanism và cần được giữ lại
> - V2 sử dụng **CE replay trực tiếp** trên old task training data (không cần teacher model hay saved logits)
> - Khác ROOT (KL on routing scores): SpecRoute replay chỉ cần CE loss vì routing là parameter-free
>
> Đây là sự thay đổi từ "spectral routing is sufficient" sang "spectral routing + replay protection is the complete solution".
> Bản chất: **decouple routing mechanism khỏi protection mechanisms**.

### Kịch bản thử nghiệm
- **Model**: T5-Small (d_model=512, 6 encoder + 6 decoder layers)
- **Method**: SpecRoute V2 — spectral routing + experience replay (CE loss trên original training data)
- **Hyperparameters**: 
  - lora_r=8, lora_alpha=32, lr=3e-4, 10 epochs
  - **threshold=0.980** (giảm từ 0.995)
  - **data_replay_freq=5** (replay mỗi 5 steps)
  - **kl_ratio=0.1** (weight cho replay CE loss)
  - **gen_data_dir=CL_Benchmark** (replay từ original training data)
- **Script**: `T5_small/gen_script_long_order3_t5_small_specroute_v2.sh`
- **Platform**: Kaggle T4 GPU

### Code Changes (Actual)

**1. Bug Fix: `generate_specroute_scripts_v2.py`**  
- `do_predict=False` → `True` cho `long_order3` và `long_order4`

**2. Trainer: `cl_trainer_specroute.py`**  
- Thêm `create_memory_replay_generators()` — tạo DataLoader cycling iterators  
- `__init__()`: nhận `data_collator_replay`, `replay_dataset_dict`, tạo `replay_dataloader_dict` và `replay_iterator_dict`
- `training_step()`: Sau main CE loss backward, replay CE loss trên old task data:
  ```
  Mỗi replay_freq steps:
    For each old task:
      sample batch from replay iterator
      replay_loss = kl_ratio * CE_loss(model, replay_batch)
      replay_loss.backward()
  ```

**3. Run entry: `run_t5.py`**  
- Mở rộng replay dataset loading condition: `load_checkpoint_from OR (specroute AND cur_task_id > 0)`
- Skip `attention_weights.pkl` loading cho SpecRoute (không cần KL on routing)
- Pass `data_collator_replay`, `replay_dataset_dict` vào SpecRoute_Trainer

**4. Shell Script: `T5_small/gen_script_long_order3_t5_small_specroute_v2.sh`** (NEW)  
- threshold: 0.995 → 0.980
- data_replay_freq: -1 → 5
- Thêm: `--kl_ratio 0.1`, `--gen_data_dir CL_Benchmark`  
- Output dir: `specroute_v2` (tách biệt V1)
- V1 script giữ nguyên để so sánh

### Kết quả
> *Chưa chạy — cần thực nghiệm*

### Phân tích
> *Pending*

### Kỳ vọng
- Tasks 8 (imdb), 9 (sst2), 12 (yahoo), 15 (wic): kỳ vọng cải thiện đáng kể nhờ threshold thấp hơn (mở rộng null-space)
- Overall AP: kỳ vọng tăng từ ~39.74 lên >50 (threshold fix), replay CE giúp chống forgetting
- FT: kỳ vọng tính được (do_predict fix) và forgetting thấp hơn nhờ replay

### Nếu kết quả không đạt → V3 Plan
- **V3a**: Thêm output-level KL distillation (so sánh logits hiện tại vs teacher model snapshot) — yêu cầu lưu teacher model
- **V3b**: Thêm adaptive threshold per-layer (thay vì cùng threshold cho tất cả layers)  
- **V3c**: SpecRoute + InfLoRA-style direction expansion khi null-space quá nhỏ

---

## Changelog

| Date | Version | Change Type | Description |
|------|---------|-------------|-------------|
| 2025-XX-XX | V1.0 | Initial | First experiment — baseline SpecRoute vs ROOT GainLoRA |
| 2025-XX-XX | V2.0 | Idea + Code | Thêm experience replay (CE), giảm threshold 0.995→0.980, fix do_predict |
