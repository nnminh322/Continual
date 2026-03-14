# Báo cáo chẩn đoán thí nghiệm: SpecRoute vs GainLoRA trên T5-Small

> **Benchmark**: Long Sequence Order 3 (15 classification tasks)  
> **Model**: T5-Small (d_model=512, 6 encoder + 6 decoder layers, lora_r=8)  
> **Thí nghiệm**: SpecRoute (improve) vs GainLoRA-InfLoRA (root)

---

## 1. Xác minh kết quả: Bảng so sánh có chính xác không?

### ✅ ROOT GainLoRA: AP = 59.70 — CHÍNH XÁC

Nguồn dữ liệu: `logs/root_t5_small/.../15-wic/all_results.json`
- Task 15 (wic) có `--do_predict` → evaluation trên ALL 15 tasks (70,861 samples)
- Metrics `predict_exact_match_for_{task}` cho tất cả tasks → **đây là R_{15,j} (final row)**
- AP = mean(R_{15,j}) = 59.70 ✓ (tính đúng theo paper)

### ⚠️ SpecRoute: "AP" = 39.74 — **KHÔNG PHẢI AP THẬT**

Nguồn dữ liệu: `predict_eval_predictions.jsonl` tại MỖI task directory
- SpecRoute THIẾU `--do_predict` cho tasks 2-15 (bug trong script generator)
- File `predict_eval_predictions.jsonl` ở mỗi task chỉ chứa **current task evaluation**
- Các con số (yelp=54.36, imdb=0.21, etc.) là **R_{j,j} (diagonal = peak performance)**, KHÔNG phải R_{15,j}
- 39.74 = mean(diagonal), **KHÔNG phải AP** theo công thức paper

**Hệ quả**: AP thật của SpecRoute sẽ THẤP HƠN 39.74 vì forgetting sẽ giảm performance của các tasks đầu. Khoảng cách thực tế với ROOT có thể lớn hơn 19.96 điểm.

### Bảng so sánh đã hiệu chỉnh

| # | Task | ROOT R_{15,j} (Final) | SpecRoute R_{j,j} (Peak) | Ghi chú |
|---|------|-----------------------|--------------------------|---------|
| 1 | yelp | 56.01 | 54.36 | Tương đương |
| 2 | amazon | 52.05 | 50.01 | Tương đương |
| 3 | mnli | 34.07 | 35.50 | SpecRoute tốt hơn |
| 4 | cb | 3.57 | 0.00 | Cả hai đều thấp |
| 5 | copa | 42.00 | 44.00 | Tương đương |
| 6 | qqp | 76.96 | 76.72 | Tương đương |
| 7 | rte | 45.85 | 50.90 | SpecRoute tốt hơn |
| 8 | imdb | 89.51 | **0.21** ⚠️ | **Không thể học** |
| 9 | sst2 | 85.21 | **0.00** ⚠️ | **Không thể học** |
| 10 | dbpedia | 98.16 | 92.22 | Chấp nhận được |
| 11 | agnews | 88.37 | 68.76 | Giảm đáng kể |
| 12 | yahoo | 57.28 | **8.12** ⚠️ | **Không thể học** |
| 13 | multirc | 50.52 | 54.23 | Tương đương |
| 14 | boolq | 60.43 | 61.13 | Tương đương |
| 15 | wic | 55.49 | **0.00** ⚠️ | **Không thể học** |

**Nhận xét quan trọng**: SpecRoute scoring ở đây là PEAK (ngay sau khi train task đó), trong khi ROOT scoring là FINAL (sau khi train xong tất cả 15 tasks). Với ROOT, imdb PEAK có thể > 89.51 rồi chỉ giảm nhẹ về 89.51. Nhưng với SpecRoute, imdb PEAK đã là 0.21 — model KHÔNG THỂ HỌC task này ngay từ đầu, đây **không phải catastrophic forgetting**.

---

## 2. Tại sao FT (Forgetting) không tính được?

### Nguyên nhân trực tiếp: `--do_predict` bị thiếu

Công thức FT cần:
- R_{j,j} = performance trên task j ngay sau khi train task j (diagonal)
- R_{T,j} = performance trên task j sau khi train xong tất cả T tasks (final row)

| Method | R_{j,j} (diagonal) | R_{T,j} (final row) | FT computable? |
|--------|--------------------|--------------------|----------------|
| ROOT | ❌ Không có (tasks 1-14 thiếu cross-task eval) | ✅ Task 15 có | ❌ Thiếu diagonal |
| SpecRoute | ⚠️ Có nhưng chỉ single-task eval | ❌ Task 15 không eval cross-task | ❌ Thiếu final row |

### Nguyên nhân gốc: Bug trong script generator

File `improve_gainlora/generate_specroute_scripts_v2.py`:

```python
"long_order3": {
    ...
    "do_predict": False,    # ← BUG: nên là True
    ...
},
```

**Fix**: Đổi thành `True` cho cả `long_order3` và `long_order4`. Khi `do_predict=True`, script sẽ generate `--do_predict --predict_with_generate` cho mỗi task → `run_t5.py` sẽ evaluate trên ALL task cumulative test sets → `score.py` sẽ build được full matrix R → FT tính được.

ROOT cũng cần fix: hiện tại chỉ task 15 có `--do_predict`. Cần thêm cho tasks 1-14 để có full R matrix.

---

## 3. Phân tích nguyên nhân gốc: Tại sao SpecRoute kém?

### 3.1 KHÔNG phải do SVD/routing bugs

Sau khi đọc toàn bộ source code:
- `compute_spectral_signatures()`: SVD đúng, lưu Vt[:r] và S[:r] đúng
- `compute_spectral_routing()`: Weighted Rayleigh quotient đúng, softmax đúng
- Không có hardcoded dimensions cho T5-large
- d_model=512, lora_r=8 → SVD rank=8 capture toàn bộ non-zero singular values
- Gradient checkpointing fix đã áp dụng đúng

### 3.2 KHÔNG phải hoàn toàn do config SVD (giả thuyết ban đầu)

User hypothesis: "configs ban đầu được thiết kế cho T5_large, T5_small nên config SVD không phù hợp"

**Sự thật**: Không có config SVD-specific nào cần thay đổi cho T5-small. Các hyperparameters (lora_r=8, lora_alpha=32, threshold=0.995, temperature=1.0) là model-agnostic. Vấn đề nằm ở chỗ khác.

### 3.3 NGUYÊN NHÂN CHÍNH: Thiếu cơ chế chống forgetting

Đây là bảng so sánh **cơ chế bảo vệ** giữa 2 phương pháp:

| Cơ chế | ROOT GainLoRA | SpecRoute | Tác động |
|--------|:---:|:---:|----------|
| GPM gradient projection (LoRA A) | ✅ | ✅ | Chặn gradient phá LoRA cũ |
| KL distillation (`kl_ratio=0.1`) | ✅ | ❌ | Duy trì routing distribution cũ |
| Data replay (`gen_data_dir`) | ✅ | ❌ | Reinforce kiến thức cũ |
| Per-step GPM on routing params | ✅ | ❌ | Bảo vệ trans_input + prompt_key |
| Trans_input (learned routing) | ✅ | ❌ | Routing có gradient, học continuous |

**ROOT có 5 lớp bảo vệ, SpecRoute chỉ có 1 lớp (GPM trên LoRA A)**

Khi loại bỏ learned routing (trans_input + prompt_key), SpecRoute đồng thời loại bỏ luôn:
1. KL distillation (vì không có routing params để distill)
2. Data replay (vì không có routing MLP cần reinforce)
3. Per-step GPM trên routing params (vì không có routing params)

Đây **không phải là design intention** — SpecRoute muốn replace routing mechanism, nhưng vô tình loại bỏ luôn CÁC CƠ CHẾ BẢO VỆ đi kèm routing.

### 3.4 NGUYÊN NHÂN PHỤ: GPM null-space bão hòa sớm ở T5-small

Training loss so sánh (bằng chứng GPM over-constraining):

| Task (thứ tự) | ROOT loss | SpecRoute loss | Tỉ lệ | SpecRoute score |
|---|---|---|---|---|
| 1 yelp | 0.586 | 0.581 | 1.0x | 54.36 |
| 2 amazon | 0.540 | 0.588 | 1.1x | 50.01 |
| 5 copa | 0.455 | 0.459 | 1.0x | 44.00 |
| 6 qqp | 0.288 | 0.304 | 1.1x | 76.72 |
| 8 imdb | 1.410 | **4.149** | **2.9x** | **0.21** |
| 9 sst2 | 1.762 | **4.449** | **2.5x** | **0.00** |
| 12 yahoo | 1.189 | **3.077** | **2.6x** | **8.12** |
| 15 wic | 0.961 | **3.654** | **3.8x** | **0.00** |

**Pattern rõ ràng**: Tasks ban đầu (1-6) loss tương đương → model học OK. Tasks sau (8, 9, 12, 15) loss cao gấp 2.5-3.8x → model KHÔNG THỂ HỌC.

Nhưng thú vị: tasks 10 (dbpedia), 13 (multirc), 14 (boolq) vẫn học được tốt (loss < 1.3). Điều này cho thấy vấn đề không chỉ đơn thuần "hết null-space":

**Các tasks THẤT BẠI (imdb, sst2, yahoo, wic)** có đặc điểm chung: overlap lớn với tasks TRƯỚC ĐÓ trong feature space:
- imdb/sst2 = sentiment binary → overlap với yelp (task 1), amazon (task 2)
- yahoo = topic QA → overlap với nhiều domain trước
- wic = word sense → cần representations already claimed bởi tasks trước

**Giải thích**: GPM từ tasks 1-2 (yelp/amazon sentiment) đã "claim" sentiment-relevant directions. Khi imdb (cũng sentiment) đến, GPM ép LoRA A vào null-space orthogonal với sentiment directions → model bị ép vào directions KHÔNG LIÊN QUAN đến sentiment → không thể phân loại sentiment → loss cao, accuracy 0.

Trong ROOT GainLoRA, vấn đề này được giải quyết bởi:
- Trans_input cho phép MAP input mới vào representation space REUSE kiến thức sentiment cũ
- KL distillation cho phép routing CHUYỂN imdb sang LoRA branch sentiment đã có
- Data replay DUY TRÌ sentiment knowledge

### 3.5 Training loss cao = model không thể học, KHÔNG PHẢI catastrophic forgetting

Đây là phát hiện quan trọng nhất: comparison_results.md ghi "imdb/sst2/wic về 0 do Catastrophic Forgetting" — **NHẬN ĐỊNH NÀY SAI**.

Bằng chứng:
- imdb train_loss = 4.149 (rất cao) → model CHƯA BAO GIỜ học được imdb
- imdb prediction: "Rififi" (copy từ review text), "Negative" (sai format, label đúng là "Good"/"Bad")
- sst2 train_loss = 4.449 → tương tự

**Đây là "inability to learn" (GPM over-constraining), KHÔNG phải "learned then forgot" (catastrophic forgetting).**

---

## 4. Lỗi so sánh không công bằng

| Aspect | ROOT | SpecRoute | Vấn đề |
|--------|------|-----------|--------|
| Score type | R_{15,j} (FINAL) | R_{j,j} (PEAK/DIAGONAL) | So sánh khác loại |
| Evaluation | Cross-task (all 15) | Single-task (chỉ current) | Scope khác nhau |
| `--do_predict` | Task 15 only | Task 1 only | Cả hai đều thiếu |

**ROOT**: Đánh giá SAU KHI train xong 15 tasks → bao gồm cả forgetting
**SpecRoute**: Đánh giá NGAY SAU KHI train từng task → peak performance, chưa bao gồm forgetting

Để so sánh công bằng, cần chạy lại SpecRoute với `--do_predict` ở task 15 để có R_{15,j} cho tất cả tasks.

---

## 5. Định hướng cải tiến

### 5.1 Fix NGAY (không đổi methodology)

**A. Thêm `--do_predict` cho tất cả tasks**
```python
# generate_specroute_scripts_v2.py
"long_order3": { "do_predict": True },  # was False
"long_order4": { "do_predict": True },  # was False
```
→ Cho phép build full R matrix, tính AP/FT đúng, so sánh công bằng.

**B. Khôi phục KL distillation**

Đây là fix quan trọng nhất. SpecRoute loại bỏ learned routing nhưng KL distillation hoàn toàn có thể adapt cho spectral routing:

```python
# Concept: KL trên routing output thay vì routing params
def spectral_kl_regularization(model, old_signatures, input_embeds):
    """Duy trì routing distribution gần với snapshot sau task trước"""
    current_routing = model.compute_spectral_routing(input_embeds)
    old_routing = compute_old_routing(old_signatures, input_embeds)
    return kl_div(current_routing.log(), old_routing)
```

Tuy nhiên, vì spectral routing là deterministic (không có learnable params), KL trên routing output không tạo gradient hữu ích. Thay vào đó:

**Option tốt hơn: KL distillation trên model OUTPUT (logits)**
```python
# Sau mỗi task, lưu model logits trên replay data
# Trong training step tiếp theo:
kl_loss = kl_div(current_logits, saved_old_logits)
```

**C. Khôi phục Data Replay**

Replay không phụ thuộc vào routing mechanism. Có thể dùng generated data hoặc coreset luôn:
```bash
--gen_data_dir generated_data/lora_gen_long_t5   # Tái sử dụng tập replay của ROOT
--data_replay_freq 5                              # Replay mỗi 5 steps
--kl_ratio 0.1                                    # Weight cho KL loss trên replay
```

### 5.2 Giảm GPM threshold cho T5-small

Với threshold=0.995, sau 15 tasks, threshold tăng lên 0.99967 → GPM giữ 99.97% variance → null-space cực nhỏ.

| threshold | Task 1 | Task 7 | Task 14 | Nhận xét |
|-----------|--------|--------|---------|----------|
| 0.995 (hiện tại) | 0.9950 | 0.9973 | 0.9997 | Quá chặt cho T5-small |
| 0.990 | 0.9900 | 0.9947 | 0.9993 | Vẫn khá chặt |
| 0.980 | 0.9800 | 0.9893 | 0.9987 | Thử nghiệm đầu tiên |
| 0.970 | 0.9700 | 0.9840 | 0.9980 | Aggressive nhưng đáng thử |

**Đề xuất**: Thử threshold=0.980 trước, nếu forgetting tăng thì kết hợp KL distillation để bù.

### 5.3 Cải tiến methodology (dài hạn)

**A. Cho phép subspace sharing**

Vấn đề gốc: GPM ép tasks tương tự (imdb/sst2 vs yelp/amazon) vào subspaces orthogonal. Cần mechanism cho phép knowledge reuse:

```python
# Ý tưởng: Nếu spectral routing gợi ý task mới SIMILAR với task cũ,
# giảm GPM protection cho directions tương tự → cho phép reuse
similarity = compute_spectral_routing(avg_input)  # routing weights
for old_task, weight in enumerate(similarity):
    if weight > threshold_reuse:
        # Giảm GPM projection cho old_task's directions
        # → cho phép refinement thay vì full orthogonality
```

**B. Hybrid routing: spectral + lightweight learned component**

Thay vì hoàn toàn parameter-free, thêm adapter nhẹ:
```python
routing = alpha * spectral_fit + (1-alpha) * learned_gate
```
- `spectral_fit`: parameter-free, ổn định, không cần GPM protection
- `learned_gate`: lightweight (MLP nhỏ), cho phép gradient flow
- `alpha`: có thể learnable hoặc fixed (e.g., 0.7)

**C. Tách biệt protection vs routing**

Thiết kế SpecRoute hiện tại **couple** routing mechanism với protection mechanisms. Cần tách:
- **Routing**: Spectral (parameter-free) — OK giữ nguyên
- **Protection**: Cần ÍT NHẤT 2 trong 3: GPM, KL distillation, data replay

---

## 6. Kế hoạch thí nghiệm tiếp theo

### Phase 1: Fix bugs + fair comparison (ưu tiên CAO)
1. Fix `generate_specroute_scripts_v2.py`: `do_predict=True` cho long benchmarks
2. Regenerate scripts
3. Chạy lại SpecRoute Long Order 3 trên T5-small
4. So sánh AP/FT chính xác giữa 2 methods

### Phase 2: Thêm protection mechanisms (ưu tiên CAO)
1. Thêm KL distillation trên model output logits (replay + KL loss)
2. Thêm data replay
3. Grid search: threshold ∈ {0.995, 0.990, 0.980}, kl_ratio ∈ {0.05, 0.1, 0.2}

### Phase 3: Validate methodology (sau phase 2)
1. Nếu Phase 2 cho kết quả tốt → methodology đúng, chỉ thiếu protection
2. Nếu Phase 2 vẫn kém → spectral routing có vấn đề ở T5-small, cần hybrid approach
3. Scale lên T5-large để so sánh ở đúng scale ROOT paper dùng

---

## 7. Tổng kết

| Câu hỏi | Trả lời |
|----------|---------|
| Kết quả tổng hợp có chính xác? | ⚠️ ROOT đúng (AP=59.70), SpecRoute SAI loại metric (diagonal vs final) |
| Tại sao kết quả tệ? | SpecRoute loại bỏ routing → vô tình loại bỏ luôn KL + replay + per-step GPM |
| Do methodology hay config? | **Cả hai**: methodology thiếu protection layers + GPM threshold quá chặt cho T5-small |
| SVD có phải nguyên nhân? | **Không trực tiếp**. SVD routing code đúng, không có bugs |
| FT tại sao chưa tính? | Bug trong script generator: `do_predict=False` cho long benchmarks |
| Hướng cải tiến? | Khôi phục KL distillation + data replay, giảm GPM threshold, fix scripts |

**Kết luận cốt lõi**: Ý tưởng spectral routing thay thế learned routing KHÔNG SAI về mặt lý thuyết. Vấn đề là khi implement, các cơ chế protection (KL, replay) bị loại bỏ theo vì chúng gắn chặt với learned routing trong code ROOT. Cần decouple routing mechanism khỏi protection mechanisms.
