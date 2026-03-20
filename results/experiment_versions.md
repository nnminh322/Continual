# Experiment Versions — SpecRoute (T5-small, Long Order 3)

**Backbone**: google/flan-t5-small (d_model=512, 8 enc+8 dec layers)
**LoRA**: r=8, target=Q+V, InfLoRA (only B trained, A frozen kaiming)
**Task order (15)**: yelp → amazon → mnli → cb → copa → qqp → rte → imdb → sst2 → dbpedia → agnews → yahoo → multirc → boolq → wic
**Settings**: zero-replay, lr=3e-4, epochs=10, threshold=0.995

---

## ROOT Baseline — GainLoRA + InfLoRA

**Script**: `gen_script_long_order3_t5_small_gainlora_inflora.sh`
**Method**: Learned MLP routing (trans_input + prompt_key), LoRA GPM (ESA), KL distill

### Kết quả cuối (sau 15 task)

| Task | EM | rougeL |
|------|---:|-------:|
| yelp | 56.01 | 70.36 |
| amazon | 52.05 | 67.08 |
| mnli | 34.07 | 34.07 |
| cb | 3.57 | 3.57 |
| copa | 42.00 | 42.00 |
| qqp | 76.96 | 76.96 |
| rte | 45.85 | 45.85 |
| imdb | 89.51 | 89.51 |
| sst2 | 85.21 | 85.21 |
| dbpedia | 98.16 | 98.16 |
| agnews | 88.37 | 88.38 |
| yahoo | 57.28 | 57.35 |
| multirc | 50.52 | 50.52 |
| boolq | 60.43 | 60.43 |
| wic | 55.49 | 55.49 |
| **AP (unweighted)** | **59.70** | **61.66** |
| **AP (weighted)** | **67.28** | **70.44** |

---

## V2 — Spectral Routing (SpecRoute ban đầu)

**Thay đổi so với ROOT**: Thay MLP routing bằng spectral routing (SVD-based Rayleigh quotient), bỏ KL distill + data replay.
**AP(EM)**: 30.73 (unweighted)
**Vấn đề chính**: 4 task EM=0 (imdb, sst2, cb, wic) do misrouting → label format mismatch. mnli, rte catastrophic forgetting.

---

## V3 — Adaptive Bias + Symmetric Inference Routing

**Thay đổi**:
- Adaptive training bias: β = T·ln(α·n_old/(1−α)), target_routing_alpha=0.8
- Symmetric inference routing: prepare_inference_routing() cho current task
- Threshold 0.995 (matches ROOT)

**AP(EM)**: 27.66 (unweighted) — **REGRESSION** từ V2
**Nguyên nhân**: V3 code chạy bằng V2 SCRIPT (threshold=0.98 thay vì 0.995). Ngoài ra, train-inference routing mismatch vẫn tồn tại.

**Kết luận V2-V3**: SVD routing CÓ HẠN CHẾ CẤU TRÚC — không phân biệt được same-domain tasks (do GPM forces A_k ⊥ A_j → spectral signatures collapse).

---

## V5 — Prototype Routing + Preconditioning + Entropy

**Script**: `T5_small/gen_script_long_order3_t5_small_specroute_v5.sh`
**Logs**: `/logs/t5_small_improve/gen_script_long_order3_t5_small_specroute_v5/`

### Motivation (GPM-Routing Paradox)
- GPM forces A_k ⊥ A_j → spectral routing via SVD(B@A) fails for same-domain tasks
- ROOT's MLP routing works vì learned parameters ≠ LoRA subspace
- **Giải pháp**: Prototype routing at inference — cosine similarity giữa encoder embedding và task prototypes (running mean during training)

### Thay đổi code (6 bugs fixed across 4 dev/review cycles)
1. **Prototype routing** (t5_specroute.py): `_update_prototype()`, `finalize_prototype()`, dual-mode inference
2. **Entropy QR fix** (cl_trainer_specroute.py): `qr(B.T)→qr(B)`, `qr(A)→qr(A.T)`
3. **Prototype temperature** T_proto=0.01 (tách biệt T_train=1.0)
4. **Preconditioning** (C4): lambda_entropy=0.01, use_preconditioning=True
5. **Init scope fix**: Prototype fields under `if not self.is_decoder`
6. **Cosine shape fix**: `.squeeze(-1)` → correct (B,) shape

### Hyperparameters
- lambda_entropy = 0.01
- use_preconditioning = True
- threshold = 0.995
- target_routing_alpha = 0.8
- lora_r = 8
- learning_rate = 3e-4
- num_train_epochs = 10

### Kết quả V5 (so sánh ROOT)

| Task | ROOT EM | V5 EM | Δ | ROOT rougeL | V5 rougeL | Δ |
|------|--------:|------:|--:|------------:|----------:|--:|
| yelp | 56.01 | 54.64 | -1.37 | 70.36 | 70.45 | +0.09 |
| amazon | 52.05 | 48.01 | -4.04 | 67.08 | 66.22 | -0.86 |
| mnli | 34.07 | 33.92 | -0.15 | 34.07 | 33.92 | -0.15 |
| cb | 3.57 | 0.00 | -3.57 | 3.57 | 0.00 | -3.57 |
| copa | 42.00 | 44.00 | +2.00 | 42.00 | 44.00 | +2.00 |
| qqp | 76.96 | 77.83 | +0.87 | 76.96 | 77.83 | +0.87 |
| rte | 45.85 | 48.01 | +2.17 | 45.85 | 52.71 | +6.86 |
| imdb | 89.51 | 88.61 | -0.90 | 89.51 | 88.61 | -0.90 |
| sst2 | 85.21 | 81.19 | -4.01 | 85.21 | 81.19 | -4.01 |
| dbpedia | 98.16 | 97.67 | -0.49 | 98.16 | 97.83 | -0.33 |
| agnews | 88.37 | 89.74 | +1.37 | 88.38 | 89.83 | +1.45 |
| yahoo | 57.28 | 49.66 | -7.62 | 57.35 | 50.37 | -6.98 |
| multirc | 50.52 | 60.44 | +9.92 | 50.52 | 60.44 | +9.92 |
| boolq | 60.43 | 61.01 | +0.58 | 60.43 | 61.01 | +0.58 |
| wic | 55.49 | 58.46 | +2.98 | 55.49 | 58.46 | +2.98 |
| **AP (unwt)** | **59.70** | **59.55** | **-0.15** | **61.66** | **62.19** | **+0.53** |
| **AP (wt)** | **67.28** | **66.65** | **-0.63** | **70.44** | **70.42** | **-0.02** |

### So sánh với V3 (prototype routing fix)

V3 có 6 task fail (EM≈0): imdb, sst2, wic, cb, rte, mnli. V5 kết quả:
- **imdb**: 0 → 88.61 ✅ FIXED
- **sst2**: 0 → 81.19 ✅ FIXED
- **wic**: 0 → 58.46 ✅ FIXED
- **rte**: 0 → 48.01 ✅ FIXED
- **mnli**: 0 → 33.92 ✅ FIXED (≈ ROOT 34.07)
- **cb**: 0 → 0.00 ❌ STILL BROKEN

### Forgetting Analysis (V5)

| Task | After Training | Final (15-wic) | Forgetting |
|------|---------------:|---------------:|-----------:|
| yelp | 55.82 | 54.64 | -1.17 |
| amazon | 48.84 | 48.01 | -0.83 |
| mnli | 34.39 | 33.92 | -0.47 |
| cb | 0.00 | 0.00 | 0.00 |
| copa | 44.00 | 44.00 | 0.00 |
| qqp | 77.82 | 77.83 | +0.01 |
| rte | 52.71 | 48.01 | -4.69 |
| imdb | 89.46 | 88.61 | -0.86 |
| sst2 | 81.42 | 81.19 | -0.23 |
| dbpedia | 98.18 | 97.67 | -0.51 |
| agnews | 89.92 | 89.74 | -0.18 |
| yahoo | 51.96 | 49.66 | -2.30 |
| multirc | 61.90 | 60.44 | -1.46 |
| boolq | 61.01 | 61.01 | 0.00 |
| wic | 58.46 | 58.46 | 0.00 |
| **Average** | | | **-0.85** |

### Training Loss per Task

| Task | Samples | Train Loss | Note |
|------|--------:|-----------:|------|
| yelp | 5000 | 0.4017 | OK |
| amazon | 5000 | 0.4064 | OK |
| mnli | 3000 | 0.6986 | Moderate |
| cb | 250 | **4.3962** | ❌ KHÔNG HỌC ĐƯỢC |
| copa | 400 | 0.4071 | OK |
| qqp | 2000 | 0.2785 | Good |
| rte | 2000 | 0.7683 | Moderate-high |
| imdb | 2000 | 0.0993 | Very good |
| sst2 | 2000 | 0.3200 | Good |
| dbpedia | 14000 | 0.0536 | Excellent |
| agnews | 4000 | 0.1756 | Good |
| yahoo | 10000 | 0.5820 | Moderate |
| multirc | 2000 | 0.5153 | Moderate |
| boolq | 2000 | 0.3958 | OK |
| wic | 2000 | 0.4847 | Moderate |

### CB Failure Analysis

CB là task 4 (sớm trong sequence), chỉ có **250 samples** → 8 steps/epoch → **80 total steps**.
- Loss curve: 5.25 → 3.63 (giảm nhưng chưa converge, eval_em=0.00 suốt)
- ROOT cũng gần-fail trên CB (EM=3.57) — CB là inherently difficult task với quá ít data
- Đây KHÔNG PHẢI lỗi routing — đây là lỗi single-task learning quality với extreme low resources

### Đánh giá tổng thể V5

**Thành công lớn**: Prototype routing giải quyết GPM-Routing Paradox — 5/6 task từ EM≈0 lên mức ≈ROOT.

**AP ngang ROOT**: V5 AP(EM)=59.55 ≈ ROOT 59.70 (-0.15). V5 AP(rougeL)=62.19 > ROOT 61.66 (+0.53).

**Forgetting rất thấp**: Average forgetting chỉ -0.85 (excellent cho 15-task sequence).

**Vấn đề còn tồn tại**:
1. **CB = 0.00**: Extreme low-resource task (250 samples). ROOT cũng gần-fail (3.57). Cần strategy riêng cho tiny datasets.
2. **yahoo -7.62 vs ROOT**: V5 yahoo after_training=51.96, ROOT yahoo=57.28. Đây là single-task quality issue, không phải forgetting.
3. **sst2 -4.01 vs ROOT**: V5=81.19 vs ROOT=85.21. Tương tự, single-task gap.
4. **amazon -4.04 vs ROOT**: V5=48.01 vs ROOT=52.05.
5. **rte forgetting -4.69**: Cao nhất trong tất cả tasks. rte trained=52.71 nhưng bị decay → 48.01.

---

## Changelog

| Date | Version | AP(EM) unwt | AP(rougeL) unwt | Key Change |
|------|---------|------------:|----------------:|------------|
| - | ROOT (baseline) | 59.70 | 61.66 | GainLoRA + InfLoRA |
| - | V2 | ~30.73 | - | Spectral routing (SVD) |
| - | V3 | ~27.66 | - | Adaptive bias + symmetric (WRONG script) |
| - | V5 | **59.55** | **62.19** | Prototype routing + entropy + preconditioning |
