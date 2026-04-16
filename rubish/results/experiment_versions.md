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

## V6 — SVD Routing + C4 (No Prototypes) — branch `new`

**Script**: V6 on branch `new`
**Logs**: `/logs/t5_small_improve/improve-gainloraim-v6.log` (interrupted at boolq epoch 4, 13/15 tasks complete)
**Method**: SVD spectral routing (symmetric) + C4 (preconditioning + entropy). Prototype routing REMOVED (violates zero-replay).

### Motivation
V5 prototype routing (AP=59.55) violates zero-replay: mean embeddings = data statistics. V6 tests if C4 alone makes SVD routing viable. Hypothesis: C4 improves expert quality → richer σ spectrum → spectral routing discriminative.

### Hyperparameters (same as V5 except no prototypes)
- lambda_entropy = 0.01, use_preconditioning = True
- threshold = 0.995, target_routing_alpha = 0.8
- lora_r = 8, lr = 3e-4, epochs = 10

### Kết quả V6 (predict scores after task 13 — multirc, last complete checkpoint)

| Task | ROOT EM | V5 EM | V6 EM | V6 rougeL | V6 Δ vs ROOT |
|------|--------:|------:|------:|----------:|-----------:|
| yelp | 56.01 | 54.64 | 36.01 | 62.40 | -19.99 |
| amazon | 52.05 | 48.01 | 36.66 | 62.00 | -15.39 |
| mnli | 34.07 | 33.92 | 0.42 | 0.43 | -33.65 |
| cb | 3.57 | 0.00 | 0.00 | 0.00 | -3.57 |
| copa | 42.00 | 44.00 | 45.00 | 45.00 | +3.00 |
| qqp | 76.96 | 77.83 | 76.83 | 76.83 | -0.13 |
| rte | 45.85 | 48.01 | 0.00 | 36.10 | -45.85 |
| imdb | 89.51 | 88.61 | 0.00 | 0.02 | -89.51 |
| sst2 | 85.21 | 81.19 | 0.00 | 0.00 | -85.21 |
| dbpedia | 98.16 | 97.67 | 52.04 | 61.34 | -46.12 |
| agnews | 88.37 | 89.74 | 54.00 | 60.02 | -34.37 |
| yahoo | 57.28 | 49.66 | 1.47 | 3.37 | -55.81 |
| multirc | 50.52 | 60.44 | 53.59 | 53.59 | +3.07 |
| **AP (13 tasks)** | **~59.70** | **~59.55** | **~27.4** | **~35.5** | **-32.3** |

### Single-Task Training Eval Scores (during training, before predict)

| Task | train_loss | eval_EM | eval_loss | Notes |
|------|--------:|------:|------:|------|
| 1-yelp | 0.5883 | 55.32 | 0.42 | OK |
| 2-amazon | 0.6224 | 49.49 | 0.46 | OK |
| 3-mnli | 0.7274 | 31.14 | 0.77 | Moderate |
| 4-cb | 4.5235 | 0.00 | 3.64 | ❌ Never learns (tiny data) |
| 5-copa | 0.4187 | 51.00 | 0.36 | OK |
| 6-qqp | 0.2910 | 76.93 | 0.25 | Good |
| 7-rte | 0.8549 | 28.88 | 0.72 | Poor (ROOT=47.29) |
| 8-imdb | 1.5361 | 0.00 | 6.37 | ❌ **NULL-SPACE COLLAPSE** |
| 9-sst2 | 2.4394 | 0.00 | 8.39 | ❌ **NULL-SPACE COLLAPSE** |
| 10-dbpedia | 0.2346 | 56.21 | 1.34 | OK but < ROOT (98.16) |
| 11-agnews | 0.3922 | 54.66 | 1.68 | OK but < ROOT (88.37) |
| 12-yahoo | 1.2078 | 1.50 | 4.66 | ❌ Near-zero |
| 13-multirc | 0.3065 | 53.59 | 0.63 | OK |

### GPM Null-Space Exhaustion (Layer 7 / Layer 1)

| After Task | L7 Dims/512 | L1 Dims/512 | Null-Space L7 | ESA Threshold |
|-----------|----:|----:|----:|---:|
| 1-yelp | 8 | 10 | 504 | 0.995 |
| 2-amazon | 36 | 33 | 476 | 0.9953 |
| 3-mnli | 69 | 57 | 443 | 0.9957 |
| 4-cb | 89 | 66 | 423 | 0.996 |
| 5-copa | 121 | 96 | 391 | 0.9964 |
| 6-qqp | 142 | 104 | 370 | 0.9967 |
| 7-rte | 147 | 104 | 365 | 0.9971 |
| 8-imdb | 161 | 110 | 351 | 0.9975 |
| 9-sst2 | 171 | 113 | 341 | 0.9978 |
| 10-dbpedia | 212 | 145 | 300 | 0.9982 |
| 11-agnews | 247 | 173 | 265 | 0.9985 |
| 12-yahoo | 248 | 173 | 264 | 0.9989 |
| 13-multirc | 344 | 286 | 168 | 0.999 |

### Root Cause Analysis — TWO Problems

**Problem 1: Single-Task Never-Learning (IMDB/SST2/Yahoo)**
- IMDB (task 8): eval_loss starts at 7.35, drops to only 6.37 over 10 epochs. EM=0.0 throughout ALL epochs.
- SST2 (task 9): Even worse — eval_loss=8.39, EM=0.0.
- **Cause**: GPM null-space collapse. By task 8, Layer 7 has 161/512 dims consumed. The remaining null-space directions are NOT aligned with IMDB's task-relevant features. LoRA-B literally cannot learn within the constrained space.
- C4 preconditioner operates WITHIN the null-space — if the null-space itself lacks task-relevant directions, preconditioning cannot help.

**Problem 2: Routing Degradation (yelp, amazon, dbpedia, agnews)**
- yelp: single-task eval=55.32, predict after 13 tasks=36.01 (-19.31)
- dbpedia: single-task eval=56.21, predict=52.04 (-4.17)
- These are NOT parameter forgetting (LoRA params frozen). This is SVD ROUTING FAILURE — later tasks' signatures cause misrouting.

### discuss_AI.txt Key Insights (external review)
1. CB failure = task-intrinsic (tiny data, ROOT also near-fail at 8.93%)
2. IMDB/SST2/RTE = GPM-depth failure (null-space collapse after 7+ tasks)
3. C4 cannot fix null-space collapse — works within same constrained space
4. Suggestion: GPM threshold tuning — relax constraint for later tasks
5. **Error in discuss**: claimed "ROOT không có GPM constraint tích luỹ này" — ROOT DOES use InfLoRA GPM. Difference is ROOT's learned MLP routing operates in SEPARATE parameter space from LoRA subspaces.

### Kết luận V6
**C4 hypothesis FAILED.** AP ≈ 27.4, below even pessimistic prediction of 30-35. SVD routing + C4 is insufficient. The problem is STRUCTURAL: GPM null-space projection destroys task-relevant directions for later tasks, and no amount of training optimization within the null-space can recover this information. V6 confirms that under zero-replay, spectral routing alone cannot match ROOT's learned routing.

---

## V8 — C5 Data-Informed Init + A-row Routing (No β) + C4 Precond

**Script**: `T5_small/gen_script_long_order3_t5_small_specroute.sh` (V8 commit)
**Logs**: `/logs/t5_small_improve/improve-gainlora-v8.log` (15/15 tasks complete, 28827s)
**Method**: C5 Constrained PCA init + A-row routing (no adaptive bias β) + C4 gradient preconditioning. No prototypes, no replay, no KL distill.

### Kết quả V8 (sau tất cả 15 tasks)

| Task | rougeL | EM | Nhận xét |
|------|-------:|---:|----------|
| yelp | 62.48 | 36.07 | -8.6 vs initial 71.08 |
| amazon | 62.05 | 36.75 | Forgetting thấp |
| mnli | 2.82 | 2.82 | ❌ Catastrophic (ban đầu 42.1) |
| cb | 0.00 | 0.00 | ❌ Never learned (tiny data) |
| copa | 48.00 | 48.00 | OK |
| qqp | 77.03 | 77.03 | Good |
| rte | 49.82 | 8.66 | Partial (routing mismatch rte EM≠rougeL) |
| imdb | 0.015 | 0.00 | ❌ Never learned |
| sst2 | 0.00 | 0.00 | ❌ Never learned |
| dbpedia | 67.06 | 55.79 | OK |
| agnews | 69.30 | 61.84 | OK |
| yahoo | 4.14 | 1.99 | ❌ Near-zero |
| multirc | 52.97 | 52.97 | OK |
| boolq | 61.80 | 61.80 | Good |
| wic | 0.00 | 0.00 | ❌ Routing failure |
| **AP** | **43.73** | **35.78** | vs ROOT 61.66 / 59.70 |

### Root Cause Analysis V8

V8 cải thiện đáng kể so với V6 (C5 fix never-learning một phần). So sánh:
- V6: imdb=0.0, sst2=0.0, dbpedia=52.0 → V8: imdb≈0.0, sst2=0.0, dbpedia=67.1
- C5 giúp nmli recovery từ 0.42 (V6) lên 2.82 — vẫn thấp (ban đầu 42.1)
- IMDB/SST2/Yahoo vẫn fail → vấn đề còn tồn tại

**Nguyên nhân chính — GPM-Routing Paradox với A-row routing:**
Khi V8 loại bỏ β (adaptive bias), current task không còn bias để win routing. Với GPM-Routing paradox:
- `A_t ⊥ h_t` (GPM buộc A_t vào null-space, h_t align với P_old directions)
- → `fit_current ≈ 0`, `fit_old > fit_current`
- → softmax chọn old task → `B_t` không nhận gradient → không học

C5 cải thiện alignment một phần (takes top eigenvectors of projected covariance) nhưng khi h_t chủ yếu nằm trong P_old subspace (yelp-like tasks), projected component nhỏ → C5 vẫn bị hạn chế.

### Kết luận V8
**Partial success.** C5 giúp mnli, rte, dbpedia, agnews, multirc cải thiện đáng kể. Nhưng imdb/sst2/yahoo vẫn EM≈0 → problem chưa giải quyết triệt để. AP=43.73 vs ROOT=61.66.

**Nhận định: V8 vẫn fail do routing bug — B_t không nhận gradient khi fit_current < fit_old.**

---

## V9 — Oracle Training + Calibrated Top-1 Inference (Bug Fix)

**Script**: Cùng script V8, code commits V9
**Logs**: Pending (V9 là fix cho routing bug trong V8)
**Method**: V8 + hai thay đổi:
1. **Oracle routing trong training**: current task luôn được route với weight=1.0, tránh GPM-Routing paradox kill gradient
2. **Calibrated Top-1 tại inference**: EMA normalization của A-row fit scores để fair comparison giữa tasks

### Thay đổi từ V8 → V9

| Aspect | V8 | V9 |
|--------|----|----|
| Training routing | Softmax/argmax A-row (β đã bỏ) | **Oracle: weight=1.0 cho current task** |
| Inference routing | argmax A-row raw | **argmax A-row calibrated (/ EMA E_fit)** |
| β adaptive bias | ❌ Đã loại bỏ | ❌ Không cần (oracle thay thế) |
| Gradient flow | Bị block khi fit_current < fit_old | **Đảm bảo: B_t luôn nhận gradient** |

### Dự kiến cải thiện

V8 fail imdb/sst2/yahoo do B_t không học (gradient bị block). V9 oracle routing fix điều này:
- Training: B_t học trong null-space với C5-init A_t → quality cao hơn V8
- Inference: calibrated argmax routing → fair comparison nhưng vẫn bị GPM-Routing paradox limit

---

## Changelog

| Date | Version | AP(EM) unwt | AP(rougeL) unwt | Key Change |
|------|---------|------------:|----------------:|------------|
| - | ROOT (baseline) | 59.70 | 61.66 | GainLoRA + InfLoRA |
| - | V2 | ~30.73 | - | Spectral routing (SVD) |
| - | V3 | ~27.66 | - | Adaptive bias + symmetric (WRONG script) |
| - | V5 | **59.55** | **62.19** | Prototype routing + entropy + preconditioning |
| - | V6 | ~27.4 | ~35.5 | SVD + C4 only (no prototypes) — **FAILED** |
| - | V8 | 35.78 | 43.73 | C5 Data-Informed Init + C4 precond + A-row routing (no β) — PARTIAL |
| - | V9 | 43.14 | 51.55 | Oracle routing (training) + calibrated Top-1 (inference) — bug fix |
| - | V10a | (pending) | (pending) | Learned Routing + GPM + C5 + C4 |
| - | V10b | (pending) | (pending) | Grassmannian Distance Routing + C5 + C4 |

---

## V10 — Duality of Routing Mechanisms

**Motivation**: V9 showed that Top-1 A-row routing struggles to isolate orthogonal subspaces despite C4+C5. V10 explores two distinct modes to address routing precision while preserving C5's benefits.

### V10a (Learned Routing - The Practical Baseline)
- **Method**: Reintroduces ROOT's `Trans_input` MLP and `prompt_key` gating, with exact GPM constraints applied to their weights post-optimizer step.
- **Why**: Proves that C5 initialization and C4 preconditioning can synergize with explicit function approximation for routing. Sacrifices the "parameter-free" claim but serves as a strong upper-bound baseline.

### V10b (Grassmannian Distance Routing - The Zero-Replay Ideal)
- **Method**: Evaluates similarity by computing the Grassmannian distance (principal angles) between the batch's local principal subspace $U_{batch}$ and expert orthogonal projection $U_A$.
- **Why**: Directly measures subset geometric alignment, entirely bypassing scale-based similarity issues (GPM-Routing paradox). Batch-level SVD aggregates representations properly. Valid for batched inference ($B \ge 8$), falling back to A-row for small batches.

### V10a Results

| Task | Final EM | Best EM | Forgetting |
|------|------:|------:|------:|
| yelp | 33.45 | 56.49 | 23.04 |
| amazon | 35.37 | 53.05 | 17.68 |
| mnli | 30.54 | 49.11 | 18.57 |
| cb | 0.00 | 57.14 | 57.14 |
| copa | 55.00 | 55.00 | 0.00 |
| qqp | 11.95 | 78.84 | 66.89 |
| rte | 10.11 | 57.76 | 47.65 |
| imdb | 89.89 | 91.51 | 1.62 |
| sst2 | 65.25 | 88.88 | 23.62 |
| dbpedia | 40.70 | 98.47 | 57.78 |
| agnews | 42.67 | 90.05 | 47.38 |
| yahoo | 61.88 | 66.01 | 4.13 |
| multirc | 43.13 | 59.12 | 15.99 |
| boolq | 62.45 | 62.45 | 0.00 |
| wic | 56.43 | 56.43 | 0.00 |
| **Cl (EM)** | **42.59** | | **27.25** |

**V10a is CATASTROPHIC**: Cl=42.59 (vs ROOT 59.70), FT=27.25 (vs ROOT ~low, V5 0.91).

### V10a Root Cause Analysis

**100% of forgetting comes from routing failure**, not weight overwriting (LoRA B matrices for old tasks are frozen in `previous_lora_weights`).

**Three critical differences from ROOT:**

1. **TransInputGPMCallback (THE KILLER)**: V10a applies GPM projection to `trans_input` + `prompt_key` every training step with threshold=0.995. By task 9, ~95% of routing feature space is locked → routing effectively frozen → cannot distinguish new tasks. ROOT does NOT constrain routing during training.

2. **Missing prompt_key re-initialization**: ROOT re-initializes `prompt_key` before each task using SVD of trans_input output covariance (task 1) or random-in-null-space (task 2+). V10a starts from `nn.init.uniform_(-1, 1)` every task → no data-informed, orthogonal starting point.

3. **No trans_input covariance collection**: ROOT collects 1000 batches of trans_input feature covariance for prompt_key initialization. V10a only collects LoRA covariance (for C5).

**The deadly combination**: Random prompt_key + Over-constrained routing = Bad starting point + Cannot learn = Routing failure = Catastrophic forgetting.

---

## V11 — ROOT Routing + C5 Init + Advanced Inference Routing

### Motivation

V10a proved that GPM on routing is fundamentally wrong: routing needs discriminative capacity, not orthogonality constraints. V11 reverts to ROOT's proven routing mechanism while keeping C5 (data-informed LoRA A init) and C4 (gradient preconditioning) for improved per-task expert quality. Additionally, V11 introduces two advanced inference-time routing strategies grounded in information theory.

### Base Fix (all V11 variants)
1. **Disable TransInputGPMCallback**: `use_routing_gpm = False` (default)
2. **ROOT-style prompt_key re-init**: SVD of trans_input output covariance (task 1) or null-space random SVD (task 2+)
3. **Keep C5**: Data-informed A init via Constrained PCA in null-space
4. **Keep C4**: Gradient preconditioning (AA^T + εI)^{-1/2}

### V11a: Base (ROOT routing + C5)
**Script**: `T5_small/gen_script_long_order3_t5_small_specroute_v11a.sh`
**Args**: `--routing_mode learned --routing_strategy base`
**Expected**: ≈ ROOT AP (routing identical), potentially better due to C5.

### V11b: Softmax Routing Normalization (Option B)

**Script**: `T5_small/gen_script_long_order3_t5_small_specroute_v11b.sh`
**Args**: `--routing_mode learned --routing_strategy softmax --routing_temp 0.1`

**Mathematical formulation:**
ROOT uses independent sigmoid routing: $w_k = |\sigma(4 \cos(x_k, p_k)) \cdot 2 - 1|$.
Each task gets weight in [0,1] independently → multiple experts may contribute equally → cross-expert interference.

V11b converts to competitive softmax gating (standard MoE):
$$p_k = \frac{\exp(s_k / \tau)}{\sum_j \exp(s_j / \tau)}$$
where $s_k = \text{logit}(w_k) = \log w_k - \log(1 - w_k)$ and $\tau$ is temperature.

**Information-theoretic justification:**
Let $Y$ = model output, $T$ = task, $X$ = input. Output: $Y = \sum_k p_k f_k(X)$.
$$H(Y|X) \geq \sum_k p_k H(f_k(X)|X) \quad \text{(concavity of entropy)}$$
Cross-expert interference term: $\sum_{j \neq k} p_j \|f_j(X) - f_k(X)\|^2$.
Minimizing this ≡ concentrating $p$ on argmax (one expert dominates) ≡ lower $\tau$.
In the limit $\tau \to 0$: softmax → argmax (hard top-1 routing, zero interference).

**Expected improvement**: Lower FT due to sharper expert selection.

### V11c: Product-of-Experts Ensemble (Option C)

**Script**: `T5_small/gen_script_long_order3_t5_small_specroute_v11c.sh`
**Args**: `--routing_mode learned --routing_strategy ensemble --routing_temp 0.1 --ensemble_weight 0.7`

**Mathematical formulation:**
Fuse learned ($p_L$) and spectral ($p_S$) routing via Product-of-Experts (Hinton, 2002):
$$p_{\text{ens}}(T=k|x) \propto p_L(T=k|x)^\gamma \cdot p_S(T=k|x)^{1-\gamma}$$
In log space:
$$\log p_{\text{ens}} = \gamma \cdot \frac{s_L^{(k)}}{\tau} + (1-\gamma) \cdot \frac{s_S^{(k)}}{\tau} + \text{const}$$

**Bayesian justification:**
If learned and spectral routing encode independent evidence about task identity $T$:
$$p(T|x) \propto p_L(T|x) \cdot p_S(T|x) \quad \text{(posterior = product of likelihoods)}$$
This is the classical Product-of-Experts derivation (assuming uniform prior on T).

**Complementary error profiles:**
- Learned routing: excels on recently trained tasks (MLP adapts); degrades on distant old tasks (feature drift)
- Spectral routing: parameter-free → zero drift; weaker on same-domain tasks (GPM forces $A_k \perp A_j$)
- When both agree: high confidence → nearly always correct
- When they disagree: hedged prediction → reduces worst-case error

**Channel capacity argument:**
Each routing method has limited channel capacity $C_L, C_S$ for encoding task identity.
Ensemble capacity: $C_{\text{ens}} \geq \max(C_L, C_S)$ (data processing inequality) with equality iff one subsumes the other.
Since learned and spectral use orthogonal feature spaces (MLP output vs A-row projection), $C_{\text{ens}} > \max(C_L, C_S)$.

**Expected improvement**: Both AP ↑ (better routing accuracy) and FT ↓ (spectral stabilizes learned).

### Hyperparameters
All V11 variants:
- lora_r = 8, lora_alpha = 32
- lr = 3e-4, epochs = 10
- threshold = 0.995, transthreshold = 0.995
- mlp_hidden_dim = 100

V11b specific: routing_temp = 0.1
V11c specific: routing_temp = 0.1, ensemble_weight = 0.7
