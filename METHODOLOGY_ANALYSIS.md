# Phân Tích Phương Pháp Luận: SRT + LoRA Initialization

## 1. Kiểm Định Lập Luận Của Bạn

### 1.1 Lập luận gốc (đúng)
> "Root GainLoRA có 2 thành phần: (a) GPM routing, (b) null-space projection trên A.
> SRT chỉ thay thế (a), nhưng (b) vẫn hoạt động → double negative: null-space exhaustion + cold-start."

**Xác nhận: Lập luận này ĐÚNG 100%.** Audit code cho thấy:

| Thành phần | Root GainLoRA | SRT (current) | Vấn đề |
|------------|--------------|---------------|--------|
| Routing (inference) | GPM attention | SRT statistical ✅ | Đã thay thế |
| Null-space init (lora_A) | InfLoRA: A'=A-A@UU^T | **Vẫn giữ nguyên** ❌ | Null-space cạn kiệt |
| GPM gradient projection | Project after each step | **Vẫn giữ nguyên** ❌ | Constrain training |
| Knowledge transfer | Qua shared prompt_key | ❌ Không có | Cold-start |

### 1.2 Bằng chứng thực nghiệm

| Thực nghiệm | Có GPM/null-space | Không GPM | Kết luận |
|-------------|-------------------|-----------|----------|
| Phase 1 CB (2 task) | 42.86% | 83.93% | GPM giết plasticity |
| Full SRT order 3 (15 task) | CB=3.57% | Chưa test | Null-space cạn kiệt |

### 1.3 Giải thích cơ chế

**Tại sao null-space + SRT là double negative:**

```
Task 1: lora_A chiếm subspace S1
Task 2: lora_A phải nằm trong S1⊥ (null-space của S1)
...
Task k: lora_A phải nằm trong (S1 ∪ S2 ∪...∪ S_{k-1})⊥
```

Với r=8 và mỗi task chiếm ~8 directions, sau ~100 tasks (d_model=1024), null-space ≈ 0.
SRT routing đã đảm bảo: inference đúng adapter → **KHÔNG CẦN** null-space constraint nữa.
Null-space lúc này chỉ có hại, không có lợi.

---

## 2. Không Gian Thiết Kế: A và B Nên Khởi Tạo/Train Thế Nào?

### 2.1 Hiểu rõ vai trò A và B

Trong LoRA: `ΔW = B @ A`, output = `(x @ A^T @ B^T) * scaling`
- **A (down-projection)**: [r, d_in] — chọn "hướng" trong input space để trích xuất
- **B (up-projection)**: [d_out, r] — ánh xạ ngược lên output space

Trong InfLoRA: A frozen → hướng trích xuất cố định → B chỉ điều chỉnh "magnitude" trên hướng đã chọn.

### 2.2 Phân tích 4 configurations

#### Config 1: Freeze A + null-space (hiện tại)
```
A = kaiming → project vào null-space → frozen
B = zeros → trainable
```
- **Expressiveness**: Thấp. A bị ép vào null-space ngày càng nhỏ. B chỉ scale hướng cố định.
- **Forgetting**: Rất tốt (by design). Nhưng SRT đã đảm bảo rồi → **dư thừa**.
- **Cold-start**: Nghiêm trọng. B=0 mỗi task, A random trong null-space.
- **Forward transfer**: 0. Không chuyển giao gì.
- **Kết quả**: CB=3.57% ở order 3 (15 tasks).

#### Config 2: Freeze A + random (bỏ null-space)
```
A = kaiming → frozen (KHÔNG project)
B = zeros → trainable
```
- **Expressiveness**: Trung bình. A random nhưng frozen → hướng trích xuất "may rủi".
- **Forgetting**: SRT routing handles this.
- **Cold-start**: Vẫn có (B=0). Nhưng A không bị ép → nhiều "phòng" hơn.
- **Forward transfer**: 0.
- **Dự kiến**: Tốt hơn Config 1 đáng kể.

#### Config 3: Full LoRA (train cả A và B, random init)
```
A = kaiming → trainable
B = zeros → trainable
```
- **Expressiveness**: Cao nhất. A tìm hướng tối ưu, B điều chỉnh magnitude.
- **Gấp 2x trainable params** nhưng mỗi task tìm được subspace riêng tốt nhất.
- **Forgetting**: SRT routing handles. Các task trước frozen trong `previous_lora_weights`.
- **Cold-start**: Nhẹ hơn vì A cũng train → tự tìm hướng tốt.
- **Forward transfer**: 0 (random init).
- **Rủi ro**: `trans_input` dùng để route trong training; nếu bỏ GPM gradient projection, trans_input evolve tự do → training-time routing có thể suboptimal. **NHƯNG**: inference dùng SRT → không ảnh hưởng cuối cùng.

#### Config 4: Full LoRA + SGWI warm-init
```
A = warm-init từ past tasks (SGWI) → trainable
B = warm-init từ past tasks (SGWI) → trainable
```
- **Expressiveness**: Cao nhất + head-start.
- **Forward transfer**: CÓ. Knowledge chuyển giao qua warm init.
- **Cold-start**: KHÔNG CÒN. Bắt đầu gần điểm tốt.
- **Forgetting**: SRT routing handles.
- **Kết hợp tốt nhất**: Plasticity (full LoRA) + Transfer (SGWI) + Anti-forgetting (SRT routing).

#### Config 5: Freeze A (SGWI warm-init) + B from zeros
```
A = SGWI warm-init từ past tasks → frozen
B = zeros → trainable
Không null-space
```
- **Expressiveness**: Trung bình-cao. A có "hướng" tốt từ task tương tự → B học trong subspace có ý nghĩa.
- **So với Config 2**: Cùng frozen A, cùng B=0, nhưng A direction là SGWI (informed) thay vì kaiming (random).
- **Forward transfer**: Có (qua A direction). A "chỉ đường" cho B dựa trên task tương tự.
- **Cold-start**: Một phần giải quyết (A warm → B hội tụ nhanh hơn trong hướng đúng).
- **Ưu điểm**: Cùng số trainable params như hiện tại → dễ so sánh fair.
- **Đáng test**: ✅ RẤT ĐÁNG — đây là cách đơn giản nhất thể hiện giá trị SGWI.

#### Config 6: SGWI-A trainable + B from zeros
```
A = SGWI warm-init → trainable
B = zeros → trainable
Không null-space
```
- **Expressiveness**: Cao. A có head-start VÀ có thể tinh chỉnh thêm.
- **So với Config 3**: Cùng full LoRA, nhưng A bắt đầu từ SGWI (informed) thay vì kaiming (random).
- **So với Config 5**: A thêm trainable → linh hoạt hơn, có thể vượt qua SGWI init nếu cần.
- **Forward transfer**: Có (qua A init). A refinable → adapt tốt hơn.
- **Cold-start**: Giải quyết cho A. B=0 nhưng A warm giúp convergence.
- **Đáng test**: ✅ ĐÁNG — test xem SGWI + trainable A có synergy không.

### 2.3 Tóm tắt so sánh đầy đủ (6 configs)

| # | A init | A train | B init | Null-space | Plasticity | Transfer | Cold-start |
|---|--------|---------|--------|-----------|-----------|----------|------------|
| 1 | kaiming→null | frozen | zeros | YES | ❌ Thấp | ❌ | ❌ Nặng |
| 2 | kaiming | frozen | zeros | NO | ⚠️ TB | ❌ | ⚠️ Có |
| 3 | kaiming | **train** | zeros | NO | ✅ Cao | ❌ | ⚠️ Nhẹ |
| 5 | **SGWI** | frozen | zeros | NO | ⚠️ TB-cao | ✅ Qua A | ⚠️ Nhẹ |
| 6 | **SGWI** | **train** | zeros | NO | ✅ Cao | ✅ Qua A | ⚠️ Nhẹ |
| 4 | **SGWI** | **train** | **SGWI** | NO | ✅ Cao | ✅ Full | ✅ Giải quyết |

### 2.4 Chuỗi ablation và câu hỏi mỗi cặp trả lời

```
Config 1 → 2: Bỏ null-space có giúp gì? (isolate null-space effect)
Config 2 → 5: SGWI-A có tốt hơn random-A? (isolate SGWI effect on A)
Config 2 → 3: Unfreeze A có giúp gì? (isolate A-training effect)
Config 5 → 6: SGWI-A + trainable có tốt hơn SGWI-A frozen? (SGWI + training synergy)
Config 3 → 6: SGWI-A init giúp gì thêm khi full LoRA? (SGWI marginal value)
Config 6 → 4: SGWI-B init thêm có giúp gì? (B warm-init marginal value)
```

**Khuyến nghị thứ tự ưu tiên** (từ ít effort → nhiều, dừng sớm nếu đủ insight):
1. **Config 2** (bỏ null-space, giữ nguyên phần còn lại) → establish new baseline
2. **Config 5** (SGWI-A frozen) → test SGWI value với ít thay đổi nhất
3. **Config 3** (full LoRA random) → test plasticity gain
4. **Config 6** (SGWI-A trainable + B=0) → test synergy
5. Config 4 (full SGWI) → maximum configuration

---

## 3. Routing: Training vs Inference

**Phát hiện quan trọng từ code:**

### Training time:
```python
# t5_gainlora_inflora.py: T5Stack.forward()
# Routing qua learned MLP: trans_input + prompt_key
avg_input = mean_pool(input_embeds)
x = trans_input(avg_input)            # MLP projection  
attn = cosine_sim(x, prompt_key)      # attention weights
lora_output = Σ attn[i] * adapter_i   # weighted combination
```

### Inference time (SRT):
```python
# cl_trainer_srt.py: predict()
srt_weights = srt_router.route(input)  # SRT statistical routing
model.encoder.override_attn_weights = srt_weights  # OVERRIDE learned routing
# Model forward uses override_attn_weights instead of trans_input routing
```

**Kết luận**: 
- trans_input/prompt_key vẫn cần cho TRAINING-TIME routing
- SRT override tại INFERENCE → kết quả cuối cùng do SRT quyết định
- Bỏ GPM gradient projection trên trans_input/prompt_key: training routing có thể thay đổi nhưng inference không ảnh hưởng

---

## 4. Hướng Đi Đề Xuất

### 4.1 Ablation Study (cần chạy)

**Phase A: SRT-Clean ablation** (bỏ GPM khỏi SRT, test 3 configs)

| Arm | A init | A train | B init | B train | GPM/null-space |
|-----|--------|---------|--------|---------|---------------|
| A1 | kaiming | frozen | zeros | yes | ❌ NONE |
| A2 | kaiming | **yes** | zeros | yes | ❌ NONE |
| A3 | SGWI warm | **yes** | SGWI warm | yes | ❌ NONE |

Chạy trên pipeline dài (15 task order 1 hoặc order 3) để thấy rõ hiệu ứng.

### 4.2 Implementation cần làm

1. **SRT_Trainer mới** (hoặc flag `--srt_clean`):
   - Override `get_reg_matrix()` → KHÔNG project null-space, chỉ set `_cur_task=0`
   - Override `_inner_training_loop()` → KHÔNG GPM gradient projection
   
2. **Unfreeeze lora_A** (flag `--train_lora_a`):
   - Trong `run_t5.py`, thêm `lora_A.requires_grad = True` cho config 3,4

3. **SGWI cho cả A và B** (đã có framework, chỉ cần mở rộng)

### 4.3 Story line cho paper

**C1 (SRT)**: 
- Claim: SRT thay thế GPM routing → non-parametric, statistical
- **Cần thêm**: SRT cũng giải phóng khỏi null-space constraint → tăng plasticity dramatically
- Evidence: CB từ 3.57% → X% (sau khi bỏ null-space)

**C2 (SGWI)**:
- Claim: Bỏ null-space tạo khoảng trống cho warm-init
- SGWI chuyển giao knowledge từ similar tasks (SRT-weighted)
- Evidence: SGWI > random init khi chạy full pipeline

**Unified story**: GPM = {routing + constraint}. SRT replaces routing. Removing constraint frees plasticity. SGWI fills the knowledge transfer gap.

---

## 5. Tài Liệu Tham Khảo

- **LoRA** (Hu et al., 2022): Freeze A (kaiming), train B (zeros). Single-task.
- **InfLoRA** (Liang et al., 2024): A → null-space projection, frozen. B trained. Continual.
- **O-LoRA** (Wang et al., 2023): Similar null-space idea.
- **LoRA+** (Hayou et al., 2024): Different learning rates for A and B improves training.
- **rsLoRA** (Kalajdzievski, 2023): Scaling factor `1/√r` for large rank.
- **AdaLoRA** (Zhang et al., 2023): Dynamic rank allocation via SVD.
- **DyLoRA** (Valipour et al., 2022): Training across ranks simultaneously.
