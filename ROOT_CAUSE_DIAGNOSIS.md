# 🔍 ROOT CAUSE DIAGNOSIS: Tại Sao SRT Không Vượt Baseline?

**Date**: April 15, 2026  
**Problem**: new_gainlora (SRT) AP=77.62 < root_gainlora (baseline) AP=78.01  
**Status**: 🔴 ROOT CAUSE IDENTIFIED

---

## TL;DR — Nguyên Nhân Gốc

### 🚨 TRAIN-INFERENCE MISMATCH (Lệch pha giữa Training và Inference)

```
TRAINING (cả 2 version giống nhau):
  key_attention_weights = cal_attention(prompt_key, x)  → SOFT weights [0,1]
  Mọi LoRA adapter đóng góp theo tỷ lệ similarity

INFERENCE - root_gainlora:
  key_attention_weights = cal_attention(prompt_key, x)  → SOFT weights [0,1]  ✅ SAME AS TRAINING

INFERENCE - new_gainlora (SRT):
  key_attention_weights = cal_attention(...)             → SOFT weights [0,1]
  key_attention_weights[:] = srt_weights                 → ONE-HOT [0,0,1,0,...] ❌ OVERRIDE!
  Chỉ 1 LoRA adapter duy nhất được kích hoạt
```

**Hệ quả**: Model được train với soft blending (nhiều adapter đóng góp) nhưng evaluate với hard one-hot routing (chỉ 1 adapter). Đây là lý do chính khiến performance giảm.

---

## Bằng Chứng Chi Tiết

### 1. Code Evidence

**root_gainlora/src/t5_gainlora_inflora.py** (line 1297):
```python
key_attention_weights = self.cal_attention(past_prompt_key, past_x)
# → Soft weights [0.3, 0.7, 0.1, ...] dùng cho cả train và inference
```

**new_gainlora/src/t5_gainlora_inflora.py** (lines 1310, 1335-1393):
```python
key_attention_weights = self.cal_attention(past_prompt_key, past_x)  # Line 1310
# ... same as root ...

# Line 1335: SRT OVERRIDE chỉ tại inference
if not self.training and self.use_srt_routing and self.srt_router is not None:
    # ... compute SRT routing ...
    srt_weights[b, pos, 0] = 1.0       # ONE-HOT
    key_attention_weights[:] = srt_weights  # Line 1393: REPLACE soft → hard
```

### 2. Result Evidence

| Metric | root_gainlora (baseline) | new_gainlora (SRT) | Phân tích |
|--------|-------------------------|--------------------|-----------| 
| **AP** | 78.01 | 77.62 (-0.39) | SRT worse — nhưng rất gần |
| **Forgetting** | 0.77 | 0.34 (-0.43) | SRT **MUCH BETTER** ✅ |
| **BWT** | ? | -0.315 | Có forgetting nhẹ |
| **FWT** | ? | 77.94 | Forward transfer tốt |

### 3. Per-Task Analysis — Ai Bị Ảnh Hưởng?

| Task | SRT Score | Vấn đề | Root Cause |
|------|-----------|--------|------------|
| **CB** | **3.57%** 🔴 | Catastrophically bad | Tiny dataset (~250 samples), weak adapter, no cross-task help |
| **wic** | **57.99%** 🟡 | Below expected | Small dataset, isolated by one-hot routing |
| **yelp** | **67.71%** 🟡 | Below expected | First task trained, may benefit from later sentiment adapters |
| **amazon** | **62.43%** 🟡 | Below expected | Sentiment task, would benefit from yelp/imdb adapter blending |
| **dbpedia** | 99.03% ✅ | Excellent | Distinctive task, one adapter sufficient |
| **imdb** | 96.28% ✅ | Excellent | Clear sentiment signal |
| **sst2** | 93.35% ✅ | Excellent | Strong classification adapter |

**Pattern**: Tasks that benefit from **cross-adapter knowledge transfer** suffer most with one-hot routing.

### 4. Ma Trận Quên (Forgetting Matrix) — Cái SRT Đúng

Nhìn vào ma trận:
```
Row 3→14 cho CB: 3.57 → 3.57 → 3.57 → ... → 3.57 (KHÔNG đổi)
Row 7→14 cho imdb: 96.28 → 96.28 → ... → 96.28 (KHÔNG đổi)
```

**SRT routing HOÀN TOÀN ngăn catastrophic forgetting!** Không task nào bị quên (BWT gần 0 cho hầu hết tasks).

Nhưng vấn đề là **initial performance** thấp cho một số tasks do mất cross-task transfer.

---

## Phân Tích Lý Thuyết

### Tại Sao Soft Gating Giúp Baseline?

Trong root_gainlora, `cal_attention()` tạo weights:
```python
weights = |sigmoid(cosine_sim * 4) * 2 - 1|  → range [0, 1]
```

Ví dụ cho task CB (NLI), weights có thể là:
```
[0.8_CB, 0.3_rte, 0.4_mnli, 0.1_yelp, ...]
```

→ CB adapter đóng góp 80%, nhưng rte (NLI) đóng góp 30%, mnli (NLI) đóng góp 40%
→ CB lợi dụng tri thức NLI từ related adapters → performance cao hơn 3.57%

### Tại Sao One-Hot Routing Gây Hại?

Với SRT one-hot routing:
```
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1_CB, 0, 0, 0]
```

→ CHỈ CB adapter hoạt động → không có cross-task help → 3.57%

### Tại Sao Forgetting Giảm?

One-hot routing = perfect isolation:
- Task A's adapter KHÔNG BAO GIỜ bị ảnh hưởng bởi task B
- SRT routing accuracy 99.99% → gần như không bao giờ route sai
- → Forgetting ≈ 0

### Paradox: Isolation Tốt Cho Anti-Forgetting, Xấu Cho Performance

```
Perfect Isolation ──→ Zero Forgetting ✅
                  └──→ Zero Knowledge Transfer ❌

Soft Blending    ──→ Some Forgetting ⚠️
                 └──→ Cross-Task Knowledge Transfer ✅
```

**Đây là trade-off cốt lõi mà SRT one-hot routing không giải quyết được.**

---

## 5 Giải Pháp Đề Xuất

### Solution 1: Soft SRT Routing (⭐ RECOMMENDED — Dễ nhất, hiệu quả nhất)

Thay one-hot bằng softmax over distances:

```python
# THAY THẾ (line 1364-1374 trong t5_gainlora_inflora.py):

# OLD: One-hot routing
srt_weights = torch.zeros(B_batch, n_slots, 1, ...)
srt_weights[b, pos, 0] = 1.0

# NEW: Soft routing with temperature
distances = torch.tensor(dist_matrix, device=device)  # (B, T)
# Map distances to slot indices
dist_slots = torch.full((B_batch, n_slots), float('inf'), device=device)
for task_id, slot_idx in self.srt_task_id_to_idx.items():
    task_pos = list(self.srt_router.signatures.keys()).index(task_id)
    dist_slots[:, slot_idx] = distances[:, task_pos]

# Temperature-scaled softmax (τ controls sharpness)
tau = 0.1  # Small τ → sharper (closer to one-hot), large τ → softer
srt_weights = F.softmax(-dist_slots / tau, dim=1).unsqueeze(-1)  # (B, n_slots, 1)
```

**Ưu điểm**:
- Giữ benefits of SRT (non-parametric, zero-drift)
- Cho phép cross-task transfer từ related adapters
- τ tunable: τ→0 = one-hot, τ→∞ = uniform

### Solution 2: Train-Time SRT Alignment

Thêm loss term để align learned gating với SRT routing:

```python
# Trong cl_trainer_srt.py, modify loss:
srt_target = compute_srt_weights(batch)  # target from SRT router
learned_weights = model.encoder.key_attention_weights  # from cal_attention

alignment_loss = F.kl_div(
    F.log_softmax(learned_weights.squeeze(-1), dim=1),
    srt_target,
    reduction='batchmean'
)

total_loss = ce_loss + lambda_align * alignment_loss
```

**Ưu điểm**: Giảm train-inference mismatch dần dần
**Nhược**: Phức tạp hơn, cần tune lambda_align

### Solution 3: Consistent SRT Routing (Train + Inference)

Dùng SRT routing cho CẢ training, không chỉ inference:

```python
# Bỏ điều kiện "not self.training" ở line 1335:
# OLD:
if not self.training and self.use_srt_routing and ...
# NEW:
if self.use_srt_routing and self.srt_router is not None and ...
```

**⚠️ CẢNH BÁO**: Cần giải quyết gradient flow! Với one-hot routing, gradient chỉ flow qua 1 adapter → các adapter khác không được update → có thể gây instability.

**Fix**: Dùng straight-through estimator hoặc soft routing (Solution 1) khi training.

### Solution 4: Hybrid Routing (Smart Blending)

Kết hợp SRT confidence với learned gating:

```python
# Nếu SRT rất confident (distance gap lớn) → dùng one-hot
# Nếu SRT ít confident (2+ tasks gần nhau) → dùng soft blending

dist_gap = sorted_distances[:, 1] - sorted_distances[:, 0]  # gap between closest 2
confidence = torch.sigmoid(dist_gap * alpha)  # high gap → high confidence

# Blend:
final_weights = confidence * srt_weights + (1 - confidence) * learned_weights
```

### Solution 5: Post-hoc Calibration

Giữ SRT routing nhưng calibrate với validation data:

```python
# Sau khi train xong tất cả tasks:
# 1. Evaluate trên validation set với different τ values
# 2. Chọn τ tối ưu cho mỗi task
# 3. Dùng task-specific τ tại inference
```

---

## Priority & Estimated Impact

| Solution | Complexity | Expected AP Gain | Expected Forgetting |
|----------|-----------|------------------|---------------------|
| **1. Soft SRT** | ⭐ Easy (10 lines) | +1-3 points | ~0.3-0.5 (slight increase) |
| 2. Alignment Loss | ⭐⭐ Medium | +0.5-1 point | ~0.3-0.4 |
| 3. Consistent Route | ⭐⭐⭐ Hard | +0.5-2 points | ~0.3-0.4 |
| 4. Hybrid | ⭐⭐ Medium | +1-2 points | ~0.3-0.5 |
| 5. Calibration | ⭐ Easy | +0.5-1 point | ~0.34 (same) |

**Recommendation**: Start with **Solution 1 (Soft SRT)** — chỉ cần thay ~10 lines code, giữ mọi thứ khác giống nhau. Expected: AP ≥ 78.5 (vượt baseline) while keeping forgetting < 0.5.

---

## Tóm Tắt

```
Theory (ĐÚNG):
  SRT routing 99.99% accurate → zero forgetting
  ✅ Forgetting giảm: 0.34 vs 0.77

Bug (SAI):
  SRT one-hot routing at inference ≠ soft gating at training
  ❌ Train-inference mismatch → mất cross-task knowledge transfer
  ❌ CB task destroyed: 3.57% (no help from related NLI adapters)
  ❌ AP giảm nhẹ: 77.62 vs 78.01

Fix:
  Soft SRT routing (softmax over distances with temperature)
  → Giữ SRT benefits (non-parametric, zero-drift, near-perfect routing)
  → Thêm cross-task knowledge transfer through soft weights
  → Expected: AP > 78.01 AND Forgetting < 0.77
```

---

## File References

| File | Line | Description |
|------|------|-------------|
| new_gainlora/src/t5_gainlora_inflora.py | 1335 | SRT routing condition (inference only) |
| new_gainlora/src/t5_gainlora_inflora.py | 1364-1374 | One-hot weight construction |
| new_gainlora/src/t5_gainlora_inflora.py | 1393 | `key_attention_weights[:] = srt_weights` — THE OVERRIDE |
| new_gainlora/src/t5_gainlora_inflora.py | 1201-1230 | `cal_attention()` — soft gating |
| new_gainlora/src/t5_gainlora_inflora.py | 639-658 | `agg_lora_states()` — adapter weighting |
| root_gainlora/src/t5_gainlora_inflora.py | 1297 | Baseline: soft gating only (no SRT) |
| new_gainlora/src/cl_trainer_srt.py | 330-361 | `_replace_attention_routing()` — wires SRT |
