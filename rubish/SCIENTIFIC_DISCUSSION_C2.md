# Scientific Discussion: GainLoRA Architecture Evolution

## 1. Vấn đề cốt lõi của GainLoRA gốc (root_gainlora)

### 1.1. GPM (Gradient Projection Memory) có vấn đề gì?

GainLoRA gốc sử dụng **InfLoRA + GPM** với cơ chế:
1. **InfLoRA null-space init**: Khởi tạo `lora_A` trong không gian trực giao (null-space) của gradient tasks cũ
2. **GPM representation**: Sau mỗi task, thu thập ma trận biểu diễn gradient và chiếu gradient task mới vào null-space

**Vấn đề chính:**
- **GPM feature collection kém chính xác**: `get_matrix3()` thu thập activations từ LoRA layers, nhưng activations này phụ thuộc vào routing → routing sai = GPM sai
- **Circular dependency**: GPM phụ thuộc routing → routing phụ thuộc attention weights → attention weights bị drift qua các tasks → GPM representation bị nhiễu tích lũy
- **Null-space shrinks**: Qua nhiều tasks, null-space co lại → task cuối gần như không có capacity → underfitting nghiêm trọng
- **lora_A frozen**: InfLoRA freeze `lora_A` sau null-space init, chỉ train `lora_B` → hạn chế capacity nghiêm trọng

### 1.2. Thực nghiệm chứng minh

| Metric | GainLoRA gốc (GPM) | SRT (no GPM) |
|--------|-------------------|--------------|
| Forgetting | Giảm nhưng tích lũy | **Gần 0%** |
| Task isolation | Phụ thuộc GPM accuracy | **100% by design** |
| Forward transfer | Có (qua routing) | Giảm (independent) |

## 2. Contribution 1: SRT (Statistical Routing Theory)

### 2.1. Idea cốt lõi
Thay thế **learned MLP router** bằng **non-parametric statistical router**:
- Tính `{μ_t, Σ_t}` từ **frozen** backbone embeddings
- Routing bằng Mahalanobis distance (hoặc ZCA whitening + L2)
- **Zero-drift**: không có learnable parameters → không bị catastrophic forgetting ở router

### 2.2. Tại sao SRT giải quyết forgetting triệt để?
- Router không drift → đúng task → đúng LoRA adapter → **zero interference**
- Thực nghiệm: forgetting **gần 0%** trên cả isolated tests và full 15-task pipeline

### 2.3. Trade-off: Independence vs Knowledge Transfer
> **Vấn đề quan trọng**: SRT tạo task isolation hoàn hảo, nhưng giảm forward transfer.

Khi tasks hoàn toàn independent:
- ✅ **Forgetting = 0**: Mỗi LoRA adapter chỉ serve đúng task của nó
- ❌ **Forward transfer giảm**: Task mới không tận dụng được knowledge từ tasks cũ
- ❌ **Capacity waste**: Mỗi task phải học from scratch

## 3. Contribution 2: SGWI (SRT-Guided Warm Initialization)

### 3.1. Key insight
> SRT giải phóng cơ chế GPM-learn → tạo **khoảng trống** (design space) cho forward transfer mechanism mới.

Thay vì GPM null-space (bị vấn đề circular dependency), ta dùng **SRT signatures** để:
1. **Đo task similarity**: d(t_new, t_old) từ frozen embeddings
2. **Warm-init LoRA**: Khởi tạo LoRA_{new} = Σ w_i × LoRA_i (weighted combination)
3. **Train freely**: Cả `lora_A` và `lora_B` trainable (không frozen bởi null-space)

### 3.2. Hai chế độ (đã implement trong code)

| Flag | Mode | Mechanism |
|------|------|-----------|
| `--sgwi True` | `sgwi_full` | Warm-init A+B từ past LoRAs, cả hai trainable |
| `--sgwi False` | `full_lora` | Standard init, cả A+B trainable (no warm-init) |

### 3.3. Tại sao SGWI tốt hơn GPM?

| Aspect | GPM (cũ) | SGWI (mới) |
|--------|----------|------------|
| Forward transfer source | Gradient null-space (noisy) | SRT similarity (clean) |
| Init mechanism | Null-space projection | Weighted combination |
| Trainable params | Only lora_B | lora_A + lora_B |
| Dependency | Router accuracy | Frozen embeddings (stable) |
| Forgetting protection | GPM (imperfect) | SRT routing (near-perfect) |

### 3.4. Dual Fisher Regularization (optional)
- `λ_emb > 0`: L2 penalty giữ params gần warm-init solution
- Giúp regularize khi task data ít → tránh overfitting

## 4. Architecture Summary (After Refactoring)

```
┌─────────────────────────────────────────────────────────┐
│                    Inference Pipeline                     │
│                                                           │
│  Input → Frozen Encoder → h(x)                           │
│           ↓                                               │
│  SRT Router: argmin_t d(h(x), {μ_t, Σ_t}) → task_id     │
│           ↓                                               │
│  Select LoRA_t → Apply to backbone → Output               │
│                                                           │
├─────────────────────────────────────────────────────────┤
│                    Training Pipeline                      │
│                                                           │
│  Task t arrives:                                          │
│    1. [SGWI] Compute SRT similarity to past tasks         │
│    2. [SGWI] Warm-init LoRA_t = Σ w_i × LoRA_i           │
│    3. Train LoRA_t (A+B trainable) on task t data         │
│    4. [SRT] Compute {μ_t, Σ_t} from frozen embeddings    │
│    5. Store LoRA_t + signature → done                     │
│                                                           │
│  ❌ NO GPM, NO null-space, NO gradient projection         │
│  ❌ NO frozen lora_A                                      │
└─────────────────────────────────────────────────────────┘
```

## 5. Code Changes Summary

### run_t5.py:
- `--sgwi True/False` replaces `--sgwi_mode` (6 modes → 2 modes)
- Always uses `SGWI_DualFisher_Trainer` (no old InfLoRA-only path)
- Removed `get_repsentation()` call for `gainlora_inflora` (no GPM)
- Always unfreezes `lora_A` (both modes need A+B trainable)

### Trainer chain (preserved):
```
GainLoRA_InfLoRA_Trainer  (base, training_step + replay)
  → SRT_Trainer           (+ SRT routing, signature computation)
    → SGWI_DualFisher_Trainer  (+ warm-init, overrides get_reg_matrix)
```

### Key: SGWI trainer's `get_reg_matrix()` dispatches:
- `sgwi_full`: `_init_gpm_attrs_skip()` + SGWI warm-init A+B
- `full_lora`: `_init_gpm_attrs_skip()` + standard kaiming init

Both skip GPM/null-space entirely.
