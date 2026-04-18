# Run Guide: SRT + SGWI + Dual Fisher

## Tổng quan 3 cờ chính

| Flag | Default | Ý nghĩa |
|------|---------|---------|
| `--sgwi True` | True | **SGWI warm-init**: Khởi tạo LoRA từ weighted combination past adapters (cả A+B) |
| `--sgwi False` | | **Full LoRA**: Standard kaiming init, cả A+B trainable, không warm-init |
| `--dual_fisher True` | False | **Bật Dual Fisher**: L2 penalty quanh past θ* (default λ=0.01) |
| `--dual_fisher False` | | **Tắt Dual Fisher**: Không regularization |
| `--lambda_emb 0.05` | 0.01 auto | Override λ strength (chỉ có tác dụng khi `--dual_fisher True`) |

## 4 Configurations có thể chạy

### Config 1: SGWI only (no Dual Fisher) — **Recommended mặc định**
```bash
SRT_FLAGS="--use_srt_router True --sgwi True --dual_fisher False --srt_shrink False --srt_metric_mode hard --srt_max_emb_samples 200"
```
- Warm-init A+B từ past LoRAs, train tự do
- Forward transfer qua initialization, forgetting prevention qua SRT routing

### Config 2: SGWI + Dual Fisher
```bash
SRT_FLAGS="--use_srt_router True --sgwi True --dual_fisher True --srt_shrink False --srt_metric_mode hard --srt_max_emb_samples 200"
```
- Warm-init A+B + L2 penalty giữ params gần warm-init solution
- λ=0.01 tự động, override bằng `--lambda_emb 0.05`

### Config 3: SGWI + Dual Fisher (custom λ)
```bash
SRT_FLAGS="--use_srt_router True --sgwi True --dual_fisher True --lambda_emb 0.05 --srt_shrink False --srt_metric_mode hard --srt_max_emb_samples 200"
```

### Config 4: Full LoRA baseline (no SGWI, no Dual Fisher)
```bash
SRT_FLAGS="--use_srt_router True --sgwi False --dual_fisher False --srt_shrink False --srt_metric_mode hard --srt_max_emb_samples 200"
```
- Standard LoRA init (kaiming A, zero B), cả A+B trainable
- Không warm-init, không regularization
- Dùng để so sánh: hiệu quả thuần SRT routing không có forward transfer

## Chạy Full Pipeline (15 tasks)

### Cú pháp
```bash
bash gen_script_<order>_t5_srt.sh <GPU_ID> <MODEL_PATH> [--sgwi True/False] [--dual_fisher True/False] [--lambda_emb <value>]
```

### Config 1: SGWI only (default — recommended)
```bash
cd new_gainlora
bash gen_script_long_order4_t5_srt.sh 0 google/flan-t5-large
# Equivalent to: --sgwi True --dual_fisher False
```

### Config 2: SGWI + Dual Fisher (λ=0.01 auto)
```bash
bash gen_script_long_order4_t5_srt.sh 0 google/flan-t5-large --dual_fisher True
```

### Config 3: SGWI + Dual Fisher (custom λ)
```bash
bash gen_script_long_order4_t5_srt.sh 0 google/flan-t5-large --dual_fisher True --lambda_emb 0.05
```

### Config 4: Full LoRA baseline (no SGWI, no Dual Fisher)
```bash
bash gen_script_long_order4_t5_srt.sh 0 google/flan-t5-large --sgwi False
```

### Long Sequence Order 3 (yelp → wic)
```bash
bash gen_script_long_order3_t5_srt.sh 0 google/flan-t5-large                           # SGWI only
bash gen_script_long_order3_t5_srt.sh 0 google/flan-t5-large --dual_fisher True         # + Dual Fisher
```

## Cách hoạt động

### Training flow cho mỗi task:
```
1. Load previous LoRA weights + SRT signatures
2. [SGWI] Tính SRT similarity → softmax weights
3. [SGWI] Warm-init: A_new = SVD(Σ w_i × B_i @ A_i), B_new = ΔW @ A_new^+
4. Train LoRA (A+B trainable)
   - [Dual Fisher] Mỗi step: loss = task_loss + λ × Σ w_s × ||θ - θ*_s||²
5. [SRT] Compute {μ_t, Σ_t} từ frozen encoder
6. Save LoRA weights + SRT signatures
```

### Inference flow:
```
Input → Frozen Encoder → h(x)
     → SRT Router: argmin_t d(h(x), {μ_t, Σ_t}) → task_id
     → Select LoRA_t → Apply to backbone → Output
```

## GPU Requirements

| GPU | Batch | Grad Accum | Notes |
|-----|-------|------------|-------|
| T4 (16GB) | 4 | 8 | `--gradient_checkpointing` required |
| A100 (80GB) | 8-16 | 2-4 | No special flags needed |
| 2x T4 | 2 | 4-8 | DataParallel + gradient_checkpointing |

## Xem kết quả
```bash
python score.py <run_name> <run_name>
```
Output: Exact Match per task + Average + Forgetting metrics.
