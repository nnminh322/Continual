# Contribution 2: SGWI + Dual Fisher — Hypothesis Testing

## Tổng quan

**SGWI** (SRT-Guided Warm Initialization): Khởi tạo LoRA mới từ tổ hợp có trọng số SRT của các LoRA cũ tương tự.
**Dual Fisher**: Regularization bảo vệ warm init khỏi bị phá hủy trong quá trình training.

## Cấu trúc thư mục

```
Continual/
├── contri2_analysis/            ← THƯ MỤC NÀY (scripts chạy từ root)
│   ├── README.md                   Hướng dẫn này
│   ├── setup.py                    Patch run_t5.py (apply/revert/check)
│   ├── analyze_results.py          Auto-analysis kết quả
│   ├── run_phase0_mnli.sh          Phase 0: Train MNLI baseline
│   ├── run_phase1_cb_arms.sh       Phase 1: 4-arm CB comparison
│   └── run_phase3_dualfisher_sweep.sh  Phase 3: λ_emb sweep
│
├── new_gainlora/                ← Source code chính
│   └── src/
│       ├── run_t5.py               ← được patch thêm args: --sgwi_mode, --lambda_emb
│       ├── cl_trainer_srt.py       SRT_Trainer gốc (C1)
│       └── sgwi_trainer.py         ← NEW: SGWI_DualFisher_Trainer (C2)
│
└── new_gainlora/contri2_analysis/  (proposal + analysis docs)
    └── C2_PROPOSAL_AND_HYPOTHESIS_TESTING.md
```

---

## Hướng dẫn chạy trên Server (Step-by-step)

### Bước 0: SSH vào server & chuẩn bị environment

```bash
# SSH vào server
ssh user@your-server

# Activate conda env (hoặc venv)
conda activate gainlora   # hoặc tên env của bạn

# Clone/pull repo
cd /path/to/Continual
git pull origin main
```

### Bước 1: Apply patch SGWI vào run_t5.py

```bash
# Từ thư mục gốc Continual/
python contri2_analysis/setup.py --apply

# Kiểm tra patch đã được áp dụng
python contri2_analysis/setup.py --check
# Expected: ✅ run_t5.py is patched with C2 support
```

> **Lưu ý:** Nếu cần revert về bản gốc: `python contri2_analysis/setup.py --revert`

### Bước 2: Chạy kiểm định giả thuyết

**Tất cả các script dưới đây phải chạy từ thư mục `new_gainlora/`:**

```bash
cd new_gainlora
```

#### Phase 0: Train MNLI baseline (~1-2h trên A100, ~3h trên T4)

```bash
# Cú pháp: bash ../contri2_analysis/run_phase0_mnli.sh <GPU_ID> <MODEL_PATH>
bash ../contri2_analysis/run_phase0_mnli.sh 0 google/flan-t5-large

# Hoặc chạy nền (khuyến khích):
nohup bash ../contri2_analysis/run_phase0_mnli.sh 0 google/flan-t5-large > ../contri2_analysis/log_phase0.txt 2>&1 &

# Monitor:
tail -f ../contri2_analysis/log_phase0.txt
```

#### Phase 1: CB 4-arm test (~2h trên A100, ~5h trên T4)

```bash
# Chỉ chạy SAU KHI Phase 0 xong
bash ../contri2_analysis/run_phase1_cb_arms.sh 0 google/flan-t5-large

# Hoặc chạy nền:
nohup bash ../contri2_analysis/run_phase1_cb_arms.sh 0 google/flan-t5-large > ../contri2_analysis/log_phase1.txt 2>&1 &
```

#### Phase 3: Dual Fisher λ sweep (~2h trên A100)

```bash
# Chỉ chạy nếu Phase 1 cho Q1=YES hoặc MARGINAL
bash ../contri2_analysis/run_phase3_dualfisher_sweep.sh 0 google/flan-t5-large

# Hoặc chạy nền:
nohup bash ../contri2_analysis/run_phase3_dualfisher_sweep.sh 0 google/flan-t5-large > ../contri2_analysis/log_phase3.txt 2>&1 &
```

### Bước 3: Xem kết quả

```bash
# Từ new_gainlora/
python ../contri2_analysis/analyze_results.py --output_base logs_and_outputs/c2_hypothesis

# Hoặc từng phase:
python ../contri2_analysis/analyze_results.py --phase 1 --output_base logs_and_outputs/c2_hypothesis
python ../contri2_analysis/analyze_results.py --phase 3 --output_base logs_and_outputs/c2_hypothesis
```

---

## Decision Flow

```
Phase 0 (MNLI)
    ↓
Phase 1 (4-arm CB test)
    ├── Q1=YES (SGWI > InfLoRA by >1pp)  → Phase 3
    ├── Q1=MARGINAL (±1pp)                → Phase 3 (cẩn trọng)
    └── Q1=NO (SGWI < InfLoRA by >1pp)   → STOP ❌
                ↓
Phase 3 (Dual Fisher λ sweep)
    ├── Q3=YES (Fisher adds >0.5pp)       → Phase 4 (full e2e)
    ├── Q3=MARGINAL                       → C2 = SGWI only
    └── Q3=NO                             → C2 = SGWI only (drop Fisher)
```

## GPU Time Estimate

| Phase | A100 (40GB) | T4 (16GB) | Câu hỏi trả lời |
|-------|------------|-----------|-----------------|
| Phase 0 | ~1h | ~3h | — (prerequisites) |
| Phase 1 | ~2h (4 arms × 30min) | ~5h | Q1, Q2, O1 |
| Phase 2 | 0 (from P1 logs) | 0 | Q4 (convergence) |
| Phase 3 | ~2h (4 λ × 30min) | ~4h | Q3, O2 |
| **Total** | **~5h** | **~12h** | |

## Troubleshooting

### "Module not found: sgwi_trainer"
```bash
# Kiểm tra file tồn tại
ls -la new_gainlora/src/sgwi_trainer.py

# Kiểm tra patch
python contri2_analysis/setup.py --check
```

### "MNLI checkpoint not found"
```bash
# Kiểm tra Phase 0 đã chạy xong
ls -la new_gainlora/logs_and_outputs/c2_hypothesis/phase0_mnli/saved_weights/
```

### GPU OOM
```bash
# Script tự detect GPU type, nhưng nếu vẫn OOM:
# Sửa trong script: BSZ=2; GA=16  (giảm batch size, tăng gradient accumulation)
```

### Revert tất cả patches
```bash
python contri2_analysis/setup.py --revert
```

---

Xem `new_gainlora/contri2_analysis/C2_PROPOSAL_AND_HYPOTHESIS_TESTING.md` cho proposal đầy đủ với lý thuyết và phân tích chi tiết.
