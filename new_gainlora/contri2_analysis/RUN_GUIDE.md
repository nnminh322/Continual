# 🚀 CONTRI2 Run Guide: GPU Server — SGWI Isolated Tests

> **Model: `google/flan-t5-large`** (d_model=1024, 24 layers, ~770M params)
> Compatible patches for transformers ≥ 4.40 are applied automatically.

---

## Quick Start

```bash
# ── 1. Setup ────────────────────────────────────────────────────────────────
cd ~/minhnguyen/test_model/Continual/new_gainlora/contri2_analysis

# ── 2. Phase 0: Tests 1-3 (~45–90 min on t5-large) ──────────────────────
# First run: extracts embeddings + runs tests
nohup python run_contri2_extended.py --phase 0 \
    --model-name google/flan-t5-large \
    > results/phase0_t5large_log.txt 2>&1 &

# ── 3. Check progress ────────────────────────────────────────────────────────
tail -f results/phase0_t5large_log.txt

# ── 4. Phase 1: Tests 6-7 (~30 min) ──────────────────────────────────────────
nohup python run_contri2_extended.py --phase 1 \
    --model-name google/flan-t5-large \
    > results/phase1_t5large_log.txt 2>&1 &

# ── 5. Phase 2: Tests 4-5 (after CL run with SGWI checkpoints) ─────────────
nohup python run_contri2_extended.py --phase 2 \
    --model-name google/flan-t5-large \
    --ckpt-dir ../logs_and_outputs/long_order3_t5_srt_hard/outputs \
    > results/phase2_t5large_log.txt 2>&1 &
```

---

## Phase Details

### Phase 0 — Tests 1, 2, 3 (~45–90 min on t5-large)

**First run:** Extracts frozen embeddings for all 15 tasks (d=1024) → cached.
**Subsequent runs:** Use cached embeddings, only run tests.

```bash
# All 15 tasks
python run_contri2_extended.py --phase 0 --model-name google/flan-t5-large

# Fast: only specific tasks
python run_contri2_extended.py --phase 0 --tasks cb,rte,mnli

# Individual tests
python run_contri2_extended.py --test 1 --model-name google/flan-t5-large   # Zero-shot
python run_contri2_extended.py --test 2 --model-name google/flan-t5-large   # Few-shot
python run_contri2_extended.py --test 3 --model-name google/flan-t5-large   # Ablation
```

### Phase 1 — Tests 6, 7 (~30 min)

```bash
python run_contri2_extended.py --phase 1 --model-name google/flan-t5-large
```

### Phase 2 — Tests 4, 5 (~20 min, **requires CL checkpoints**)

```bash
# First run a full CL training run with SGWI patch:
python patch_sgwi_into_run_t5.py --apply   # patches run_t5.py (backup: run_t5.py.bak_sgwi)
bash gen_script_long_order3_t5_srt_hard.sh google/flan-t5-large

# Then run Phase 2 with checkpoint path:
python run_contri2_extended.py --phase 2 \
    --model-name google/flan-t5-large \
    --ckpt-dir ../logs_and_outputs/long_order3_t5_srt_hard/outputs
```

---

## Patch: SGWI into run_t5.py

```bash
# Apply patch (creates backup)
python patch_sgwi_into_run_t5.py --apply

# Preview changes without applying
python patch_sgwi_into_run_t5.py --dry-run

# Revert to original
python patch_sgwi_into_run_t5.py --revert
```

After patching, the full CL pipeline automatically uses SGWI (SVD Fusion Init) when:
- `--use_srt_router` is set AND
- `cur_task_id > 0` AND
- `--use_sgwi` is True (default)

---

## Compatibility Fixes (Applied Automatically)

The code automatically handles missing/moved modules in newer `transformers`:

| Missing Module | Fix |
|---|---|
| `loralib` | Stub provided |
| `transformers.utils.model_parallel_utils` | Stub provided |
| `find_pruneable_heads_and_indices` | No-op stub |
| `ipdb` | Dummy stub |

No manual installation needed.

---

## Troubleshooting

### "CUDA out of memory" on t5-large
```bash
# Switch to t5-base for debugging
python run_contri2_extended.py --phase 0 --model-name google/flan-t5-base

# Or reduce batch size in contri2_utils.py train_lora_isolated()
```

### Wrong embedding dimensions (512 vs 1024)
```bash
# Clear cached embeddings (wrong-dim from previous run)
rm -f results/cache/emb_*.npy

# Will re-extract with correct dimension for current model
```

### "ModuleNotFoundError: No module named 'srt_router'"
```bash
# srt_router.py lives in ../src/
export PYTHONPATH="$(pwd)/../src:$PYTHONPATH"
```

### Cached checkpoints interfering
```bash
# Clear only checkpoints (keep embeddings)
rm -f results/cache/ckpt_*.pt
```

---

## Test Matrix

| Test | What | Time (t5-large) | Needs Checkpoints |
|------|------|-----------------|-----------------|
| **1** | Zero-shot transfer | ~10 min | ❌ |
| **2** | Few-shot convergence | ~20 min | ❌ |
| **3** | Ablation (R/NTI/SFI) | ~30 min | ❌ |
| **6** | τ sensitivity sweep | ~20 min | ❌ |
| **7** | Negative transfer detection | ~15 min | ❌ |
| **4** | H1: d_SRT ∝ d_LoRA | ~5 min | ✅ |
| **5** | Real vs Proxy SGWI | ~20 min | ✅ |

---

## Results

All results saved to `results/contri2_extended_results.json`.

```bash
# Pretty-print
python -c "
import json
with open('results/contri2_extended_results.json') as f:
    r = json.load(f)
for k, v in r.items():
    print(f'=== {k} ===')
    if isinstance(v, dict):
        for tk, tv in v.items():
            if isinstance(tv, dict) and tv.get('status') not in ('completed',):
                continue
            if isinstance(tv, dict) and 'summary' in tv:
                print(f'  {tk}: Δ={tv[\"summary\"]}')
            elif isinstance(tv, dict) and tv.get('delta') not in (None, ''):
                print(f'  {tk}: Δ={tv[\"delta\"]:+.2f}%')
"
```

## Pipeline Summary

```
┌─────────────────────────────────────────────────────────┐
│  CONTRI2 Tests (flan-t5-large)                        │
│                                                         │
│  Phase 0 ─── Tests 1-3 ─── Isolated: AP improvement?   │
│    Test 1: Zero-shot (SGWI vs Random baseline)         │
│    Test 2: Few-shot convergence (5 epochs)              │
│    Test 3: Ablation R/NTI/SFI                         │
│                          ↓                            │
│  Phase 1 ─── Tests 6-7 ─── Robustness:                │
│    Test 6: τ sensitivity (median is robust?)           │
│    Test 7: Negative transfer (SRT guides correctly?)  │
│                          ↓                            │
│  Phase 2 ─── Tests 4-5 ─── With Real Checkpoints:      │
│    Test 4: H1 correlation d_SRT ∝ ||ΔW||_F            │
│    Test 5: Real LoRA vs Proxy (quality of μ·μᵀ proxy?)│
│                          ↓                            │
│  Full CL + SGWI ── AP > 78.01, Fgt < 0.34?            │
└─────────────────────────────────────────────────────────┘
```
