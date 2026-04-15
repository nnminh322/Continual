# 🚀 CONTRI2 Run Guide: Local → GPU Server Workflow

## Quick Reference

```bash
# LOCAL: commit & push
git add -A && git commit -m "contri2: add extended SGWI tests (4-7)" && git push

# SERVER: pull & run
cd ~/Continual && git pull
cd new_gainlora/contri2_analysis

# Phase 0: Quick proxy tests (~30 min GPU)
python run_contri2_extended.py --phase 0

# Phase 1: Extended tests (~45 min GPU)
python run_contri2_extended.py --phase 1

# Phase 2: Real LoRA tests (after CL run)
python run_contri2_extended.py --phase 2 --ckpt-dir ../logs_and_outputs/long_order3_t5_srt_hard/outputs

# Fast test on 3 tasks only (~10 min)
python run_contri2_extended.py --phase 0 --tasks cb,rte,mnli
```

---

## Step-by-Step Workflow

### 1. LOCAL: Commit & Push

```bash
cd /Users/nnminh322/Desktop/personal/Continual

# Check what's changed
git status

# Add all contri2 files
git add new_gainlora/contri2_analysis/
git add new_gainlora/CLAUDE.md
git add isolate_hypothesis_testing.md
git add SRT_WARM_INIT_PROPOSAL.md

# Commit
git commit -m "contri2: add SGWI isolated tests (Tests 1-7)

- Tests 1-3: proxy-based (zero-shot, few-shot, ablation)
- Test 4: H1 validation (d_SRT vs ||ΔW||_F correlation)
- Test 5: Real SGWI with actual LoRA checkpoints
- Test 6: τ sensitivity sweep
- Test 7: Negative transfer detection"

# Push
git push origin main
```

### 2. SERVER: Pull & Setup

```bash
# SSH to GPU server
ssh your_server

# Navigate to project
cd ~/Continual   # or wherever your repo is

# Pull latest
git pull origin main

# Navigate to contri2
cd new_gainlora/contri2_analysis

# Check Python environment
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}')"
python -c "import transformers; print(f'transformers {transformers.__version__}')"

# Install scipy if missing
pip install scipy --quiet
```

### 3. SERVER: Run Tests

#### Phase 0 — Quick Proxy Validation (~30 min)

```bash
# All 15 tasks, Tests 1-3
python run_contri2_extended.py --phase 0

# Or fast: only CB + 2 related NLI tasks (~10 min)
python run_contri2_extended.py --phase 0 --tasks cb,rte,mnli

# Or individual tests
python run_contri2_extended.py --test 1    # Zero-shot only
python run_contri2_extended.py --test 2    # Few-shot convergence only
python run_contri2_extended.py --test 3    # Ablation only
```

#### Phase 1 — Extended Validation (~45 min)

```bash
# τ sensitivity + negative transfer
python run_contri2_extended.py --phase 1

# Or individual
python run_contri2_extended.py --test 6    # τ sweep
python run_contri2_extended.py --test 7    # Negative transfer
```

#### Phase 2 — Real LoRA (after full CL run)

```bash
# First: need CL checkpoints from a completed training run
# Check if checkpoints exist:
ls ../logs_and_outputs/long_order3_t5_srt_hard/outputs/

# Then run with checkpoint path
python run_contri2_extended.py --phase 2 \
    --ckpt-dir ../logs_and_outputs/long_order3_t5_srt_hard/outputs
```

#### Run with nohup (background, don't lose on disconnect)

```bash
# Recommended: use nohup + redirect output
nohup python run_contri2_extended.py --phase 0 \
    > results/phase0_log.txt 2>&1 &

# Check progress
tail -f results/phase0_log.txt

# Or use screen/tmux
tmux new -s contri2
python run_contri2_extended.py --phase 0
# Ctrl+B, D to detach
# tmux attach -t contri2 to reattach
```

### 4. SERVER: Check Results

```bash
# Results saved as JSON
cat results/contri2_extended_results.json | python -m json.tool | head -100

# Quick summary
python -c "
import json
with open('results/contri2_extended_results.json') as f:
    r = json.load(f)
for test_name, test_data in r.items():
    print(f'\n=== {test_name} ===')
    if isinstance(test_data, dict):
        for task, res in test_data.items():
            if isinstance(res, dict) and 'delta' in res:
                print(f'  {task}: Δ = {res[\"delta\"]:+.2f}%')
"
```

### 5. SERVER → LOCAL: Pull Results

```bash
# SERVER: commit results
cd ~/Continual
git add new_gainlora/contri2_analysis/results/
git commit -m "contri2: Phase 0 results"
git push

# LOCAL: pull results
cd /Users/nnminh322/Desktop/personal/Continual
git pull
cat new_gainlora/contri2_analysis/results/contri2_extended_results.json
```

---

## Test Matrix & Expected Results

| Test | What | Time | GPU | Hypothesis |
|------|------|------|-----|-----------|
| **1** | Zero-shot transfer | 5 min | ✅ | H2: SGWI > random at init |
| **2** | Few-shot convergence | 15 min | ✅ | H2: SGWI converges faster |
| **3** | Ablation (R/NTI/SFI) | 15 min | ✅ | H3: SFI > NTI > Random |
| **6** | τ sensitivity | 20 min | ✅ | H8: τ_median robust |
| **7** | Negative transfer | 15 min | ✅ | SRT guides correctly |
| **4** | H1 correlation | 5 min | ❌ | H1: d_SRT ∝ d_LoRA |
| **5** | Real SGWI vs proxy | 20 min | ✅ | Real ≥ Proxy > Random |

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'srt_router'"
```bash
# srt_router.py is in ../src/
export PYTHONPATH=$PYTHONPATH:$(pwd)/../src
```

### "CUDA out of memory"
```bash
# Use smaller model
python run_contri2_extended.py --phase 0 --model-name google/flan-t5-small

# Or reduce batch size in contri2_utils.py train_lora_isolated()
```

### "FileNotFoundError: CL_Benchmark"
```bash
# Ensure benchmark data exists
ls ../CL_Benchmark/Long_Sequence/
# Should show: agnews/ amazon/ boolq/ cb/ copa/ ...
```

### Cached results interfering
```bash
# Clear cache to re-run
rm -rf results/cache/ckpt_*
# Keep embeddings cache (expensive to recompute)
```
