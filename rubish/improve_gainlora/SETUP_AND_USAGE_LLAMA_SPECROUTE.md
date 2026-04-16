# Llama SpecRoute on H100: Complete Setup & Usage Guide

## Overview

This guide provides step-by-step instructions to:
1. Setup an isolated Python environment on H100 server
2. Run Llama SpecRoute Continual Learning experiments (Order 1 & 2)
3. Compare results with ROOT Llama GainLoRA baselines
4. Interpret performance metrics (Cl, Fgt, Fwt, Bwt)

**What's being tested:**
- **Model**: Llama-2-7B, Llama-2-13B, Llama-3-8B
- **Benchmark**: SuperNI (15 NLP tasks)
- **Task Orders**: Order 1 (shuffled), Order 2 (shuffled differently)
- **Baseline**: GainLoRA (ROOT implementation in this repo)
- **New Method**: SpecRoute (parameter-free spectral routing)

---

## Part 1: Server Environment Setup (Isolated, No System Conflicts)

### Step 1.1: Create isolated workspace within improve_gainlora/

```bash
cd /path/to/improve_gainlora

# Create a venv in the repo (not system-wide)
python3.10 -m venv venv_llama_specroute

# Activate
source venv_llama_specroute/bin/activate
```

**Why isolated venv?**
- Stays within improve_gainlora/ folder
- No conda base environment conflicts
- Easy to share scripts (just include venv_llama_specroute/)
- Can be deleted/recreated without affecting system

### Step 1.2: Upgrade pip and install core dependencies

```bash
# Always upgrade pip first
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 (H100 standard)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install DeepSpeed (required for multi-GPU distributed training)
pip install deepspeed==0.13.1

# Install HuggingFace transformers (for Llama model loading)
pip install transformers==4.36.0

# Install sentencepiece (required for LlamaTokenizer)
pip install sentencepiece==0.1.99

# Install datasets and evaluation metrics
pip install datasets==2.14.7
pip install nltk==3.8.1
pip install rouge-score==0.1.2
pip install ipdb
# Install tqdm for progress bars
pip install tqdm==4.66.1

# Optional: cupy for GPU-accelerated operations
pip install cupy-cuda12x==12.1.0
```

**Expected installation time**: 5-10 minutes

### Step 1.3: Verify installation

```bash
# Check PyTorch with GPU
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}'); print(f'Current GPU: {torch.cuda.get_device_name(0)}')"

# Check DeepSpeed
python -c "import deepspeed; print(f'DeepSpeed version: {deepspeed.__version__}')"

# Check transformers
python -c "from transformers import LlamaForCausalLM; print('Transformers OK')"
```

**Expected output:**
```
CUDA Available: True
GPU Count: 1  (or more for multi-GPU)
Current GPU: NVIDIA H100 SXM5
DeepSpeed version: 0.13.1
Transformers OK
```

### Step 1.4: Download model weights (if not already cached)

```bash
# Set Hugging Face cache directory (optional, avoids default ~/.cache/)
export HF_HOME=$(pwd)/.hf_cache

# Pre-download Llama-2-7B
python -c "from transformers import LlamaForCausalLM, AutoTokenizer; \
    model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf'); \
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')"

# Pre-download Llama-2-13B (optional, larger)
# python -c "from transformers import LlamaForCausalLM; \
#     model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-13b-hf')"

# Check NLTK data
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('wordnet', quiet=True)"
```

**Expected time**: 10-30 minutes (depends on internet speed and model size)

---

## Part 2: Running Llama SpecRoute Experiments

### Step 2.1: Understand the generated scripts

Two scripts are ready-to-run:

1. **gen_script_superni_order1_llama_specroute.sh** — 15 sequential tasks (different order than order 2)
2. **gen_script_superni_order2_llama_specroute.sh** — 15 sequential tasks (shuffled differently for robustness)

View script structure:
```bash
head -30 gen_script_superni_order1_llama_specroute.sh
```

Key parameters already preset:
- **model_name=specroute** — Uses spectral routing (not GainLoRA)
- **threshold=0.995** — ESA dynamic GPM threshold
- **lora_r=4, lora_alpha=32** — Low-rank adaptation (same as ROOT)
- **max_source_length=1024, max_target_length=50** — Token limits
- **deepspeed stage 2** — Distributed training with gradient checkpointing
- **master_port=49500** — Unique port for distributed communication
- **no data replay** — Pure LoRA continual learning (zero forgetting baseline)

### Step 2.2: Single task test run (2-5 minutes)

Before running full 15 tasks, test a single task:

```bash
# Activate environment
source venv_llama_specroute/bin/activate

# Run only task 1 (quick test)
deepspeed --include localhost:0 --master_port 49500 src/run_llama.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path meta-llama/Llama-2-7b-hf \
   --data_dir CL_Benchmark \
   --task_order task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification \
   --task_config_dir configs/gen_script_superni_order1_llama_configs/task1572_samsum_summary \
   --output_dir logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/1-task1572_samsum_summary \
   --training_epochs 50 \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 4 \
   --lora_r 4 \
   --lora_alpha 32 \
   --threshold 0.995 \
   --model_name specroute \
   --num_train_epochs 50
```

**Expected output:**
```
[2026-03-16 14:32:10] Training task 1/1: task1572_samsum_summary
[2026-03-16 14:32:15]   Loss: 2.345 | Epoch 1/50
[2026-03-16 14:35:42]   Loss: 0.892 | Epoch 50/50
[2026-03-16 14:36:01] Evaluation (ALL tasks):
  - predict_eval_rougeL_for_task1572_samsum_summary: 0.45
[2026-03-16 14:36:02] Saving checkpoint...
[2026-03-16 14:36:05] DONE
```

**If successful**, proceed to full run.

### Step 2.3: Run full Llama SpecRoute Order 1 (6-10 hours on H100)

```bash
source venv_llama_specroute/bin/activate

# Make scripts executable
chmod +x gen_script_superni_order1_llama_specroute.sh

# Run (background with nohup)
nohup bash gen_script_superni_order1_llama_specroute.sh 0 meta-llama/Llama-2-7b-hf > run_order1.log 2>&1 &

# Parameters:
#   $1 = GPU ID (0 for single GPU, or 0,1 for multi-GPU)
#   $2 = Model path or HuggingFace ID

# Monitor progress in real-time
tail -f run_order1.log

# Or check completion
grep -c "DONE" logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/*/trainer_state.json
# Should show: 15 (one per task)
```

**Estimated time**: 6-10 hours (depending on H100 speed, batch size)

### Step 2.4: Run full Llama SpecRoute Order 2 (6-10 hours on H100)

After Order 1 completes:

```bash
source venv_llama_specroute/bin/activate

chmod +x gen_script_superni_order2_llama_specroute.sh

nohup bash gen_script_superni_order2_llama_specroute.sh 0 meta-llama/Llama-2-7b-hf > run_order2.log 2>&1 &

# Monitor
tail -f run_order2.log
```

**Total experimental time for full comparison (1 model + 2 orders):**
- Setup + verification: 30 mins
- Order 1: 6-10 hours
- Order 2: 6-10 hours
- **Total: 12-20 hours**

---

## Part 3: Collect & Compare Results

### Step 3.1: Run evaluation script

After both orders complete:

```bash
source venv_llama_specroute/bin/activate

# Compute Continual Learning metrics for Order 1
python score.py gen_script_superni_order1_llama_specroute gen_script_superni_order1_llama_specroute

# Example output:
# [INFO] base_dir: logs_and_outputs
# [INFO] run_name: gen_script_superni_order1_llama_specroute
# === Continual Learning Metrics (Order 1) ===
# Cl (Current Learning):    0.4523
# Fgt (Forgetting):         0.1245
# Fwt (Forward Transfer):   0.4234
# Bwt (Backward Transfer):  0.0856
# === Cross-Task Score Matrix ===
#            T1      T2      T3  ... T15
# Task 1:  0.450   0.000   0.000      0.000
# Task 2:  0.438   0.462   0.000      0.000
# ...
```

```bash
# Compute for Order 2
python score.py gen_script_superni_order2_llama_specroute gen_script_superni_order2_llama_specroute
```

### Step 3.2: Compare with ROOT GainLoRA Llama baseline

Assuming ROOT GainLoRA results exist:

```bash
# Llama GainLoRA InfLoRA Order 1 results (reference)
python score.py gen_script_superni_order1_llama_gainlora_inflora gen_script_superni_order1_llama_gainlora_inflora

echo ""
echo "=== COMPARISON: SpecRoute vs GainLoRA InfLoRA (Order 1) ==="
echo "| Metric | GainLoRA  | SpecRoute | Delta |"
echo "|--------|-----------|-----------|-------|"
# Manually paste numbers from above outputs
```

### Step 3.3: Collect final results into comparison table

```bash
# Optional: Create a CSV summary
python -c "
import json
import os

def get_metrics(run_name):
    path = f'logs_and_outputs/{run_name}/outputs/task_order.txt'
    if not os.path.exists(path):
        return None
    # Parse results from score.py output
    # (You can modify this to auto-parse JSON results)
    pass

# Create summary
print('Model,Order,Method,Cl,Fgt,Fwt,Bwt')
# Fill in from score.py outputs above
"
```

---

## Part 4: Interpreting Results

### Continual Learning Metrics

| Metric | Definition | What it means |
|--------|------------|---------------|
| **Cl** | Average accuracy on all tasks at the final step | Overall final performance. Higher is better. |
| **Fgt** | Average forgetting on previous tasks after learning all tasks | Catastrophic forgetting measure. Lower is better (ideally 0). |
| **Fwt** | Average forward transfer (using tasks learned so far) | How much earlier tasks help future tasks. Higher is better. |
| **Bwt** | Average backward transfer (final task helps previous) | How much current learning damages previous task performance. Lower is better. |

### Expected Results (from paper baseline, Table 3)

**Llama-2-7B GainLoRA (InfLoRA):**
- Cl: ~0.45
- Fgt: ~0.12
- Fwt: ~0.42
- Bwt: ~0.09

**SpecRoute should achieve similar or better:**
- Replaces learned routing with parameter-free spectral routing
- Removes KL distillation + data replay for pure LoRA-only continual learning
- Same LoRA GPM (task-specific neuron masks)

### What to accept/concern:

✅ **Good signs:**
- Cl ≈ GainLoRA baseline (0.42-0.48)
- Order 1 and Order 2 have similar Cl (robust to task ordering)
- Fgt is small and stable (< 0.15)
- Training loss decreases smoothly

⚠️ **Warning signs:**
- Cl much lower (< 0.40) → routing may not be converging
- Fgt very high (> 0.20) → catastrophic forgetting problem
- NaN in loss → numerical issue (check bf16 vs fp32)
- Early divergence → learning rate too high or initialization issue

---

## Part 5: Quick Troubleshooting

### Issue: "CUDA out of memory"
```bash
# Reduce batch size in script
# Change: --per_device_train_batch_size 2 
# To:     --per_device_train_batch_size 1
```

### Issue: "score.py not found"
```bash
# Make sure you run from improve_gainlora/ directory
cd /path/to/improve_gainlora
python score.py ...
```

### Issue: "task_order.txt not found"
```bash
# Means tasks didn't complete. Check logs:
tail -100 run_order1.log | grep -i error
```

### Issue: NaN loss
```bash
# SpecRoute training uses bf16 (bfloat16).
# If server doesn't support bf16, modify src/run_llama.py:
# Change: --bf16
# To:     --fp32  (but needs more GPU memory)
```

### Issue: Results directory structure empty
```bash
# Check if training actually ran for each task:
ls -la logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/

# You should see: 1-task1572_samsum_summary/, 2-task363_sst2_polarity_classification/, etc.
```

---

## Part 6: Advanced Usage

### Run on Multi-GPU (if H100 has 8 GPUs)

```bash
# Modify GPU IDs in script or run with:
deepspeed --include localhost:0,1,2,3 --master_port 49500 src/run_llama.py ...

# Or in script, change:
# deepspeed --include localhost:${1} 
# To specify multiple GPUs: 0,1 or 0,1,2,3
```

### Run Llama-2-13B or Llama-3-8B

```bash
# Simply change model path:
bash gen_script_superni_order1_llama_specroute.sh 0 meta-llama/Llama-2-13b-hf

# Or Llama-3:
bash gen_script_superni_order1_llama_specroute.sh 0 meta-llama/Llama-3-8b-hf

# Note: Llama-3 support not yet implemented (will raise NotImplementedError)
# Requires creating llama_3_specroute.py (similar steps as llama_specroute.py)
```

### Profile execution time per task

```bash
# Add timestamps to log
for i in {1..15}; do
    START=$(date +%s)
    # ... run task $i ...
    END=$(date +%s)
    ELAPSED=$((END - START))
    echo "Task $i: $ELAPSED seconds" >> timings.log
done
```

---

## Summary Checklist

- [ ] Created isolated venv_llama_specroute/
- [ ] Installed PyTorch, DeepSpeed, transformers
- [ ] Verified CUDA availability
- [ ] Pre-downloaded model weights (Llama-2-7B)
- [ ] Ran single task test ✓
- [ ] Ran full Order 1 (6-10 hours)
- [ ] Ran full Order 2 (6-10 hours)
- [ ] Computed metrics with score.py for both orders
- [ ] Compared with GainLoRA baseline
- [ ] Recorded results in comparison table
- [ ] Interpreted performance (Cl, Fgt, Fwt, Bwt)

---

## Files Reference

| File | Purpose |
|------|---------|
| `venv_llama_specroute/` | Isolated Python environment |
| `src/llama_specroute.py` | Llama model with spectral routing |
| `src/cl_trainer_specroute_llama.py` | SpecRoute trainer (GPM + ESA) |
| `gen_script_superni_order1_llama_specroute.sh` | Task sequence 1 (15 tasks) |
| `gen_script_superni_order2_llama_specroute.sh` | Task sequence 2 (15 tasks) |
| `score.py` | Evaluation script (computes Cl, Fgt, etc.) |
| `logs_and_outputs/gen_script_superni_order{1,2}_llama_specroute/outputs/` | Results per task |
| `results/comparison_results.md` | Summary table for all methods |

---

## Questions?

Check existing baselines first:
```bash
# ROOT GainLoRA InfLoRA results (reference)
python score.py gen_script_superni_order1_llama_gainlora_inflora gen_script_superni_order1_llama_gainlora_inflora

# T5 SpecRoute results (if available)
python score.py gen_script_superni_order1_t5_specroute gen_script_superni_order1_t5_specroute
```

For theoretical background, see [SPECROUTE_IDEA.md](SPECROUTE_IDEA.md).
