# Comparison Protocol: SpecRoute vs GainLoRA on Llama

This document specifies **exactly how** to compare SpecRoute results with GainLoRA baselines.

## What We're Comparing

| Aspect | Value |
|--------|-------|
| **New Method** | SpecRoute (spectral routing, parameter-free) |
| **Baseline** | GainLoRA InfLoRA (learned routing, trainable params) |
| **Models** | Llama-2-7B, Llama-2-13B, Llama-3-8B |
| **Benchmark** | SuperNI (15 NLP tasks) |
| **Metric** | Continual Learning metrics: Cl, Fgt, Fwt, Bwt |

---

## Step-by-Step Comparison Procedure

### 1. Ensure both methods have completed

```bash
# Check SpecRoute Order 1 is done
ls logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/ | wc -l
# Should output: 15 (one directory per task)

# Check GainLoRA InfLoRA Order 1 is done
ls logs_and_outputs/gen_script_superni_order1_llama_gainlora_inflora/outputs/ | wc -l
# Should also output: 15

# Same for Order 2
ls logs_and_outputs/gen_script_superni_order2_llama_specroute/outputs/ | wc -l
ls logs_and_outputs/gen_script_superni_order2_llama_gainlora_inflora/outputs/ | wc -l
```

### 2. Generate metrics for all 4 runs

```bash
# SpecRoute Order 1
python score.py gen_script_superni_order1_llama_specroute gen_script_superni_order1_llama_specroute > results_specroute_order1.txt

# SpecRoute Order 2
python score.py gen_script_superni_order2_llama_specroute gen_script_superni_order2_llama_specroute > results_specroute_order2.txt

# GainLoRA Order 1
python score.py gen_script_superni_order1_llama_gainlora_inflora gen_script_superni_order1_llama_gainlora_inflora > results_baseline_order1.txt

# GainLoRA Order 2
python score.py gen_script_superni_order2_llama_gainlora_inflora gen_script_superni_order2_llama_gainlora_inflora > results_baseline_order2.txt

# View all results
echo "=== SpecRoute Order 1 ===" && grep -A5 "=== Continual Learning Metrics ===" results_specroute_order1.txt
echo "\n=== SpecRoute Order 2 ===" && grep -A5 "=== Continual Learning Metrics ===" results_specroute_order2.txt
echo "\n=== GainLoRA Order 1 ===" && grep -A5 "=== Continual Learning Metrics ===" results_baseline_order1.txt
echo "\n=== GainLoRA Order 2 ===" && grep -A5 "=== Continual Learning Metrics ===" results_baseline_order2.txt
```

### 3. Create comparison table

Fill in the values from above into this template:

```markdown
## Llama-2-7B SuperNI Continual Learning Results

| Method | Order | Cl | Fgt | Fwt | Bwt | Avg(Cl,Fwt) |
|--------|-------|-----|-----|-----|-----|-------------|
| GainLoRA (InfLoRA) | Order 1 | ___ | ___ | ___ | ___ | ___ |
| GainLoRA (InfLoRA) | Order 2 | ___ | ___ | ___ | ___ | ___ |
| **SpecRoute** | **Order 1** | ___ | ___ | ___ | ___ | ___ |
| **SpecRoute** | **Order 2** | ___ | ___ | ___ | ___ | ___ |

### Average across orders:
- GainLoRA: Cl=___, Fgt=___, Fwt=___, Bwt=___
- SpecRoute: Cl=___, Fgt=___, Fwt=___, Bwt=___

### Comparison summary:
- **Cl (Current Learning)**: SpecRoute vs GainLoRA = ___ (±_%)
- **Fgt (Forgetting)**: SpecRoute vs GainLoRA = ___ (±_%)
- **Fwt (Forward Transfer)**: SpecRoute vs GainLoRA = ___ (±_%)
- **Bwt (Backward Transfer)**: SpecRoute vs GainLoRA = ___ (±_%)
```

### 4. Example: What acceptable results look like

**GOOD result:** SpecRoute ≈ GainLoRA (within 1-2%)
```
GainLoRA Order 1: Cl=0.451, Fgt=0.124, Fwt=0.424, Bwt=0.087
SpecRoute Order 1: Cl=0.450, Fgt=0.126, Fwt=0.422, Bwt=0.089
→ Difference: -0.1% Cl, +0.2% Fgt, -0.2% Fwt, +0.2% Bwt
✓ Acceptable (within noise margin, different routing but same effectiveness)
```

**CONCERNING result:** SpecRoute much worse (>3% drop in Cl)
```
GainLoRA Order 1: Cl=0.451
SpecRoute Order 1: Cl=0.410
→ Difference: -8.2% Cl (BAD!)
✗ Not acceptable - suggests routing issue or training instability
```

---

## Robustness Check: Order Invariance

A good continual learning method should be robust to task ordering.

```bash
# Compare Order 1 vs Order 2 for EACH method

# SpecRoute robustness
ORDER1_CL=$(grep "^Cl" results_specroute_order1.txt | cut -d':' -f2)
ORDER2_CL=$(grep "^Cl" results_specroute_order2.txt | cut -d':' -f2)
echo "SpecRoute Order Robustness: Order1=$ORDER1_CL, Order2=$ORDER2_CL (should be similar)"

# GainLoRA robustness
ORDER1_CL=$(grep "^Cl" results_baseline_order1.txt | cut -d':' -f2)
ORDER2_CL=$(grep "^Cl" results_baseline_order2.txt | cut -d':' -f2)
echo "GainLoRA Order Robustness: Order1=$ORDER1_CL, Order2=$ORDER2_CL (should be similar)"

# Expected: Both methods should have similar Cl in Order 1 and Order 2
# (within 1-2% variance due to data shuffling)
```

---

## Per-Task Analysis (Advanced)

### Extract per-task accuracies

```bash
# Extract cross-task score matrix from SpecRoute Order 1
python -c "
import json
import os

run_name = 'gen_script_superni_order1_llama_specroute'
base_dir = 'logs_and_outputs'

# Read task list
with open(f'{base_dir}/{run_name}/outputs/task_order.txt') as f:
    tasks = f.read().strip().split(',')

print('Task order:')
for i, t in enumerate(tasks, 1):
    print(f'{i}. {t}')

print()
print('Per-task scores (diagonal = final accuracy on that task):')
print('Task | Final Acc | Forgetting from peak?')
print('-----|-----------|---------------------')

task_num = len(tasks)
per_task_scores = []

for i in range(task_num):
    res_file = f'{base_dir}/{run_name}/outputs/{i+1}-{tasks[i]}/all_results.json'
    if os.path.exists(res_file):
        with open(res_file) as f:
            result = json.load(f)
        key = f'predict_eval_rougeL_for_{tasks[i]}'
        score = result.get(key, 0.0)
        per_task_scores.append(score)
        print(f'{i+1:<5}| {score:.4f}    | -')
    else:
        print(f'{i+1:<5}| MISSING   | -')
"
```

### Compare task-by-task

```bash
# Create side-by-side comparison
python -c "
import json
import os

def get_task_scores(run_name):
    base_dir = 'logs_and_outputs'
    with open(f'{base_dir}/{run_name}/outputs/task_order.txt') as f:
        tasks = f.read().strip().split(',')
    
    scores = {}
    for i, task in enumerate(tasks):
        res_file = f'{base_dir}/{run_name}/outputs/{i+1}-{task}/all_results.json'
        if os.path.exists(res_file):
            with open(res_file) as f:
                result = json.load(f)
            key = f'predict_eval_rougeL_for_{task}'
            scores[task] = result.get(key, 0.0)
    return scores, tasks

specroute_scores, task_order = get_task_scores('gen_script_superni_order1_llama_specroute')
gainlora_scores, _ = get_task_scores('gen_script_superni_order1_llama_gainlora_inflora')

print(f'{'Task':<45} | {'GainLoRA':>8} | {'SpecRoute':>8} | {'Delta':>7}')
print('-' * 72)

for task in task_order:
    gl = gainlora_scores.get(task, 0.0)
    sr = specroute_scores.get(task, 0.0)
    delta = sr - gl
    print(f'{task:<45} | {gl:>8.4f} | {sr:>8.4f} | {delta:>+7.4f}')
"
```

---

## Interpretation Guide

### What each metric means:

- **Cl (Current Learning)**: Average final accuracy across all tasks
  - ✓ Higher is better
  - Range: 0.0 to 1.0 (for ROUGE-L, typically 0.3-0.5 for SuperNI)

- **Fgt (Forgetting)**: How much performance drops on earlier tasks
  - ✓ Lower is better (ideally 0, meaning no forgetting)
  - Range: 0.0 to 1.0
  - Formula: average of (best_perf_on_task - final_perf_on_task) over all tasks

- **Fwt (Forward Transfer)**: How much earlier tasks help future tasks
  - ✓ Higher is better (positive transfer)
  - Can be negative (negative transfer)
  - Formula: average (final_perf_after_learning_all - perf_without_prior_tasks)

- **Bwt (Backward Transfer)**: How much learning new tasks affects old tasks
  - ✓ Lower is better (less negative impact)
  - Can be negative if current learning hurts past tasks
  - Formula: average (final_perf - initial_perf_after_learning) over past tasks

### Expected values for SuperNI (Llama-2-7B):

| Metric | Poor (<) | Fair | Good | Excellent (>) |
|--------|----------|------|------|---------------|
| Cl | 0.40 | 0.40-0.45 | 0.45-0.50 | 0.50 |
| Fgt | 0.20 | 0.15-0.20 | 0.10-0.15 | 0.10 |
| Fwt | 0.35 | 0.35-0.40 | 0.40-0.45 | 0.45 |
| Bwt | 0.15 | 0.10-0.15 | 0.05-0.10 | 0.05 |

---

## Final Comparison Report Template

```markdown
# Llama-2-7B SpecRoute vs GainLoRA (InfLoRA) - Final Report

## Summary
- **Model**: Llama-2-7B
- **Benchmark**: SuperNI (15 tasks)
- **Task Orders Tested**: Order 1, Order 2
- **Baseline**: GainLoRA InfLoRA (ROOT implementation)
- **New Method**: SpecRoute (spectral routing, parameter-free)

## Results

### Overall Performance (averaged across both orders)

| Metric | GainLoRA | SpecRoute | Difference | Status |
|--------|----------|-----------|------------|--------|
| Cl     | 0.451    | 0.450     | -0.1%      | ✓ PASS |
| Fgt    | 0.124    | 0.126     | +0.2%      | ✓ PASS |
| Fwt    | 0.424    | 0.422     | -0.2%      | ✓ PASS |
| Bwt    | 0.087    | 0.089     | +0.2%      | ✓ PASS |

### Order Robustness

| Method | Order 1 Cl | Order 2 Cl | Variance |
|--------|-----------|-----------|----------|
| GainLoRA | 0.451 | 0.450 | ±0.1% |
| SpecRoute | 0.450 | 0.449 | ±0.1% |

### Key Findings

1. **Performance Parity**: SpecRoute achieves nearly identical accuracy to GainLoRA
2. **Robustness**: Both methods stable across task orderings
3. **Insights**: SpecRoute replaces learned routing with parameter-free SVD-based routing
   - Parameter count reduced (no Trans_input, no prompt_key)
   - Training more interpretable (spectral signatures reveal task relationships)
   - No additional hyperparameters for routing (unlike GainLoRA's trans_hidden_dim, attn_lr, etc.)

## Conclusion

✓ **SpecRoute successfully ports to Llama architecture**
✓ **Maintains parity with GainLoRA baseline**
✓ **Ready for deployment and extension to larger models**
```

---

## Quick Metric Extraction Command

```bash
cat results_*.txt | grep -E "(Cl |Fgt |Fwt |Bwt )" | sed 's/.*: //'
```

---

**Next steps after comparison:**
1. Record results in `results/comparison_results.md` (Table 5: Llama SpecRoute)
2. If satisfied, commit to git: `git add -A && git commit -m "Add Llama SpecRoute results"`
3. (Optional) Run on Llama-2-13B or Llama-3-8B for full ablation
