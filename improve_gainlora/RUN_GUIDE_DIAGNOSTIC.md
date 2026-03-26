# SpecRoute Diagnostic Run Guide

## Quick Start — Priority Experiment

**Run Long Sequence Order 3 (T5-small) first** — hardest benchmark with many same-domain tasks (yelp/amazon/imdb/sst2 all sentiment).

### On H100 / A100:

```bash
cd /path/to/Continual/improve_gainlora

# Using T5_small scripts (has --do_predict in all 15 tasks = full diagnostics)
bash T5_small/gen_script_long_order3_t5_small_specroute.sh 0 google/flan-t5-small

# OR using top-level scripts (--do_predict only in task 1)
bash gen_script_long_order3_t5_specroute.sh 0 google/flan-t5-small
```

> **Recommendation**: Use `T5_small/` scripts — they have `--do_predict` in all 15 blocks,
> so routing diagnostics (`routing_decisions.pt`) are saved for every task.
> Top-level scripts only save routing data for task 1.

### On Kaggle / Colab (T4 GPU):

```bash
bash setup_kaggle_colab.sh   # install deps
bash T5_small/gen_script_long_order3_t5_small_specroute.sh 0 google/flan-t5-small
```

---

## All Experiments

| Script | Model | Benchmark | Priority |
|--------|-------|-----------|----------|
| `T5_small/gen_script_long_order3_t5_small_specroute.sh` | flan-t5-small | Long Seq (15 tasks, order 3) | **1st** |
| `T5_small/gen_script_long_order4_t5_small_specroute.sh` | flan-t5-small | Long Seq (15 tasks, order 4) | 2nd |
| `T5_small/gen_script_superni_order1_t5_small_specroute.sh` | flan-t5-small | SuperNI (15 tasks, order 1) | 3rd |
| `T5_small/gen_script_superni_order2_t5_small_specroute.sh` | flan-t5-small | SuperNI (15 tasks, order 2) | 4th |
| `gen_script_superni_order1_llama_specroute.sh` | Llama-2-7B | SuperNI order 1 | Later |
| `gen_script_superni_order2_llama_specroute.sh` | Llama-2-7B | SuperNI order 2 | Later |

---

## CPI/OAP Parameters (already set in all scripts)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `--cpi_gamma` | 0.5 | CPI contrastive weight (γ in Def. 3) |
| `--oap_eta` | 0.5 | OAP projection strength (η in Def. 4) |
| `--oap_beta_min` | 0.3 | OAP minimum retention floor (β_min in Thm. 4) |
| `--oap_warmup` | 3 | Tasks before full OAP kicks in (empirical safeguard) |

To change parameters:
```bash
python _patch_cpi_oap.py --gamma 0.5 --eta 0.5 --beta_min 0.3 --warmup 3
# This patches top-level scripts only. For T5_small, edit manually or extend the script.
```

---

## Diagnostic Outputs

### 1. CPI/OAP Init Diagnostics (during training, task ≥ 2)

**Log lines** (grep for these in stdout):
```
[DIAG-INIT] Layer N chunk K: ρ_l=0.1234, β_l=0.7000, SSE_before=0.45, SSE_after=0.12, λ_min+/r=0.000123, n_pos=6/8
```

| Metric | What it tells you | Healthy range |
|--------|-------------------|---------------|
| `ρ_l` | Domain proximity to old tasks (weighted) | 0.0–1.0 (higher = more overlap) |
| `β_l` | OAP retention factor for layer l | ≥ β_min (0.3). Higher ρ → higher β |
| `SSE_before` | Subspace overlap BEFORE OAP | Varies |
| `SSE_after` | Subspace overlap AFTER OAP | Should be < SSE_before |
| `λ_min+/r` | CPI routing margin (Thm. 3 lower bound) | > 1e-5. If ≈ 0, CPI failing |
| `n_pos/total` | Positive eigenvalues in D_t | ≥ r (lora_r=8). If < r, falling back to Kaiming |

**Saved file**: `<output_dir>/saved_weights/init_diagnostics.pt`
- List of dicts, one per layer, with per-chunk diagnostics

### 2. Routing Diagnostics (during prediction, requires `--do_predict`)

**Log lines**:
```
[DIAG-ROUTING] Task amazon (id=1): routed_to_current=0.850 (850/1000) n_tasks=2
  task_idx=0: 0.850
  task_idx=1: 0.150
```

| Metric | What it tells you | Healthy range |
|--------|-------------------|---------------|
| `routed_to_current` | Fraction correctly routed to current task's expert | > 0.7 |
| `p_e = 1 - routed_to_current` | Routing error rate (Thm. 4 input) | < 0.3 |

**Saved file**: `<output_dir>/saved_weights/routing_decisions.pt`
- Tensor of routing indices (0 = current task, 1+ = old tasks)

### 3. Standard Metrics

After all 15 tasks complete, stdout contains `predict_exact_match_for_<task>` lines.
Use the scoring script:
```bash
# Parse from log file
python ../parse_and_score_v2.py <logfile>
```

---

## Post-Run Analysis

### Quick analysis script:

```bash
python analyze_diagnostics.py gen_script_long_order3_t5_small_specroute
```

This reads all `init_diagnostics.pt` and `routing_decisions.pt` files and prints:
- Per-task CPI/OAP health (ρ, β, SSE, λ_min+/r, eigenvalue count)
- Per-task routing error (p_e)
- Trend analysis (is p_e increasing? Is λ_min+/r collapsing?)
- Summary table

### Manual inspection:

```python
import torch

# Load init diagnostics for task 5
diag = torch.load('logs_and_outputs/gen_script_long_order3_t5_small_specroute/outputs/5-copa/saved_weights/init_diagnostics.pt')
for layer_idx, layer_data in enumerate(diag):
    for chunk_idx, d in layer_data.items():
        print(f"Layer {layer_idx} chunk {chunk_idx}: ρ={d['rho_l']:.4f} β={d['beta_l']:.4f} λ+/r={d['lambda_min_pos_over_r']:.6f}")

# Load routing decisions for task 10
rd = torch.load('logs_and_outputs/gen_script_long_order3_t5_small_specroute/outputs/10-dbpedia/saved_weights/routing_decisions.pt')
p_e = 1.0 - (rd == 0).float().mean().item()
print(f"p_e = {p_e:.3f}")
```

---

## Ablation Grid (§7.1)

Run 4 configs to isolate CPI vs OAP contribution:

| Config | γ | η | What it tests |
|--------|---|---|---------------|
| baseline | 0 | 0 | Pure spectral routing (no CPI, no OAP) |
| CPI only | 0.5 | 0 | CPI init without OAP projection |
| OAP only | 0 | 0.5 | OAP projection without CPI init |
| full | 0.5 | 0.5 | Full SpecRoute (default) |

To run the baseline config:
```bash
python _patch_cpi_oap.py --gamma 0 --eta 0
# Then run the same gen_script with different --run_name to separate outputs
```

> Note: When γ=0, CPI falls back to standard Kaiming init.
> When η=0, OAP is skipped (β_l always = 1, no projection applied).

---

## What to Report Back

After the first experiment (Long Order 3 T5-small) finishes:

1. **Final AP and Forgetting**: run `python ../parse_and_score_v2.py <logfile>`
2. **Diagnostic summary**: run `python analyze_diagnostics.py gen_script_long_order3_t5_small_specroute`
3. **Key questions to answer**:
   - Is `p_e` staying below 0.3 across all 15 tasks?
   - Is `λ_min+/r` maintaining a healthy margin (> 1e-5)?
   - Does SSE decrease after OAP (SSE_after < SSE_before)?
   - Is β_l respecting the β_min=0.3 floor?
   - Any tasks with routing accuracy < 0.5? (indicates method failure)

4. **If things look bad** (p_e > 0.3 and rising):
   → We discuss Option 1: decoupled routing, prototype-based routing, or hierarchical routing

5. **If things look good** (p_e < 0.3, stable):
   → Proceed to remaining experiments (Order 4, SuperNI, Llama)

---

## Output Directory Structure

```
logs_and_outputs/gen_script_long_order3_t5_small_specroute/
  outputs/
    task_order.txt
    1-yelp/
      saved_weights/
        routing_decisions.pt      # routing stats (if --do_predict)
        init_diagnostics.pt       # CPI/OAP diagnostics (task >= 2)
        spectral_signatures.pt    # frozen A matrices + calibration
        ...
    2-amazon/
      saved_weights/
        init_diagnostics.pt
        routing_decisions.pt
        ...
    ...
    15-wic/
      saved_weights/
        ...
```
