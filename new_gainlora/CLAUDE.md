# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a Continual Learning (CL) research project implementing **GainLoRA + SRT Router** — a method combining LoRA adapters with Statistical Routing Theory (SRT) for multi-task learning on T5 and LLaMA models.

## Key Architecture

### GainLoRA + InfLoRA (t5_gainlora_inflora.py)
- LoRA adapters (`lora_A ∈ ℝ^{r×d}`, `lora_B ∈ ℝ^{d×r}`) attached to attention projections (q, v)
- `cal_attention()` produces soft routing weights via: `|sigmoid(4*cos_sim) * 2 - 1| → [0,1]`
- `agg_lora_states()` blends all prior LoRA contributions during training
- **CRITICAL BUG**: `reset_parameters()` is bypassed by `no_init_weights=True` in `from_pretrained()` → `lora_A` stays zeros. Fix at run_t5.py lines 550–555.

### SRT Router (srt_router.py)
- **PooledMahalanobis_RIDGE** (default): d_t(h) = (h-μ_t)ᵀ Σ_pool⁻¹ (h-μ_t)
  - Welford-Hart incremental pooled covariance (zero-rehearsal compliant)
  - Ridge shrinkage: δ* = d/(n+d) — analytical optimal for n≈d regime
  - GPU-accelerated via torch.linalg.eigh on CUDA
  - Achieves 97.51% final / 97.17% avg routing accuracy on SuperNI
- Alternative shrinkage: `--srt_shrinkage ridge|oas|lw|none`
- All tasks share the same Σ_pool⁻¹ (SRT Theorem 4: shared-covariance Gaussians → pooled Mahalanobis is Bayes-optimal)
- `SRTRouter.add_task()` → Welford-Hart update → Ridge shrink → eigendecomposition → store
- `SRTRouter.route()` → Pooled Mahalanobis distance to each centroid → argmin

### Train-Inference Mismatch Bug
**Location**: `t5_gainlora_inflora.py` lines 1335–1393
- Training: `cal_attention()` → soft weights → cross-task LoRA blending
- Inference: `key_attention_weights[:] = srt_weights` (one-hot) → overrides soft routing
- **Effect**: CB task destroyed (3.57%) because it loses cross-NLI knowledge from rte/mnli adapters
- **Root cause of AP gap**: new_gainlora (77.62) < root_gainlora (78.01) despite SRT's 99.99% routing accuracy

### SRT-Guided Warm Initialization (SGWI) — see contri2_analysis/
- Uses same SRT distance metric for **both** routing (inference) and warm init (training)
- NTI: copy LoRA from nearest SRT-distance previous task
- SFI: `ΔW_init = Σ w_s·B_s·A_s` → rank-r SVD → initialize LoRA_t
- Solves the single-task performance problem in hard routing

## Task Order (Long_Sequence, order 3)
```
yelp → amazon → mnli → cb → copa → qqp → rte → imdb → sst2 → dbpedia
     → agnews → yahoo → multirc → boolq → wic
```

## Common Commands

### Run SRT LLaMA training (HPC)
```bash
# LLaMA: SRT with PooledMahalanobis_RIDGE (default, --srt_shrinkage ridge)
cd new_llama_gainlora && bash run_superni_order1_llama_cl.sh

# Explicit GPU override (use before MODEL_PATH)
bash gen_script_superni_order1_llama_srt.sh --gpu 5090 meta-llama/Llama-2-7b-hf   # 32GB → fp16 + grad_ckpt, BSZ=2 GA=8
bash gen_script_superni_order1_llama_srt.sh --gpu h100  meta-llama/Llama-2-7b-hf   # 80GB  → bf16, BSZ=8 GA=2
bash gen_script_superni_order1_llama_srt.sh --gpu a100  meta-llama/Llama-2-7b-hf   # 80GB  → bf16/fp16 per CUDA version

# With SGWI + Dual Fisher
bash gen_script_superni_order1_llama_srt.sh --gpu 5090 --dual_fisher True --lambda_emb 0.05 meta-llama/Llama-2-7b-hf
```

### Run SRT T5 training (HPC)
```bash
# generates gen_script_long_order3_t5_srt_ridge.sh (PooledMahalanobis_RIDGE)
python generate_srt_order3.py ridge
# all modes: ridge, oas, lw, none
python generate_srt_order3.py all
```

### Run Contri2 isolated tests
```bash
cd contri2_analysis
python run_contri2.py                    # all 3 tests, all 15 tasks
python run_contri2.py --test 1          # only Zero-Shot Transfer
python run_contri2.py --test 2          # only Few-Shot Convergence
python run_contri2.py --test 3          # only Ablation on Init Methods
python run_contri2.py --tasks cb,mnli,rte  # specific tasks only
python run_contri2.py --skip-train      # skip training steps
```

### Run score.py after training
```bash
python score.py long_order3_t5_srt_hard long_order3_t5_srt_hard
```

## Project Structure
```
new_gainlora/
├── src/
│   ├── run_t5.py                 # Entry point — trains each task
│   ├── t5_gainlora_inflora.py    # T5 + GainLoRA + SRT integration
│   ├── cl_trainer_srt.py          # SRT_Trainer: extracts signatures, wires router
│   ├── srt_router.py              # SRTRouter: non-parametric routing
│   ├── cl_trainer_gainlora_inflora.py  # Training loop with GPM
│   └── assets.py                 # task_config, lora_state_dict helpers
├── CL_Benchmark/
│   └── Long_Sequence/{task}/      # 15 tasks × {train,dev,test}.json
├── contri2_analysis/
│   ├── run_contri2.py            # Main runner: 3 isolated tests
│   └── contri2_utils.py          # Core utilities: SGWI init, eval, training
└── gen_script_long_order3_t5_srt_*.sh  # HPC training scripts
```

## Key LoRA Init Code Locations
| What | File | Lines |
|------|------|-------|
| LoRA reset_parameters (kaiming A, zeros B) | `t5_gainlora_inflora.py` | 120–123 |
| A=zeros bug fix | `run_t5.py` | 550–555 |
| SRT routing override | `t5_gainlora_inflora.py` | 1335–1393 |
| cal_attention (soft routing) | `t5_gainlora_inflora.py` | 1201–1230 |
| agg_lora_states (blending) | `t5_gainlora_inflora.py` | 639–658 |
| SRT PooledMahalanobis core | `srt_router.py` | PooledMahalanobisRouter class |
| Ridge shrinkage (δ*=d/(n+d)) | `srt_router.py` | `_shrink_ridge()` |
| Welford-Hart pooled update | `srt_router.py` | `welford_pooled_update()` |
| SRT signature extraction | `cl_trainer_srt.py` | `_extract_task_embeddings()` |
| SGWI init (NTI/SFI) | `contri2_analysis/contri2_utils.py` | _init_nti, _init_sfi |
