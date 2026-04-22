# Continual LLaMA/T5 Experiment Init Context

Last updated: 2026-04-22
Scope: This document is a portable init context so a new session/model can understand the project state immediately.

## 1) Purpose and Ground Rules

This file captures:
- what `origin_gainlora` does (baseline reference),
- what SRT + SGWI means in the T5 path (current method standard),
- why LLaMA has been problematic,
- what has already been changed in `llama_epoch_ablation`,
- what is already verified vs what is still pending.

Important:
- Canonical experiment workspace is `llama_epoch_ablation` (singular).
- This file is stored in `llama_epochs_ablation/project.md` (plural) because the user requested this destination.

## 2) Codebase Map (High Signal)

### 2.1 Baseline reference
- `origin_gainlora/`
  - Original continual-learning implementation used as historical baseline.
  - Key LLaMA files:
    - `origin_gainlora/src/run_llama.py`
    - `origin_gainlora/src/llama_gainlora_inflora.py`
    - `origin_gainlora/src/cl_trainer_gainlora_inflora_llama.py`

### 2.2 New method reference (T5 and LLaMA)
- `new_gainlora/`
  - Contains SRT routing and SGWI-related trainers.
  - Key files:
    - `new_gainlora/src/run_t5.py`
    - `new_gainlora/src/t5_gainlora.py`
    - `new_gainlora/src/sgwi_trainer.py`
    - `new_gainlora/src/run_llama.py`
    - `new_gainlora/src/llama_gainlora.py`
    - `new_gainlora/src/sgwi_trainer_llama.py`

### 2.3 Current ablation workspace
- `llama_epoch_ablation/`
  - Contains two tracks:
    1. `root_gainlora_bugfix` (restore origin-like behavior under modern stack)
    2. `llama_clean_port` (clean local port of T5-style SRT+SGWI idea to LLaMA)

## 3) What origin_gainlora Does (LLaMA Baseline Semantics)

## 3.1 Entry and argument pattern
- `origin_gainlora/src/run_llama.py` defines model/data/training arguments including:
  - `task_order`
  - `attn_lr`
  - `trans_hidden_dim`
  - `model_name` variants (`gainlora_inflora`, etc.)
- It builds prompt config and chooses trainer/model class based on `model_name` and backbone.

## 3.2 Routing behavior in origin (soft weighting)
- In `origin_gainlora/src/llama_gainlora_inflora.py`:
  - `cal_attention(prompt_key, x, return_logits=False)` computes similarity-based weights.
  - It normalizes `x` and `prompt_key`, then uses a sigmoid-derived transform:
    - effectively a soft weight in [0, 1], not hard one-hot.
- In forward pass, query features are built from masked pooled input embeddings, transformed by `trans_input`, then matched against prompt keys.
- When previous tasks exist, current and previous prompt keys are concatenated and weighted by `cal_attention`.

## 3.3 Continual constraints / projection behavior
- In `origin_gainlora/src/cl_trainer_gainlora_inflora_llama.py`:
  - `get_reg_matrix()` and related paths build/update feature subspaces.
  - The trainer applies projection-matrix style constraints to reduce forgetting (GPM-like mechanism).

Practical meaning:
- Origin LLaMA is not only LoRA updates; it is also shaped by prompt-key attention and feature-space constraints.

## 4) SRT + SGWI Standard (T5 Path)

This is treated as the methodological standard in this project context.

## 4.1 SRT routing core
- SRT is non-parametric routing based on task signatures (mu/cov-like stats) from embeddings.
- In `new_gainlora/src/t5_gainlora.py` forward path:
  - Training path: current task slot gets hard one-hot routing (`[1, 0, 0, ...]`).
  - Inference path: route with SRT (`self.srt_router.route(...)`) and map prediction to one-hot adapter slot.
- This means routing is hard assignment, not soft interpolation.

## 4.2 SGWI core
- In `new_gainlora/src/sgwi_trainer.py`:
  - `SGWI_DualFisher_Trainer` extends SRT trainer.
  - Supports modes such as `full_lora`, `sgwi_full`, `sgwi_freeze_a`, `sgwi_train_a`, `inflora`, `random`.
  - `sgwi_full`: warm-init both A and B from weighted past adapters.
  - `full_lora`: no SGWI warm init, but full LoRA-style training path.

## 4.3 run_t5 wiring (important)
- In `new_gainlora/src/run_t5.py` for `model_name == gainlora`:
  - always uses `SGWI_DualFisher_Trainer` (both SGWI-enabled and SGWI-disabled paths go through same trainer family),
  - `sgwi=True` maps to `sgwi_full`, `sgwi=False` maps to `full_lora`,
  - lora_A is explicitly unfrozen in gainlora path, supporting full A+B behavior.

Why this matters:
- T5 path has a relatively coherent method path: SRT hard routing + SGWI-capable trainer family + full LoRA behavior.

## 5) LLaMA Problem Statement

Observed phenomenon:
- New/T5 path can perform strongly in intended settings.
- LLaMA path often behaves near-untrainable or random-like in comparable experiments.

Concrete evidence already logged:
- `llama_epoch_ablation/run_results.txt` contains an earlier root-task1572 run snapshot showing collapse-like behavior:
  - eval/predict metrics around zero (rouge and exact-match all 0),
  - very high training loss (`train_loss` around 209),
  - attention weights log looks degenerate (`[0.]`).

Interpretation:
- This is not normal healthy adaptation behavior for task 1 and strongly suggests LLaMA path confounds.

## 6) Current Reasoning (Confidence Tagged)

## 6.1 High confidence findings
1. Recipe/path drift is a major confound in LLaMA comparisons.
- Historical origin-like recipe and new LLaMA recipe diverged in meaningful hyperparameters and flow.

2. A known SGWI covariance-source mismatch existed in LLaMA SGWI trainer family.
- In `new_gainlora/src/sgwi_trainer_llama.py`, distance computation can depend on router pooled covariance.
- Clean-port patch adds fallback from `pooled_cov` to `_Sigma_pool` to avoid silent mismatch.

3. T5 method path is internally more coherent for SRT+SGWI evaluation.
- The trainer and routing semantics are aligned and explicit in code.

## 6.2 Medium confidence findings
1. LLaMA failures are likely multi-factor, not one single bug.
- Includes method drift, recipe drift, and architecture sensitivity.

2. Some origin-compatible modernized paths historically had padding/position semantics risk (left-padding and RoPE interactions).
- This was part of prior diagnosis context and is still relevant as a caution.

## 6.3 Open questions
- After restoring origin-like baseline and running clean-port under matched recipe, does LLaMA recover meaningfully?
- If not, what remains: data formatting, collator behavior, hidden API drift, or LLaMA-specific optimization instability?

## 7) What Was Implemented in llama_epoch_ablation

## 7.1 Baseline restoration track (origin-like)
Main files:
- `llama_epoch_ablation/root_gainlora_bugfix/src/run_llama.py`
  - dataset loading path was restored toward origin-style `load_dataset(..., cl_dataset.py)` semantics.
- `llama_epoch_ablation/root_gainlora_bugfix/src/llama_gainlora_inflora.py`
  - query pooling behavior restored toward origin masked-mean pattern.
- `llama_epoch_ablation/generate_root_order1_baseline.py`
  - origin-like recipe alignment for order1 generator,
  - includes replay flags for task > 1,
  - includes small sleeps between tasks.
- `llama_epoch_ablation/run_root_task1572_baseline.sh`
  - currently set to origin-like key knobs (`attn_lr 0.0`, `kl_ratio 1`, `trans_hidden_dim 50`) with current `num_train_epochs 10` in file state.

Note:
- Earlier logs in `run_results.txt` still reflect a previous run with older task1572 settings (5 epochs, old ratios).

## 7.2 Clean-port track (T5 idea to LLaMA)
Main files:
- `llama_epoch_ablation/llama_clean_port/src/run_llama.py`
  - `sgwi` default set to `True` in this local fork,
  - gainlora path always uses `SGWI_DualFisher_LLaMA_Trainer` (mirrors T5-style trainer unification idea).
- `llama_epoch_ablation/llama_clean_port/src/sgwi_trainer_llama.py`
  - covariance lookup hardened: `pooled_cov` fallback to `_Sigma_pool`.

Launchers/generators added:
- `llama_epoch_ablation/run_clean_task1572_llama_port.sh`
- `llama_epoch_ablation/run_clean_order1_llama_port.sh`
- `llama_epoch_ablation/generate_clean_order1_llama_port.py`

## 7.3 Important known issue discovered in current file state
- `llama_epoch_ablation/generate_clean_order1_llama_port.py` currently builds command lines with:
  - `command = " \\\n+".join(cmd_parts)`
- This introduces literal leading `+` tokens into generated scripts.
- Result: generated script `llama_epoch_ablation/generated_scripts/gen_script_superni_order1_llama_gainlora_srt_cleanport.sh` contains lines like `+   --do_train`.
- This can break/contaminate CLI execution for full order1 clean-port runs.

Also observed:
- generator header uses `printf '%s\n'` inside Python string literals that become a literal newline in output script text (`printf '%s` line break `' ...`). Usually not fatal, but clearly malformed.

Implication:
- `run_clean_task1572_llama_port.sh` (single-task direct launcher) is the safer immediate clean-port check.
- Full `run_clean_order1_llama_port.sh` should be treated as at-risk until generator string formatting is fixed.

## 8) Results So Far

## 8.1 What is completed
- Diagnosis and code-level comparison between origin and new paths.
- Baseline restoration patches in `llama_epoch_ablation`.
- Clean-port scaffolding and key trainer/routing patches.

## 8.2 What is not completed
- Fresh end-to-end reruns after all latest adjustments.
- Final metric comparison baseline vs clean-port under controlled settings.

## 8.3 Latest empirical snapshot available
From `llama_epoch_ablation/run_results.txt` (older run snapshot):
- root task1572 gainlora_inflora run showed:
  - train loss very high,
  - eval/predict scores near zero,
  - behavior consistent with failure/collapse.

Treat this as "pre-latest-rerun evidence", not final post-fix verdict.

## 9) Recommended Execution Order (Current)

1. Run restored baseline task1572 first.
- `bash llama_epoch_ablation/run_root_task1572_baseline.sh`

2. Run clean-port task1572 second.
- `bash llama_epoch_ablation/run_clean_task1572_llama_port.sh`

3. Run full order1 baseline.
- `bash llama_epoch_ablation/run_root_order1_baseline.sh`

4. Run full order1 clean-port only after generator script formatting issue is fixed.
- current generator output has `+`-prefixed args.

## 10) Quick Checklist Before Launch

- Confirm data path availability:
  - baseline expects `root_gainlora/CL_Benchmark`
  - clean-port task1572 launcher expects `new_gainlora/CL_Benchmark`
- Confirm model path (default is `meta-llama/Llama-2-7b-hf`).
- Confirm DeepSpeed executable or Python module launcher availability.
- Use unique `master_port` if multiple runs are active.
- Keep logs under `llama_epoch_ablation/logs/` and outputs under `llama_epoch_ablation/logs_and_outputs/`.

## 11) Minimal Session Boot Prompt (Optional)

If a new model/session is loaded, provide this summary:
- Baseline reference is `origin_gainlora` semantics.
- T5 standard for method is hard SRT routing + SGWI-capable trainer (`run_t5.py`, `t5_gainlora.py`, `sgwi_trainer.py`).
- LLaMA currently under two-track validation in `llama_epoch_ablation`:
  - `root_gainlora_bugfix` for origin-like restoration,
  - `llama_clean_port` for T5-idea clean port.
- There is a known current bug in clean order1 generator output (`+` prefixed CLI args).
- Latest stored run evidence in `run_results.txt` still reflects pre-latest-fix collapse-like behavior.

---

End of init context.
