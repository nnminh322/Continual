# LLaMA Epoch Ablation

This folder isolates epoch-count ablations for the current LLaMA GainLoRA path without modifying anything inside `new_gainlora`.

All scripts here call back into `../new_gainlora/src/run_llama.py` and keep the current runtime stack unchanged. Only the launch surface changes:

- outputs are written under `llama_epoch_ablation/logs_and_outputs/`
- logs are written under `llama_epoch_ablation/logs/`
- generated full-order scripts are written under `llama_epoch_ablation/generated_scripts/`

## What Is Included

- `run_task1572_epoch_sweep.sh`
  - Fast single-task ablation for `task1572_samsum_summary`
  - Uses the same task-1 command as `new_gainlora/gen_script_superni_order1_llama2_srt.sh`
  - Only changes `RUN_NAME`, `OUTPUT_DIR`, and `--num_train_epochs`

- `generate_order1_epoch_variant.py`
  - Clones `new_gainlora/gen_script_superni_order1_llama2_srt.sh`
  - Rewrites only:
    - `RUN_NAME`
    - `BASE_OUT`
    - `#SBATCH` job/log names
    - every `--num_train_epochs 100`

- `run_order1_epoch_variant.sh`
  - Generates an order-1 script for a chosen epoch count
  - Runs that generated script from inside `new_gainlora`

- `root_gainlora_bugfix/`
  - Isolated copy of the old `root_gainlora` LLaMA path under `llama_epoch_ablation/`
  - Keeps the old `gainlora_inflora` method, with only runtime-compat patches for the current Python 3.12 / Transformers 5 / datasets 3 stack

- `generate_root_order1_baseline.py`
  - Generates a full order-1 shell script for the copied old-method baseline
  - Uses the recovered root-like LLaMA settings: `lr=5e-05`, `epochs=50`, `lora_r=4`, `lora_alpha=32`, `chunk=4`, `threshold=0.995`

- `run_root_order1_baseline.sh`
  - Generates and runs the full order-1 old-method baseline against `root_gainlora_bugfix/src/run_llama.py`

- `run_root_task1572_baseline.sh`
  - Fast single-task quality check for the copied old method on `task1572_samsum_summary`

## Why Task 1572 First

`task1572_samsum_summary` is the shortest and clearest place to test the epoch hypothesis:

- train: 160
- eval/dev: 20
- test: 20

With the current A100 path in `new_gainlora`, this task gets about 10 optimizer steps per epoch, so 100 epochs is about 1000 steps. The single-task sweep lets you test whether more steps help before paying for a full order-1 run.

## Example Usage

Single-task sweep with the current settings, testing 100, 150, and 200 epochs:

```bash
bash llama_epoch_ablation/run_task1572_epoch_sweep.sh --epochs_list 100,150,200
```

Single-task sweep with explicit model path and SGWI disabled:

```bash
bash llama_epoch_ablation/run_task1572_epoch_sweep.sh \
  --model_path meta-llama/Llama-2-7b-hf \
  --epochs_list 100,200,300 \
  --sgwi False
```

Generate and run a full order-1 variant with 200 epochs:

```bash
bash llama_epoch_ablation/run_order1_epoch_variant.sh --epochs 200
```

Generate and run a full order-1 variant with 300 epochs and Dual Fisher:

```bash
bash llama_epoch_ablation/run_order1_epoch_variant.sh \
  --epochs 300 \
  --dual_fisher True \
  --lambda_emb 0.01
```

Run the copied old-method baseline on task1572 first:

```bash
bash llama_epoch_ablation/run_root_task1572_baseline.sh \
  --model_path meta-llama/Llama-2-7b-hf \
  --gpu_ids 0
```

Generate and run the full order-1 copied old-method baseline:

```bash
bash llama_epoch_ablation/run_root_order1_baseline.sh \
  --model_path meta-llama/Llama-2-7b-hf \
  --gpu_ids 0
```

## Traceability

Each single-task run writes:

- a dedicated output directory under `logs_and_outputs/`
- a dedicated log file under `logs/`
- the exact launch command to `launch_command.sh` inside that run's output directory

Each full order-1 run writes:

- the generated shell script under `generated_scripts/`
- a dedicated console log under `logs/`
- all model outputs under `logs_and_outputs/`

## Important Constraint

This folder intentionally does not patch, fork, or edit any file inside `new_gainlora`. If you want a second ablation axis later, add it here as a new wrapper or generator rather than editing the source tree.

The copied old-method baseline under `root_gainlora_bugfix/` is the exception to that rule: it is a separate forked trace of `root_gainlora`, created specifically so the old method can be rerun with runtime-only compat fixes and without touching the original source tree.
