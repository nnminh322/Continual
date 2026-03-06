#!/bin/bash
# Fast smoke run for early bug detection (2-5 minutes, not 30+ minutes).

set -euo pipefail

GPU_IDS="${1:-0}"
MODEL_PATH="${2:-google/flan-t5-small}"

RUN_NAME="ot_sign_smoke_t5"
OUTDIR="logs_and_outputs/${RUN_NAME}/outputs"
TASK="yelp"
TASK_ORDER="yelp"
CONFIG_DIR="configs/gen_script_long_order3_t5_configs/${TASK}"

echo "[1/2] Running preflight checks..."
python src/preflight_check.py

echo "[2/2] Running smoke training (tiny data, tiny steps)..."
CUDA_VISIBLE_DEVICES="${GPU_IDS}" torchrun --nproc_per_node=1 src/run_t5.py \
  --do_train \
  --do_eval \
  --model_name_or_path "${MODEL_PATH}" \
  --data_dir CL_Benchmark \
  --task_order "${TASK_ORDER}" \
  --task_config_dir "${CONFIG_DIR}" \
  --output_dir "${OUTDIR}" \
  --overwrite_output_dir True \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 3e-4 \
  --num_train_epochs 1 \
  --max_steps 3 \
  --max_num_instances_per_task 8 \
  --max_num_instances_per_eval_task 8 \
  --max_train_samples 8 \
  --max_eval_samples 8 \
  --max_predict_samples 8 \
  --run_name "${RUN_NAME}" \
  --max_source_length 128 \
  --max_target_length 32 \
  --generation_max_length 32 \
  --add_task_name False \
  --add_dataset_name False \
  --lr_scheduler_type constant \
  --warmup_steps 0 \
  --logging_strategy steps \
  --logging_steps 1 \
  --eval_strategy steps \
  --eval_steps 1 \
  --save_strategy no \
  --lora_r 4 \
  --lora_alpha 32 \
  --lora_dropout 0.0 \
  --attn_temperature 1 \
  --mlp_hidden_dim 32 \
  --model_name gainlora_inflora \
  --threshold 0.995 \
  --transthreshold 0.995 \
  --fp16 \
  --gradient_checkpointing True \
  --use_ot_routing True \
  --ot_epsilon 0.05 \
  --ot_n_iter 5 \
  --default_kappa 10.0 \
  --lambda_drift 0.01 \
  --lambda_inv 0.001 \
  --invasion_threshold 2.3

echo "Smoke run done."
