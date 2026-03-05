#!/bin/bash
###############################################################################
# OT-SIGN + GainLoRA (T5-Large) — Order 2 (SuperNI)
# ===================================================
# Task order: glucose → commonsenseqa → diplomacy → ... → emotion (15 tasks)
# Benchmark:  SuperNI (RougeL metric)
# Hardware:   2× T4 16GB  |  Estimated time: ~9-10h
# Run:        bash run_ot_sign_order2_t5large.sh [GPU_IDS] [MODEL_PATH]
###############################################################################

set -e

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
GPU_IDS="${1:-0,1}"
MODEL_PATH="${2:-google/flan-t5-large}"

# ── Config ──────────────────────────────────────────────────────────────────
RUN_NAME="ot_sign_order2_t5large"
OUTPUT_BASE="logs_and_outputs/${RUN_NAME}/outputs"
CONFIG_PREFIX="configs/gen_script_superni_order2_t5_configs"
GEN_DATA_DIR="generated_data/lora_gen_superni_t5"
METRIC_TYPE="rougeL"
KL_RATIO=0.5
EPOCHS=50

BATCH_SIZE=8
EVAL_BATCH=4
GRAD_ACCUM=4
LR=0.0003

OT_EPSILON=0.05
OT_N_ITER=10
DEFAULT_KAPPA=10.0
LAMBDA_DRIFT=0.01
LAMBDA_INV=0.001
THRESHOLD=2.3

# ── Task sequence ────────────────────────────────────────────────────────────
TASKS=(
  "task748_glucose_reverse_cause_event_detection"
  "task073_commonsenseqa_answer_generation"
  "task1590_diplomacy_text_generation"
  "task639_multi_woz_user_utterance_generation"
  "task1572_samsum_summary"
  "task1687_sentiment140_classification"
  "task591_sciq_answer_generation"
  "task363_sst2_polarity_classification"
  "task1510_evalution_relation_extraction"
  "task1729_personachat_generate_next"
  "task181_outcome_extraction"
  "task511_reddit_tifu_long_text_summarization"
  "task002_quoref_answer_generation"
  "task1290_xsum_summarization"
  "task875_emotion_classification"
)
TASK_ORDER=$(IFS=,; echo "${TASKS[*]}")
T=${#TASKS[@]}

mkdir -p "${OUTPUT_BASE}"
echo "════════════════════════════════════════════════"
echo "  OT-SIGN+GainLoRA  |  Order 2  |  T5-Large"
echo "  GPUs: ${GPU_IDS}  |  Tasks: ${T}  |  Epochs: ${EPOCHS}"
echo "════════════════════════════════════════════════"

# ── Training loop ────────────────────────────────────────────────────────────
PREV_LORA_PATHS=""

for i in "${!TASKS[@]}"; do
  TASK="${TASKS[$i]}"
  TASKNUM=$((i + 1))
  OUTDIR="${OUTPUT_BASE}/${TASKNUM}-${TASK}"

  if [[ $i -eq 0 ]]; then
    METRIC_KEY="eval_${METRIC_TYPE}"
  else
    METRIC_KEY="eval_${METRIC_TYPE}_for_${TASK}"
  fi

  echo ""
  echo "──────────────────────────────────────────────"
  echo "  Task ${TASKNUM}/${T}: ${TASK}"
  echo "──────────────────────────────────────────────"

  OT_FLAGS="--use_ot_routing True \
    --ot_epsilon ${OT_EPSILON} \
    --ot_n_iter ${OT_N_ITER} \
    --default_kappa ${DEFAULT_KAPPA} \
    --lambda_drift ${LAMBDA_DRIFT} \
    --lambda_inv ${LAMBDA_INV} \
    --invasion_threshold ${THRESHOLD}"

  PREV_FLAGS=""
  if [[ $i -gt 0 ]]; then
    PREV_OUTDIR="${OUTPUT_BASE}/${i}-${TASKS[$((i-1))]}/saved_weights"
    PREV_FLAGS="--load_checkpoint_from ${PREV_OUTDIR}/trans_input.pt \
      --previous_lora_path ${PREV_LORA_PATHS} \
      --previous_prompt_key_path ${PREV_OUTDIR}/prompts_keys_till_now.pt \
      --previous_vmf_signatures_path ${PREV_OUTDIR}/vmf_signatures.pt \
      --add_instruction_replay \
      --gen_data_dir ${GEN_DATA_DIR} \
      --kl_ratio ${KL_RATIO} \
      --data_replay_freq -1 \
      --replay_after_n_epoch 0"
  fi

  CUDA_VISIBLE_DEVICES=${GPU_IDS} python src/run_t5.py \
    --do_train \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path "${MODEL_PATH}" \
    --data_dir CL_Benchmark \
    --task_order "${TASK_ORDER}" \
    --task_config_dir "${CONFIG_PREFIX}/${TASK}" \
    --output_dir "${OUTDIR}" \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${EVAL_BATCH} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LR} \
    --num_train_epochs ${EPOCHS} \
    --run_name "${RUN_NAME}" \
    --max_source_length 512 \
    --max_target_length 50 \
    --generation_max_length 50 \
    --add_task_name False \
    --add_dataset_name False \
    --overwrite_output_dir \
    --overwrite_cache \
    --lr_scheduler_type constant \
    --warmup_steps 0 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --save_strategy steps \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --metric_for_best_model "${METRIC_KEY}" \
    --lora_r 4 \
    --lora_alpha 32 \
    --lora_dropout 0.0 \
    --attn_temperature 1 \
    --mlp_hidden_dim 100 \
    --model_name gainlora_inflora \
    --threshold 0.995 \
    --transthreshold 0.995 \
    --fp16 \
    ${OT_FLAGS} \
    ${PREV_FLAGS}

  if [[ -z "${PREV_LORA_PATHS}" ]]; then
    PREV_LORA_PATHS="${OUTDIR}/saved_weights"
  else
    PREV_LORA_PATHS="${PREV_LORA_PATHS},${OUTDIR}/saved_weights"
  fi

  echo "  [DONE] Task ${TASKNUM}/${T}: ${TASK}"
done

# ── Final AP/FT ──────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════"
echo "  Computing AP / FT ..."
echo "════════════════════════════════════════════════"
python src/compute_ap_ft.py \
  --output_base "${OUTPUT_BASE}" \
  --task_order "${TASK_ORDER}" \
  --method_name "${RUN_NAME}" \
  --save

echo ""
echo "✓ All done. Results in: logs_and_outputs/${RUN_NAME}/"
