#!/bin/bash
# =============================================================================
# LLaMA GainLoRA Continual Learning — SuperNI Order 1 (15 tasks)
# Default: SRT + SGWI warm-init (matching T5 gold standard)
#
# Usage:
#   bash run_superni_order1_llama_cl.sh              # SRT + SGWI (default)
#   bash run_superni_order1_llama_cl.sh --no_sgwi    # SRT only, no SGWI
#
# Each task is run sequentially, accumulating previous LoRA paths and
# SRT signatures from the previous task's saved_weights/.
#
# Directory layout (score.py compatible, 1-indexed):
#   logs_and_outputs/{RUN_NAME}/outputs/1-task1572.../all_results.json
#   logs_and_outputs/{RUN_NAME}/outputs/task_order.txt
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"meta-llama/Llama-2-7b-hf"}
CONTINUAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_DIR="${CONTINUAL_DIR}/root_gainlora/CL_Benchmark"
CONFIG_BASE="${CONTINUAL_DIR}/new_gainlora/configs/gen_script_superni_order1_llama_configs"
DS_CONFIG=${DS_CONFIG:-"${CONTINUAL_DIR}/new_gainlora/configs/ds_configs/stage2_cpu_offload.config"}
LOG_DIR="${SCRIPT_DIR}/logs_and_outputs"
RUN_NAME=${RUN_NAME:-"superni_order1_llama_srt"}

# DeepSpeed settings
NUM_GPUS=${NUM_GPUS:-1}

# Training hyperparameters
LORA_R=${LORA_R:-4}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE:-1}   # quality-neutral VRAM reduction
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-32}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-15}
MAX_SOURCE_LENGTH=${MAX_SOURCE_LENGTH:-1024}
MAX_TARGET_LENGTH=${MAX_TARGET_LENGTH:-50}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-50}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-}     # default: no cap; set e.g. 500 for quick runs
SRT_SHRINKAGE=${SRT_SHRINKAGE:-"ridge"}    # PooledMahalanobis shrinkage method
SRT_MAX_EMB_SAMPLES=${SRT_MAX_EMB_SAMPLES:-2000}  # increased for better Σ estimate (was 200)
SRT_PCA_COMPONENTS=${SRT_PCA_COMPONENTS:-}  # PCA dims before Mahalanobis (e.g. 128); empty = no PCA
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-1}

# Parse extra flags (e.g. --no_sgwi)
EXTRA_FLAGS="$*"

# ── Task order (15 tasks, SuperNI Order 1) ─────────────────────────────────────
TASK_ORDER=(
    "task1572_samsum_summary"
    "task363_sst2_polarity_classification"
    "task1290_xsum_summarization"
    "task181_outcome_extraction"
    "task002_quoref_answer_generation"
    "task1510_evalution_relation_extraction"
    "task639_multi_woz_user_utterance_generation"
    "task1729_personachat_generate_next"
    "task073_commonsenseqa_answer_generation"
    "task1590_diplomacy_text_generation"
    "task748_glucose_reverse_cause_event_detection"
    "task511_reddit_tifu_long_text_summarization"
    "task591_sciq_answer_generation"
    "task1687_sentiment140_classification"
    "task875_emotion_classification"
)

TASK_ORDER_STR=$(IFS=','; echo "${TASK_ORDER[*]}")
NUM_TASKS=${#TASK_ORDER[@]}

echo "=================================================="
echo "LLaMA GainLoRA Continual Learning — SuperNI Order 1"
echo "MODEL:     ${MODEL_NAME_OR_PATH}"
echo "TASKS:     ${NUM_TASKS}"
echo "RUN_NAME:  ${RUN_NAME}"
echo "LOG_DIR:   ${LOG_DIR}"
echo "DS_CONFIG: ${DS_CONFIG}"
echo "GRAD_CKPT:${GRADIENT_CHECKPOINTING}"
echo "EXTRA:     ${EXTRA_FLAGS}"
echo "=================================================="

# Output root (score.py compatible): logs_and_outputs/{RUN_NAME}/outputs/
OUTPUTS_DIR="${LOG_DIR}/${RUN_NAME}/outputs"
mkdir -p "${OUTPUTS_DIR}"

# ── Sequential task loop ───────────────────────────────────────────────────────
PREVIOUS_LORA_PATHS=""   # accumulated comma-sep list (oldest→newest)
PREV_SRT_PATH=""         # srt_signatures.npz dir from last task

for ((TASK_ID=0; TASK_ID<NUM_TASKS; TASK_ID++)); do
    TASK_NAME="${TASK_ORDER[$TASK_ID]}"
    TASK_CONFIG_DIR="${CONFIG_BASE}/${TASK_NAME}"
    TASK_NUM=$((TASK_ID + 1))  # 1-indexed for score.py compatibility
    OUTPUT_DIR="${OUTPUTS_DIR}/${TASK_NUM}-${TASK_NAME}"

    echo ""
    echo "──────────────────────────────────────────────────"
    echo "TASK ${TASK_NUM}/${NUM_TASKS}: ${TASK_NAME}"
    echo "OUTPUT:  ${OUTPUT_DIR}"
    echo "PREV LORA: ${PREVIOUS_LORA_PATHS:-<none>}"
    echo "SRT PATH:  ${PREV_SRT_PATH:-<none>}"
    echo "──────────────────────────────────────────────────"

    mkdir -p "${OUTPUT_DIR}"

    # Build optional args
    PREV_LORA_ARG=""
    if [ -n "${PREVIOUS_LORA_PATHS}" ]; then
        PREV_LORA_ARG="--previous_lora_path ${PREVIOUS_LORA_PATHS}"
    fi

    SRT_LOAD_ARG=""
    if [ -n "${PREV_SRT_PATH}" ]; then
        SRT_LOAD_ARG="--srt_load_path ${PREV_SRT_PATH}"
    fi

    GC_ARG=""
    if [ "${GRADIENT_CHECKPOINTING}" != "0" ]; then
        GC_ARG="--gradient_checkpointing"
    else
        GC_ARG="--no_gradient_checkpointing"
    fi

    # Run task
    deepspeed \
        --num_gpus=${NUM_GPUS} \
        --master_port $((29500 + TASK_ID)) \
        "${SCRIPT_DIR}/run_llama_gainlora_cl.py" \
        --model_name_or_path     "${MODEL_NAME_OR_PATH}" \
        --data_dir               "${DATA_DIR}" \
        --task_config_dir        "${TASK_CONFIG_DIR}" \
        --output_dir             "${OUTPUT_DIR}" \
        --cur_task_id            ${TASK_ID} \
        --task_order             "${TASK_ORDER_STR}" \
        --lora_r                 ${LORA_R} \
        --lora_alpha             ${LORA_ALPHA} \
        --lora_dropout           ${LORA_DROPOUT} \
        --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
        --per_device_eval_batch_size  ${PER_DEVICE_EVAL_BATCH_SIZE} \
        --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
        --learning_rate          ${LEARNING_RATE} \
        --num_train_epochs       ${NUM_TRAIN_EPOCHS} \
        --max_source_length      ${MAX_SOURCE_LENGTH} \
        --max_target_length      ${MAX_TARGET_LENGTH} \
        --max_new_tokens         ${MAX_NEW_TOKENS} \
        --srt_shrinkage          ${SRT_SHRINKAGE} \
        --srt_max_emb_samples    ${SRT_MAX_EMB_SAMPLES} \
        ${SRT_PCA_COMPONENTS:+--srt_pca_components ${SRT_PCA_COMPONENTS}} \
        --use_srt_router \
        ${GC_ARG} \
        --bf16 \
        --deepspeed              "${DS_CONFIG}" \
        --logging_steps          10 \
        --seed                   42 \
        ${PREV_LORA_ARG} \
        ${SRT_LOAD_ARG} \
        ${MAX_TRAIN_SAMPLES:+--max_train_samples ${MAX_TRAIN_SAMPLES}} \
        ${EXTRA_FLAGS} \
    2>&1 | tee "${OUTPUT_DIR}/train.log"

    # Accumulate previous LoRA paths for next task
    CUR_SAVED="${OUTPUT_DIR}/saved_weights"

    if [ -z "${PREVIOUS_LORA_PATHS}" ]; then
        PREVIOUS_LORA_PATHS="${CUR_SAVED}"
    else
        PREVIOUS_LORA_PATHS="${PREVIOUS_LORA_PATHS},${CUR_SAVED}"
    fi
    PREV_SRT_PATH="${CUR_SAVED}"

    echo "[Done] Task ${TASK_NUM}: saved → ${CUR_SAVED}"
done

echo ""
echo "=================================================="
echo "ALL ${NUM_TASKS} TASKS COMPLETE"
echo "=================================================="

# ── Score (uses same score.py as T5) ──────────────────────────────────────────
echo "Computing CL metrics via score.py ..."
python "${CONTINUAL_DIR}/new_gainlora/score.py" "${RUN_NAME}" "${RUN_NAME}" "${LOG_DIR}" || \
    echo "[WARN] score.py failed — check that all tasks produced all_results.json"

echo "Done."
