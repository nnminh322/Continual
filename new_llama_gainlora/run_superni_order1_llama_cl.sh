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
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"meta-llama/Llama-2-7b-hf"}
CONTINUAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_DIR="${CONTINUAL_DIR}/root_gainlora/CL_Benchmark"
# Use local canonical configs to avoid depending on llama_epoch_ablation
CONFIG_BASE="${CONTINUAL_DIR}/new_gainlora/configs/gen_script_superni_order1_llama_configs"
DS_CONFIG="${CONTINUAL_DIR}/new_gainlora/configs/ds_configs/stage2.config"
LOG_DIR="${SCRIPT_DIR}/logs_and_outputs"

# DeepSpeed settings
NUM_GPUS=${NUM_GPUS:-1}

# Training hyperparameters (match T5 GainLoRA standard)
LORA_R=${LORA_R:-4}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-32}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-15}   # T5 gold: 100 epochs; set e.g. 10 for quick runs
MAX_SOURCE_LENGTH=${MAX_SOURCE_LENGTH:-1024}
MAX_TARGET_LENGTH=${MAX_TARGET_LENGTH:-50}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-50}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-}     # default: no cap (T5 gold); set e.g. 500 for quick runs
SRT_METRIC_MODE=${SRT_METRIC_MODE:-"hard"}

# Parse extra flags (e.g. --sgwi)
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
echo "LOG_DIR:   ${LOG_DIR}"
echo "EXTRA:     ${EXTRA_FLAGS}"
echo "=================================================="

mkdir -p "${LOG_DIR}"

# ── Sequential task loop ───────────────────────────────────────────────────────
PREVIOUS_LORA_PATHS=""   # accumulated comma-sep list (oldest→newest)
PREV_SRT_PATH=""         # srt_signatures.npz dir from last task

for ((TASK_ID=0; TASK_ID<NUM_TASKS; TASK_ID++)); do
    TASK_NAME="${TASK_ORDER[$TASK_ID]}"
    TASK_CONFIG_DIR="${CONFIG_BASE}/${TASK_NAME}"
    OUTPUT_DIR="${LOG_DIR}/${TASK_ID}-${TASK_NAME}"

    echo ""
    echo "──────────────────────────────────────────────────"
    echo "TASK ${TASK_ID}/${NUM_TASKS}: ${TASK_NAME}"
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
        --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
        --learning_rate          ${LEARNING_RATE} \
        --num_train_epochs       ${NUM_TRAIN_EPOCHS} \
        --max_source_length      ${MAX_SOURCE_LENGTH} \
        --max_target_length      ${MAX_TARGET_LENGTH} \
        --max_new_tokens         ${MAX_NEW_TOKENS} \
        --srt_metric_mode        ${SRT_METRIC_MODE} \
        --use_srt_router \
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

    echo "[Done] Task ${TASK_ID}: saved → ${CUR_SAVED}"
done

echo ""
echo "=================================================="
echo "ALL ${NUM_TASKS} TASKS COMPLETE"
echo "=================================================="

# ── Aggregate final metrics ────────────────────────────────────────────────────
RESULTS_FILE="${LOG_DIR}/all_results.json"
echo "Aggregating results → ${RESULTS_FILE}"

python3 - << 'PYEOF'
import json
import glob
import os
import sys

log_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "logs_and_outputs"
)

results = []
for metrics_path in sorted(glob.glob(os.path.join(log_dir, "*", "final_metrics.json"))):
    with open(metrics_path) as f:
        m = json.load(f)
    results.append(m)

print(f"Found {len(results)} task results.")
if results:
    avg_dev_rougeL  = sum(m.get("dev_rougeL",  0) for m in results) / len(results)
    avg_test_rougeL = sum(m.get("test_rougeL", 0) for m in results) / len(results)
    print(f"Avg dev_rougeL:  {avg_dev_rougeL:.4f}")
    print(f"Avg test_rougeL: {avg_test_rougeL:.4f}")

with open(os.path.join(log_dir, "all_results.json"), "w") as f:
    json.dump({"tasks": results, "avg_dev_rougeL": avg_dev_rougeL,
               "avg_test_rougeL": avg_test_rougeL}, f, indent=2)
PYEOF "${LOG_DIR}"

echo "Done."
