#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
ROOT_BASE="$SCRIPT_DIR/root_gainlora_bugfix"
DATA_DIR="$WORKSPACE_DIR/root_gainlora/CL_Benchmark"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-deepspeed}"
MODEL_PATH="meta-llama/Llama-2-7b-hf"
GPU_IDS="0"
MASTER_PORT="49501"
RUN_NAME="root_task363_gainlora_inflora_bugfix"
OUT_DIR="$SCRIPT_DIR/logs_and_outputs/$RUN_NAME/outputs/1-task363_sst2_polarity_classification"
LOG_FILE="$LOG_DIR/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

# reduce GPU memory pressure for quick local test (override for full runs)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# skip GPM SVD for the first task
export SKIP_GPM=1

resolve_deepspeed_launcher() {
    if command -v "$DEEPSPEED_BIN" >/dev/null 2>&1; then
        printf '%s\n' "$DEEPSPEED_BIN"
        return 0
    fi

    if "$PYTHON_BIN" -c "import deepspeed.launcher.runner" >/dev/null 2>&1; then
        printf '%s\n' "$PYTHON_BIN -m deepspeed.launcher.runner"
        return 0
    fi

    echo "Could not find a runnable DeepSpeed launcher." >&2
    echo "Tried CLI: $DEEPSPEED_BIN" >&2
    echo "Tried module: $PYTHON_BIN -m deepspeed.launcher.runner" >&2
    echo "Set DEEPSPEED_BIN explicitly or install deepspeed into the active environment." >&2
    return 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path)  MODEL_PATH="$2"; shift 2 ;;
        --gpu_ids)     GPU_IDS="$2"; shift 2 ;;
        --master_port) MASTER_PORT="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

mkdir -p "$OUT_DIR"

DEEPSPEED_LAUNCHER=$(resolve_deepspeed_launcher)

CMD=(
    ${DEEPSPEED_LAUNCHER} --include "localhost:${GPU_IDS}" --master_port "$MASTER_PORT" "$ROOT_BASE/src/run_llama.py"
    --do_train
    --do_predict
    --predict_with_generate
    --model_name_or_path "$MODEL_PATH"
    --data_dir "$DATA_DIR"
    --task_order "task363_sst2_polarity_classification,task1572_samsum_summary,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification"
    --task_config_dir "$ROOT_BASE/configs/gen_script_superni_order1_llama_configs/task363_sst2_polarity_classification"
    --output_dir "$OUT_DIR"
    --per_device_train_batch_size 1
    --per_device_eval_batch_size 8
    --gradient_accumulation_steps 16
    --learning_rate 1e-4
    --attn_lr 0.0
    --num_train_epochs 30
    --bf16
    --deepspeed "$ROOT_BASE/configs/ds_configs/stage2.config"
    --run_name "$RUN_NAME"
    --max_source_length 768
    --max_target_length 8
    --generation_max_length 8
    --add_task_name False
    --add_dataset_name False
    --overwrite_output_dir
    --overwrite_cache
    --lr_scheduler_type constant
    --warmup_steps 50
    --logging_strategy steps
    --logging_steps 10
    --metric_for_best_model eval_exact_match
    --eval_strategy steps
    --eval_steps 10
    --save_strategy steps
    --save_steps 10
    --save_total_limit 1
    --load_best_model_at_end
    --lora_r 16
    --lora_alpha 16
    --lora_dropout 0.0
    --data_replay_freq -1
    --replay_after_n_epoch 0
    --kl_ratio 1
    --attn_temperature 1
    --trans_hidden_dim 50
    --chunk 4
    --model_name gainlora_inflora
    --threshold 0.995
    --transthreshold 0.995
)

printf '%q ' "${CMD[@]}" > "$OUT_DIR/launch_command.sh"
printf '\n' >> "$OUT_DIR/launch_command.sh"
chmod +x "$OUT_DIR/launch_command.sh"

{
    echo "[ROOT-T363] log_file=$LOG_FILE"
    echo "[ROOT-T363] model_path=$MODEL_PATH"
    echo "[ROOT-T363] gpu_ids=$GPU_IDS"
    echo "[ROOT-T363] master_port=$MASTER_PORT"
    echo "[ROOT-T363] deepspeed_launcher=$DEEPSPEED_LAUNCHER"
    echo "[ROOT-T363] output_dir=$OUT_DIR"
    printf '[ROOT-T363] command='
    printf '%q ' "${CMD[@]}"
    printf '\n'
} | tee "$LOG_FILE"

"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"

echo "[ROOT-T363] done. See $LOG_FILE"