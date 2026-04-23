#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-deepspeed}"
MODEL_PATH="meta-llama/Llama-2-7b-hf"
GPU_IDS="0"
MASTER_PORT="49502"
RUN_NAME="task1572_llama_lora_simple"
OUT_DIR="$SCRIPT_DIR/logs_and_outputs/$RUN_NAME"
LOG_FILE="$LOG_DIR/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
    ${DEEPSPEED_LAUNCHER} --include "localhost:${GPU_IDS}" --master_port "$MASTER_PORT" "$SCRIPT_DIR/run_task1572_llama_lora_simple.py"
    --model_name_or_path "$MODEL_PATH"
    --output_dir "$OUT_DIR"
    --run_name "$RUN_NAME"
    --per_device_train_batch_size 1
    --per_device_eval_batch_size 1
    --gradient_accumulation_steps 32
    --learning_rate 5e-5
    --num_train_epochs 50
    --warmup_steps 0
    --logging_steps 10
    --max_source_length 1024
    --max_target_length 50
    --max_new_tokens 50
    --lora_r 4
    --lora_alpha 32
    --lora_dropout 0.0
    --bf16
)

printf '%q ' "${CMD[@]}" > "$OUT_DIR/launch_command.sh"
printf '\n' >> "$OUT_DIR/launch_command.sh"
chmod +x "$OUT_DIR/launch_command.sh"

{
    echo "[TASK1572-LORA] log_file=$LOG_FILE"
    echo "[TASK1572-LORA] model_path=$MODEL_PATH"
    echo "[TASK1572-LORA] gpu_ids=$GPU_IDS"
    echo "[TASK1572-LORA] master_port=$MASTER_PORT"
    echo "[TASK1572-LORA] deepspeed_launcher=$DEEPSPEED_LAUNCHER"
    echo "[TASK1572-LORA] output_dir=$OUT_DIR"
    printf '[TASK1572-LORA] command='
    printf '%q ' "${CMD[@]}"
    printf '\n'
} | tee "$LOG_FILE"

"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"

echo "[TASK1572-LORA] done. See $LOG_FILE"
