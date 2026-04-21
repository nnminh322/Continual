#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN="${PYTHON_BIN:-python}"
GENERATOR="$SCRIPT_DIR/generate_root_order1_baseline.py"

MODEL_PATH="meta-llama/Llama-2-7b-hf"
GPU_IDS="0"
MASTER_PORT="49500"

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

GENERATED_SCRIPT=$($PYTHON_BIN "$GENERATOR")
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/root_order1_baseline_$(date +%Y%m%d_%H%M%S).log"
CMD=(bash "$GENERATED_SCRIPT" "$MODEL_PATH" "$GPU_IDS" "$MASTER_PORT")

{
    echo "[ROOT-ORDER1] generated_script=$GENERATED_SCRIPT"
    echo "[ROOT-ORDER1] log_file=$LOG_FILE"
    echo "[ROOT-ORDER1] model_path=$MODEL_PATH"
    echo "[ROOT-ORDER1] gpu_ids=$GPU_IDS"
    echo "[ROOT-ORDER1] master_port=$MASTER_PORT"
    printf '[ROOT-ORDER1] command='
    printf '%q ' "${CMD[@]}"
    printf '\n'
} | tee "$LOG_FILE"

"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"

echo "[ROOT-ORDER1] done. See $LOG_FILE"
