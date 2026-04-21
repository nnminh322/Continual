#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
REPO_DIR="$WORKSPACE_DIR/new_gainlora"
GENERATOR="$SCRIPT_DIR/generate_order1_epoch_variant.py"

EPOCHS="200"
MODEL_PATH="meta-llama/Llama-2-7b-hf"
GPU_IDS="0"
SGWI_FLAG="True"
DUAL_FISHER_FLAG="False"
LAMBDA_EMB=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)       EPOCHS="$2"; shift 2 ;;
        --model_path)   MODEL_PATH="$2"; shift 2 ;;
        --gpu_ids)      GPU_IDS="$2"; shift 2 ;;
        --sgwi)         SGWI_FLAG="$2"; shift 2 ;;
        --dual_fisher)  DUAL_FISHER_FLAG="$2"; shift 2 ;;
        --lambda_emb)   LAMBDA_EMB="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

GENERATED_SCRIPT=$(python "$GENERATOR" --epochs "$EPOCHS")
LOG_FILE="$SCRIPT_DIR/logs/order1_epochs${EPOCHS}_$(date +%Y%m%d_%H%M%S).log"

CMD=(bash "$GENERATED_SCRIPT" "$MODEL_PATH" "$GPU_IDS" --sgwi "$SGWI_FLAG" --dual_fisher "$DUAL_FISHER_FLAG")
if [[ -n "$LAMBDA_EMB" ]]; then
    CMD+=(--lambda_emb "$LAMBDA_EMB")
fi

{
    echo "[ABLAT-ORDER1] generated_script=$GENERATED_SCRIPT"
    echo "[ABLAT-ORDER1] log_file=$LOG_FILE"
    echo "[ABLAT-ORDER1] model_path=$MODEL_PATH"
    echo "[ABLAT-ORDER1] gpu_ids=$GPU_IDS"
    echo "[ABLAT-ORDER1] epochs=$EPOCHS"
    echo "[ABLAT-ORDER1] sgwi=$SGWI_FLAG dual_fisher=$DUAL_FISHER_FLAG lambda_emb=${LAMBDA_EMB:-auto}"
    printf '[ABLAT-ORDER1] command='
    printf '%q ' "${CMD[@]}"
    printf '\n'
} | tee "$LOG_FILE"

pushd "$REPO_DIR" >/dev/null
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
popd >/dev/null

echo "[ABLAT-ORDER1] done. See $LOG_FILE"
