#!/bin/bash
#SBATCH -J srt
#SBATCH -o srt-%j.out
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 20:00:00
#SBATCH --mem 128G
#SBATCH --gres=gpu:a100-sxm4-80gb:1

export CUDA_DEVICE_ORDER="PCI_BUS_ID"

# ============================================================
# Parse named arguments: --sgwi, --dual_fisher, --lambda_emb
# Usage: bash script.sh <GPU_ID> <MODEL_PATH> [--sgwi true/false] [--dual_fisher true/false] [--lambda_emb 0.01]
# ============================================================
SGWI_FLAG="True"
DUAL_FISHER_FLAG="False"
LAMBDA_EMB=""

POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --sgwi)        SGWI_FLAG="$2";        shift 2 ;;
        --dual_fisher) DUAL_FISHER_FLAG="$2"; shift 2 ;;
        --lambda_emb)  LAMBDA_EMB="$2";       shift 2 ;;
        *)             POSITIONAL+=("$1");    shift ;;
    esac
done
set -- "${POSITIONAL[@]}"

LAMBDA_ARG=""
if [ -n "$LAMBDA_EMB" ]; then
    LAMBDA_ARG="--lambda_emb $LAMBDA_EMB"
fi

echo "============================================================"
echo "[CONFIG] sgwi=$SGWI_FLAG, dual_fisher=$DUAL_FISHER_FLAG, lambda_emb=${LAMBDA_EMB:-auto}"
echo "============================================================"

# ============================================================
# Auto-detect GPU type
# ============================================================
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)

if [ -z "$GPU_MEM" ]; then
    echo "ERROR: No GPU detected! Defaulting."
    GPU_MEM=16000; NUM_GPUS=1
fi

if [ "$GPU_MEM" -lt 20000 ]; then
    GPU_TIER="t4"
    echo "[GPU] Detected T4-class (${GPU_MEM}MB VRAM)"
elif [ "$GPU_MEM" -lt 50000 ]; then
    GPU_TIER="mid"
    echo "[GPU] Detected mid-class (${GPU_MEM}MB VRAM, e.g. 3090/4090/5090)"
else
    GPU_TIER="a100"
    echo "[GPU] Detected high-mem (${GPU_MEM}MB VRAM, e.g. A100/H100)"
fi

if [ "$GPU_TIER" = "t4" ] && [ "$NUM_GPUS" -ge 2 ]; then
    GPU_MODE="t4_2gpu"; GPU_IDS="0,1"
    FP16_FLAG="--gradient_checkpointing"
    echo "[GPU] Strategy: 2x T4 DataParallel + gradient_checkpointing"
elif [ "$GPU_TIER" = "t4" ]; then
    GPU_MODE="t4_1gpu"; GPU_IDS="${1:-0}"
    FP16_FLAG="--gradient_checkpointing"
    echo "[GPU] Strategy: 1x T4 + gradient_checkpointing"
elif [ "$GPU_TIER" = "mid" ]; then
    GPU_MODE="mid"; GPU_IDS="${1:-0}"
    FP16_FLAG="--gradient_checkpointing"
    echo "[GPU] Strategy: mid-GPU + gradient_checkpointing"
else
    GPU_MODE="a100"; GPU_IDS="${1:-0}"
    FP16_FLAG=""
    echo "[GPU] Strategy: A100 (single GPU, fp32)"
fi

echo "[GPU] Using CUDA_VISIBLE_DEVICES=$GPU_IDS"
echo "============================================================"
echo ""

MODEL_NAME_ARG="${2:-google/flan-t5-large}"

SRT_FLAGS="--use_srt_router True --sgwi $SGWI_FLAG --dual_fisher $DUAL_FISHER_FLAG $LAMBDA_ARG --srt_shrink False --srt_metric_mode hard --srt_max_emb_samples 200"

is_task_complete() {
    local task_dir="$1"
    local required=(
        "$task_dir/all_results.json"
        "$task_dir/saved_weights/trans_input.pt"
        "$task_dir/saved_weights/lora_weights_A.pt"
        "$task_dir/saved_weights/lora_weights_B.pt"
        "$task_dir/saved_weights/prompts_keys_till_now.pt"
        "$task_dir/saved_weights/srt_signatures.npz"
    )

    local marker
    for marker in "${required[@]}"; do
        if [ ! -f "$marker" ]; then
            return 1
        fi
    done
    return 0
}

print_missing_task_markers() {
    local task_dir="$1"
    local required=(
        "$task_dir/all_results.json"
        "$task_dir/saved_weights/trans_input.pt"
        "$task_dir/saved_weights/lora_weights_A.pt"
        "$task_dir/saved_weights/lora_weights_B.pt"
        "$task_dir/saved_weights/prompts_keys_till_now.pt"
        "$task_dir/saved_weights/srt_signatures.npz"
    )

    local marker
    for marker in "${required[@]}"; do
        if [ ! -f "$marker" ]; then
            echo "         missing: $marker"
        fi
    done
}

# ============================================================
# Task config
# ============================================================
TASK_ORDER="yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic"
BASE_OUTPUT_DIR="logs_and_outputs/long_order3_t5_srt/outputs"
RUN_NAME="long_order3_t5_srt"
CONFIG_DIR="configs/gen_script_long_order3_t5_configs"
GEN_DATA_DIR="generated_data/lora_gen_long_t5"

IFS=',' read -ra TASKS <<< "$TASK_ORDER"
NUM_TASKS=${#TASKS[@]}

# ============================================================
# Common fixed args
# ============================================================
COMMON_ARGS=(
    --do_train
    --do_predict
    --predict_with_generate
    --model_name_or_path "$MODEL_NAME_ARG"
    --data_dir CL_Benchmark
    --task_order "$TASK_ORDER"
    --gen_data_dir "$GEN_DATA_DIR"
    --max_source_length 512
    --max_target_length 50
    --generation_max_length 50
    --add_task_name False
    --add_dataset_name False
    --overwrite_cache
    --learning_rate 0.0003
    --num_train_epochs 10
    --run_name "$RUN_NAME"
    --lr_scheduler_type constant
    --warmup_steps 0
    --logging_strategy steps
    --logging_steps 10
    --evaluation_strategy steps
    --save_strategy steps
    --save_total_limit 1
    --lora_r 8
    --lora_alpha 32
    --lora_dropout 0.0
    --data_replay_freq -1
    --mlp_hidden_dim 100
    --model_name gainlora
    --threshold 0.995
    --transthreshold 0.995
    $FP16_FLAG
    $SRT_FLAGS
)

# ============================================================
# Loop over all tasks
# ============================================================
PREV_SAVE_DIRS=""

for ((i=0; i<NUM_TASKS; i++)); do
    TASK="${TASKS[$i]}"
    TASK_NUM=$((i+1))
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${TASK_NUM}-${TASK}"

    # ── SMART SKIP ────────────────────────────────────────
    if is_task_complete "$OUTPUT_DIR"; then
        echo "============================================================"
        echo "✓ SKIP  [${TASK_NUM}/${NUM_TASKS}] $TASK"
        echo "         (all_results + saved_weights complete)"
        echo "============================================================"
        if [ -z "$PREV_SAVE_DIRS" ]; then
            PREV_SAVE_DIRS="${OUTPUT_DIR}/saved_weights"
        else
            PREV_SAVE_DIRS="${PREV_SAVE_DIRS},${OUTPUT_DIR}/saved_weights"
        fi
        continue
    fi

    if [ -d "$OUTPUT_DIR" ]; then
        echo "============================================================"
        echo "! INCOMPLETE [${TASK_NUM}/${NUM_TASKS}] $TASK"
        echo "         output_dir exists but completion markers are missing"
        print_missing_task_markers "$OUTPUT_DIR"
        echo "============================================================"
    fi

    echo "============================================================"
    echo "▶ START  [${TASK_NUM}/${NUM_TASKS}] $TASK"
    echo "         output_dir: $OUTPUT_DIR"
    echo "         mode: restart-from-scratch (matches original script)"
    echo "============================================================"

    # ── Batch sizes per GPU mode (task 1 vs task 2+) ─────
    if [ "$i" -eq 0 ]; then
        # Task 1 — slightly smaller BSZ (no replay overhead)
        if [ "$GPU_MODE" = "t4_2gpu" ]; then
            BSZ=2; GA=8; EVAL_BSZ=16
        elif [ "$GPU_MODE" = "t4_1gpu" ]; then
            BSZ=4; GA=8; EVAL_BSZ=16
        elif [ "$GPU_MODE" = "mid" ]; then
            BSZ=8; GA=4; EVAL_BSZ=64
        else
            BSZ=8; GA=4; EVAL_BSZ=128
        fi
    else
        # Task 2+ — larger BSZ when VRAM allows
        if [ "$GPU_MODE" = "t4_2gpu" ]; then
            BSZ=2; GA=4; EVAL_BSZ=16
        elif [ "$GPU_MODE" = "t4_1gpu" ]; then
            BSZ=4; GA=8; EVAL_BSZ=16
        elif [ "$GPU_MODE" = "mid" ]; then
            BSZ=8; GA=4; EVAL_BSZ=64
        else
            BSZ=16; GA=2; EVAL_BSZ=128
        fi
    fi

    # ── Task metric ───────────────────────────────────────
    if [ "$i" -eq 0 ]; then
        METRIC="eval_exact_match"
    else
        METRIC="eval_exact_match_for_${TASK}"
        PREV_TASK_DIR="${BASE_OUTPUT_DIR}/$((i))-${TASKS[$((i-1))]}"
        if ! is_task_complete "$PREV_TASK_DIR"; then
            echo "ERROR: Previous task is incomplete: $PREV_TASK_DIR"
            print_missing_task_markers "$PREV_TASK_DIR"
            exit 1
        fi
    fi

    # ── Build per-task args ───────────────────────────────
    TASK_ARGS=(
        --task_config_dir "${CONFIG_DIR}/${TASK}"
        --output_dir "$OUTPUT_DIR"
        --metric_for_best_model "$METRIC"
        --per_device_train_batch_size $BSZ
        --per_device_eval_batch_size $EVAL_BSZ
        --gradient_accumulation_steps $GA
    )

    if [ "$i" -eq 0 ]; then
        # Task 1: no previous weights, instruction replay
        TASK_ARGS+=(
            --add_instruction_replay
            --replay_after_n_epoch 0
            --overwrite_output_dir
        )
    else
        # Task 2+: load previous weights, kl_ratio, srt_load_path
        TASK_ARGS+=(
            --load_checkpoint_from "${PREV_TASK_DIR}/saved_weights/trans_input.pt"
            --previous_lora_path "$PREV_SAVE_DIRS"
            --previous_prompt_key_path "${PREV_TASK_DIR}/saved_weights/prompts_keys_till_now.pt"
            --kl_ratio 0.1
            --attn_temperature 1
            --srt_load_path "${PREV_TASK_DIR}/saved_weights"
            --overwrite_output_dir
        )
    fi

    # ── Run training ──────────────────────────────────────
    set -x
    CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py "${COMMON_ARGS[@]}" "${TASK_ARGS[@]}"
    EXIT_CODE=$?
    set +x

    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: Task ${TASK_NUM}-${TASK} failed with exit code $EXIT_CODE"
        exit $EXIT_CODE
    fi

    # ── Cleanup & bookkeeping ─────────────────────────────
    rm -rf "${OUTPUT_DIR}"/checkpoint*
    sleep 5

    if [ -z "$PREV_SAVE_DIRS" ]; then
        PREV_SAVE_DIRS="${OUTPUT_DIR}/saved_weights"
    else
        PREV_SAVE_DIRS="${PREV_SAVE_DIRS},${OUTPUT_DIR}/saved_weights"
    fi

    echo ""
done

echo "============================================================"
echo "ALL TASKS COMPLETE"
echo "============================================================"
