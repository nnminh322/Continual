#!/bin/bash
# ==============================================================================
# 6-Config Ablation: Kaggle-compatible version
# Adjusts paths for Kaggle/Colab environment
# ==============================================================================

set -e

if [ ! -f "src/run_t5.py" ]; then
    echo "ERROR: Must run from new_gainlora/ directory!"
    exit 1
fi

GPU_ID=${1:-0}
MODEL=${2:-google/flan-t5-large}
ORDER=${3:-4}

export CUDA_VISIBLE_DEVICES=$GPU_ID
BASE="logs_and_outputs/ablation_order${ORDER}"

# ── Task sequences ────────────────────────────────────────────────
if [ "$ORDER" = "4" ]; then
    TASKS=("mnli" "cb")
    TASK_ORDER="mnli,cb"
elif [ "$ORDER" = "3" ]; then
    TASKS=("yelp" "amazon" "mnli" "cb")
    TASK_ORDER="yelp,amazon,mnli,cb"
else
    echo "ERROR: ORDER must be 3 or 4"
    exit 1
fi

# Detect data dir (try multiple locations)
# NOTE: cl_dataset.py expects path to CL_Benchmark root (not Long_Sequence subdir)
# because task_config has top-level key "Long_Sequence"
if [ -d "CL_Benchmark" ]; then
    DATA_DIR="CL_Benchmark"
    echo "[INFO] Using local CL_Benchmark"
elif [ -d "/kaggle/input/continual" ]; then
    DATA_DIR="/kaggle/input/continual/CL_Benchmark"
    echo "[INFO] Using Kaggle input: $DATA_DIR"
elif [ -d "/content/drive/My Drive/Continual" ]; then
    DATA_DIR="/content/drive/My Drive/Continual/CL_Benchmark"
    echo "[INFO] Using Colab Drive: $DATA_DIR"
else
    echo "ERROR: Could not find CL_Benchmark directory!"
    echo "Checked:"
    echo "  - CL_Benchmark (local)"
    echo "  - /kaggle/input/continual/CL_Benchmark (Kaggle)"
    echo "  - /content/drive/My Drive/Continual/CL_Benchmark (Colab)"
    exit 1
fi

CONFIGS=("inflora" "random" "full_lora" "sgwi_freeze_a" "sgwi_train_a" "sgwi_full")

# ── P100 (16GB) — FAST ablation settings ──────────────────────────
# Aligned with working gen_script_long_order4_t5_srt.sh
# Only reduced: train_samples=1000, epochs=5, batch tuned for P100
TRAIN_BATCH=4
GRAD_ACC=8
EVAL_BATCH=2

# CUDA allocator: reduce fragmentation on P100
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON_ARGS_BASE="
    --model_name gainlora_inflora
    --model_name_or_path $MODEL
    --per_device_train_batch_size $TRAIN_BATCH
    --gradient_accumulation_steps $GRAD_ACC
    --per_device_eval_batch_size $EVAL_BATCH
    --predict_with_generate
    --learning_rate 3e-4
    --num_train_epochs 5
    --lr_scheduler_type constant
    --warmup_steps 0
    --max_source_length 512
    --max_target_length 50
    --generation_max_length 50
    --max_num_instances_per_task 1000
    --max_num_instances_per_eval_task 200
    --max_predict_samples 200
    --overwrite_cache
    --add_dataset_name False
    --add_task_name False
    --num_examples 0
    --seed 42
    --logging_strategy steps
    --logging_steps 10
    --save_total_limit 1
    --lora_r 8
    --lora_alpha 32
    --lora_dropout 0.0
    --data_replay_freq -1
    --replay_after_n_epoch 0
    --mlp_hidden_dim 100
    --chunk 1
    --threshold 0.99
    --transthreshold 0.99
    --use_srt_router True
    --srt_metric_mode hard
    --srt_shrink False
    --srt_max_emb_samples 200
"

# ── Smart skip: detect task state ─────────────────────────────────
# Returns:
#   "done"          → train + eval complete, skip
#   "eval_only"     → train done, eval pending → run --do_predict only
#   "fresh"         → nothing done → run --do_train --do_predict
task_state() {
    local OUT_DIR=$1
    local SAVED="${OUT_DIR}/saved_weights"
    local TRAIN_DONE=false
    local EVAL_DONE=false

    # Training done markers: lora weights + SRT signatures saved
    if [ -f "${SAVED}/lora_weights_A.pt" ] && [ -f "${SAVED}/lora_weights_B.pt" ]; then
        TRAIN_DONE=true
    fi

    # Eval done markers: predict metrics written
    if [ -f "${OUT_DIR}/predict_results.json" ] || [ -f "${OUT_DIR}/all_results.json" ]; then
        EVAL_DONE=true
    fi

    if $TRAIN_DONE && $EVAL_DONE; then
        echo "done"
    elif $TRAIN_DONE; then
        echo "eval_only"
    else
        echo "fresh"
    fi
}

train_task() {
    local CONFIG=$1
    local TASK_IDX=$2
    local TASK_NAME=${TASKS[$TASK_IDX]}
    local RUN_NAME="ablation_order${ORDER}_${CONFIG}"
    local OUT_DIR="${BASE}/${CONFIG}/$((TASK_IDX+1))-${TASK_NAME}"

    # ── Smart skip detection ──────────────────────────────
    local STATE=$(task_state "$OUT_DIR")
    echo "============================================================"
    echo "[Ablation] Config=${CONFIG}, Task=${TASK_NAME} [${TASK_IDX}/${#TASKS[@]}] — state=${STATE}"
    echo "============================================================"

    if [ "$STATE" = "done" ]; then
        echo "  [SKIP] Already complete. Skipping."
        return 0
    fi

    # ── Build path arguments ──────────────────────────────
    local PREV_LORA=""
    local LOAD_CKPT=""
    local PREV_KEYS=""
    local SRT_LOAD=""

    if [ $TASK_IDX -gt 0 ]; then
        for ((i=0; i<TASK_IDX; i++)); do
            local prev_task=${TASKS[$i]}
            local prev_dir="${BASE}/${CONFIG}/$((i+1))-${prev_task}/saved_weights"
            if [ -z "$PREV_LORA" ]; then
                PREV_LORA="$prev_dir"
            else
                PREV_LORA="${PREV_LORA},${prev_dir}"
            fi
        done

        local prev_idx=$((TASK_IDX-1))
        local prev_task=${TASKS[$prev_idx]}
        LOAD_CKPT="${BASE}/${CONFIG}/$((prev_idx+1))-${prev_task}/saved_weights/trans_input.pt"
        PREV_KEYS="${BASE}/${CONFIG}/$((prev_idx+1))-${prev_task}/saved_weights/prompts_keys_till_now.pt"
        SRT_LOAD="${BASE}/${CONFIG}/$((prev_idx+1))-${prev_task}/saved_weights"
    fi

    # ── Build task_config_dir path ────────────────────────
    # Working gen scripts use: configs/gen_script_long_order{N}_t5_configs/{task}
    local TASK_CONFIG_DIR="configs/gen_script_long_order${ORDER}_t5_configs/${TASK_NAME}"
    if [ ! -d "$TASK_CONFIG_DIR" ]; then
        # Fallback: try order3 configs (yelp,amazon,mnli,cb are there too)
        TASK_CONFIG_DIR="configs/gen_script_long_order3_t5_configs/${TASK_NAME}"
    fi
    if [ ! -d "$TASK_CONFIG_DIR" ]; then
        echo "ERROR: Could not find task config dir!"
        echo "  Tried: configs/gen_script_long_order${ORDER}_t5_configs/${TASK_NAME}"
        echo "  Tried: configs/gen_script_long_order3_t5_configs/${TASK_NAME}"
        return 1
    fi
    echo "  [CONFIG] Using task_config_dir=$TASK_CONFIG_DIR"

    # ── Decide do_train / do_predict flags ────────────────
    local MODE_FLAGS=""
    local CURRENT_LORA_FLAG=""
    if [ "$STATE" = "eval_only" ]; then
        echo "  [RESUME] Training done. Running eval-only."
        MODE_FLAGS="--do_predict"
        CURRENT_LORA_FLAG="--current_lora_path ${OUT_DIR}/saved_weights"
    else
        MODE_FLAGS="--do_train --do_predict --overwrite_output_dir"
    fi

    # ── Build command ──────────────────────────────────────
    local CMD="python src/run_t5.py \
        $COMMON_ARGS_BASE \
        $MODE_FLAGS \
        --data_dir $DATA_DIR \
        --task_config_dir $TASK_CONFIG_DIR \
        --task_order $TASK_ORDER \
        --output_dir $OUT_DIR \
        --run_name $RUN_NAME \
        --sgwi_mode $CONFIG \
        --lambda_emb 0.0"

    if [ -n "$CURRENT_LORA_FLAG" ]; then
        CMD="$CMD $CURRENT_LORA_FLAG"
    fi

    if [ -n "$PREV_LORA" ]; then
        CMD="$CMD --previous_lora_path $PREV_LORA"
    fi
    if [ -n "$LOAD_CKPT" ] && [ -f "$LOAD_CKPT" ]; then
        CMD="$CMD --load_checkpoint_from $LOAD_CKPT"
    fi
    if [ -n "$PREV_KEYS" ] && [ -f "$PREV_KEYS" ]; then
        CMD="$CMD --previous_prompt_key_path $PREV_KEYS"
    fi
    if [ -n "$SRT_LOAD" ] && [ -d "$SRT_LOAD" ]; then
        CMD="$CMD --srt_load_path $SRT_LOAD"
    fi

    echo "[CMD] $CMD"
    eval $CMD
    echo "[Done] Config=${CONFIG}, Task=${TASK_NAME}"
}

echo "============================================================"
echo " 6-Config Ablation Study — Order ${ORDER} (Kaggle)"
echo " Tasks: ${TASKS[*]}"
echo " Data Dir: $DATA_DIR"
echo " GPU: $GPU_ID"
echo "============================================================"

for CONFIG in "${CONFIGS[@]}"; do
    echo ""
    echo "########################################################"
    echo "# Starting Config: $CONFIG"
    echo "########################################################"
    
    for ((t=0; t<${#TASKS[@]}; t++)); do
        train_task "$CONFIG" "$t"
    done
    
    echo "[Config $CONFIG] All ${#TASKS[@]} tasks complete."
done

echo ""
echo "============================================================"
echo " All 6 configs complete!"
echo " Results in: ${BASE}/"
echo "============================================================"
