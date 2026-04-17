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
if [ -d "CL_Benchmark/Long_Sequence" ]; then
    DATA_DIR="CL_Benchmark/Long_Sequence"
    echo "[INFO] Using local CL_Benchmark/Long_Sequence"
elif [ -d "/kaggle/input/continual" ]; then
    DATA_DIR="/kaggle/input/continual/CL_Benchmark/Long_Sequence"
    echo "[INFO] Using Kaggle input: $DATA_DIR"
elif [ -d "/content/drive/My Drive/Continual" ]; then
    DATA_DIR="/content/drive/My Drive/Continual/CL_Benchmark/Long_Sequence"
    echo "[INFO] Using Colab Drive: $DATA_DIR"
else
    echo "ERROR: Could not find CL_Benchmark directory!"
    echo "Checked:"
    echo "  - CL_Benchmark/Long_Sequence (local)"
    echo "  - /kaggle/input/continual/CL_Benchmark/Long_Sequence (Kaggle)"
    echo "  - /content/drive/My Drive/Continual/CL_Benchmark/Long_Sequence (Colab)"
    exit 1
fi

CONFIGS=("inflora" "random" "full_lora" "sgwi_freeze_a" "sgwi_train_a" "sgwi_full")

COMMON_ARGS="
    --model_name gainlora_inflora
    --model_name_or_path $MODEL
    --do_train --do_predict
    --per_device_train_batch_size 4
    --gradient_accumulation_steps 8
    --per_device_eval_batch_size 16
    --learning_rate 3e-4
    --num_train_epochs 10
    --max_source_length 512
    --max_target_length 50
    --generation_max_length 50
    --max_num_instances_per_task 10000
    --max_num_instances_per_eval_task 200
    --add_dataset_name False
    --add_task_name False
    --num_examples 0
    --overwrite_output_dir
    --seed 42
    --lora_r 8
    --lora_alpha 16
    --chunk 1
    --threshold 0.99
    --use_srt_router True
    --srt_metric_mode hard
    --srt_max_emb_samples 500
"

train_task() {
    local CONFIG=$1
    local TASK_IDX=$2
    local TASK_NAME=${TASKS[$TASK_IDX]}
    local RUN_NAME="ablation_order${ORDER}_${CONFIG}"
    local OUT_DIR="${BASE}/${CONFIG}/$((TASK_IDX+1))-${TASK_NAME}"
    
    echo "============================================================"
    echo "[Ablation] Config=${CONFIG}, Task=${TASK_NAME} (${TASK_IDX}/${#TASKS[@]})"
    echo "============================================================"
    
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
    local TASK_CONFIG_DIR="configs/Long_Sequence/${TASK_NAME}"
    if [ ! -d "$TASK_CONFIG_DIR" ]; then
        # Try in data_dir
        TASK_CONFIG_DIR="${DATA_DIR}/../config_${TASK_NAME}"
        if [ ! -d "$TASK_CONFIG_DIR" ]; then
            echo "ERROR: Could not find task config for $TASK_NAME"
            return 1
        fi
    fi
    
    local CMD="python src/run_t5.py \
        $COMMON_ARGS \
        --data_dir $DATA_DIR \
        --task_config_dir $TASK_CONFIG_DIR \
        --task_order $TASK_ORDER \
        --output_dir $OUT_DIR \
        --run_name $RUN_NAME \
        --sgwi_mode $CONFIG \
        --lambda_emb 0.0"
    
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
