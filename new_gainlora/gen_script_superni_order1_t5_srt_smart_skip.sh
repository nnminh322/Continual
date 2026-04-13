#!/bin/bash
set -e
#SBATCH -J srt
#SBATCH -o srt-%j.out
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 20:00:00
#SBATCH --mem 128G
#SBATCH --gres=gpu:a100-sxm4-80gb:1

export CUDA_DEVICE_ORDER="PCI_BUS_ID"

# ============================================================
# Auto-detect GPU count and type for optimal parallelism
# ============================================================
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)

if [ -z "$GPU_MEM" ]; then
    echo "ERROR: No GPU detected!"
    GPU_MEM=16000
    NUM_GPUS=1
fi

if [ "$GPU_MEM" -lt 20000 ]; then
    IS_T4=1
    echo "[GPU] Detected T4 GPUs (${GPU_MEM}MB VRAM each)"
else
    IS_T4=0
    echo "[GPU] Detected high-memory GPUs (${GPU_MEM}MB VRAM each)"
fi

if [ "$IS_T4" -eq 1 ] && [ "$NUM_GPUS" -ge 2 ]; then
    GPU_MODE="t4_2gpu"; GPU_IDS="0,1"
    echo "[GPU] Strategy: 2x T4 DataParallel + gradient_checkpointing"
elif [ "$IS_T4" -eq 1 ]; then
    GPU_MODE="t4_1gpu"; GPU_IDS="${1:-0}"
    echo "[GPU] Strategy: 1x T4 + gradient_checkpointing"
else
    GPU_MODE="a100"; GPU_IDS="${1:-0}"
    echo "[GPU] Strategy: A100 + gradient_checkpointing"
fi

echo "[GPU] Using CUDA_VISIBLE_DEVICES=$GPU_IDS"
echo "============================================================"
echo ""

# ============================================================
# Hyperparameters (from original script)
# ============================================================
MODEL_NAME_ARG="${2:-google/flan-t5-large}"
BASE_OUTPUT_DIR="logs_and_outputs/superni_order1_t5_srt/outputs"
BASE_SAVE_DIR="logs_and_outputs/superni_order1_t5_srt/outputs"
RUN_NAME="superni_order1_t5_srt"
DATA_DIR="CL_Benchmark"
GEN_DATA_DIR="generated_data/lora_gen_superni_t5"
CONFIG_DIR="configs/gen_script_superni_order1_t5_configs"
TASK_ORDER="task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification"

# Parse task list into array
IFS=',' read -ra TASKS <<< "$TASK_ORDER"

# Batch sizes per GPU mode
if [ "$GPU_MODE" = "t4_2gpu" ]; then
    BSZ=1; GA=16; EVAL_BSZ=4
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=1; GA=16; EVAL_BSZ=4
else
    BSZ=1; GA=16; EVAL_BSZ=8
fi

SRT_FLAGS="--use_srt_router --srt_metric_mode hard --srt_max_emb_samples 500 --srt_skip_forward"
echo "NOTE: --srt_skip_forward=True: embeddings loaded from disk (embeddings/\${MODEL}/SuperNI/{task}/)"
echo ""

# ============================================================
# Common fixed arguments (used by every task)
# ============================================================
COMMON_ARGS=(
    --do_train
    --do_predict
    --predict_with_generate
    --model_name_or_path "$MODEL_NAME_ARG"
    --data_dir "$DATA_DIR"
    --task_order "$TASK_ORDER"
    --per_device_train_batch_size $BSZ
    --per_device_eval_batch_size $EVAL_BSZ
    --gradient_accumulation_steps $GA
    --learning_rate 0.0003
    --num_train_epochs 100
    --run_name "$RUN_NAME"
    --max_source_length 512
    --max_target_length 50
    --generation_max_length 50
    --add_task_name False
    --add_dataset_name False
    --overwrite_cache
    --lr_scheduler_type constant
    --warmup_steps 0
    --logging_strategy steps
    --logging_steps 10
    --save_strategy best
    --save_total_limit 1
    --lora_r 4
    --lora_alpha 32
    --lora_dropout 0.0
    --data_replay_freq -1
    --replay_after_n_epoch 0
    --kl_ratio 0.5
    --attn_temperature 1
    --mlp_hidden_dim 100
    --model_name gainlora_inflora
    --threshold 0.995
    --transthreshold 0.995
    $SRT_FLAGS
)

# ============================================================
# Loop over all tasks
# ============================================================
PREV_SAVE_DIRS=""
NUM_TASKS=${#TASKS[@]}

for ((i=0; i<NUM_TASKS; i++)); do
    TASK="${TASKS[$i]}"
    TASK_NUM=$((i+1))
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${TASK_NUM}-${TASK}"
    TASK_CONFIG_DIR="${CONFIG_DIR}/${TASK}"
    SRT_MARKER="${OUTPUT_DIR}/saved_weights/srt_signatures.npz"

    # ── SMART SKIP LOGIC ──────────────────────────────────
    if [ -f "$SRT_MARKER" ]; then
        echo "============================================================"
        echo "✓ SKIP  [${TASK_NUM}/${NUM_TASKS}] $TASK"
        echo "         (saved_weights found → task already completed)"
        echo "============================================================"
        # Still update prev dirs for next task
        if [ -z "$PREV_SAVE_DIRS" ]; then
            PREV_SAVE_DIRS="${OUTPUT_DIR}/saved_weights"
        else
            PREV_SAVE_DIRS="${PREV_SAVE_DIRS},${OUTPUT_DIR}/saved_weights"
        fi
        continue
    fi

    # Check for partial run (checkpoint exists but not finished)
    PARTIAL_CHECKPOINT=""
    if [ -d "$OUTPUT_DIR" ]; then
        LAST_CKPT=$(ls -d ${OUTPUT_DIR}/checkpoint-* 2>/dev/null | sort -V | tail -1)
        if [ -n "$LAST_CKPT" ]; then
            PARTIAL_CHECKPOINT="$LAST_CKPT"
        fi
    fi

    echo "============================================================"
    if [ -n "$PARTIAL_CHECKPOINT" ]; then
        echo "▶ RESUME [${TASK_NUM}/${NUM_TASKS}] $TASK"
        echo "         checkpoint found: $PARTIAL_CHECKPOINT"
    else
        echo "▶ START  [${TASK_NUM}/${NUM_TASKS}] $TASK"
    fi
    echo "         output_dir: $OUTPUT_DIR"
    echo "============================================================"

    # ── Build per-task args ────────────────────────────────
    if [ "$i" -eq 0 ]; then
        CURRENT_METRIC="eval_rougeL"
    else
        CURRENT_METRIC="eval_rougeL_for_${TASK}"
    fi

    ARGS=(
        --task_config_dir "$TASK_CONFIG_DIR"
        --output_dir "$OUTPUT_DIR"
        --metric_for_best_model "$CURRENT_METRIC"
        --evaluation_strategy steps
    )

    if [ "$i" -eq 0 ]; then
        # First task: no previous checkpoints, no gen_data_dir, no instruction_replay
        ARGS+=(
            --gen_data_dir "$GEN_DATA_DIR"
        )
        if [ -n "$PARTIAL_CHECKPOINT" ]; then
            echo "         (Trainer will auto-resume from latest checkpoint)"
        fi
    else
        # Subsequent tasks: load from previous task's saved weights
        PREV_TASK_DIR="${BASE_SAVE_DIR}/$((i))-${TASKS[$((i-1))]}"
        PREV_PROMPT_KEY="${PREV_TASK_DIR}/saved_weights/prompts_keys_till_now.pt"
        ARGS+=(
            --gen_data_dir "$GEN_DATA_DIR"
            --load_checkpoint_from "${PREV_TASK_DIR}/saved_weights/trans_input.pt"
            --previous_lora_path "$PREV_SAVE_DIRS"
            --previous_prompt_key_path "$PREV_PROMPT_KEY"
            --add_instruction_replay
            --srt_load_path "${PREV_TASK_DIR}/saved_weights"
        )
        if [ -n "$PARTIAL_CHECKPOINT" ]; then
            echo "         (Trainer will auto-resume from latest checkpoint)"
        fi
    fi

    # ── Run training ───────────────────────────────────────
    set -x
    CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py "${COMMON_ARGS[@]}" "${ARGS[@]}"
    set +x

    # ── Update accumulated paths for next task ──────────────
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

# Final scoring
python score.py superni_order1_t5_srt superni_order1_t5_srt
