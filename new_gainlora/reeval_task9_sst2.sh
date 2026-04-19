#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Re-evaluate Task 9 (sst2) PREDICT-ONLY — no training
# 
# Purpose: Task 9 training completed successfully (saved_weights exist),
#          but predict crashed due to all_attn_weights bug (now fixed).
#          This script loads the trained weights and runs predict-only
#          to regenerate all_results.json, then re-runs score.py.
#
# Usage:
#   cd new_gainlora
#   bash reeval_task9_sst2.sh <GPU_ID> <MODEL_PATH>
#   # e.g.: bash reeval_task9_sst2.sh 0 google/flan-t5-large
# ─────────────────────────────────────────────────────────────────────────────
set -e

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
GPU_IDS="${1:-0}"
MODEL_PATH="${2:-google/flan-t5-large}"
RUN_NAME="long_order3_t5_srt_hard"
BASE="logs_and_outputs/$RUN_NAME/outputs"

echo "════════════════════════════════════════════════════════════"
echo " Re-eval Task 9 (sst2) — predict only"
echo " GPU=$GPU_IDS  MODEL=$MODEL_PATH"
echo " RUN=$RUN_NAME"
echo "════════════════════════════════════════════════════════════"

# ── Verify saved_weights exist ────────────────────────────────────────────────
TASK9_DIR="$BASE/9-sst2"
if [ ! -d "$TASK9_DIR/saved_weights" ]; then
    echo "[ERROR] $TASK9_DIR/saved_weights not found!"
    echo "  Task 9 training must have completed. Cannot re-eval without weights."
    exit 1
fi

echo "[OK] saved_weights found at $TASK9_DIR/saved_weights"
echo "[INFO] Will run --do_predict only (no --do_train)"

# ── GPU auto-detect for batch size ────────────────────────────────────────────
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
: ${GPU_MEM:=16000}
if [ "$GPU_MEM" -lt 20000 ]; then
    EVAL_BSZ=64; FP16_FLAG="--gradient_checkpointing"
elif [ "$GPU_MEM" -lt 50000 ]; then
    EVAL_BSZ=128; FP16_FLAG="--gradient_checkpointing"
else
    EVAL_BSZ=128; FP16_FLAG=""
fi
echo "[GPU] ${GPU_MEM}MB → EVAL_BSZ=$EVAL_BSZ"

# SRT flags (same as original training)
SRT_FLAGS="--use_srt_router --srt_metric_mode hard --srt_max_emb_samples 500 --srt_skip_forward"

# ── Run predict-only ──────────────────────────────────────────────────────────
# NOTE: --do_predict only, --do_train is NOT set
# This will:
#   1. Load the model + all previous LoRA weights
#   2. Load SRT signatures 
#   3. Run predict on the test set for tasks 0..8 (cumulative eval)
#   4. Write all_results.json to the output dir
echo ""
echo "[STEP 1/2] Running predict-only for task 9-sst2..."
CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
    --do_predict --predict_with_generate \
    --model_name_or_path $MODEL_PATH \
    --load_checkpoint_from $BASE/8-imdb/saved_weights/trans_input.pt \
    --previous_lora_path $BASE/1-yelp/saved_weights,$BASE/2-amazon/saved_weights,$BASE/3-mnli/saved_weights,$BASE/4-cb/saved_weights,$BASE/5-copa/saved_weights,$BASE/6-qqp/saved_weights,$BASE/7-rte/saved_weights,$BASE/8-imdb/saved_weights \
    --previous_prompt_key_path $BASE/8-imdb/saved_weights/prompts_keys_till_now.pt \
    --data_dir CL_Benchmark \
    --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
    --gen_data_dir generated_data/lora_gen_long_t5 \
    --task_config_dir configs/gen_script_long_order3_t5_configs/sst2 \
    --output_dir $TASK9_DIR \
    --per_device_train_batch_size 16 --per_device_eval_batch_size $EVAL_BSZ \
    --gradient_accumulation_steps 2 --learning_rate 0.0003 --num_train_epochs 10 \
    --run_name $RUN_NAME \
    --max_source_length 512 --max_target_length 50 --generation_max_length 50 \
    --add_task_name False --add_dataset_name False \
    --overwrite_output_dir --overwrite_cache \
    --lr_scheduler_type constant --warmup_steps 0 \
    --logging_strategy steps --logging_steps 10 \
    --metric_for_best_model eval_exact_match_for_sst2 \
    --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \
    --lora_r 8 --lora_alpha 32 --lora_dropout 0.0 \
    --data_replay_freq -1 --kl_ratio 0.1 --attn_temperature 1 \
    --mlp_hidden_dim 100 --model_name gainlora \
    --threshold 0.995 --transthreshold 0.995 \
    $FP16_FLAG $SRT_FLAGS \
    --srt_load_path $BASE/8-imdb/saved_weights

echo ""
echo "[STEP 1/2] Done. Checking all_results.json..."
if [ -f "$TASK9_DIR/all_results.json" ]; then
    echo "[OK] all_results.json created:"
    cat "$TASK9_DIR/all_results.json" | python -m json.tool 2>/dev/null || cat "$TASK9_DIR/all_results.json"
else
    echo "[ERROR] all_results.json NOT found! Predict may have failed."
    exit 1
fi

# ── Re-run score.py ──────────────────────────────────────────────────────────
echo ""
echo "[STEP 2/2] Re-running score.py..."
python score.py $RUN_NAME $RUN_NAME

echo ""
echo "════════════════════════════════════════════════════════════"
echo " DONE. Task 9 re-eval complete. Check results/ folder."
echo "════════════════════════════════════════════════════════════"
