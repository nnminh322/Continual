#!/bin/bash
# ==============================================================================
# Phase 0: Train MNLI (task 1 of Order 4) with SRT baseline
# This creates the shared checkpoint that ALL Phase 1 CB arms will build upon.
# ==============================================================================
# IMPORTANT: Run this from the new_gainlora/ directory!
# Usage: cd new_gainlora && bash ../contri2_analysis/run_phase0_mnli.sh <GPU_ID> <MODEL_PATH>
# Example: cd new_gainlora && bash ../contri2_analysis/run_phase0_mnli.sh 0 google/flan-t5-large
# ==============================================================================

set -e

# Verify we're in new_gainlora/
if [ ! -f "src/run_t5.py" ]; then
    echo "ERROR: Must run from new_gainlora/ directory!"
    echo "Usage: cd new_gainlora && bash ../contri2_analysis/run_phase0_mnli.sh <GPU_ID> <MODEL_PATH>"
    exit 1
fi

GPU_ID=${1:-0}
MODEL_PATH=${2:-google/flan-t5-large}
OUTPUT_BASE=logs_and_outputs/c2_hypothesis

echo "============================================================"
echo "[Phase 0] Training MNLI (Task 1 of Order 4)"
echo "  GPU: $GPU_ID"
echo "  Model: $MODEL_PATH"
echo "  Output: $OUTPUT_BASE/phase0_mnli"
echo "============================================================"

# Auto-detect GPU type
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -z "$GPU_MEM" ]; then
    echo "[WARN] No GPU detected, defaulting to small batch"
    BSZ=4; GA=8; EVAL_BSZ=16; FP16_FLAG="--gradient_checkpointing"
elif [ "$GPU_MEM" -lt 20000 ]; then
    echo "[GPU] T4 detected (${GPU_MEM}MB)"
    BSZ=4; GA=8; EVAL_BSZ=16; FP16_FLAG="--gradient_checkpointing"
else
    echo "[GPU] High-mem GPU detected (${GPU_MEM}MB)"
    BSZ=8; GA=4; EVAL_BSZ=128; FP16_FLAG=""
fi

SRT_FLAGS="--use_srt_router --srt_shrink --srt_shrink_factor 0.1 --srt_metric hard --srt_max_emb_samples 500"

CUDA_VISIBLE_DEVICES=$GPU_ID python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $MODEL_PATH \
   --data_dir CL_Benchmark \
   --task_order mnli,cb,wic,copa,qqp,boolq,rte,imdb,yelp,amazon,sst2,dbpedia,agnews,multirc,yahoo \
   --task_config_dir configs/gen_script_long_order4_t5_configs/mnli \
   --output_dir $OUTPUT_BASE/phase0_mnli \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name c2_hyp_phase0_mnli \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --add_instruction_replay \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --mlp_hidden_dim 100 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG \
   $SRT_FLAGS

rm -rf $OUTPUT_BASE/phase0_mnli/checkpoint*

echo ""
echo "============================================================"
echo "[Phase 0] ✅ MNLI training complete!"
echo "  Checkpoint: $OUTPUT_BASE/phase0_mnli/saved_weights/"
echo "  Next: bash ../contri2_analysis/run_phase1_cb_arms.sh $GPU_ID $MODEL_PATH"
echo "============================================================"
