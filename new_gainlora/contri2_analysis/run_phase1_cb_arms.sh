#!/bin/bash
# ==============================================================================
# Phase 1: CB 4-Arm Initialization Comparison
# Tests 4 initialization strategies for CB (task #2 in order 4)
#   Arm A: InfLoRA baseline (current SRT behavior)
#   Arm B: SGWI only (warm init from mnli, no InfLoRA)
#   Arm C: SGWI + InfLoRA (SGWI first, then InfLoRA projects)
#   Arm D: Random init (standard LoRA init)
# ==============================================================================
# Prerequisite: Phase 0 completed (mnli checkpoint available)
# Usage: bash contri2_analysis/run_phase1_cb_arms.sh <GPU_ID> <MODEL_PATH>
# Example: bash contri2_analysis/run_phase1_cb_arms.sh 0 google/flan-t5-large
# ==============================================================================

set -e

GPU_ID=${1:-0}
MODEL_PATH=${2:-google/flan-t5-large}
OUTPUT_BASE=logs_and_outputs/c2_hypothesis
MNLI_CKPT=$OUTPUT_BASE/phase0_mnli/saved_weights

# Verify Phase 0 checkpoint exists
if [ ! -d "$MNLI_CKPT" ]; then
    echo "ERROR: MNLI checkpoint not found at $MNLI_CKPT"
    echo "Run Phase 0 first: bash contri2_analysis/run_phase0_mnli.sh $GPU_ID $MODEL_PATH"
    exit 1
fi

echo "============================================================"
echo "[Phase 1] CB 4-Arm Initialization Comparison"
echo "  GPU: $GPU_ID"
echo "  Model: $MODEL_PATH"
echo "  MNLI checkpoint: $MNLI_CKPT"
echo "============================================================"

# Auto-detect GPU type
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -z "$GPU_MEM" ]; then
    BSZ=4; GA=8; EVAL_BSZ=16; FP16_FLAG="--gradient_checkpointing"
elif [ "$GPU_MEM" -lt 20000 ]; then
    echo "[GPU] T4 detected (${GPU_MEM}MB)"
    BSZ=4; GA=8; EVAL_BSZ=16; FP16_FLAG="--gradient_checkpointing"
else
    echo "[GPU] High-mem GPU detected (${GPU_MEM}MB)"
    BSZ=16; GA=2; EVAL_BSZ=128; FP16_FLAG=""
fi

SRT_FLAGS="--use_srt_router --srt_shrink --srt_shrink_factor 0.1 --srt_metric auto --srt_max_emb_samples 500"

# ==============================================================================
# Function to run CB training with a given arm configuration
# ==============================================================================
run_cb_arm() {
    ARM_NAME=$1
    ARM_DIR=$2
    SGWI_MODE=$3
    
    echo ""
    echo "============================================================"
    echo "[Phase 1] ARM: $ARM_NAME (sgwi_mode=$SGWI_MODE)"
    echo "  Output: $ARM_DIR"
    echo "============================================================"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python src/run_t5.py \
       --do_train \
       --do_predict \
       --predict_with_generate \
       --model_name_or_path $MODEL_PATH \
       --load_checkpoint_from $MNLI_CKPT/trans_input.pt \
       --previous_lora_path $MNLI_CKPT \
       --previous_prompt_key_path $MNLI_CKPT/prompts_keys_till_now.pt \
       --data_dir CL_Benchmark \
       --task_order mnli,cb,wic,copa,qqp,boolq,rte,imdb,yelp,amazon,sst2,dbpedia,agnews,multirc,yahoo \
       --task_config_dir configs/gen_script_long_order4_t5_configs/cb \
       --output_dir $ARM_DIR \
       --per_device_train_batch_size $BSZ \
       --per_device_eval_batch_size $EVAL_BSZ \
       --gradient_accumulation_steps $GA \
       --learning_rate 0.0003 \
       --num_train_epochs 10 \
       --run_name c2_arm_${ARM_NAME} \
       --max_source_length 512 \
       --max_target_length 50 \
       --generation_max_length 50 \
       --add_task_name False \
       --add_dataset_name False \
       --overwrite_output_dir \
       --overwrite_cache \
       --lr_scheduler_type constant \
       --warmup_steps 0 \
       --logging_strategy steps \
       --logging_steps 5 \
       --metric_for_best_model eval_exact_match_for_cb \
       --evaluation_strategy steps \
       --save_strategy best \
       --save_total_limit 1 \
       --lora_r 8 \
       --lora_alpha 32 \
       --lora_dropout 0.0 \
       --data_replay_freq -1 \
       --kl_ratio 0.1 \
       --attn_temperature 1 \
       --mlp_hidden_dim 100 \
       --model_name gainlora_inflora \
       --threshold 0.995 \
       --transthreshold 0.995 \
       --sgwi_mode $SGWI_MODE \
       $FP16_FLAG \
       $SRT_FLAGS \
       --srt_load_path $MNLI_CKPT
    
    rm -rf $ARM_DIR/checkpoint*
    echo "[Phase 1] ARM $ARM_NAME complete."
    sleep 3
}

# ==============================================================================
# Run all 4 arms
# ==============================================================================

# ARM A: InfLoRA baseline (current SRT behavior — get_reg_matrix runs normally)
run_cb_arm "a_inflora" "$OUTPUT_BASE/phase1_arm_a_inflora" "inflora"

# ARM B: SGWI only (warm init from mnli LoRA, skip InfLoRA)
run_cb_arm "b_sgwi" "$OUTPUT_BASE/phase1_arm_b_sgwi" "sgwi"

# ARM C: SGWI + InfLoRA (SGWI first, then InfLoRA projects on top)
run_cb_arm "c_sgwi_inflora" "$OUTPUT_BASE/phase1_arm_c_sgwi_inflora" "sgwi+inflora"

# ARM D: Random init (standard LoRA init, no InfLoRA, no SGWI)
run_cb_arm "d_random" "$OUTPUT_BASE/phase1_arm_d_random" "random"

# ==============================================================================
# Auto-analyze results
# ==============================================================================
echo ""
echo "============================================================"
echo "[Phase 1] All 4 arms complete. Running analysis..."
echo "============================================================"
echo ""

python contri2_analysis/analyze_results.py --phase 1 --output_base $OUTPUT_BASE

echo ""
echo "============================================================"
echo "[Phase 1] ✅ Complete!"
echo "  If Q1=YES → Run Phase 3: bash contri2_analysis/run_phase3_dualfisher_sweep.sh $GPU_ID $MODEL_PATH"
echo "  If Q1=NO  → STOP. Reconsider C2 direction."
echo "============================================================"
