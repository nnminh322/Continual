#!/bin/bash
# ==============================================================================
# Phase 3: Dual Fisher λ_emb Sweep
# Tests 4 values of λ_emb with SGWI initialization on CB task
# ==============================================================================
# IMPORTANT: Run this from the new_gainlora/ directory!
# Prerequisite: Phase 0 + Phase 1 completed, Q1=YES (SGWI shows positive signal)
# Usage: cd new_gainlora && bash ../contri2_analysis/run_phase3_dualfisher_sweep.sh <GPU_ID> <MODEL_PATH>
# ==============================================================================

set -e

# Verify we're in new_gainlora/
if [ ! -f "src/run_t5.py" ]; then
    echo "ERROR: Must run from new_gainlora/ directory!"
    echo "Usage: cd new_gainlora && bash ../contri2_analysis/run_phase3_dualfisher_sweep.sh <GPU_ID> <MODEL_PATH>"
    exit 1
fi

GPU_ID=${1:-0}
MODEL_PATH=${2:-google/flan-t5-large}
OUTPUT_BASE=logs_and_outputs/c2_hypothesis
MNLI_CKPT=$OUTPUT_BASE/phase0_mnli/saved_weights

# Verify prerequisites
if [ ! -d "$MNLI_CKPT" ]; then
    echo "ERROR: MNLI checkpoint not found. Run Phase 0 first."
    exit 1
fi

echo "============================================================"
echo "[Phase 3] Dual Fisher λ_emb Sweep on CB (with SGWI init)"
echo "  GPU: $GPU_ID"  
echo "  Model: $MODEL_PATH"
echo "  λ values: 0.001, 0.005, 0.01, 0.05"
echo "============================================================"

# Auto-detect GPU type
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -z "$GPU_MEM" ]; then
    BSZ=4; GA=8; EVAL_BSZ=16; FP16_FLAG="--gradient_checkpointing"
elif [ "$GPU_MEM" -lt 20000 ]; then
    BSZ=4; GA=8; EVAL_BSZ=16; FP16_FLAG="--gradient_checkpointing"
else
    BSZ=16; GA=2; EVAL_BSZ=128; FP16_FLAG=""
fi

SRT_FLAGS="--use_srt_router --srt_shrink --srt_shrink_factor 0.1 --srt_metric auto --srt_max_emb_samples 500"

for LAMBDA in 0.001 0.005 0.01 0.05; do
    echo ""
    echo "============================================================"
    echo "[Phase 3] λ_emb = $LAMBDA"
    echo "  Output: $OUTPUT_BASE/phase3_lambda_${LAMBDA}"
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
       --output_dir $OUTPUT_BASE/phase3_lambda_${LAMBDA} \
       --per_device_train_batch_size $BSZ \
       --per_device_eval_batch_size $EVAL_BSZ \
       --gradient_accumulation_steps $GA \
       --learning_rate 0.0003 \
       --num_train_epochs 10 \
       --run_name c2_df_lambda_${LAMBDA} \
       --max_source_length 512 \
       --max_target_length 50 \
       --generation_max_length 50 \
       --add_task_name False \
       --add_dataset_name False \
       --overwrite_cache \
       --lr_scheduler_type constant \
       --warmup_steps 0 \
       --logging_strategy steps \
       --logging_steps 5 \
       --metric_for_best_model eval_exact_match \
       --evaluation_strategy steps \
       --save_strategy steps \
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
       --sgwi_mode sgwi \
       --lambda_emb $LAMBDA \
       $FP16_FLAG \
       $SRT_FLAGS \
       --srt_load_path $MNLI_CKPT
    
    rm -rf $OUTPUT_BASE/phase3_lambda_${LAMBDA}/checkpoint*
    echo "[Phase 3] λ_emb=$LAMBDA complete."
    sleep 3
done

echo ""
echo "============================================================"
echo "[Phase 3] All λ values tested. Running analysis..."
echo "============================================================"
echo ""

python ../contri2_analysis/analyze_results.py --phase 3 --output_base $OUTPUT_BASE

echo ""
echo "============================================================"
echo "[Phase 3] ✅ Complete!"
echo "  Review results above for best λ_emb."
echo "  If Q3=YES → Proceed to Phase 4 (5-task e2e validation)"
echo "  If Q3=NO  → C2 = SGWI only (still valid, drop Dual Fisher)"
echo "============================================================"
