#!/bin/bash
#SBATCH -J srt
#SBATCH -o srt-%j.out
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 40:00:00
#SBATCH --mem 256G
#SBATCH --gres=gpu:a100-sxm4-80gb:1

export CUDA_DEVICE_ORDER="PCI_BUS_ID"

port=$(shuf -i25000-30000 -n1)

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
    echo "[GPU] Detected T4 (${GPU_MEM}MB)"
else
    IS_T4=0
    echo "[GPU] Detected high-memory GPU (${GPU_MEM}MB)"
fi

if [ "$IS_T4" -eq 1 ] && [ "$NUM_GPUS" -ge 2 ]; then
    GPU_MODE="t4_2gpu"; GPU_IDS="0,1"; FP16_FLAG="--gradient_checkpointing"
elif [ "$IS_T4" -eq 1 ]; then
    GPU_MODE="t4_1gpu"; GPU_IDS="${1:-0}"; FP16_FLAG="--gradient_checkpointing"
else
    GPU_MODE="a100"; GPU_IDS="${1:-0}"; FP16_FLAG=""
fi

echo "[GPU] Using CUDA_VISIBLE_DEVICES=$GPU_IDS"
echo "============================================================"
echo ""

RUN_NAME="superni_order1_t5_srt"
TASK_ORDER="task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification"

BASE_OUT="logs_and_outputs/${RUN_NAME}"

# T5-XL: BSZ=2, GA=8 (effective 16 per step on A100)
if [ "$GPU_MODE" = "t4_2gpu" ]; then
    BSZ=2; GA=8; EVAL_BSZ=8
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=2; GA=8; EVAL_BSZ=8
else
    BSZ=2; GA=8; EVAL_BSZ=16
fi

SRT_FLAGS="--use_srt_router --srt_shrink --srt_shrink_factor 0.1 --srt_metric auto --srt_max_emb_samples 500"

# ──────────────────────────────────────────────────────────────────
# TASK 1: task1572_samsum_summary
# ──────────────────────────────────────────────────────────────────
echo "[TASK 1/15] task1572_samsum_summary"
CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path "${2:-t5-xl}" \
   --data_dir CL_Benchmark \
   --task_order $TASK_ORDER \
   --task_config_dir configs/gen_script_superni_order1_t5_configs/task1572_samsum_summary \
   --output_dir ${BASE_OUT}/outputs/1-task1572_samsum_summary \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 100 \
   --run_name $RUN_NAME \
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
   --logging_steps 10 \
   --metric_for_best_model eval_rougeL \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --load_best_model_at_end \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG \
   $SRT_FLAGS

# Clean checkpoint to save disk
rm -rf ${BASE_OUT}/outputs/1-task1572_samsum_summary/checkpoint*
sleep 5

# ──────────────────────────────────────────────────────────────────
# TASK 2: task363_sst2_polarity_classification
# ──────────────────────────────────────────────────────────────────
echo "[TASK 2/15] task363_sst2_polarity_classification"
CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path "${2:-t5-xl}" \
   --load_checkpoint_from ${BASE_OUT}/outputs/1-task1572_samsum_summary/saved_weights/trans_input.pt \
   --previous_lora_path ${BASE_OUT}/outputs/1-task1572_samsum_summary/saved_weights \
   --previous_prompt_key_path ${BASE_OUT}/outputs/1-task1572_samsum_summary/saved_weights/prompts_keys_till_now.pt \
   --data_dir CL_Benchmark \
   --task_order $TASK_ORDER \
   --gen_data_dir generated_data/lora_gen_superni_t5 \
   --task_config_dir configs/gen_script_superni_order1_t5_configs/task363_sst2_polarity_classification \
   --output_dir ${BASE_OUT}/outputs/2-task363_sst2_polarity_classification \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 100 \
   --run_name $RUN_NAME \
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
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --load_best_model_at_end \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG \
   --use_srt_router \
   --srt_shrink \
   --srt_shrink_factor 0.1 \
   --srt_metric auto \
   --srt_max_emb_samples 500 \
   --srt_load_path ${BASE_OUT}/outputs/1-task1572_samsum_summary/saved_weights

rm -rf ${BASE_OUT}/outputs/2-task363_sst2_polarity_classification/checkpoint*
sleep 5

# ──────────────────────────────────────────────────────────────────
# TASK 3: task1290_xsum_summarization
# ──────────────────────────────────────────────────────────────────
echo "[TASK 3/15] task1290_xsum_summarization"
CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path "${2:-t5-xl}" \
   --load_checkpoint_from ${BASE_OUT}/outputs/2-task363_sst2_polarity_classification/saved_weights/trans_input.pt \
   --previous_lora_path ${BASE_OUT}/outputs/1-task1572_samsum_summary/saved_weights,${BASE_OUT}/outputs/2-task363_sst2_polarity_classification/saved_weights \
   --previous_prompt_key_path ${BASE_OUT}/outputs/2-task363_sst2_polarity_classification/saved_weights/prompts_keys_till_now.pt \
   --data_dir CL_Benchmark \
   --task_order $TASK_ORDER \
   --gen_data_dir generated_data/lora_gen_superni_t5 \
   --task_config_dir configs/gen_script_superni_order1_t5_configs/task1290_xsum_summarization \
   --output_dir ${BASE_OUT}/outputs/3-task1290_xsum_summarization \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 100 \
   --run_name $RUN_NAME \
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
   --logging_steps 10 \
   --metric_for_best_model eval_rougeL \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --load_best_model_at_end \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG \
   --use_srt_router \
   --srt_shrink \
   --srt_shrink_factor 0.1 \
   --srt_metric auto \
   --srt_max_emb_samples 500 \
   --srt_load_path ${BASE_OUT}/outputs/2-task363_sst2_polarity_classification/saved_weights

rm -rf ${BASE_OUT}/outputs/3-task1290_xsum_summarization/checkpoint*
echo "[DONE] Script complete. Run score.py to compute final metrics."
