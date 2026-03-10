#!/bin/bash
#SBATCH -J cl
#SBATCH -o cl-%j.out
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 20:00:00
#SBATCH --mem 128G
#SBATCH --gres=gpu:2

export CUDA_DEVICE_ORDER="PCI_BUS_ID"

port=$(shuf -i25000-30000 -n1)

# ============================================================
# Auto-detect GPU count and type for optimal parallelism
# ============================================================
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)

if [ -z "$GPU_MEM" ]; then
    echo "ERROR: No GPU detected!"
    exit 1
fi

# Determine GPU type
if [ "$GPU_MEM" -lt 20000 ]; then
    IS_T4=1
    echo "[GPU] Detected T4 GPUs (${GPU_MEM}MB VRAM each)"
else
    IS_T4=0
    echo "[GPU] Detected high-memory GPUs (${GPU_MEM}MB VRAM each)"
fi

# Determine parallelism strategy
# NOTE: DataParallel is incompatible (segfault with GC, OOM without GC).
# When 2x T4: use DDP (torchrun) with gradient_checkpointing. Each GPU ~13.4GB.
if [ "$IS_T4" -eq 1 ] && [ "$NUM_GPUS" -ge 2 ]; then
    GPU_MODE="t4_ddp"
    GPU_IDS="0,1"
    FP16_FLAG="--gradient_checkpointing"
    echo "[GPU] Strategy: 2x T4 DDP + fp32 + gradient_checkpointing"
elif [ "$IS_T4" -eq 1 ]; then
    GPU_MODE="t4_1gpu"
    GPU_IDS="${1:-0}"
    FP16_FLAG="--gradient_checkpointing"
    echo "[GPU] Strategy: 1x T4 + fp32 + gradient_checkpointing"
else
    GPU_MODE="a100"
    GPU_IDS="${1:-0}"
    FP16_FLAG=""
    echo "[GPU] Strategy: A100 (single GPU, fp32)"
fi

echo "[GPU] Using CUDA_VISIBLE_DEVICES=$GPU_IDS"
echo "============================================================"
echo ""

# Set launch command: torchrun for DDP, python otherwise
if [ "$GPU_MODE" = "t4_ddp" ]; then
    PYTHON_CMD="torchrun --standalone --nproc_per_node=${NUM_GPUS}"
else
    PYTHON_CMD="python"
fi

if [ "$GPU_MODE" = "t4_ddp" ]; then
    BSZ=2; GA=8; EVAL_BSZ=2
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=8; GA=4; EVAL_BSZ=4
else
    BSZ=16; GA=2; EVAL_BSZ=4
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS $PYTHON_CMD src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --data_dir CL_Benchmark \
   --task_order task748_glucose_reverse_cause_event_detection,task073_commonsenseqa_answer_generation,task875_emotion_classification,task002_quoref_answer_generation,task1687_sentiment140_classification,task591_sciq_answer_generation,task363_sst2_polarity_classification,task1572_samsum_summary,task511_reddit_tifu_long_text_summarization,task1290_xsum_summarization,task639_multi_woz_user_utterance_generation,task1510_evalution_relation_extraction,task181_outcome_extraction,task1729_personachat_generate_next,task1590_diplomacy_text_generation \
   --task_config_dir configs/gen_script_superni_order2_t5_configs/task748_glucose_reverse_cause_event_detection \
   --output_dir logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/1-task748_glucose_reverse_cause_event_detection \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 100 \
   --run_name gen_script_superni_order2_t5_specroute \
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
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --run_single True \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

if [ "$GPU_MODE" = "t4_ddp" ]; then
    BSZ=2; GA=8; EVAL_BSZ=2
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=8; GA=4; EVAL_BSZ=4
else
    BSZ=32; GA=1; EVAL_BSZ=4
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS $PYTHON_CMD src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/1-task748_glucose_reverse_cause_event_detection/saved_weights \
   --data_dir CL_Benchmark \
   --task_order task748_glucose_reverse_cause_event_detection,task073_commonsenseqa_answer_generation,task875_emotion_classification,task002_quoref_answer_generation,task1687_sentiment140_classification,task591_sciq_answer_generation,task363_sst2_polarity_classification,task1572_samsum_summary,task511_reddit_tifu_long_text_summarization,task1290_xsum_summarization,task639_multi_woz_user_utterance_generation,task1510_evalution_relation_extraction,task181_outcome_extraction,task1729_personachat_generate_next,task1590_diplomacy_text_generation \
   --task_config_dir configs/gen_script_superni_order2_t5_configs/task073_commonsenseqa_answer_generation \
   --output_dir logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/2-task073_commonsenseqa_answer_generation \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 100 \
   --run_name gen_script_superni_order2_t5_specroute \
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
   --metric_for_best_model eval_rougeL_for_task073_commonsenseqa_answer_generation \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

if [ "$GPU_MODE" = "t4_ddp" ]; then
    BSZ=2; GA=8; EVAL_BSZ=2
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=8; GA=4; EVAL_BSZ=4
else
    BSZ=32; GA=1; EVAL_BSZ=4
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS $PYTHON_CMD src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/1-task748_glucose_reverse_cause_event_detection/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/2-task073_commonsenseqa_answer_generation/saved_weights \
   --data_dir CL_Benchmark \
   --task_order task748_glucose_reverse_cause_event_detection,task073_commonsenseqa_answer_generation,task875_emotion_classification,task002_quoref_answer_generation,task1687_sentiment140_classification,task591_sciq_answer_generation,task363_sst2_polarity_classification,task1572_samsum_summary,task511_reddit_tifu_long_text_summarization,task1290_xsum_summarization,task639_multi_woz_user_utterance_generation,task1510_evalution_relation_extraction,task181_outcome_extraction,task1729_personachat_generate_next,task1590_diplomacy_text_generation \
   --task_config_dir configs/gen_script_superni_order2_t5_configs/task875_emotion_classification \
   --output_dir logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/3-task875_emotion_classification \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 100 \
   --run_name gen_script_superni_order2_t5_specroute \
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
   --metric_for_best_model eval_rougeL_for_task875_emotion_classification \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

if [ "$GPU_MODE" = "t4_ddp" ]; then
    BSZ=2; GA=8; EVAL_BSZ=2
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=8; GA=4; EVAL_BSZ=4
else
    BSZ=32; GA=1; EVAL_BSZ=4
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS $PYTHON_CMD src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/1-task748_glucose_reverse_cause_event_detection/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/2-task073_commonsenseqa_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/3-task875_emotion_classification/saved_weights \
   --data_dir CL_Benchmark \
   --task_order task748_glucose_reverse_cause_event_detection,task073_commonsenseqa_answer_generation,task875_emotion_classification,task002_quoref_answer_generation,task1687_sentiment140_classification,task591_sciq_answer_generation,task363_sst2_polarity_classification,task1572_samsum_summary,task511_reddit_tifu_long_text_summarization,task1290_xsum_summarization,task639_multi_woz_user_utterance_generation,task1510_evalution_relation_extraction,task181_outcome_extraction,task1729_personachat_generate_next,task1590_diplomacy_text_generation \
   --task_config_dir configs/gen_script_superni_order2_t5_configs/task002_quoref_answer_generation \
   --output_dir logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/4-task002_quoref_answer_generation \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 100 \
   --run_name gen_script_superni_order2_t5_specroute \
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
   --metric_for_best_model eval_rougeL_for_task002_quoref_answer_generation \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

if [ "$GPU_MODE" = "t4_ddp" ]; then
    BSZ=2; GA=8; EVAL_BSZ=2
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=8; GA=4; EVAL_BSZ=4
else
    BSZ=32; GA=1; EVAL_BSZ=4
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS $PYTHON_CMD src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/1-task748_glucose_reverse_cause_event_detection/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/2-task073_commonsenseqa_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/3-task875_emotion_classification/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/4-task002_quoref_answer_generation/saved_weights \
   --data_dir CL_Benchmark \
   --task_order task748_glucose_reverse_cause_event_detection,task073_commonsenseqa_answer_generation,task875_emotion_classification,task002_quoref_answer_generation,task1687_sentiment140_classification,task591_sciq_answer_generation,task363_sst2_polarity_classification,task1572_samsum_summary,task511_reddit_tifu_long_text_summarization,task1290_xsum_summarization,task639_multi_woz_user_utterance_generation,task1510_evalution_relation_extraction,task181_outcome_extraction,task1729_personachat_generate_next,task1590_diplomacy_text_generation \
   --task_config_dir configs/gen_script_superni_order2_t5_configs/task1687_sentiment140_classification \
   --output_dir logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/5-task1687_sentiment140_classification \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 100 \
   --run_name gen_script_superni_order2_t5_specroute \
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
   --metric_for_best_model eval_rougeL_for_task1687_sentiment140_classification \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

if [ "$GPU_MODE" = "t4_ddp" ]; then
    BSZ=2; GA=8; EVAL_BSZ=2
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=8; GA=4; EVAL_BSZ=4
else
    BSZ=32; GA=1; EVAL_BSZ=4
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS $PYTHON_CMD src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/1-task748_glucose_reverse_cause_event_detection/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/2-task073_commonsenseqa_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/3-task875_emotion_classification/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/4-task002_quoref_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/5-task1687_sentiment140_classification/saved_weights \
   --data_dir CL_Benchmark \
   --task_order task748_glucose_reverse_cause_event_detection,task073_commonsenseqa_answer_generation,task875_emotion_classification,task002_quoref_answer_generation,task1687_sentiment140_classification,task591_sciq_answer_generation,task363_sst2_polarity_classification,task1572_samsum_summary,task511_reddit_tifu_long_text_summarization,task1290_xsum_summarization,task639_multi_woz_user_utterance_generation,task1510_evalution_relation_extraction,task181_outcome_extraction,task1729_personachat_generate_next,task1590_diplomacy_text_generation \
   --task_config_dir configs/gen_script_superni_order2_t5_configs/task591_sciq_answer_generation \
   --output_dir logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/6-task591_sciq_answer_generation \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 100 \
   --run_name gen_script_superni_order2_t5_specroute \
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
   --metric_for_best_model eval_rougeL_for_task591_sciq_answer_generation \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

if [ "$GPU_MODE" = "t4_ddp" ]; then
    BSZ=2; GA=8; EVAL_BSZ=2
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=8; GA=4; EVAL_BSZ=4
else
    BSZ=32; GA=1; EVAL_BSZ=4
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS $PYTHON_CMD src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/1-task748_glucose_reverse_cause_event_detection/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/2-task073_commonsenseqa_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/3-task875_emotion_classification/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/4-task002_quoref_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/5-task1687_sentiment140_classification/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/6-task591_sciq_answer_generation/saved_weights \
   --data_dir CL_Benchmark \
   --task_order task748_glucose_reverse_cause_event_detection,task073_commonsenseqa_answer_generation,task875_emotion_classification,task002_quoref_answer_generation,task1687_sentiment140_classification,task591_sciq_answer_generation,task363_sst2_polarity_classification,task1572_samsum_summary,task511_reddit_tifu_long_text_summarization,task1290_xsum_summarization,task639_multi_woz_user_utterance_generation,task1510_evalution_relation_extraction,task181_outcome_extraction,task1729_personachat_generate_next,task1590_diplomacy_text_generation \
   --task_config_dir configs/gen_script_superni_order2_t5_configs/task363_sst2_polarity_classification \
   --output_dir logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/7-task363_sst2_polarity_classification \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 100 \
   --run_name gen_script_superni_order2_t5_specroute \
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
   --metric_for_best_model eval_rougeL_for_task363_sst2_polarity_classification \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

if [ "$GPU_MODE" = "t4_ddp" ]; then
    BSZ=2; GA=8; EVAL_BSZ=2
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=8; GA=4; EVAL_BSZ=4
else
    BSZ=32; GA=1; EVAL_BSZ=4
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS $PYTHON_CMD src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/1-task748_glucose_reverse_cause_event_detection/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/2-task073_commonsenseqa_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/3-task875_emotion_classification/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/4-task002_quoref_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/5-task1687_sentiment140_classification/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/6-task591_sciq_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/7-task363_sst2_polarity_classification/saved_weights \
   --data_dir CL_Benchmark \
   --task_order task748_glucose_reverse_cause_event_detection,task073_commonsenseqa_answer_generation,task875_emotion_classification,task002_quoref_answer_generation,task1687_sentiment140_classification,task591_sciq_answer_generation,task363_sst2_polarity_classification,task1572_samsum_summary,task511_reddit_tifu_long_text_summarization,task1290_xsum_summarization,task639_multi_woz_user_utterance_generation,task1510_evalution_relation_extraction,task181_outcome_extraction,task1729_personachat_generate_next,task1590_diplomacy_text_generation \
   --task_config_dir configs/gen_script_superni_order2_t5_configs/task1572_samsum_summary \
   --output_dir logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/8-task1572_samsum_summary \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 100 \
   --run_name gen_script_superni_order2_t5_specroute \
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
   --metric_for_best_model eval_rougeL_for_task1572_samsum_summary \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

if [ "$GPU_MODE" = "t4_ddp" ]; then
    BSZ=2; GA=8; EVAL_BSZ=2
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=8; GA=4; EVAL_BSZ=4
else
    BSZ=32; GA=1; EVAL_BSZ=4
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS $PYTHON_CMD src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/1-task748_glucose_reverse_cause_event_detection/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/2-task073_commonsenseqa_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/3-task875_emotion_classification/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/4-task002_quoref_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/5-task1687_sentiment140_classification/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/6-task591_sciq_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/7-task363_sst2_polarity_classification/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/8-task1572_samsum_summary/saved_weights \
   --data_dir CL_Benchmark \
   --task_order task748_glucose_reverse_cause_event_detection,task073_commonsenseqa_answer_generation,task875_emotion_classification,task002_quoref_answer_generation,task1687_sentiment140_classification,task591_sciq_answer_generation,task363_sst2_polarity_classification,task1572_samsum_summary,task511_reddit_tifu_long_text_summarization,task1290_xsum_summarization,task639_multi_woz_user_utterance_generation,task1510_evalution_relation_extraction,task181_outcome_extraction,task1729_personachat_generate_next,task1590_diplomacy_text_generation \
   --task_config_dir configs/gen_script_superni_order2_t5_configs/task511_reddit_tifu_long_text_summarization \
   --output_dir logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/9-task511_reddit_tifu_long_text_summarization \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 100 \
   --run_name gen_script_superni_order2_t5_specroute \
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
   --metric_for_best_model eval_rougeL_for_task511_reddit_tifu_long_text_summarization \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

if [ "$GPU_MODE" = "t4_ddp" ]; then
    BSZ=2; GA=8; EVAL_BSZ=2
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=8; GA=4; EVAL_BSZ=4
else
    BSZ=32; GA=1; EVAL_BSZ=4
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS $PYTHON_CMD src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/1-task748_glucose_reverse_cause_event_detection/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/2-task073_commonsenseqa_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/3-task875_emotion_classification/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/4-task002_quoref_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/5-task1687_sentiment140_classification/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/6-task591_sciq_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/7-task363_sst2_polarity_classification/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/8-task1572_samsum_summary/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/9-task511_reddit_tifu_long_text_summarization/saved_weights \
   --data_dir CL_Benchmark \
   --task_order task748_glucose_reverse_cause_event_detection,task073_commonsenseqa_answer_generation,task875_emotion_classification,task002_quoref_answer_generation,task1687_sentiment140_classification,task591_sciq_answer_generation,task363_sst2_polarity_classification,task1572_samsum_summary,task511_reddit_tifu_long_text_summarization,task1290_xsum_summarization,task639_multi_woz_user_utterance_generation,task1510_evalution_relation_extraction,task181_outcome_extraction,task1729_personachat_generate_next,task1590_diplomacy_text_generation \
   --task_config_dir configs/gen_script_superni_order2_t5_configs/task1290_xsum_summarization \
   --output_dir logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/10-task1290_xsum_summarization \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 100 \
   --run_name gen_script_superni_order2_t5_specroute \
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
   --metric_for_best_model eval_rougeL_for_task1290_xsum_summarization \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

if [ "$GPU_MODE" = "t4_ddp" ]; then
    BSZ=2; GA=8; EVAL_BSZ=2
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=8; GA=4; EVAL_BSZ=4
else
    BSZ=32; GA=1; EVAL_BSZ=4
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS $PYTHON_CMD src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/1-task748_glucose_reverse_cause_event_detection/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/2-task073_commonsenseqa_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/3-task875_emotion_classification/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/4-task002_quoref_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/5-task1687_sentiment140_classification/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/6-task591_sciq_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/7-task363_sst2_polarity_classification/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/8-task1572_samsum_summary/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/9-task511_reddit_tifu_long_text_summarization/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/10-task1290_xsum_summarization/saved_weights \
   --data_dir CL_Benchmark \
   --task_order task748_glucose_reverse_cause_event_detection,task073_commonsenseqa_answer_generation,task875_emotion_classification,task002_quoref_answer_generation,task1687_sentiment140_classification,task591_sciq_answer_generation,task363_sst2_polarity_classification,task1572_samsum_summary,task511_reddit_tifu_long_text_summarization,task1290_xsum_summarization,task639_multi_woz_user_utterance_generation,task1510_evalution_relation_extraction,task181_outcome_extraction,task1729_personachat_generate_next,task1590_diplomacy_text_generation \
   --task_config_dir configs/gen_script_superni_order2_t5_configs/task639_multi_woz_user_utterance_generation \
   --output_dir logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/11-task639_multi_woz_user_utterance_generation \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 100 \
   --run_name gen_script_superni_order2_t5_specroute \
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
   --metric_for_best_model eval_rougeL_for_task639_multi_woz_user_utterance_generation \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

if [ "$GPU_MODE" = "t4_ddp" ]; then
    BSZ=2; GA=8; EVAL_BSZ=2
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=8; GA=4; EVAL_BSZ=4
else
    BSZ=32; GA=1; EVAL_BSZ=4
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS $PYTHON_CMD src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/1-task748_glucose_reverse_cause_event_detection/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/2-task073_commonsenseqa_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/3-task875_emotion_classification/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/4-task002_quoref_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/5-task1687_sentiment140_classification/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/6-task591_sciq_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/7-task363_sst2_polarity_classification/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/8-task1572_samsum_summary/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/9-task511_reddit_tifu_long_text_summarization/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/10-task1290_xsum_summarization/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/11-task639_multi_woz_user_utterance_generation/saved_weights \
   --data_dir CL_Benchmark \
   --task_order task748_glucose_reverse_cause_event_detection,task073_commonsenseqa_answer_generation,task875_emotion_classification,task002_quoref_answer_generation,task1687_sentiment140_classification,task591_sciq_answer_generation,task363_sst2_polarity_classification,task1572_samsum_summary,task511_reddit_tifu_long_text_summarization,task1290_xsum_summarization,task639_multi_woz_user_utterance_generation,task1510_evalution_relation_extraction,task181_outcome_extraction,task1729_personachat_generate_next,task1590_diplomacy_text_generation \
   --task_config_dir configs/gen_script_superni_order2_t5_configs/task1510_evalution_relation_extraction \
   --output_dir logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/12-task1510_evalution_relation_extraction \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 100 \
   --run_name gen_script_superni_order2_t5_specroute \
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
   --metric_for_best_model eval_rougeL_for_task1510_evalution_relation_extraction \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

if [ "$GPU_MODE" = "t4_ddp" ]; then
    BSZ=2; GA=8; EVAL_BSZ=2
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=8; GA=4; EVAL_BSZ=4
else
    BSZ=32; GA=1; EVAL_BSZ=4
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS $PYTHON_CMD src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/1-task748_glucose_reverse_cause_event_detection/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/2-task073_commonsenseqa_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/3-task875_emotion_classification/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/4-task002_quoref_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/5-task1687_sentiment140_classification/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/6-task591_sciq_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/7-task363_sst2_polarity_classification/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/8-task1572_samsum_summary/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/9-task511_reddit_tifu_long_text_summarization/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/10-task1290_xsum_summarization/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/11-task639_multi_woz_user_utterance_generation/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/12-task1510_evalution_relation_extraction/saved_weights \
   --data_dir CL_Benchmark \
   --task_order task748_glucose_reverse_cause_event_detection,task073_commonsenseqa_answer_generation,task875_emotion_classification,task002_quoref_answer_generation,task1687_sentiment140_classification,task591_sciq_answer_generation,task363_sst2_polarity_classification,task1572_samsum_summary,task511_reddit_tifu_long_text_summarization,task1290_xsum_summarization,task639_multi_woz_user_utterance_generation,task1510_evalution_relation_extraction,task181_outcome_extraction,task1729_personachat_generate_next,task1590_diplomacy_text_generation \
   --task_config_dir configs/gen_script_superni_order2_t5_configs/task181_outcome_extraction \
   --output_dir logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/13-task181_outcome_extraction \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 100 \
   --run_name gen_script_superni_order2_t5_specroute \
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
   --metric_for_best_model eval_rougeL_for_task181_outcome_extraction \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

if [ "$GPU_MODE" = "t4_ddp" ]; then
    BSZ=2; GA=8; EVAL_BSZ=2
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=8; GA=4; EVAL_BSZ=4
else
    BSZ=32; GA=1; EVAL_BSZ=4
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS $PYTHON_CMD src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/1-task748_glucose_reverse_cause_event_detection/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/2-task073_commonsenseqa_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/3-task875_emotion_classification/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/4-task002_quoref_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/5-task1687_sentiment140_classification/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/6-task591_sciq_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/7-task363_sst2_polarity_classification/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/8-task1572_samsum_summary/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/9-task511_reddit_tifu_long_text_summarization/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/10-task1290_xsum_summarization/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/11-task639_multi_woz_user_utterance_generation/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/12-task1510_evalution_relation_extraction/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/13-task181_outcome_extraction/saved_weights \
   --data_dir CL_Benchmark \
   --task_order task748_glucose_reverse_cause_event_detection,task073_commonsenseqa_answer_generation,task875_emotion_classification,task002_quoref_answer_generation,task1687_sentiment140_classification,task591_sciq_answer_generation,task363_sst2_polarity_classification,task1572_samsum_summary,task511_reddit_tifu_long_text_summarization,task1290_xsum_summarization,task639_multi_woz_user_utterance_generation,task1510_evalution_relation_extraction,task181_outcome_extraction,task1729_personachat_generate_next,task1590_diplomacy_text_generation \
   --task_config_dir configs/gen_script_superni_order2_t5_configs/task1729_personachat_generate_next \
   --output_dir logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/14-task1729_personachat_generate_next \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 100 \
   --run_name gen_script_superni_order2_t5_specroute \
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
   --metric_for_best_model eval_rougeL_for_task1729_personachat_generate_next \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

if [ "$GPU_MODE" = "t4_ddp" ]; then
    BSZ=2; GA=8; EVAL_BSZ=2
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=8; GA=4; EVAL_BSZ=4
else
    BSZ=32; GA=1; EVAL_BSZ=4
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS $PYTHON_CMD src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/1-task748_glucose_reverse_cause_event_detection/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/2-task073_commonsenseqa_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/3-task875_emotion_classification/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/4-task002_quoref_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/5-task1687_sentiment140_classification/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/6-task591_sciq_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/7-task363_sst2_polarity_classification/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/8-task1572_samsum_summary/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/9-task511_reddit_tifu_long_text_summarization/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/10-task1290_xsum_summarization/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/11-task639_multi_woz_user_utterance_generation/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/12-task1510_evalution_relation_extraction/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/13-task181_outcome_extraction/saved_weights,logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/14-task1729_personachat_generate_next/saved_weights \
   --data_dir CL_Benchmark \
   --task_order task748_glucose_reverse_cause_event_detection,task073_commonsenseqa_answer_generation,task875_emotion_classification,task002_quoref_answer_generation,task1687_sentiment140_classification,task591_sciq_answer_generation,task363_sst2_polarity_classification,task1572_samsum_summary,task511_reddit_tifu_long_text_summarization,task1290_xsum_summarization,task639_multi_woz_user_utterance_generation,task1510_evalution_relation_extraction,task181_outcome_extraction,task1729_personachat_generate_next,task1590_diplomacy_text_generation \
   --task_config_dir configs/gen_script_superni_order2_t5_configs/task1590_diplomacy_text_generation \
   --output_dir logs_and_outputs/gen_script_superni_order2_t5_specroute/outputs/15-task1590_diplomacy_text_generation \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 100 \
   --run_name gen_script_superni_order2_t5_specroute \
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
   --metric_for_best_model eval_rougeL_for_task1590_diplomacy_text_generation \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG
