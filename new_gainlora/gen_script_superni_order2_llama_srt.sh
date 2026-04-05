#!/bin/bash
#SBATCH -J srt-llama
#SBATCH -o srt-llama-%j.out
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 80:00:00
#SBATCH --mem 256G
#SBATCH --gres=gpu:a100-sxm4-80gb:1

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
MODEL_PATH="${1:-meta-llama/Meta-Llama-3-8B}"

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
: ${GPU_MEM:=16000}; : ${NUM_GPUS:=1}

if [ "$GPU_MEM" -lt 20000 ]; then
    IS_T4=1; GPU_MODE="t4_1gpu"; GPU_IDS="${1:-0}"; FP16_FLAG="--gradient_checkpointing"
else
    IS_T4=0; GPU_MODE="a100"; GPU_IDS="${1:-0}"; FP16_FLAG=""
fi

echo "[GPU] $GPU_MODE | CUDA_VISIBLE_DEVICES=$GPU_IDS | $MODEL_PATH"
echo "============================================================"

# Llama: smaller BSZ due to larger model
if [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=1; GA=16; EVAL_BSZ=4
else
    BSZ=1; GA=16; EVAL_BSZ=8
fi

RUN_NAME="superni_order2_llama_srt"
TASK_ORDER="task748_glucose_reverse_cause_event_detection,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task639_multi_woz_user_utterance_generation,task1572_samsum_summary,task1687_sentiment140_classification,task591_sciq_answer_generation,task363_sst2_polarity_classification,task1510_evalution_relation_extraction,task1729_personachat_generate_next,task181_outcome_extraction,task511_reddit_tifu_long_text_summarization,task002_quoref_answer_generation,task1290_xsum_summarization,task875_emotion_classification"
BASE_OUT="logs_and_outputs/$RUN_NAME"
SRT_FLAGS="--use_srt_router --srt_shrink --srt_shrink_factor 0.1 --srt_metric auto --srt_max_emb_samples 500"

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_llama.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $MODEL_PATH \
   --task_order $TASK_ORDER \
   --task_config_dir configs/gen_script_superni_order2_llama_configs/task748_glucose_reverse_cause_event_detection \
   --output_dir $BASE_OUT/outputs/1-task748_glucose_reverse_cause_event_detection \
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
   $SRT_FLAGS \
   

rm -rf $BASE_OUT/outputs/1-task748_glucose_reverse_cause_event_detection/checkpoint*
sleep 5

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_llama.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $MODEL_PATH \
   --load_checkpoint_from $BASE_OUT/outputs/1-task748_glucose_reverse_cause_event_detection/saved_weights/trans_input.pt \
   --previous_lora_path $BASE_OUT/outputs/1-task748_glucose_reverse_cause_event_detection/saved_weights \
   --previous_prompt_key_path $BASE_OUT/outputs/1-task748_glucose_reverse_cause_event_detection/saved_weights/prompts_keys_till_now.pt \
   --task_order $TASK_ORDER \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/gen_script_superni_order2_llama_configs/task073_commonsenseqa_answer_generation \
   --output_dir $BASE_OUT/outputs/2-task073_commonsenseqa_answer_generation \
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
   $SRT_FLAGS \
   --srt_load_path $BASE_OUT/outputs/1-task748_glucose_reverse_cause_event_detection/saved_weights

rm -rf $BASE_OUT/outputs/2-task073_commonsenseqa_answer_generation/checkpoint*
sleep 5

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_llama.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $MODEL_PATH \
   --load_checkpoint_from $BASE_OUT/outputs/2-task073_commonsenseqa_answer_generation/saved_weights/trans_input.pt \
   --previous_lora_path $BASE_OUT/outputs/1-task748_glucose_reverse_cause_event_detection/saved_weights,$BASE_OUT/outputs/2-task073_commonsenseqa_answer_generation/saved_weights \
   --previous_prompt_key_path $BASE_OUT/outputs/2-task073_commonsenseqa_answer_generation/saved_weights/prompts_keys_till_now.pt \
   --task_order $TASK_ORDER \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/gen_script_superni_order2_llama_configs/task1590_diplomacy_text_generation \
   --output_dir $BASE_OUT/outputs/3-task1590_diplomacy_text_generation \
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
   $SRT_FLAGS \
   --srt_load_path $BASE_OUT/outputs/2-task073_commonsenseqa_answer_generation/saved_weights

rm -rf $BASE_OUT/outputs/3-task1590_diplomacy_text_generation/checkpoint*
sleep 5

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_llama.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $MODEL_PATH \
   --load_checkpoint_from $BASE_OUT/outputs/3-task1590_diplomacy_text_generation/saved_weights/trans_input.pt \
   --previous_lora_path $BASE_OUT/outputs/1-task748_glucose_reverse_cause_event_detection/saved_weights,$BASE_OUT/outputs/2-task073_commonsenseqa_answer_generation/saved_weights,$BASE_OUT/outputs/3-task1590_diplomacy_text_generation/saved_weights \
   --previous_prompt_key_path $BASE_OUT/outputs/3-task1590_diplomacy_text_generation/saved_weights/prompts_keys_till_now.pt \
   --task_order $TASK_ORDER \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/gen_script_superni_order2_llama_configs/task639_multi_woz_user_utterance_generation \
   --output_dir $BASE_OUT/outputs/4-task639_multi_woz_user_utterance_generation \
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
   $SRT_FLAGS \
   --srt_load_path $BASE_OUT/outputs/3-task1590_diplomacy_text_generation/saved_weights

rm -rf $BASE_OUT/outputs/4-task639_multi_woz_user_utterance_generation/checkpoint*
sleep 5

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_llama.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $MODEL_PATH \
   --load_checkpoint_from $BASE_OUT/outputs/4-task639_multi_woz_user_utterance_generation/saved_weights/trans_input.pt \
   --previous_lora_path $BASE_OUT/outputs/1-task748_glucose_reverse_cause_event_detection/saved_weights,$BASE_OUT/outputs/2-task073_commonsenseqa_answer_generation/saved_weights,$BASE_OUT/outputs/3-task1590_diplomacy_text_generation/saved_weights,$BASE_OUT/outputs/4-task639_multi_woz_user_utterance_generation/saved_weights \
   --previous_prompt_key_path $BASE_OUT/outputs/4-task639_multi_woz_user_utterance_generation/saved_weights/prompts_keys_till_now.pt \
   --task_order $TASK_ORDER \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/gen_script_superni_order2_llama_configs/task1572_samsum_summary \
   --output_dir $BASE_OUT/outputs/5-task1572_samsum_summary \
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
   $SRT_FLAGS \
   --srt_load_path $BASE_OUT/outputs/4-task639_multi_woz_user_utterance_generation/saved_weights

rm -rf $BASE_OUT/outputs/5-task1572_samsum_summary/checkpoint*
sleep 5

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_llama.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $MODEL_PATH \
   --load_checkpoint_from $BASE_OUT/outputs/5-task1572_samsum_summary/saved_weights/trans_input.pt \
   --previous_lora_path $BASE_OUT/outputs/1-task748_glucose_reverse_cause_event_detection/saved_weights,$BASE_OUT/outputs/2-task073_commonsenseqa_answer_generation/saved_weights,$BASE_OUT/outputs/3-task1590_diplomacy_text_generation/saved_weights,$BASE_OUT/outputs/4-task639_multi_woz_user_utterance_generation/saved_weights,$BASE_OUT/outputs/5-task1572_samsum_summary/saved_weights \
   --previous_prompt_key_path $BASE_OUT/outputs/5-task1572_samsum_summary/saved_weights/prompts_keys_till_now.pt \
   --task_order $TASK_ORDER \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/gen_script_superni_order2_llama_configs/task1687_sentiment140_classification \
   --output_dir $BASE_OUT/outputs/6-task1687_sentiment140_classification \
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
   $SRT_FLAGS \
   --srt_load_path $BASE_OUT/outputs/5-task1572_samsum_summary/saved_weights

rm -rf $BASE_OUT/outputs/6-task1687_sentiment140_classification/checkpoint*
sleep 5

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_llama.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $MODEL_PATH \
   --load_checkpoint_from $BASE_OUT/outputs/6-task1687_sentiment140_classification/saved_weights/trans_input.pt \
   --previous_lora_path $BASE_OUT/outputs/1-task748_glucose_reverse_cause_event_detection/saved_weights,$BASE_OUT/outputs/2-task073_commonsenseqa_answer_generation/saved_weights,$BASE_OUT/outputs/3-task1590_diplomacy_text_generation/saved_weights,$BASE_OUT/outputs/4-task639_multi_woz_user_utterance_generation/saved_weights,$BASE_OUT/outputs/5-task1572_samsum_summary/saved_weights,$BASE_OUT/outputs/6-task1687_sentiment140_classification/saved_weights \
   --previous_prompt_key_path $BASE_OUT/outputs/6-task1687_sentiment140_classification/saved_weights/prompts_keys_till_now.pt \
   --task_order $TASK_ORDER \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/gen_script_superni_order2_llama_configs/task591_sciq_answer_generation \
   --output_dir $BASE_OUT/outputs/7-task591_sciq_answer_generation \
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
   $SRT_FLAGS \
   --srt_load_path $BASE_OUT/outputs/6-task1687_sentiment140_classification/saved_weights

rm -rf $BASE_OUT/outputs/7-task591_sciq_answer_generation/checkpoint*
sleep 5

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_llama.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $MODEL_PATH \
   --load_checkpoint_from $BASE_OUT/outputs/7-task591_sciq_answer_generation/saved_weights/trans_input.pt \
   --previous_lora_path $BASE_OUT/outputs/1-task748_glucose_reverse_cause_event_detection/saved_weights,$BASE_OUT/outputs/2-task073_commonsenseqa_answer_generation/saved_weights,$BASE_OUT/outputs/3-task1590_diplomacy_text_generation/saved_weights,$BASE_OUT/outputs/4-task639_multi_woz_user_utterance_generation/saved_weights,$BASE_OUT/outputs/5-task1572_samsum_summary/saved_weights,$BASE_OUT/outputs/6-task1687_sentiment140_classification/saved_weights,$BASE_OUT/outputs/7-task591_sciq_answer_generation/saved_weights \
   --previous_prompt_key_path $BASE_OUT/outputs/7-task591_sciq_answer_generation/saved_weights/prompts_keys_till_now.pt \
   --task_order $TASK_ORDER \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/gen_script_superni_order2_llama_configs/task363_sst2_polarity_classification \
   --output_dir $BASE_OUT/outputs/8-task363_sst2_polarity_classification \
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
   $SRT_FLAGS \
   --srt_load_path $BASE_OUT/outputs/7-task591_sciq_answer_generation/saved_weights

rm -rf $BASE_OUT/outputs/8-task363_sst2_polarity_classification/checkpoint*
sleep 5

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_llama.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $MODEL_PATH \
   --load_checkpoint_from $BASE_OUT/outputs/8-task363_sst2_polarity_classification/saved_weights/trans_input.pt \
   --previous_lora_path $BASE_OUT/outputs/1-task748_glucose_reverse_cause_event_detection/saved_weights,$BASE_OUT/outputs/2-task073_commonsenseqa_answer_generation/saved_weights,$BASE_OUT/outputs/3-task1590_diplomacy_text_generation/saved_weights,$BASE_OUT/outputs/4-task639_multi_woz_user_utterance_generation/saved_weights,$BASE_OUT/outputs/5-task1572_samsum_summary/saved_weights,$BASE_OUT/outputs/6-task1687_sentiment140_classification/saved_weights,$BASE_OUT/outputs/7-task591_sciq_answer_generation/saved_weights,$BASE_OUT/outputs/8-task363_sst2_polarity_classification/saved_weights \
   --previous_prompt_key_path $BASE_OUT/outputs/8-task363_sst2_polarity_classification/saved_weights/prompts_keys_till_now.pt \
   --task_order $TASK_ORDER \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/gen_script_superni_order2_llama_configs/task1510_evalution_relation_extraction \
   --output_dir $BASE_OUT/outputs/9-task1510_evalution_relation_extraction \
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
   $SRT_FLAGS \
   --srt_load_path $BASE_OUT/outputs/8-task363_sst2_polarity_classification/saved_weights

rm -rf $BASE_OUT/outputs/9-task1510_evalution_relation_extraction/checkpoint*
sleep 5

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_llama.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $MODEL_PATH \
   --load_checkpoint_from $BASE_OUT/outputs/9-task1510_evalution_relation_extraction/saved_weights/trans_input.pt \
   --previous_lora_path $BASE_OUT/outputs/1-task748_glucose_reverse_cause_event_detection/saved_weights,$BASE_OUT/outputs/2-task073_commonsenseqa_answer_generation/saved_weights,$BASE_OUT/outputs/3-task1590_diplomacy_text_generation/saved_weights,$BASE_OUT/outputs/4-task639_multi_woz_user_utterance_generation/saved_weights,$BASE_OUT/outputs/5-task1572_samsum_summary/saved_weights,$BASE_OUT/outputs/6-task1687_sentiment140_classification/saved_weights,$BASE_OUT/outputs/7-task591_sciq_answer_generation/saved_weights,$BASE_OUT/outputs/8-task363_sst2_polarity_classification/saved_weights,$BASE_OUT/outputs/9-task1510_evalution_relation_extraction/saved_weights \
   --previous_prompt_key_path $BASE_OUT/outputs/9-task1510_evalution_relation_extraction/saved_weights/prompts_keys_till_now.pt \
   --task_order $TASK_ORDER \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/gen_script_superni_order2_llama_configs/task1729_personachat_generate_next \
   --output_dir $BASE_OUT/outputs/10-task1729_personachat_generate_next \
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
   $SRT_FLAGS \
   --srt_load_path $BASE_OUT/outputs/9-task1510_evalution_relation_extraction/saved_weights

rm -rf $BASE_OUT/outputs/10-task1729_personachat_generate_next/checkpoint*
sleep 5

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_llama.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $MODEL_PATH \
   --load_checkpoint_from $BASE_OUT/outputs/10-task1729_personachat_generate_next/saved_weights/trans_input.pt \
   --previous_lora_path $BASE_OUT/outputs/1-task748_glucose_reverse_cause_event_detection/saved_weights,$BASE_OUT/outputs/2-task073_commonsenseqa_answer_generation/saved_weights,$BASE_OUT/outputs/3-task1590_diplomacy_text_generation/saved_weights,$BASE_OUT/outputs/4-task639_multi_woz_user_utterance_generation/saved_weights,$BASE_OUT/outputs/5-task1572_samsum_summary/saved_weights,$BASE_OUT/outputs/6-task1687_sentiment140_classification/saved_weights,$BASE_OUT/outputs/7-task591_sciq_answer_generation/saved_weights,$BASE_OUT/outputs/8-task363_sst2_polarity_classification/saved_weights,$BASE_OUT/outputs/9-task1510_evalution_relation_extraction/saved_weights,$BASE_OUT/outputs/10-task1729_personachat_generate_next/saved_weights \
   --previous_prompt_key_path $BASE_OUT/outputs/10-task1729_personachat_generate_next/saved_weights/prompts_keys_till_now.pt \
   --task_order $TASK_ORDER \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/gen_script_superni_order2_llama_configs/task181_outcome_extraction \
   --output_dir $BASE_OUT/outputs/11-task181_outcome_extraction \
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
   $SRT_FLAGS \
   --srt_load_path $BASE_OUT/outputs/10-task1729_personachat_generate_next/saved_weights

rm -rf $BASE_OUT/outputs/11-task181_outcome_extraction/checkpoint*
sleep 5

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_llama.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $MODEL_PATH \
   --load_checkpoint_from $BASE_OUT/outputs/11-task181_outcome_extraction/saved_weights/trans_input.pt \
   --previous_lora_path $BASE_OUT/outputs/1-task748_glucose_reverse_cause_event_detection/saved_weights,$BASE_OUT/outputs/2-task073_commonsenseqa_answer_generation/saved_weights,$BASE_OUT/outputs/3-task1590_diplomacy_text_generation/saved_weights,$BASE_OUT/outputs/4-task639_multi_woz_user_utterance_generation/saved_weights,$BASE_OUT/outputs/5-task1572_samsum_summary/saved_weights,$BASE_OUT/outputs/6-task1687_sentiment140_classification/saved_weights,$BASE_OUT/outputs/7-task591_sciq_answer_generation/saved_weights,$BASE_OUT/outputs/8-task363_sst2_polarity_classification/saved_weights,$BASE_OUT/outputs/9-task1510_evalution_relation_extraction/saved_weights,$BASE_OUT/outputs/10-task1729_personachat_generate_next/saved_weights,$BASE_OUT/outputs/11-task181_outcome_extraction/saved_weights \
   --previous_prompt_key_path $BASE_OUT/outputs/11-task181_outcome_extraction/saved_weights/prompts_keys_till_now.pt \
   --task_order $TASK_ORDER \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/gen_script_superni_order2_llama_configs/task511_reddit_tifu_long_text_summarization \
   --output_dir $BASE_OUT/outputs/12-task511_reddit_tifu_long_text_summarization \
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
   $SRT_FLAGS \
   --srt_load_path $BASE_OUT/outputs/11-task181_outcome_extraction/saved_weights

rm -rf $BASE_OUT/outputs/12-task511_reddit_tifu_long_text_summarization/checkpoint*
sleep 5

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_llama.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $MODEL_PATH \
   --load_checkpoint_from $BASE_OUT/outputs/12-task511_reddit_tifu_long_text_summarization/saved_weights/trans_input.pt \
   --previous_lora_path $BASE_OUT/outputs/1-task748_glucose_reverse_cause_event_detection/saved_weights,$BASE_OUT/outputs/2-task073_commonsenseqa_answer_generation/saved_weights,$BASE_OUT/outputs/3-task1590_diplomacy_text_generation/saved_weights,$BASE_OUT/outputs/4-task639_multi_woz_user_utterance_generation/saved_weights,$BASE_OUT/outputs/5-task1572_samsum_summary/saved_weights,$BASE_OUT/outputs/6-task1687_sentiment140_classification/saved_weights,$BASE_OUT/outputs/7-task591_sciq_answer_generation/saved_weights,$BASE_OUT/outputs/8-task363_sst2_polarity_classification/saved_weights,$BASE_OUT/outputs/9-task1510_evalution_relation_extraction/saved_weights,$BASE_OUT/outputs/10-task1729_personachat_generate_next/saved_weights,$BASE_OUT/outputs/11-task181_outcome_extraction/saved_weights,$BASE_OUT/outputs/12-task511_reddit_tifu_long_text_summarization/saved_weights \
   --previous_prompt_key_path $BASE_OUT/outputs/12-task511_reddit_tifu_long_text_summarization/saved_weights/prompts_keys_till_now.pt \
   --task_order $TASK_ORDER \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/gen_script_superni_order2_llama_configs/task002_quoref_answer_generation \
   --output_dir $BASE_OUT/outputs/13-task002_quoref_answer_generation \
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
   $SRT_FLAGS \
   --srt_load_path $BASE_OUT/outputs/12-task511_reddit_tifu_long_text_summarization/saved_weights

rm -rf $BASE_OUT/outputs/13-task002_quoref_answer_generation/checkpoint*
sleep 5

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_llama.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $MODEL_PATH \
   --load_checkpoint_from $BASE_OUT/outputs/13-task002_quoref_answer_generation/saved_weights/trans_input.pt \
   --previous_lora_path $BASE_OUT/outputs/1-task748_glucose_reverse_cause_event_detection/saved_weights,$BASE_OUT/outputs/2-task073_commonsenseqa_answer_generation/saved_weights,$BASE_OUT/outputs/3-task1590_diplomacy_text_generation/saved_weights,$BASE_OUT/outputs/4-task639_multi_woz_user_utterance_generation/saved_weights,$BASE_OUT/outputs/5-task1572_samsum_summary/saved_weights,$BASE_OUT/outputs/6-task1687_sentiment140_classification/saved_weights,$BASE_OUT/outputs/7-task591_sciq_answer_generation/saved_weights,$BASE_OUT/outputs/8-task363_sst2_polarity_classification/saved_weights,$BASE_OUT/outputs/9-task1510_evalution_relation_extraction/saved_weights,$BASE_OUT/outputs/10-task1729_personachat_generate_next/saved_weights,$BASE_OUT/outputs/11-task181_outcome_extraction/saved_weights,$BASE_OUT/outputs/12-task511_reddit_tifu_long_text_summarization/saved_weights,$BASE_OUT/outputs/13-task002_quoref_answer_generation/saved_weights \
   --previous_prompt_key_path $BASE_OUT/outputs/13-task002_quoref_answer_generation/saved_weights/prompts_keys_till_now.pt \
   --task_order $TASK_ORDER \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/gen_script_superni_order2_llama_configs/task1290_xsum_summarization \
   --output_dir $BASE_OUT/outputs/14-task1290_xsum_summarization \
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
   $SRT_FLAGS \
   --srt_load_path $BASE_OUT/outputs/13-task002_quoref_answer_generation/saved_weights

rm -rf $BASE_OUT/outputs/14-task1290_xsum_summarization/checkpoint*
sleep 5

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_llama.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $MODEL_PATH \
   --load_checkpoint_from $BASE_OUT/outputs/14-task1290_xsum_summarization/saved_weights/trans_input.pt \
   --previous_lora_path $BASE_OUT/outputs/1-task748_glucose_reverse_cause_event_detection/saved_weights,$BASE_OUT/outputs/2-task073_commonsenseqa_answer_generation/saved_weights,$BASE_OUT/outputs/3-task1590_diplomacy_text_generation/saved_weights,$BASE_OUT/outputs/4-task639_multi_woz_user_utterance_generation/saved_weights,$BASE_OUT/outputs/5-task1572_samsum_summary/saved_weights,$BASE_OUT/outputs/6-task1687_sentiment140_classification/saved_weights,$BASE_OUT/outputs/7-task591_sciq_answer_generation/saved_weights,$BASE_OUT/outputs/8-task363_sst2_polarity_classification/saved_weights,$BASE_OUT/outputs/9-task1510_evalution_relation_extraction/saved_weights,$BASE_OUT/outputs/10-task1729_personachat_generate_next/saved_weights,$BASE_OUT/outputs/11-task181_outcome_extraction/saved_weights,$BASE_OUT/outputs/12-task511_reddit_tifu_long_text_summarization/saved_weights,$BASE_OUT/outputs/13-task002_quoref_answer_generation/saved_weights,$BASE_OUT/outputs/14-task1290_xsum_summarization/saved_weights \
   --previous_prompt_key_path $BASE_OUT/outputs/14-task1290_xsum_summarization/saved_weights/prompts_keys_till_now.pt \
   --task_order $TASK_ORDER \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/gen_script_superni_order2_llama_configs/task875_emotion_classification \
   --output_dir $BASE_OUT/outputs/15-task875_emotion_classification \
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
   $SRT_FLAGS \
   --srt_load_path $BASE_OUT/outputs/14-task1290_xsum_summarization/saved_weights

rm -rf $BASE_OUT/outputs/15-task875_emotion_classification/checkpoint*
sleep 5
echo "[DONE] All 15 tasks complete. Run: python score.py $RUN_NAME $RUN_NAME"
