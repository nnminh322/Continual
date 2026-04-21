#!/bin/bash
# GENERATED FILE. DO NOT EDIT.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
ROOT_BASE="$WORKSPACE_DIR/llama_epoch_ablation/root_gainlora_bugfix"
DATA_DIR="$WORKSPACE_DIR/root_gainlora/CL_Benchmark"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-deepspeed}"
MODEL_PATH="${1:-meta-llama/Llama-2-7b-hf}"
GPU_IDS="${2:-0}"
MASTER_PORT="${3:-49500}"

resolve_deepspeed_launcher() {
    if command -v "$DEEPSPEED_BIN" >/dev/null 2>&1; then
        printf '%s
' "$DEEPSPEED_BIN"
        return 0
    fi

    if "$PYTHON_BIN" -c "import deepspeed.launcher.runner" >/dev/null 2>&1; then
        printf '%s
' "$PYTHON_BIN -m deepspeed.launcher.runner"
        return 0
    fi

    echo "Could not find a runnable DeepSpeed launcher." >&2
    echo "Tried CLI: $DEEPSPEED_BIN" >&2
    echo "Tried module: $PYTHON_BIN -m deepspeed.launcher.runner" >&2
    echo "Set DEEPSPEED_BIN explicitly or install deepspeed into the active environment." >&2
    return 1
}

DEEPSPEED_LAUNCHER=$(resolve_deepspeed_launcher)

RUN_NAME="gen_script_superni_order1_llama_gainlora_inflora_rootbugfix"
BASE_OUT="$WORKSPACE_DIR/llama_epoch_ablation/logs_and_outputs/$RUN_NAME"

mkdir -p "$BASE_OUT/outputs"

echo "[ROOT-ORDER1] model_path=$MODEL_PATH"
echo "[ROOT-ORDER1] gpu_ids=$GPU_IDS"
echo "[ROOT-ORDER1] master_port=$MASTER_PORT"
echo "[ROOT-ORDER1] deepspeed_launcher=$DEEPSPEED_LAUNCHER"
echo "[ROOT-ORDER1] output_root=$BASE_OUT"
echo "============================================================"

${DEEPSPEED_BIN} --include localhost:${GPU_IDS} --master_port ${MASTER_PORT} "$ROOT_BASE/src/run_llama.py" \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path "$MODEL_PATH" \
   --data_dir "$DATA_DIR" \
   --task_order task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification \
   --task_config_dir "$ROOT_BASE/configs/gen_script_superni_order1_llama_configs/task1572_samsum_summary" \
   --output_dir "$BASE_OUT/outputs/1-task1572_samsum_summary" \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 16 \
   --learning_rate 5e-05 \
   --num_train_epochs 50 \
   --bf16 \
   --deepspeed "$ROOT_BASE/configs/ds_configs/stage2.config" \
   --run_name "$RUN_NAME" \
   --max_source_length 1024 \
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
   --eval_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --trans_hidden_dim 100 \
   --attn_lr 0 \
   --chunk 4 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995

rm -rf "$BASE_OUT/outputs/1-task1572_samsum_summary"/checkpoint*

${DEEPSPEED_BIN} --include localhost:${GPU_IDS} --master_port ${MASTER_PORT} "$ROOT_BASE/src/run_llama.py" \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path "$MODEL_PATH" \
   --data_dir "$DATA_DIR" \
   --task_order task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification \
   --task_config_dir "$ROOT_BASE/configs/gen_script_superni_order1_llama_configs/task363_sst2_polarity_classification" \
   --output_dir "$BASE_OUT/outputs/2-task363_sst2_polarity_classification" \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 2 \
   --learning_rate 5e-05 \
   --num_train_epochs 50 \
   --bf16 \
   --deepspeed "$ROOT_BASE/configs/ds_configs/stage2.config" \
   --run_name "$RUN_NAME" \
   --max_source_length 1024 \
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
   --eval_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --trans_hidden_dim 100 \
   --attn_lr 0 \
   --chunk 4 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995 \
   --previous_lora_path "$BASE_OUT/outputs/1-task1572_samsum_summary/saved_weights" \
   --previous_prompt_key_path "$BASE_OUT/outputs/1-task1572_samsum_summary/saved_weights/prompts_keys_till_now.pt" \
   --load_checkpoint_from "$BASE_OUT/outputs/1-task1572_samsum_summary/saved_weights/trans_input.pt"

rm -rf "$BASE_OUT/outputs/2-task363_sst2_polarity_classification"/checkpoint*

${DEEPSPEED_BIN} --include localhost:${GPU_IDS} --master_port ${MASTER_PORT} "$ROOT_BASE/src/run_llama.py" \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path "$MODEL_PATH" \
   --data_dir "$DATA_DIR" \
   --task_order task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification \
   --task_config_dir "$ROOT_BASE/configs/gen_script_superni_order1_llama_configs/task1290_xsum_summarization" \
   --output_dir "$BASE_OUT/outputs/3-task1290_xsum_summarization" \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 32 \
   --learning_rate 5e-05 \
   --num_train_epochs 50 \
   --bf16 \
   --deepspeed "$ROOT_BASE/configs/ds_configs/stage2.config" \
   --run_name "$RUN_NAME" \
   --max_source_length 1024 \
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
   --eval_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --trans_hidden_dim 100 \
   --attn_lr 0 \
   --chunk 4 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995 \
   --previous_lora_path "$BASE_OUT/outputs/1-task1572_samsum_summary/saved_weights,$BASE_OUT/outputs/2-task363_sst2_polarity_classification/saved_weights" \
   --previous_prompt_key_path "$BASE_OUT/outputs/2-task363_sst2_polarity_classification/saved_weights/prompts_keys_till_now.pt" \
   --load_checkpoint_from "$BASE_OUT/outputs/2-task363_sst2_polarity_classification/saved_weights/trans_input.pt"

rm -rf "$BASE_OUT/outputs/3-task1290_xsum_summarization"/checkpoint*

${DEEPSPEED_BIN} --include localhost:${GPU_IDS} --master_port ${MASTER_PORT} "$ROOT_BASE/src/run_llama.py" \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path "$MODEL_PATH" \
   --data_dir "$DATA_DIR" \
   --task_order task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification \
   --task_config_dir "$ROOT_BASE/configs/gen_script_superni_order1_llama_configs/task181_outcome_extraction" \
   --output_dir "$BASE_OUT/outputs/4-task181_outcome_extraction" \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 8 \
   --learning_rate 5e-05 \
   --num_train_epochs 50 \
   --bf16 \
   --deepspeed "$ROOT_BASE/configs/ds_configs/stage2.config" \
   --run_name "$RUN_NAME" \
   --max_source_length 1024 \
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
   --eval_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --trans_hidden_dim 100 \
   --attn_lr 0 \
   --chunk 4 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995 \
   --previous_lora_path "$BASE_OUT/outputs/1-task1572_samsum_summary/saved_weights,$BASE_OUT/outputs/2-task363_sst2_polarity_classification/saved_weights,$BASE_OUT/outputs/3-task1290_xsum_summarization/saved_weights" \
   --previous_prompt_key_path "$BASE_OUT/outputs/3-task1290_xsum_summarization/saved_weights/prompts_keys_till_now.pt" \
   --load_checkpoint_from "$BASE_OUT/outputs/3-task1290_xsum_summarization/saved_weights/trans_input.pt"

rm -rf "$BASE_OUT/outputs/4-task181_outcome_extraction"/checkpoint*

${DEEPSPEED_BIN} --include localhost:${GPU_IDS} --master_port ${MASTER_PORT} "$ROOT_BASE/src/run_llama.py" \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path "$MODEL_PATH" \
   --data_dir "$DATA_DIR" \
   --task_order task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification \
   --task_config_dir "$ROOT_BASE/configs/gen_script_superni_order1_llama_configs/task002_quoref_answer_generation" \
   --output_dir "$BASE_OUT/outputs/5-task002_quoref_answer_generation" \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 32 \
   --learning_rate 5e-05 \
   --num_train_epochs 50 \
   --bf16 \
   --deepspeed "$ROOT_BASE/configs/ds_configs/stage2.config" \
   --run_name "$RUN_NAME" \
   --max_source_length 1024 \
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
   --eval_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --trans_hidden_dim 100 \
   --attn_lr 0 \
   --chunk 4 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995 \
   --previous_lora_path "$BASE_OUT/outputs/1-task1572_samsum_summary/saved_weights,$BASE_OUT/outputs/2-task363_sst2_polarity_classification/saved_weights,$BASE_OUT/outputs/3-task1290_xsum_summarization/saved_weights,$BASE_OUT/outputs/4-task181_outcome_extraction/saved_weights" \
   --previous_prompt_key_path "$BASE_OUT/outputs/4-task181_outcome_extraction/saved_weights/prompts_keys_till_now.pt" \
   --load_checkpoint_from "$BASE_OUT/outputs/4-task181_outcome_extraction/saved_weights/trans_input.pt"

rm -rf "$BASE_OUT/outputs/5-task002_quoref_answer_generation"/checkpoint*

${DEEPSPEED_BIN} --include localhost:${GPU_IDS} --master_port ${MASTER_PORT} "$ROOT_BASE/src/run_llama.py" \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path "$MODEL_PATH" \
   --data_dir "$DATA_DIR" \
   --task_order task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification \
   --task_config_dir "$ROOT_BASE/configs/gen_script_superni_order1_llama_configs/task1510_evalution_relation_extraction" \
   --output_dir "$BASE_OUT/outputs/6-task1510_evalution_relation_extraction" \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 2 \
   --learning_rate 5e-05 \
   --num_train_epochs 50 \
   --bf16 \
   --deepspeed "$ROOT_BASE/configs/ds_configs/stage2.config" \
   --run_name "$RUN_NAME" \
   --max_source_length 1024 \
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
   --eval_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --trans_hidden_dim 100 \
   --attn_lr 0 \
   --chunk 4 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995 \
   --previous_lora_path "$BASE_OUT/outputs/1-task1572_samsum_summary/saved_weights,$BASE_OUT/outputs/2-task363_sst2_polarity_classification/saved_weights,$BASE_OUT/outputs/3-task1290_xsum_summarization/saved_weights,$BASE_OUT/outputs/4-task181_outcome_extraction/saved_weights,$BASE_OUT/outputs/5-task002_quoref_answer_generation/saved_weights" \
   --previous_prompt_key_path "$BASE_OUT/outputs/5-task002_quoref_answer_generation/saved_weights/prompts_keys_till_now.pt" \
   --load_checkpoint_from "$BASE_OUT/outputs/5-task002_quoref_answer_generation/saved_weights/trans_input.pt"

rm -rf "$BASE_OUT/outputs/6-task1510_evalution_relation_extraction"/checkpoint*

${DEEPSPEED_BIN} --include localhost:${GPU_IDS} --master_port ${MASTER_PORT} "$ROOT_BASE/src/run_llama.py" \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path "$MODEL_PATH" \
   --data_dir "$DATA_DIR" \
   --task_order task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification \
   --task_config_dir "$ROOT_BASE/configs/gen_script_superni_order1_llama_configs/task639_multi_woz_user_utterance_generation" \
   --output_dir "$BASE_OUT/outputs/7-task639_multi_woz_user_utterance_generation" \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 2 \
   --learning_rate 5e-05 \
   --num_train_epochs 50 \
   --bf16 \
   --deepspeed "$ROOT_BASE/configs/ds_configs/stage2.config" \
   --run_name "$RUN_NAME" \
   --max_source_length 1024 \
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
   --eval_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --trans_hidden_dim 100 \
   --attn_lr 0 \
   --chunk 4 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995 \
   --previous_lora_path "$BASE_OUT/outputs/1-task1572_samsum_summary/saved_weights,$BASE_OUT/outputs/2-task363_sst2_polarity_classification/saved_weights,$BASE_OUT/outputs/3-task1290_xsum_summarization/saved_weights,$BASE_OUT/outputs/4-task181_outcome_extraction/saved_weights,$BASE_OUT/outputs/5-task002_quoref_answer_generation/saved_weights,$BASE_OUT/outputs/6-task1510_evalution_relation_extraction/saved_weights" \
   --previous_prompt_key_path "$BASE_OUT/outputs/6-task1510_evalution_relation_extraction/saved_weights/prompts_keys_till_now.pt" \
   --load_checkpoint_from "$BASE_OUT/outputs/6-task1510_evalution_relation_extraction/saved_weights/trans_input.pt"

rm -rf "$BASE_OUT/outputs/7-task639_multi_woz_user_utterance_generation"/checkpoint*

${DEEPSPEED_BIN} --include localhost:${GPU_IDS} --master_port ${MASTER_PORT} "$ROOT_BASE/src/run_llama.py" \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path "$MODEL_PATH" \
   --data_dir "$DATA_DIR" \
   --task_order task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification \
   --task_config_dir "$ROOT_BASE/configs/gen_script_superni_order1_llama_configs/task1729_personachat_generate_next" \
   --output_dir "$BASE_OUT/outputs/8-task1729_personachat_generate_next" \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 16 \
   --learning_rate 5e-05 \
   --num_train_epochs 50 \
   --bf16 \
   --deepspeed "$ROOT_BASE/configs/ds_configs/stage2.config" \
   --run_name "$RUN_NAME" \
   --max_source_length 1024 \
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
   --eval_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --trans_hidden_dim 100 \
   --attn_lr 0 \
   --chunk 4 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995 \
   --previous_lora_path "$BASE_OUT/outputs/1-task1572_samsum_summary/saved_weights,$BASE_OUT/outputs/2-task363_sst2_polarity_classification/saved_weights,$BASE_OUT/outputs/3-task1290_xsum_summarization/saved_weights,$BASE_OUT/outputs/4-task181_outcome_extraction/saved_weights,$BASE_OUT/outputs/5-task002_quoref_answer_generation/saved_weights,$BASE_OUT/outputs/6-task1510_evalution_relation_extraction/saved_weights,$BASE_OUT/outputs/7-task639_multi_woz_user_utterance_generation/saved_weights" \
   --previous_prompt_key_path "$BASE_OUT/outputs/7-task639_multi_woz_user_utterance_generation/saved_weights/prompts_keys_till_now.pt" \
   --load_checkpoint_from "$BASE_OUT/outputs/7-task639_multi_woz_user_utterance_generation/saved_weights/trans_input.pt"

rm -rf "$BASE_OUT/outputs/8-task1729_personachat_generate_next"/checkpoint*

${DEEPSPEED_BIN} --include localhost:${GPU_IDS} --master_port ${MASTER_PORT} "$ROOT_BASE/src/run_llama.py" \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path "$MODEL_PATH" \
   --data_dir "$DATA_DIR" \
   --task_order task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification \
   --task_config_dir "$ROOT_BASE/configs/gen_script_superni_order1_llama_configs/task073_commonsenseqa_answer_generation" \
   --output_dir "$BASE_OUT/outputs/9-task073_commonsenseqa_answer_generation" \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 4 \
   --learning_rate 5e-05 \
   --num_train_epochs 50 \
   --bf16 \
   --deepspeed "$ROOT_BASE/configs/ds_configs/stage2.config" \
   --run_name "$RUN_NAME" \
   --max_source_length 1024 \
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
   --eval_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --trans_hidden_dim 100 \
   --attn_lr 0 \
   --chunk 4 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995 \
   --previous_lora_path "$BASE_OUT/outputs/1-task1572_samsum_summary/saved_weights,$BASE_OUT/outputs/2-task363_sst2_polarity_classification/saved_weights,$BASE_OUT/outputs/3-task1290_xsum_summarization/saved_weights,$BASE_OUT/outputs/4-task181_outcome_extraction/saved_weights,$BASE_OUT/outputs/5-task002_quoref_answer_generation/saved_weights,$BASE_OUT/outputs/6-task1510_evalution_relation_extraction/saved_weights,$BASE_OUT/outputs/7-task639_multi_woz_user_utterance_generation/saved_weights,$BASE_OUT/outputs/8-task1729_personachat_generate_next/saved_weights" \
   --previous_prompt_key_path "$BASE_OUT/outputs/8-task1729_personachat_generate_next/saved_weights/prompts_keys_till_now.pt" \
   --load_checkpoint_from "$BASE_OUT/outputs/8-task1729_personachat_generate_next/saved_weights/trans_input.pt"

rm -rf "$BASE_OUT/outputs/9-task073_commonsenseqa_answer_generation"/checkpoint*

${DEEPSPEED_BIN} --include localhost:${GPU_IDS} --master_port ${MASTER_PORT} "$ROOT_BASE/src/run_llama.py" \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path "$MODEL_PATH" \
   --data_dir "$DATA_DIR" \
   --task_order task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification \
   --task_config_dir "$ROOT_BASE/configs/gen_script_superni_order1_llama_configs/task1590_diplomacy_text_generation" \
   --output_dir "$BASE_OUT/outputs/10-task1590_diplomacy_text_generation" \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 16 \
   --learning_rate 5e-05 \
   --num_train_epochs 50 \
   --bf16 \
   --deepspeed "$ROOT_BASE/configs/ds_configs/stage2.config" \
   --run_name "$RUN_NAME" \
   --max_source_length 1024 \
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
   --eval_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --trans_hidden_dim 100 \
   --attn_lr 0 \
   --chunk 4 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995 \
   --previous_lora_path "$BASE_OUT/outputs/1-task1572_samsum_summary/saved_weights,$BASE_OUT/outputs/2-task363_sst2_polarity_classification/saved_weights,$BASE_OUT/outputs/3-task1290_xsum_summarization/saved_weights,$BASE_OUT/outputs/4-task181_outcome_extraction/saved_weights,$BASE_OUT/outputs/5-task002_quoref_answer_generation/saved_weights,$BASE_OUT/outputs/6-task1510_evalution_relation_extraction/saved_weights,$BASE_OUT/outputs/7-task639_multi_woz_user_utterance_generation/saved_weights,$BASE_OUT/outputs/8-task1729_personachat_generate_next/saved_weights,$BASE_OUT/outputs/9-task073_commonsenseqa_answer_generation/saved_weights" \
   --previous_prompt_key_path "$BASE_OUT/outputs/9-task073_commonsenseqa_answer_generation/saved_weights/prompts_keys_till_now.pt" \
   --load_checkpoint_from "$BASE_OUT/outputs/9-task073_commonsenseqa_answer_generation/saved_weights/trans_input.pt"

rm -rf "$BASE_OUT/outputs/10-task1590_diplomacy_text_generation"/checkpoint*

${DEEPSPEED_BIN} --include localhost:${GPU_IDS} --master_port ${MASTER_PORT} "$ROOT_BASE/src/run_llama.py" \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path "$MODEL_PATH" \
   --data_dir "$DATA_DIR" \
   --task_order task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification \
   --task_config_dir "$ROOT_BASE/configs/gen_script_superni_order1_llama_configs/task748_glucose_reverse_cause_event_detection" \
   --output_dir "$BASE_OUT/outputs/11-task748_glucose_reverse_cause_event_detection" \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 8 \
   --learning_rate 5e-05 \
   --num_train_epochs 50 \
   --bf16 \
   --deepspeed "$ROOT_BASE/configs/ds_configs/stage2.config" \
   --run_name "$RUN_NAME" \
   --max_source_length 1024 \
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
   --metric_for_best_model eval_rougeL_for_task748_glucose_reverse_cause_event_detection \
   --eval_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --trans_hidden_dim 100 \
   --attn_lr 0 \
   --chunk 4 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995 \
   --previous_lora_path "$BASE_OUT/outputs/1-task1572_samsum_summary/saved_weights,$BASE_OUT/outputs/2-task363_sst2_polarity_classification/saved_weights,$BASE_OUT/outputs/3-task1290_xsum_summarization/saved_weights,$BASE_OUT/outputs/4-task181_outcome_extraction/saved_weights,$BASE_OUT/outputs/5-task002_quoref_answer_generation/saved_weights,$BASE_OUT/outputs/6-task1510_evalution_relation_extraction/saved_weights,$BASE_OUT/outputs/7-task639_multi_woz_user_utterance_generation/saved_weights,$BASE_OUT/outputs/8-task1729_personachat_generate_next/saved_weights,$BASE_OUT/outputs/9-task073_commonsenseqa_answer_generation/saved_weights,$BASE_OUT/outputs/10-task1590_diplomacy_text_generation/saved_weights" \
   --previous_prompt_key_path "$BASE_OUT/outputs/10-task1590_diplomacy_text_generation/saved_weights/prompts_keys_till_now.pt" \
   --load_checkpoint_from "$BASE_OUT/outputs/10-task1590_diplomacy_text_generation/saved_weights/trans_input.pt"

rm -rf "$BASE_OUT/outputs/11-task748_glucose_reverse_cause_event_detection"/checkpoint*

${DEEPSPEED_BIN} --include localhost:${GPU_IDS} --master_port ${MASTER_PORT} "$ROOT_BASE/src/run_llama.py" \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path "$MODEL_PATH" \
   --data_dir "$DATA_DIR" \
   --task_order task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification \
   --task_config_dir "$ROOT_BASE/configs/gen_script_superni_order1_llama_configs/task511_reddit_tifu_long_text_summarization" \
   --output_dir "$BASE_OUT/outputs/12-task511_reddit_tifu_long_text_summarization" \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 32 \
   --learning_rate 5e-05 \
   --num_train_epochs 50 \
   --bf16 \
   --deepspeed "$ROOT_BASE/configs/ds_configs/stage2.config" \
   --run_name "$RUN_NAME" \
   --max_source_length 1024 \
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
   --eval_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --trans_hidden_dim 100 \
   --attn_lr 0 \
   --chunk 4 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995 \
   --previous_lora_path "$BASE_OUT/outputs/1-task1572_samsum_summary/saved_weights,$BASE_OUT/outputs/2-task363_sst2_polarity_classification/saved_weights,$BASE_OUT/outputs/3-task1290_xsum_summarization/saved_weights,$BASE_OUT/outputs/4-task181_outcome_extraction/saved_weights,$BASE_OUT/outputs/5-task002_quoref_answer_generation/saved_weights,$BASE_OUT/outputs/6-task1510_evalution_relation_extraction/saved_weights,$BASE_OUT/outputs/7-task639_multi_woz_user_utterance_generation/saved_weights,$BASE_OUT/outputs/8-task1729_personachat_generate_next/saved_weights,$BASE_OUT/outputs/9-task073_commonsenseqa_answer_generation/saved_weights,$BASE_OUT/outputs/10-task1590_diplomacy_text_generation/saved_weights,$BASE_OUT/outputs/11-task748_glucose_reverse_cause_event_detection/saved_weights" \
   --previous_prompt_key_path "$BASE_OUT/outputs/11-task748_glucose_reverse_cause_event_detection/saved_weights/prompts_keys_till_now.pt" \
   --load_checkpoint_from "$BASE_OUT/outputs/11-task748_glucose_reverse_cause_event_detection/saved_weights/trans_input.pt"

rm -rf "$BASE_OUT/outputs/12-task511_reddit_tifu_long_text_summarization"/checkpoint*

${DEEPSPEED_BIN} --include localhost:${GPU_IDS} --master_port ${MASTER_PORT} "$ROOT_BASE/src/run_llama.py" \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path "$MODEL_PATH" \
   --data_dir "$DATA_DIR" \
   --task_order task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification \
   --task_config_dir "$ROOT_BASE/configs/gen_script_superni_order1_llama_configs/task591_sciq_answer_generation" \
   --output_dir "$BASE_OUT/outputs/13-task591_sciq_answer_generation" \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 4 \
   --learning_rate 5e-05 \
   --num_train_epochs 50 \
   --bf16 \
   --deepspeed "$ROOT_BASE/configs/ds_configs/stage2.config" \
   --run_name "$RUN_NAME" \
   --max_source_length 1024 \
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
   --eval_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --trans_hidden_dim 100 \
   --attn_lr 0 \
   --chunk 4 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995 \
   --previous_lora_path "$BASE_OUT/outputs/1-task1572_samsum_summary/saved_weights,$BASE_OUT/outputs/2-task363_sst2_polarity_classification/saved_weights,$BASE_OUT/outputs/3-task1290_xsum_summarization/saved_weights,$BASE_OUT/outputs/4-task181_outcome_extraction/saved_weights,$BASE_OUT/outputs/5-task002_quoref_answer_generation/saved_weights,$BASE_OUT/outputs/6-task1510_evalution_relation_extraction/saved_weights,$BASE_OUT/outputs/7-task639_multi_woz_user_utterance_generation/saved_weights,$BASE_OUT/outputs/8-task1729_personachat_generate_next/saved_weights,$BASE_OUT/outputs/9-task073_commonsenseqa_answer_generation/saved_weights,$BASE_OUT/outputs/10-task1590_diplomacy_text_generation/saved_weights,$BASE_OUT/outputs/11-task748_glucose_reverse_cause_event_detection/saved_weights,$BASE_OUT/outputs/12-task511_reddit_tifu_long_text_summarization/saved_weights" \
   --previous_prompt_key_path "$BASE_OUT/outputs/12-task511_reddit_tifu_long_text_summarization/saved_weights/prompts_keys_till_now.pt" \
   --load_checkpoint_from "$BASE_OUT/outputs/12-task511_reddit_tifu_long_text_summarization/saved_weights/trans_input.pt"

rm -rf "$BASE_OUT/outputs/13-task591_sciq_answer_generation"/checkpoint*

${DEEPSPEED_BIN} --include localhost:${GPU_IDS} --master_port ${MASTER_PORT} "$ROOT_BASE/src/run_llama.py" \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path "$MODEL_PATH" \
   --data_dir "$DATA_DIR" \
   --task_order task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification \
   --task_config_dir "$ROOT_BASE/configs/gen_script_superni_order1_llama_configs/task1687_sentiment140_classification" \
   --output_dir "$BASE_OUT/outputs/14-task1687_sentiment140_classification" \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 4 \
   --learning_rate 5e-05 \
   --num_train_epochs 50 \
   --bf16 \
   --deepspeed "$ROOT_BASE/configs/ds_configs/stage2.config" \
   --run_name "$RUN_NAME" \
   --max_source_length 1024 \
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
   --eval_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --trans_hidden_dim 100 \
   --attn_lr 0 \
   --chunk 4 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995 \
   --previous_lora_path "$BASE_OUT/outputs/1-task1572_samsum_summary/saved_weights,$BASE_OUT/outputs/2-task363_sst2_polarity_classification/saved_weights,$BASE_OUT/outputs/3-task1290_xsum_summarization/saved_weights,$BASE_OUT/outputs/4-task181_outcome_extraction/saved_weights,$BASE_OUT/outputs/5-task002_quoref_answer_generation/saved_weights,$BASE_OUT/outputs/6-task1510_evalution_relation_extraction/saved_weights,$BASE_OUT/outputs/7-task639_multi_woz_user_utterance_generation/saved_weights,$BASE_OUT/outputs/8-task1729_personachat_generate_next/saved_weights,$BASE_OUT/outputs/9-task073_commonsenseqa_answer_generation/saved_weights,$BASE_OUT/outputs/10-task1590_diplomacy_text_generation/saved_weights,$BASE_OUT/outputs/11-task748_glucose_reverse_cause_event_detection/saved_weights,$BASE_OUT/outputs/12-task511_reddit_tifu_long_text_summarization/saved_weights,$BASE_OUT/outputs/13-task591_sciq_answer_generation/saved_weights" \
   --previous_prompt_key_path "$BASE_OUT/outputs/13-task591_sciq_answer_generation/saved_weights/prompts_keys_till_now.pt" \
   --load_checkpoint_from "$BASE_OUT/outputs/13-task591_sciq_answer_generation/saved_weights/trans_input.pt"

rm -rf "$BASE_OUT/outputs/14-task1687_sentiment140_classification"/checkpoint*

${DEEPSPEED_BIN} --include localhost:${GPU_IDS} --master_port ${MASTER_PORT} "$ROOT_BASE/src/run_llama.py" \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path "$MODEL_PATH" \
   --data_dir "$DATA_DIR" \
   --task_order task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification \
   --task_config_dir "$ROOT_BASE/configs/gen_script_superni_order1_llama_configs/task875_emotion_classification" \
   --output_dir "$BASE_OUT/outputs/15-task875_emotion_classification" \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 8 \
   --learning_rate 5e-05 \
   --num_train_epochs 50 \
   --bf16 \
   --deepspeed "$ROOT_BASE/configs/ds_configs/stage2.config" \
   --run_name "$RUN_NAME" \
   --max_source_length 1024 \
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
   --eval_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --trans_hidden_dim 100 \
   --attn_lr 0 \
   --chunk 4 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995 \
   --previous_lora_path "$BASE_OUT/outputs/1-task1572_samsum_summary/saved_weights,$BASE_OUT/outputs/2-task363_sst2_polarity_classification/saved_weights,$BASE_OUT/outputs/3-task1290_xsum_summarization/saved_weights,$BASE_OUT/outputs/4-task181_outcome_extraction/saved_weights,$BASE_OUT/outputs/5-task002_quoref_answer_generation/saved_weights,$BASE_OUT/outputs/6-task1510_evalution_relation_extraction/saved_weights,$BASE_OUT/outputs/7-task639_multi_woz_user_utterance_generation/saved_weights,$BASE_OUT/outputs/8-task1729_personachat_generate_next/saved_weights,$BASE_OUT/outputs/9-task073_commonsenseqa_answer_generation/saved_weights,$BASE_OUT/outputs/10-task1590_diplomacy_text_generation/saved_weights,$BASE_OUT/outputs/11-task748_glucose_reverse_cause_event_detection/saved_weights,$BASE_OUT/outputs/12-task511_reddit_tifu_long_text_summarization/saved_weights,$BASE_OUT/outputs/13-task591_sciq_answer_generation/saved_weights,$BASE_OUT/outputs/14-task1687_sentiment140_classification/saved_weights" \
   --previous_prompt_key_path "$BASE_OUT/outputs/14-task1687_sentiment140_classification/saved_weights/prompts_keys_till_now.pt" \
   --load_checkpoint_from "$BASE_OUT/outputs/14-task1687_sentiment140_classification/saved_weights/trans_input.pt"

rm -rf "$BASE_OUT/outputs/15-task875_emotion_classification"/checkpoint*

"$PYTHON_BIN" "$ROOT_BASE/score.py" "$RUN_NAME" "$RUN_NAME" "$WORKSPACE_DIR/llama_epoch_ablation/logs_and_outputs"
