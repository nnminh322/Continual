#!/bin/bash
#SBATCH -J cl
#SBATCH -o cl-%j.out
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 20:00:00
#SBATCH --mem 128G
#SBATCH --gres=gpu:a100-sxm4-80gb:1

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# ── GPU auto-detection ─────────────────────────────────────────────────────
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)

if [ -z "$GPU_MEM" ]; then
    echo "[ERROR] No GPU detected. Exiting."
    exit 1
fi

if [ "$NUM_GPUS" -ge 2 ] && [ "$GPU_MEM" -ge 20000 ]; then
    # 2× 3090/4090 (24 GB+): run DDP on both GPUs
    DS_INCLUDE="localhost:0,1"
    echo "[GPU] 2× GPU detected (${GPU_MEM} MB each) → DDP on localhost:0,1"
elif [ "$GPU_MEM" -lt 20000 ]; then
    # P100/V100 16 GB: use the dedicated P100 script instead
    echo "[GPU] WARNING: 16 GB GPU detected. Use the _p100.sh script instead."
    echo "      This script requires ≥ 20 GB VRAM per GPU."
    exit 1
else
    # Single high-memory GPU (A100/H100 or single 3090)
    DS_INCLUDE="localhost:${1:-0}"
    echo "[GPU] Single GPU detected (${GPU_MEM} MB) → localhost:${1:-0}"
fi
# ──────────────────────────────────────────────────────────────────────────


deepspeed --include $DS_INCLUDE --master_port 49500 src/run_llama.py \
   --do_train \
   --gradient_checkpointing \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --data_dir CL_Benchmark \
   --task_order task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification \
   --task_config_dir configs/gen_script_superni_order1_llama_configs/task1572_samsum_summary \
   --output_dir logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/1-task1572_samsum_summary \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 8 \
   --learning_rate 5e-05 \
   --num_train_epochs 50 \
   --bf16 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name gen_script_superni_order1_llama_specroute \
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
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --chunk 4 \
   --model_name specroute \
   --cpi_gamma 0.5 \
   --oap_eta 0.5 \
   --oap_beta_min 0.3 \
   --oap_warmup 3 \
   --threshold 0.995 

rm -rf logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/1-task1572_samsum_summary/checkpoint*

sleep 5


deepspeed --include $DS_INCLUDE --master_port 49500 src/run_llama.py \
   --do_train \
   --gradient_checkpointing \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/1-task1572_samsum_summary/saved_weights \
   --previous_prompt_key_path logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/1-task1572_samsum_summary/saved_weights/spectral_signatures.pt \
   --data_dir CL_Benchmark \
   --task_order task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/gen_script_superni_order1_llama_configs/task363_sst2_polarity_classification \
   --output_dir logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/2-task363_sst2_polarity_classification \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 1 \
   --learning_rate 5e-05 \
   --num_train_epochs 50 \
   --bf16 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name gen_script_superni_order1_llama_specroute \
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
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --chunk 4 \
   --model_name specroute \
   --cpi_gamma 0.5 \
   --oap_eta 0.5 \
   --oap_beta_min 0.3 \
   --oap_warmup 3 \
   --threshold 0.995 

rm -rf logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/2-task363_sst2_polarity_classification/checkpoint*

sleep 5


deepspeed --include $DS_INCLUDE --master_port 49500 src/run_llama.py \
   --do_train \
   --gradient_checkpointing \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/1-task1572_samsum_summary/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/2-task363_sst2_polarity_classification/saved_weights \
   --previous_prompt_key_path logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/2-task363_sst2_polarity_classification/saved_weights/spectral_signatures.pt \
   --data_dir CL_Benchmark \
   --task_order task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/gen_script_superni_order1_llama_configs/task1290_xsum_summarization \
   --output_dir logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/3-task1290_xsum_summarization \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 16 \
   --learning_rate 5e-05 \
   --num_train_epochs 50 \
   --bf16 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name gen_script_superni_order1_llama_specroute \
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
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --chunk 4 \
   --model_name specroute \
   --cpi_gamma 0.5 \
   --oap_eta 0.5 \
   --oap_beta_min 0.3 \
   --oap_warmup 3 \
   --threshold 0.995 

rm -rf logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/3-task1290_xsum_summarization/checkpoint*

sleep 5


deepspeed --include $DS_INCLUDE --master_port 49500 src/run_llama.py \
   --do_train \
   --gradient_checkpointing \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/1-task1572_samsum_summary/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/2-task363_sst2_polarity_classification/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/3-task1290_xsum_summarization/saved_weights \
   --previous_prompt_key_path logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/3-task1290_xsum_summarization/saved_weights/spectral_signatures.pt \
   --data_dir CL_Benchmark \
   --task_order task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/gen_script_superni_order1_llama_configs/task181_outcome_extraction \
   --output_dir logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/4-task181_outcome_extraction \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 4 \
   --learning_rate 5e-05 \
   --num_train_epochs 50 \
   --bf16 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name gen_script_superni_order1_llama_specroute \
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
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --chunk 4 \
   --model_name specroute \
   --cpi_gamma 0.5 \
   --oap_eta 0.5 \
   --oap_beta_min 0.3 \
   --oap_warmup 3 \
   --threshold 0.995 

rm -rf logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/4-task181_outcome_extraction/checkpoint*

sleep 5


deepspeed --include $DS_INCLUDE --master_port 49500 src/run_llama.py \
   --do_train \
   --gradient_checkpointing \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/1-task1572_samsum_summary/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/2-task363_sst2_polarity_classification/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/3-task1290_xsum_summarization/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/4-task181_outcome_extraction/saved_weights \
   --previous_prompt_key_path logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/4-task181_outcome_extraction/saved_weights/spectral_signatures.pt \
   --data_dir CL_Benchmark \
   --task_order task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/gen_script_superni_order1_llama_configs/task002_quoref_answer_generation \
   --output_dir logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/5-task002_quoref_answer_generation \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 16 \
   --learning_rate 5e-05 \
   --num_train_epochs 50 \
   --bf16 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name gen_script_superni_order1_llama_specroute \
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
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --chunk 4 \
   --model_name specroute \
   --cpi_gamma 0.5 \
   --oap_eta 0.5 \
   --oap_beta_min 0.3 \
   --oap_warmup 3 \
   --threshold 0.995 

rm -rf logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/5-task002_quoref_answer_generation/checkpoint*

sleep 5


deepspeed --include $DS_INCLUDE --master_port 49500 src/run_llama.py \
   --do_train \
   --gradient_checkpointing \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/1-task1572_samsum_summary/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/2-task363_sst2_polarity_classification/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/3-task1290_xsum_summarization/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/4-task181_outcome_extraction/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/5-task002_quoref_answer_generation/saved_weights \
   --previous_prompt_key_path logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/5-task002_quoref_answer_generation/saved_weights/spectral_signatures.pt \
   --data_dir CL_Benchmark \
   --task_order task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/gen_script_superni_order1_llama_configs/task1510_evalution_relation_extraction \
   --output_dir logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/6-task1510_evalution_relation_extraction \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 1 \
   --learning_rate 5e-05 \
   --num_train_epochs 50 \
   --bf16 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name gen_script_superni_order1_llama_specroute \
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
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --chunk 4 \
   --model_name specroute \
   --cpi_gamma 0.5 \
   --oap_eta 0.5 \
   --oap_beta_min 0.3 \
   --oap_warmup 3 \
   --threshold 0.995 

rm -rf logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/6-task1510_evalution_relation_extraction/checkpoint*

sleep 5


deepspeed --include $DS_INCLUDE --master_port 49500 src/run_llama.py \
   --do_train \
   --gradient_checkpointing \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/1-task1572_samsum_summary/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/2-task363_sst2_polarity_classification/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/3-task1290_xsum_summarization/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/4-task181_outcome_extraction/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/5-task002_quoref_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/6-task1510_evalution_relation_extraction/saved_weights \
   --previous_prompt_key_path logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/6-task1510_evalution_relation_extraction/saved_weights/spectral_signatures.pt \
   --data_dir CL_Benchmark \
   --task_order task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/gen_script_superni_order1_llama_configs/task639_multi_woz_user_utterance_generation \
   --output_dir logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/7-task639_multi_woz_user_utterance_generation \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 1 \
   --learning_rate 5e-05 \
   --num_train_epochs 50 \
   --bf16 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name gen_script_superni_order1_llama_specroute \
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
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --chunk 4 \
   --model_name specroute \
   --cpi_gamma 0.5 \
   --oap_eta 0.5 \
   --oap_beta_min 0.3 \
   --oap_warmup 3 \
   --threshold 0.995 

rm -rf logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/7-task639_multi_woz_user_utterance_generation/checkpoint*

sleep 5


deepspeed --include $DS_INCLUDE --master_port 49500 src/run_llama.py \
   --do_train \
   --gradient_checkpointing \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/1-task1572_samsum_summary/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/2-task363_sst2_polarity_classification/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/3-task1290_xsum_summarization/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/4-task181_outcome_extraction/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/5-task002_quoref_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/6-task1510_evalution_relation_extraction/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/7-task639_multi_woz_user_utterance_generation/saved_weights \
   --previous_prompt_key_path logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/7-task639_multi_woz_user_utterance_generation/saved_weights/spectral_signatures.pt \
   --data_dir CL_Benchmark \
   --task_order task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/gen_script_superni_order1_llama_configs/task1729_personachat_generate_next \
   --output_dir logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/8-task1729_personachat_generate_next \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 8 \
   --learning_rate 5e-05 \
   --num_train_epochs 50 \
   --bf16 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name gen_script_superni_order1_llama_specroute \
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
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --chunk 4 \
   --model_name specroute \
   --cpi_gamma 0.5 \
   --oap_eta 0.5 \
   --oap_beta_min 0.3 \
   --oap_warmup 3 \
   --threshold 0.995 

rm -rf logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/8-task1729_personachat_generate_next/checkpoint*

sleep 5


deepspeed --include $DS_INCLUDE --master_port 49500 src/run_llama.py \
   --do_train \
   --gradient_checkpointing \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/1-task1572_samsum_summary/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/2-task363_sst2_polarity_classification/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/3-task1290_xsum_summarization/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/4-task181_outcome_extraction/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/5-task002_quoref_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/6-task1510_evalution_relation_extraction/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/7-task639_multi_woz_user_utterance_generation/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/8-task1729_personachat_generate_next/saved_weights \
   --previous_prompt_key_path logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/8-task1729_personachat_generate_next/saved_weights/spectral_signatures.pt \
   --data_dir CL_Benchmark \
   --task_order task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/gen_script_superni_order1_llama_configs/task073_commonsenseqa_answer_generation \
   --output_dir logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/9-task073_commonsenseqa_answer_generation \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 2 \
   --learning_rate 5e-05 \
   --num_train_epochs 50 \
   --bf16 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name gen_script_superni_order1_llama_specroute \
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
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --chunk 4 \
   --model_name specroute \
   --cpi_gamma 0.5 \
   --oap_eta 0.5 \
   --oap_beta_min 0.3 \
   --oap_warmup 3 \
   --threshold 0.995 

rm -rf logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/9-task073_commonsenseqa_answer_generation/checkpoint*

sleep 5


deepspeed --include $DS_INCLUDE --master_port 49500 src/run_llama.py \
   --do_train \
   --gradient_checkpointing \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/1-task1572_samsum_summary/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/2-task363_sst2_polarity_classification/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/3-task1290_xsum_summarization/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/4-task181_outcome_extraction/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/5-task002_quoref_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/6-task1510_evalution_relation_extraction/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/7-task639_multi_woz_user_utterance_generation/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/8-task1729_personachat_generate_next/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/9-task073_commonsenseqa_answer_generation/saved_weights \
   --previous_prompt_key_path logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/9-task073_commonsenseqa_answer_generation/saved_weights/spectral_signatures.pt \
   --data_dir CL_Benchmark \
   --task_order task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/gen_script_superni_order1_llama_configs/task1590_diplomacy_text_generation \
   --output_dir logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/10-task1590_diplomacy_text_generation \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 8 \
   --learning_rate 5e-05 \
   --num_train_epochs 50 \
   --bf16 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name gen_script_superni_order1_llama_specroute \
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
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --chunk 4 \
   --model_name specroute \
   --cpi_gamma 0.5 \
   --oap_eta 0.5 \
   --oap_beta_min 0.3 \
   --oap_warmup 3 \
   --threshold 0.995 

rm -rf logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/10-task1590_diplomacy_text_generation/checkpoint*

sleep 5


deepspeed --include $DS_INCLUDE --master_port 49500 src/run_llama.py \
   --do_train \
   --gradient_checkpointing \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/1-task1572_samsum_summary/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/2-task363_sst2_polarity_classification/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/3-task1290_xsum_summarization/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/4-task181_outcome_extraction/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/5-task002_quoref_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/6-task1510_evalution_relation_extraction/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/7-task639_multi_woz_user_utterance_generation/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/8-task1729_personachat_generate_next/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/9-task073_commonsenseqa_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/10-task1590_diplomacy_text_generation/saved_weights \
   --previous_prompt_key_path logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/10-task1590_diplomacy_text_generation/saved_weights/spectral_signatures.pt \
   --data_dir CL_Benchmark \
   --task_order task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/gen_script_superni_order1_llama_configs/task748_glucose_reverse_cause_event_detection \
   --output_dir logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/11-task748_glucose_reverse_cause_event_detection \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 4 \
   --learning_rate 5e-05 \
   --num_train_epochs 50 \
   --bf16 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name gen_script_superni_order1_llama_specroute \
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
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --chunk 4 \
   --model_name specroute \
   --cpi_gamma 0.5 \
   --oap_eta 0.5 \
   --oap_beta_min 0.3 \
   --oap_warmup 3 \
   --threshold 0.995 

rm -rf logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/11-task748_glucose_reverse_cause_event_detection/checkpoint*

sleep 5


deepspeed --include $DS_INCLUDE --master_port 49500 src/run_llama.py \
   --do_train \
   --gradient_checkpointing \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/1-task1572_samsum_summary/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/2-task363_sst2_polarity_classification/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/3-task1290_xsum_summarization/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/4-task181_outcome_extraction/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/5-task002_quoref_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/6-task1510_evalution_relation_extraction/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/7-task639_multi_woz_user_utterance_generation/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/8-task1729_personachat_generate_next/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/9-task073_commonsenseqa_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/10-task1590_diplomacy_text_generation/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/11-task748_glucose_reverse_cause_event_detection/saved_weights \
   --previous_prompt_key_path logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/11-task748_glucose_reverse_cause_event_detection/saved_weights/spectral_signatures.pt \
   --data_dir CL_Benchmark \
   --task_order task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/gen_script_superni_order1_llama_configs/task511_reddit_tifu_long_text_summarization \
   --output_dir logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/12-task511_reddit_tifu_long_text_summarization \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 16 \
   --learning_rate 5e-05 \
   --num_train_epochs 50 \
   --bf16 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name gen_script_superni_order1_llama_specroute \
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
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --chunk 4 \
   --model_name specroute \
   --cpi_gamma 0.5 \
   --oap_eta 0.5 \
   --oap_beta_min 0.3 \
   --oap_warmup 3 \
   --threshold 0.995 

rm -rf logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/12-task511_reddit_tifu_long_text_summarization/checkpoint*

sleep 5


deepspeed --include $DS_INCLUDE --master_port 49500 src/run_llama.py \
   --do_train \
   --gradient_checkpointing \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/1-task1572_samsum_summary/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/2-task363_sst2_polarity_classification/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/3-task1290_xsum_summarization/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/4-task181_outcome_extraction/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/5-task002_quoref_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/6-task1510_evalution_relation_extraction/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/7-task639_multi_woz_user_utterance_generation/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/8-task1729_personachat_generate_next/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/9-task073_commonsenseqa_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/10-task1590_diplomacy_text_generation/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/11-task748_glucose_reverse_cause_event_detection/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/12-task511_reddit_tifu_long_text_summarization/saved_weights \
   --previous_prompt_key_path logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/12-task511_reddit_tifu_long_text_summarization/saved_weights/spectral_signatures.pt \
   --data_dir CL_Benchmark \
   --task_order task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/gen_script_superni_order1_llama_configs/task591_sciq_answer_generation \
   --output_dir logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/13-task591_sciq_answer_generation \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 2 \
   --learning_rate 5e-05 \
   --num_train_epochs 50 \
   --bf16 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name gen_script_superni_order1_llama_specroute \
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
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --chunk 4 \
   --model_name specroute \
   --cpi_gamma 0.5 \
   --oap_eta 0.5 \
   --oap_beta_min 0.3 \
   --oap_warmup 3 \
   --threshold 0.995 

rm -rf logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/13-task591_sciq_answer_generation/checkpoint*

sleep 5


deepspeed --include $DS_INCLUDE --master_port 49500 src/run_llama.py \
   --do_train \
   --gradient_checkpointing \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/1-task1572_samsum_summary/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/2-task363_sst2_polarity_classification/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/3-task1290_xsum_summarization/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/4-task181_outcome_extraction/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/5-task002_quoref_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/6-task1510_evalution_relation_extraction/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/7-task639_multi_woz_user_utterance_generation/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/8-task1729_personachat_generate_next/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/9-task073_commonsenseqa_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/10-task1590_diplomacy_text_generation/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/11-task748_glucose_reverse_cause_event_detection/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/12-task511_reddit_tifu_long_text_summarization/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/13-task591_sciq_answer_generation/saved_weights \
   --previous_prompt_key_path logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/13-task591_sciq_answer_generation/saved_weights/spectral_signatures.pt \
   --data_dir CL_Benchmark \
   --task_order task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/gen_script_superni_order1_llama_configs/task1687_sentiment140_classification \
   --output_dir logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/14-task1687_sentiment140_classification \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 2 \
   --learning_rate 5e-05 \
   --num_train_epochs 50 \
   --bf16 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name gen_script_superni_order1_llama_specroute \
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
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --chunk 4 \
   --model_name specroute \
   --cpi_gamma 0.5 \
   --oap_eta 0.5 \
   --oap_beta_min 0.3 \
   --oap_warmup 3 \
   --threshold 0.995 

rm -rf logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/14-task1687_sentiment140_classification/checkpoint*

sleep 5


deepspeed --include $DS_INCLUDE --master_port 49500 src/run_llama.py \
   --do_train \
   --gradient_checkpointing \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/1-task1572_samsum_summary/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/2-task363_sst2_polarity_classification/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/3-task1290_xsum_summarization/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/4-task181_outcome_extraction/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/5-task002_quoref_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/6-task1510_evalution_relation_extraction/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/7-task639_multi_woz_user_utterance_generation/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/8-task1729_personachat_generate_next/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/9-task073_commonsenseqa_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/10-task1590_diplomacy_text_generation/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/11-task748_glucose_reverse_cause_event_detection/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/12-task511_reddit_tifu_long_text_summarization/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/13-task591_sciq_answer_generation/saved_weights,logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/14-task1687_sentiment140_classification/saved_weights \
   --previous_prompt_key_path logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/14-task1687_sentiment140_classification/saved_weights/spectral_signatures.pt \
   --data_dir CL_Benchmark \
   --task_order task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/gen_script_superni_order1_llama_configs/task875_emotion_classification \
   --output_dir logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/15-task875_emotion_classification \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 4 \
   --learning_rate 5e-05 \
   --num_train_epochs 50 \
   --bf16 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name gen_script_superni_order1_llama_specroute \
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
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --chunk 4 \
   --model_name specroute \
   --cpi_gamma 0.5 \
   --oap_eta 0.5 \
   --oap_beta_min 0.3 \
   --oap_warmup 3 \
   --threshold 0.995 

rm -rf logs_and_outputs/gen_script_superni_order1_llama_specroute/outputs/15-task875_emotion_classification/checkpoint*

sleep 5


python score.py gen_script_superni_order1_llama_specroute gen_script_superni_order1_llama_specroute
