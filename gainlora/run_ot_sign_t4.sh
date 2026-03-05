#!/bin/bash
###############################################################################
# OT-SIGN + GainLoRA (T5-Large) — Optimized for 2×T4 (16GB each)
# ================================================================
# Changes from original:
#   - batch_size: 16 → 8 (T4 memory constraint)
#   - fp16: enabled (T4 has good FP16 throughput)
#   - gradient_accumulation: 4 (compensate for smaller batch)
#   - OT-SIGN: use_ot_routing enabled from task 2 onward
#   - OT-SIGN: vMF signatures saved/loaded between tasks
#   - OT-SIGN: anti-drift + anti-invasion losses activated
#   - num_train_epochs: 50 (reduced for faster iteration on T4)
#   - ipdb/cupy guards already handled in code
###############################################################################

export CUDA_DEVICE_ORDER="PCI_BUS_ID"

# === Configuration ===
GPU_IDS="${1:-0,1}"         # Default: use GPU 0,1
MODEL_PATH="${2:-google/flan-t5-large}"  # T5-Large
RUN_NAME="ot_sign_t5_gainlora_t4"
TASK_ORDER="task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification"
OUTPUT_BASE="logs_and_outputs/${RUN_NAME}/outputs"

# T4-optimized hyperparameters
BATCH_SIZE=8
EVAL_BATCH_SIZE=4
GRAD_ACCUM=4
LR=0.0003
EPOCHS=50
MAX_SRC=512
MAX_TGT=50
LORA_R=4
LORA_ALPHA=32

# OT-SIGN hyperparameters
OT_EPSILON=0.05
OT_N_ITER=10
DEFAULT_KAPPA=10.0
LAMBDA_DRIFT=0.01
LAMBDA_INV=0.001
INVASION_THRESHOLD=2.3

# Common flags
COMMON_FLAGS="--predict_with_generate \
   --model_name_or_path ${MODEL_PATH} \
   --data_dir CL_Benchmark \
   --task_order ${TASK_ORDER} \
   --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
   --learning_rate ${LR} \
   --num_train_epochs ${EPOCHS} \
   --run_name ${RUN_NAME} \
   --max_source_length ${MAX_SRC} \
   --max_target_length ${MAX_TGT} \
   --generation_max_length ${MAX_TGT} \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --eval_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r ${LORA_R} \
   --lora_alpha ${LORA_ALPHA} \
   --lora_dropout 0.0 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995 \
   --fp16"

# OT-SIGN flags (used from task 2 onward)
OT_SIGN_FLAGS="--use_ot_routing True \
   --ot_epsilon ${OT_EPSILON} \
   --ot_n_iter ${OT_N_ITER} \
   --default_kappa ${DEFAULT_KAPPA} \
   --lambda_drift ${LAMBDA_DRIFT} \
   --lambda_inv ${LAMBDA_INV} \
   --invasion_threshold ${INVASION_THRESHOLD}"

echo "============================================="
echo "OT-SIGN + GainLoRA T5-Large on 2×T4"
echo "Run: ${RUN_NAME}"
echo "Model: ${MODEL_PATH}"
echo "Batch: ${BATCH_SIZE} × ${GRAD_ACCUM} grad_accum"
echo "FP16: enabled"
echo "OT-SIGN: epsilon=${OT_EPSILON}, kappa_def=${DEFAULT_KAPPA}"
echo "============================================="

# Create output directories
mkdir -p ${OUTPUT_BASE}
mkdir -p logs_and_outputs/${RUN_NAME}/ot_sign_logs

###############################################################################
# Task 1: task1572_samsum_summary (NO OT routing — first task, no signatures)
###############################################################################
echo ""
echo "===== Task 1/15: task1572_samsum_summary (baseline, no OT) ====="
CUDA_VISIBLE_DEVICES=${GPU_IDS} python src/run_t5.py \
   --do_train \
   --do_predict \
   ${COMMON_FLAGS} \
   --task_config_dir configs/gen_script_superni_order1_t5_configs/task1572_samsum_summary \
   --output_dir ${OUTPUT_BASE}/1-task1572_samsum_summary \
   --per_device_train_batch_size ${BATCH_SIZE} \
   --gradient_accumulation_steps ${GRAD_ACCUM} \
   --metric_for_best_model eval_rougeL \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   --use_ot_routing True \
   --ot_epsilon ${OT_EPSILON} \
   --ot_n_iter ${OT_N_ITER} \
   --default_kappa ${DEFAULT_KAPPA}

###############################################################################
# Task 2: task363_sst2 (OT routing with 1 signature from task 1)
###############################################################################
echo ""
echo "===== Task 2/15: task363_sst2_polarity_classification (OT-SIGN active) ====="
PREV_1="${OUTPUT_BASE}/1-task1572_samsum_summary/saved_weights"
CUDA_VISIBLE_DEVICES=${GPU_IDS} python src/run_t5.py \
   --do_train \
   --do_predict \
   ${COMMON_FLAGS} \
   --load_checkpoint_from ${PREV_1}/trans_input.pt \
   --previous_lora_path ${PREV_1} \
   --previous_prompt_key_path ${PREV_1}/prompts_keys_till_now.pt \
   --previous_vmf_signatures_path ${PREV_1}/vmf_signatures.pt \
   --gen_data_dir generated_data/lora_gen_superni_t5 \
   --task_config_dir configs/gen_script_superni_order1_t5_configs/task363_sst2_polarity_classification \
   --output_dir ${OUTPUT_BASE}/2-task363_sst2_polarity_classification \
   --per_device_train_batch_size ${BATCH_SIZE} \
   --gradient_accumulation_steps ${GRAD_ACCUM} \
   --metric_for_best_model eval_rougeL_for_task363_sst2_polarity_classification \
   --add_instruction_replay \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   ${OT_SIGN_FLAGS}

###############################################################################
# Task 3: task1290_xsum_summarization
###############################################################################
echo ""
echo "===== Task 3/15: task1290_xsum_summarization (OT-SIGN active) ====="
PREV_2="${OUTPUT_BASE}/2-task363_sst2_polarity_classification/saved_weights"
CUDA_VISIBLE_DEVICES=${GPU_IDS} python src/run_t5.py \
   --do_train \
   --do_predict \
   ${COMMON_FLAGS} \
   --load_checkpoint_from ${PREV_2}/trans_input.pt \
   --previous_lora_path ${PREV_1},${PREV_2} \
   --previous_prompt_key_path ${PREV_2}/prompts_keys_till_now.pt \
   --previous_vmf_signatures_path ${PREV_2}/vmf_signatures.pt \
   --gen_data_dir generated_data/lora_gen_superni_t5 \
   --task_config_dir configs/gen_script_superni_order1_t5_configs/task1290_xsum_summarization \
   --output_dir ${OUTPUT_BASE}/3-task1290_xsum_summarization \
   --per_device_train_batch_size ${BATCH_SIZE} \
   --gradient_accumulation_steps ${GRAD_ACCUM} \
   --metric_for_best_model eval_rougeL_for_task1290_xsum_summarization \
   --add_instruction_replay \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   ${OT_SIGN_FLAGS}

###############################################################################
# Task 4: task181_outcome_extraction
###############################################################################
echo ""
echo "===== Task 4/15: task181_outcome_extraction (OT-SIGN active) ====="
PREV_3="${OUTPUT_BASE}/3-task1290_xsum_summarization/saved_weights"
CUDA_VISIBLE_DEVICES=${GPU_IDS} python src/run_t5.py \
   --do_train \
   --do_predict \
   ${COMMON_FLAGS} \
   --load_checkpoint_from ${PREV_3}/trans_input.pt \
   --previous_lora_path ${PREV_1},${PREV_2},${PREV_3} \
   --previous_prompt_key_path ${PREV_3}/prompts_keys_till_now.pt \
   --previous_vmf_signatures_path ${PREV_3}/vmf_signatures.pt \
   --gen_data_dir generated_data/lora_gen_superni_t5 \
   --task_config_dir configs/gen_script_superni_order1_t5_configs/task181_outcome_extraction \
   --output_dir ${OUTPUT_BASE}/4-task181_outcome_extraction \
   --per_device_train_batch_size ${BATCH_SIZE} \
   --gradient_accumulation_steps ${GRAD_ACCUM} \
   --metric_for_best_model eval_rougeL_for_task181_outcome_extraction \
   --add_instruction_replay \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   ${OT_SIGN_FLAGS}

###############################################################################
# Task 5: task002_quoref_answer_generation
###############################################################################
echo ""
echo "===== Task 5/15: task002_quoref_answer_generation (OT-SIGN active) ====="
PREV_4="${OUTPUT_BASE}/4-task181_outcome_extraction/saved_weights"
CUDA_VISIBLE_DEVICES=${GPU_IDS} python src/run_t5.py \
   --do_train \
   --do_predict \
   ${COMMON_FLAGS} \
   --load_checkpoint_from ${PREV_4}/trans_input.pt \
   --previous_lora_path ${PREV_1},${PREV_2},${PREV_3},${PREV_4} \
   --previous_prompt_key_path ${PREV_4}/prompts_keys_till_now.pt \
   --previous_vmf_signatures_path ${PREV_4}/vmf_signatures.pt \
   --gen_data_dir generated_data/lora_gen_superni_t5 \
   --task_config_dir configs/gen_script_superni_order1_t5_configs/task002_quoref_answer_generation \
   --output_dir ${OUTPUT_BASE}/5-task002_quoref_answer_generation \
   --per_device_train_batch_size ${BATCH_SIZE} \
   --gradient_accumulation_steps ${GRAD_ACCUM} \
   --metric_for_best_model eval_rougeL_for_task002_quoref_answer_generation \
   --add_instruction_replay \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   ${OT_SIGN_FLAGS}

###############################################################################
# Task 6: task1510_evalution_relation_extraction
###############################################################################
echo ""
echo "===== Task 6/15: task1510_evalution_relation_extraction (OT-SIGN active) ====="
PREV_5="${OUTPUT_BASE}/5-task002_quoref_answer_generation/saved_weights"
CUDA_VISIBLE_DEVICES=${GPU_IDS} python src/run_t5.py \
   --do_train \
   --do_predict \
   ${COMMON_FLAGS} \
   --load_checkpoint_from ${PREV_5}/trans_input.pt \
   --previous_lora_path ${PREV_1},${PREV_2},${PREV_3},${PREV_4},${PREV_5} \
   --previous_prompt_key_path ${PREV_5}/prompts_keys_till_now.pt \
   --previous_vmf_signatures_path ${PREV_5}/vmf_signatures.pt \
   --gen_data_dir generated_data/lora_gen_superni_t5 \
   --task_config_dir configs/gen_script_superni_order1_t5_configs/task1510_evalution_relation_extraction \
   --output_dir ${OUTPUT_BASE}/6-task1510_evalution_relation_extraction \
   --per_device_train_batch_size ${BATCH_SIZE} \
   --gradient_accumulation_steps ${GRAD_ACCUM} \
   --metric_for_best_model eval_rougeL_for_task1510_evalution_relation_extraction \
   --add_instruction_replay \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   ${OT_SIGN_FLAGS}

###############################################################################
# Task 7: task639_multi_woz_user_utterance_generation
###############################################################################
echo ""
echo "===== Task 7/15: task639_multi_woz_user_utterance_generation (OT-SIGN active) ====="
PREV_6="${OUTPUT_BASE}/6-task1510_evalution_relation_extraction/saved_weights"
CUDA_VISIBLE_DEVICES=${GPU_IDS} python src/run_t5.py \
   --do_train \
   --do_predict \
   ${COMMON_FLAGS} \
   --load_checkpoint_from ${PREV_6}/trans_input.pt \
   --previous_lora_path ${PREV_1},${PREV_2},${PREV_3},${PREV_4},${PREV_5},${PREV_6} \
   --previous_prompt_key_path ${PREV_6}/prompts_keys_till_now.pt \
   --previous_vmf_signatures_path ${PREV_6}/vmf_signatures.pt \
   --gen_data_dir generated_data/lora_gen_superni_t5 \
   --task_config_dir configs/gen_script_superni_order1_t5_configs/task639_multi_woz_user_utterance_generation \
   --output_dir ${OUTPUT_BASE}/7-task639_multi_woz_user_utterance_generation \
   --per_device_train_batch_size ${BATCH_SIZE} \
   --gradient_accumulation_steps ${GRAD_ACCUM} \
   --metric_for_best_model eval_rougeL_for_task639_multi_woz_user_utterance_generation \
   --add_instruction_replay \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   ${OT_SIGN_FLAGS}

###############################################################################
# Task 8: task1729_personachat_generate_next
###############################################################################
echo ""
echo "===== Task 8/15: task1729_personachat_generate_next (OT-SIGN active) ====="
PREV_7="${OUTPUT_BASE}/7-task639_multi_woz_user_utterance_generation/saved_weights"
CUDA_VISIBLE_DEVICES=${GPU_IDS} python src/run_t5.py \
   --do_train \
   --do_predict \
   ${COMMON_FLAGS} \
   --load_checkpoint_from ${PREV_7}/trans_input.pt \
   --previous_lora_path ${PREV_1},${PREV_2},${PREV_3},${PREV_4},${PREV_5},${PREV_6},${PREV_7} \
   --previous_prompt_key_path ${PREV_7}/prompts_keys_till_now.pt \
   --previous_vmf_signatures_path ${PREV_7}/vmf_signatures.pt \
   --gen_data_dir generated_data/lora_gen_superni_t5 \
   --task_config_dir configs/gen_script_superni_order1_t5_configs/task1729_personachat_generate_next \
   --output_dir ${OUTPUT_BASE}/8-task1729_personachat_generate_next \
   --per_device_train_batch_size ${BATCH_SIZE} \
   --gradient_accumulation_steps ${GRAD_ACCUM} \
   --metric_for_best_model eval_rougeL_for_task1729_personachat_generate_next \
   --add_instruction_replay \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   ${OT_SIGN_FLAGS}

###############################################################################
# Task 9: task073_commonsenseqa_answer_generation
###############################################################################
echo ""
echo "===== Task 9/15: task073_commonsenseqa_answer_generation (OT-SIGN active) ====="
PREV_8="${OUTPUT_BASE}/8-task1729_personachat_generate_next/saved_weights"
CUDA_VISIBLE_DEVICES=${GPU_IDS} python src/run_t5.py \
   --do_train \
   --do_predict \
   ${COMMON_FLAGS} \
   --load_checkpoint_from ${PREV_8}/trans_input.pt \
   --previous_lora_path ${PREV_1},${PREV_2},${PREV_3},${PREV_4},${PREV_5},${PREV_6},${PREV_7},${PREV_8} \
   --previous_prompt_key_path ${PREV_8}/prompts_keys_till_now.pt \
   --previous_vmf_signatures_path ${PREV_8}/vmf_signatures.pt \
   --gen_data_dir generated_data/lora_gen_superni_t5 \
   --task_config_dir configs/gen_script_superni_order1_t5_configs/task073_commonsenseqa_answer_generation \
   --output_dir ${OUTPUT_BASE}/9-task073_commonsenseqa_answer_generation \
   --per_device_train_batch_size ${BATCH_SIZE} \
   --gradient_accumulation_steps ${GRAD_ACCUM} \
   --metric_for_best_model eval_rougeL_for_task073_commonsenseqa_answer_generation \
   --add_instruction_replay \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   ${OT_SIGN_FLAGS}

###############################################################################
# Task 10: task1590_diplomacy_text_generation
###############################################################################
echo ""
echo "===== Task 10/15: task1590_diplomacy_text_generation (OT-SIGN active) ====="
PREV_9="${OUTPUT_BASE}/9-task073_commonsenseqa_answer_generation/saved_weights"
CUDA_VISIBLE_DEVICES=${GPU_IDS} python src/run_t5.py \
   --do_train \
   --do_predict \
   ${COMMON_FLAGS} \
   --load_checkpoint_from ${PREV_9}/trans_input.pt \
   --previous_lora_path ${PREV_1},${PREV_2},${PREV_3},${PREV_4},${PREV_5},${PREV_6},${PREV_7},${PREV_8},${PREV_9} \
   --previous_prompt_key_path ${PREV_9}/prompts_keys_till_now.pt \
   --previous_vmf_signatures_path ${PREV_9}/vmf_signatures.pt \
   --gen_data_dir generated_data/lora_gen_superni_t5 \
   --task_config_dir configs/gen_script_superni_order1_t5_configs/task1590_diplomacy_text_generation \
   --output_dir ${OUTPUT_BASE}/10-task1590_diplomacy_text_generation \
   --per_device_train_batch_size ${BATCH_SIZE} \
   --gradient_accumulation_steps ${GRAD_ACCUM} \
   --metric_for_best_model eval_rougeL_for_task1590_diplomacy_text_generation \
   --add_instruction_replay \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   ${OT_SIGN_FLAGS}

###############################################################################
# Task 11: task748_glucose_reverse_cause_event_detection
###############################################################################
echo ""
echo "===== Task 11/15: task748_glucose_reverse_cause_event_detection (OT-SIGN active) ====="
PREV_10="${OUTPUT_BASE}/10-task1590_diplomacy_text_generation/saved_weights"
CUDA_VISIBLE_DEVICES=${GPU_IDS} python src/run_t5.py \
   --do_train \
   --do_predict \
   ${COMMON_FLAGS} \
   --load_checkpoint_from ${PREV_10}/trans_input.pt \
   --previous_lora_path ${PREV_1},${PREV_2},${PREV_3},${PREV_4},${PREV_5},${PREV_6},${PREV_7},${PREV_8},${PREV_9},${PREV_10} \
   --previous_prompt_key_path ${PREV_10}/prompts_keys_till_now.pt \
   --previous_vmf_signatures_path ${PREV_10}/vmf_signatures.pt \
   --gen_data_dir generated_data/lora_gen_superni_t5 \
   --task_config_dir configs/gen_script_superni_order1_t5_configs/task748_glucose_reverse_cause_event_detection \
   --output_dir ${OUTPUT_BASE}/11-task748_glucose_reverse_cause_event_detection \
   --per_device_train_batch_size ${BATCH_SIZE} \
   --gradient_accumulation_steps ${GRAD_ACCUM} \
   --metric_for_best_model eval_rougeL_for_task748_glucose_reverse_cause_event_detection \
   --add_instruction_replay \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   ${OT_SIGN_FLAGS}

###############################################################################
# Task 12: task511_reddit_tifu_long_text_summarization
###############################################################################
echo ""
echo "===== Task 12/15: task511_reddit_tifu_long_text_summarization (OT-SIGN active) ====="
PREV_11="${OUTPUT_BASE}/11-task748_glucose_reverse_cause_event_detection/saved_weights"
CUDA_VISIBLE_DEVICES=${GPU_IDS} python src/run_t5.py \
   --do_train \
   --do_predict \
   ${COMMON_FLAGS} \
   --load_checkpoint_from ${PREV_11}/trans_input.pt \
   --previous_lora_path ${PREV_1},${PREV_2},${PREV_3},${PREV_4},${PREV_5},${PREV_6},${PREV_7},${PREV_8},${PREV_9},${PREV_10},${PREV_11} \
   --previous_prompt_key_path ${PREV_11}/prompts_keys_till_now.pt \
   --previous_vmf_signatures_path ${PREV_11}/vmf_signatures.pt \
   --gen_data_dir generated_data/lora_gen_superni_t5 \
   --task_config_dir configs/gen_script_superni_order1_t5_configs/task511_reddit_tifu_long_text_summarization \
   --output_dir ${OUTPUT_BASE}/12-task511_reddit_tifu_long_text_summarization \
   --per_device_train_batch_size ${BATCH_SIZE} \
   --gradient_accumulation_steps ${GRAD_ACCUM} \
   --metric_for_best_model eval_rougeL_for_task511_reddit_tifu_long_text_summarization \
   --add_instruction_replay \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   ${OT_SIGN_FLAGS}

###############################################################################
# Task 13: task591_sciq_answer_generation
###############################################################################
echo ""
echo "===== Task 13/15: task591_sciq_answer_generation (OT-SIGN active) ====="
PREV_12="${OUTPUT_BASE}/12-task511_reddit_tifu_long_text_summarization/saved_weights"
CUDA_VISIBLE_DEVICES=${GPU_IDS} python src/run_t5.py \
   --do_train \
   --do_predict \
   ${COMMON_FLAGS} \
   --load_checkpoint_from ${PREV_12}/trans_input.pt \
   --previous_lora_path ${PREV_1},${PREV_2},${PREV_3},${PREV_4},${PREV_5},${PREV_6},${PREV_7},${PREV_8},${PREV_9},${PREV_10},${PREV_11},${PREV_12} \
   --previous_prompt_key_path ${PREV_12}/prompts_keys_till_now.pt \
   --previous_vmf_signatures_path ${PREV_12}/vmf_signatures.pt \
   --gen_data_dir generated_data/lora_gen_superni_t5 \
   --task_config_dir configs/gen_script_superni_order1_t5_configs/task591_sciq_answer_generation \
   --output_dir ${OUTPUT_BASE}/13-task591_sciq_answer_generation \
   --per_device_train_batch_size ${BATCH_SIZE} \
   --gradient_accumulation_steps ${GRAD_ACCUM} \
   --metric_for_best_model eval_rougeL_for_task591_sciq_answer_generation \
   --add_instruction_replay \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   ${OT_SIGN_FLAGS}

###############################################################################
# Task 14: task1687_sentiment140_classification
###############################################################################
echo ""
echo "===== Task 14/15: task1687_sentiment140_classification (OT-SIGN active) ====="
PREV_13="${OUTPUT_BASE}/13-task591_sciq_answer_generation/saved_weights"
CUDA_VISIBLE_DEVICES=${GPU_IDS} python src/run_t5.py \
   --do_train \
   --do_predict \
   ${COMMON_FLAGS} \
   --load_checkpoint_from ${PREV_13}/trans_input.pt \
   --previous_lora_path ${PREV_1},${PREV_2},${PREV_3},${PREV_4},${PREV_5},${PREV_6},${PREV_7},${PREV_8},${PREV_9},${PREV_10},${PREV_11},${PREV_12},${PREV_13} \
   --previous_prompt_key_path ${PREV_13}/prompts_keys_till_now.pt \
   --previous_vmf_signatures_path ${PREV_13}/vmf_signatures.pt \
   --gen_data_dir generated_data/lora_gen_superni_t5 \
   --task_config_dir configs/gen_script_superni_order1_t5_configs/task1687_sentiment140_classification \
   --output_dir ${OUTPUT_BASE}/14-task1687_sentiment140_classification \
   --per_device_train_batch_size ${BATCH_SIZE} \
   --gradient_accumulation_steps ${GRAD_ACCUM} \
   --metric_for_best_model eval_rougeL_for_task1687_sentiment140_classification \
   --add_instruction_replay \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   ${OT_SIGN_FLAGS}

###############################################################################
# Task 15: task875_emotion_classification
###############################################################################
echo ""
echo "===== Task 15/15: task875_emotion_classification (OT-SIGN active) ====="
PREV_14="${OUTPUT_BASE}/14-task1687_sentiment140_classification/saved_weights"
CUDA_VISIBLE_DEVICES=${GPU_IDS} python src/run_t5.py \
   --do_train \
   --do_predict \
   ${COMMON_FLAGS} \
   --load_checkpoint_from ${PREV_14}/trans_input.pt \
   --previous_lora_path ${PREV_1},${PREV_2},${PREV_3},${PREV_4},${PREV_5},${PREV_6},${PREV_7},${PREV_8},${PREV_9},${PREV_10},${PREV_11},${PREV_12},${PREV_13},${PREV_14} \
   --previous_prompt_key_path ${PREV_14}/prompts_keys_till_now.pt \
   --previous_vmf_signatures_path ${PREV_14}/vmf_signatures.pt \
   --gen_data_dir generated_data/lora_gen_superni_t5 \
   --task_config_dir configs/gen_script_superni_order1_t5_configs/task875_emotion_classification \
   --output_dir ${OUTPUT_BASE}/15-task875_emotion_classification \
   --per_device_train_batch_size ${BATCH_SIZE} \
   --gradient_accumulation_steps ${GRAD_ACCUM} \
   --metric_for_best_model eval_rougeL_for_task875_emotion_classification \
   --add_instruction_replay \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   ${OT_SIGN_FLAGS}

echo ""
echo "============================================="
echo "OT-SIGN Training Complete!"
echo "Logs: logs_and_outputs/${RUN_NAME}/ot_sign_logs/"
echo "============================================="
