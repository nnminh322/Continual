#!/bin/bash
#SBATCH -J srt-llama
#SBATCH -o srt-llama-%j.out
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 80:00:00
#SBATCH --mem 256G
#SBATCH --gres=gpu:a100-sxm4-80gb:1

export CUDA_DEVICE_ORDER="PCI_BUS_ID"

# ============================================================
# Parse named arguments: --sgwi, --dual_fisher, --lambda_emb, --gpu
# Usage: bash script.sh [MODEL_PATH] [--gpu 5090|a100|h100] [--sgwi true/false] [--dual_fisher true/false] [--lambda_emb 0.01]
#
# Defaults:  SRT + SGWI (no Dual Fisher)
# Examples:
#   bash script.sh                                              # SRT + SGWI only (auto-detect GPU)
#   bash script.sh --gpu 5090                                   # RTX 5090 32GB → fp16 + grad_ckpt, effective BS=16
#   bash script.sh --gpu h100                                    # H100 80GB → bf16
#   bash script.sh --gpu a100                                    # A100 80GB → bf16
#   bash script.sh --dual_fisher True                            # SRT + SGWI + Dual Fisher
#   bash script.sh --dual_fisher True --lambda_emb 0.05
#   bash script.sh --sgwi False                                  # SRT only (full random LoRA init)
# ============================================================
SGWI_FLAG="True"          # default: SGWI warm-init enabled
DUAL_FISHER_FLAG="False"  # default: Dual Fisher disabled
LAMBDA_EMB=""             # default: auto (0.01 when dual_fisher=True)
GPU_FORCE=""              # override GPU mode (e.g. 5090, a100, h100)

POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --sgwi)        SGWI_FLAG="$2"; shift 2 ;;
        --dual_fisher) DUAL_FISHER_FLAG="$2"; shift 2 ;;
        --lambda_emb)  LAMBDA_EMB="$2"; shift 2 ;;
        --gpu)         GPU_FORCE="$2"; shift 2 ;;
        *)             POSITIONAL+=("$1"); shift ;;
    esac
done
set -- "${POSITIONAL[@]}"

MODEL_PATH="${1:-meta-llama/Meta-Llama-3-8B}"

# Build LAMBDA_ARG
LAMBDA_ARG=""
if [ -n "$LAMBDA_EMB" ]; then
    LAMBDA_ARG="--lambda_emb $LAMBDA_EMB"
fi

echo "============================================================"
echo "[CONFIG] sgwi=$SGWI_FLAG, dual_fisher=$DUAL_FISHER_FLAG, lambda_emb=${LAMBDA_EMB:-auto}"
echo "============================================================"

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
: ${GPU_MEM:=16000}; : ${NUM_GPUS:=1}

# ── Determine GPU capability ──────────────────────────────────────────────────
# Check if bf16 is supported (Ampere+ A100, H100, RTX 3090/4090, or newer)
BF16_SUPPORTED=0
COMPUTE_CAPABILITY=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 | tr '.' ' ')
if [ -n "$COMPUTE_CAPABILITY" ]; then
    MAJOR=$(echo $COMPUTE_CAPABILITY | awk '{print $1}')
    if [ "$MAJOR" -ge 8 ]; then
        BF16_SUPPORTED=1
    fi
fi

# Ampere+ hardware ≠ PyTorch bf16 support — bf16 requires CUDA 12.1+ with Ampere.
# To be safe, only use bf16 on 80GB+ A100/H100 cards (where CUDA version is reliable).
# All other tiers default to fp16 with gradient checkpointing.
if [ -n "$GPU_FORCE" ]; then
    # Explicit GPU type override: --gpu 5090 | --gpu a100 | --gpu h100
    case "$GPU_FORCE" in
        5090|rtx5090)
            IS_T4=0; GPU_MODE="5090_fp16"
            BSZ=1; GA=16; EVAL_BSZ=2  # effective BS=16, targets ~20-22GB VRAM
            FP16_FLAG="--gradient_checkpointing --fp16"
            ;;
        h100)
            IS_T4=0; GPU_MODE="h100_bf16"
            BSZ=8; GA=2; EVAL_BSZ=16
            FP16_FLAG="--bf16"
            ;;
        a100|*)
            # Treat as 80GB A100
            if [ "$BF16_SUPPORTED" -eq 1 ]; then
                IS_T4=0; GPU_MODE="a100_bf16"
                BSZ=8; GA=2; EVAL_BSZ=16
                FP16_FLAG="--bf16"
            else
                IS_T4=0; GPU_MODE="a100_fp16"
                BSZ=8; GA=2; EVAL_BSZ=16
                FP16_FLAG="--gradient_checkpointing --fp16"
            fi
            ;;
    esac
    GPU_IDS="${2:-0}"
elif [ "$GPU_MEM" -lt 50000 ]; then
    IS_T4=0; GPU_MODE="mid_fp16"; GPU_IDS="${2:-0}"
    BSZ=4; GA=4; EVAL_BSZ=8
    FP16_FLAG="--gradient_checkpointing --fp16"
elif [ "$GPU_MEM" -ge 50000 ] && [ "$BF16_SUPPORTED" -eq 1 ]; then
    IS_T4=0; GPU_MODE="a100_bf16"; GPU_IDS="${2:-0}"
    BSZ=8; GA=2; EVAL_BSZ=16
    FP16_FLAG="--bf16"
else
    IS_T4=0; GPU_MODE="a100_fp16"; GPU_IDS="${2:-0}"
    BSZ=8; GA=2; EVAL_BSZ=16
    FP16_FLAG="--gradient_checkpointing --fp16"
fi

echo "[GPU] $GPU_MODE ($GPU_MEM MB) | CUDA_VISIBLE_DEVICES=$GPU_IDS | $MODEL_PATH"
echo "============================================================"

RUN_NAME="superni_order1_llama_srt"
TASK_ORDER="task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification"
BASE_OUT="logs_and_outputs/$RUN_NAME"
# SRT flags: same pattern as T5 order 3/4
SRT_FLAGS="--use_srt_router --srt_metric_mode hard --srt_max_emb_samples 500 --srt_skip_forward --sgwi $SGWI_FLAG --dual_fisher $DUAL_FISHER_FLAG $LAMBDA_ARG"
echo "[SRT] use_srt_router=True, sgwi=$SGWI_FLAG, dual_fisher=$DUAL_FISHER_FLAG, srt_skip_forward=True"

# ── TASK 1: task1572_samsum_summary ──────────────────────────────────────────
OUTPUT_DIR="$BASE_OUT/outputs/1-task1572_samsum_summary"
if [ -f "$OUTPUT_DIR/done" ]; then
    echo "[SKIP] Task 1-task1572_samsum_summary already complete"
else
    CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_llama.py \
       --do_train \
       --do_predict \
       --predict_with_generate \
       --model_name_or_path $MODEL_PATH \
       --data_dir CL_Benchmark \
       --task_order $TASK_ORDER \
       --task_config_dir configs/gen_script_superni_order1_llama_configs/task1572_samsum_summary \
       --output_dir $OUTPUT_DIR \
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
       --overwrite_cache \
       --lr_scheduler_type constant \
       --warmup_steps 0 \
       --logging_strategy steps \
       --logging_steps 10 \
       --metric_for_best_model eval_rougeL \
       --eval_strategy steps \
       --save_strategy steps \
       --save_total_limit 1 \
       --lora_r 8 \
       --lora_alpha 32 \
       --lora_dropout 0.0 \
          --data_replay_freq -1 \
       --replay_after_n_epoch 0 \
       --kl_ratio 0.5 \
       --attn_temperature 1 \
       --mlp_hidden_dim 100 \
       --model_name gainlora \
       --threshold 0.995 \
       --transthreshold 0.995 \
       $FP16_FLAG \
       $SRT_FLAGS
    rm -rf $OUTPUT_DIR/checkpoint*
    touch "$OUTPUT_DIR/done"
fi
sleep 5

# ── TASK 2: task363_sst2_polarity_classification ─────────────────────────────
OUTPUT_DIR="$BASE_OUT/outputs/2-task363_sst2_polarity_classification"
if [ -f "$OUTPUT_DIR/done" ]; then
    echo "[SKIP] Task 2-task363_sst2_polarity_classification already complete"
else
    CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_llama.py \
       --do_train \
       --do_predict \
       --predict_with_generate \
       --model_name_or_path $MODEL_PATH \
       --data_dir CL_Benchmark \
       --load_checkpoint_from $BASE_OUT/outputs/1-task1572_samsum_summary/saved_weights/trans_input.pt \
       --previous_lora_path $BASE_OUT/outputs/1-task1572_samsum_summary/saved_weights \
       --previous_prompt_key_path $BASE_OUT/outputs/1-task1572_samsum_summary/saved_weights/prompts_keys_till_now.pt \
       --task_order $TASK_ORDER \
       --gen_data_dir generated_data/lora_gen_superni_llama \
       --task_config_dir configs/gen_script_superni_order1_llama_configs/task363_sst2_polarity_classification \
       --output_dir $OUTPUT_DIR \
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
       --overwrite_cache \
       --lr_scheduler_type constant \
       --warmup_steps 0 \
       --logging_strategy steps \
       --logging_steps 10 \
       --metric_for_best_model eval_exact_match \
       --eval_strategy steps \
       --save_strategy steps \
       --save_total_limit 1 \
       --lora_r 8 \
       --lora_alpha 32 \
       --lora_dropout 0.0 \
          --data_replay_freq -1 \
       --kl_ratio 0.5 \
       --attn_temperature 1 \
       --mlp_hidden_dim 100 \
       --model_name gainlora \
       --threshold 0.995 \
       --transthreshold 0.995 \
       $FP16_FLAG \
       $SRT_FLAGS \
       --srt_load_path $BASE_OUT/outputs/1-task1572_samsum_summary/saved_weights
    rm -rf $OUTPUT_DIR/checkpoint*
    touch "$OUTPUT_DIR/done"
fi
sleep 5

# ── TASK 3: task1290_xsum_summarization ──────────────────────────────────────
OUTPUT_DIR="$BASE_OUT/outputs/3-task1290_xsum_summarization"
if [ -f "$OUTPUT_DIR/done" ]; then
    echo "[SKIP] Task 3-task1290_xsum_summarization already complete"
else
    CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_llama.py \
       --do_train \
       --do_predict \
       --predict_with_generate \
       --model_name_or_path $MODEL_PATH \
       --data_dir CL_Benchmark \
       --load_checkpoint_from $BASE_OUT/outputs/2-task363_sst2_polarity_classification/saved_weights/trans_input.pt \
       --previous_lora_path $BASE_OUT/outputs/1-task1572_samsum_summary/saved_weights,$BASE_OUT/outputs/2-task363_sst2_polarity_classification/saved_weights \
       --previous_prompt_key_path $BASE_OUT/outputs/2-task363_sst2_polarity_classification/saved_weights/prompts_keys_till_now.pt \
       --task_order $TASK_ORDER \
       --gen_data_dir generated_data/lora_gen_superni_llama \
       --task_config_dir configs/gen_script_superni_order1_llama_configs/task1290_xsum_summarization \
       --output_dir $OUTPUT_DIR \
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
       --overwrite_cache \
       --lr_scheduler_type constant \
       --warmup_steps 0 \
       --logging_strategy steps \
       --logging_steps 10 \
       --metric_for_best_model eval_rougeL \
       --eval_strategy steps \
       --save_strategy steps \
       --save_total_limit 1 \
       --lora_r 8 \
       --lora_alpha 32 \
       --lora_dropout 0.0 \
          --data_replay_freq -1 \
       --kl_ratio 0.5 \
       --attn_temperature 1 \
       --mlp_hidden_dim 100 \
       --model_name gainlora \
       --threshold 0.995 \
       --transthreshold 0.995 \
       $FP16_FLAG \
       $SRT_FLAGS \
       --srt_load_path $BASE_OUT/outputs/2-task363_sst2_polarity_classification/saved_weights
    rm -rf $OUTPUT_DIR/checkpoint*
    touch "$OUTPUT_DIR/done"
fi
sleep 5

# ── TASK 4: task181_outcome_extraction ───────────────────────────────────────
OUTPUT_DIR="$BASE_OUT/outputs/4-task181_outcome_extraction"
if [ -f "$OUTPUT_DIR/done" ]; then
    echo "[SKIP] Task 4-task181_outcome_extraction already complete"
else
    CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_llama.py \
       --do_train \
       --do_predict \
       --predict_with_generate \
       --model_name_or_path $MODEL_PATH \
       --data_dir CL_Benchmark \
       --load_checkpoint_from $BASE_OUT/outputs/3-task1290_xsum_summarization/saved_weights/trans_input.pt \
       --previous_lora_path $BASE_OUT/outputs/1-task1572_samsum_summary/saved_weights,$BASE_OUT/outputs/2-task363_sst2_polarity_classification/saved_weights,$BASE_OUT/outputs/3-task1290_xsum_summarization/saved_weights \
       --previous_prompt_key_path $BASE_OUT/outputs/3-task1290_xsum_summarization/saved_weights/prompts_keys_till_now.pt \
       --task_order $TASK_ORDER \
       --gen_data_dir generated_data/lora_gen_superni_llama \
       --task_config_dir configs/gen_script_superni_order1_llama_configs/task181_outcome_extraction \
       --output_dir $OUTPUT_DIR \
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
       --overwrite_cache \
       --lr_scheduler_type constant \
       --warmup_steps 0 \
       --logging_strategy steps \
       --logging_steps 10 \
       --metric_for_best_model eval_rougeL \
       --eval_strategy steps \
       --save_strategy steps \
       --save_total_limit 1 \
       --lora_r 8 \
       --lora_alpha 32 \
       --lora_dropout 0.0 \
          --data_replay_freq -1 \
       --kl_ratio 0.5 \
       --attn_temperature 1 \
       --mlp_hidden_dim 100 \
       --model_name gainlora \
       --threshold 0.995 \
       --transthreshold 0.995 \
       $FP16_FLAG \
       $SRT_FLAGS \
       --srt_load_path $BASE_OUT/outputs/3-task1290_xsum_summarization/saved_weights
    rm -rf $OUTPUT_DIR/checkpoint*
    touch "$OUTPUT_DIR/done"
fi
sleep 5

# ── TASK 5: task002_quoref_answer_generation ──────────────────────────────────
OUTPUT_DIR="$BASE_OUT/outputs/5-task002_quoref_answer_generation"
if [ -f "$OUTPUT_DIR/done" ]; then
    echo "[SKIP] Task 5-task002_quoref_answer_generation already complete"
else
    CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_llama.py \
       --do_train \
       --do_predict \
       --predict_with_generate \
       --model_name_or_path $MODEL_PATH \
       --data_dir CL_Benchmark \
       --load_checkpoint_from $BASE_OUT/outputs/4-task181_outcome_extraction/saved_weights/trans_input.pt \
       --previous_lora_path $BASE_OUT/outputs/1-task1572_samsum_summary/saved_weights,$BASE_OUT/outputs/2-task363_sst2_polarity_classification/saved_weights,$BASE_OUT/outputs/3-task1290_xsum_summarization/saved_weights,$BASE_OUT/outputs/4-task181_outcome_extraction/saved_weights \
       --previous_prompt_key_path $BASE_OUT/outputs/4-task181_outcome_extraction/saved_weights/prompts_keys_till_now.pt \
       --task_order $TASK_ORDER \
       --gen_data_dir generated_data/lora_gen_superni_llama \
       --task_config_dir configs/gen_script_superni_order1_llama_configs/task002_quoref_answer_generation \
       --output_dir $OUTPUT_DIR \
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
       --overwrite_cache \
       --lr_scheduler_type constant \
       --warmup_steps 0 \
       --logging_strategy steps \
       --logging_steps 10 \
       --metric_for_best_model eval_rougeL \
       --eval_strategy steps \
       --save_strategy steps \
       --save_total_limit 1 \
       --lora_r 8 \
       --lora_alpha 32 \
       --lora_dropout 0.0 \
          --data_replay_freq -1 \
       --kl_ratio 0.5 \
       --attn_temperature 1 \
       --mlp_hidden_dim 100 \
       --model_name gainlora \
       --threshold 0.995 \
       --transthreshold 0.995 \
       $FP16_FLAG \
       $SRT_FLAGS \
       --srt_load_path $BASE_OUT/outputs/4-task181_outcome_extraction/saved_weights
    rm -rf $OUTPUT_DIR/checkpoint*
    touch "$OUTPUT_DIR/done"
fi
sleep 5

# ── TASK 6: task1510_evalution_relation_extraction ────────────────────────────
OUTPUT_DIR="$BASE_OUT/outputs/6-task1510_evalution_relation_extraction"
if [ -f "$OUTPUT_DIR/done" ]; then
    echo "[SKIP] Task 6-task1510_evalution_relation_extraction already complete"
else
    CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_llama.py \
       --do_train \
       --do_predict \
       --predict_with_generate \
       --model_name_or_path $MODEL_PATH \
       --data_dir CL_Benchmark \
       --load_checkpoint_from $BASE_OUT/outputs/5-task002_quoref_answer_generation/saved_weights/trans_input.pt \
       --previous_lora_path $BASE_OUT/outputs/1-task1572_samsum_summary/saved_weights,$BASE_OUT/outputs/2-task363_sst2_polarity_classification/saved_weights,$BASE_OUT/outputs/3-task1290_xsum_summarization/saved_weights,$BASE_OUT/outputs/4-task181_outcome_extraction/saved_weights,$BASE_OUT/outputs/5-task002_quoref_answer_generation/saved_weights \
       --previous_prompt_key_path $BASE_OUT/outputs/5-task002_quoref_answer_generation/saved_weights/prompts_keys_till_now.pt \
       --task_order $TASK_ORDER \
       --gen_data_dir generated_data/lora_gen_superni_llama \
       --task_config_dir configs/gen_script_superni_order1_llama_configs/task1510_evalution_relation_extraction \
       --output_dir $OUTPUT_DIR \
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
       --overwrite_cache \
       --lr_scheduler_type constant \
       --warmup_steps 0 \
       --logging_strategy steps \
       --logging_steps 10 \
       --metric_for_best_model eval_rougeL \
       --eval_strategy steps \
       --save_strategy steps \
       --save_total_limit 1 \
       --lora_r 8 \
       --lora_alpha 32 \
       --lora_dropout 0.0 \
          --data_replay_freq -1 \
       --kl_ratio 0.5 \
       --attn_temperature 1 \
       --mlp_hidden_dim 100 \
       --model_name gainlora \
       --threshold 0.995 \
       --transthreshold 0.995 \
       $FP16_FLAG \
       $SRT_FLAGS \
       --srt_load_path $BASE_OUT/outputs/5-task002_quoref_answer_generation/saved_weights
    rm -rf $OUTPUT_DIR/checkpoint*
    touch "$OUTPUT_DIR/done"
fi
sleep 5

# ── TASK 7: task639_multi_woz_user_utterance_generation ──────────────────────
OUTPUT_DIR="$BASE_OUT/outputs/7-task639_multi_woz_user_utterance_generation"
if [ -f "$OUTPUT_DIR/done" ]; then
    echo "[SKIP] Task 7-task639_multi_woz_user_utterance_generation already complete"
else
    CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_llama.py \
       --do_train \
       --do_predict \
       --predict_with_generate \
       --model_name_or_path $MODEL_PATH \
       --data_dir CL_Benchmark \
       --load_checkpoint_from $BASE_OUT/outputs/6-task1510_evalution_relation_extraction/saved_weights/trans_input.pt \
       --previous_lora_path $BASE_OUT/outputs/1-task1572_samsum_summary/saved_weights,$BASE_OUT/outputs/2-task363_sst2_polarity_classification/saved_weights,$BASE_OUT/outputs/3-task1290_xsum_summarization/saved_weights,$BASE_OUT/outputs/4-task181_outcome_extraction/saved_weights,$BASE_OUT/outputs/5-task002_quoref_answer_generation/saved_weights,$BASE_OUT/outputs/6-task1510_evalution_relation_extraction/saved_weights \
       --previous_prompt_key_path $BASE_OUT/outputs/6-task1510_evalution_relation_extraction/saved_weights/prompts_keys_till_now.pt \
       --task_order $TASK_ORDER \
       --gen_data_dir generated_data/lora_gen_superni_llama \
       --task_config_dir configs/gen_script_superni_order1_llama_configs/task639_multi_woz_user_utterance_generation \
       --output_dir $OUTPUT_DIR \
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
       --overwrite_cache \
       --lr_scheduler_type constant \
       --warmup_steps 0 \
       --logging_strategy steps \
       --logging_steps 10 \
       --metric_for_best_model eval_rougeL \
       --eval_strategy steps \
       --save_strategy steps \
       --save_total_limit 1 \
       --lora_r 8 \
       --lora_alpha 32 \
       --lora_dropout 0.0 \
          --data_replay_freq -1 \
       --kl_ratio 0.5 \
       --attn_temperature 1 \
       --mlp_hidden_dim 100 \
       --model_name gainlora \
       --threshold 0.995 \
       --transthreshold 0.995 \
       $FP16_FLAG \
       $SRT_FLAGS \
       --srt_load_path $BASE_OUT/outputs/6-task1510_evalution_relation_extraction/saved_weights
    rm -rf $OUTPUT_DIR/checkpoint*
    touch "$OUTPUT_DIR/done"
fi
sleep 5

# ── TASK 8: task1729_personachat_generate_next ────────────────────────────────
OUTPUT_DIR="$BASE_OUT/outputs/8-task1729_personachat_generate_next"
if [ -f "$OUTPUT_DIR/done" ]; then
    echo "[SKIP] Task 8-task1729_personachat_generate_next already complete"
else
    CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_llama.py \
       --do_train \
       --do_predict \
       --predict_with_generate \
       --model_name_or_path $MODEL_PATH \
       --data_dir CL_Benchmark \
       --load_checkpoint_from $BASE_OUT/outputs/7-task639_multi_woz_user_utterance_generation/saved_weights/trans_input.pt \
       --previous_lora_path $BASE_OUT/outputs/1-task1572_samsum_summary/saved_weights,$BASE_OUT/outputs/2-task363_sst2_polarity_classification/saved_weights,$BASE_OUT/outputs/3-task1290_xsum_summarization/saved_weights,$BASE_OUT/outputs/4-task181_outcome_extraction/saved_weights,$BASE_OUT/outputs/5-task002_quoref_answer_generation/saved_weights,$BASE_OUT/outputs/6-task1510_evalution_relation_extraction/saved_weights,$BASE_OUT/outputs/7-task639_multi_woz_user_utterance_generation/saved_weights \
       --previous_prompt_key_path $BASE_OUT/outputs/7-task639_multi_woz_user_utterance_generation/saved_weights/prompts_keys_till_now.pt \
       --task_order $TASK_ORDER \
       --gen_data_dir generated_data/lora_gen_superni_llama \
       --task_config_dir configs/gen_script_superni_order1_llama_configs/task1729_personachat_generate_next \
       --output_dir $OUTPUT_DIR \
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
       --overwrite_cache \
       --lr_scheduler_type constant \
       --warmup_steps 0 \
       --logging_strategy steps \
       --logging_steps 10 \
       --metric_for_best_model eval_rougeL \
       --eval_strategy steps \
       --save_strategy steps \
       --save_total_limit 1 \
       --lora_r 8 \
       --lora_alpha 32 \
       --lora_dropout 0.0 \
          --data_replay_freq -1 \
       --kl_ratio 0.5 \
       --attn_temperature 1 \
       --mlp_hidden_dim 100 \
       --model_name gainlora \
       --threshold 0.995 \
       --transthreshold 0.995 \
       $FP16_FLAG \
       $SRT_FLAGS \
       --srt_load_path $BASE_OUT/outputs/7-task639_multi_woz_user_utterance_generation/saved_weights
    rm -rf $OUTPUT_DIR/checkpoint*
    touch "$OUTPUT_DIR/done"
fi
sleep 5

# ── TASK 9: task073_commonsenseqa_answer_generation ─────────────────────────
OUTPUT_DIR="$BASE_OUT/outputs/9-task073_commonsenseqa_answer_generation"
if [ -f "$OUTPUT_DIR/done" ]; then
    echo "[SKIP] Task 9-task073_commonsenseqa_answer_generation already complete"
else
    CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_llama.py \
       --do_train \
       --do_predict \
       --predict_with_generate \
       --model_name_or_path $MODEL_PATH \
       --data_dir CL_Benchmark \
       --load_checkpoint_from $BASE_OUT/outputs/8-task1729_personachat_generate_next/saved_weights/trans_input.pt \
       --previous_lora_path $BASE_OUT/outputs/1-task1572_samsum_summary/saved_weights,$BASE_OUT/outputs/2-task363_sst2_polarity_classification/saved_weights,$BASE_OUT/outputs/3-task1290_xsum_summarization/saved_weights,$BASE_OUT/outputs/4-task181_outcome_extraction/saved_weights,$BASE_OUT/outputs/5-task002_quoref_answer_generation/saved_weights,$BASE_OUT/outputs/6-task1510_evalution_relation_extraction/saved_weights,$BASE_OUT/outputs/7-task639_multi_woz_user_utterance_generation/saved_weights,$BASE_OUT/outputs/8-task1729_personachat_generate_next/saved_weights \
       --previous_prompt_key_path $BASE_OUT/outputs/8-task1729_personachat_generate_next/saved_weights/prompts_keys_till_now.pt \
       --task_order $TASK_ORDER \
       --gen_data_dir generated_data/lora_gen_superni_llama \
       --task_config_dir configs/gen_script_superni_order1_llama_configs/task073_commonsenseqa_answer_generation \
       --output_dir $OUTPUT_DIR \
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
       --overwrite_cache \
       --lr_scheduler_type constant \
       --warmup_steps 0 \
       --logging_strategy steps \
       --logging_steps 10 \
       --metric_for_best_model eval_rougeL \
       --eval_strategy steps \
       --save_strategy steps \
       --save_total_limit 1 \
       --lora_r 8 \
       --lora_alpha 32 \
       --lora_dropout 0.0 \
          --data_replay_freq -1 \
       --kl_ratio 0.5 \
       --attn_temperature 1 \
       --mlp_hidden_dim 100 \
       --model_name gainlora \
       --threshold 0.995 \
       --transthreshold 0.995 \
       $FP16_FLAG \
       $SRT_FLAGS \
       --srt_load_path $BASE_OUT/outputs/8-task1729_personachat_generate_next/saved_weights
    rm -rf $OUTPUT_DIR/checkpoint*
    touch "$OUTPUT_DIR/done"
fi
sleep 5

# ── TASK 10: task1590_diplomacy_text_generation ──────────────────────────────
OUTPUT_DIR="$BASE_OUT/outputs/10-task1590_diplomacy_text_generation"
if [ -f "$OUTPUT_DIR/done" ]; then
    echo "[SKIP] Task 10-task1590_diplomacy_text_generation already complete"
else
    CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_llama.py \
       --do_train \
       --do_predict \
       --predict_with_generate \
       --model_name_or_path $MODEL_PATH \
       --data_dir CL_Benchmark \
       --load_checkpoint_from $BASE_OUT/outputs/9-task073_commonsenseqa_answer_generation/saved_weights/trans_input.pt \
       --previous_lora_path $BASE_OUT/outputs/1-task1572_samsum_summary/saved_weights,$BASE_OUT/outputs/2-task363_sst2_polarity_classification/saved_weights,$BASE_OUT/outputs/3-task1290_xsum_summarization/saved_weights,$BASE_OUT/outputs/4-task181_outcome_extraction/saved_weights,$BASE_OUT/outputs/5-task002_quoref_answer_generation/saved_weights,$BASE_OUT/outputs/6-task1510_evalution_relation_extraction/saved_weights,$BASE_OUT/outputs/7-task639_multi_woz_user_utterance_generation/saved_weights,$BASE_OUT/outputs/8-task1729_personachat_generate_next/saved_weights,$BASE_OUT/outputs/9-task073_commonsenseqa_answer_generation/saved_weights \
       --previous_prompt_key_path $BASE_OUT/outputs/9-task073_commonsenseqa_answer_generation/saved_weights/prompts_keys_till_now.pt \
       --task_order $TASK_ORDER \
       --gen_data_dir generated_data/lora_gen_superni_llama \
       --task_config_dir configs/gen_script_superni_order1_llama_configs/task1590_diplomacy_text_generation \
       --output_dir $OUTPUT_DIR \
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
       --overwrite_cache \
       --lr_scheduler_type constant \
       --warmup_steps 0 \
       --logging_strategy steps \
       --logging_steps 10 \
       --metric_for_best_model eval_rougeL \
       --eval_strategy steps \
       --save_strategy steps \
       --save_total_limit 1 \
       --lora_r 8 \
       --lora_alpha 32 \
       --lora_dropout 0.0 \
          --data_replay_freq -1 \
       --kl_ratio 0.5 \
       --attn_temperature 1 \
       --mlp_hidden_dim 100 \
       --model_name gainlora \
       --threshold 0.995 \
       --transthreshold 0.995 \
       $FP16_FLAG \
       $SRT_FLAGS \
       --srt_load_path $BASE_OUT/outputs/9-task073_commonsenseqa_answer_generation/saved_weights
    rm -rf $OUTPUT_DIR/checkpoint*
    touch "$OUTPUT_DIR/done"
fi
sleep 5

# ── TASK 11: task748_glucose_reverse_cause_event_detection ────────────────────
OUTPUT_DIR="$BASE_OUT/outputs/11-task748_glucose_reverse_cause_event_detection"
if [ -f "$OUTPUT_DIR/done" ]; then
    echo "[SKIP] Task 11-task748_glucose_reverse_cause_event_detection already complete"
else
    CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_llama.py \
       --do_train \
       --do_predict \
       --predict_with_generate \
       --model_name_or_path $MODEL_PATH \
       --data_dir CL_Benchmark \
       --load_checkpoint_from $BASE_OUT/outputs/10-task1590_diplomacy_text_generation/saved_weights/trans_input.pt \
       --previous_lora_path $BASE_OUT/outputs/1-task1572_samsum_summary/saved_weights,$BASE_OUT/outputs/2-task363_sst2_polarity_classification/saved_weights,$BASE_OUT/outputs/3-task1290_xsum_summarization/saved_weights,$BASE_OUT/outputs/4-task181_outcome_extraction/saved_weights,$BASE_OUT/outputs/5-task002_quoref_answer_generation/saved_weights,$BASE_OUT/outputs/6-task1510_evalution_relation_extraction/saved_weights,$BASE_OUT/outputs/7-task639_multi_woz_user_utterance_generation/saved_weights,$BASE_OUT/outputs/8-task1729_personachat_generate_next/saved_weights,$BASE_OUT/outputs/9-task073_commonsenseqa_answer_generation/saved_weights,$BASE_OUT/outputs/10-task1590_diplomacy_text_generation/saved_weights \
       --previous_prompt_key_path $BASE_OUT/outputs/10-task1590_diplomacy_text_generation/saved_weights/prompts_keys_till_now.pt \
       --task_order $TASK_ORDER \
       --gen_data_dir generated_data/lora_gen_superni_llama \
       --task_config_dir configs/gen_script_superni_order1_llama_configs/task748_glucose_reverse_cause_event_detection \
       --output_dir $OUTPUT_DIR \
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
       --overwrite_cache \
       --lr_scheduler_type constant \
       --warmup_steps 0 \
       --logging_strategy steps \
       --logging_steps 10 \
       --metric_for_best_model eval_rougeL \
       --eval_strategy steps \
       --save_strategy steps \
       --save_total_limit 1 \
       --lora_r 8 \
       --lora_alpha 32 \
       --lora_dropout 0.0 \
          --data_replay_freq -1 \
       --kl_ratio 0.5 \
       --attn_temperature 1 \
       --mlp_hidden_dim 100 \
       --model_name gainlora \
       --threshold 0.995 \
       --transthreshold 0.995 \
       $FP16_FLAG \
       $SRT_FLAGS \
       --srt_load_path $BASE_OUT/outputs/10-task1590_diplomacy_text_generation/saved_weights
    rm -rf $OUTPUT_DIR/checkpoint*
    touch "$OUTPUT_DIR/done"
fi
sleep 5

# ── TASK 12: task511_reddit_tifu_long_text_summarization ──────────────────────
OUTPUT_DIR="$BASE_OUT/outputs/12-task511_reddit_tifu_long_text_summarization"
if [ -f "$OUTPUT_DIR/done" ]; then
    echo "[SKIP] Task 12-task511_reddit_tifu_long_text_summarization already complete"
else
    CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_llama.py \
       --do_train \
       --do_predict \
       --predict_with_generate \
       --model_name_or_path $MODEL_PATH \
       --data_dir CL_Benchmark \
       --load_checkpoint_from $BASE_OUT/outputs/11-task748_glucose_reverse_cause_event_detection/saved_weights/trans_input.pt \
       --previous_lora_path $BASE_OUT/outputs/1-task1572_samsum_summary/saved_weights,$BASE_OUT/outputs/2-task363_sst2_polarity_classification/saved_weights,$BASE_OUT/outputs/3-task1290_xsum_summarization/saved_weights,$BASE_OUT/outputs/4-task181_outcome_extraction/saved_weights,$BASE_OUT/outputs/5-task002_quoref_answer_generation/saved_weights,$BASE_OUT/outputs/6-task1510_evalution_relation_extraction/saved_weights,$BASE_OUT/outputs/7-task639_multi_woz_user_utterance_generation/saved_weights,$BASE_OUT/outputs/8-task1729_personachat_generate_next/saved_weights,$BASE_OUT/outputs/9-task073_commonsenseqa_answer_generation/saved_weights,$BASE_OUT/outputs/10-task1590_diplomacy_text_generation/saved_weights,$BASE_OUT/outputs/11-task748_glucose_reverse_cause_event_detection/saved_weights \
       --previous_prompt_key_path $BASE_OUT/outputs/11-task748_glucose_reverse_cause_event_detection/saved_weights/prompts_keys_till_now.pt \
       --task_order $TASK_ORDER \
       --gen_data_dir generated_data/lora_gen_superni_llama \
       --task_config_dir configs/gen_script_superni_order1_llama_configs/task511_reddit_tifu_long_text_summarization \
       --output_dir $OUTPUT_DIR \
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
       --overwrite_cache \
       --lr_scheduler_type constant \
       --warmup_steps 0 \
       --logging_strategy steps \
       --logging_steps 10 \
       --metric_for_best_model eval_rougeL \
       --eval_strategy steps \
       --save_strategy steps \
       --save_total_limit 1 \
       --lora_r 8 \
       --lora_alpha 32 \
       --lora_dropout 0.0 \
          --data_replay_freq -1 \
       --kl_ratio 0.5 \
       --attn_temperature 1 \
       --mlp_hidden_dim 100 \
       --model_name gainlora \
       --threshold 0.995 \
       --transthreshold 0.995 \
       $FP16_FLAG \
       $SRT_FLAGS \
       --srt_load_path $BASE_OUT/outputs/11-task748_glucose_reverse_cause_event_detection/saved_weights
    rm -rf $OUTPUT_DIR/checkpoint*
    touch "$OUTPUT_DIR/done"
fi
sleep 5

# ── TASK 13: task591_sciq_answer_generation ──────────────────────────────────
OUTPUT_DIR="$BASE_OUT/outputs/13-task591_sciq_answer_generation"
if [ -f "$OUTPUT_DIR/done" ]; then
    echo "[SKIP] Task 13-task591_sciq_answer_generation already complete"
else
    CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_llama.py \
       --do_train \
       --do_predict \
       --predict_with_generate \
       --model_name_or_path $MODEL_PATH \
       --data_dir CL_Benchmark \
       --load_checkpoint_from $BASE_OUT/outputs/12-task511_reddit_tifu_long_text_summarization/saved_weights/trans_input.pt \
       --previous_lora_path $BASE_OUT/outputs/1-task1572_samsum_summary/saved_weights,$BASE_OUT/outputs/2-task363_sst2_polarity_classification/saved_weights,$BASE_OUT/outputs/3-task1290_xsum_summarization/saved_weights,$BASE_OUT/outputs/4-task181_outcome_extraction/saved_weights,$BASE_OUT/outputs/5-task002_quoref_answer_generation/saved_weights,$BASE_OUT/outputs/6-task1510_evalution_relation_extraction/saved_weights,$BASE_OUT/outputs/7-task639_multi_woz_user_utterance_generation/saved_weights,$BASE_OUT/outputs/8-task1729_personachat_generate_next/saved_weights,$BASE_OUT/outputs/9-task073_commonsenseqa_answer_generation/saved_weights,$BASE_OUT/outputs/10-task1590_diplomacy_text_generation/saved_weights,$BASE_OUT/outputs/11-task748_glucose_reverse_cause_event_detection/saved_weights,$BASE_OUT/outputs/12-task511_reddit_tifu_long_text_summarization/saved_weights \
       --previous_prompt_key_path $BASE_OUT/outputs/12-task511_reddit_tifu_long_text_summarization/saved_weights/prompts_keys_till_now.pt \
       --task_order $TASK_ORDER \
       --gen_data_dir generated_data/lora_gen_superni_llama \
       --task_config_dir configs/gen_script_superni_order1_llama_configs/task591_sciq_answer_generation \
       --output_dir $OUTPUT_DIR \
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
       --overwrite_cache \
       --lr_scheduler_type constant \
       --warmup_steps 0 \
       --logging_strategy steps \
       --logging_steps 10 \
       --metric_for_best_model eval_rougeL \
       --eval_strategy steps \
       --save_strategy steps \
       --save_total_limit 1 \
       --lora_r 8 \
       --lora_alpha 32 \
       --lora_dropout 0.0 \
          --data_replay_freq -1 \
       --kl_ratio 0.5 \
       --attn_temperature 1 \
       --mlp_hidden_dim 100 \
       --model_name gainlora \
       --threshold 0.995 \
       --transthreshold 0.995 \
       $FP16_FLAG \
       $SRT_FLAGS \
       --srt_load_path $BASE_OUT/outputs/12-task511_reddit_tifu_long_text_summarization/saved_weights
    rm -rf $OUTPUT_DIR/checkpoint*
    touch "$OUTPUT_DIR/done"
fi
sleep 5

# ── TASK 14: task1687_sentiment140_classification ────────────────────────────
OUTPUT_DIR="$BASE_OUT/outputs/14-task1687_sentiment140_classification"
if [ -f "$OUTPUT_DIR/done" ]; then
    echo "[SKIP] Task 14-task1687_sentiment140_classification already complete"
else
    CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_llama.py \
       --do_train \
       --do_predict \
       --predict_with_generate \
       --model_name_or_path $MODEL_PATH \
       --data_dir CL_Benchmark \
       --load_checkpoint_from $BASE_OUT/outputs/13-task591_sciq_answer_generation/saved_weights/trans_input.pt \
       --previous_lora_path $BASE_OUT/outputs/1-task1572_samsum_summary/saved_weights,$BASE_OUT/outputs/2-task363_sst2_polarity_classification/saved_weights,$BASE_OUT/outputs/3-task1290_xsum_summarization/saved_weights,$BASE_OUT/outputs/4-task181_outcome_extraction/saved_weights,$BASE_OUT/outputs/5-task002_quoref_answer_generation/saved_weights,$BASE_OUT/outputs/6-task1510_evalution_relation_extraction/saved_weights,$BASE_OUT/outputs/7-task639_multi_woz_user_utterance_generation/saved_weights,$BASE_OUT/outputs/8-task1729_personachat_generate_next/saved_weights,$BASE_OUT/outputs/9-task073_commonsenseqa_answer_generation/saved_weights,$BASE_OUT/outputs/10-task1590_diplomacy_text_generation/saved_weights,$BASE_OUT/outputs/11-task748_glucose_reverse_cause_event_detection/saved_weights,$BASE_OUT/outputs/12-task511_reddit_tifu_long_text_summarization/saved_weights,$BASE_OUT/outputs/13-task591_sciq_answer_generation/saved_weights \
       --previous_prompt_key_path $BASE_OUT/outputs/13-task591_sciq_answer_generation/saved_weights/prompts_keys_till_now.pt \
       --task_order $TASK_ORDER \
       --gen_data_dir generated_data/lora_gen_superni_llama \
       --task_config_dir configs/gen_script_superni_order1_llama_configs/task1687_sentiment140_classification \
       --output_dir $OUTPUT_DIR \
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
       --overwrite_cache \
       --lr_scheduler_type constant \
       --warmup_steps 0 \
       --logging_strategy steps \
       --logging_steps 10 \
       --metric_for_best_model eval_exact_match \
       --eval_strategy steps \
       --save_strategy steps \
       --save_total_limit 1 \
       --lora_r 8 \
       --lora_alpha 32 \
       --lora_dropout 0.0 \
          --data_replay_freq -1 \
       --kl_ratio 0.5 \
       --attn_temperature 1 \
       --mlp_hidden_dim 100 \
       --model_name gainlora \
       --threshold 0.995 \
       --transthreshold 0.995 \
       $FP16_FLAG \
       $SRT_FLAGS \
       --srt_load_path $BASE_OUT/outputs/13-task591_sciq_answer_generation/saved_weights
    rm -rf $OUTPUT_DIR/checkpoint*
    touch "$OUTPUT_DIR/done"
fi
sleep 5

# ── TASK 15: task875_emotion_classification ───────────────────────────────────
OUTPUT_DIR="$BASE_OUT/outputs/15-task875_emotion_classification"
if [ -f "$OUTPUT_DIR/done" ]; then
    echo "[SKIP] Task 15-task875_emotion_classification already complete"
else
    CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_llama.py \
       --do_train \
       --do_predict \
       --predict_with_generate \
       --model_name_or_path $MODEL_PATH \
       --data_dir CL_Benchmark \
       --load_checkpoint_from $BASE_OUT/outputs/14-task1687_sentiment140_classification/saved_weights/trans_input.pt \
       --previous_lora_path $BASE_OUT/outputs/1-task1572_samsum_summary/saved_weights,$BASE_OUT/outputs/2-task363_sst2_polarity_classification/saved_weights,$BASE_OUT/outputs/3-task1290_xsum_summarization/saved_weights,$BASE_OUT/outputs/4-task181_outcome_extraction/saved_weights,$BASE_OUT/outputs/5-task002_quoref_answer_generation/saved_weights,$BASE_OUT/outputs/6-task1510_evalution_relation_extraction/saved_weights,$BASE_OUT/outputs/7-task639_multi_woz_user_utterance_generation/saved_weights,$BASE_OUT/outputs/8-task1729_personachat_generate_next/saved_weights,$BASE_OUT/outputs/9-task073_commonsenseqa_answer_generation/saved_weights,$BASE_OUT/outputs/10-task1590_diplomacy_text_generation/saved_weights,$BASE_OUT/outputs/11-task748_glucose_reverse_cause_event_detection/saved_weights,$BASE_OUT/outputs/12-task511_reddit_tifu_long_text_summarization/saved_weights,$BASE_OUT/outputs/13-task591_sciq_answer_generation/saved_weights,$BASE_OUT/outputs/14-task1687_sentiment140_classification/saved_weights \
       --previous_prompt_key_path $BASE_OUT/outputs/14-task1687_sentiment140_classification/saved_weights/prompts_keys_till_now.pt \
       --task_order $TASK_ORDER \
       --gen_data_dir generated_data/lora_gen_superni_llama \
       --task_config_dir configs/gen_script_superni_order1_llama_configs/task875_emotion_classification \
       --output_dir $OUTPUT_DIR \
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
       --overwrite_cache \
       --lr_scheduler_type constant \
       --warmup_steps 0 \
       --logging_strategy steps \
       --logging_steps 10 \
       --metric_for_best_model eval_exact_match \
       --eval_strategy steps \
       --save_strategy steps \
       --save_total_limit 1 \
       --lora_r 8 \
       --lora_alpha 32 \
       --lora_dropout 0.0 \
          --data_replay_freq -1 \
       --kl_ratio 0.5 \
       --attn_temperature 1 \
       --mlp_hidden_dim 100 \
       --model_name gainlora \
       --threshold 0.995 \
       --transthreshold 0.995 \
       $FP16_FLAG \
       $SRT_FLAGS \
       --srt_load_path $BASE_OUT/outputs/14-task1687_sentiment140_classification/saved_weights
    rm -rf $OUTPUT_DIR/checkpoint*
    touch "$OUTPUT_DIR/done"
fi

echo "[DONE] All 15 tasks complete. Run: python score.py $RUN_NAME $RUN_NAME"
