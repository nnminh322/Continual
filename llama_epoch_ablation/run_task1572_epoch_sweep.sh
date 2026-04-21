#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
REPO_DIR="$WORKSPACE_DIR/new_gainlora"

MODEL_PATH="meta-llama/Llama-2-7b-hf"
GPU_IDS="0"
EPOCHS_LIST="100,150,200"
SGWI_FLAG="True"
DUAL_FISHER_FLAG="False"
LAMBDA_EMB=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path)   MODEL_PATH="$2"; shift 2 ;;
        --gpu_ids)      GPU_IDS="$2"; shift 2 ;;
        --epochs_list)  EPOCHS_LIST="$2"; shift 2 ;;
        --sgwi)         SGWI_FLAG="$2"; shift 2 ;;
        --dual_fisher)  DUAL_FISHER_FLAG="$2"; shift 2 ;;
        --lambda_emb)   LAMBDA_EMB="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
: ${GPU_MEM:=16000}
: ${NUM_GPUS:=1}

if [[ "$GPU_MEM" -lt 20000 ]]; then
    GPU_MODE="t4_1gpu"
    FP16_ARGS=(--gradient_checkpointing)
    BSZ=1
    GA=16
    EVAL_BSZ=2
elif [[ "$GPU_MEM" -lt 50000 ]]; then
    GPU_MODE="mid"
    FP16_ARGS=(--gradient_checkpointing)
    BSZ=1
    GA=16
    EVAL_BSZ=4
else
    GPU_MODE="a100"
    FP16_ARGS=()
    BSZ=1
    GA=16
    EVAL_BSZ=8
fi

TASK_ORDER="task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification"

IFS=',' read -r -a EPOCHS_ARRAY <<< "$EPOCHS_LIST"

pushd "$REPO_DIR" >/dev/null

for EPOCHS in "${EPOCHS_ARRAY[@]}"; do
    if [[ ! "$EPOCHS" =~ ^[0-9]+$ ]]; then
        echo "Invalid epoch value: $EPOCHS" >&2
        exit 1
    fi

    RUN_NAME="task1572_llama2_srt_epochs${EPOCHS}"
    BASE_OUT="../llama_epoch_ablation/logs_and_outputs/$RUN_NAME"
    OUTPUT_DIR="$BASE_OUT/outputs/1-task1572_samsum_summary"
    LOG_FILE="$SCRIPT_DIR/logs/task1572_epochs${EPOCHS}_$(date +%Y%m%d_%H%M%S).log"

    mkdir -p "$OUTPUT_DIR"

    CMD=(
        python src/run_llama.py
        --do_train
        --do_predict
        --predict_with_generate
        --model_name_or_path "$MODEL_PATH"
        --data_dir CL_Benchmark
        --task_order "$TASK_ORDER"
        --task_config_dir configs/gen_script_superni_order1_llama_configs/task1572_samsum_summary
        --output_dir "$OUTPUT_DIR"
        --per_device_train_batch_size "$BSZ"
        --per_device_eval_batch_size "$EVAL_BSZ"
        --gradient_accumulation_steps "$GA"
        --learning_rate 0.0003
        --num_train_epochs "$EPOCHS"
        --run_name "$RUN_NAME"
        --max_source_length 512
        --max_target_length 50
        --generation_max_length 50
        --add_task_name False
        --add_dataset_name False
        --overwrite_output_dir
        --overwrite_cache
        --lr_scheduler_type constant
        --warmup_steps 0
        --logging_strategy steps
        --logging_steps 10
        --metric_for_best_model eval_rougeL
        --eval_strategy steps
        --save_strategy steps
        --save_total_limit 1
        --lora_r 8
        --lora_alpha 32
        --lora_dropout 0.0
        --data_replay_freq -1
        --replay_after_n_epoch 0
        --kl_ratio 0.5
        --attn_temperature 1
        --mlp_hidden_dim 100
        --model_name gainlora
        --threshold 0.995
        --transthreshold 0.995
        --use_srt_router
        --srt_metric_mode hard
        --srt_max_emb_samples 500
        --srt_skip_forward
        --sgwi "$SGWI_FLAG"
        --dual_fisher "$DUAL_FISHER_FLAG"
    )

    if [[ -n "$LAMBDA_EMB" ]]; then
        CMD+=(--lambda_emb "$LAMBDA_EMB")
    fi
    if [[ ${#FP16_ARGS[@]} -gt 0 ]]; then
        CMD+=("${FP16_ARGS[@]}")
    fi

    {
        echo "[ABLAT-TASK1572] gpu_mode=$GPU_MODE gpu_ids=$GPU_IDS"
        echo "[ABLAT-TASK1572] model_path=$MODEL_PATH"
        echo "[ABLAT-TASK1572] epochs=$EPOCHS"
        echo "[ABLAT-TASK1572] output_dir=$OUTPUT_DIR"
        echo "[ABLAT-TASK1572] log_file=$LOG_FILE"
        echo "[ABLAT-TASK1572] sgwi=$SGWI_FLAG dual_fisher=$DUAL_FISHER_FLAG lambda_emb=${LAMBDA_EMB:-auto}"
        printf '%q ' "CUDA_VISIBLE_DEVICES=$GPU_IDS"
        printf '%q ' "${CMD[@]}"
        printf '\n'
    } | tee "$LOG_FILE"

    {
        echo '#!/usr/bin/env bash'
        printf '%q ' env CUDA_VISIBLE_DEVICES="$GPU_IDS" "${CMD[@]}"
        printf '\n'
    } > "$OUTPUT_DIR/launch_command.sh"

    env CUDA_VISIBLE_DEVICES="$GPU_IDS" "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"

    rm -rf "$OUTPUT_DIR"/checkpoint*
    touch "$OUTPUT_DIR/done"
done

popd >/dev/null

echo "[ABLAT-TASK1572] done. See $SCRIPT_DIR/logs/ and $SCRIPT_DIR/logs_and_outputs/"
