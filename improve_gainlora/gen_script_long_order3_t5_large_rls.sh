#!/bin/bash
#SBATCH -J cl-rls-large
#SBATCH -o cl-rls-large-%j.out
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 30:00:00
#SBATCH --mem 128G
#SBATCH --gres=gpu:2

# ============================================================
# SpecRoute V11: RLS Analytical Router + InfLoRA/CPI/GPM
# Long Sequence Order 3 — T5-LARGE (optimized, no redundancy)
# ============================================================

export CUDA_DEVICE_ORDER="PCI_BUS_ID"

# Auto-detect GPU count and type for optimal parallelism
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)

if [ -z "$GPU_MEM" ]; then
    echo "ERROR: No GPU detected!"
    exit 1
fi

# Determine GPU type and parallelism strategy
if [ "$GPU_MEM" -lt 15500 ]; then
    GPU_MODE="t4_2gpu"
    GPU_IDS="0,1"
    STRAT="2x T4 DataParallel"
    # T5-large won't fit well on T4 16GB with training
    BSZ=1; GA=16; EVAL_BSZ=8
elif [ "$GPU_MEM" -le 17000 ]; then
    GPU_MODE="p100"
    GPU_IDS="${1:-0}"
    STRAT="P100 16GB"
    BSZ=4; GA=8; EVAL_BSZ=16
else
    GPU_MODE="a100"
    if [ "$NUM_GPUS" -ge 2 ]; then
        GPU_IDS="0,1"
        STRAT="${NUM_GPUS}x ${GPU_MEM}MB DataParallel"
    else
        GPU_IDS="${1:-0}"
        STRAT="1x ${GPU_MEM}MB GPU"
    fi
    # T5-large: higher batch sizes with extra VRAM
    if [ "$GPU_MEM" -ge 40000 ]; then
        BSZ=32; GA=1; EVAL_BSZ=64
    else
        BSZ=16; GA=2; EVAL_BSZ=32
    fi
fi

echo "[GPU] Detected (~${GPU_MEM}MB per GPU): $STRAT"
echo "[HP] BSZ=$BSZ, GA=$GA, EVAL_BSZ=$EVAL_BSZ"
echo "[GPU] Using CUDA_VISIBLE_DEVICES=$GPU_IDS"
echo "============================================================"

# ============================================================
# Configuration
# ============================================================
RLS_EXPANSION_DIM=2048
RLS_LAMBDA=0.1
ROUTING_MODE=rls

TASK_ORDER=yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic
RUN_NAME=gen_script_long_order3_t5_large_rls
CONFIG_BASE=configs/gen_script_long_order3_t5_configs
OUTPUT_BASE=logs_and_outputs/${RUN_NAME}/outputs

# Common hyperparameters (all tasks)
COMMON_ARGS="
   --do_train
   --predict_with_generate
   --model_name_or_path $2
   --data_dir CL_Benchmark
   --task_order ${TASK_ORDER}
   --per_device_train_batch_size $BSZ
   --per_device_eval_batch_size $EVAL_BSZ
   --gradient_accumulation_steps $GA
   --learning_rate 0.0003
   --num_train_epochs 10
   --run_name ${RUN_NAME}
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
   --evaluation_strategy steps
   --save_strategy steps
   --save_total_limit 1
   --load_best_model_at_end
   --lora_r 8
   --lora_alpha 32
   --lora_dropout 0.0
   --run_single False
   --n_batches_c5 100
   --data_replay_freq -1
   --mlp_hidden_dim 100
   --model_name specroute
   --routing_mode ${ROUTING_MODE}
   --rls_expansion_dim ${RLS_EXPANSION_DIM}
   --rls_lambda ${RLS_LAMBDA}
   --cpi_gamma 0.5
   --oap_eta 0.5
   --oap_beta_min 0.3
   --oap_warmup 3
   --threshold 0.995
   --transthreshold 0.995
   --do_predict
"

# ============================================================
# Generate previous_lora_path string for each task
# ============================================================
build_prev_lora_list() {
    local task_num=$1
    local list=""
    for i in $(seq 1 $((task_num - 1))); do
        if [ $i -gt 1 ]; then
            list="${list},"
        fi
        list="${list}${OUTPUT_BASE}/$i-${TASKS[$((i-1))]}/saved_weights"
    done
    echo "$list"
}

# ============================================================
# Task array: indexed from 0
# ============================================================
TASKS=(yelp amazon mnli cb copa qqp rte imdb sst2 dbpedia agnews yahoo multirc boolq wic)

# ============================================================
# Run all 15 tasks
# ============================================================
for task_idx in ${!TASKS[@]}; do
    task_num=$((task_idx + 1))
    task_name=${TASKS[$task_idx]}
    
    echo ""
    echo "============================================================"
    echo "Task $task_num: $task_name"
    echo "============================================================"
    
    # Build metric key and previous_lora_path
    metric_key="eval_exact_match"
    if [ $task_num -gt 1 ]; then
        metric_key="${metric_key}_for_${task_name}"
        prev_lora=$(build_prev_lora_list $task_num)
        prev_lora_arg="--previous_lora_path $prev_lora"
    else
        prev_lora_arg=""
    fi
    
    # Task 1 has different metric (no suffix)
    if [ $task_num -eq 1 ]; then
        metric_key="eval_exact_match"
    fi
    
    # Run training + prediction
    CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
       $COMMON_ARGS \
       $prev_lora_arg \
       --task_config_dir ${CONFIG_BASE}/${task_name} \
       --output_dir ${OUTPUT_BASE}/${task_num}-${task_name} \
       --metric_for_best_model $metric_key
    
    # Cleanup checkpoints (save space)
    rm -rf ${OUTPUT_BASE}/${task_num}-${task_name}/checkpoint*
    
    # Brief pause before next task
    sleep 2
done

echo ""
echo "============================================================"
echo "All 15 tasks completed!"
echo "============================================================"
