#!/bin/bash
#SBATCH -J cl-rls
#SBATCH -o cl-rls-%j.out
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 20:00:00
#SBATCH --mem 128G
#SBATCH --gres=gpu:2

# ============================================================
# SpecRoute V11: RLS Analytical Router + InfLoRA/CPI/GPM
# Long Sequence Order 3 — T5-small
# ============================================================

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
if [ "$GPU_MEM" -lt 15500 ]; then
    IS_T4=1
    echo "[GPU] Detected T4 GPUs (${GPU_MEM}MB VRAM each)"
else
    IS_T4=0
    echo "[GPU] Detected high-memory GPUs (${GPU_MEM}MB VRAM each)"
fi

# Determine parallelism strategy
if [ "$IS_T4" -eq 1 ] && [ "$NUM_GPUS" -ge 2 ]; then
    GPU_MODE="t4_2gpu"
    GPU_IDS="0,1"
    FP16_FLAG="--gradient_checkpointing"
    echo "[GPU] Strategy: 2x T4 DataParallel + fp32 + gradient_checkpointing"
elif [ "$IS_T4" -eq 1 ]; then
    GPU_MODE="t4_1gpu"
    GPU_IDS="${1:-0}"
    FP16_FLAG="--gradient_checkpointing"
    echo "[GPU] Strategy: 1x T4 + fp32 + gradient_checkpointing"
elif [ "$GPU_MEM" -le 17000 ]; then
    GPU_MODE="p100"
    GPU_IDS="${1:-0}"
    FP16_FLAG="--gradient_checkpointing"
    echo "[GPU] Strategy: P100 16GB (fp32 + gradient_checkpointing)"
else
    GPU_MODE="a100"
    if [ "$NUM_GPUS" -ge 2 ]; then
        GPU_IDS="0,1"
        echo "[GPU] Strategy: ${NUM_GPUS}x ${GPU_MEM}MB DataParallel (RTX3090/A100, fp32)"
    else
        GPU_IDS="${1:-0}"
        echo "[GPU] Strategy: 1x ${GPU_MEM}MB GPU (fp32)"
    fi
    FP16_FLAG=""
fi

echo "[GPU] Using CUDA_VISIBLE_DEVICES=$GPU_IDS"
echo "============================================================"
echo ""

# V11 RLS-specific hyperparameters
RLS_EXPANSION_DIM=2048
RLS_LAMBDA=0.1
ROUTING_MODE=rls

TASK_ORDER=yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic
RUN_NAME=gen_script_long_order3_t5_rls
CONFIG_BASE=configs/gen_script_long_order3_t5_configs
OUTPUT_BASE=logs_and_outputs/${RUN_NAME}/outputs

# ============================================================
# Task 1: yelp (no previous LoRA)
# ============================================================
if [ "$GPU_MODE" = "t4_2gpu" ]; then
    BSZ=2; GA=8; EVAL_BSZ=16
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=4; GA=8; EVAL_BSZ=16
elif [ "$GPU_MODE" = "p100" ]; then
    BSZ=16; GA=2; EVAL_BSZ=32
else
    BSZ=8; GA=4; EVAL_BSZ=128
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --data_dir CL_Benchmark \
   --task_order ${TASK_ORDER} \
   --task_config_dir ${CONFIG_BASE}/yelp \
   --output_dir ${OUTPUT_BASE}/1-yelp \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name ${RUN_NAME} \
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
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --do_predict \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --routing_mode ${ROUTING_MODE} \
   --rls_expansion_dim ${RLS_EXPANSION_DIM} \
   --rls_lambda ${RLS_LAMBDA} \
   --cpi_gamma 0.5 \
   --oap_eta 0.5 \
   --oap_beta_min 0.3 \
   --oap_warmup 3 \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

rm -rf ${OUTPUT_BASE}/1-yelp/checkpoint*

sleep 5

# ============================================================
# Task 2: amazon
# ============================================================
if [ "$GPU_MODE" = "t4_2gpu" ]; then
    BSZ=2; GA=4; EVAL_BSZ=16
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=4; GA=8; EVAL_BSZ=16
elif [ "$GPU_MODE" = "p100" ]; then
    BSZ=16; GA=2; EVAL_BSZ=32
else
    BSZ=16; GA=2; EVAL_BSZ=128
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path ${OUTPUT_BASE}/1-yelp/saved_weights \
   --data_dir CL_Benchmark \
   --task_order ${TASK_ORDER} \
   --task_config_dir ${CONFIG_BASE}/amazon \
   --output_dir ${OUTPUT_BASE}/2-amazon \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name ${RUN_NAME} \
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
   --metric_for_best_model eval_exact_match_for_amazon \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --run_single False \
   --n_batches_c5 100 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --routing_mode ${ROUTING_MODE} \
   --rls_expansion_dim ${RLS_EXPANSION_DIM} \
   --rls_lambda ${RLS_LAMBDA} \
   --cpi_gamma 0.5 \
   --oap_eta 0.5 \
   --oap_beta_min 0.3 \
   --oap_warmup 3 \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

rm -rf ${OUTPUT_BASE}/2-amazon/checkpoint*

sleep 5

# ============================================================
# Task 3: mnli
# ============================================================
CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path ${OUTPUT_BASE}/1-yelp/saved_weights,${OUTPUT_BASE}/2-amazon/saved_weights \
   --data_dir CL_Benchmark \
   --task_order ${TASK_ORDER} \
   --task_config_dir ${CONFIG_BASE}/mnli \
   --output_dir ${OUTPUT_BASE}/3-mnli \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name ${RUN_NAME} \
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
   --metric_for_best_model eval_exact_match_for_mnli \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --run_single False \
   --n_batches_c5 100 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --routing_mode ${ROUTING_MODE} \
   --rls_expansion_dim ${RLS_EXPANSION_DIM} \
   --rls_lambda ${RLS_LAMBDA} \
   --cpi_gamma 0.5 \
   --oap_eta 0.5 \
   --oap_beta_min 0.3 \
   --oap_warmup 3 \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

rm -rf ${OUTPUT_BASE}/3-mnli/checkpoint*

sleep 5

# ============================================================
# Task 4: cb
# ============================================================
CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path ${OUTPUT_BASE}/1-yelp/saved_weights,${OUTPUT_BASE}/2-amazon/saved_weights,${OUTPUT_BASE}/3-mnli/saved_weights \
   --data_dir CL_Benchmark \
   --task_order ${TASK_ORDER} \
   --task_config_dir ${CONFIG_BASE}/cb \
   --output_dir ${OUTPUT_BASE}/4-cb \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name ${RUN_NAME} \
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
   --metric_for_best_model eval_exact_match_for_cb \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --run_single False \
   --n_batches_c5 100 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --routing_mode ${ROUTING_MODE} \
   --rls_expansion_dim ${RLS_EXPANSION_DIM} \
   --rls_lambda ${RLS_LAMBDA} \
   --cpi_gamma 0.5 \
   --oap_eta 0.5 \
   --oap_beta_min 0.3 \
   --oap_warmup 3 \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

rm -rf ${OUTPUT_BASE}/4-cb/checkpoint*

sleep 5

# ============================================================
# Task 5: copa
# ============================================================
CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path ${OUTPUT_BASE}/1-yelp/saved_weights,${OUTPUT_BASE}/2-amazon/saved_weights,${OUTPUT_BASE}/3-mnli/saved_weights,${OUTPUT_BASE}/4-cb/saved_weights \
   --data_dir CL_Benchmark \
   --task_order ${TASK_ORDER} \
   --task_config_dir ${CONFIG_BASE}/copa \
   --output_dir ${OUTPUT_BASE}/5-copa \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name ${RUN_NAME} \
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
   --metric_for_best_model eval_exact_match_for_copa \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --run_single False \
   --n_batches_c5 100 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --routing_mode ${ROUTING_MODE} \
   --rls_expansion_dim ${RLS_EXPANSION_DIM} \
   --rls_lambda ${RLS_LAMBDA} \
   --cpi_gamma 0.5 \
   --oap_eta 0.5 \
   --oap_beta_min 0.3 \
   --oap_warmup 3 \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

rm -rf ${OUTPUT_BASE}/5-copa/checkpoint*

sleep 5

# ============================================================
# Task 6: qqp
# ============================================================
CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path ${OUTPUT_BASE}/1-yelp/saved_weights,${OUTPUT_BASE}/2-amazon/saved_weights,${OUTPUT_BASE}/3-mnli/saved_weights,${OUTPUT_BASE}/4-cb/saved_weights,${OUTPUT_BASE}/5-copa/saved_weights \
   --data_dir CL_Benchmark \
   --task_order ${TASK_ORDER} \
   --task_config_dir ${CONFIG_BASE}/qqp \
   --output_dir ${OUTPUT_BASE}/6-qqp \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name ${RUN_NAME} \
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
   --metric_for_best_model eval_exact_match_for_qqp \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --run_single False \
   --n_batches_c5 100 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --routing_mode ${ROUTING_MODE} \
   --rls_expansion_dim ${RLS_EXPANSION_DIM} \
   --rls_lambda ${RLS_LAMBDA} \
   --cpi_gamma 0.5 \
   --oap_eta 0.5 \
   --oap_beta_min 0.3 \
   --oap_warmup 3 \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

rm -rf ${OUTPUT_BASE}/6-qqp/checkpoint*

sleep 5

# ============================================================
# Task 7: rte
# ============================================================
CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path ${OUTPUT_BASE}/1-yelp/saved_weights,${OUTPUT_BASE}/2-amazon/saved_weights,${OUTPUT_BASE}/3-mnli/saved_weights,${OUTPUT_BASE}/4-cb/saved_weights,${OUTPUT_BASE}/5-copa/saved_weights,${OUTPUT_BASE}/6-qqp/saved_weights \
   --data_dir CL_Benchmark \
   --task_order ${TASK_ORDER} \
   --task_config_dir ${CONFIG_BASE}/rte \
   --output_dir ${OUTPUT_BASE}/7-rte \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name ${RUN_NAME} \
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
   --metric_for_best_model eval_exact_match_for_rte \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --run_single False \
   --n_batches_c5 100 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --routing_mode ${ROUTING_MODE} \
   --rls_expansion_dim ${RLS_EXPANSION_DIM} \
   --rls_lambda ${RLS_LAMBDA} \
   --cpi_gamma 0.5 \
   --oap_eta 0.5 \
   --oap_beta_min 0.3 \
   --oap_warmup 3 \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

rm -rf ${OUTPUT_BASE}/7-rte/checkpoint*

sleep 5

# ============================================================
# Task 8: imdb
# ============================================================
CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path ${OUTPUT_BASE}/1-yelp/saved_weights,${OUTPUT_BASE}/2-amazon/saved_weights,${OUTPUT_BASE}/3-mnli/saved_weights,${OUTPUT_BASE}/4-cb/saved_weights,${OUTPUT_BASE}/5-copa/saved_weights,${OUTPUT_BASE}/6-qqp/saved_weights,${OUTPUT_BASE}/7-rte/saved_weights \
   --data_dir CL_Benchmark \
   --task_order ${TASK_ORDER} \
   --task_config_dir ${CONFIG_BASE}/imdb \
   --output_dir ${OUTPUT_BASE}/8-imdb \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name ${RUN_NAME} \
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
   --metric_for_best_model eval_exact_match_for_imdb \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --run_single False \
   --n_batches_c5 100 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --routing_mode ${ROUTING_MODE} \
   --rls_expansion_dim ${RLS_EXPANSION_DIM} \
   --rls_lambda ${RLS_LAMBDA} \
   --cpi_gamma 0.5 \
   --oap_eta 0.5 \
   --oap_beta_min 0.3 \
   --oap_warmup 3 \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

rm -rf ${OUTPUT_BASE}/8-imdb/checkpoint*

sleep 5

# ============================================================
# Task 9: sst2
# ============================================================
CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path ${OUTPUT_BASE}/1-yelp/saved_weights,${OUTPUT_BASE}/2-amazon/saved_weights,${OUTPUT_BASE}/3-mnli/saved_weights,${OUTPUT_BASE}/4-cb/saved_weights,${OUTPUT_BASE}/5-copa/saved_weights,${OUTPUT_BASE}/6-qqp/saved_weights,${OUTPUT_BASE}/7-rte/saved_weights,${OUTPUT_BASE}/8-imdb/saved_weights \
   --data_dir CL_Benchmark \
   --task_order ${TASK_ORDER} \
   --task_config_dir ${CONFIG_BASE}/sst2 \
   --output_dir ${OUTPUT_BASE}/9-sst2 \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name ${RUN_NAME} \
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
   --metric_for_best_model eval_exact_match_for_sst2 \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --run_single False \
   --n_batches_c5 100 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --routing_mode ${ROUTING_MODE} \
   --rls_expansion_dim ${RLS_EXPANSION_DIM} \
   --rls_lambda ${RLS_LAMBDA} \
   --cpi_gamma 0.5 \
   --oap_eta 0.5 \
   --oap_beta_min 0.3 \
   --oap_warmup 3 \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

rm -rf ${OUTPUT_BASE}/9-sst2/checkpoint*

sleep 5

# ============================================================
# Task 10: dbpedia
# ============================================================
CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path ${OUTPUT_BASE}/1-yelp/saved_weights,${OUTPUT_BASE}/2-amazon/saved_weights,${OUTPUT_BASE}/3-mnli/saved_weights,${OUTPUT_BASE}/4-cb/saved_weights,${OUTPUT_BASE}/5-copa/saved_weights,${OUTPUT_BASE}/6-qqp/saved_weights,${OUTPUT_BASE}/7-rte/saved_weights,${OUTPUT_BASE}/8-imdb/saved_weights,${OUTPUT_BASE}/9-sst2/saved_weights \
   --data_dir CL_Benchmark \
   --task_order ${TASK_ORDER} \
   --task_config_dir ${CONFIG_BASE}/dbpedia \
   --output_dir ${OUTPUT_BASE}/10-dbpedia \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name ${RUN_NAME} \
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
   --metric_for_best_model eval_exact_match_for_dbpedia \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --run_single False \
   --n_batches_c5 100 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --routing_mode ${ROUTING_MODE} \
   --rls_expansion_dim ${RLS_EXPANSION_DIM} \
   --rls_lambda ${RLS_LAMBDA} \
   --cpi_gamma 0.5 \
   --oap_eta 0.5 \
   --oap_beta_min 0.3 \
   --oap_warmup 3 \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

rm -rf ${OUTPUT_BASE}/10-dbpedia/checkpoint*

sleep 5

# ============================================================
# Task 11: agnews
# ============================================================
CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path ${OUTPUT_BASE}/1-yelp/saved_weights,${OUTPUT_BASE}/2-amazon/saved_weights,${OUTPUT_BASE}/3-mnli/saved_weights,${OUTPUT_BASE}/4-cb/saved_weights,${OUTPUT_BASE}/5-copa/saved_weights,${OUTPUT_BASE}/6-qqp/saved_weights,${OUTPUT_BASE}/7-rte/saved_weights,${OUTPUT_BASE}/8-imdb/saved_weights,${OUTPUT_BASE}/9-sst2/saved_weights,${OUTPUT_BASE}/10-dbpedia/saved_weights \
   --data_dir CL_Benchmark \
   --task_order ${TASK_ORDER} \
   --task_config_dir ${CONFIG_BASE}/agnews \
   --output_dir ${OUTPUT_BASE}/11-agnews \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name ${RUN_NAME} \
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
   --metric_for_best_model eval_exact_match_for_agnews \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --run_single False \
   --n_batches_c5 100 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --routing_mode ${ROUTING_MODE} \
   --rls_expansion_dim ${RLS_EXPANSION_DIM} \
   --rls_lambda ${RLS_LAMBDA} \
   --cpi_gamma 0.5 \
   --oap_eta 0.5 \
   --oap_beta_min 0.3 \
   --oap_warmup 3 \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

rm -rf ${OUTPUT_BASE}/11-agnews/checkpoint*

sleep 5

# ============================================================
# Task 12: yahoo
# ============================================================
CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path ${OUTPUT_BASE}/1-yelp/saved_weights,${OUTPUT_BASE}/2-amazon/saved_weights,${OUTPUT_BASE}/3-mnli/saved_weights,${OUTPUT_BASE}/4-cb/saved_weights,${OUTPUT_BASE}/5-copa/saved_weights,${OUTPUT_BASE}/6-qqp/saved_weights,${OUTPUT_BASE}/7-rte/saved_weights,${OUTPUT_BASE}/8-imdb/saved_weights,${OUTPUT_BASE}/9-sst2/saved_weights,${OUTPUT_BASE}/10-dbpedia/saved_weights,${OUTPUT_BASE}/11-agnews/saved_weights \
   --data_dir CL_Benchmark \
   --task_order ${TASK_ORDER} \
   --task_config_dir ${CONFIG_BASE}/yahoo \
   --output_dir ${OUTPUT_BASE}/12-yahoo \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name ${RUN_NAME} \
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
   --metric_for_best_model eval_exact_match_for_yahoo \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --run_single False \
   --n_batches_c5 100 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --routing_mode ${ROUTING_MODE} \
   --rls_expansion_dim ${RLS_EXPANSION_DIM} \
   --rls_lambda ${RLS_LAMBDA} \
   --cpi_gamma 0.5 \
   --oap_eta 0.5 \
   --oap_beta_min 0.3 \
   --oap_warmup 3 \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

rm -rf ${OUTPUT_BASE}/12-yahoo/checkpoint*

sleep 5

# ============================================================
# Task 13: multirc
# ============================================================
CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path ${OUTPUT_BASE}/1-yelp/saved_weights,${OUTPUT_BASE}/2-amazon/saved_weights,${OUTPUT_BASE}/3-mnli/saved_weights,${OUTPUT_BASE}/4-cb/saved_weights,${OUTPUT_BASE}/5-copa/saved_weights,${OUTPUT_BASE}/6-qqp/saved_weights,${OUTPUT_BASE}/7-rte/saved_weights,${OUTPUT_BASE}/8-imdb/saved_weights,${OUTPUT_BASE}/9-sst2/saved_weights,${OUTPUT_BASE}/10-dbpedia/saved_weights,${OUTPUT_BASE}/11-agnews/saved_weights,${OUTPUT_BASE}/12-yahoo/saved_weights \
   --data_dir CL_Benchmark \
   --task_order ${TASK_ORDER} \
   --task_config_dir ${CONFIG_BASE}/multirc \
   --output_dir ${OUTPUT_BASE}/13-multirc \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name ${RUN_NAME} \
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
   --metric_for_best_model eval_exact_match_for_multirc \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --run_single False \
   --n_batches_c5 100 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --routing_mode ${ROUTING_MODE} \
   --rls_expansion_dim ${RLS_EXPANSION_DIM} \
   --rls_lambda ${RLS_LAMBDA} \
   --cpi_gamma 0.5 \
   --oap_eta 0.5 \
   --oap_beta_min 0.3 \
   --oap_warmup 3 \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

rm -rf ${OUTPUT_BASE}/13-multirc/checkpoint*

sleep 5

# ============================================================
# Task 14: boolq
# ============================================================
CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path ${OUTPUT_BASE}/1-yelp/saved_weights,${OUTPUT_BASE}/2-amazon/saved_weights,${OUTPUT_BASE}/3-mnli/saved_weights,${OUTPUT_BASE}/4-cb/saved_weights,${OUTPUT_BASE}/5-copa/saved_weights,${OUTPUT_BASE}/6-qqp/saved_weights,${OUTPUT_BASE}/7-rte/saved_weights,${OUTPUT_BASE}/8-imdb/saved_weights,${OUTPUT_BASE}/9-sst2/saved_weights,${OUTPUT_BASE}/10-dbpedia/saved_weights,${OUTPUT_BASE}/11-agnews/saved_weights,${OUTPUT_BASE}/12-yahoo/saved_weights,${OUTPUT_BASE}/13-multirc/saved_weights \
   --data_dir CL_Benchmark \
   --task_order ${TASK_ORDER} \
   --task_config_dir ${CONFIG_BASE}/boolq \
   --output_dir ${OUTPUT_BASE}/14-boolq \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name ${RUN_NAME} \
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
   --metric_for_best_model eval_exact_match_for_boolq \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --run_single False \
   --n_batches_c5 100 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --routing_mode ${ROUTING_MODE} \
   --rls_expansion_dim ${RLS_EXPANSION_DIM} \
   --rls_lambda ${RLS_LAMBDA} \
   --cpi_gamma 0.5 \
   --oap_eta 0.5 \
   --oap_beta_min 0.3 \
   --oap_warmup 3 \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

rm -rf ${OUTPUT_BASE}/14-boolq/checkpoint*

sleep 5

# ============================================================
# Task 15: wic (FINAL)
# ============================================================
CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path ${OUTPUT_BASE}/1-yelp/saved_weights,${OUTPUT_BASE}/2-amazon/saved_weights,${OUTPUT_BASE}/3-mnli/saved_weights,${OUTPUT_BASE}/4-cb/saved_weights,${OUTPUT_BASE}/5-copa/saved_weights,${OUTPUT_BASE}/6-qqp/saved_weights,${OUTPUT_BASE}/7-rte/saved_weights,${OUTPUT_BASE}/8-imdb/saved_weights,${OUTPUT_BASE}/9-sst2/saved_weights,${OUTPUT_BASE}/10-dbpedia/saved_weights,${OUTPUT_BASE}/11-agnews/saved_weights,${OUTPUT_BASE}/12-yahoo/saved_weights,${OUTPUT_BASE}/13-multirc/saved_weights,${OUTPUT_BASE}/14-boolq/saved_weights \
   --data_dir CL_Benchmark \
   --task_order ${TASK_ORDER} \
   --task_config_dir ${CONFIG_BASE}/wic \
   --output_dir ${OUTPUT_BASE}/15-wic \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name ${RUN_NAME} \
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
   --metric_for_best_model eval_exact_match_for_wic \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --run_single False \
   --n_batches_c5 100 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --routing_mode ${ROUTING_MODE} \
   --rls_expansion_dim ${RLS_EXPANSION_DIM} \
   --rls_lambda ${RLS_LAMBDA} \
   --cpi_gamma 0.5 \
   --oap_eta 0.5 \
   --oap_beta_min 0.3 \
   --oap_warmup 3 \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

rm -rf ${OUTPUT_BASE}/15-wic/checkpoint*

echo "============================================================"
echo "V11 RLS Training Complete!"
echo "Results: ${OUTPUT_BASE}/"
echo "============================================================"
