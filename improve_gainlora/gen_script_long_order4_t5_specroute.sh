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
    GPU_IDS="${1:-0}"
    FP16_FLAG=""
    echo "[GPU] Strategy: A100 (single GPU, fp32)"
fi

echo "[GPU] Using CUDA_VISIBLE_DEVICES=$GPU_IDS"
echo "============================================================"
echo ""

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
   --predict_with_generate \
   --model_name_or_path $2 \
   --data_dir CL_Benchmark \
   --task_order mnli,cb,wic,copa,qqp,boolq,rte,imdb,yelp,amazon,sst2,dbpedia,agnews,multirc,yahoo \
   --task_config_dir configs/gen_script_long_order4_t5_configs/mnli \
   --output_dir logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/1-mnli \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order4_t5_specroute \
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
   --run_single True \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

rm -rf logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/1-mnli/checkpoint*

sleep 5

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
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/1-mnli/saved_weights \
   --data_dir CL_Benchmark \
   --task_order mnli,cb,wic,copa,qqp,boolq,rte,imdb,yelp,amazon,sst2,dbpedia,agnews,multirc,yahoo \
   --task_config_dir configs/gen_script_long_order4_t5_configs/cb \
   --output_dir logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/2-cb \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order4_t5_specroute \
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
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

rm -rf logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/2-cb/checkpoint*

sleep 5

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
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/1-mnli/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/2-cb/saved_weights \
   --data_dir CL_Benchmark \
   --task_order mnli,cb,wic,copa,qqp,boolq,rte,imdb,yelp,amazon,sst2,dbpedia,agnews,multirc,yahoo \
   --task_config_dir configs/gen_script_long_order4_t5_configs/wic \
   --output_dir logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/3-wic \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order4_t5_specroute \
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
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

rm -rf logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/3-wic/checkpoint*

sleep 5

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
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/1-mnli/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/2-cb/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/3-wic/saved_weights \
   --data_dir CL_Benchmark \
   --task_order mnli,cb,wic,copa,qqp,boolq,rte,imdb,yelp,amazon,sst2,dbpedia,agnews,multirc,yahoo \
   --task_config_dir configs/gen_script_long_order4_t5_configs/copa \
   --output_dir logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/4-copa \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order4_t5_specroute \
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
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

rm -rf logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/4-copa/checkpoint*

sleep 5

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
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/1-mnli/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/2-cb/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/3-wic/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/4-copa/saved_weights \
   --data_dir CL_Benchmark \
   --task_order mnli,cb,wic,copa,qqp,boolq,rte,imdb,yelp,amazon,sst2,dbpedia,agnews,multirc,yahoo \
   --task_config_dir configs/gen_script_long_order4_t5_configs/qqp \
   --output_dir logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/5-qqp \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order4_t5_specroute \
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
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

rm -rf logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/5-qqp/checkpoint*

sleep 5

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
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/1-mnli/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/2-cb/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/3-wic/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/4-copa/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/5-qqp/saved_weights \
   --data_dir CL_Benchmark \
   --task_order mnli,cb,wic,copa,qqp,boolq,rte,imdb,yelp,amazon,sst2,dbpedia,agnews,multirc,yahoo \
   --task_config_dir configs/gen_script_long_order4_t5_configs/boolq \
   --output_dir logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/6-boolq \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order4_t5_specroute \
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
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

rm -rf logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/6-boolq/checkpoint*

sleep 5

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
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/1-mnli/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/2-cb/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/3-wic/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/4-copa/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/5-qqp/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/6-boolq/saved_weights \
   --data_dir CL_Benchmark \
   --task_order mnli,cb,wic,copa,qqp,boolq,rte,imdb,yelp,amazon,sst2,dbpedia,agnews,multirc,yahoo \
   --task_config_dir configs/gen_script_long_order4_t5_configs/rte \
   --output_dir logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/7-rte \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order4_t5_specroute \
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
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

rm -rf logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/7-rte/checkpoint*

sleep 5

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
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/1-mnli/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/2-cb/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/3-wic/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/4-copa/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/5-qqp/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/6-boolq/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/7-rte/saved_weights \
   --data_dir CL_Benchmark \
   --task_order mnli,cb,wic,copa,qqp,boolq,rte,imdb,yelp,amazon,sst2,dbpedia,agnews,multirc,yahoo \
   --task_config_dir configs/gen_script_long_order4_t5_configs/imdb \
   --output_dir logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/8-imdb \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order4_t5_specroute \
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
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

rm -rf logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/8-imdb/checkpoint*

sleep 5

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
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/1-mnli/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/2-cb/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/3-wic/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/4-copa/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/5-qqp/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/6-boolq/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/7-rte/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/8-imdb/saved_weights \
   --data_dir CL_Benchmark \
   --task_order mnli,cb,wic,copa,qqp,boolq,rte,imdb,yelp,amazon,sst2,dbpedia,agnews,multirc,yahoo \
   --task_config_dir configs/gen_script_long_order4_t5_configs/yelp \
   --output_dir logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/9-yelp \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order4_t5_specroute \
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
   --metric_for_best_model eval_exact_match_for_yelp \
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
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

rm -rf logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/9-yelp/checkpoint*

sleep 5

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
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/1-mnli/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/2-cb/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/3-wic/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/4-copa/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/5-qqp/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/6-boolq/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/7-rte/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/8-imdb/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/9-yelp/saved_weights \
   --data_dir CL_Benchmark \
   --task_order mnli,cb,wic,copa,qqp,boolq,rte,imdb,yelp,amazon,sst2,dbpedia,agnews,multirc,yahoo \
   --task_config_dir configs/gen_script_long_order4_t5_configs/amazon \
   --output_dir logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/10-amazon \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order4_t5_specroute \
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
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

rm -rf logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/10-amazon/checkpoint*

sleep 5

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
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/1-mnli/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/2-cb/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/3-wic/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/4-copa/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/5-qqp/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/6-boolq/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/7-rte/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/8-imdb/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/9-yelp/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/10-amazon/saved_weights \
   --data_dir CL_Benchmark \
   --task_order mnli,cb,wic,copa,qqp,boolq,rte,imdb,yelp,amazon,sst2,dbpedia,agnews,multirc,yahoo \
   --task_config_dir configs/gen_script_long_order4_t5_configs/sst2 \
   --output_dir logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/11-sst2 \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order4_t5_specroute \
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
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

rm -rf logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/11-sst2/checkpoint*

sleep 5

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
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/1-mnli/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/2-cb/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/3-wic/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/4-copa/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/5-qqp/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/6-boolq/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/7-rte/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/8-imdb/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/9-yelp/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/10-amazon/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/11-sst2/saved_weights \
   --data_dir CL_Benchmark \
   --task_order mnli,cb,wic,copa,qqp,boolq,rte,imdb,yelp,amazon,sst2,dbpedia,agnews,multirc,yahoo \
   --task_config_dir configs/gen_script_long_order4_t5_configs/dbpedia \
   --output_dir logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/12-dbpedia \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order4_t5_specroute \
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
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

rm -rf logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/12-dbpedia/checkpoint*

sleep 5

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
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/1-mnli/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/2-cb/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/3-wic/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/4-copa/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/5-qqp/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/6-boolq/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/7-rte/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/8-imdb/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/9-yelp/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/10-amazon/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/11-sst2/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/12-dbpedia/saved_weights \
   --data_dir CL_Benchmark \
   --task_order mnli,cb,wic,copa,qqp,boolq,rte,imdb,yelp,amazon,sst2,dbpedia,agnews,multirc,yahoo \
   --task_config_dir configs/gen_script_long_order4_t5_configs/agnews \
   --output_dir logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/13-agnews \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order4_t5_specroute \
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
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

rm -rf logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/13-agnews/checkpoint*

sleep 5

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
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/1-mnli/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/2-cb/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/3-wic/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/4-copa/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/5-qqp/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/6-boolq/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/7-rte/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/8-imdb/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/9-yelp/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/10-amazon/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/11-sst2/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/12-dbpedia/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/13-agnews/saved_weights \
   --data_dir CL_Benchmark \
   --task_order mnli,cb,wic,copa,qqp,boolq,rte,imdb,yelp,amazon,sst2,dbpedia,agnews,multirc,yahoo \
   --task_config_dir configs/gen_script_long_order4_t5_configs/multirc \
   --output_dir logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/14-multirc \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order4_t5_specroute \
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
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

rm -rf logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/14-multirc/checkpoint*

sleep 5

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
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/1-mnli/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/2-cb/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/3-wic/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/4-copa/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/5-qqp/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/6-boolq/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/7-rte/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/8-imdb/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/9-yelp/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/10-amazon/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/11-sst2/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/12-dbpedia/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/13-agnews/saved_weights,logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/14-multirc/saved_weights \
   --data_dir CL_Benchmark \
   --task_order mnli,cb,wic,copa,qqp,boolq,rte,imdb,yelp,amazon,sst2,dbpedia,agnews,multirc,yahoo \
   --task_config_dir configs/gen_script_long_order4_t5_configs/yahoo \
   --output_dir logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/15-yahoo \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order4_t5_specroute \
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
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG

rm -rf logs_and_outputs/gen_script_long_order4_t5_specroute/outputs/15-yahoo/checkpoint*

sleep 5
