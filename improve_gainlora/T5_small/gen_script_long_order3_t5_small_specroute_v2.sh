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
if [ "$IS_T4" -eq 1 ] && [ "$NUM_GPUS" -ge 2 ]; then
    GPU_MODE="t4_2gpu"
    GPU_IDS="0,1"
    FP16_FLAG=""
    echo "[GPU] Strategy: 2x T4 DataParallel + fp32 + gradient_checkpointing"
elif [ "$IS_T4" -eq 1 ]; then
    GPU_MODE="t4_1gpu"
    GPU_IDS="${1:-0}"
    FP16_FLAG=""
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

if [ "$GPU_MODE" = "t4_2gpu" ]; then
    BSZ=2; GA=8; EVAL_BSZ=128
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=4; GA=8; EVAL_BSZ=128
else
    BSZ=8; GA=4; EVAL_BSZ=128
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --task_config_dir configs/gen_script_long_order3_t5_configs/yelp \
   --output_dir logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/1-yelp \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order3_t5_small_specroute_v2 \
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
   --evaluation_strategy epoch \
   --save_strategy epoch \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --training_bias 1.0 \
   --gen_data_dir CL_Benchmark \
   --threshold 0.980 \
   --transthreshold 0.980 \
   --eval_accumulation_steps 10 \
   $FP16_FLAG

rm -rf logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/1-yelp/checkpoint*

# Free HF datasets cache and OS page cache between tasks
find /tmp -name "*.arrow" -mmin +5 -delete 2>/dev/null; find ~/.cache/huggingface/datasets -name "cache-*.arrow" -delete 2>/dev/null
sleep 5

if [ "$GPU_MODE" = "t4_2gpu" ]; then
    BSZ=2; GA=8; EVAL_BSZ=128
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=4; GA=8; EVAL_BSZ=128
else
    BSZ=8; GA=4; EVAL_BSZ=128
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/1-yelp/saved_weights \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --task_config_dir configs/gen_script_long_order3_t5_configs/amazon \
   --output_dir logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/2-amazon \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order3_t5_small_specroute_v2 \
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
   --evaluation_strategy epoch \
   --save_strategy epoch \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --training_bias 1.0 \
   --gen_data_dir CL_Benchmark \
   --threshold 0.980 \
   --transthreshold 0.980 \
   --eval_accumulation_steps 10 \
   $FP16_FLAG

rm -rf logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/2-amazon/checkpoint*

# Free HF datasets cache and OS page cache between tasks
find /tmp -name "*.arrow" -mmin +5 -delete 2>/dev/null; find ~/.cache/huggingface/datasets -name "cache-*.arrow" -delete 2>/dev/null
sleep 5

if [ "$GPU_MODE" = "t4_2gpu" ]; then
    BSZ=2; GA=8; EVAL_BSZ=128
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=4; GA=8; EVAL_BSZ=128
else
    BSZ=8; GA=4; EVAL_BSZ=128
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/1-yelp/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/2-amazon/saved_weights \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --task_config_dir configs/gen_script_long_order3_t5_configs/mnli \
   --output_dir logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/3-mnli \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order3_t5_small_specroute_v2 \
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
   --evaluation_strategy epoch \
   --save_strategy epoch \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --training_bias 1.0 \
   --gen_data_dir CL_Benchmark \
   --threshold 0.980 \
   --transthreshold 0.980 \
   --eval_accumulation_steps 10 \
   $FP16_FLAG

rm -rf logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/3-mnli/checkpoint*

# Free HF datasets cache and OS page cache between tasks
find /tmp -name "*.arrow" -mmin +5 -delete 2>/dev/null; find ~/.cache/huggingface/datasets -name "cache-*.arrow" -delete 2>/dev/null
sleep 5

if [ "$GPU_MODE" = "t4_2gpu" ]; then
    BSZ=2; GA=8; EVAL_BSZ=128
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=4; GA=8; EVAL_BSZ=128
else
    BSZ=8; GA=4; EVAL_BSZ=128
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/1-yelp/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/2-amazon/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/3-mnli/saved_weights \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --task_config_dir configs/gen_script_long_order3_t5_configs/cb \
   --output_dir logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/4-cb \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order3_t5_small_specroute_v2 \
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
   --evaluation_strategy epoch \
   --save_strategy epoch \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --training_bias 1.0 \
   --gen_data_dir CL_Benchmark \
   --threshold 0.980 \
   --transthreshold 0.980 \
   --eval_accumulation_steps 10 \
   $FP16_FLAG

rm -rf logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/4-cb/checkpoint*

# Free HF datasets cache and OS page cache between tasks
find /tmp -name "*.arrow" -mmin +5 -delete 2>/dev/null; find ~/.cache/huggingface/datasets -name "cache-*.arrow" -delete 2>/dev/null
sleep 5

if [ "$GPU_MODE" = "t4_2gpu" ]; then
    BSZ=2; GA=8; EVAL_BSZ=128
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=4; GA=8; EVAL_BSZ=128
else
    BSZ=8; GA=4; EVAL_BSZ=128
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/1-yelp/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/2-amazon/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/3-mnli/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/4-cb/saved_weights \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --task_config_dir configs/gen_script_long_order3_t5_configs/copa \
   --output_dir logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/5-copa \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order3_t5_small_specroute_v2 \
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
   --evaluation_strategy epoch \
   --save_strategy epoch \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --training_bias 1.0 \
   --gen_data_dir CL_Benchmark \
   --threshold 0.980 \
   --transthreshold 0.980 \
   --eval_accumulation_steps 10 \
   $FP16_FLAG

rm -rf logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/5-copa/checkpoint*

# Free HF datasets cache and OS page cache between tasks
find /tmp -name "*.arrow" -mmin +5 -delete 2>/dev/null; find ~/.cache/huggingface/datasets -name "cache-*.arrow" -delete 2>/dev/null
sleep 5

if [ "$GPU_MODE" = "t4_2gpu" ]; then
    BSZ=2; GA=8; EVAL_BSZ=128
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=4; GA=8; EVAL_BSZ=128
else
    BSZ=8; GA=4; EVAL_BSZ=128
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/1-yelp/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/2-amazon/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/3-mnli/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/4-cb/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/5-copa/saved_weights \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --task_config_dir configs/gen_script_long_order3_t5_configs/qqp \
   --output_dir logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/6-qqp \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order3_t5_small_specroute_v2 \
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
   --evaluation_strategy epoch \
   --save_strategy epoch \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --training_bias 1.0 \
   --gen_data_dir CL_Benchmark \
   --threshold 0.980 \
   --transthreshold 0.980 \
   --eval_accumulation_steps 10 \
   $FP16_FLAG

rm -rf logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/6-qqp/checkpoint*

# Free HF datasets cache and OS page cache between tasks
find /tmp -name "*.arrow" -mmin +5 -delete 2>/dev/null; find ~/.cache/huggingface/datasets -name "cache-*.arrow" -delete 2>/dev/null
sleep 5

if [ "$GPU_MODE" = "t4_2gpu" ]; then
    BSZ=2; GA=8; EVAL_BSZ=128
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=4; GA=8; EVAL_BSZ=128
else
    BSZ=8; GA=4; EVAL_BSZ=128
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/1-yelp/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/2-amazon/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/3-mnli/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/4-cb/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/5-copa/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/6-qqp/saved_weights \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --task_config_dir configs/gen_script_long_order3_t5_configs/rte \
   --output_dir logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/7-rte \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order3_t5_small_specroute_v2 \
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
   --evaluation_strategy epoch \
   --save_strategy epoch \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --training_bias 1.0 \
   --gen_data_dir CL_Benchmark \
   --threshold 0.980 \
   --transthreshold 0.980 \
   --eval_accumulation_steps 10 \
   $FP16_FLAG

rm -rf logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/7-rte/checkpoint*

# Free HF datasets cache and OS page cache between tasks
find /tmp -name "*.arrow" -mmin +5 -delete 2>/dev/null; find ~/.cache/huggingface/datasets -name "cache-*.arrow" -delete 2>/dev/null
sleep 5

if [ "$GPU_MODE" = "t4_2gpu" ]; then
    BSZ=2; GA=8; EVAL_BSZ=128
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=4; GA=8; EVAL_BSZ=128
else
    BSZ=8; GA=4; EVAL_BSZ=128
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/1-yelp/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/2-amazon/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/3-mnli/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/4-cb/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/5-copa/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/6-qqp/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/7-rte/saved_weights \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --task_config_dir configs/gen_script_long_order3_t5_configs/imdb \
   --output_dir logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/8-imdb \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order3_t5_small_specroute_v2 \
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
   --evaluation_strategy epoch \
   --save_strategy epoch \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --training_bias 1.0 \
   --gen_data_dir CL_Benchmark \
   --threshold 0.980 \
   --transthreshold 0.980 \
   --eval_accumulation_steps 10 \
   $FP16_FLAG

rm -rf logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/8-imdb/checkpoint*

# Free HF datasets cache and OS page cache between tasks
find /tmp -name "*.arrow" -mmin +5 -delete 2>/dev/null; find ~/.cache/huggingface/datasets -name "cache-*.arrow" -delete 2>/dev/null
sleep 5

if [ "$GPU_MODE" = "t4_2gpu" ]; then
    BSZ=2; GA=8; EVAL_BSZ=128
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=4; GA=8; EVAL_BSZ=128
else
    BSZ=8; GA=4; EVAL_BSZ=128
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/1-yelp/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/2-amazon/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/3-mnli/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/4-cb/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/5-copa/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/6-qqp/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/7-rte/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/8-imdb/saved_weights \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --task_config_dir configs/gen_script_long_order3_t5_configs/sst2 \
   --output_dir logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/9-sst2 \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order3_t5_small_specroute_v2 \
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
   --evaluation_strategy epoch \
   --save_strategy epoch \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --training_bias 1.0 \
   --gen_data_dir CL_Benchmark \
   --threshold 0.980 \
   --transthreshold 0.980 \
   --eval_accumulation_steps 10 \
   $FP16_FLAG

rm -rf logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/9-sst2/checkpoint*

# Free HF datasets cache and OS page cache between tasks
find /tmp -name "*.arrow" -mmin +5 -delete 2>/dev/null; find ~/.cache/huggingface/datasets -name "cache-*.arrow" -delete 2>/dev/null
sleep 5

if [ "$GPU_MODE" = "t4_2gpu" ]; then
    BSZ=2; GA=8; EVAL_BSZ=128
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=4; GA=8; EVAL_BSZ=128
else
    BSZ=8; GA=4; EVAL_BSZ=128
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/1-yelp/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/2-amazon/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/3-mnli/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/4-cb/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/5-copa/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/6-qqp/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/7-rte/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/8-imdb/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/9-sst2/saved_weights \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --task_config_dir configs/gen_script_long_order3_t5_configs/dbpedia \
   --output_dir logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/10-dbpedia \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order3_t5_small_specroute_v2 \
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
   --evaluation_strategy epoch \
   --save_strategy epoch \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --training_bias 1.0 \
   --gen_data_dir CL_Benchmark \
   --threshold 0.980 \
   --transthreshold 0.980 \
   --eval_accumulation_steps 10 \
   $FP16_FLAG

rm -rf logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/10-dbpedia/checkpoint*

# Free HF datasets cache and OS page cache between tasks
find /tmp -name "*.arrow" -mmin +5 -delete 2>/dev/null; find ~/.cache/huggingface/datasets -name "cache-*.arrow" -delete 2>/dev/null
sleep 5

if [ "$GPU_MODE" = "t4_2gpu" ]; then
    BSZ=2; GA=8; EVAL_BSZ=128
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=4; GA=8; EVAL_BSZ=128
else
    BSZ=8; GA=4; EVAL_BSZ=128
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/1-yelp/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/2-amazon/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/3-mnli/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/4-cb/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/5-copa/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/6-qqp/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/7-rte/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/8-imdb/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/9-sst2/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/10-dbpedia/saved_weights \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --task_config_dir configs/gen_script_long_order3_t5_configs/agnews \
   --output_dir logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/11-agnews \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order3_t5_small_specroute_v2 \
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
   --evaluation_strategy epoch \
   --save_strategy epoch \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --training_bias 1.0 \
   --gen_data_dir CL_Benchmark \
   --threshold 0.980 \
   --transthreshold 0.980 \
   --eval_accumulation_steps 10 \
   $FP16_FLAG

rm -rf logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/11-agnews/checkpoint*

# Free HF datasets cache and OS page cache between tasks
find /tmp -name "*.arrow" -mmin +5 -delete 2>/dev/null; find ~/.cache/huggingface/datasets -name "cache-*.arrow" -delete 2>/dev/null
sleep 5

if [ "$GPU_MODE" = "t4_2gpu" ]; then
    BSZ=2; GA=8; EVAL_BSZ=128
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=4; GA=8; EVAL_BSZ=128
else
    BSZ=8; GA=4; EVAL_BSZ=128
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/1-yelp/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/2-amazon/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/3-mnli/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/4-cb/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/5-copa/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/6-qqp/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/7-rte/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/8-imdb/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/9-sst2/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/10-dbpedia/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/11-agnews/saved_weights \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --task_config_dir configs/gen_script_long_order3_t5_configs/yahoo \
   --output_dir logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/12-yahoo \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order3_t5_small_specroute_v2 \
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
   --evaluation_strategy epoch \
   --save_strategy epoch \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --training_bias 1.0 \
   --gen_data_dir CL_Benchmark \
   --threshold 0.980 \
   --transthreshold 0.980 \
   --eval_accumulation_steps 10 \
   $FP16_FLAG

rm -rf logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/12-yahoo/checkpoint*

# Free HF datasets cache and OS page cache between tasks
find /tmp -name "*.arrow" -mmin +5 -delete 2>/dev/null; find ~/.cache/huggingface/datasets -name "cache-*.arrow" -delete 2>/dev/null
sleep 5

if [ "$GPU_MODE" = "t4_2gpu" ]; then
    BSZ=2; GA=8; EVAL_BSZ=128
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=4; GA=8; EVAL_BSZ=128
else
    BSZ=8; GA=4; EVAL_BSZ=128
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/1-yelp/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/2-amazon/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/3-mnli/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/4-cb/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/5-copa/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/6-qqp/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/7-rte/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/8-imdb/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/9-sst2/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/10-dbpedia/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/11-agnews/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/12-yahoo/saved_weights \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --task_config_dir configs/gen_script_long_order3_t5_configs/multirc \
   --output_dir logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/13-multirc \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order3_t5_small_specroute_v2 \
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
   --evaluation_strategy epoch \
   --save_strategy epoch \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --training_bias 1.0 \
   --gen_data_dir CL_Benchmark \
   --threshold 0.980 \
   --transthreshold 0.980 \
   --eval_accumulation_steps 10 \
   $FP16_FLAG

rm -rf logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/13-multirc/checkpoint*

# Free HF datasets cache and OS page cache between tasks
find /tmp -name "*.arrow" -mmin +5 -delete 2>/dev/null; find ~/.cache/huggingface/datasets -name "cache-*.arrow" -delete 2>/dev/null
sleep 5

if [ "$GPU_MODE" = "t4_2gpu" ]; then
    BSZ=2; GA=8; EVAL_BSZ=128
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=4; GA=8; EVAL_BSZ=128
else
    BSZ=8; GA=4; EVAL_BSZ=128
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/1-yelp/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/2-amazon/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/3-mnli/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/4-cb/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/5-copa/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/6-qqp/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/7-rte/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/8-imdb/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/9-sst2/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/10-dbpedia/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/11-agnews/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/12-yahoo/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/13-multirc/saved_weights \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --task_config_dir configs/gen_script_long_order3_t5_configs/boolq \
   --output_dir logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/14-boolq \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order3_t5_small_specroute_v2 \
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
   --evaluation_strategy epoch \
   --save_strategy epoch \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --training_bias 1.0 \
   --gen_data_dir CL_Benchmark \
   --threshold 0.980 \
   --transthreshold 0.980 \
   --eval_accumulation_steps 10 \
   $FP16_FLAG

rm -rf logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/14-boolq/checkpoint*

# Free HF datasets cache and OS page cache between tasks
find /tmp -name "*.arrow" -mmin +5 -delete 2>/dev/null; find ~/.cache/huggingface/datasets -name "cache-*.arrow" -delete 2>/dev/null
sleep 5

if [ "$GPU_MODE" = "t4_2gpu" ]; then
    BSZ=2; GA=8; EVAL_BSZ=128
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=4; GA=8; EVAL_BSZ=128
else
    BSZ=8; GA=4; EVAL_BSZ=128
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --previous_lora_path logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/1-yelp/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/2-amazon/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/3-mnli/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/4-cb/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/5-copa/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/6-qqp/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/7-rte/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/8-imdb/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/9-sst2/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/10-dbpedia/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/11-agnews/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/12-yahoo/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/13-multirc/saved_weights,logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/14-boolq/saved_weights \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --task_config_dir configs/gen_script_long_order3_t5_configs/wic \
   --output_dir logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/15-wic \
   --per_device_train_batch_size $BSZ \
   --per_device_eval_batch_size $EVAL_BSZ \
   --gradient_accumulation_steps $GA \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order3_t5_small_specroute_v2 \
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
   --evaluation_strategy epoch \
   --save_strategy epoch \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --mlp_hidden_dim 100 \
   --model_name specroute \
   --training_bias 1.0 \
   --gen_data_dir CL_Benchmark \
   --threshold 0.980 \
   --transthreshold 0.980 \
   --eval_accumulation_steps 10 \
   $FP16_FLAG

rm -rf logs_and_outputs/gen_script_long_order3_t5_small_specroute_v2/outputs/15-wic/checkpoint*

# Free HF datasets cache and OS page cache between tasks
find /tmp -name "*.arrow" -mmin +5 -delete 2>/dev/null; find ~/.cache/huggingface/datasets -name "cache-*.arrow" -delete 2>/dev/null
sleep 5
