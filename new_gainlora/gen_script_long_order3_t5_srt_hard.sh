#!/bin/bash
#SBATCH -J srt_hard
#SBATCH -o srt_hard-%j.out
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 20:00:00
#SBATCH --mem 128G
#SBATCH --gres=gpu:a100-sxm4-80gb:1

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
port=$(shuf -i25000-30000 -n1)

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -z "$GPU_MEM" ]; then
    echo "ERROR: No GPU detected!"
    GPU_MEM=16000; NUM_GPUS=1
fi

if [ "$GPU_MEM" -lt 20000 ]; then
    IS_T4=1; echo "[GPU] Detected T4 GPUs (${GPU_MEM}MB)"
else
    IS_T4=0; echo "[GPU] Detected high-memory GPUs (${GPU_MEM}MB)"
fi

if [ "$IS_T4" -eq 1 ] && [ "$NUM_GPUS" -ge 2 ]; then
    GPU_MODE="t4_2gpu"; GPU_IDS="0,1"; FP16_FLAG="--gradient_checkpointing"
elif [ "$IS_T4" -eq 1 ]; then
    GPU_MODE="t4_1gpu"; GPU_IDS="${1:-0}"; FP16_FLAG="--gradient_checkpointing"
else
    GPU_MODE="a100"; GPU_IDS="${1:-0}"; FP16_FLAG=""
fi

echo "[GPU] CUDA_VISIBLE_DEVICES=$GPU_IDS, mode=$GPU_MODE"
echo "============================================================"
echo ""

# SRT hard mode: ZCA whitening + L2 (matches routing_analysis experiment)
SRT_FLAGS="--use_srt_router --srt_metric_mode hard --srt_max_emb_samples 500"


# ── TASK 1: yelp ──────────────────────────────────────────
if [ -d "logs_and_outputs/long_order3_t5_srt_hard/outputs/1-yelp/saved_weights" ]; then
    echo "[SKIP] Task 1-yelp already complete"
else
if [ "$GPU_MODE" = "t4_2gpu" ]; then
    BSZ=16; GA=2; EVAL_BSZ=128
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=16; GA=2; EVAL_BSZ=128
else
    BSZ=16; GA=2; EVAL_BSZ=128
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
    --do_train --do_predict --predict_with_generate \
    --model_name_or_path $2 \
    --data_dir CL_Benchmark \
    --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
    --task_config_dir configs/gen_script_long_order3_t5_configs/yelp \
    --output_dir logs_and_outputs/long_order3_t5_srt_hard/outputs/1-yelp \
    --per_device_train_batch_size $BSZ --per_device_eval_batch_size $EVAL_BSZ \
    --gradient_accumulation_steps $GA --learning_rate 0.0003 --num_train_epochs 10 \
    --run_name long_order3_t5_srt_hard \
    --max_source_length 512 --max_target_length 50 --generation_max_length 50 \
    --add_task_name False --add_dataset_name False --gradient_checkpointing \
    --overwrite_output_dir --overwrite_cache \
    --lr_scheduler_type constant --warmup_steps 0 \
    --logging_strategy steps --logging_steps 10 \
    --metric_for_best_model eval_exact_match_for_yelp \
    --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \
    --load_best_model_at_end \
    --lora_r 8 --lora_alpha 32 --lora_dropout 0.0 \
    --add_instruction_replay --data_replay_freq -1 --replay_after_n_epoch 0 \
    --mlp_hidden_dim 100 --model_name gainlora_inflora \
    --threshold 0.995 --transthreshold 0.995 \
    $FP16_FLAG $SRT_FLAGS

rm -rf logs_and_outputs/long_order3_t5_srt_hard/outputs/1-yelp/checkpoint*
sleep 5
fi
# ── TASK 2: amazon ──────────────────────────────────────────
if [ -d "logs_and_outputs/long_order3_t5_srt_hard/outputs/2-amazon/saved_weights" ]; then
    echo "[SKIP] Task 2-amazon already complete"
else
if [ "$GPU_MODE" = "t4_2gpu" ]; then
    BSZ=16; GA=2; EVAL_BSZ=128
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=16; GA=2; EVAL_BSZ=128
else
    BSZ=16; GA=2; EVAL_BSZ=128
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
    --do_train --do_predict --predict_with_generate \
    --model_name_or_path $2 \
    --load_checkpoint_from logs_and_outputs/long_order3_t5_srt_hard/outputs/1-yelp/saved_weights/trans_input.pt \
    --previous_lora_path logs_and_outputs/long_order3_t5_srt_hard/outputs/1-yelp/saved_weights \
    --previous_prompt_key_path logs_and_outputs/long_order3_t5_srt_hard/outputs/1-yelp/saved_weights/prompts_keys_till_now.pt \
    --data_dir CL_Benchmark \
    --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
    --gen_data_dir generated_data/lora_gen_long_t5 \
    --task_config_dir configs/gen_script_long_order3_t5_configs/amazon \
    --output_dir logs_and_outputs/long_order3_t5_srt_hard/outputs/2-amazon \
    --per_device_train_batch_size $BSZ --per_device_eval_batch_size $EVAL_BSZ \
    --gradient_accumulation_steps $GA --learning_rate 0.0003 --num_train_epochs 10 \
    --run_name long_order3_t5_srt_hard \
    --max_source_length 512 --max_target_length 50 --generation_max_length 50 \
    --add_task_name False --add_dataset_name False --gradient_checkpointing \
    --overwrite_output_dir --overwrite_cache \
    --lr_scheduler_type constant --warmup_steps 0 \
    --logging_strategy steps --logging_steps 10 \
    --metric_for_best_model eval_exact_match_for_amazon \
    --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \
    --load_best_model_at_end \
    --lora_r 8 --lora_alpha 32 --lora_dropout 0.0 \
    --data_replay_freq -1 --kl_ratio 0.1 --attn_temperature 1 \
    --mlp_hidden_dim 100 --model_name gainlora_inflora \
    --threshold 0.995 --transthreshold 0.995 \
    $FP16_FLAG $SRT_FLAGS \
    --srt_load_path logs_and_outputs/long_order3_t5_srt_hard/outputs/1-yelp/saved_weights

rm -rf logs_and_outputs/long_order3_t5_srt_hard/outputs/2-amazon/checkpoint*
sleep 5
fi
# ── TASK 3: mnli ──────────────────────────────────────────
if [ -d "logs_and_outputs/long_order3_t5_srt_hard/outputs/3-mnli/saved_weights" ]; then
    echo "[SKIP] Task 3-mnli already complete"
else
if [ "$GPU_MODE" = "t4_2gpu" ]; then
    BSZ=16; GA=2; EVAL_BSZ=128
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=16; GA=2; EVAL_BSZ=128
else
    BSZ=16; GA=2; EVAL_BSZ=128
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
    --do_train --do_predict --predict_with_generate \
    --model_name_or_path $2 \
    --load_checkpoint_from logs_and_outputs/long_order3_t5_srt_hard/outputs/2-amazon/saved_weights/trans_input.pt \
    --previous_lora_path logs_and_outputs/long_order3_t5_srt_hard/outputs/1-yelp/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/2-amazon/saved_weights \
    --previous_prompt_key_path logs_and_outputs/long_order3_t5_srt_hard/outputs/2-amazon/saved_weights/prompts_keys_till_now.pt \
    --data_dir CL_Benchmark \
    --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
    --gen_data_dir generated_data/lora_gen_long_t5 \
    --task_config_dir configs/gen_script_long_order3_t5_configs/mnli \
    --output_dir logs_and_outputs/long_order3_t5_srt_hard/outputs/3-mnli \
    --per_device_train_batch_size $BSZ --per_device_eval_batch_size $EVAL_BSZ \
    --gradient_accumulation_steps $GA --learning_rate 0.0003 --num_train_epochs 10 \
    --run_name long_order3_t5_srt_hard \
    --max_source_length 512 --max_target_length 50 --generation_max_length 50 \
    --add_task_name False --add_dataset_name False --gradient_checkpointing \
    --overwrite_output_dir --overwrite_cache \
    --lr_scheduler_type constant --warmup_steps 0 \
    --logging_strategy steps --logging_steps 10 \
    --metric_for_best_model eval_exact_match_for_mnli \
    --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \
    --load_best_model_at_end \
    --lora_r 8 --lora_alpha 32 --lora_dropout 0.0 \
    --data_replay_freq -1 --kl_ratio 0.1 --attn_temperature 1 \
    --mlp_hidden_dim 100 --model_name gainlora_inflora \
    --threshold 0.995 --transthreshold 0.995 \
    $FP16_FLAG $SRT_FLAGS \
    --srt_load_path logs_and_outputs/long_order3_t5_srt_hard/outputs/2-amazon/saved_weights

rm -rf logs_and_outputs/long_order3_t5_srt_hard/outputs/3-mnli/checkpoint*
sleep 5
fi
# ── TASK 4: cb ──────────────────────────────────────────
if [ -d "logs_and_outputs/long_order3_t5_srt_hard/outputs/4-cb/saved_weights" ]; then
    echo "[SKIP] Task 4-cb already complete"
else
if [ "$GPU_MODE" = "t4_2gpu" ]; then
    BSZ=16; GA=2; EVAL_BSZ=128
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=16; GA=2; EVAL_BSZ=128
else
    BSZ=16; GA=2; EVAL_BSZ=128
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
    --do_train --do_predict --predict_with_generate \
    --model_name_or_path $2 \
    --load_checkpoint_from logs_and_outputs/long_order3_t5_srt_hard/outputs/3-mnli/saved_weights/trans_input.pt \
    --previous_lora_path logs_and_outputs/long_order3_t5_srt_hard/outputs/1-yelp/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/2-amazon/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/3-mnli/saved_weights \
    --previous_prompt_key_path logs_and_outputs/long_order3_t5_srt_hard/outputs/3-mnli/saved_weights/prompts_keys_till_now.pt \
    --data_dir CL_Benchmark \
    --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
    --gen_data_dir generated_data/lora_gen_long_t5 \
    --task_config_dir configs/gen_script_long_order3_t5_configs/cb \
    --output_dir logs_and_outputs/long_order3_t5_srt_hard/outputs/4-cb \
    --per_device_train_batch_size $BSZ --per_device_eval_batch_size $EVAL_BSZ \
    --gradient_accumulation_steps $GA --learning_rate 0.0003 --num_train_epochs 10 \
    --run_name long_order3_t5_srt_hard \
    --max_source_length 512 --max_target_length 50 --generation_max_length 50 \
    --add_task_name False --add_dataset_name False --gradient_checkpointing \
    --overwrite_output_dir --overwrite_cache \
    --lr_scheduler_type constant --warmup_steps 0 \
    --logging_strategy steps --logging_steps 10 \
    --metric_for_best_model eval_exact_match_for_cb \
    --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \
    --load_best_model_at_end \
    --lora_r 8 --lora_alpha 32 --lora_dropout 0.0 \
    --data_replay_freq -1 --kl_ratio 0.1 --attn_temperature 1 \
    --mlp_hidden_dim 100 --model_name gainlora_inflora \
    --threshold 0.995 --transthreshold 0.995 \
    $FP16_FLAG $SRT_FLAGS \
    --srt_load_path logs_and_outputs/long_order3_t5_srt_hard/outputs/3-mnli/saved_weights

rm -rf logs_and_outputs/long_order3_t5_srt_hard/outputs/4-cb/checkpoint*
sleep 5
fi
# ── TASK 5: copa ──────────────────────────────────────────
if [ -d "logs_and_outputs/long_order3_t5_srt_hard/outputs/5-copa/saved_weights" ]; then
    echo "[SKIP] Task 5-copa already complete"
else
if [ "$GPU_MODE" = "t4_2gpu" ]; then
    BSZ=16; GA=2; EVAL_BSZ=128
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=16; GA=2; EVAL_BSZ=128
else
    BSZ=16; GA=2; EVAL_BSZ=128
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
    --do_train --do_predict --predict_with_generate \
    --model_name_or_path $2 \
    --load_checkpoint_from logs_and_outputs/long_order3_t5_srt_hard/outputs/4-cb/saved_weights/trans_input.pt \
    --previous_lora_path logs_and_outputs/long_order3_t5_srt_hard/outputs/1-yelp/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/2-amazon/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/3-mnli/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/4-cb/saved_weights \
    --previous_prompt_key_path logs_and_outputs/long_order3_t5_srt_hard/outputs/4-cb/saved_weights/prompts_keys_till_now.pt \
    --data_dir CL_Benchmark \
    --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
    --gen_data_dir generated_data/lora_gen_long_t5 \
    --task_config_dir configs/gen_script_long_order3_t5_configs/copa \
    --output_dir logs_and_outputs/long_order3_t5_srt_hard/outputs/5-copa \
    --per_device_train_batch_size $BSZ --per_device_eval_batch_size $EVAL_BSZ \
    --gradient_accumulation_steps $GA --learning_rate 0.0003 --num_train_epochs 10 \
    --run_name long_order3_t5_srt_hard \
    --max_source_length 512 --max_target_length 50 --generation_max_length 50 \
    --add_task_name False --add_dataset_name False --gradient_checkpointing \
    --overwrite_output_dir --overwrite_cache \
    --lr_scheduler_type constant --warmup_steps 0 \
    --logging_strategy steps --logging_steps 10 \
    --metric_for_best_model eval_exact_match_for_copa \
    --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \
    --load_best_model_at_end \
    --lora_r 8 --lora_alpha 32 --lora_dropout 0.0 \
    --data_replay_freq -1 --kl_ratio 0.1 --attn_temperature 1 \
    --mlp_hidden_dim 100 --model_name gainlora_inflora \
    --threshold 0.995 --transthreshold 0.995 \
    $FP16_FLAG $SRT_FLAGS \
    --srt_load_path logs_and_outputs/long_order3_t5_srt_hard/outputs/4-cb/saved_weights

rm -rf logs_and_outputs/long_order3_t5_srt_hard/outputs/5-copa/checkpoint*
sleep 5
fi
# ── TASK 6: qqp ──────────────────────────────────────────
if [ -d "logs_and_outputs/long_order3_t5_srt_hard/outputs/6-qqp/saved_weights" ]; then
    echo "[SKIP] Task 6-qqp already complete"
else
if [ "$GPU_MODE" = "t4_2gpu" ]; then
    BSZ=16; GA=2; EVAL_BSZ=128
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=16; GA=2; EVAL_BSZ=128
else
    BSZ=16; GA=2; EVAL_BSZ=128
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
    --do_train --do_predict --predict_with_generate \
    --model_name_or_path $2 \
    --load_checkpoint_from logs_and_outputs/long_order3_t5_srt_hard/outputs/5-copa/saved_weights/trans_input.pt \
    --previous_lora_path logs_and_outputs/long_order3_t5_srt_hard/outputs/1-yelp/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/2-amazon/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/3-mnli/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/4-cb/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/5-copa/saved_weights \
    --previous_prompt_key_path logs_and_outputs/long_order3_t5_srt_hard/outputs/5-copa/saved_weights/prompts_keys_till_now.pt \
    --data_dir CL_Benchmark \
    --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
    --gen_data_dir generated_data/lora_gen_long_t5 \
    --task_config_dir configs/gen_script_long_order3_t5_configs/qqp \
    --output_dir logs_and_outputs/long_order3_t5_srt_hard/outputs/6-qqp \
    --per_device_train_batch_size $BSZ --per_device_eval_batch_size $EVAL_BSZ \
    --gradient_accumulation_steps $GA --learning_rate 0.0003 --num_train_epochs 10 \
    --run_name long_order3_t5_srt_hard \
    --max_source_length 512 --max_target_length 50 --generation_max_length 50 \
    --add_task_name False --add_dataset_name False --gradient_checkpointing \
    --overwrite_output_dir --overwrite_cache \
    --lr_scheduler_type constant --warmup_steps 0 \
    --logging_strategy steps --logging_steps 10 \
    --metric_for_best_model eval_exact_match_for_qqp \
    --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \
    --load_best_model_at_end \
    --lora_r 8 --lora_alpha 32 --lora_dropout 0.0 \
    --data_replay_freq -1 --kl_ratio 0.1 --attn_temperature 1 \
    --mlp_hidden_dim 100 --model_name gainlora_inflora \
    --threshold 0.995 --transthreshold 0.995 \
    $FP16_FLAG $SRT_FLAGS \
    --srt_load_path logs_and_outputs/long_order3_t5_srt_hard/outputs/5-copa/saved_weights

rm -rf logs_and_outputs/long_order3_t5_srt_hard/outputs/6-qqp/checkpoint*
sleep 5
fi
# ── TASK 7: rte ──────────────────────────────────────────
if [ -d "logs_and_outputs/long_order3_t5_srt_hard/outputs/7-rte/saved_weights" ]; then
    echo "[SKIP] Task 7-rte already complete"
else
if [ "$GPU_MODE" = "t4_2gpu" ]; then
    BSZ=16; GA=2; EVAL_BSZ=128
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=16; GA=2; EVAL_BSZ=128
else
    BSZ=16; GA=2; EVAL_BSZ=128
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
    --do_train --do_predict --predict_with_generate \
    --model_name_or_path $2 \
    --load_checkpoint_from logs_and_outputs/long_order3_t5_srt_hard/outputs/6-qqp/saved_weights/trans_input.pt \
    --previous_lora_path logs_and_outputs/long_order3_t5_srt_hard/outputs/1-yelp/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/2-amazon/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/3-mnli/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/4-cb/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/5-copa/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/6-qqp/saved_weights \
    --previous_prompt_key_path logs_and_outputs/long_order3_t5_srt_hard/outputs/6-qqp/saved_weights/prompts_keys_till_now.pt \
    --data_dir CL_Benchmark \
    --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
    --gen_data_dir generated_data/lora_gen_long_t5 \
    --task_config_dir configs/gen_script_long_order3_t5_configs/rte \
    --output_dir logs_and_outputs/long_order3_t5_srt_hard/outputs/7-rte \
    --per_device_train_batch_size $BSZ --per_device_eval_batch_size $EVAL_BSZ \
    --gradient_accumulation_steps $GA --learning_rate 0.0003 --num_train_epochs 10 \
    --run_name long_order3_t5_srt_hard \
    --max_source_length 512 --max_target_length 50 --generation_max_length 50 \
    --add_task_name False --add_dataset_name False --gradient_checkpointing \
    --overwrite_output_dir --overwrite_cache \
    --lr_scheduler_type constant --warmup_steps 0 \
    --logging_strategy steps --logging_steps 10 \
    --metric_for_best_model eval_exact_match_for_rte \
    --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \
    --load_best_model_at_end \
    --lora_r 8 --lora_alpha 32 --lora_dropout 0.0 \
    --data_replay_freq -1 --kl_ratio 0.1 --attn_temperature 1 \
    --mlp_hidden_dim 100 --model_name gainlora_inflora \
    --threshold 0.995 --transthreshold 0.995 \
    $FP16_FLAG $SRT_FLAGS \
    --srt_load_path logs_and_outputs/long_order3_t5_srt_hard/outputs/6-qqp/saved_weights

rm -rf logs_and_outputs/long_order3_t5_srt_hard/outputs/7-rte/checkpoint*
sleep 5
fi
# ── TASK 8: imdb ──────────────────────────────────────────
if [ -d "logs_and_outputs/long_order3_t5_srt_hard/outputs/8-imdb/saved_weights" ]; then
    echo "[SKIP] Task 8-imdb already complete"
else
if [ "$GPU_MODE" = "t4_2gpu" ]; then
    BSZ=16; GA=2; EVAL_BSZ=128
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=16; GA=2; EVAL_BSZ=128
else
    BSZ=16; GA=2; EVAL_BSZ=128
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
    --do_train --do_predict --predict_with_generate \
    --model_name_or_path $2 \
    --load_checkpoint_from logs_and_outputs/long_order3_t5_srt_hard/outputs/7-rte/saved_weights/trans_input.pt \
    --previous_lora_path logs_and_outputs/long_order3_t5_srt_hard/outputs/1-yelp/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/2-amazon/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/3-mnli/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/4-cb/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/5-copa/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/6-qqp/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/7-rte/saved_weights \
    --previous_prompt_key_path logs_and_outputs/long_order3_t5_srt_hard/outputs/7-rte/saved_weights/prompts_keys_till_now.pt \
    --data_dir CL_Benchmark \
    --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
    --gen_data_dir generated_data/lora_gen_long_t5 \
    --task_config_dir configs/gen_script_long_order3_t5_configs/imdb \
    --output_dir logs_and_outputs/long_order3_t5_srt_hard/outputs/8-imdb \
    --per_device_train_batch_size $BSZ --per_device_eval_batch_size $EVAL_BSZ \
    --gradient_accumulation_steps $GA --learning_rate 0.0003 --num_train_epochs 10 \
    --run_name long_order3_t5_srt_hard \
    --max_source_length 512 --max_target_length 50 --generation_max_length 50 \
    --add_task_name False --add_dataset_name False --gradient_checkpointing \
    --overwrite_output_dir --overwrite_cache \
    --lr_scheduler_type constant --warmup_steps 0 \
    --logging_strategy steps --logging_steps 10 \
    --metric_for_best_model eval_exact_match_for_imdb \
    --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \
    --load_best_model_at_end \
    --lora_r 8 --lora_alpha 32 --lora_dropout 0.0 \
    --data_replay_freq -1 --kl_ratio 0.1 --attn_temperature 1 \
    --mlp_hidden_dim 100 --model_name gainlora_inflora \
    --threshold 0.995 --transthreshold 0.995 \
    $FP16_FLAG $SRT_FLAGS \
    --srt_load_path logs_and_outputs/long_order3_t5_srt_hard/outputs/7-rte/saved_weights

rm -rf logs_and_outputs/long_order3_t5_srt_hard/outputs/8-imdb/checkpoint*
sleep 5
fi
# ── TASK 9: sst2 ──────────────────────────────────────────
if [ -d "logs_and_outputs/long_order3_t5_srt_hard/outputs/9-sst2/saved_weights" ]; then
    echo "[SKIP] Task 9-sst2 already complete"
else
if [ "$GPU_MODE" = "t4_2gpu" ]; then
    BSZ=16; GA=2; EVAL_BSZ=128
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=16; GA=2; EVAL_BSZ=128
else
    BSZ=16; GA=2; EVAL_BSZ=128
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
    --do_train --do_predict --predict_with_generate \
    --model_name_or_path $2 \
    --load_checkpoint_from logs_and_outputs/long_order3_t5_srt_hard/outputs/8-imdb/saved_weights/trans_input.pt \
    --previous_lora_path logs_and_outputs/long_order3_t5_srt_hard/outputs/1-yelp/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/2-amazon/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/3-mnli/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/4-cb/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/5-copa/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/6-qqp/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/7-rte/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/8-imdb/saved_weights \
    --previous_prompt_key_path logs_and_outputs/long_order3_t5_srt_hard/outputs/8-imdb/saved_weights/prompts_keys_till_now.pt \
    --data_dir CL_Benchmark \
    --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
    --gen_data_dir generated_data/lora_gen_long_t5 \
    --task_config_dir configs/gen_script_long_order3_t5_configs/sst2 \
    --output_dir logs_and_outputs/long_order3_t5_srt_hard/outputs/9-sst2 \
    --per_device_train_batch_size $BSZ --per_device_eval_batch_size $EVAL_BSZ \
    --gradient_accumulation_steps $GA --learning_rate 0.0003 --num_train_epochs 10 \
    --run_name long_order3_t5_srt_hard \
    --max_source_length 512 --max_target_length 50 --generation_max_length 50 \
    --add_task_name False --add_dataset_name False --gradient_checkpointing \
    --overwrite_output_dir --overwrite_cache \
    --lr_scheduler_type constant --warmup_steps 0 \
    --logging_strategy steps --logging_steps 10 \
    --metric_for_best_model eval_exact_match_for_sst2 \
    --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \
    --load_best_model_at_end \
    --lora_r 8 --lora_alpha 32 --lora_dropout 0.0 \
    --data_replay_freq -1 --kl_ratio 0.1 --attn_temperature 1 \
    --mlp_hidden_dim 100 --model_name gainlora_inflora \
    --threshold 0.995 --transthreshold 0.995 \
    $FP16_FLAG $SRT_FLAGS \
    --srt_load_path logs_and_outputs/long_order3_t5_srt_hard/outputs/8-imdb/saved_weights

rm -rf logs_and_outputs/long_order3_t5_srt_hard/outputs/9-sst2/checkpoint*
sleep 5
fi
# ── TASK 10: dbpedia ──────────────────────────────────────────
if [ -d "logs_and_outputs/long_order3_t5_srt_hard/outputs/10-dbpedia/saved_weights" ]; then
    echo "[SKIP] Task 10-dbpedia already complete"
else
if [ "$GPU_MODE" = "t4_2gpu" ]; then
    BSZ=16; GA=2; EVAL_BSZ=128
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=16; GA=2; EVAL_BSZ=128
else
    BSZ=16; GA=2; EVAL_BSZ=128
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
    --do_train --do_predict --predict_with_generate \
    --model_name_or_path $2 \
    --load_checkpoint_from logs_and_outputs/long_order3_t5_srt_hard/outputs/9-sst2/saved_weights/trans_input.pt \
    --previous_lora_path logs_and_outputs/long_order3_t5_srt_hard/outputs/1-yelp/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/2-amazon/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/3-mnli/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/4-cb/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/5-copa/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/6-qqp/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/7-rte/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/8-imdb/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/9-sst2/saved_weights \
    --previous_prompt_key_path logs_and_outputs/long_order3_t5_srt_hard/outputs/9-sst2/saved_weights/prompts_keys_till_now.pt \
    --data_dir CL_Benchmark \
    --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
    --gen_data_dir generated_data/lora_gen_long_t5 \
    --task_config_dir configs/gen_script_long_order3_t5_configs/dbpedia \
    --output_dir logs_and_outputs/long_order3_t5_srt_hard/outputs/10-dbpedia \
    --per_device_train_batch_size $BSZ --per_device_eval_batch_size $EVAL_BSZ \
    --gradient_accumulation_steps $GA --learning_rate 0.0003 --num_train_epochs 10 \
    --run_name long_order3_t5_srt_hard \
    --max_source_length 512 --max_target_length 50 --generation_max_length 50 \
    --add_task_name False --add_dataset_name False --gradient_checkpointing \
    --overwrite_output_dir --overwrite_cache \
    --lr_scheduler_type constant --warmup_steps 0 \
    --logging_strategy steps --logging_steps 10 \
    --metric_for_best_model eval_exact_match_for_dbpedia \
    --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \
    --load_best_model_at_end \
    --lora_r 8 --lora_alpha 32 --lora_dropout 0.0 \
    --data_replay_freq -1 --kl_ratio 0.1 --attn_temperature 1 \
    --mlp_hidden_dim 100 --model_name gainlora_inflora \
    --threshold 0.995 --transthreshold 0.995 \
    $FP16_FLAG $SRT_FLAGS \
    --srt_load_path logs_and_outputs/long_order3_t5_srt_hard/outputs/9-sst2/saved_weights

rm -rf logs_and_outputs/long_order3_t5_srt_hard/outputs/10-dbpedia/checkpoint*
sleep 5
fi
# ── TASK 11: agnews ──────────────────────────────────────────
if [ -d "logs_and_outputs/long_order3_t5_srt_hard/outputs/11-agnews/saved_weights" ]; then
    echo "[SKIP] Task 11-agnews already complete"
else
if [ "$GPU_MODE" = "t4_2gpu" ]; then
    BSZ=16; GA=2; EVAL_BSZ=128
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=16; GA=2; EVAL_BSZ=128
else
    BSZ=16; GA=2; EVAL_BSZ=128
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
    --do_train --do_predict --predict_with_generate \
    --model_name_or_path $2 \
    --load_checkpoint_from logs_and_outputs/long_order3_t5_srt_hard/outputs/10-dbpedia/saved_weights/trans_input.pt \
    --previous_lora_path logs_and_outputs/long_order3_t5_srt_hard/outputs/1-yelp/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/2-amazon/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/3-mnli/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/4-cb/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/5-copa/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/6-qqp/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/7-rte/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/8-imdb/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/9-sst2/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/10-dbpedia/saved_weights \
    --previous_prompt_key_path logs_and_outputs/long_order3_t5_srt_hard/outputs/10-dbpedia/saved_weights/prompts_keys_till_now.pt \
    --data_dir CL_Benchmark \
    --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
    --gen_data_dir generated_data/lora_gen_long_t5 \
    --task_config_dir configs/gen_script_long_order3_t5_configs/agnews \
    --output_dir logs_and_outputs/long_order3_t5_srt_hard/outputs/11-agnews \
    --per_device_train_batch_size $BSZ --per_device_eval_batch_size $EVAL_BSZ \
    --gradient_accumulation_steps $GA --learning_rate 0.0003 --num_train_epochs 10 \
    --run_name long_order3_t5_srt_hard \
    --max_source_length 512 --max_target_length 50 --generation_max_length 50 \
    --add_task_name False --add_dataset_name False --gradient_checkpointing \
    --overwrite_output_dir --overwrite_cache \
    --lr_scheduler_type constant --warmup_steps 0 \
    --logging_strategy steps --logging_steps 10 \
    --metric_for_best_model eval_exact_match_for_agnews \
    --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \
    --load_best_model_at_end \
    --lora_r 8 --lora_alpha 32 --lora_dropout 0.0 \
    --data_replay_freq -1 --kl_ratio 0.1 --attn_temperature 1 \
    --mlp_hidden_dim 100 --model_name gainlora_inflora \
    --threshold 0.995 --transthreshold 0.995 \
    $FP16_FLAG $SRT_FLAGS \
    --srt_load_path logs_and_outputs/long_order3_t5_srt_hard/outputs/10-dbpedia/saved_weights

rm -rf logs_and_outputs/long_order3_t5_srt_hard/outputs/11-agnews/checkpoint*
sleep 5
fi
# ── TASK 12: yahoo ──────────────────────────────────────────
if [ -d "logs_and_outputs/long_order3_t5_srt_hard/outputs/12-yahoo/saved_weights" ]; then
    echo "[SKIP] Task 12-yahoo already complete"
else
if [ "$GPU_MODE" = "t4_2gpu" ]; then
    BSZ=16; GA=2; EVAL_BSZ=128
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=16; GA=2; EVAL_BSZ=128
else
    BSZ=16; GA=2; EVAL_BSZ=128
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
    --do_train --do_predict --predict_with_generate \
    --model_name_or_path $2 \
    --load_checkpoint_from logs_and_outputs/long_order3_t5_srt_hard/outputs/11-agnews/saved_weights/trans_input.pt \
    --previous_lora_path logs_and_outputs/long_order3_t5_srt_hard/outputs/1-yelp/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/2-amazon/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/3-mnli/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/4-cb/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/5-copa/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/6-qqp/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/7-rte/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/8-imdb/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/9-sst2/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/10-dbpedia/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/11-agnews/saved_weights \
    --previous_prompt_key_path logs_and_outputs/long_order3_t5_srt_hard/outputs/11-agnews/saved_weights/prompts_keys_till_now.pt \
    --data_dir CL_Benchmark \
    --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
    --gen_data_dir generated_data/lora_gen_long_t5 \
    --task_config_dir configs/gen_script_long_order3_t5_configs/yahoo \
    --output_dir logs_and_outputs/long_order3_t5_srt_hard/outputs/12-yahoo \
    --per_device_train_batch_size $BSZ --per_device_eval_batch_size $EVAL_BSZ \
    --gradient_accumulation_steps $GA --learning_rate 0.0003 --num_train_epochs 10 \
    --run_name long_order3_t5_srt_hard \
    --max_source_length 512 --max_target_length 50 --generation_max_length 50 \
    --add_task_name False --add_dataset_name False --gradient_checkpointing \
    --overwrite_output_dir --overwrite_cache \
    --lr_scheduler_type constant --warmup_steps 0 \
    --logging_strategy steps --logging_steps 10 \
    --metric_for_best_model eval_exact_match_for_yahoo \
    --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \
    --load_best_model_at_end \
    --lora_r 8 --lora_alpha 32 --lora_dropout 0.0 \
    --data_replay_freq -1 --kl_ratio 0.1 --attn_temperature 1 \
    --mlp_hidden_dim 100 --model_name gainlora_inflora \
    --threshold 0.995 --transthreshold 0.995 \
    $FP16_FLAG $SRT_FLAGS \
    --srt_load_path logs_and_outputs/long_order3_t5_srt_hard/outputs/11-agnews/saved_weights

rm -rf logs_and_outputs/long_order3_t5_srt_hard/outputs/12-yahoo/checkpoint*
sleep 5
fi
# ── TASK 13: multirc ──────────────────────────────────────────
if [ -d "logs_and_outputs/long_order3_t5_srt_hard/outputs/13-multirc/saved_weights" ]; then
    echo "[SKIP] Task 13-multirc already complete"
else
if [ "$GPU_MODE" = "t4_2gpu" ]; then
    BSZ=16; GA=2; EVAL_BSZ=128
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=16; GA=2; EVAL_BSZ=128
else
    BSZ=16; GA=2; EVAL_BSZ=128
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
    --do_train --do_predict --predict_with_generate \
    --model_name_or_path $2 \
    --load_checkpoint_from logs_and_outputs/long_order3_t5_srt_hard/outputs/12-yahoo/saved_weights/trans_input.pt \
    --previous_lora_path logs_and_outputs/long_order3_t5_srt_hard/outputs/1-yelp/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/2-amazon/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/3-mnli/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/4-cb/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/5-copa/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/6-qqp/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/7-rte/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/8-imdb/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/9-sst2/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/10-dbpedia/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/11-agnews/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/12-yahoo/saved_weights \
    --previous_prompt_key_path logs_and_outputs/long_order3_t5_srt_hard/outputs/12-yahoo/saved_weights/prompts_keys_till_now.pt \
    --data_dir CL_Benchmark \
    --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
    --gen_data_dir generated_data/lora_gen_long_t5 \
    --task_config_dir configs/gen_script_long_order3_t5_configs/multirc \
    --output_dir logs_and_outputs/long_order3_t5_srt_hard/outputs/13-multirc \
    --per_device_train_batch_size $BSZ --per_device_eval_batch_size $EVAL_BSZ \
    --gradient_accumulation_steps $GA --learning_rate 0.0003 --num_train_epochs 10 \
    --run_name long_order3_t5_srt_hard \
    --max_source_length 512 --max_target_length 50 --generation_max_length 50 \
    --add_task_name False --add_dataset_name False --gradient_checkpointing \
    --overwrite_output_dir --overwrite_cache \
    --lr_scheduler_type constant --warmup_steps 0 \
    --logging_strategy steps --logging_steps 10 \
    --metric_for_best_model eval_exact_match_for_multirc \
    --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \
    --load_best_model_at_end \
    --lora_r 8 --lora_alpha 32 --lora_dropout 0.0 \
    --data_replay_freq -1 --kl_ratio 0.1 --attn_temperature 1 \
    --mlp_hidden_dim 100 --model_name gainlora_inflora \
    --threshold 0.995 --transthreshold 0.995 \
    $FP16_FLAG $SRT_FLAGS \
    --srt_load_path logs_and_outputs/long_order3_t5_srt_hard/outputs/12-yahoo/saved_weights

rm -rf logs_and_outputs/long_order3_t5_srt_hard/outputs/13-multirc/checkpoint*
sleep 5
fi
# ── TASK 14: boolq ──────────────────────────────────────────
if [ -d "logs_and_outputs/long_order3_t5_srt_hard/outputs/14-boolq/saved_weights" ]; then
    echo "[SKIP] Task 14-boolq already complete"
else
if [ "$GPU_MODE" = "t4_2gpu" ]; then
    BSZ=16; GA=2; EVAL_BSZ=128
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=16; GA=2; EVAL_BSZ=128
else
    BSZ=16; GA=2; EVAL_BSZ=128
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
    --do_train --do_predict --predict_with_generate \
    --model_name_or_path $2 \
    --load_checkpoint_from logs_and_outputs/long_order3_t5_srt_hard/outputs/13-multirc/saved_weights/trans_input.pt \
    --previous_lora_path logs_and_outputs/long_order3_t5_srt_hard/outputs/1-yelp/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/2-amazon/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/3-mnli/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/4-cb/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/5-copa/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/6-qqp/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/7-rte/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/8-imdb/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/9-sst2/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/10-dbpedia/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/11-agnews/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/12-yahoo/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/13-multirc/saved_weights \
    --previous_prompt_key_path logs_and_outputs/long_order3_t5_srt_hard/outputs/13-multirc/saved_weights/prompts_keys_till_now.pt \
    --data_dir CL_Benchmark \
    --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
    --gen_data_dir generated_data/lora_gen_long_t5 \
    --task_config_dir configs/gen_script_long_order3_t5_configs/boolq \
    --output_dir logs_and_outputs/long_order3_t5_srt_hard/outputs/14-boolq \
    --per_device_train_batch_size $BSZ --per_device_eval_batch_size $EVAL_BSZ \
    --gradient_accumulation_steps $GA --learning_rate 0.0003 --num_train_epochs 10 \
    --run_name long_order3_t5_srt_hard \
    --max_source_length 512 --max_target_length 50 --generation_max_length 50 \
    --add_task_name False --add_dataset_name False --gradient_checkpointing \
    --overwrite_output_dir --overwrite_cache \
    --lr_scheduler_type constant --warmup_steps 0 \
    --logging_strategy steps --logging_steps 10 \
    --metric_for_best_model eval_exact_match_for_boolq \
    --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \
    --load_best_model_at_end \
    --lora_r 8 --lora_alpha 32 --lora_dropout 0.0 \
    --data_replay_freq -1 --kl_ratio 0.1 --attn_temperature 1 \
    --mlp_hidden_dim 100 --model_name gainlora_inflora \
    --threshold 0.995 --transthreshold 0.995 \
    $FP16_FLAG $SRT_FLAGS \
    --srt_load_path logs_and_outputs/long_order3_t5_srt_hard/outputs/13-multirc/saved_weights

rm -rf logs_and_outputs/long_order3_t5_srt_hard/outputs/14-boolq/checkpoint*
sleep 5
fi
# ── TASK 15: wic ──────────────────────────────────────────
if [ -d "logs_and_outputs/long_order3_t5_srt_hard/outputs/15-wic/saved_weights" ]; then
    echo "[SKIP] Task 15-wic already complete"
else
if [ "$GPU_MODE" = "t4_2gpu" ]; then
    BSZ=16; GA=2; EVAL_BSZ=128
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=16; GA=2; EVAL_BSZ=128
else
    BSZ=16; GA=2; EVAL_BSZ=128
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
    --do_train --do_predict --predict_with_generate \
    --model_name_or_path $2 \
    --load_checkpoint_from logs_and_outputs/long_order3_t5_srt_hard/outputs/14-boolq/saved_weights/trans_input.pt \
    --previous_lora_path logs_and_outputs/long_order3_t5_srt_hard/outputs/1-yelp/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/2-amazon/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/3-mnli/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/4-cb/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/5-copa/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/6-qqp/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/7-rte/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/8-imdb/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/9-sst2/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/10-dbpedia/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/11-agnews/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/12-yahoo/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/13-multirc/saved_weights,logs_and_outputs/long_order3_t5_srt_hard/outputs/14-boolq/saved_weights \
    --previous_prompt_key_path logs_and_outputs/long_order3_t5_srt_hard/outputs/14-boolq/saved_weights/prompts_keys_till_now.pt \
    --data_dir CL_Benchmark \
    --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
    --gen_data_dir generated_data/lora_gen_long_t5 \
    --task_config_dir configs/gen_script_long_order3_t5_configs/wic \
    --output_dir logs_and_outputs/long_order3_t5_srt_hard/outputs/15-wic \
    --per_device_train_batch_size $BSZ --per_device_eval_batch_size $EVAL_BSZ \
    --gradient_accumulation_steps $GA --learning_rate 0.0003 --num_train_epochs 10 \
    --run_name long_order3_t5_srt_hard \
    --max_source_length 512 --max_target_length 50 --generation_max_length 50 \
    --add_task_name False --add_dataset_name False --gradient_checkpointing \
    --overwrite_output_dir --overwrite_cache \
    --lr_scheduler_type constant --warmup_steps 0 \
    --logging_strategy steps --logging_steps 10 \
    --metric_for_best_model eval_exact_match_for_wic \
    --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \
    --load_best_model_at_end \
    --lora_r 8 --lora_alpha 32 --lora_dropout 0.0 \
    --data_replay_freq -1 --kl_ratio 0.1 --attn_temperature 1 \
    --mlp_hidden_dim 100 --model_name gainlora_inflora \
    --threshold 0.995 --transthreshold 0.995 \
    $FP16_FLAG $SRT_FLAGS \
    --srt_load_path logs_and_outputs/long_order3_t5_srt_hard/outputs/14-boolq/saved_weights

rm -rf logs_and_outputs/long_order3_t5_srt_hard/outputs/15-wic/checkpoint*
sleep 5
fi
python score.py long_order3_t5_srt_hard long_order3_t5_srt_hard
