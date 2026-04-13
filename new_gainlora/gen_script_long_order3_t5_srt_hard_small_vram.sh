#!/bin/bash
#SBATCH -J srt_hard_small
#SBATCH -o srt_hard_small-%j.out
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 96:00:00
#SBATCH --mem 64G
#SBATCH --gres=gpu:rtx3090:1

# ================================================================
# LOW VRAM MODE — for RTX 3090 / ~8GB GPUs
#
# Strategy: batch_size=1 + gradient_accumulation_steps=32
#   → effective batch size = 32 (same as original bs=16,ga=2)
#   → VRAM: ~4.7GB (vs ~6GB original)
#   → Training speed: ~2x slower (32 micro-steps vs 2)
#   → Model quality: IDENTICAL (GA is mathematically equivalent)
# ================================================================

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
port=$(shuf -i25000-30000 -n1)

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -z "$GPU_MEM" ]; then
    echo "ERROR: No GPU detected!"
    GPU_MEM=24000; NUM_GPUS=1
fi

if [ "$GPU_MEM" -lt 20000 ]; then
    IS_LOW_VRAM=1; echo "[GPU] Low-VRAM mode (${GPU_MEM}MB) — using BS=1, GA=32"
else
    IS_LOW_VRAM=1; echo "[GPU] Low-VRAM fallback (${GPU_MEM}MB) — using BS=1, GA=32"
fi

# ── FIXED low-VRAM hyperparameters ──────────────────────────
#   bs=1  : keeps VRAM ~4.7GB (T5-Large)
#   ga=32 : effective_bs=32 (identical to original bs=16,ga=2)
#   eval_bs=16 : inference is memory-light (no optimizer/gradients)
#   --gradient_checkpointing : always ON (saves ~0.5GB activations)
# ───────────────────────────────────────────────────────────
BSZ=1
GA=32
EVAL_BSZ=16
GC_FLAG="--gradient_checkpointing"

echo "[LOW-VRAM] BSZ=$BSZ GA=$GA eff_bs=$((BSZ*GA)) eval_bs=$EVAL_BSZ $GC_FLAG"
echo "============================================================"
echo ""

# SRT hard mode
SRT_FLAGS="--use_srt_router --srt_metric_mode hard --srt_max_emb_samples 500 --srt_skip_forward"
echo "NOTE: --srt_skip_forward=True: embeddings loaded from disk"


# ── TASK 1: yelp ──────────────────────────────────────────
if [ -d "logs_and_outputs/long_order3_t5_srt_hard_small/outputs/1-yelp/saved_weights" ]; then
    echo "[SKIP] Task 1-yelp already complete"
else
CUDA_VISIBLE_DEVICES="${1:-0}" python src/run_t5.py \
    --do_train --do_predict --predict_with_generate \
    --model_name_or_path $2 \
    --data_dir CL_Benchmark \
    --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
    --task_config_dir configs/gen_script_long_order3_t5_configs/yelp \
    --output_dir logs_and_outputs/long_order3_t5_srt_hard_small/outputs/1-yelp \
    --per_device_train_batch_size $BSZ --per_device_eval_batch_size $EVAL_BSZ \
    --gradient_accumulation_steps $GA \
    --learning_rate 0.0003 --num_train_epochs 10 \
    --run_name long_order3_t5_srt_hard_small \
    --max_source_length 512 --max_target_length 50 --generation_max_length 50 \
    --add_task_name False --add_dataset_name False \
    $GC_FLAG \
    --overwrite_output_dir --overwrite_cache \
    --lr_scheduler_type constant --warmup_steps 0 \
    --logging_strategy steps --logging_steps 10 \
    --metric_for_best_model eval_exact_match_for_yelp \
    --evaluation_strategy steps --save_strategy best --save_total_limit 1 \
    --lora_r 8 --lora_alpha 32 --lora_dropout 0.0 \
    --add_instruction_replay --data_replay_freq -1 --replay_after_n_epoch 0 \
    --mlp_hidden_dim 100 --model_name gainlora_inflora \
    --threshold 0.995 --transthreshold 0.995 \
    $SRT_FLAGS

rm -rf logs_and_outputs/long_order3_t5_srt_hard_small/outputs/1-yelp/checkpoint*
sleep 5
fi

# ── TASK 2: amazon ──────────────────────────────────────────
if [ -d "logs_and_outputs/long_order3_t5_srt_hard_small/outputs/2-amazon/saved_weights" ]; then
    echo "[SKIP] Task 2-amazon already complete"
else
CUDA_VISIBLE_DEVICES="${1:-0}" python src/run_t5.py \
    --do_train --do_predict --predict_with_generate \
    --model_name_or_path $2 \
    --load_checkpoint_from logs_and_outputs/long_order3_t5_srt_hard_small/outputs/1-yelp/saved_weights/trans_input.pt \
    --previous_lora_path logs_and_outputs/long_order3_t5_srt_hard_small/outputs/1-yelp/saved_weights \
    --previous_prompt_key_path logs_and_outputs/long_order3_t5_srt_hard_small/outputs/1-yelp/saved_weights/prompts_keys_till_now.pt \
    --data_dir CL_Benchmark \
    --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
    --gen_data_dir generated_data/lora_gen_long_t5 \
    --task_config_dir configs/gen_script_long_order3_t5_configs/amazon \
    --output_dir logs_and_outputs/long_order3_t5_srt_hard_small/outputs/2-amazon \
    --per_device_train_batch_size $BSZ --per_device_eval_batch_size $EVAL_BSZ \
    --gradient_accumulation_steps $GA \
    --learning_rate 0.0003 --num_train_epochs 10 \
    --run_name long_order3_t5_srt_hard_small \
    --max_source_length 512 --max_target_length 50 --generation_max_length 50 \
    --add_task_name False --add_dataset_name False \
    $GC_FLAG \
    --overwrite_output_dir --overwrite_cache \
    --lr_scheduler_type constant --warmup_steps 0 \
    --logging_strategy steps --logging_steps 10 \
    --metric_for_best_model eval_exact_match_for_amazon \
    --evaluation_strategy steps --save_strategy best --save_total_limit 1 \
    --lora_r 8 --lora_alpha 32 --lora_dropout 0.0 \
    --data_replay_freq -1 --kl_ratio 0.1 --attn_temperature 1 \
    --mlp_hidden_dim 100 --model_name gainlora_inflora \
    --threshold 0.995 --transthreshold 0.995 \
    $SRT_FLAGS \
    --srt_load_path logs_and_outputs/long_order3_t5_srt_hard_small/outputs/1-yelp/saved_weights

rm -rf logs_and_outputs/long_order3_t5_srt_hard_small/outputs/2-amazon/checkpoint*
sleep 5
fi

# ── TASK 3: mnli ──────────────────────────────────────────
if [ -d "logs_and_outputs/long_order3_t5_srt_hard_small/outputs/3-mnli/saved_weights" ]; then
    echo "[SKIP] Task 3-mnli already complete"
else
CUDA_VISIBLE_DEVICES="${1:-0}" python src/run_t5.py \
    --do_train --do_predict --predict_with_generate \
    --model_name_or_path $2 \
    --load_checkpoint_from logs_and_outputs/long_order3_t5_srt_hard_small/outputs/2-amazon/saved_weights/trans_input.pt \
    --previous_lora_path logs_and_outputs/long_order3_t5_srt_hard_small/outputs/1-yelp/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/2-amazon/saved_weights \
    --previous_prompt_key_path logs_and_outputs/long_order3_t5_srt_hard_small/outputs/2-amazon/saved_weights/prompts_keys_till_now.pt \
    --data_dir CL_Benchmark \
    --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
    --gen_data_dir generated_data/lora_gen_long_t5 \
    --task_config_dir configs/gen_script_long_order3_t5_configs/mnli \
    --output_dir logs_and_outputs/long_order3_t5_srt_hard_small/outputs/3-mnli \
    --per_device_train_batch_size $BSZ --per_device_eval_batch_size $EVAL_BSZ \
    --gradient_accumulation_steps $GA \
    --learning_rate 0.0003 --num_train_epochs 10 \
    --run_name long_order3_t5_srt_hard_small \
    --max_source_length 512 --max_target_length 50 --generation_max_length 50 \
    --add_task_name False --add_dataset_name False \
    $GC_FLAG \
    --overwrite_output_dir --overwrite_cache \
    --lr_scheduler_type constant --warmup_steps 0 \
    --logging_strategy steps --logging_steps 10 \
    --metric_for_best_model eval_exact_match_for_mnli \
    --evaluation_strategy steps --save_strategy best --save_total_limit 1 \
    --lora_r 8 --lora_alpha 32 --lora_dropout 0.0 \
    --data_replay_freq -1 --kl_ratio 0.1 --attn_temperature 1 \
    --mlp_hidden_dim 100 --model_name gainlora_inflora \
    --threshold 0.995 --transthreshold 0.995 \
    $SRT_FLAGS \
    --srt_load_path logs_and_outputs/long_order3_t5_srt_hard_small/outputs/2-amazon/saved_weights

rm -rf logs_and_outputs/long_order3_t5_srt_hard_small/outputs/3-mnli/checkpoint*
sleep 5
fi

# ── TASK 4: cb ──────────────────────────────────────────
if [ -d "logs_and_outputs/long_order3_t5_srt_hard_small/outputs/4-cb/saved_weights" ]; then
    echo "[SKIP] Task 4-cb already complete"
else
CUDA_VISIBLE_DEVICES="${1:-0}" python src/run_t5.py \
    --do_train --do_predict --predict_with_generate \
    --model_name_or_path $2 \
    --load_checkpoint_from logs_and_outputs/long_order3_t5_srt_hard_small/outputs/3-mnli/saved_weights/trans_input.pt \
    --previous_lora_path logs_and_outputs/long_order3_t5_srt_hard_small/outputs/1-yelp/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/2-amazon/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/3-mnli/saved_weights \
    --previous_prompt_key_path logs_and_outputs/long_order3_t5_srt_hard_small/outputs/3-mnli/saved_weights/prompts_keys_till_now.pt \
    --data_dir CL_Benchmark \
    --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
    --gen_data_dir generated_data/lora_gen_long_t5 \
    --task_config_dir configs/gen_script_long_order3_t5_configs/cb \
    --output_dir logs_and_outputs/long_order3_t5_srt_hard_small/outputs/4-cb \
    --per_device_train_batch_size $BSZ --per_device_eval_batch_size $EVAL_BSZ \
    --gradient_accumulation_steps $GA \
    --learning_rate 0.0003 --num_train_epochs 10 \
    --run_name long_order3_t5_srt_hard_small \
    --max_source_length 512 --max_target_length 50 --generation_max_length 50 \
    --add_task_name False --add_dataset_name False \
    $GC_FLAG \
    --overwrite_output_dir --overwrite_cache \
    --lr_scheduler_type constant --warmup_steps 0 \
    --logging_strategy steps --logging_steps 10 \
    --metric_for_best_model eval_exact_match_for_cb \
    --evaluation_strategy steps --save_strategy best --save_total_limit 1 \
    --lora_r 8 --lora_alpha 32 --lora_dropout 0.0 \
    --data_replay_freq -1 --kl_ratio 0.1 --attn_temperature 1 \
    --mlp_hidden_dim 100 --model_name gainlora_inflora \
    --threshold 0.995 --transthreshold 0.995 \
    $SRT_FLAGS \
    --srt_load_path logs_and_outputs/long_order3_t5_srt_hard_small/outputs/3-mnli/saved_weights

rm -rf logs_and_outputs/long_order3_t5_srt_hard_small/outputs/4-cb/checkpoint*
sleep 5
fi

# ── TASK 5: copa ──────────────────────────────────────────
if [ -d "logs_and_outputs/long_order3_t5_srt_hard_small/outputs/5-copa/saved_weights" ]; then
    echo "[SKIP] Task 5-copa already complete"
else
CUDA_VISIBLE_DEVICES="${1:-0}" python src/run_t5.py \
    --do_train --do_predict --predict_with_generate \
    --model_name_or_path $2 \
    --load_checkpoint_from logs_and_outputs/long_order3_t5_srt_hard_small/outputs/4-cb/saved_weights/trans_input.pt \
    --previous_lora_path logs_and_outputs/long_order3_t5_srt_hard_small/outputs/1-yelp/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/2-amazon/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/3-mnli/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/4-cb/saved_weights \
    --previous_prompt_key_path logs_and_outputs/long_order3_t5_srt_hard_small/outputs/4-cb/saved_weights/prompts_keys_till_now.pt \
    --data_dir CL_Benchmark \
    --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
    --gen_data_dir generated_data/lora_gen_long_t5 \
    --task_config_dir configs/gen_script_long_order3_t5_configs/copa \
    --output_dir logs_and_outputs/long_order3_t5_srt_hard_small/outputs/5-copa \
    --per_device_train_batch_size $BSZ --per_device_eval_batch_size $EVAL_BSZ \
    --gradient_accumulation_steps $GA \
    --learning_rate 0.0003 --num_train_epochs 10 \
    --run_name long_order3_t5_srt_hard_small \
    --max_source_length 512 --max_target_length 50 --generation_max_length 50 \
    --add_task_name False --add_dataset_name False \
    $GC_FLAG \
    --overwrite_output_dir --overwrite_cache \
    --lr_scheduler_type constant --warmup_steps 0 \
    --logging_strategy steps --logging_steps 10 \
    --metric_for_best_model eval_exact_match_for_copa \
    --evaluation_strategy steps --save_strategy best --save_total_limit 1 \
    --lora_r 8 --lora_alpha 32 --lora_dropout 0.0 \
    --data_replay_freq -1 --kl_ratio 0.1 --attn_temperature 1 \
    --mlp_hidden_dim 100 --model_name gainlora_inflora \
    --threshold 0.995 --transthreshold 0.995 \
    $SRT_FLAGS \
    --srt_load_path logs_and_outputs/long_order3_t5_srt_hard_small/outputs/4-cb/saved_weights

rm -rf logs_and_outputs/long_order3_t5_srt_hard_small/outputs/5-copa/checkpoint*
sleep 5
fi

# ── TASK 6: qqp ──────────────────────────────────────────
if [ -d "logs_and_outputs/long_order3_t5_srt_hard_small/outputs/6-qqp/saved_weights" ]; then
    echo "[SKIP] Task 6-qqp already complete"
else
CUDA_VISIBLE_DEVICES="${1:-0}" python src/run_t5.py \
    --do_train --do_predict --predict_with_generate \
    --model_name_or_path $2 \
    --load_checkpoint_from logs_and_outputs/long_order3_t5_srt_hard_small/outputs/5-copa/saved_weights/trans_input.pt \
    --previous_lora_path logs_and_outputs/long_order3_t5_srt_hard_small/outputs/1-yelp/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/2-amazon/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/3-mnli/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/4-cb/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/5-copa/saved_weights \
    --previous_prompt_key_path logs_and_outputs/long_order3_t5_srt_hard_small/outputs/5-copa/saved_weights/prompts_keys_till_now.pt \
    --data_dir CL_Benchmark \
    --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
    --gen_data_dir generated_data/lora_gen_long_t5 \
    --task_config_dir configs/gen_script_long_order3_t5_configs/qqp \
    --output_dir logs_and_outputs/long_order3_t5_srt_hard_small/outputs/6-qqp \
    --per_device_train_batch_size $BSZ --per_device_eval_batch_size $EVAL_BSZ \
    --gradient_accumulation_steps $GA \
    --learning_rate 0.0003 --num_train_epochs 10 \
    --run_name long_order3_t5_srt_hard_small \
    --max_source_length 512 --max_target_length 50 --generation_max_length 50 \
    --add_task_name False --add_dataset_name False \
    $GC_FLAG \
    --overwrite_output_dir --overwrite_cache \
    --lr_scheduler_type constant --warmup_steps 0 \
    --logging_strategy steps --logging_steps 10 \
    --metric_for_best_model eval_exact_match_for_qqp \
    --evaluation_strategy steps --save_strategy best --save_total_limit 1 \
    --lora_r 8 --lora_alpha 32 --lora_dropout 0.0 \
    --data_replay_freq -1 --kl_ratio 0.1 --attn_temperature 1 \
    --mlp_hidden_dim 100 --model_name gainlora_inflora \
    --threshold 0.995 --transthreshold 0.995 \
    $SRT_FLAGS \
    --srt_load_path logs_and_outputs/long_order3_t5_srt_hard_small/outputs/5-copa/saved_weights

rm -rf logs_and_outputs/long_order3_t5_srt_hard_small/outputs/6-qqp/checkpoint*
sleep 5
fi

# ── TASK 7: rte ──────────────────────────────────────────
if [ -d "logs_and_outputs/long_order3_t5_srt_hard_small/outputs/7-rte/saved_weights" ]; then
    echo "[SKIP] Task 7-rte already complete"
else
CUDA_VISIBLE_DEVICES="${1:-0}" python src/run_t5.py \
    --do_train --do_predict --predict_with_generate \
    --model_name_or_path $2 \
    --load_checkpoint_from logs_and_outputs/long_order3_t5_srt_hard_small/outputs/6-qqp/saved_weights/trans_input.pt \
    --previous_lora_path logs_and_outputs/long_order3_t5_srt_hard_small/outputs/1-yelp/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/2-amazon/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/3-mnli/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/4-cb/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/5-copa/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/6-qqp/saved_weights \
    --previous_prompt_key_path logs_and_outputs/long_order3_t5_srt_hard_small/outputs/6-qqp/saved_weights/prompts_keys_till_now.pt \
    --data_dir CL_Benchmark \
    --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
    --gen_data_dir generated_data/lora_gen_long_t5 \
    --task_config_dir configs/gen_script_long_order3_t5_configs/rte \
    --output_dir logs_and_outputs/long_order3_t5_srt_hard_small/outputs/7-rte \
    --per_device_train_batch_size $BSZ --per_device_eval_batch_size $EVAL_BSZ \
    --gradient_accumulation_steps $GA \
    --learning_rate 0.0003 --num_train_epochs 10 \
    --run_name long_order3_t5_srt_hard_small \
    --max_source_length 512 --max_target_length 50 --generation_max_length 50 \
    --add_task_name False --add_dataset_name False \
    $GC_FLAG \
    --overwrite_output_dir --overwrite_cache \
    --lr_scheduler_type constant --warmup_steps 0 \
    --logging_strategy steps --logging_steps 10 \
    --metric_for_best_model eval_exact_match_for_rte \
    --evaluation_strategy steps --save_strategy best --save_total_limit 1 \
    --lora_r 8 --lora_alpha 32 --lora_dropout 0.0 \
    --data_replay_freq -1 --kl_ratio 0.1 --attn_temperature 1 \
    --mlp_hidden_dim 100 --model_name gainlora_inflora \
    --threshold 0.995 --transthreshold 0.995 \
    $SRT_FLAGS \
    --srt_load_path logs_and_outputs/long_order3_t5_srt_hard_small/outputs/6-qqp/saved_weights

rm -rf logs_and_outputs/long_order3_t5_srt_hard_small/outputs/7-rte/checkpoint*
sleep 5
fi

# ── TASK 8: imdb ──────────────────────────────────────────
if [ -d "logs_and_outputs/long_order3_t5_srt_hard_small/outputs/8-imdb/saved_weights" ]; then
    echo "[SKIP] Task 8-imdb already complete"
else
CUDA_VISIBLE_DEVICES="${1:-0}" python src/run_t5.py \
    --do_train --do_predict --predict_with_generate \
    --model_name_or_path $2 \
    --load_checkpoint_from logs_and_outputs/long_order3_t5_srt_hard_small/outputs/7-rte/saved_weights/trans_input.pt \
    --previous_lora_path logs_and_outputs/long_order3_t5_srt_hard_small/outputs/1-yelp/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/2-amazon/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/3-mnli/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/4-cb/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/5-copa/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/6-qqp/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/7-rte/saved_weights \
    --previous_prompt_key_path logs_and_outputs/long_order3_t5_srt_hard_small/outputs/7-rte/saved_weights/prompts_keys_till_now.pt \
    --data_dir CL_Benchmark \
    --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
    --gen_data_dir generated_data/lora_gen_long_t5 \
    --task_config_dir configs/gen_script_long_order3_t5_configs/imdb \
    --output_dir logs_and_outputs/long_order3_t5_srt_hard_small/outputs/8-imdb \
    --per_device_train_batch_size $BSZ --per_device_eval_batch_size $EVAL_BSZ \
    --gradient_accumulation_steps $GA \
    --learning_rate 0.0003 --num_train_epochs 10 \
    --run_name long_order3_t5_srt_hard_small \
    --max_source_length 512 --max_target_length 50 --generation_max_length 50 \
    --add_task_name False --add_dataset_name False \
    $GC_FLAG \
    --overwrite_output_dir --overwrite_cache \
    --lr_scheduler_type constant --warmup_steps 0 \
    --logging_strategy steps --logging_steps 10 \
    --metric_for_best_model eval_exact_match_for_imdb \
    --evaluation_strategy steps --save_strategy best --save_total_limit 1 \
    --lora_r 8 --lora_alpha 32 --lora_dropout 0.0 \
    --data_replay_freq -1 --kl_ratio 0.1 --attn_temperature 1 \
    --mlp_hidden_dim 100 --model_name gainlora_inflora \
    --threshold 0.995 --transthreshold 0.995 \
    $SRT_FLAGS \
    --srt_load_path logs_and_outputs/long_order3_t5_srt_hard_small/outputs/7-rte/saved_weights

rm -rf logs_and_outputs/long_order3_t5_srt_hard_small/outputs/8-imdb/checkpoint*
sleep 5
fi

# ── TASK 9: sst2 ──────────────────────────────────────────
if [ -d "logs_and_outputs/long_order3_t5_srt_hard_small/outputs/9-sst2/saved_weights" ]; then
    echo "[SKIP] Task 9-sst2 already complete"
else
CUDA_VISIBLE_DEVICES="${1:-0}" python src/run_t5.py \
    --do_train --do_predict --predict_with_generate \
    --model_name_or_path $2 \
    --load_checkpoint_from logs_and_outputs/long_order3_t5_srt_hard_small/outputs/8-imdb/saved_weights/trans_input.pt \
    --previous_lora_path logs_and_outputs/long_order3_t5_srt_hard_small/outputs/1-yelp/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/2-amazon/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/3-mnli/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/4-cb/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/5-copa/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/6-qqp/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/7-rte/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/8-imdb/saved_weights \
    --previous_prompt_key_path logs_and_outputs/long_order3_t5_srt_hard_small/outputs/8-imdb/saved_weights/prompts_keys_till_now.pt \
    --data_dir CL_Benchmark \
    --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
    --gen_data_dir generated_data/lora_gen_long_t5 \
    --task_config_dir configs/gen_script_long_order3_t5_configs/sst2 \
    --output_dir logs_and_outputs/long_order3_t5_srt_hard_small/outputs/9-sst2 \
    --per_device_train_batch_size $BSZ --per_device_eval_batch_size $EVAL_BSZ \
    --gradient_accumulation_steps $GA \
    --learning_rate 0.0003 --num_train_epochs 10 \
    --run_name long_order3_t5_srt_hard_small \
    --max_source_length 512 --max_target_length 50 --generation_max_length 50 \
    --add_task_name False --add_dataset_name False \
    $GC_FLAG \
    --overwrite_output_dir --overwrite_cache \
    --lr_scheduler_type constant --warmup_steps 0 \
    --logging_strategy steps --logging_steps 10 \
    --metric_for_best_model eval_exact_match_for_sst2 \
    --evaluation_strategy steps --save_strategy best --save_total_limit 1 \
    --lora_r 8 --lora_alpha 32 --lora_dropout 0.0 \
    --data_replay_freq -1 --kl_ratio 0.1 --attn_temperature 1 \
    --mlp_hidden_dim 100 --model_name gainlora_inflora \
    --threshold 0.995 --transthreshold 0.995 \
    $SRT_FLAGS \
    --srt_load_path logs_and_outputs/long_order3_t5_srt_hard_small/outputs/8-imdb/saved_weights

rm -rf logs_and_outputs/long_order3_t5_srt_hard_small/outputs/9-sst2/checkpoint*
sleep 5
fi

# ── TASK 10: dbpedia ──────────────────────────────────────────
if [ -d "logs_and_outputs/long_order3_t5_srt_hard_small/outputs/10-dbpedia/saved_weights" ]; then
    echo "[SKIP] Task 10-dbpedia already complete"
else
CUDA_VISIBLE_DEVICES="${1:-0}" python src/run_t5.py \
    --do_train --do_predict --predict_with_generate \
    --model_name_or_path $2 \
    --load_checkpoint_from logs_and_outputs/long_order3_t5_srt_hard_small/outputs/9-sst2/saved_weights/trans_input.pt \
    --previous_lora_path logs_and_outputs/long_order3_t5_srt_hard_small/outputs/1-yelp/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/2-amazon/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/3-mnli/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/4-cb/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/5-copa/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/6-qqp/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/7-rte/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/8-imdb/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/9-sst2/saved_weights \
    --previous_prompt_key_path logs_and_outputs/long_order3_t5_srt_hard_small/outputs/9-sst2/saved_weights/prompts_keys_till_now.pt \
    --data_dir CL_Benchmark \
    --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
    --gen_data_dir generated_data/lora_gen_long_t5 \
    --task_config_dir configs/gen_script_long_order3_t5_configs/dbpedia \
    --output_dir logs_and_outputs/long_order3_t5_srt_hard_small/outputs/10-dbpedia \
    --per_device_train_batch_size $BSZ --per_device_eval_batch_size $EVAL_BSZ \
    --gradient_accumulation_steps $GA \
    --learning_rate 0.0003 --num_train_epochs 10 \
    --run_name long_order3_t5_srt_hard_small \
    --max_source_length 512 --max_target_length 50 --generation_max_length 50 \
    --add_task_name False --add_dataset_name False \
    $GC_FLAG \
    --overwrite_output_dir --overwrite_cache \
    --lr_scheduler_type constant --warmup_steps 0 \
    --logging_strategy steps --logging_steps 10 \
    --metric_for_best_model eval_exact_match_for_dbpedia \
    --evaluation_strategy steps --save_strategy best --save_total_limit 1 \
    --lora_r 8 --lora_alpha 32 --lora_dropout 0.0 \
    --data_replay_freq -1 --kl_ratio 0.1 --attn_temperature 1 \
    --mlp_hidden_dim 100 --model_name gainlora_inflora \
    --threshold 0.995 --transthreshold 0.995 \
    $SRT_FLAGS \
    --srt_load_path logs_and_outputs/long_order3_t5_srt_hard_small/outputs/9-sst2/saved_weights

rm -rf logs_and_outputs/long_order3_t5_srt_hard_small/outputs/10-dbpedia/checkpoint*
sleep 5
fi

# ── TASK 11: agnews ──────────────────────────────────────────
if [ -d "logs_and_outputs/long_order3_t5_srt_hard_small/outputs/11-agnews/saved_weights" ]; then
    echo "[SKIP] Task 11-agnews already complete"
else
CUDA_VISIBLE_DEVICES="${1:-0}" python src/run_t5.py \
    --do_train --do_predict --predict_with_generate \
    --model_name_or_path $2 \
    --load_checkpoint_from logs_and_outputs/long_order3_t5_srt_hard_small/outputs/10-dbpedia/saved_weights/trans_input.pt \
    --previous_lora_path logs_and_outputs/long_order3_t5_srt_hard_small/outputs/1-yelp/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/2-amazon/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/3-mnli/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/4-cb/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/5-copa/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/6-qqp/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/7-rte/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/8-imdb/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/9-sst2/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/10-dbpedia/saved_weights \
    --previous_prompt_key_path logs_and_outputs/long_order3_t5_srt_hard_small/outputs/10-dbpedia/saved_weights/prompts_keys_till_now.pt \
    --data_dir CL_Benchmark \
    --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
    --gen_data_dir generated_data/lora_gen_long_t5 \
    --task_config_dir configs/gen_script_long_order3_t5_configs/agnews \
    --output_dir logs_and_outputs/long_order3_t5_srt_hard_small/outputs/11-agnews \
    --per_device_train_batch_size $BSZ --per_device_eval_batch_size $EVAL_BSZ \
    --gradient_accumulation_steps $GA \
    --learning_rate 0.0003 --num_train_epochs 10 \
    --run_name long_order3_t5_srt_hard_small \
    --max_source_length 512 --max_target_length 50 --generation_max_length 50 \
    --add_task_name False --add_dataset_name False \
    $GC_FLAG \
    --overwrite_output_dir --overwrite_cache \
    --lr_scheduler_type constant --warmup_steps 0 \
    --logging_strategy steps --logging_steps 10 \
    --metric_for_best_model eval_exact_match_for_agnews \
    --evaluation_strategy steps --save_strategy best --save_total_limit 1 \
    --lora_r 8 --lora_alpha 32 --lora_dropout 0.0 \
    --data_replay_freq -1 --kl_ratio 0.1 --attn_temperature 1 \
    --mlp_hidden_dim 100 --model_name gainlora_inflora \
    --threshold 0.995 --transthreshold 0.995 \
    $SRT_FLAGS \
    --srt_load_path logs_and_outputs/long_order3_t5_srt_hard_small/outputs/10-dbpedia/saved_weights

rm -rf logs_and_outputs/long_order3_t5_srt_hard_small/outputs/11-agnews/checkpoint*
sleep 5
fi

# ── TASK 12: yahoo ──────────────────────────────────────────
if [ -d "logs_and_outputs/long_order3_t5_srt_hard_small/outputs/12-yahoo/saved_weights" ]; then
    echo "[SKIP] Task 12-yahoo already complete"
else
CUDA_VISIBLE_DEVICES="${1:-0}" python src/run_t5.py \
    --do_train --do_predict --predict_with_generate \
    --model_name_or_path $2 \
    --load_checkpoint_from logs_and_outputs/long_order3_t5_srt_hard_small/outputs/11-agnews/saved_weights/trans_input.pt \
    --previous_lora_path logs_and_outputs/long_order3_t5_srt_hard_small/outputs/1-yelp/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/2-amazon/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/3-mnli/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/4-cb/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/5-copa/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/6-qqp/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/7-rte/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/8-imdb/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/9-sst2/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/10-dbpedia/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/11-agnews/saved_weights \
    --previous_prompt_key_path logs_and_outputs/long_order3_t5_srt_hard_small/outputs/11-agnews/saved_weights/prompts_keys_till_now.pt \
    --data_dir CL_Benchmark \
    --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
    --gen_data_dir generated_data/lora_gen_long_t5 \
    --task_config_dir configs/gen_script_long_order3_t5_configs/yahoo \
    --output_dir logs_and_outputs/long_order3_t5_srt_hard_small/outputs/12-yahoo \
    --per_device_train_batch_size $BSZ --per_device_eval_batch_size $EVAL_BSZ \
    --gradient_accumulation_steps $GA \
    --learning_rate 0.0003 --num_train_epochs 10 \
    --run_name long_order3_t5_srt_hard_small \
    --max_source_length 512 --max_target_length 50 --generation_max_length 50 \
    --add_task_name False --add_dataset_name False \
    $GC_FLAG \
    --overwrite_output_dir --overwrite_cache \
    --lr_scheduler_type constant --warmup_steps 0 \
    --logging_strategy steps --logging_steps 10 \
    --metric_for_best_model eval_exact_match_for_yahoo \
    --evaluation_strategy steps --save_strategy best --save_total_limit 1 \
    --lora_r 8 --lora_alpha 32 --lora_dropout 0.0 \
    --data_replay_freq -1 --kl_ratio 0.1 --attn_temperature 1 \
    --mlp_hidden_dim 100 --model_name gainlora_inflora \
    --threshold 0.995 --transthreshold 0.995 \
    $SRT_FLAGS \
    --srt_load_path logs_and_outputs/long_order3_t5_srt_hard_small/outputs/11-agnews/saved_weights

rm -rf logs_and_outputs/long_order3_t5_srt_hard_small/outputs/12-yahoo/checkpoint*
sleep 5
fi

# ── TASK 13: multirc ──────────────────────────────────────────
if [ -d "logs_and_outputs/long_order3_t5_srt_hard_small/outputs/13-multirc/saved_weights" ]; then
    echo "[SKIP] Task 13-multirc already complete"
else
CUDA_VISIBLE_DEVICES="${1:-0}" python src/run_t5.py \
    --do_train --do_predict --predict_with_generate \
    --model_name_or_path $2 \
    --load_checkpoint_from logs_and_outputs/long_order3_t5_srt_hard_small/outputs/12-yahoo/saved_weights/trans_input.pt \
    --previous_lora_path logs_and_outputs/long_order3_t5_srt_hard_small/outputs/1-yelp/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/2-amazon/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/3-mnli/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/4-cb/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/5-copa/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/6-qqp/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/7-rte/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/8-imdb/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/9-sst2/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/10-dbpedia/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/11-agnews/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/12-yahoo/saved_weights \
    --previous_prompt_key_path logs_and_outputs/long_order3_t5_srt_hard_small/outputs/12-yahoo/saved_weights/prompts_keys_till_now.pt \
    --data_dir CL_Benchmark \
    --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
    --gen_data_dir generated_data/lora_gen_long_t5 \
    --task_config_dir configs/gen_script_long_order3_t5_configs/multirc \
    --output_dir logs_and_outputs/long_order3_t5_srt_hard_small/outputs/13-multirc \
    --per_device_train_batch_size $BSZ --per_device_eval_batch_size $EVAL_BSZ \
    --gradient_accumulation_steps $GA \
    --learning_rate 0.0003 --num_train_epochs 10 \
    --run_name long_order3_t5_srt_hard_small \
    --max_source_length 512 --max_target_length 50 --generation_max_length 50 \
    --add_task_name False --add_dataset_name False \
    $GC_FLAG \
    --overwrite_output_dir --overwrite_cache \
    --lr_scheduler_type constant --warmup_steps 0 \
    --logging_strategy steps --logging_steps 10 \
    --metric_for_best_model eval_exact_match_for_multirc \
    --evaluation_strategy steps --save_strategy best --save_total_limit 1 \
    --lora_r 8 --lora_alpha 32 --lora_dropout 0.0 \
    --data_replay_freq -1 --kl_ratio 0.1 --attn_temperature 1 \
    --mlp_hidden_dim 100 --model_name gainlora_inflora \
    --threshold 0.995 --transthreshold 0.995 \
    $SRT_FLAGS \
    --srt_load_path logs_and_outputs/long_order3_t5_srt_hard_small/outputs/12-yahoo/saved_weights

rm -rf logs_and_outputs/long_order3_t5_srt_hard_small/outputs/13-multirc/checkpoint*
sleep 5
fi

# ── TASK 14: boolq ──────────────────────────────────────────
if [ -d "logs_and_outputs/long_order3_t5_srt_hard_small/outputs/14-boolq/saved_weights" ]; then
    echo "[SKIP] Task 14-boolq already complete"
else
CUDA_VISIBLE_DEVICES="${1:-0}" python src/run_t5.py \
    --do_train --do_predict --predict_with_generate \
    --model_name_or_path $2 \
    --load_checkpoint_from logs_and_outputs/long_order3_t5_srt_hard_small/outputs/13-multirc/saved_weights/trans_input.pt \
    --previous_lora_path logs_and_outputs/long_order3_t5_srt_hard_small/outputs/1-yelp/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/2-amazon/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/3-mnli/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/4-cb/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/5-copa/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/6-qqp/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/7-rte/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/8-imdb/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/9-sst2/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/10-dbpedia/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/11-agnews/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/12-yahoo/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/13-multirc/saved_weights \
    --previous_prompt_key_path logs_and_outputs/long_order3_t5_srt_hard_small/outputs/13-multirc/saved_weights/prompts_keys_till_now.pt \
    --data_dir CL_Benchmark \
    --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
    --gen_data_dir generated_data/lora_gen_long_t5 \
    --task_config_dir configs/gen_script_long_order3_t5_configs/boolq \
    --output_dir logs_and_outputs/long_order3_t5_srt_hard_small/outputs/14-boolq \
    --per_device_train_batch_size $BSZ --per_device_eval_batch_size $EVAL_BSZ \
    --gradient_accumulation_steps $GA \
    --learning_rate 0.0003 --num_train_epochs 10 \
    --run_name long_order3_t5_srt_hard_small \
    --max_source_length 512 --max_target_length 50 --generation_max_length 50 \
    --add_task_name False --add_dataset_name False \
    $GC_FLAG \
    --overwrite_output_dir --overwrite_cache \
    --lr_scheduler_type constant --warmup_steps 0 \
    --logging_strategy steps --logging_steps 10 \
    --metric_for_best_model eval_exact_match_for_boolq \
    --evaluation_strategy steps --save_strategy best --save_total_limit 1 \
    --lora_r 8 --lora_alpha 32 --lora_dropout 0.0 \
    --data_replay_freq -1 --kl_ratio 0.1 --attn_temperature 1 \
    --mlp_hidden_dim 100 --model_name gainlora_inflora \
    --threshold 0.995 --transthreshold 0.995 \
    $SRT_FLAGS \
    --srt_load_path logs_and_outputs/long_order3_t5_srt_hard_small/outputs/13-multirc/saved_weights

rm -rf logs_and_outputs/long_order3_t5_srt_hard_small/outputs/14-boolq/checkpoint*
sleep 5
fi

# ── TASK 15: wic ──────────────────────────────────────────
if [ -d "logs_and_outputs/long_order3_t5_srt_hard_small/outputs/15-wic/saved_weights" ]; then
    echo "[SKIP] Task 15-wic already complete"
else
CUDA_VISIBLE_DEVICES="${1:-0}" python src/run_t5.py \
    --do_train --do_predict --predict_with_generate \
    --model_name_or_path $2 \
    --load_checkpoint_from logs_and_outputs/long_order3_t5_srt_hard_small/outputs/14-boolq/saved_weights/trans_input.pt \
    --previous_lora_path logs_and_outputs/long_order3_t5_srt_hard_small/outputs/1-yelp/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/2-amazon/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/3-mnli/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/4-cb/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/5-copa/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/6-qqp/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/7-rte/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/8-imdb/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/9-sst2/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/10-dbpedia/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/11-agnews/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/12-yahoo/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/13-multirc/saved_weights,logs_and_outputs/long_order3_t5_srt_hard_small/outputs/14-boolq/saved_weights \
    --previous_prompt_key_path logs_and_outputs/long_order3_t5_srt_hard_small/outputs/14-boolq/saved_weights/prompts_keys_till_now.pt \
    --data_dir CL_Benchmark \
    --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
    --gen_data_dir generated_data/lora_gen_long_t5 \
    --task_config_dir configs/gen_script_long_order3_t5_configs/wic \
    --output_dir logs_and_outputs/long_order3_t5_srt_hard_small/outputs/15-wic \
    --per_device_train_batch_size $BSZ --per_device_eval_batch_size $EVAL_BSZ \
    --gradient_accumulation_steps $GA \
    --learning_rate 0.0003 --num_train_epochs 10 \
    --run_name long_order3_t5_srt_hard_small \
    --max_source_length 512 --max_target_length 50 --generation_max_length 50 \
    --add_task_name False --add_dataset_name False \
    $GC_FLAG \
    --overwrite_output_dir --overwrite_cache \
    --lr_scheduler_type constant --warmup_steps 0 \
    --logging_strategy steps --logging_steps 10 \
    --metric_for_best_model eval_exact_match_for_wic \
    --evaluation_strategy steps --save_strategy best --save_total_limit 1 \
    --lora_r 8 --lora_alpha 32 --lora_dropout 0.0 \
    --data_replay_freq -1 --kl_ratio 0.1 --attn_temperature 1 \
    --mlp_hidden_dim 100 --model_name gainlora_inflora \
    --threshold 0.995 --transthreshold 0.995 \
    $SRT_FLAGS \
    --srt_load_path logs_and_outputs/long_order3_t5_srt_hard_small/outputs/14-boolq/saved_weights

rm -rf logs_and_outputs/long_order3_t5_srt_hard_small/outputs/15-wic/checkpoint*
sleep 5
fi

# ── FINAL: compute CL metrics ─────────────────────────────────
python score.py long_order3_t5_srt_hard_small long_order3_t5_srt_hard_small
