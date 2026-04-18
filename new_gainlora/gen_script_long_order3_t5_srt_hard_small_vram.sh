cl#!/bin/bash
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
# Trains ONLY tasks 13-15 (multirc, boolq, wic) of order 3.
#
# VRAM breakdown for task 13 (12 prev LoRA adapters loaded):
#   Backbone T5-Large BF16:              ~1.54 GB
#   Optimizer states (trainable params):  ~0.01 GB  (only lora_B + trans_input + prompt_key)
#   Gradients (trainable params):         ~0.004 GB
#   12 prev LoRA adapters (q+v, 24 layers):
#                                          ~0.096 GB  ← key: loaded CPU→GPU in agg_lora_states
#   Hidden states (GC ON, seq=512, BS=1, 24 layers):
#                                          ~0.384 GB  ← key: checkpoint recomputes activations
#   SRT embeddings (~500×768 FP32 × 15 tasks):
#                                          ~0.024 GB
#   KV cache inference (BS=16):           ~0.3 GB
#   ─────────────────────────────────────────────
#   TOTAL ESTIMATED:                      ~2.9 GB  ✓ well under 7.2 GB budget
#
# Strategy: batch_size=1 + gradient_accumulation_steps=32
#   → effective batch size = 32 (same as original bs=16, ga=2)
#   → Model quality: IDENTICAL (gradient accumulation is mathematically equivalent)
#   → Training time: ~16x slower per step (32 vs 2 micro-steps), acceptable tradeoff
# ================================================================

set -e

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
#   bs=1  : minimizes activations (seq=512 → ~0.4GB hidden states with GC)
#   ga=32 : effective_bs=32 (identical to original bs=16,ga=2)
#   eval_bs=16 : inference memory is lighter (no optimizer/gradients)
#   --gradient_checkpointing : always ON (saves ~0.5GB by recomputing activations)
# ───────────────────────────────────────────────────────────
BSZ=1
GA=32
EVAL_BSZ=16
GC_FLAG="--gradient_checkpointing"

echo "[LOW-VRAM] BSZ=$BSZ GA=$GA eff_bs=$((BSZ*GA)) eval_bs=$EVAL_BSZ $GC_FLAG"
echo "[VRAM BREAKDOWN] backbone=1.54GB + optim+grad=0.014GB + 12×prev_lora=0.096GB + activations=0.384GB + srt_emb=0.024GB ≈ 3.0GB"
echo "============================================================"
echo ""

# SRT hard mode
SRT_FLAGS="--use_srt_router --srt_metric_mode hard --srt_max_emb_samples 500 --srt_skip_forward"
echo "NOTE: --srt_skip_forward=True: SRT embeddings loaded from disk (no forward pass needed)"


# ── TASK 13: multirc ──────────────────────────────────────────
# previous_lora_path: 12 adapters from tasks 1-12
# srt_load_path: task 12 signatures
# load_checkpoint_from: trans_input from task 12
# output_dir: SAME as original run to reuse existing outputs
if [ -d "logs_and_outputs/long_order3_t5_srt_hard/outputs/13-multirc/saved_weights" ]; then
    echo "[SKIP] Task 13-multirc already complete"
else
CUDA_VISIBLE_DEVICES="${1:-0}" python src/run_t5.py \
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
    --gradient_accumulation_steps $GA \
    --learning_rate 0.0003 --num_train_epochs 10 \
    --run_name long_order3_t5_srt_hard \
    --max_source_length 512 --max_target_length 50 --generation_max_length 50 \
    --add_task_name False --add_dataset_name False \
    $GC_FLAG \
    --overwrite_output_dir --overwrite_cache \
    --lr_scheduler_type constant --warmup_steps 0 \
    --logging_strategy steps --logging_steps 10 \
    --metric_for_best_model eval_exact_match_for_multirc \
    --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \
    --lora_r 8 --lora_alpha 32 --lora_dropout 0.0 \
    --data_replay_freq -1 --kl_ratio 0.1 --attn_temperature 1 \
    --mlp_hidden_dim 100 --model_name gainlora \
    --threshold 0.995 --transthreshold 0.995 \
    $SRT_FLAGS \
    --srt_load_path logs_and_outputs/long_order3_t5_srt_hard/outputs/12-yahoo/saved_weights

rm -rf logs_and_outputs/long_order3_t5_srt_hard/outputs/13-multirc/checkpoint*
sleep 5
fi

# ── TASK 14: boolq ──────────────────────────────────────────
# previous_lora_path: 13 adapters from tasks 1-13
# srt_load_path: task 13 signatures
# load_checkpoint_from: trans_input from task 13
if [ -d "logs_and_outputs/long_order3_t5_srt_hard/outputs/14-boolq/saved_weights" ]; then
    echo "[SKIP] Task 14-boolq already complete"
else
CUDA_VISIBLE_DEVICES="${1:-0}" python src/run_t5.py \
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
    --gradient_accumulation_steps $GA \
    --learning_rate 0.0003 --num_train_epochs 10 \
    --run_name long_order3_t5_srt_hard \
    --max_source_length 512 --max_target_length 50 --generation_max_length 50 \
    --add_task_name False --add_dataset_name False \
    $GC_FLAG \
    --overwrite_output_dir --overwrite_cache \
    --lr_scheduler_type constant --warmup_steps 0 \
    --logging_strategy steps --logging_steps 10 \
    --metric_for_best_model eval_exact_match_for_boolq \
    --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \
    --lora_r 8 --lora_alpha 32 --lora_dropout 0.0 \
    --data_replay_freq -1 --kl_ratio 0.1 --attn_temperature 1 \
    --mlp_hidden_dim 100 --model_name gainlora \
    --threshold 0.995 --transthreshold 0.995 \
    $SRT_FLAGS \
    --srt_load_path logs_and_outputs/long_order3_t5_srt_hard/outputs/13-multirc/saved_weights

rm -rf logs_and_outputs/long_order3_t5_srt_hard/outputs/14-boolq/checkpoint*
sleep 5
fi

# ── TASK 15: wic ──────────────────────────────────────────
# previous_lora_path: 14 adapters from tasks 1-14
# srt_load_path: task 14 signatures
# load_checkpoint_from: trans_input from task 14
if [ -d "logs_and_outputs/long_order3_t5_srt_hard/outputs/15-wic/saved_weights" ]; then
    echo "[SKIP] Task 15-wic already complete"
else
CUDA_VISIBLE_DEVICES="${1:-0}" python src/run_t5.py \
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
    --gradient_accumulation_steps $GA \
    --learning_rate 0.0003 --num_train_epochs 10 \
    --run_name long_order3_t5_srt_hard \
    --max_source_length 512 --max_target_length 50 --generation_max_length 50 \
    --add_task_name False --add_dataset_name False \
    $GC_FLAG \
    --overwrite_output_dir --overwrite_cache \
    --lr_scheduler_type constant --warmup_steps 0 \
    --logging_strategy steps --logging_steps 10 \
    --metric_for_best_model eval_exact_match_for_wic \
    --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \
    --lora_r 8 --lora_alpha 32 --lora_dropout 0.0 \
    --data_replay_freq -1 --kl_ratio 0.1 --attn_temperature 1 \
    --mlp_hidden_dim 100 --model_name gainlora \
    --threshold 0.995 --transthreshold 0.995 \
    $SRT_FLAGS \
    --srt_load_path logs_and_outputs/long_order3_t5_srt_hard/outputs/14-boolq/saved_weights

rm -rf logs_and_outputs/long_order3_t5_srt_hard/outputs/15-wic/checkpoint*
sleep 5
fi

# ── FINAL: compute CL metrics ─────────────────────────────────
# Re-evaluates ALL 15 tasks using the newly completed tasks 13-15
# alongside existing results from tasks 1-12
python score.py long_order3_t5_srt_hard long_order3_t5_srt_hard
