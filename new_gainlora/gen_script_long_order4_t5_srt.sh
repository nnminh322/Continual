#!/bin/bash
#SBATCH -J srt
#SBATCH -o srt-%j.out
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 80:00:00
#SBATCH --mem 256G
#SBATCH --gres=gpu:a100-sxm4-80gb:1

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
GPU_ID="${1:-0}"
MODEL_PATH="${2:-google/flan-t5-xl}"

# ── GPU detection ────────────────────────────────────────────────────────────
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
: ${GPU_MEM:=16000}; : ${NUM_GPUS:=1}

if [ "$GPU_MEM" -lt 20000 ]; then
    IS_T4=1; GPU_MODE="t4_1gpu"; GPU_IDS="$GPU_ID"; FP16_FLAG="--gradient_checkpointing"
else
    IS_T4=0; GPU_MODE="a100"; GPU_IDS="$GPU_ID"; FP16_FLAG=""
fi

echo "[GPU] $GPU_MODE | CUDA_VISIBLE_DEVICES=$GPU_IDS | $MODEL_PATH"
echo "============================================================"

if [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=2; GA=8; EVAL_BSZ=8
else
    BSZ=2; GA=8; EVAL_BSZ=16
fi

RUN_NAME="long_order4_t5_srt"
TASK_ORDER="mnli,cb,wic,copa,qqp,boolq,rte,imdb,yelp,amazon,sst2,dbpedia,agnews,multirc,yahoo"
BASE_OUT="logs_and_outputs/$RUN_NAME"
SRT_FLAGS="--use_srt_router --srt_shrink --srt_shrink_factor 0.1 --srt_metric auto --srt_max_emb_samples 500"

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $MODEL_PATH \
   --data_dir CL_Benchmark \
   --task_order $TASK_ORDER \
   --task_config_dir configs/gen_script_long_order4_t5_configs/yelp \
   --output_dir $BASE_OUT/outputs/1-yelp \
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
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_rougeL \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --load_best_model_at_end \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG \
   $SRT_FLAGS \
   

rm -rf $BASE_OUT/outputs/1-yelp/checkpoint*
sleep 5

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $MODEL_PATH \
   --data_dir CL_Benchmark \
   --load_checkpoint_from $BASE_OUT/outputs/1-yelp/saved_weights/trans_input.pt \
   --previous_lora_path $BASE_OUT/outputs/1-yelp/saved_weights \
   --previous_prompt_key_path $BASE_OUT/outputs/1-yelp/saved_weights/prompts_keys_till_now.pt \
   --task_order $TASK_ORDER \
   --gen_data_dir generated_data/lora_gen_long_t5 \
   --task_config_dir configs/gen_script_long_order4_t5_configs/amazon \
   --output_dir $BASE_OUT/outputs/2-amazon \
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
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_rougeL \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --load_best_model_at_end \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG \
   $SRT_FLAGS \
   --srt_load_path $BASE_OUT/outputs/1-yelp/saved_weights

rm -rf $BASE_OUT/outputs/2-amazon/checkpoint*
sleep 5

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $MODEL_PATH \
   --data_dir CL_Benchmark \
   --load_checkpoint_from $BASE_OUT/outputs/2-amazon/saved_weights/trans_input.pt \
   --previous_lora_path $BASE_OUT/outputs/1-yelp/saved_weights,$BASE_OUT/outputs/2-amazon/saved_weights \
   --previous_prompt_key_path $BASE_OUT/outputs/2-amazon/saved_weights/prompts_keys_till_now.pt \
   --task_order $TASK_ORDER \
   --gen_data_dir generated_data/lora_gen_long_t5 \
   --task_config_dir configs/gen_script_long_order4_t5_configs/mnli \
   --output_dir $BASE_OUT/outputs/3-mnli \
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
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_rougeL \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --load_best_model_at_end \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG \
   $SRT_FLAGS \
   --srt_load_path $BASE_OUT/outputs/2-amazon/saved_weights

rm -rf $BASE_OUT/outputs/3-mnli/checkpoint*
sleep 5

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $MODEL_PATH \
   --data_dir CL_Benchmark \
   --load_checkpoint_from $BASE_OUT/outputs/3-mnli/saved_weights/trans_input.pt \
   --previous_lora_path $BASE_OUT/outputs/1-yelp/saved_weights,$BASE_OUT/outputs/2-amazon/saved_weights,$BASE_OUT/outputs/3-mnli/saved_weights \
   --previous_prompt_key_path $BASE_OUT/outputs/3-mnli/saved_weights/prompts_keys_till_now.pt \
   --task_order $TASK_ORDER \
   --gen_data_dir generated_data/lora_gen_long_t5 \
   --task_config_dir configs/gen_script_long_order4_t5_configs/cb \
   --output_dir $BASE_OUT/outputs/4-cb \
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
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_rougeL \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --load_best_model_at_end \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG \
   $SRT_FLAGS \
   --srt_load_path $BASE_OUT/outputs/3-mnli/saved_weights

rm -rf $BASE_OUT/outputs/4-cb/checkpoint*
sleep 5

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $MODEL_PATH \
   --data_dir CL_Benchmark \
   --load_checkpoint_from $BASE_OUT/outputs/4-cb/saved_weights/trans_input.pt \
   --previous_lora_path $BASE_OUT/outputs/1-yelp/saved_weights,$BASE_OUT/outputs/2-amazon/saved_weights,$BASE_OUT/outputs/3-mnli/saved_weights,$BASE_OUT/outputs/4-cb/saved_weights \
   --previous_prompt_key_path $BASE_OUT/outputs/4-cb/saved_weights/prompts_keys_till_now.pt \
   --task_order $TASK_ORDER \
   --gen_data_dir generated_data/lora_gen_long_t5 \
   --task_config_dir configs/gen_script_long_order4_t5_configs/copa \
   --output_dir $BASE_OUT/outputs/5-copa \
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
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_rougeL \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --load_best_model_at_end \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG \
   $SRT_FLAGS \
   --srt_load_path $BASE_OUT/outputs/4-cb/saved_weights

rm -rf $BASE_OUT/outputs/5-copa/checkpoint*
sleep 5

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $MODEL_PATH \
   --data_dir CL_Benchmark \
   --load_checkpoint_from $BASE_OUT/outputs/5-copa/saved_weights/trans_input.pt \
   --previous_lora_path $BASE_OUT/outputs/1-yelp/saved_weights,$BASE_OUT/outputs/2-amazon/saved_weights,$BASE_OUT/outputs/3-mnli/saved_weights,$BASE_OUT/outputs/4-cb/saved_weights,$BASE_OUT/outputs/5-copa/saved_weights \
   --previous_prompt_key_path $BASE_OUT/outputs/5-copa/saved_weights/prompts_keys_till_now.pt \
   --task_order $TASK_ORDER \
   --gen_data_dir generated_data/lora_gen_long_t5 \
   --task_config_dir configs/gen_script_long_order4_t5_configs/qqp \
   --output_dir $BASE_OUT/outputs/6-qqp \
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
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_rougeL \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --load_best_model_at_end \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG \
   $SRT_FLAGS \
   --srt_load_path $BASE_OUT/outputs/5-copa/saved_weights

rm -rf $BASE_OUT/outputs/6-qqp/checkpoint*
sleep 5

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $MODEL_PATH \
   --data_dir CL_Benchmark \
   --load_checkpoint_from $BASE_OUT/outputs/6-qqp/saved_weights/trans_input.pt \
   --previous_lora_path $BASE_OUT/outputs/1-yelp/saved_weights,$BASE_OUT/outputs/2-amazon/saved_weights,$BASE_OUT/outputs/3-mnli/saved_weights,$BASE_OUT/outputs/4-cb/saved_weights,$BASE_OUT/outputs/5-copa/saved_weights,$BASE_OUT/outputs/6-qqp/saved_weights \
   --previous_prompt_key_path $BASE_OUT/outputs/6-qqp/saved_weights/prompts_keys_till_now.pt \
   --task_order $TASK_ORDER \
   --gen_data_dir generated_data/lora_gen_long_t5 \
   --task_config_dir configs/gen_script_long_order4_t5_configs/rte \
   --output_dir $BASE_OUT/outputs/7-rte \
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
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_rougeL \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --load_best_model_at_end \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG \
   $SRT_FLAGS \
   --srt_load_path $BASE_OUT/outputs/6-qqp/saved_weights

rm -rf $BASE_OUT/outputs/7-rte/checkpoint*
sleep 5

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $MODEL_PATH \
   --data_dir CL_Benchmark \
   --load_checkpoint_from $BASE_OUT/outputs/7-rte/saved_weights/trans_input.pt \
   --previous_lora_path $BASE_OUT/outputs/1-yelp/saved_weights,$BASE_OUT/outputs/2-amazon/saved_weights,$BASE_OUT/outputs/3-mnli/saved_weights,$BASE_OUT/outputs/4-cb/saved_weights,$BASE_OUT/outputs/5-copa/saved_weights,$BASE_OUT/outputs/6-qqp/saved_weights,$BASE_OUT/outputs/7-rte/saved_weights \
   --previous_prompt_key_path $BASE_OUT/outputs/7-rte/saved_weights/prompts_keys_till_now.pt \
   --task_order $TASK_ORDER \
   --gen_data_dir generated_data/lora_gen_long_t5 \
   --task_config_dir configs/gen_script_long_order4_t5_configs/imdb \
   --output_dir $BASE_OUT/outputs/8-imdb \
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
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_rougeL \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --load_best_model_at_end \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG \
   $SRT_FLAGS \
   --srt_load_path $BASE_OUT/outputs/7-rte/saved_weights

rm -rf $BASE_OUT/outputs/8-imdb/checkpoint*
sleep 5

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $MODEL_PATH \
   --data_dir CL_Benchmark \
   --load_checkpoint_from $BASE_OUT/outputs/8-imdb/saved_weights/trans_input.pt \
   --previous_lora_path $BASE_OUT/outputs/1-yelp/saved_weights,$BASE_OUT/outputs/2-amazon/saved_weights,$BASE_OUT/outputs/3-mnli/saved_weights,$BASE_OUT/outputs/4-cb/saved_weights,$BASE_OUT/outputs/5-copa/saved_weights,$BASE_OUT/outputs/6-qqp/saved_weights,$BASE_OUT/outputs/7-rte/saved_weights,$BASE_OUT/outputs/8-imdb/saved_weights \
   --previous_prompt_key_path $BASE_OUT/outputs/8-imdb/saved_weights/prompts_keys_till_now.pt \
   --task_order $TASK_ORDER \
   --gen_data_dir generated_data/lora_gen_long_t5 \
   --task_config_dir configs/gen_script_long_order4_t5_configs/sst2 \
   --output_dir $BASE_OUT/outputs/9-sst2 \
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
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_rougeL \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --load_best_model_at_end \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG \
   $SRT_FLAGS \
   --srt_load_path $BASE_OUT/outputs/8-imdb/saved_weights

rm -rf $BASE_OUT/outputs/9-sst2/checkpoint*
sleep 5

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $MODEL_PATH \
   --data_dir CL_Benchmark \
   --load_checkpoint_from $BASE_OUT/outputs/9-sst2/saved_weights/trans_input.pt \
   --previous_lora_path $BASE_OUT/outputs/1-yelp/saved_weights,$BASE_OUT/outputs/2-amazon/saved_weights,$BASE_OUT/outputs/3-mnli/saved_weights,$BASE_OUT/outputs/4-cb/saved_weights,$BASE_OUT/outputs/5-copa/saved_weights,$BASE_OUT/outputs/6-qqp/saved_weights,$BASE_OUT/outputs/7-rte/saved_weights,$BASE_OUT/outputs/8-imdb/saved_weights,$BASE_OUT/outputs/9-sst2/saved_weights \
   --previous_prompt_key_path $BASE_OUT/outputs/9-sst2/saved_weights/prompts_keys_till_now.pt \
   --task_order $TASK_ORDER \
   --gen_data_dir generated_data/lora_gen_long_t5 \
   --task_config_dir configs/gen_script_long_order4_t5_configs/dbpedia \
   --output_dir $BASE_OUT/outputs/10-dbpedia \
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
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_rougeL \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --load_best_model_at_end \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG \
   $SRT_FLAGS \
   --srt_load_path $BASE_OUT/outputs/9-sst2/saved_weights

rm -rf $BASE_OUT/outputs/10-dbpedia/checkpoint*
sleep 5

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $MODEL_PATH \
   --data_dir CL_Benchmark \
   --load_checkpoint_from $BASE_OUT/outputs/10-dbpedia/saved_weights/trans_input.pt \
   --previous_lora_path $BASE_OUT/outputs/1-yelp/saved_weights,$BASE_OUT/outputs/2-amazon/saved_weights,$BASE_OUT/outputs/3-mnli/saved_weights,$BASE_OUT/outputs/4-cb/saved_weights,$BASE_OUT/outputs/5-copa/saved_weights,$BASE_OUT/outputs/6-qqp/saved_weights,$BASE_OUT/outputs/7-rte/saved_weights,$BASE_OUT/outputs/8-imdb/saved_weights,$BASE_OUT/outputs/9-sst2/saved_weights,$BASE_OUT/outputs/10-dbpedia/saved_weights \
   --previous_prompt_key_path $BASE_OUT/outputs/10-dbpedia/saved_weights/prompts_keys_till_now.pt \
   --task_order $TASK_ORDER \
   --gen_data_dir generated_data/lora_gen_long_t5 \
   --task_config_dir configs/gen_script_long_order4_t5_configs/agnews \
   --output_dir $BASE_OUT/outputs/11-agnews \
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
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_rougeL \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --load_best_model_at_end \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG \
   $SRT_FLAGS \
   --srt_load_path $BASE_OUT/outputs/10-dbpedia/saved_weights

rm -rf $BASE_OUT/outputs/11-agnews/checkpoint*
sleep 5

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $MODEL_PATH \
   --data_dir CL_Benchmark \
   --load_checkpoint_from $BASE_OUT/outputs/11-agnews/saved_weights/trans_input.pt \
   --previous_lora_path $BASE_OUT/outputs/1-yelp/saved_weights,$BASE_OUT/outputs/2-amazon/saved_weights,$BASE_OUT/outputs/3-mnli/saved_weights,$BASE_OUT/outputs/4-cb/saved_weights,$BASE_OUT/outputs/5-copa/saved_weights,$BASE_OUT/outputs/6-qqp/saved_weights,$BASE_OUT/outputs/7-rte/saved_weights,$BASE_OUT/outputs/8-imdb/saved_weights,$BASE_OUT/outputs/9-sst2/saved_weights,$BASE_OUT/outputs/10-dbpedia/saved_weights,$BASE_OUT/outputs/11-agnews/saved_weights \
   --previous_prompt_key_path $BASE_OUT/outputs/11-agnews/saved_weights/prompts_keys_till_now.pt \
   --task_order $TASK_ORDER \
   --gen_data_dir generated_data/lora_gen_long_t5 \
   --task_config_dir configs/gen_script_long_order4_t5_configs/yahoo \
   --output_dir $BASE_OUT/outputs/12-yahoo \
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
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_rougeL \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --load_best_model_at_end \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG \
   $SRT_FLAGS \
   --srt_load_path $BASE_OUT/outputs/11-agnews/saved_weights

rm -rf $BASE_OUT/outputs/12-yahoo/checkpoint*
sleep 5

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $MODEL_PATH \
   --data_dir CL_Benchmark \
   --load_checkpoint_from $BASE_OUT/outputs/12-yahoo/saved_weights/trans_input.pt \
   --previous_lora_path $BASE_OUT/outputs/1-yelp/saved_weights,$BASE_OUT/outputs/2-amazon/saved_weights,$BASE_OUT/outputs/3-mnli/saved_weights,$BASE_OUT/outputs/4-cb/saved_weights,$BASE_OUT/outputs/5-copa/saved_weights,$BASE_OUT/outputs/6-qqp/saved_weights,$BASE_OUT/outputs/7-rte/saved_weights,$BASE_OUT/outputs/8-imdb/saved_weights,$BASE_OUT/outputs/9-sst2/saved_weights,$BASE_OUT/outputs/10-dbpedia/saved_weights,$BASE_OUT/outputs/11-agnews/saved_weights,$BASE_OUT/outputs/12-yahoo/saved_weights \
   --previous_prompt_key_path $BASE_OUT/outputs/12-yahoo/saved_weights/prompts_keys_till_now.pt \
   --task_order $TASK_ORDER \
   --gen_data_dir generated_data/lora_gen_long_t5 \
   --task_config_dir configs/gen_script_long_order4_t5_configs/multirc \
   --output_dir $BASE_OUT/outputs/13-multirc \
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
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_rougeL \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --load_best_model_at_end \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG \
   $SRT_FLAGS \
   --srt_load_path $BASE_OUT/outputs/12-yahoo/saved_weights

rm -rf $BASE_OUT/outputs/13-multirc/checkpoint*
sleep 5

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $MODEL_PATH \
   --data_dir CL_Benchmark \
   --load_checkpoint_from $BASE_OUT/outputs/13-multirc/saved_weights/trans_input.pt \
   --previous_lora_path $BASE_OUT/outputs/1-yelp/saved_weights,$BASE_OUT/outputs/2-amazon/saved_weights,$BASE_OUT/outputs/3-mnli/saved_weights,$BASE_OUT/outputs/4-cb/saved_weights,$BASE_OUT/outputs/5-copa/saved_weights,$BASE_OUT/outputs/6-qqp/saved_weights,$BASE_OUT/outputs/7-rte/saved_weights,$BASE_OUT/outputs/8-imdb/saved_weights,$BASE_OUT/outputs/9-sst2/saved_weights,$BASE_OUT/outputs/10-dbpedia/saved_weights,$BASE_OUT/outputs/11-agnews/saved_weights,$BASE_OUT/outputs/12-yahoo/saved_weights,$BASE_OUT/outputs/13-multirc/saved_weights \
   --previous_prompt_key_path $BASE_OUT/outputs/13-multirc/saved_weights/prompts_keys_till_now.pt \
   --task_order $TASK_ORDER \
   --gen_data_dir generated_data/lora_gen_long_t5 \
   --task_config_dir configs/gen_script_long_order4_t5_configs/boolq \
   --output_dir $BASE_OUT/outputs/14-boolq \
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
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_rougeL \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --load_best_model_at_end \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG \
   $SRT_FLAGS \
   --srt_load_path $BASE_OUT/outputs/13-multirc/saved_weights

rm -rf $BASE_OUT/outputs/14-boolq/checkpoint*
sleep 5

CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $MODEL_PATH \
   --data_dir CL_Benchmark \
   --load_checkpoint_from $BASE_OUT/outputs/14-boolq/saved_weights/trans_input.pt \
   --previous_lora_path $BASE_OUT/outputs/1-yelp/saved_weights,$BASE_OUT/outputs/2-amazon/saved_weights,$BASE_OUT/outputs/3-mnli/saved_weights,$BASE_OUT/outputs/4-cb/saved_weights,$BASE_OUT/outputs/5-copa/saved_weights,$BASE_OUT/outputs/6-qqp/saved_weights,$BASE_OUT/outputs/7-rte/saved_weights,$BASE_OUT/outputs/8-imdb/saved_weights,$BASE_OUT/outputs/9-sst2/saved_weights,$BASE_OUT/outputs/10-dbpedia/saved_weights,$BASE_OUT/outputs/11-agnews/saved_weights,$BASE_OUT/outputs/12-yahoo/saved_weights,$BASE_OUT/outputs/13-multirc/saved_weights,$BASE_OUT/outputs/14-boolq/saved_weights \
   --previous_prompt_key_path $BASE_OUT/outputs/14-boolq/saved_weights/prompts_keys_till_now.pt \
   --task_order $TASK_ORDER \
   --gen_data_dir generated_data/lora_gen_long_t5 \
   --task_config_dir configs/gen_script_long_order4_t5_configs/wic \
   --output_dir $BASE_OUT/outputs/15-wic \
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
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_rougeL \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --load_best_model_at_end \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0.5 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995 \
   $FP16_FLAG \
   $SRT_FLAGS \
   --srt_load_path $BASE_OUT/outputs/14-boolq/saved_weights

rm -rf $BASE_OUT/outputs/15-wic/checkpoint*
sleep 5
echo "[DONE] All 15 tasks complete. Run: python score.py $RUN_NAME $RUN_NAME"
