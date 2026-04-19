#!/bin/bash
set -e

echo "============================================================"
echo " Re-eval task 9 (sst2)"
echo "============================================================"

cd ~/minhnguyen/test_model/Continual/new_gainlora

PREV_LORA="logs_and_outputs/long_order3_t5_srt/outputs/1-yelp/saved_weights,logs_and_outputs/long_order3_t5_srt/outputs/2-amazon/saved_weights,logs_and_outputs/long_order3_t5_srt/outputs/3-mnli/saved_weights,logs_and_outputs/long_order3_t5_srt/outputs/4-cb/saved_weights,logs_and_outputs/long_order3_t5_srt/outputs/5-copa/saved_weights,logs_and_outputs/long_order3_t5_srt/outputs/6-qqp/saved_weights,logs_and_outputs/long_order3_t5_srt/outputs/7-rte/saved_weights,logs_and_outputs/long_order3_t5_srt/outputs/8-imdb/saved_weights"

# Always re-eval — no skip, ensure fresh results
CUDA_VISIBLE_DEVICES=0 python src/run_t5.py \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path google/flan-t5-large \
    --previous_lora_path "$PREV_LORA" \
    --previous_prompt_key_path logs_and_outputs/long_order3_t5_srt/outputs/9-sst2/saved_weights/prompts_keys_till_now.pt \
    --current_lora_path logs_and_outputs/long_order3_t5_srt/outputs/9-sst2/saved_weights \
    --data_dir CL_Benchmark \
    --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
    --task_config_dir configs/gen_script_long_order3_t5_configs/sst2 \
    --output_dir logs_and_outputs/long_order3_t5_srt/outputs/9-sst2 \
    --srt_load_path logs_and_outputs/long_order3_t5_srt/outputs/9-sst2/saved_weights \
    --srt_metric_mode hard \
    --srt_skip_forward \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --num_beams 1 \
    --generation_max_length 50 \
    --max_source_length 512 \
    --use_srt_router True \
    --sgwi True \
    --dual_fisher False \
    --run_name long_order3_t5_srt \
    --mlp_hidden_dim 100 \
    --model_name gainlora \
    --metric_for_best_model eval_exact_match_for_sst2

echo ""
echo "=== Task 9 Result ==="
cat logs_and_outputs/long_order3_t5_srt/outputs/9-sst2/all_results.json
