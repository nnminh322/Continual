#!/bin/bash
set -e

echo "============================================================"
echo " Re-eval task 15 (wic)"
echo "============================================================"

cd ~/minhnguyen/test_model/Continual/new_gainlora

PREV_LORA="logs_and_outputs/long_order3_t5_srt/outputs/1-yelp/saved_weights,logs_and_outputs/long_order3_t5_srt/outputs/2-amazon/saved_weights,logs_and_outputs/long_order3_t5_srt/outputs/3-mnli/saved_weights,logs_and_outputs/long_order3_t5_srt/outputs/4-cb/saved_weights,logs_and_outputs/long_order3_t5_srt/outputs/5-copa/saved_weights,logs_and_outputs/long_order3_t5_srt/outputs/6-qqp/saved_weights,logs_and_outputs/long_order3_t5_srt/outputs/7-rte/saved_weights,logs_and_outputs/long_order3_t5_srt/outputs/8-imdb/saved_weights,logs_and_outputs/long_order3_t5_srt/outputs/9-sst2/saved_weights,logs_and_outputs/long_order3_t5_srt/outputs/10-dbpedia/saved_weights,logs_and_outputs/long_order3_t5_srt/outputs/11-agnews/saved_weights,logs_and_outputs/long_order3_t5_srt/outputs/12-yahoo/saved_weights,logs_and_outputs/long_order3_t5_srt/outputs/13-multirc/saved_weights,logs_and_outputs/long_order3_t5_srt/outputs/14-boolq/saved_weights"

# Always re-eval — no skip, ensure fresh results
CUDA_VISIBLE_DEVICES=0 python src/run_t5.py \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path google/flan-t5-large \
    --previous_lora_path "$PREV_LORA" \
    --previous_prompt_key_path logs_and_outputs/long_order3_t5_srt/outputs/15-wic/saved_weights/prompts_keys_till_now.pt \
    --current_lora_path logs_and_outputs/long_order3_t5_srt/outputs/15-wic/saved_weights \
    --data_dir CL_Benchmark \
    --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
    --task_config_dir configs/gen_script_long_order3_t5_configs/wic \
    --output_dir logs_and_outputs/long_order3_t5_srt/outputs/15-wic \
    --srt_load_path logs_and_outputs/long_order3_t5_srt/outputs/15-wic/saved_weights \
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
    --metric_for_best_model eval_exact_match_for_wic

echo ""
echo "=== Task 15 Result ==="
cat logs_and_outputs/long_order3_t5_srt/outputs/15-wic/all_results.json

echo ""
echo "============================================================"
echo " Scoring pipeline"
echo "============================================================"
cd ~/minhnguyen/test_model/Continual/new_gainlora
python score.py long_order3_t5_srt long_order3_t5_single
