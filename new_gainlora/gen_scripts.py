#!/usr/bin/env python3
"""
Generate SRT experiment scripts for new_gainlora.
Generates 6 scripts:
  - T5-XL SuperNI order 1 (15 tasks)
  - T5-XL SuperNI order 2 (15 tasks)
  - T5-XL Long-Seq order 3 (15 tasks)
  - T5-XL Long-Seq order 4 (15 tasks)
  - Llama-3 8B SuperNI order 1 (15 tasks)
  - Llama-3 8B SuperNI order 2 (15 tasks)
"""

import textwrap

# ── Task orders ──────────────────────────────────────────────────────────────

SUPERNI_ORDER1 = (
    "task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,"
    "task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,"
    "task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,"
    "task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,"
    "task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,"
    "task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification"
)

SUPERNI_ORDER2 = (
    "task748_glucose_reverse_cause_event_detection,task073_commonsenseqa_answer_generation,"
    "task1590_diplomacy_text_generation,task639_multi_woz_user_utterance_generation,"
    "task1572_samsum_summary,task1687_sentiment140_classification,task591_sciq_answer_generation,"
    "task363_sst2_polarity_classification,task1510_evalution_relation_extraction,"
    "task1729_personachat_generate_next,task181_outcome_extraction,"
    "task511_reddit_tifu_long_text_summarization,task002_quoref_answer_generation,"
    "task1290_xsum_summarization,task875_emotion_classification"
)

LONG_ORDER3 = (
    "yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic"
)

LONG_ORDER4 = (
    "mnli,cb,wic,copa,qqp,boolq,rte,imdb,yelp,amazon,sst2,dbpedia,agnews,multirc,yahoo"
)

LONG_TASK_LIST = [
    "yelp", "amazon", "mnli", "cb", "copa", "qqp", "rte",
    "imdb", "sst2", "dbpedia", "agnews", "yahoo", "multirc", "boolq", "wic"
]

SUPERNI_TASK_LIST_ORDER1 = [
    "task1572_samsum_summary", "task363_sst2_polarity_classification",
    "task1290_xsum_summarization", "task181_outcome_extraction",
    "task002_quoref_answer_generation", "task1510_evalution_relation_extraction",
    "task639_multi_woz_user_utterance_generation", "task1729_personachat_generate_next",
    "task073_commonsenseqa_answer_generation", "task1590_diplomacy_text_generation",
    "task748_glucose_reverse_cause_event_detection", "task511_reddit_tifu_long_text_summarization",
    "task591_sciq_answer_generation", "task1687_sentiment140_classification",
    "task875_emotion_classification"
]

SUPERNI_TASK_LIST_ORDER2 = [
    "task748_glucose_reverse_cause_event_detection", "task073_commonsenseqa_answer_generation",
    "task1590_diplomacy_text_generation", "task639_multi_woz_user_utterance_generation",
    "task1572_samsum_summary", "task1687_sentiment140_classification",
    "task591_sciq_answer_generation", "task363_sst2_polarity_classification",
    "task1510_evalution_relation_extraction", "task1729_personachat_generate_next",
    "task181_outcome_extraction", "task511_reddit_tifu_long_text_summarization",
    "task002_quoref_answer_generation", "task1290_xsum_summarization",
    "task875_emotion_classification"
]

# ── T5 SRT script generator ───────────────────────────────────────────────────

def gen_t5_script(run_name, task_order, task_list, config_dir, gen_data_dir, metric):
    assert len(task_list) == 15
    SRT_FLAGS = (
        "--use_srt_router --srt_shrink --srt_shrink_factor 0.1 "
        "--srt_metric_mode hard --srt_max_emb_samples 500"
    )
    prev_lora_chains = ",".join(
        f"$BASE_OUT/outputs/{i+1}-{task_list[i]}/saved_weights"
        for i in range(14)
    )
    prev_lora_task2 = ",".join(
        [f"$BASE_OUT/outputs/1-{task_list[0]}/saved_weights"]
    )

    task_cmds = []
    for i, task in enumerate(task_list):
        # Previous LoRA paths (all previous tasks)
        if i == 0:
            prev_lora = ""
            prev_key = ""
            load_trans = ""
        elif i == 1:
            prev_lora = (
                f"$BASE_OUT/outputs/1-{task_list[0]}/saved_weights"
            )
            prev_key = (
                f"$BASE_OUT/outputs/1-{task_list[0]}/saved_weights/"
                f"prompts_keys_till_now.pt"
            )
            load_trans = (
                f"$BASE_OUT/outputs/1-{task_list[0]}/saved_weights/trans_input.pt"
            )
        else:
            prev_lora_list = ",".join(
                f"$BASE_OUT/outputs/{j+1}-{task_list[j]}/saved_weights"
                for j in range(i)
            )
            prev_lora = prev_lora_list
            prev_key = (
                f"$BASE_OUT/outputs/{i}-{task_list[i-1]}/saved_weights/"
                f"prompts_keys_till_now.pt"
            )
            load_trans = (
                f"$BASE_OUT/outputs/{i}-{task_list[i-1]}/saved_weights/trans_input.pt"
            )

        # SRT flags: task 1 has no load_path, task 2+ load from prev saved_weights dir
        if i == 0:
            srt_load = ""
        else:
            srt_load = f"--srt_load_path $BASE_OUT/outputs/{i}-{task_list[i-1]}/saved_weights"

        if i == 0:
            common = f"""\
   --data_dir CL_Benchmark \\
   --task_order $TASK_ORDER \\
   --task_config_dir {config_dir}/{task} \\
   --output_dir $BASE_OUT/outputs/{i+1}-{task} \\"""
        else:
            common = f"""\
   --data_dir CL_Benchmark \\
   --load_checkpoint_from $BASE_OUT/outputs/{i}-{task_list[i-1]}/saved_weights/trans_input.pt \\
   --previous_lora_path {prev_lora} \\
   --previous_prompt_key_path $BASE_OUT/outputs/{i}-{task_list[i-1]}/saved_weights/prompts_keys_till_now.pt \\
   --task_order $TASK_ORDER \\
   --gen_data_dir {gen_data_dir} \\
   --task_config_dir {config_dir}/{task} \\
   --output_dir $BASE_OUT/outputs/{i+1}-{task} \\"""

        metric_key = f"eval_exact_match" if "classification" in task.lower() or "polarity" in task.lower() else f"eval_{metric}"

        cmd = f"""\
CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \\
   --do_train \\
   --do_predict \\
   --predict_with_generate \\
   --model_name_or_path $MODEL_PATH \\
{common}
   --per_device_train_batch_size $BSZ \\
   --per_device_eval_batch_size $EVAL_BSZ \\
   --gradient_accumulation_steps $GA \\
   --learning_rate 0.0003 \\
   --num_train_epochs 100 \\
   --run_name $RUN_NAME \\
   --max_source_length 512 \\
   --max_target_length 50 \\
   --generation_max_length 50 \\
   --add_task_name False \\
   --add_dataset_name False \\
   --overwrite_output_dir \\
   --overwrite_cache \\
   --lr_scheduler_type constant \\
   --warmup_steps 0 \\
   --logging_strategy steps \\
   --logging_steps 10 \\
   --metric_for_best_model {metric_key} \\
   --evaluation_strategy steps \\
   --save_strategy steps \\
   --save_total_limit 1 \\
   --lora_r 8 \\
   --lora_alpha 32 \\
   --lora_dropout 0.0 \\
   --load_best_model_at_end \\
   --data_replay_freq -1 \\
   --replay_after_n_epoch 0 \\
   --kl_ratio 0.5 \\
   --attn_temperature 1 \\
   --mlp_hidden_dim 100 \\
   --model_name gainlora_inflora \\
   --threshold 0.995 \\
   --transthreshold 0.995 \\
   $FP16_FLAG \\
   $SRT_FLAGS \\
   {srt_load}

rm -rf $BASE_OUT/outputs/{i+1}-{task}/checkpoint*
sleep 5"""
        task_cmds.append(cmd)

    return textwrap.dedent(f"""\
#!/bin/bash
#SBATCH -J srt
#SBATCH -o srt-%j.out
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 80:00:00
#SBATCH --mem 256G
#SBATCH --gres=gpu:a100-sxm4-80gb:1

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
GPU_ID="${{1:-0}}"
MODEL_PATH="${{2:-google/flan-t5-xl}}"

# ── GPU detection ────────────────────────────────────────────────────────────
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
: ${{GPU_MEM:=16000}}; : ${{NUM_GPUS:=1}}

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

RUN_NAME="{run_name}"
TASK_ORDER="{task_order}"
BASE_OUT="logs_and_outputs/$RUN_NAME"
SRT_FLAGS="--use_srt_router --srt_shrink --srt_shrink_factor 0.1 --srt_metric_mode hard --srt_max_emb_samples 500"

""") + "\n\n".join(task_cmds) + f"""\

echo "[DONE] All 15 tasks complete. Run: python score.py $RUN_NAME $RUN_NAME"
"""


# ── Llama SRT script generator ────────────────────────────────────────────────

def gen_llama_script(run_name, task_order, task_list, config_dir, gen_data_dir):
    SRT_FLAGS = (
        "--use_srt_router --srt_shrink --srt_shrink_factor 0.1 "
        "--srt_metric_mode hard --srt_max_emb_samples 500"
    )

    task_cmds = []
    for i, task in enumerate(task_list):
        if i == 0:
            prev_lora = ""
            prev_key = ""
            load_trans = ""
        elif i == 1:
            prev_lora = f"$BASE_OUT/outputs/1-{task_list[0]}/saved_weights"
            prev_key = f"$BASE_OUT/outputs/1-{task_list[0]}/saved_weights/prompts_keys_till_now.pt"
            load_trans = f"$BASE_OUT/outputs/1-{task_list[0]}/saved_weights/trans_input.pt"
        else:
            prev_lora = ",".join(
                f"$BASE_OUT/outputs/{j+1}-{task_list[j]}/saved_weights"
                for j in range(i)
            )
            prev_key = f"$BASE_OUT/outputs/{i}-{task_list[i-1]}/saved_weights/prompts_keys_till_now.pt"
            load_trans = f"$BASE_OUT/outputs/{i}-{task_list[i-1]}/saved_weights/trans_input.pt"

        srt_load = "" if i == 0 else f"--srt_load_path $BASE_OUT/outputs/{i}-{task_list[i-1]}/saved_weights"

        if i == 0:
            common = f"""\
   --data_dir CL_Benchmark \\
   --task_order $TASK_ORDER \\
   --task_config_dir {config_dir}/{task} \\
   --output_dir $BASE_OUT/outputs/{i+1}-{task} \\"""
        else:
            common = f"""\
   --data_dir CL_Benchmark \\
   --load_checkpoint_from $BASE_OUT/outputs/{i}-{task_list[i-1]}/saved_weights/trans_input.pt \\
   --previous_lora_path {prev_lora} \\
   --previous_prompt_key_path $BASE_OUT/outputs/{i}-{task_list[i-1]}/saved_weights/prompts_keys_till_now.pt \\
   --task_order $TASK_ORDER \\
   --gen_data_dir {gen_data_dir} \\
   --task_config_dir {config_dir}/{task} \\
   --output_dir $BASE_OUT/outputs/{i+1}-{task} \\"""

        metric_key = "eval_exact_match" if "classification" in task.lower() or "polarity" in task.lower() else "eval_rougeL"

        cmd = f"""\
CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_llama.py \\
   --do_train \\
   --do_predict \\
   --predict_with_generate \\
   --model_name_or_path $MODEL_PATH \\
{common}
   --per_device_train_batch_size $BSZ \\
   --per_device_eval_batch_size $EVAL_BSZ \\
   --gradient_accumulation_steps $GA \\
   --learning_rate 0.0003 \\
   --num_train_epochs 100 \\
   --run_name $RUN_NAME \\
   --max_source_length 512 \\
   --max_target_length 50 \\
   --generation_max_length 50 \\
   --add_task_name False \\
   --add_dataset_name False \\
   --overwrite_output_dir \\
   --overwrite_cache \\
   --lr_scheduler_type constant \\
   --warmup_steps 0 \\
   --logging_strategy steps \\
   --logging_steps 10 \\
   --metric_for_best_model {metric_key} \\
   --evaluation_strategy steps \\
   --save_strategy steps \\
   --save_total_limit 1 \\
   --lora_r 8 \\
   --lora_alpha 32 \\
   --lora_dropout 0.0 \\
   --load_best_model_at_end \\
   --data_replay_freq -1 \\
   --replay_after_n_epoch 0 \\
   --kl_ratio 0.5 \\
   --attn_temperature 1 \\
   --mlp_hidden_dim 100 \\
   --model_name gainlora_inflora \\
   --threshold 0.995 \\
   --transthreshold 0.995 \\
   $FP16_FLAG \\
   $SRT_FLAGS \\
   {srt_load}

rm -rf $BASE_OUT/outputs/{i+1}-{task}/checkpoint*
sleep 5"""
        task_cmds.append(cmd)

    return textwrap.dedent(f"""\
#!/bin/bash
#SBATCH -J srt-llama
#SBATCH -o srt-llama-%j.out
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 80:00:00
#SBATCH --mem 256G
#SBATCH --gres=gpu:a100-sxm4-80gb:1

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
MODEL_PATH="${{1:-meta-llama/Meta-Llama-3-8B}}"

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
: ${{GPU_MEM:=16000}}; : ${{NUM_GPUS:=1}}

if [ "$GPU_MEM" -lt 20000 ]; then
    IS_T4=1; GPU_MODE="t4_1gpu"; GPU_IDS="$GPU_ID"; FP16_FLAG="--gradient_checkpointing"
else
    IS_T4=0; GPU_MODE="a100"; GPU_IDS="$GPU_ID"; FP16_FLAG=""
fi

echo "[GPU] $GPU_MODE | CUDA_VISIBLE_DEVICES=$GPU_IDS | $MODEL_PATH"
echo "============================================================"

# Llama: smaller BSZ due to larger model
if [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ=1; GA=16; EVAL_BSZ=4
else
    BSZ=1; GA=16; EVAL_BSZ=8
fi

RUN_NAME="{run_name}"
TASK_ORDER="{task_order}"
BASE_OUT="logs_and_outputs/$RUN_NAME"
SRT_FLAGS="--use_srt_router --srt_shrink --srt_shrink_factor 0.1 --srt_metric_mode hard --srt_max_emb_samples 500"

""") + "\n\n".join(task_cmds) + f"""\

echo "[DONE] All 15 tasks complete. Run: python score.py $RUN_NAME $RUN_NAME"
"""


# ── Generate all scripts ───────────────────────────────────────────────────────

scripts = [
    ("gen_script_superni_order1_t5_srt.sh",
     gen_t5_script(
         "superni_order1_t5_srt",
         SUPERNI_ORDER1,
         SUPERNI_TASK_LIST_ORDER1,
         "configs/gen_script_superni_order1_t5_configs",
         "generated_data/lora_gen_superni_t5",
         "rougeL")),
    ("gen_script_superni_order2_t5_srt.sh",
     gen_t5_script(
         "superni_order2_t5_srt",
         SUPERNI_ORDER2,
         SUPERNI_TASK_LIST_ORDER2,
         "configs/gen_script_superni_order2_t5_configs",
         "generated_data/lora_gen_superni_t5",
         "rougeL")),
    ("gen_script_long_order3_t5_srt.sh",
     gen_t5_script(
         "long_order3_t5_srt",
         LONG_ORDER3,
         LONG_TASK_LIST,
         "configs/gen_script_long_order3_t5_configs",
         "generated_data/lora_gen_long_t5",
         "rougeL")),
    ("gen_script_long_order4_t5_srt.sh",
     gen_t5_script(
         "long_order4_t5_srt",
         LONG_ORDER4,
         LONG_TASK_LIST,
         "configs/gen_script_long_order4_t5_configs",
         "generated_data/lora_gen_long_t5",
         "rougeL")),
    ("gen_script_superni_order1_llama_srt.sh",
     gen_llama_script(
         "superni_order1_llama_srt",
         SUPERNI_ORDER1,
         SUPERNI_TASK_LIST_ORDER1,
         "configs/gen_script_superni_order1_llama_configs",
         "generated_data/lora_gen_superni_llama")),
    ("gen_script_superni_order2_llama_srt.sh",
     gen_llama_script(
         "superni_order2_llama_srt",
         SUPERNI_ORDER2,
         SUPERNI_TASK_LIST_ORDER2,
         "configs/gen_script_superni_order2_llama_configs",
         "generated_data/lora_gen_superni_llama")),
]

out_dir = "/Users/nnminh322/Desktop/personal/Continual/new_gainlora"
for name, content in scripts:
    path = f"{out_dir}/{name}"
    with open(path, "w") as f:
        f.write(content)
    import os
    os.chmod(path, 0o755)
    print(f"Generated: {name}")

print("\nUsage:")
print("  cd new_gainlora")
print("  bash gen_script_superni_order1_t5_srt.sh 0 google/flan-t5-xl")
print("  bash gen_script_superni_order2_t5_srt.sh 0 google/flan-t5-xl")
print("  bash gen_script_long_order3_t5_srt.sh  0 google/flan-t5-xl")
print("  bash gen_script_long_order4_t5_srt.sh  0 google/flan-t5-xl")
print("  bash gen_script_superni_order1_llama_srt.sh 0 meta-llama/Meta-Llama-3-8B")
print("  bash gen_script_superni_order2_llama_srt.sh 0 meta-llama/Meta-Llama-3-8B")
