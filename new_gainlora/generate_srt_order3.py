#!/usr/bin/env python3
"""
Generator cho SRT order-3 experiment scripts.

Usage:
    python generate_srt_order3.py hard
    python generate_srt_order3.py dynamics
    python generate_srt_order3.py all

Output:
    gen_script_long_order3_t5_srt_{mode}.sh — full 15-task script
"""

import sys
import os

MODEL = "google/flan-t5-small"  # change as needed

# (task_name, output_dir, srt_load_dir, config_dir)
# srt_load_dir = output_dir of the PREVIOUS task (not including /saved_weights)
TASKS = [
    ("yelp",    "1-yelp",    None,                                       "gen_script_long_order3_t5_configs/yelp"),
    ("amazon",  "2-amazon",  "1-yelp",                                    "gen_script_long_order3_t5_configs/amazon"),
    ("mnli",    "3-mnli",   "2-amazon",                                  "gen_script_long_order3_t5_configs/mnli"),
    ("cb",      "4-cb",     "3-mnli",                                    "gen_script_long_order3_t5_configs/cb"),
    ("copa",    "5-copa",   "4-cb",                                      "gen_script_long_order3_t5_configs/copa"),
    ("qqp",     "6-qqp",    "5-copa",                                    "gen_script_long_order3_t5_configs/qqp"),
    ("rte",     "7-rte",    "6-qqp",                                     "gen_script_long_order3_t5_configs/rte"),
    ("imdb",    "8-imdb",   "7-rte",                                     "gen_script_long_order3_t5_configs/imdb"),
    ("sst2",    "9-sst2",   "8-imdb",                                    "gen_script_long_order3_t5_configs/sst2"),
    ("dbpedia", "10-dbpedia","9-sst2",                                    "gen_script_long_order3_t5_configs/dbpedia"),
    ("agnews",  "11-agnews", "10-dbpedia",                                "gen_script_long_order3_t5_configs/agnews"),
    ("yahoo",   "12-yahoo",  "11-agnews",                                 "gen_script_long_order3_t5_configs/yahoo"),
    ("multirc", "13-multirc", "12-yahoo",                                 "gen_script_long_order3_t5_configs/multirc"),
    ("boolq",   "14-boolq",  "13-multirc",                                "gen_script_long_order3_t5_configs/boolq"),
    ("wic",     "15-wic",    "14-boolq",                                  "gen_script_long_order3_t5_configs/wic"),
]

GPU_PARAMS = {
    "t4_2gpu":  {"bsz": 2,  "ga": 4,  "eval_bsz": 16},
    "t4_1gpu":  {"bsz": 4,  "ga": 8,  "eval_bsz": 16},
    "a100":     {"bsz": 16, "ga": 2,  "eval_bsz": 128},
}


def prev_lora_paths(up_to_task_idx, mode):
    """Build comma-separated previous LoRA paths for tasks 0..up_to_task_idx-1."""
    parts = []
    for i in range(up_to_task_idx):
        _, out_dir, _, _ = TASKS[i]
        parts.append(f"logs_and_outputs/long_order3_t5_srt_{mode}/outputs/{out_dir}/saved_weights")
    return ",".join(parts)


def task_block(task_name, task_idx, out_dir, srt_load, config_dir, gpu_mode, mode):
    gp = GPU_PARAMS[gpu_mode]
    is_first = (task_idx == 0)

    block = f"""
# ── TASK {task_idx+1}: {task_name} ──────────────────────────────────────────
if [ "$GPU_MODE" = "t4_2gpu" ]; then
    BSZ={gp["bsz"]}; GA={gp["ga"]}; EVAL_BSZ={gp["eval_bsz"]}
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ={gp["bsz"]}; GA={gp["ga"]}; EVAL_BSZ={gp["eval_bsz"]}
else
    BSZ={gp["bsz"]}; GA={gp["ga"]}; EVAL_BSZ={gp["eval_bsz"]}
fi
"""
    if is_first:
        block += f"""
CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \\
    --do_train --do_predict --predict_with_generate \\
    --model_name_or_path $2 \\
    --data_dir CL_Benchmark \\
    --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \\
    --task_config_dir configs/{config_dir} \\
    --output_dir logs_and_outputs/long_order3_t5_srt_{mode}/outputs/{out_dir} \\
    --per_device_train_batch_size $BSZ --per_device_eval_batch_size $EVAL_BSZ \\
    --gradient_accumulation_steps $GA --learning_rate 0.0003 --num_train_epochs 10 \\
    --run_name long_order3_t5_srt_{mode} \\
    --max_source_length 512 --max_target_length 50 --generation_max_length 50 \\
    --add_task_name False --add_dataset_name False --gradient_checkpointing \\
    --overwrite_output_dir --overwrite_cache \\
    --lr_scheduler_type constant --warmup_steps 0 \\
    --logging_strategy steps --logging_steps 10 \\
    --metric_for_best_model eval_exact_match_for_{task_name} \\
    --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \\
    --load_best_model_at_end \\
    --lora_r 8 --lora_alpha 32 --lora_dropout 0.0 \\
    --add_instruction_replay --data_replay_freq -1 --replay_after_n_epoch 0 \\
    --mlp_hidden_dim 100 --model_name gainlora_inflora \\
    --threshold 0.995 --transthreshold 0.995 \\
    $FP16_FLAG $SRT_FLAGS

rm -rf logs_and_outputs/long_order3_t5_srt_{mode}/outputs/{out_dir}/checkpoint*
sleep 5"""
    else:
        lora_list = prev_lora_paths(task_idx, mode)
        block += f"""
CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \\
    --do_train --do_predict --predict_with_generate \\
    --model_name_or_path $2 \\
    --load_checkpoint_from logs_and_outputs/long_order3_t5_srt_{mode}/outputs/{srt_load}/saved_weights/trans_input.pt \\
    --previous_lora_path {lora_list} \\
    --previous_prompt_key_path logs_and_outputs/long_order3_t5_srt_{mode}/outputs/{srt_load}/saved_weights/prompts_keys_till_now.pt \\
    --data_dir CL_Benchmark \\
    --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \\
    --gen_data_dir generated_data/lora_gen_long_t5 \\
    --task_config_dir configs/{config_dir} \\
    --output_dir logs_and_outputs/long_order3_t5_srt_{mode}/outputs/{out_dir} \\
    --per_device_train_batch_size $BSZ --per_device_eval_batch_size $EVAL_BSZ \\
    --gradient_accumulation_steps $GA --learning_rate 0.0003 --num_train_epochs 10 \\
    --run_name long_order3_t5_srt_{mode} \\
    --max_source_length 512 --max_target_length 50 --generation_max_length 50 \\
    --add_task_name False --add_dataset_name False --gradient_checkpointing \\
    --overwrite_output_dir --overwrite_cache \\
    --lr_scheduler_type constant --warmup_steps 0 \\
    --logging_strategy steps --logging_steps 10 \\
    --metric_for_best_model eval_exact_match_for_{task_name} \\
    --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \\
    --load_best_model_at_end \\
    --lora_r 8 --lora_alpha 32 --lora_dropout 0.0 \\
    --data_replay_freq -1 --kl_ratio 0.1 --attn_temperature 1 \\
    --mlp_hidden_dim 100 --model_name gainlora_inflora \\
    --threshold 0.995 --transthreshold 0.995 \\
    $FP16_FLAG $SRT_FLAGS \\
    --srt_load_path logs_and_outputs/long_order3_t5_srt_{mode}/outputs/{srt_load}/saved_weights

rm -rf logs_and_outputs/long_order3_t5_srt_{mode}/outputs/{out_dir}/checkpoint*
sleep 5"""
    return block


def generate_script(mode):
    # Mode maps to PooledMahalanobis shrinkage method
    srt_flags = {
        "ridge":    "--use_srt_router --srt_shrinkage ridge --srt_max_emb_samples 500",
        "oas":      "--use_srt_router --srt_shrinkage oas --srt_max_emb_samples 500",
        "lw":       "--use_srt_router --srt_shrinkage lw --srt_max_emb_samples 500",
        "none":     "--use_srt_router --srt_shrinkage none --srt_max_emb_samples 500",
    }[mode]

    mode_desc = {
        "ridge":    "PooledMahalanobis + Ridge δ*=d/(n+d) (recommended)",
        "oas":      "PooledMahalanobis + OAS shrinkage (Chen et al., 2010)",
        "lw":       "PooledMahalanobis + Ledoit-Wolf (2004)",
        "none":     "PooledMahalanobis + no shrinkage",
    }[mode]

    script = f'''#!/bin/bash
#SBATCH -J srt_{mode}
#SBATCH -o srt_{mode}-%j.out
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
    IS_T4=1; echo "[GPU] Detected T4 GPUs (${{GPU_MEM}}MB)"
else
    IS_T4=0; echo "[GPU] Detected high-memory GPUs (${{GPU_MEM}}MB)"
fi

if [ "$IS_T4" -eq 1 ] && [ "$NUM_GPUS" -ge 2 ]; then
    GPU_MODE="t4_2gpu"; GPU_IDS="0,1"; FP16_FLAG="--gradient_checkpointing"
elif [ "$IS_T4" -eq 1 ]; then
    GPU_MODE="t4_1gpu"; GPU_IDS="${{1:-0}}"; FP16_FLAG="--gradient_checkpointing"
else
    GPU_MODE="a100"; GPU_IDS="${{1:-0}}"; FP16_FLAG=""
fi

echo "[GPU] CUDA_VISIBLE_DEVICES=$GPU_IDS, mode=$GPU_MODE"
echo "============================================================"
echo ""

# SRT {mode} mode: {mode_desc}
SRT_FLAGS="{srt_flags}"

'''
    for task_idx, (tname, odir, sload, cdir) in enumerate(TASKS):
        script += task_block(tname, task_idx, odir, sload, cdir, "a100", mode)

    script += f"""
python score.py long_order3_t5_srt_{mode} long_order3_t5_srt_{mode}
"""
    return script


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("ridge", "oas", "lw", "none", "all"):
        print("Usage: python generate_srt_order3.py <ridge|oas|lw|none|all>")
        print("  ridge → PooledMahalanobis + Ridge δ*=d/(n+d) [RECOMMENDED]")
        print("  oas   → PooledMahalanobis + Oracle-Approximating Shrinkage")
        print("  lw    → PooledMahalanobis + Ledoit-Wolf")
        print("  none  → PooledMahalanobis + no shrinkage")
        print("  all   → generate all four scripts")
        sys.exit(1)

    modes = ["ridge", "oas", "lw", "none"] if sys.argv[1] == "all" else [sys.argv[1]]

    for mode in modes:
        script = generate_script(mode)
        out_path = f"gen_script_long_order3_t5_srt_{mode}.sh"
        with open(out_path, "w") as f:
            f.write(script)
        os.chmod(out_path, 0o755)
        print(f"Generated: {out_path}")

    print()
    print("Usage:")
    print("  bash gen_script_long_order3_t5_srt_hard.sh google/flan-t5-small")
    print("  bash gen_script_long_order3_t5_srt_dynamics.sh google/flan-t5-small")
