#!/usr/bin/env python3
"""
Generate SpecRoute bash scripts with proper 2-GPU T4 parallelism.

Changes from v1:
- Auto-detect number of GPUs AND GPU type (T4 vs A100)
- If 2+ T4 GPUs: CUDA_VISIBLE_DEVICES=0,1, DataParallel, fp16
- Reduced eval_batch_size for T4 (avoid OOM during eval)
- Proper batch size scaling for 2-GPU DataParallel

Usage: python generate_specroute_scripts_v2.py
"""

# ========================= DATASET CONFIGS =========================

SUPERNI_ORDER1_TASKS = [
    "task1572_samsum_summary",
    "task363_sst2_polarity_classification",
    "task1290_xsum_summarization",
    "task181_outcome_extraction",
    "task002_quoref_answer_generation",
    "task1510_evalution_relation_extraction",
    "task639_multi_woz_user_utterance_generation",
    "task1729_personachat_generate_next",
    "task073_commonsenseqa_answer_generation",
    "task1590_diplomacy_text_generation",
    "task748_glucose_reverse_cause_event_detection",
    "task511_reddit_tifu_long_text_summarization",
    "task591_sciq_answer_generation",
    "task1687_sentiment140_classification",
    "task875_emotion_classification",
]

SUPERNI_ORDER2_TASKS = [
    "task748_glucose_reverse_cause_event_detection",
    "task073_commonsenseqa_answer_generation",
    "task875_emotion_classification",
    "task002_quoref_answer_generation",
    "task1687_sentiment140_classification",
    "task591_sciq_answer_generation",
    "task363_sst2_polarity_classification",
    "task1572_samsum_summary",
    "task511_reddit_tifu_long_text_summarization",
    "task1290_xsum_summarization",
    "task639_multi_woz_user_utterance_generation",
    "task1510_evalution_relation_extraction",
    "task181_outcome_extraction",
    "task1729_personachat_generate_next",
    "task1590_diplomacy_text_generation",
]

LONG_ORDER3_TASKS = [
    "yelp", "amazon", "mnli", "cb", "copa",
    "qqp", "rte", "imdb", "sst2", "dbpedia",
    "agnews", "yahoo", "multirc", "boolq", "wic",
]

LONG_ORDER4_TASKS = [
    "mnli", "cb", "wic", "copa", "qqp",
    "boolq", "rte", "imdb", "yelp", "amazon",
    "sst2", "dbpedia", "agnews", "multirc", "yahoo",
]


SCRIPT_CONFIGS = {
    "superni_order1": {
        "tasks": SUPERNI_ORDER1_TASKS,
        "config_dir": "gen_script_superni_order1_t5_configs",
        "run_name": "gen_script_superni_order1_t5_specroute",
        "lora_r": 4,
        "epochs": 100,
        "metric_base": "eval_rougeL",
        "do_predict": True,
        "cleanup_checkpoints": False,
        # (per_device_bsz, grad_accum)
        "batch_a100_task1": (16, 2),
        "batch_a100_rest": (32, 1),
        "batch_t4_1gpu_task1": (8, 4),
        "batch_t4_1gpu_rest": (8, 4),
        "batch_t4_2gpu_task1": (4, 4),  # 4*2gpu*4=32 effective
        "batch_t4_2gpu_rest": (4, 4),   # 4*2gpu*4=32 effective
        "eval_batch_a100": 4,
        "eval_batch_t4": 4,
    },
    "superni_order2": {
        "tasks": SUPERNI_ORDER2_TASKS,
        "config_dir": "gen_script_superni_order2_t5_configs",
        "run_name": "gen_script_superni_order2_t5_specroute",
        "lora_r": 4,
        "epochs": 100,
        "metric_base": "eval_rougeL",
        "do_predict": True,
        "cleanup_checkpoints": False,
        "batch_a100_task1": (16, 2),
        "batch_a100_rest": (32, 1),
        "batch_t4_1gpu_task1": (8, 4),
        "batch_t4_1gpu_rest": (8, 4),
        "batch_t4_2gpu_task1": (4, 4),
        "batch_t4_2gpu_rest": (4, 4),
        "eval_batch_a100": 4,
        "eval_batch_t4": 4,
    },
    "long_order3": {
        "tasks": LONG_ORDER3_TASKS,
        "config_dir": "gen_script_long_order3_t5_configs",
        "run_name": "gen_script_long_order3_t5_specroute",
        "lora_r": 8,
        "epochs": 10,
        "metric_base": "eval_exact_match",
        "do_predict": True,
        "cleanup_checkpoints": True,
        "batch_a100_task1": (8, 4),
        "batch_a100_rest": (16, 2),
        "batch_t4_1gpu_task1": (4, 8),
        "batch_t4_1gpu_rest": (4, 8),
        "batch_t4_2gpu_task1": (2, 8),  # 2*2gpu*8=32 effective
        "batch_t4_2gpu_rest": (2, 4),   # 2*2gpu*4=16 effective
        "eval_batch_a100": 128,
        "eval_batch_t4": 16,   # reduced for T4 to avoid OOM
    },
    "long_order4": {
        "tasks": LONG_ORDER4_TASKS,
        "config_dir": "gen_script_long_order4_t5_configs",
        "run_name": "gen_script_long_order4_t5_specroute",
        "lora_r": 8,
        "epochs": 10,
        "metric_base": "eval_exact_match",
        "do_predict": True,
        "cleanup_checkpoints": True,
        "batch_a100_task1": (8, 4),
        "batch_a100_rest": (16, 2),
        "batch_t4_1gpu_task1": (4, 8),
        "batch_t4_1gpu_rest": (4, 8),
        "batch_t4_2gpu_task1": (2, 8),
        "batch_t4_2gpu_rest": (2, 4),
        "eval_batch_a100": 128,
        "eval_batch_t4": 16,
    },
}


def generate_header():
    """Generate script header with GPU auto-detection for 2-GPU T4."""
    return '''#!/bin/bash
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
    FP16_FLAG="--fp16"
    echo "[GPU] Strategy: 2x T4 DataParallel + fp16"
elif [ "$IS_T4" -eq 1 ]; then
    GPU_MODE="t4_1gpu"
    GPU_IDS="${1:-0}"
    FP16_FLAG="--fp16"
    echo "[GPU] Strategy: 1x T4 + fp16"
else
    GPU_MODE="a100"
    GPU_IDS="${1:-0}"
    FP16_FLAG=""
    echo "[GPU] Strategy: A100 (single GPU, fp32)"
fi

echo "[GPU] Using CUDA_VISIBLE_DEVICES=$GPU_IDS"
echo "============================================================"
echo ""
'''


def generate_batch_selection(cfg, is_task1):
    """Generate batch size selection block."""
    if is_task1:
        a100_bsz, a100_ga = cfg["batch_a100_task1"]
        t4_1_bsz, t4_1_ga = cfg["batch_t4_1gpu_task1"]
        t4_2_bsz, t4_2_ga = cfg["batch_t4_2gpu_task1"]
    else:
        a100_bsz, a100_ga = cfg["batch_a100_rest"]
        t4_1_bsz, t4_1_ga = cfg["batch_t4_1gpu_rest"]
        t4_2_bsz, t4_2_ga = cfg["batch_t4_2gpu_rest"]

    eval_a100 = cfg["eval_batch_a100"]
    eval_t4 = cfg["eval_batch_t4"]

    return f'''if [ "$GPU_MODE" = "t4_2gpu" ]; then
    BSZ={t4_2_bsz}; GA={t4_2_ga}; EVAL_BSZ={eval_t4}
elif [ "$GPU_MODE" = "t4_1gpu" ]; then
    BSZ={t4_1_bsz}; GA={t4_1_ga}; EVAL_BSZ={eval_t4}
else
    BSZ={a100_bsz}; GA={a100_ga}; EVAL_BSZ={eval_a100}
fi
'''


def generate_task_block(cfg, task_idx, task_name, tasks, output_base):
    """Generate a single task training block."""
    lines = []

    # Batch size selection
    is_task1 = (task_idx == 0)
    lines.append(generate_batch_selection(cfg, is_task1))

    # Build previous_lora_path
    prev_paths = []
    for j in range(task_idx):
        prev_paths.append(
            f"logs_and_outputs/{output_base}/outputs/{j+1}-{tasks[j]}/saved_weights"
        )

    # Build metric name
    if task_idx == 0:
        metric = cfg["metric_base"]
    else:
        metric = f"{cfg['metric_base']}_for_{task_name}"

    # Build command
    lines.append(f'CUDA_VISIBLE_DEVICES=$GPU_IDS python src/run_t5.py \\')
    lines.append(f'   --do_train \\')
    if cfg["do_predict"]:
        lines.append(f'   --do_predict \\')
    lines.append(f'   --predict_with_generate \\')
    lines.append(f'   --model_name_or_path $2 \\')

    if prev_paths:
        lines.append(f'   --previous_lora_path {",".join(prev_paths)} \\')

    task_order_str = ",".join(tasks)
    lines.append(f'   --data_dir CL_Benchmark \\')
    lines.append(f'   --task_order {task_order_str} \\')
    lines.append(
        f'   --task_config_dir configs/{cfg["config_dir"]}/{task_name} \\'
    )
    lines.append(
        f'   --output_dir logs_and_outputs/{output_base}/outputs/{task_idx+1}-{task_name} \\'
    )
    lines.append(f'   --per_device_train_batch_size $BSZ \\')
    lines.append(f'   --per_device_eval_batch_size $EVAL_BSZ \\')
    lines.append(f'   --gradient_accumulation_steps $GA \\')
    lines.append(f'   --learning_rate 0.0003 \\')
    lines.append(f'   --num_train_epochs {cfg["epochs"]} \\')
    lines.append(f'   --run_name {cfg["run_name"]} \\')
    lines.append(f'   --max_source_length 512 \\')
    lines.append(f'   --max_target_length 50 \\')
    lines.append(f'   --generation_max_length 50 \\')
    lines.append(f'   --add_task_name False \\')
    lines.append(f'   --add_dataset_name False \\')
    lines.append(f'   --overwrite_output_dir \\')
    lines.append(f'   --overwrite_cache \\')
    lines.append(f'   --lr_scheduler_type constant \\')
    lines.append(f'   --warmup_steps 0 \\')
    lines.append(f'   --logging_strategy steps \\')
    lines.append(f'   --logging_steps 10 \\')
    lines.append(f'   --metric_for_best_model {metric} \\')
    lines.append(f'   --evaluation_strategy steps \\')
    lines.append(f'   --save_strategy steps \\')
    lines.append(f'   --save_total_limit 1 \\')
    lines.append(f'   --load_best_model_at_end \\')
    lines.append(f'   --lora_r {cfg["lora_r"]} \\')
    lines.append(f'   --lora_alpha 32 \\')
    lines.append(f'   --lora_dropout 0.0 \\')

    if is_task1:
        lines.append(f'   --run_single True \\')

    lines.append(f'   --data_replay_freq -1 \\')
    lines.append(f'   --mlp_hidden_dim 100 \\')
    lines.append(f'   --model_name specroute \\')
    lines.append(f'   --threshold 0.995 \\')
    lines.append(f'   --transthreshold 0.995 \\')
    lines.append(f'   $FP16_FLAG')

    # Cleanup checkpoints (Long benchmark)
    if cfg["cleanup_checkpoints"]:
        lines.append('')
        lines.append(
            f'rm -rf logs_and_outputs/{output_base}/outputs/'
            f'{task_idx+1}-{task_name}/checkpoint*'
        )
        lines.append('')
        lines.append('sleep 5')

    return "\n".join(lines)


def generate_script(config_key):
    """Generate a full bash script for a given config."""
    cfg = SCRIPT_CONFIGS[config_key]
    tasks = cfg["tasks"]
    output_base = cfg["run_name"]

    parts = [generate_header()]

    for i, task in enumerate(tasks):
        parts.append(
            generate_task_block(cfg, i, task, tasks, output_base)
        )
        parts.append("")  # blank line between tasks

    return "\n".join(parts)


def main():
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))

    for key in SCRIPT_CONFIGS:
        script_name = f"gen_script_{key}_t5_specroute.sh"
        script_path = os.path.join(script_dir, script_name)
        content = generate_script(key)

        with open(script_path, "w") as f:
            f.write(content)

        os.chmod(script_path, 0o755)

        num_lines = content.count("\n") + 1
        print(f"Generated {script_name} ({num_lines} lines)")

    print("\nAll 4 scripts generated with 2-GPU T4 support.")
    print("Usage: bash gen_script_<name>.sh <gpu_id> <model_path>")
    print("  - With 2 T4 GPUs: auto-detects and uses both GPUs")
    print("  - With 1 GPU: falls back to single GPU mode")
    print("  - <gpu_id> is only used for single-GPU fallback")


if __name__ == "__main__":
    main()
