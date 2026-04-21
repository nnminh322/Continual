#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path


TASK_ORDER_1 = [
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

BSZ_ORDER_1 = [2, 16, 1, 4, 1, 16, 16, 2, 8, 2, 4, 1, 8, 8, 4]
RUN_NAME = "gen_script_superni_order1_llama_gainlora_inflora_rootbugfix"


def build_task_command(task_idx: int, task_name: str, task_order_str: str) -> str:
    task_num = task_idx + 1
    bsz = BSZ_ORDER_1[task_idx]
    grad_accum = 32 // bsz
    metric = "eval_rougeL" if task_idx == 0 else f"eval_rougeL_for_{task_name}"
    output_dir = f'$BASE_OUT/outputs/{task_num}-{task_name}'
    config_dir = f'$ROOT_BASE/configs/gen_script_superni_order1_llama_configs/{task_name}'

    cmd_parts = [
        '${DEEPSPEED_BIN} --include localhost:${GPU_IDS} --master_port ${MASTER_PORT} "$ROOT_BASE/src/run_llama.py"',
        '   --do_train',
        '   --do_predict',
        '   --predict_with_generate',
        '   --model_name_or_path "$MODEL_PATH"',
        '   --data_dir "$DATA_DIR"',
        f'   --task_order {task_order_str}',
        f'   --task_config_dir "{config_dir}"',
        f'   --output_dir "{output_dir}"',
        f'   --per_device_train_batch_size {bsz}',
        '   --per_device_eval_batch_size 8',
        f'   --gradient_accumulation_steps {grad_accum}',
        '   --learning_rate 5e-05',
        '   --num_train_epochs 50',
        '   --bf16',
        '   --deepspeed "$ROOT_BASE/configs/ds_configs/stage2.config"',
        '   --run_name "$RUN_NAME"',
        '   --max_source_length 1024',
        '   --max_target_length 50',
        '   --generation_max_length 50',
        '   --add_task_name False',
        '   --add_dataset_name False',
        '   --overwrite_output_dir',
        '   --overwrite_cache',
        '   --lr_scheduler_type constant',
        '   --warmup_steps 0',
        '   --logging_strategy steps',
        '   --logging_steps 10',
        f'   --metric_for_best_model {metric}',
        '   --eval_strategy steps',
        '   --save_strategy steps',
        '   --save_total_limit 1',
        '   --load_best_model_at_end',
        '   --lora_r 4',
        '   --lora_alpha 32',
        '   --lora_dropout 0.0',
        '   --data_replay_freq -1',
        '   --replay_after_n_epoch 0',
        '   --kl_ratio 0.5',
        '   --attn_temperature 1',
        '   --mlp_hidden_dim 100',
        '   --trans_hidden_dim 100',
        '   --attn_lr 0',
        '   --chunk 4',
        '   --model_name gainlora_inflora',
        '   --threshold 0.995',
        '   --transthreshold 0.995',
    ]

    if task_idx > 0:
        prev_saved_dirs = [
            f'$BASE_OUT/outputs/{prev_idx + 1}-{TASK_ORDER_1[prev_idx]}/saved_weights'
            for prev_idx in range(task_idx)
        ]
        prev_task = TASK_ORDER_1[task_idx - 1]
        cmd_parts.extend(
            [
                f'   --previous_lora_path "{",".join(prev_saved_dirs)}"',
                f'   --previous_prompt_key_path "$BASE_OUT/outputs/{task_idx}-{prev_task}/saved_weights/prompts_keys_till_now.pt"',
                f'   --load_checkpoint_from "$BASE_OUT/outputs/{task_idx}-{prev_task}/saved_weights/trans_input.pt"',
            ]
        )

    command = " \\\n".join(cmd_parts)
    cleanup = f'rm -rf "{output_dir}"/checkpoint*'
    return f"{command}\n\n{cleanup}\n"


def build_script() -> str:
    task_order_str = ",".join(TASK_ORDER_1)
    header = """#!/bin/bash
# GENERATED FILE. DO NOT EDIT.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
ROOT_BASE="$WORKSPACE_DIR/llama_epoch_ablation/root_gainlora_bugfix"
DATA_DIR="$WORKSPACE_DIR/root_gainlora/CL_Benchmark"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-deepspeed}"
MODEL_PATH="${1:-meta-llama/Llama-2-7b-hf}"
GPU_IDS="${2:-0}"
MASTER_PORT="${3:-49500}"

RUN_NAME="gen_script_superni_order1_llama_gainlora_inflora_rootbugfix"
BASE_OUT="$WORKSPACE_DIR/llama_epoch_ablation/logs_and_outputs/$RUN_NAME"

mkdir -p "$BASE_OUT/outputs"

echo "[ROOT-ORDER1] model_path=$MODEL_PATH"
echo "[ROOT-ORDER1] gpu_ids=$GPU_IDS"
echo "[ROOT-ORDER1] master_port=$MASTER_PORT"
echo "[ROOT-ORDER1] output_root=$BASE_OUT"
echo "============================================================"

"""
    body = "\n".join(build_task_command(idx, task_name, task_order_str) for idx, task_name in enumerate(TASK_ORDER_1))
    footer = """
"$PYTHON_BIN" "$ROOT_BASE/score.py" "$RUN_NAME" "$RUN_NAME" "$WORKSPACE_DIR/llama_epoch_ablation/logs_and_outputs"
"""
    return header + body + footer


def main() -> None:
    ablation_dir = Path(__file__).resolve().parent
    output_dir = ablation_dir / "generated_scripts"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{RUN_NAME}.sh"
    output_path.write_text(build_script(), encoding="utf-8")
    output_path.chmod(0o755)
    print(output_path)


if __name__ == "__main__":
    main()
