#!/usr/bin/env python3
"""
Generate SpecRoute shell scripts for Llama experiments.
Creates gen_script_superni_order{1,2}_llama_specroute.sh

Key differences from ROOT gainlora_inflora scripts:
- model_name = specroute
- No --load_checkpoint_from (no trans_input)
- No --add_instruction_replay
- No --kl_ratio, --attn_temperature, --trans_hidden_dim, --transthreshold, --attn_lr
- --previous_prompt_key_path replaced with spectral signatures path (reusing same flag name)
- Same BSZ x GA = 32 per task, same epochs, lr, threshold
"""

import os

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

TASK_ORDER_2 = [
    "task748_glucose_reverse_cause_event_detection",
    "task073_commonsenseqa_answer_generation",
    "task1590_diplomacy_text_generation",
    "task639_multi_woz_user_utterance_generation",
    "task1572_samsum_summary",
    "task1687_sentiment140_classification",
    "task591_sciq_answer_generation",
    "task363_sst2_polarity_classification",
    "task1510_evalution_relation_extraction",
    "task1729_personachat_generate_next",
    "task181_outcome_extraction",
    "task511_reddit_tifu_long_text_summarization",
    "task002_quoref_answer_generation",
    "task1290_xsum_summarization",
    "task875_emotion_classification",
]

# BSZ per task for ORDER 1 (same as ROOT; BSZ x GA = 32 always)
BSZ_ORDER_1 = [2, 16, 1, 4, 1, 16, 16, 2, 8, 2, 4, 1, 8, 8, 4]
# BSZ per task for ORDER 2 (from ROOT order2 script)
BSZ_ORDER_2 = [4, 8, 2, 16, 2, 8, 8, 16, 16, 2, 4, 1, 1, 1, 4]


def generate_script(task_order, bsz_list, order_num, config_dir_prefix):
    script_name = f"gen_script_superni_order{order_num}_llama_specroute"
    task_order_str = ",".join(task_order)
    
    lines = []
    lines.append("#!/bin/bash")
    lines.append("#SBATCH -J cl")
    lines.append("#SBATCH -o cl-%j.out")
    lines.append("#SBATCH -p compute")
    lines.append("#SBATCH -N 1")
    lines.append("#SBATCH -t 20:00:00")
    lines.append("#SBATCH --mem 128G")
    lines.append("#SBATCH --gres=gpu:a100-sxm4-80gb:1")
    lines.append("")
    lines.append('export CUDA_DEVICE_ORDER="PCI_BUS_ID"')
    lines.append("")

    for task_idx, task_name in enumerate(task_order):
        task_num = task_idx + 1
        bsz = bsz_list[task_idx]
        ga = 32 // bsz
        output_dir = f"logs_and_outputs/{script_name}/outputs/{task_num}-{task_name}"
        
        # Build previous_lora_path (comma-separated list of all previous tasks' saved_weights)
        if task_idx > 0:
            prev_paths = []
            for prev_idx in range(task_idx):
                prev_name = task_order[prev_idx]
                prev_paths.append(f"logs_and_outputs/{script_name}/outputs/{prev_idx+1}-{prev_name}/saved_weights")
            previous_lora_path = ",".join(prev_paths)
            # Use previous task's spectral signatures
            prev_task_name = task_order[task_idx - 1]
            spectral_sig_path = f"logs_and_outputs/{script_name}/outputs/{task_idx}-{prev_task_name}/saved_weights/spectral_signatures.pt"
        
        # metric_for_best_model
        if task_num == 1:
            metric = "eval_rougeL"
        else:
            metric = f"eval_rougeL_for_{task_name}"
        
        lines.append(f"deepspeed --include localhost:${{1}} --master_port 49500 src/run_llama.py \\")
        lines.append("   --do_train \\")
        lines.append("   --do_predict \\")
        lines.append("   --predict_with_generate \\")
        lines.append("   --model_name_or_path $2 \\")
        
        if task_idx > 0:
            lines.append(f"   --previous_lora_path {previous_lora_path} \\")
            lines.append(f"   --previous_prompt_key_path {spectral_sig_path} \\")
        
        lines.append("   --data_dir CL_Benchmark \\")
        lines.append(f"   --task_order {task_order_str} \\")
        
        if task_idx > 0:
            lines.append(f"   --gen_data_dir generated_data/lora_gen_superni_llama \\")
        
        lines.append(f"   --task_config_dir configs/{config_dir_prefix}/{task_name} \\")
        lines.append(f"   --output_dir {output_dir} \\")
        lines.append(f"   --per_device_train_batch_size {bsz} \\")
        lines.append("   --per_device_eval_batch_size 8 \\")
        lines.append(f"   --gradient_accumulation_steps {ga} \\")
        lines.append("   --learning_rate 5e-05 \\")
        lines.append("   --num_train_epochs 50 \\")
        lines.append("   --bf16 \\")
        lines.append("   --deepspeed configs/ds_configs/stage2.config \\")
        lines.append(f"   --run_name {script_name} \\")
        lines.append("   --max_source_length 1024 \\")
        lines.append("   --max_target_length 50 \\")
        lines.append("   --generation_max_length 50 \\")
        lines.append("   --add_task_name False \\")
        lines.append("   --add_dataset_name False \\")
        lines.append("   --overwrite_output_dir \\")
        lines.append("   --overwrite_cache \\")
        lines.append("   --lr_scheduler_type constant \\")
        lines.append("   --warmup_steps 0 \\")
        lines.append("   --logging_strategy steps \\")
        lines.append("   --logging_steps 10 \\")
        lines.append(f"   --metric_for_best_model {metric} \\")
        lines.append("   --evaluation_strategy steps \\")
        lines.append("   --save_strategy steps \\")
        lines.append("   --save_total_limit 1 \\")
        lines.append("   --load_best_model_at_end \\")
        lines.append("   --lora_r 4 \\")
        lines.append("   --lora_alpha 32 \\")
        lines.append("   --lora_dropout 0.0 \\")
        lines.append("   --data_replay_freq -1 \\")
        lines.append("   --chunk 4 \\")
        lines.append("   --model_name specroute \\")
        lines.append("   --threshold 0.995 ")
        lines.append("")
        lines.append(f"rm -rf {output_dir}/checkpoint*")
        lines.append("")
        lines.append("sleep 5")
        lines.append("")
        lines.append("")

    lines.append(f"python score.py {script_name} {script_name}")
    lines.append("")
    
    return script_name, "\n".join(lines)


if __name__ == "__main__":
    for order_num, task_order, bsz_list in [
        (1, TASK_ORDER_1, BSZ_ORDER_1),
        (2, TASK_ORDER_2, BSZ_ORDER_2),
    ]:
        config_dir_prefix = f"gen_script_superni_order{order_num}_llama_configs"
        script_name, content = generate_script(task_order, bsz_list, order_num, config_dir_prefix)
        output_path = f"{script_name}.sh"
        with open(output_path, "w") as f:
            f.write(content)
        os.chmod(output_path, 0o755)
        print(f"Generated {output_path}")
