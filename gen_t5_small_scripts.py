#!/usr/bin/env python3
"""
Generate T5_small benchmark scripts from original T5 scripts.

For all scripts:
  - Rename experiment: t5_METHOD -> t5_small_METHOD  (output_dir, run_name, checkpoint paths)
  - Remove --gradient_checkpointing standalone flag (not needed for small model)

Batch sizes (flan-t5-small fits easily on T4 with large batches):
  Long sequence non-specroute : train=32, ga=1, eval=256
  SuperNI non-specroute        : train=16, ga=2, eval=8

Specroute GPU-mode blocks (long):
  t4_2gpu : BSZ=16; GA=1; EVAL_BSZ=256
  t4_1gpu : BSZ=32; GA=1; EVAL_BSZ=256
  a100    : BSZ=64; GA=1; EVAL_BSZ=512

Specroute GPU-mode blocks (superni):
  t4_2gpu : BSZ=8 ; GA=2; EVAL_BSZ=16
  t4_1gpu : BSZ=16; GA=2; EVAL_BSZ=16
  a100    : BSZ=32; GA=1; EVAL_BSZ=32
"""

import re
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SPECROUTE_LONG_MODES = {
    "t4_2gpu": "BSZ=16; GA=1; EVAL_BSZ=256",
    "t4_1gpu": "BSZ=32; GA=1; EVAL_BSZ=256",
    "a100":    "BSZ=64; GA=1; EVAL_BSZ=512",
}

SPECROUTE_SUPERNI_MODES = {
    "t4_2gpu": "BSZ=8;  GA=2; EVAL_BSZ=16",
    "t4_1gpu": "BSZ=16; GA=2; EVAL_BSZ=16",
    "a100":    "BSZ=32; GA=1; EVAL_BSZ=32",
}


def replace_experiment_names(content: str) -> str:
    """Replace gen_script_X_t5_METHOD  →  gen_script_X_t5_small_METHOD."""
    # Works in output_dir, run_name, checkpoint paths
    return re.sub(
        r'(gen_script_(?:long_order[34]|superni_order[12])_t5_)(?!small_)',
        r'\1small_',
        content,
    )


def remove_gradient_checkpointing_flag(content: str) -> str:
    """Remove   --gradient_checkpointing \\   as a standalone argument line."""
    # Matches lines like:  "   --gradient_checkpointing \\\n"  or  "   --gradient_checkpointing\n"
    content = re.sub(r'[ \t]*--gradient_checkpointing \\\n', '', content)
    content = re.sub(r'[ \t]*--gradient_checkpointing\n', '', content)
    return content


def set_non_specroute_batch_sizes(content: str, script_type: str) -> str:
    """Replace hardcoded --per_device_* and --gradient_accumulation_steps."""
    if script_type == 'long':
        train_bsz, ga, eval_bsz = 32, 1, 256
    else:  # superni
        train_bsz, ga, eval_bsz = 16, 2, 8

    content = re.sub(r'--per_device_train_batch_size \d+',
                     f'--per_device_train_batch_size {train_bsz}', content)
    content = re.sub(r'--per_device_eval_batch_size \d+',
                     f'--per_device_eval_batch_size {eval_bsz}', content)
    content = re.sub(r'--gradient_accumulation_steps \d+',
                     f'--gradient_accumulation_steps {ga}', content)
    return content


def fix_specroute_gpu_modes(content: str, script_type: str) -> str:
    """Replace BSZ/GA/EVAL_BSZ inside the GPU-mode if/elif/else block."""
    modes = SPECROUTE_LONG_MODES if script_type == 'long' else SPECROUTE_SUPERNI_MODES

    # t4_2gpu block: "    BSZ=N; GA=N; EVAL_BSZ=N"
    content = re.sub(
        r'(if \[ "\$GPU_MODE" = "t4_2gpu" \]; then\n)[ \t]*BSZ=\d+; GA=\d+; EVAL_BSZ=\d+',
        r'\g<1>    ' + modes['t4_2gpu'],
        content,
    )
    # t4_1gpu block
    content = re.sub(
        r'(elif \[ "\$GPU_MODE" = "t4_1gpu" \]; then\n)[ \t]*BSZ=\d+; GA=\d+; EVAL_BSZ=\d+',
        r'\g<1>    ' + modes['t4_1gpu'],
        content,
    )
    # a100 block (else)
    content = re.sub(
        r'(else\n)[ \t]*BSZ=\d+; GA=\d+; EVAL_BSZ=\d+',
        r'\g<1>    ' + modes['a100'],
        content,
    )

    # Remove gradient_checkpointing from FP16_FLAG  (set it to empty for all modes)
    content = re.sub(r'FP16_FLAG="--gradient_checkpointing"', 'FP16_FLAG=""', content)
    return content


def transform(content: str, is_specroute: bool, script_type: str) -> str:
    content = replace_experiment_names(content)
    content = remove_gradient_checkpointing_flag(content)
    if is_specroute:
        content = fix_specroute_gpu_modes(content, script_type)
    else:
        content = set_non_specroute_batch_sizes(content, script_type)
    return content


def process_dir(src_dir: Path, dst_dir: Path, scripts_long: list, scripts_superni: list):
    dst_dir.mkdir(exist_ok=True)

    for script in scripts_long:
        src = src_dir / script
        if not src.exists():
            print(f"  SKIP (not found): {src}")
            continue
        content = src.read_text()
        is_specroute = 'specroute' in script
        new_content = transform(content, is_specroute, 'long')
        new_name = script.replace('_t5_', '_t5_small_')
        dst = dst_dir / new_name
        dst.write_text(new_content)
        os.chmod(dst, 0o755)
        print(f"  Created  {dst.relative_to(src_dir.parent.parent)}")

    for script in scripts_superni:
        src = src_dir / script
        if not src.exists():
            print(f"  SKIP (not found): {src}")
            continue
        content = src.read_text()
        is_specroute = 'specroute' in script
        new_content = transform(content, is_specroute, 'superni')
        new_name = script.replace('_t5_', '_t5_small_')
        dst = dst_dir / new_name
        dst.write_text(new_content)
        os.chmod(dst, 0o755)
        print(f"  Created  {dst.relative_to(src_dir.parent.parent)}")


# ---------------------------------------------------------------------------
# Root gainlora
# ---------------------------------------------------------------------------
ROOT = Path('/Users/nnminh322/Desktop/personal/Continual/root_gainlora')

ROOT_LONG = [
    'gen_script_long_order3_t5_inflora.sh',
    'gen_script_long_order3_t5_gainlora_inflora.sh',
    'gen_script_long_order4_t5_inflora.sh',
    'gen_script_long_order4_t5_gainlora_inflora.sh',
]
ROOT_SUPERNI = [
    'gen_script_superni_order1_t5_inflora.sh',
    'gen_script_superni_order1_t5_gainlora_inflora.sh',
    'gen_script_superni_order2_t5_inflora.sh',
    'gen_script_superni_order2_t5_gainlora_inflora.sh',
]

print("=== root_gainlora/T5_small/ ===")
process_dir(ROOT, ROOT / 'T5_small', ROOT_LONG, ROOT_SUPERNI)

# ---------------------------------------------------------------------------
# Improve gainlora
# ---------------------------------------------------------------------------
IMPROVE = Path('/Users/nnminh322/Desktop/personal/Continual/improve_gainlora')

IMPROVE_LONG = [
    'gen_script_long_order3_t5_inflora.sh',
    'gen_script_long_order3_t5_gainlora_inflora.sh',
    'gen_script_long_order3_t5_specroute.sh',
    'gen_script_long_order4_t5_inflora.sh',
    'gen_script_long_order4_t5_gainlora_inflora.sh',
    'gen_script_long_order4_t5_specroute.sh',
]
IMPROVE_SUPERNI = [
    'gen_script_superni_order1_t5_inflora.sh',
    'gen_script_superni_order1_t5_gainlora_inflora.sh',
    'gen_script_superni_order1_t5_specroute.sh',
    'gen_script_superni_order2_t5_inflora.sh',
    'gen_script_superni_order2_t5_gainlora_inflora.sh',
    'gen_script_superni_order2_t5_specroute.sh',
]

print("\n=== improve_gainlora/T5_small/ ===")
process_dir(IMPROVE, IMPROVE / 'T5_small', IMPROVE_LONG, IMPROVE_SUPERNI)

print("\nDone!")
