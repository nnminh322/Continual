#!/usr/bin/env python3
"""
Patches all T5 SpecRoute shell scripts to:
 1. T5_small scripts: Replace GPU detection block (add P100 + 3090 2-GPU + gradient_checkpointing)
 2. T5_small Long scripts: Reduce BSZ for P100/T4 (EVAL_BSZ 256->32, add p100 branch)
 3. T5_small SuperNI scripts: Add p100 BSZ branch
 4. Root T5 scripts: Add 3090 2-GPU support to a100 else branch
"""
import os

BASE = "/Users/nnminh322/Desktop/personal/Continual/improve_gainlora"

# ── Replacements for T5_small scripts ────────────────────────────────────────

# GPU detection block — Long scripts (no NOTE comment)
OLD_DETECT_LONG = (
    "# Determine GPU type\n"
    'if [ "$GPU_MEM" -lt 20000 ]; then\n'
    "    IS_T4=1\n"
    '    echo "[GPU] Detected T4 GPUs (${GPU_MEM}MB VRAM each)"\n'
    "else\n"
    "    IS_T4=0\n"
    '    echo "[GPU] Detected high-memory GPUs (${GPU_MEM}MB VRAM each)"\n'
    "fi\n"
    "\n"
    "# Determine parallelism strategy\n"
    'if [ "$IS_T4" -eq 1 ] && [ "$NUM_GPUS" -ge 2 ]; then\n'
    '    GPU_MODE="t4_2gpu"\n'
    '    GPU_IDS="0,1"\n'
    '    FP16_FLAG=""\n'
    '    echo "[GPU] Strategy: 2x T4 DataParallel + fp32 + gradient_checkpointing"\n'
    'elif [ "$IS_T4" -eq 1 ]; then\n'
    '    GPU_MODE="t4_1gpu"\n'
    '    GPU_IDS="${1:-0}"\n'
    '    FP16_FLAG=""\n'
    '    echo "[GPU] Strategy: 1x T4 + fp32 + gradient_checkpointing"\n'
    "else\n"
    '    GPU_MODE="a100"\n'
    '    GPU_IDS="${1:-0}"\n'
    '    FP16_FLAG=""\n'
    '    echo "[GPU] Strategy: A100 (single GPU, fp32)"\n'
    "fi"
)

# SuperNI T5_small scripts have this NOTE comment block instead
OLD_DETECT_SUPERNI = (
    "# NOTE: T5 models trained in bfloat16 produce NaN with fp16 (overflow).\n"
    "# T4 GPUs do not support bf16. Use fp32 + gradient_checkpointing instead.\n"
    'if [ "$IS_T4" -eq 1 ] && [ "$NUM_GPUS" -ge 2 ]; then\n'
    '    GPU_MODE="t4_2gpu"\n'
    '    GPU_IDS="0,1"\n'
    '    FP16_FLAG=""\n'
    '    echo "[GPU] Strategy: 2x T4 DataParallel + fp32 + gradient_checkpointing"\n'
    'elif [ "$IS_T4" -eq 1 ]; then\n'
    '    GPU_MODE="t4_1gpu"\n'
    '    GPU_IDS="${1:-0}"\n'
    '    FP16_FLAG=""\n'
    '    echo "[GPU] Strategy: 1x T4 + fp32 + gradient_checkpointing"\n'
    "else\n"
    '    GPU_MODE="a100"\n'
    '    GPU_IDS="${1:-0}"\n'
    '    FP16_FLAG=""\n'
    '    echo "[GPU] Strategy: A100 (single GPU, fp32)"\n'
    "fi"
)

# New unified GPU detection block for all T5_small scripts
NEW_DETECT = (
    "# GPU type detection\n"
    "# T4 <15500 MB | P100 15500-17000 MB | RTX3090 ~24576 | A100 40000 | H100 80000\n"
    'if [ "$GPU_MEM" -lt 15500 ]; then\n'
    '    GPU_TYPE="t4"\n'
    '    echo "[GPU] Detected T4 (${GPU_MEM}MB)"\n'
    'elif [ "$GPU_MEM" -le 17000 ]; then\n'
    '    GPU_TYPE="p100"\n'
    '    echo "[GPU] Detected P100 (${GPU_MEM}MB)"\n'
    "else\n"
    '    GPU_TYPE="highvram"\n'
    '    echo "[GPU] Detected high-VRAM GPU (${GPU_MEM}MB)"\n'
    "fi\n"
    "\n"
    "# Parallelism: T4/P100 use gradient_checkpointing (16 GB fp32); highvram uses DataParallel if 2+ GPUs\n"
    'if [ "$GPU_TYPE" = "t4" ] && [ "$NUM_GPUS" -ge 2 ]; then\n'
    '    GPU_MODE="t4_2gpu"\n'
    '    GPU_IDS="0,1"\n'
    '    FP16_FLAG="--gradient_checkpointing"\n'
    '    echo "[GPU] Strategy: 2x T4 DataParallel + fp32 + gradient_checkpointing"\n'
    'elif [ "$GPU_TYPE" = "t4" ]; then\n'
    '    GPU_MODE="t4_1gpu"\n'
    '    GPU_IDS="${1:-0}"\n'
    '    FP16_FLAG="--gradient_checkpointing"\n'
    '    echo "[GPU] Strategy: 1x T4 (${GPU_MEM}MB) + fp32 + gradient_checkpointing"\n'
    'elif [ "$GPU_TYPE" = "p100" ]; then\n'
    '    GPU_MODE="p100"\n'
    '    GPU_IDS="${1:-0}"\n'
    '    FP16_FLAG="--gradient_checkpointing"\n'
    '    echo "[GPU] Strategy: P100 16GB + fp32 + gradient_checkpointing"\n'
    "else\n"
    '    GPU_MODE="a100"\n'
    '    if [ "$NUM_GPUS" -ge 2 ]; then\n'
    '        GPU_IDS="0,1"\n'
    '        echo "[GPU] Strategy: ${NUM_GPUS}x ${GPU_MEM}MB DataParallel (RTX3090/A100, fp32)"\n'
    "    else\n"
    '        GPU_IDS="${1:-0}"\n'
    '        echo "[GPU] Strategy: 1x ${GPU_MEM}MB GPU (fp32)"\n'
    "    fi\n"
    '    FP16_FLAG=""\n'
    "fi"
)

# ── Per-task BSZ blocks ───────────────────────────────────────────────────────

# Long T5_small scripts: old BSZ block (with space before EVAL_BSZ=256)
OLD_BSZ_LONG_A = (
    'if [ "$GPU_MODE" = "t4_2gpu" ]; then\n'
    "    BSZ=16; GA=1; EVAL_BSZ=256\n"
    'elif [ "$GPU_MODE" = "t4_1gpu" ]; then\n'
    "    BSZ=32; GA=1; EVAL_BSZ=256\n"
    "else\n"
    "    BSZ=64; GA=1; EVAL_BSZ=512\n"
    "fi"
)

NEW_BSZ_LONG = (
    'if [ "$GPU_MODE" = "t4_2gpu" ]; then\n'
    "    BSZ=8; GA=2; EVAL_BSZ=64\n"
    'elif [ "$GPU_MODE" = "t4_1gpu" ]; then\n'
    "    BSZ=8; GA=2; EVAL_BSZ=32\n"
    'elif [ "$GPU_MODE" = "p100" ]; then\n'
    "    BSZ=16; GA=2; EVAL_BSZ=32\n"
    "else\n"
    "    BSZ=64; GA=1; EVAL_BSZ=128\n"
    "fi"
)

# SuperNI T5_small: old BSZ block (2 space variants)
OLD_BSZ_SUPERNI_A = (
    'if [ "$GPU_MODE" = "t4_2gpu" ]; then\n'
    "    BSZ=8;  GA=2; EVAL_BSZ=16\n"
    'elif [ "$GPU_MODE" = "t4_1gpu" ]; then\n'
    "    BSZ=16; GA=2; EVAL_BSZ=16\n"
    "else\n"
    "    BSZ=32; GA=1; EVAL_BSZ=32\n"
    "fi"
)

OLD_BSZ_SUPERNI_B = (
    'if [ "$GPU_MODE" = "t4_2gpu" ]; then\n'
    "    BSZ=8; GA=2; EVAL_BSZ=16\n"
    'elif [ "$GPU_MODE" = "t4_1gpu" ]; then\n'
    "    BSZ=16; GA=2; EVAL_BSZ=16\n"
    "else\n"
    "    BSZ=32; GA=1; EVAL_BSZ=32\n"
    "fi"
)

NEW_BSZ_SUPERNI = (
    'if [ "$GPU_MODE" = "t4_2gpu" ]; then\n'
    "    BSZ=8;  GA=2; EVAL_BSZ=16\n"
    'elif [ "$GPU_MODE" = "t4_1gpu" ]; then\n'
    "    BSZ=8;  GA=2; EVAL_BSZ=16\n"
    'elif [ "$GPU_MODE" = "p100" ]; then\n'
    "    BSZ=16; GA=2; EVAL_BSZ=16\n"
    "else\n"
    "    BSZ=32; GA=1; EVAL_BSZ=32\n"
    "fi"
)

# ── Root T5 scripts: add 3090 2-GPU to a100 branch ───────────────────────────

OLD_A100_ROOT = (
    "else\n"
    '    GPU_MODE="a100"\n'
    '    GPU_IDS="${1:-0}"\n'
    '    FP16_FLAG=""\n'
    '    echo "[GPU] Strategy: A100 (single GPU, fp32)"\n'
    "fi"
)

NEW_A100_ROOT = (
    "else\n"
    '    GPU_MODE="a100"\n'
    '    if [ "$NUM_GPUS" -ge 2 ]; then\n'
    '        GPU_IDS="0,1"\n'
    '        echo "[GPU] Strategy: ${NUM_GPUS}x ${GPU_MEM}MB DataParallel (RTX3090/A100, fp32)"\n'
    "    else\n"
    '        GPU_IDS="${1:-0}"\n'
    '        echo "[GPU] Strategy: 1x ${GPU_MEM}MB GPU (fp32)"\n'
    "    fi\n"
    '    FP16_FLAG=""\n'
    "fi"
)


def patch(path, replacements):
    full = os.path.join(BASE, path)
    if not os.path.exists(full):
        print(f"  SKIP (not found): {path}")
        return
    with open(full) as f:
        content = f.read()
    original = content
    for old, new in replacements:
        n = content.count(old)
        if n > 0:
            content = content.replace(old, new)
            print(f"  {path}: replaced {n}x '{old[:50].strip()}...'")
    if content != original:
        with open(full, "w") as f:
            f.write(content)
    else:
        print(f"  {path}: no changes (already patched or pattern mismatch)")


T5_SMALL_LONG = [
    "T5_small/gen_script_long_order3_t5_small_specroute.sh",
    "T5_small/gen_script_long_order3_t5_small_specroute_v10a.sh",
    "T5_small/gen_script_long_order3_t5_small_specroute_v10b.sh",
    "T5_small/gen_script_long_order4_t5_small_specroute.sh",
]

T5_SMALL_SUPERNI = [
    "T5_small/gen_script_superni_order1_t5_small_specroute.sh",
    "T5_small/gen_script_superni_order2_t5_small_specroute.sh",
]

ROOT_T5 = [
    "gen_script_long_order3_t5_specroute.sh",
    "gen_script_long_order4_t5_specroute.sh",
    "gen_script_superni_order1_t5_specroute.sh",
    "gen_script_superni_order2_t5_specroute.sh",
]

print("=== T5_small Long scripts ===")
for p in T5_SMALL_LONG:
    patch(p, [
        (OLD_DETECT_LONG, NEW_DETECT),
        (OLD_BSZ_LONG_A,  NEW_BSZ_LONG),
    ])

print("\n=== T5_small SuperNI scripts ===")
for p in T5_SMALL_SUPERNI:
    patch(p, [
        (OLD_DETECT_SUPERNI, NEW_DETECT),
        (OLD_DETECT_LONG,    NEW_DETECT),
        (OLD_BSZ_SUPERNI_A,  NEW_BSZ_SUPERNI),
        (OLD_BSZ_SUPERNI_B,  NEW_BSZ_SUPERNI),
    ])

print("\n=== Root T5 scripts (add 3090 2-GPU to a100 branch) ===")
for p in ROOT_T5:
    patch(p, [(OLD_A100_ROOT, NEW_A100_ROOT)])

print("\nDone.")
