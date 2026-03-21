"""Patch T5 specroute gen_scripts to add P100 GPU detection and BSZ."""
import re, os

BASE = '/Users/nnminh322/Desktop/personal/Continual/improve_gainlora'

T5_SCRIPTS = [
    os.path.join(BASE, 'gen_script_superni_order1_t5_specroute.sh'),
    os.path.join(BASE, 'gen_script_superni_order2_t5_specroute.sh'),
    os.path.join(BASE, 'gen_script_long_order3_t5_specroute.sh'),
    os.path.join(BASE, 'gen_script_long_order4_t5_specroute.sh'),
]

GPU_OLD = (
    'else\n'
    '    GPU_MODE="a100"\n'
    '    GPU_IDS="${1:-0}"\n'
    '    FP16_FLAG=""\n'
    '    echo "[GPU] Strategy: A100 (single GPU, fp32)"\n'
    'fi'
)

GPU_NEW = (
    'elif [ "$GPU_MEM" -gt 16000 ]; then\n'
    '    GPU_MODE="p100"\n'
    '    GPU_IDS="${1:-0}"\n'
    '    FP16_FLAG="--gradient_checkpointing"\n'
    '    echo "[GPU] Strategy: P100 16GB (fp32 + gradient_checkpointing)"\n'
    'else\n'
    '    GPU_MODE="a100"\n'
    '    GPU_IDS="${1:-0}"\n'
    '    FP16_FLAG=""\n'
    '    echo "[GPU] Strategy: A100 (single GPU, fp32)"\n'
    'fi'
)

BSZ_PAT = re.compile(
    r'(elif \[ "\$GPU_MODE" = "t4_1gpu" \]; then\n    BSZ=\d+; GA=\d+; EVAL_BSZ=\d+\n)'
    r'(else\n    BSZ=\d+; GA=\d+; EVAL_BSZ=\d+\n)'
)

def add_p100(m):
    return (
        m.group(1)
        + 'elif [ "$GPU_MODE" = "p100" ]; then\n    BSZ=8; GA=4; EVAL_BSZ=4\n'
        + m.group(2)
    )

for name in T5_SCRIPTS:
    if not os.path.exists(name):
        print(f'SKIP (not found): {name}')
        continue
    with open(name) as f:
        c = f.read()
    n_detect = c.count(GPU_OLD)
    c = c.replace(GPU_OLD, GPU_NEW, 1)
    n_bsz = len(BSZ_PAT.findall(c))
    c = BSZ_PAT.sub(add_p100, c)
    with open(name, 'w') as f:
        f.write(c)
    print(f'{name}: gpu_detect={n_detect} bsz_blocks={n_bsz}')

print('Done.')
