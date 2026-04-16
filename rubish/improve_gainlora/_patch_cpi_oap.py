"""Patch all specroute gen_scripts to add CPI+OAP parameters.
Usage: python _patch_cpi_oap.py [--gamma 0.5] [--eta 0.5] [--beta_min 0.3] [--warmup 3]
"""
import re, os, sys

BASE = os.path.dirname(os.path.abspath(__file__))

# Default values
GAMMA = 0.5
ETA = 0.5
BETA_MIN = 0.3
WARMUP = 3

# Parse CLI overrides
args = sys.argv[1:]
i = 0
while i < len(args):
    if args[i] == '--gamma' and i + 1 < len(args):
        GAMMA = float(args[i+1]); i += 2
    elif args[i] == '--eta' and i + 1 < len(args):
        ETA = float(args[i+1]); i += 2
    elif args[i] == '--beta_min' and i + 1 < len(args):
        BETA_MIN = float(args[i+1]); i += 2
    elif args[i] == '--warmup' and i + 1 < len(args):
        WARMUP = int(args[i+1]); i += 2
    else:
        i += 1

CPI_OAP_BLOCK = (
    f'   --cpi_gamma {GAMMA} \\\n'
    f'   --oap_eta {ETA} \\\n'
    f'   --oap_beta_min {BETA_MIN} \\\n'
    f'   --oap_warmup {WARMUP} \\\n'
)

# Find all specroute gen scripts
scripts = sorted([
    os.path.join(BASE, f) for f in os.listdir(BASE)
    if f.startswith('gen_script_') and 'specroute' in f and f.endswith('.sh')
])

ANCHOR = re.compile(r'(   --model_name specroute \\\n)')

for script_path in scripts:
    with open(script_path) as f:
        content = f.read()

    if '--cpi_gamma' in content:
        # Already patched — update values
        content = re.sub(r'--cpi_gamma [\d.]+', f'--cpi_gamma {GAMMA}', content)
        content = re.sub(r'--oap_eta [\d.]+', f'--oap_eta {ETA}', content)
        content = re.sub(r'--oap_beta_min [\d.]+', f'--oap_beta_min {BETA_MIN}', content)
        content = re.sub(r'--oap_warmup \d+', f'--oap_warmup {WARMUP}', content)
        action = 'UPDATED'
    else:
        # Insert CPI+OAP params after --model_name specroute
        matches = list(ANCHOR.finditer(content))
        if not matches:
            print(f'SKIP (no anchor): {os.path.basename(script_path)}')
            continue
        # Insert after each occurrence (multiple task blocks)
        for m in reversed(matches):
            insert_pos = m.end()
            content = content[:insert_pos] + CPI_OAP_BLOCK + content[insert_pos:]
        action = 'PATCHED'

    with open(script_path, 'w') as f:
        f.write(content)

    n_blocks = len(ANCHOR.findall(content)) if action == 'UPDATED' else len(matches)
    print(f'{action} ({n_blocks} blocks): {os.path.basename(script_path)} '
          f'[gamma={GAMMA}, eta={ETA}, beta_min={BETA_MIN}, warmup={WARMUP}]')
