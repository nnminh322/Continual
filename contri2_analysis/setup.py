#!/usr/bin/env python3
"""
Setup script for C2 (SGWI + Dual Fisher) hypothesis testing.
Patches new_gainlora/src/run_t5.py to add SGWI args and trainer.

Usage:
    python contri2_analysis/setup.py --apply     # Apply patches
    python contri2_analysis/setup.py --revert    # Revert to original
    python contri2_analysis/setup.py --check     # Check if already patched
"""

import os
import sys
import shutil
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_T5_PATH = os.path.join(REPO_ROOT, 'new_gainlora', 'src', 'run_t5.py')
BACKUP_PATH = RUN_T5_PATH + '.bak_c2'

# ============================================================================
# Patch 1: Add SGWI arguments to TrainingArguments
# ============================================================================
ARGS_MARKER = "srt_skip_forward: Optional[bool]"
ARGS_PATCH = '''
    # ── C2: SGWI + Dual Fisher ───────────────────────────────────
    sgwi_mode: Optional[str] = field(
        default='inflora',
        metadata={"help": "SGWI init mode: 'inflora' (baseline), 'sgwi', 'sgwi+inflora', 'random'"}
    )
    lambda_emb: Optional[float] = field(
        default=0.0,
        metadata={"help": "Dual Fisher embedding regularization strength (0=disabled)"}
    )
'''

# ============================================================================
# Patch 2: Add SGWI trainer import
# ============================================================================
IMPORT_MARKER = "from cl_trainer_srt import SRT_Trainer"
IMPORT_PATCH = """from cl_trainer_srt import SRT_Trainer
from sgwi_trainer import SGWI_DualFisher_Trainer"""

# ============================================================================
# Patch 3: Wire SGWI trainer into the SRT training path
# Replace SRT_Trainer instantiation with SGWI_DualFisher_Trainer
# ============================================================================
# We need to find where SRT_Trainer is instantiated and add a condition
TRAINER_MARKER = "trainer = SRT_Trainer("
TRAINER_PATCH_BEFORE = """            # C2: Use SGWI trainer if sgwi_mode != 'inflora' or lambda_emb > 0
            if hasattr(training_args, 'sgwi_mode') and (training_args.sgwi_mode != 'inflora' or training_args.lambda_emb > 0):
                trainer = SGWI_DualFisher_Trainer("""
TRAINER_PATCH_AFTER_CLOSE = """                    sgwi_mode=training_args.sgwi_mode,
                    lambda_emb=training_args.lambda_emb,"""


def check_patched():
    """Check if run_t5.py is already patched."""
    if not os.path.exists(RUN_T5_PATH):
        return False
    with open(RUN_T5_PATH, 'r') as f:
        content = f.read()
    return 'sgwi_mode' in content and 'SGWI_DualFisher_Trainer' in content


def apply_patch():
    """Apply C2 patches to run_t5.py."""
    if check_patched():
        print("[setup] Already patched. Use --revert first to re-apply.")
        return True

    if not os.path.exists(RUN_T5_PATH):
        print(f"[setup] ERROR: {RUN_T5_PATH} not found!")
        return False

    # Backup
    shutil.copy2(RUN_T5_PATH, BACKUP_PATH)
    print(f"[setup] Backed up to {BACKUP_PATH}")

    with open(RUN_T5_PATH, 'r') as f:
        content = f.read()

    # Patch 1: Add arguments
    if ARGS_MARKER in content:
        # Find the line with srt_skip_forward and add after its field() closing
        lines = content.split('\n')
        new_lines = []
        found_marker = False
        for i, line in enumerate(lines):
            new_lines.append(line)
            if ARGS_MARKER in line and not found_marker:
                # Find end of this field definition (next line with closing paren)
                j = i + 1
                while j < len(lines) and ')' not in lines[j]:
                    j += 1
                # The args patch will be inserted after j
                # But we need to handle this differently - add after this block
                found_marker = True
        
        if found_marker:
            # Simpler approach: just insert after the marker line's field() block
            content_with_args = content.replace(
                ARGS_MARKER,
                ARGS_MARKER
            )
            # Find the complete srt_skip_forward field definition end
            idx = content.find(ARGS_MARKER)
            # Find the next field or class end after this
            idx_after = content.find('\n\n', idx)
            if idx_after == -1:
                idx_after = content.find('\n    #', idx + len(ARGS_MARKER))
            if idx_after == -1:
                idx_after = content.find('\n    denser_evaluation', idx)
            
            if idx_after > idx:
                content = content[:idx_after] + '\n' + ARGS_PATCH + content[idx_after:]
                print("[setup] ✅ Patch 1: Added sgwi_mode + lambda_emb arguments")
            else:
                # Fallback: insert right after the marker line
                content = content.replace(
                    ARGS_MARKER,
                    ARGS_MARKER + '\n' + ARGS_PATCH
                )
                print("[setup] ✅ Patch 1 (fallback): Added sgwi_mode + lambda_emb arguments")
    else:
        print(f"[setup] ⚠️  Patch 1: Could not find marker '{ARGS_MARKER}'. Manual edit needed.")

    # Patch 2: Add import
    if IMPORT_MARKER in content and 'SGWI_DualFisher_Trainer' not in content:
        content = content.replace(IMPORT_MARKER, IMPORT_PATCH)
        print("[setup] ✅ Patch 2: Added SGWI_DualFisher_Trainer import")
    elif 'SGWI_DualFisher_Trainer' in content:
        print("[setup] ✅ Patch 2: Import already present")
    else:
        print(f"[setup] ⚠️  Patch 2: Could not find import marker. Manual edit needed.")

    # Patch 3: Wire trainer (more complex - add before SRT_Trainer instantiation)
    if TRAINER_MARKER in content and 'SGWI_DualFisher_Trainer' not in content.split(TRAINER_MARKER)[0][-200:]:
        # Find the trainer = SRT_Trainer( line and add conditional before it
        idx = content.find(TRAINER_MARKER)
        # Find the line start
        line_start = content.rfind('\n', 0, idx) + 1
        indent = ' ' * (idx - line_start)
        
        # Build the replacement: add SGWI path as an elif before the existing SRT_Trainer
        # We wrap the existing SRT_Trainer in an else block
        old_trainer_line = content[line_start:content.find('\n', idx)]
        
        # Find the complete SRT_Trainer constructor call (until matching closing paren)
        paren_count = 0
        end_idx = idx
        for ci in range(idx, len(content)):
            if content[ci] == '(':
                paren_count += 1
            elif content[ci] == ')':
                paren_count -= 1
                if paren_count == 0:
                    end_idx = ci + 1
                    break
        
        original_srt_block = content[line_start:end_idx]
        
        # Create the patched version with conditional
        sgwi_block = original_srt_block.replace(
            'trainer = SRT_Trainer(',
            'trainer = SGWI_DualFisher_Trainer('
        )
        # Add sgwi_mode and lambda_emb params - insert before the closing paren
        last_paren = sgwi_block.rfind(')')
        sgwi_block = (
            sgwi_block[:last_paren] + 
            f'\n{indent}    sgwi_mode=training_args.sgwi_mode,\n'
            f'{indent}    lambda_emb=training_args.lambda_emb,\n'
            f'{indent})' 
        )
        # Remove the duplicate closing paren
        sgwi_block = sgwi_block[:sgwi_block.rfind(')')] + ')'
        
        # Build conditional
        patched = (
            f'{indent}# C2: Use SGWI trainer if sgwi_mode specified\n'
            f'{indent}if hasattr(training_args, "sgwi_mode") and '
            f'(training_args.sgwi_mode != "inflora" or getattr(training_args, "lambda_emb", 0) > 0):\n'
            f'{indent}    ' + sgwi_block.replace('\n', f'\n{indent}    ').strip() + '\n'
            f'{indent}else:\n'
            f'{indent}    ' + original_srt_block.replace('\n', f'\n{indent}    ').strip()
        )
        
        content = content[:line_start] + patched + content[end_idx:]
        print("[setup] ✅ Patch 3: Wired SGWI_DualFisher_Trainer into SRT path")
    else:
        print(f"[setup] ⚠️  Patch 3: Could not auto-patch trainer instantiation.")
        print(f"         Manual edit: In run_t5.py, where SRT_Trainer is created,")
        print(f"         add conditional to use SGWI_DualFisher_Trainer when sgwi_mode != 'inflora'")

    # Write patched file
    with open(RUN_T5_PATH, 'w') as f:
        f.write(content)
    
    print(f"\n[setup] ✅ Patches applied to {RUN_T5_PATH}")
    print(f"[setup] Backup at: {BACKUP_PATH}")
    return True


def revert_patch():
    """Revert run_t5.py to pre-patch state."""
    if os.path.exists(BACKUP_PATH):
        shutil.copy2(BACKUP_PATH, RUN_T5_PATH)
        os.remove(BACKUP_PATH)
        print(f"[setup] ✅ Reverted {RUN_T5_PATH} from backup")
        return True
    else:
        print(f"[setup] ERROR: No backup found at {BACKUP_PATH}")
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Setup C2 SGWI patches')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--apply', action='store_true', help='Apply patches')
    group.add_argument('--revert', action='store_true', help='Revert patches')
    group.add_argument('--check', action='store_true', help='Check patch status')
    args = parser.parse_args()

    if args.check:
        if check_patched():
            print("[setup] ✅ run_t5.py is patched with C2 support")
        else:
            print("[setup] ❌ run_t5.py is NOT patched. Run: python contri2_analysis/setup.py --apply")
    elif args.apply:
        apply_patch()
    elif args.revert:
        revert_patch()
