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

    # ========================================================================
    # Patch 1: Add SGWI arguments to TrainingArguments
    # ========================================================================
    ARGS_MARKER = """    srt_skip_forward: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Skip forward-pass embedding extraction. "
                    "Load pre-extracted embeddings from embeddings/{backbone}/{split}/{task}/train.npz instead. "
                    "Requires pre-extracted embeddings to exist."
        },
    )

    denser_evaluation: Optional[bool]"""
    
    ARGS_PATCH = """    srt_skip_forward: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Skip forward-pass embedding extraction. "
                    "Load pre-extracted embeddings from embeddings/{backbone}/{split}/{task}/train.npz instead. "
                    "Requires pre-extracted embeddings to exist."
        },
    )

    # ── C2: SGWI + Dual Fisher ───────────────────────────────────
    sgwi_mode: Optional[str] = field(
        default='inflora',
        metadata={"help": "SGWI init mode: 'inflora' (baseline), 'sgwi', 'sgwi+inflora', 'random'"}
    )
    lambda_emb: Optional[float] = field(
        default=0.0,
        metadata={"help": "Dual Fisher embedding regularization strength (0=disabled)"}
    )

    denser_evaluation: Optional[bool]"""

    if ARGS_MARKER in content:
        content = content.replace(ARGS_MARKER, ARGS_PATCH)
        print("[setup] ✅ Patch 1: Added sgwi_mode + lambda_emb arguments")
    else:
        print(f"[setup] ⚠️  Patch 1: Could not find marker. Manual edit needed.")

    # ========================================================================
    # Patch 2+3: Replace SRT_Trainer block with SGWI conditional
    # NOTE: Patch 2 (import) is embedded inside the replaced block below.
    #       Do NOT add a separate import replacement - it breaks block matching.
    # ========================================================================
    ORIGINAL_SRT_BLOCK = """    elif training_args.model_name == 'gainlora_inflora' and training_args.use_srt_router:
        # SRT Trainer: GainLoRA + SRT non-parametric router
        from cl_trainer_srt import SRT_Trainer
        trainer = SRT_Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            cur_task_id=cur_task_id,
            task_order=task_order,
            data_collator_replay=data_collator_replay,
            replay_dataset_dict=replay_dataset_dict,
            replay_label_dict=replay_label_dict,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_rouge_metrics,
            callbacks=[DenserEvalCallback] if training_args.denser_evaluation else None,
            srt_metric_mode=training_args.srt_metric_mode,
            srt_shrink=training_args.srt_shrink,
            srt_shrink_factor=training_args.srt_shrink_factor,
            srt_max_emb_samples=training_args.srt_max_emb_samples,
            srt_load_path=training_args.srt_load_path,
            srt_skip_forward=training_args.srt_skip_forward,
        )"""

    PATCHED_SRT_BLOCK = """    elif training_args.model_name == 'gainlora_inflora' and training_args.use_srt_router:
        # SRT Trainer: GainLoRA + SRT non-parametric router
        # C2: Use SGWI_DualFisher_Trainer if SGWI mode specified
        if (hasattr(training_args, 'sgwi_mode') and
                (training_args.sgwi_mode != 'inflora' or getattr(training_args, 'lambda_emb', 0) > 0)):
            from sgwi_trainer import SGWI_DualFisher_Trainer
            trainer = SGWI_DualFisher_Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset if training_args.do_train else None,
                cur_task_id=cur_task_id,
                task_order=task_order,
                data_collator_replay=data_collator_replay,
                replay_dataset_dict=replay_dataset_dict,
                replay_label_dict=replay_label_dict,
                eval_dataset=eval_dataset if training_args.do_eval else None,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_rouge_metrics,
                callbacks=[DenserEvalCallback] if training_args.denser_evaluation else None,
                srt_metric_mode=training_args.srt_metric_mode,
                srt_shrink=training_args.srt_shrink,
                srt_shrink_factor=training_args.srt_shrink_factor,
                srt_max_emb_samples=training_args.srt_max_emb_samples,
                srt_load_path=training_args.srt_load_path,
                srt_skip_forward=training_args.srt_skip_forward,
                sgwi_mode=training_args.sgwi_mode,
                lambda_emb=training_args.lambda_emb,
            )
        else:
            from cl_trainer_srt import SRT_Trainer
            trainer = SRT_Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset if training_args.do_train else None,
                cur_task_id=cur_task_id,
                task_order=task_order,
                data_collator_replay=data_collator_replay,
                replay_dataset_dict=replay_dataset_dict,
                replay_label_dict=replay_label_dict,
                eval_dataset=eval_dataset if training_args.do_eval else None,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_rouge_metrics,
                callbacks=[DenserEvalCallback] if training_args.denser_evaluation else None,
                srt_metric_mode=training_args.srt_metric_mode,
                srt_shrink=training_args.srt_shrink,
                srt_shrink_factor=training_args.srt_shrink_factor,
                srt_max_emb_samples=training_args.srt_max_emb_samples,
                srt_load_path=training_args.srt_load_path,
                srt_skip_forward=training_args.srt_skip_forward,
            )"""

    if ORIGINAL_SRT_BLOCK in content:
        content = content.replace(ORIGINAL_SRT_BLOCK, PATCHED_SRT_BLOCK)
        print("[setup] ✅ Patch 2+3: Added SGWI conditional into SRT trainer path")
    else:
        # Debug: show what we actually find around the SRT block
        idx = content.find("'gainlora_inflora' and training_args.use_srt_router")
        if idx > 0:
            snippet = content[idx:idx+500].replace('\n', '\\n\n  ')
            print(f"[setup] ⚠️  Patch 2+3: Block mismatch. Actual content around SRT block:")
            print(f"  {snippet[:300]}")
        else:
            print(f"[setup] ⚠️  Patch 2+3: Cannot find SRT block at all!")
        print()
        print(f"[setup] MANUAL FIX: In run_t5.py, find the 'elif gainlora_inflora and use_srt_router' block")
        print(f"        and wrap the trainer = SRT_Trainer(...) with:")
        print(f"          if (hasattr(training_args, 'sgwi_mode') and ...")
        print(f"               trainer = SGWI_DualFisher_Trainer(..., sgwi_mode=..., lambda_emb=...)")
        print(f"          else:")
        print(f"               trainer = SRT_Trainer(...)")

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
        print(f"[setup] Unable to revert. You may need to restore from git:")
        print(f"       git checkout new_gainlora/src/run_t5.py")
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
