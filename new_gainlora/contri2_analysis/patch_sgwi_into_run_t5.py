#!/usr/bin/env python3
"""
patch_sgwi_into_run_t5.py — Patch SGWI warm initialization into run_t5.py

This script modifies run_t5.py to add SGWI (SRT-Guided Warm Initialization)
AFTER the existing LoRA re-initialization block (line ~555) and BEFORE training.

Usage:
    # Apply patch (creates backup at run_t5.py.bak)
    python patch_sgwi_into_run_t5.py --apply

    # Revert patch
    python patch_sgwi_into_run_t5.py --revert

    # Dry run (show what would change)
    python patch_sgwi_into_run_t5.py --dry-run

After patching, run the full CL pipeline with SGWI:
    bash gen_script_long_order3_t5_srt_hard.sh 0 google/flan-t5-large
    # SGWI activates automatically when --use_srt_router is set

The patch adds a new flag: --use_sgwi (default: True when --use_srt_router is set)
"""

import os
import sys
import shutil
import re
from pathlib import Path

RUN_T5_PATH = Path(__file__).parent.parent / "src" / "run_t5.py"

# ═════════════════════════════════════════════════════════════════════════════
#  SGWI CODE BLOCK — inserted after LoRA re-init (line ~555)
# ═════════════════════════════════════════════════════════════════════════════

SGWI_IMPORT_BLOCK = '''
# ── SGWI: SRT-Guided Warm Initialization imports ──────────────────────
try:
    from srt_router import SRTRouter
except ImportError:
    SRTRouter = None
'''

SGWI_ARG_BLOCK = '''    use_sgwi: Optional[bool] = field(
        default=True,
        metadata={"help": "Use SRT-Guided Warm Initialization (SGWI) for LoRA. "
                  "Only active when use_srt_router=True and cur_task_id > 0."},
    )
    sgwi_mode: Optional[str] = field(
        default="sfi",
        metadata={"help": "SGWI mode: 'nti' (nearest-task init) or 'sfi' (SVD fusion init)"},
    )
'''

SGWI_INIT_BLOCK = '''
    # ═══════════════════════════════════════════════════════════════════════
    #  SGWI: SRT-Guided Warm Initialization (Contribution 2)
    #
    #  When use_srt_router=True and cur_task_id > 0:
    #    1. Load SRT signatures from previous checkpoint
    #    2. Compute SRT distance to all previous tasks
    #    3. Initialize LoRA_t from weighted combination of previous LoRAs
    #       (SVD Fusion for rank-r approximation)
    #
    #  This replaces the random init (kaiming A, zeros B) with a warm start
    #  from related task adapters, guided by SRT task similarity metric.
    # ═══════════════════════════════════════════════════════════════════════
    if (cur_task_id > 0 and training_args.use_srt_router
            and getattr(training_args, 'use_sgwi', True)
            and model_args.previous_lora_path
            and SRTRouter is not None):

        print("=" * 60)
        print("[SGWI] SRT-Guided Warm Initialization")
        print(f"[SGWI] Mode: {getattr(training_args, 'sgwi_mode', 'sfi')}")

        import numpy as np
        from scipy.special import softmax as scipy_softmax

        sgwi_mode = getattr(training_args, 'sgwi_mode', 'sfi')

        # 1. Load SRT signatures from previous task's checkpoint
        srt_sig_path = None
        prev_output_dir = os.path.dirname(model_args.load_checkpoint_from) if model_args.load_checkpoint_from else None
        if prev_output_dir:
            srt_sig_path = os.path.join(prev_output_dir, 'srt_signatures.npz')
        if srt_sig_path and os.path.exists(srt_sig_path):
            sgwi_router = SRTRouter(srt_metric_mode='hard')
            sgwi_router.load(srt_sig_path)
            print(f"[SGWI] Loaded {len(sgwi_router.signatures)} task signatures from {srt_sig_path}")
        else:
            print(f"[SGWI] No SRT signatures found at {srt_sig_path} → skip SGWI")
            sgwi_router = None

        if sgwi_router and len(sgwi_router.signatures) > 0:
            # 2. Extract current task centroid from frozen encoder
            print("[SGWI] Extracting current task embeddings from frozen encoder...")
            from cl_trainer_srt import extract_embeddings_from_batch
            from torch.utils.data import DataLoader

            _sgwi_dl = DataLoader(train_dataset, batch_size=16, shuffle=False,
                                   collate_fn=data_collator, num_workers=0)
            _sgwi_embs = []
            _sgwi_max = 500
            for _step, _inputs in enumerate(_sgwi_dl):
                if _step >= _sgwi_max:
                    break
                _inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in _inputs.items()}
                _h = extract_embeddings_from_batch(model, _inputs)
                _sgwi_embs.append(_h.cpu())
            _sgwi_embs = torch.cat(_sgwi_embs, dim=0).numpy()
            _mu_t = _sgwi_embs.mean(axis=0)

            # Add current task temporarily to router for distance computation
            sgwi_router.add_task(task_id=cur_task_id, h_train=_sgwi_embs)

            # 3. Compute distances to all previous tasks
            cur_sig = sgwi_router.signatures[cur_task_id]
            _dists = []
            _task_ids = []
            for _tid, _sig in sgwi_router.signatures.items():
                if _tid == cur_task_id:
                    continue
                _d = float(np.linalg.norm(cur_sig.mu - _sig.mu))
                _dists.append(_d)
                _task_ids.append(_tid)

            if _dists:
                _dists = np.array(_dists)

                # 4. Compute similarity weights
                _tau = float(np.median(_dists)) if len(_dists) > 1 else 1.0
                _tau = max(_tau, 1e-8)
                _weights = scipy_softmax(-_dists / _tau)

                print(f"[SGWI] τ={_tau:.3f}, {len(_dists)} prior tasks")
                for _tid, _w in sorted(zip(_task_ids, _weights), key=lambda x: -x[1])[:3]:
                    print(f"[SGWI]   task {_tid}: w={_w:.3f}")

                # 5. Load previous LoRA weights and compute weighted ΔW
                previous_lora_list_sgwi = model_args.previous_lora_path.split(',')
                previous_lora_list_sgwi.reverse()  # reversed: [task_{t-1}, ..., task_0]
                #  ↑  index 0 = newest (task_{t-1}), index (t-1) = oldest (task_0)

                # ── Bug 3 fix: build task-ID → path mapping explicitly ──────────
                # _task_ids = [0, 1, 2, ..., t-1] (ascending by task ID)
                # previous_lora_list_sgwi = [path_{t-1}, ..., path_0] (newest→oldest)
                # We need: task_id → its corresponding path
                # The mapping is: task_id k → previous_lora_list_sgwi[t-1 - k]
                # because index 0 = task_{t-1}, index 1 = task_{t-2}, ...
                _tid_to_path = {}
                for _k, _path in enumerate(previous_lora_list_sgwi):
                    _tid_val = len(previous_lora_list_sgwi) - 1 - _k   # task ID
                    _tid_to_path[_tid_val] = _path

                lora_r = model_args.lora_r or 8

                # ── Per-layer SVD fusion ──────────────────────────────────────
                # Iterate: (task_id, weight) from _task_ids, _weights (same length)
                _delta_W = {}  # {param_name: accumulated weighted ΔW}
                for _tid, _w in zip(_task_ids, _weights):
                    _path = _tid_to_path.get(_tid)
                    if _path is None:
                        continue
                    _ckpt_a = os.path.join(_path, "lora_weights_A.pt")
                    _ckpt_b = os.path.join(_path, "lora_weights_B.pt")
                    if not (os.path.exists(_ckpt_a) and os.path.exists(_ckpt_b)):
                        continue
                    _A = torch.load(_ckpt_a, map_location=device)
                    _B = torch.load(_ckpt_b, map_location=device)
                    for _key_a in _A:
                        _key_b = _key_a.replace("lora_A", "lora_B")
                        if _key_b not in _B:
                            continue
                        _dW = _w * (_B[_key_b].float() @ _A[_key_a].float())
                        if _key_a not in _delta_W:
                            _delta_W[_key_a] = _dW
                        else:
                            _delta_W[_key_a] += _dW

                _n_layers_init = 0

                # ── NTI: copy LoRA from nearest task ────────────────────────────
                if sgwi_mode == 'nti':
                    _nearest_idx_in_array = int(np.argmin(_dists))
                    _nearest_tid = _task_ids[_nearest_idx_in_array]
                    _npath = _tid_to_path.get(_nearest_tid)
                    if _npath is None:
                        print(f"[SGWI] NTI: could not find path for task {_nearest_tid} → skip")
                    else:
                        _nA = torch.load(os.path.join(_npath, "lora_weights_A.pt"), map_location=device)
                        _nB = torch.load(os.path.join(_npath, "lora_weights_B.pt"), map_location=device)
                        for j in range(model.config.num_layers):
                            for _attn_type in ["q", "v"]:
                                _key_a = f"encoder.block.{j}.layer.0.SelfAttention.lora_{_attn_type}.lora_A"
                                _key_b = f"encoder.block.{j}.layer.0.SelfAttention.lora_{_attn_type}.lora_B"
                                if _key_a in _nA:
                                    _attn = model.encoder.block[j].layer[0].SelfAttention
                                    _lora_attr = getattr(_attn, f"lora_{_attn_type}", None)
                                    if _lora_attr is not None:
                                        _lora_attr.lora_A.data.copy_(_nA[_key_a])
                                        _lora_attr.lora_B.data.copy_(_nB[_key_b])
                                        _n_layers_init += 1
                        print(f"[SGWI] NTI: copied from task {_nearest_tid}, {_n_layers_init} layers")

                # ── SFI: SVD Fusion ───────────────────────────────────────────
                else:
                    for _key_a, _dw in _delta_W.items():
                        # ── Bug 2 fix: rename SVD output to avoid shadowing ─────
                        U, S_local, Vt = torch.linalg.svd(_dw, full_matrices=False)
                        _sqrt_S = torch.sqrt(S_local[:lora_r])
                        _B_init = U[:, :lora_r] * _sqrt_S.unsqueeze(0)
                        _A_init = _sqrt_S.unsqueeze(1) * Vt[:lora_r, :]

                        _parts = _key_a.split('.')
                        # e.g. "encoder.block.0.layer.0.SelfAttention.lora_q.lora_A"
                        try:
                            _block_idx = int(_parts[2])
                            _is_q = 'lora_q' in _key_a
                            _is_v = 'lora_v' in _key_a
                            _attn = model.encoder.block[_block_idx].layer[0].SelfAttention
                            if _is_q and hasattr(_attn, 'lora_q'):
                                _attn.lora_q.lora_A.data.copy_(_A_init)
                                _attn.lora_q.lora_B.data.copy_(_B_init)
                                _n_layers_init += 1
                            elif _is_v and hasattr(_attn, 'lora_v'):
                                _attn.lora_v.lora_A.data.copy_(_A_init)
                                _attn.lora_v.lora_B.data.copy_(_B_init)
                                _n_layers_init += 1
                        except (IndexError, ValueError, AttributeError):
                            pass

                    print(f"[SGWI] SFI: SVD fusion init, {_n_layers_init} layers")

            # Cleanup
            del _sgwi_embs, _sgwi_dl
            if cur_task_id in sgwi_router.signatures:
                del sgwi_router.signatures[cur_task_id]

        print("=" * 60)
'''


def apply_patch():
    """Apply SGWI patch to run_t5.py."""
    if not RUN_T5_PATH.exists():
        print(f"ERROR: {RUN_T5_PATH} not found")
        sys.exit(1)

    # Backup
    backup = RUN_T5_PATH.with_suffix('.py.bak_sgwi')
    if not backup.exists():
        shutil.copy2(RUN_T5_PATH, backup)
        print(f"  Backup: {backup}")

    content = RUN_T5_PATH.read_text()

    # Check if already patched
    if '[SGWI]' in content:
        print("  Already patched! Use --revert first to re-apply.")
        return

    # 1. Add import after existing SRT imports
    marker_import = "from cl_trainer_gainlora import"
    if marker_import in content:
        content = content.replace(marker_import, SGWI_IMPORT_BLOCK + "\n" + marker_import)
        print("  ✅ Added SGWI imports")

    # 2. Add SGWI args after srt_skip_forward field
    marker_args = "srt_skip_forward"
    if marker_args in content:
        # Find the end of srt_skip_forward field definition
        idx = content.find(marker_args)
        # Find the next field or class end
        next_field = content.find("    )", idx)
        if next_field > 0:
            insert_pos = content.find("\n", next_field) + 1
            content = content[:insert_pos] + SGWI_ARG_BLOCK + content[insert_pos:]
            print("  ✅ Added SGWI training arguments")

    # 3. Add SGWI init block after LoRA re-initialization
    marker_init = '[FIX] Re-initialized lora_A in'
    if marker_init in content:
        idx = content.find(marker_init)
        # Find the print statement end
        next_nl = content.find("\n", idx)
        insert_pos = next_nl + 1
        content = content[:insert_pos] + SGWI_INIT_BLOCK + content[insert_pos:]
        print("  ✅ Added SGWI initialization block")

    RUN_T5_PATH.write_text(content)
    print(f"\n  Patch applied to {RUN_T5_PATH}")
    print("  Run your training script as usual — SGWI activates automatically.")


def revert_patch():
    """Revert SGWI patch."""
    backup = RUN_T5_PATH.with_suffix('.py.bak_sgwi')
    if backup.exists():
        shutil.copy2(backup, RUN_T5_PATH)
        print(f"  Reverted from {backup}")
    else:
        print("  No backup found — cannot revert")


def dry_run():
    """Show what the patch would do."""
    print("  SGWI Patch would:")
    print("  1. Add SRTRouter import after existing imports")
    print("  2. Add --use_sgwi and --sgwi_mode training arguments")
    print("  3. Add SGWI init block (~80 lines) after LoRA re-initialization")
    print(f"\n  Target file: {RUN_T5_PATH}")
    print(f"  File exists: {RUN_T5_PATH.exists()}")
    if RUN_T5_PATH.exists():
        content = RUN_T5_PATH.read_text()
        print(f"  Already patched: {'[SGWI]' in content}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true")
    p.add_argument("--revert", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.apply:
        apply_patch()
    elif args.revert:
        revert_patch()
    elif args.dry_run:
        dry_run()
    else:
        dry_run()
        print("\n  Use --apply to patch, --revert to undo, --dry-run to preview")
