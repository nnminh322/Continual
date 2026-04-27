#!/usr/bin/env python3
"""Restore all removed code + fix bugs."""
import os, sys

# ──────────────────────────────────────────────────────────────────────────────
# FIX 1: Add --srt_pca_components to argument parser
# ──────────────────────────────────────────────────────────────────────────────
run_path = "/Users/nhatminhnguyen/Library/Mobile Documents/com~apple~CloudDocs/Thạc sĩ/Continual/new_llama_gainlora/run_llama_gainlora_cl.py"

with open(run_path, "r") as f:
    content = f.read()

# Add pca_components arg after srt_max_emb_samples
old_emb_arg = '    parser.add_argument("--srt_max_emb_samples",type=int,   default=500)'
new_emb_arg = '''    parser.add_argument("--srt_max_emb_samples",type=int,   default=500)
    parser.add_argument("--srt_pca_components", type=int, default=None,
                        help="PCA dims before Mahalanobis (e.g. 128). Reduces 4096→128 for stable Σ estimate.")'''

if old_emb_arg in content:
    content = content.replace(old_emb_arg, new_emb_arg)
    print("✓ run_llama_gainlora_cl.py: --srt_pca_components arg added")
else:
    # Try to find the line with whitespace variations
    import re
    pat = re.compile(r'parser\.add_argument\("--srt_max_emb_samples"[^)]+\)')
    m = pat.search(content)
    if m:
        old = m.group(0)
        new = old + '\n    parser.add_argument("--srt_pca_components", type=int, default=None,\n                        help="PCA dims before Mahalanobis (e.g. 128). Reduces 4096→128 for stable Σ estimate.")'
        content = content.replace(old, new)
        print("✓ run_llama_gainlora_cl.py: --srt_pca_components arg added (regex)")
    else:
        print("✗ run_llama_gainlora_cl.py: srt_max_emb_samples arg not found")

# FIX 2: Add srt_pca_components to trainer kwargs
old_trainer = """        srt_skip_forward   = args.srt_skip_forward,
        # in-training eval"""

new_trainer = """        srt_skip_forward   = args.srt_skip_forward,
        srt_pca_components = getattr(args, 'srt_pca_components', None),
        # in-training eval"""

if old_trainer in content:
    content = content.replace(old_trainer, new_trainer)
    print("✓ run_llama_gainlora_cl.py: srt_pca_components kwarg added")
else:
    print("✗ run_llama_gainlora_cl.py: trainer kwarg pattern not found")

# FIX 3: Add srt_routing_stats init before batch loop
old_init = """    n_batches = math.ceil(len(samples) / batch_size)
    # SRT router status"""

new_init = """    n_batches = math.ceil(len(samples) / batch_size)
    # SRT routing accuracy tracker (per GT task, per slot)
    srt_routing_stats: dict = {}

    # SRT router status"""

if old_init in content:
    content = content.replace(old_init, new_init)
    print("✓ run_llama_gainlora_cl.py: srt_routing_stats init added")
else:
    print("✗ run_llama_gainlora_cl.py: n_batches init not found")

# FIX 4: Add routing accuracy tracking + summary inside generate_predictions_cl
# Find the spot AFTER "model.model.is_inference = False" and BEFORE "runtime = ..."
old_inference_block = """    model.model.is_inference = False
    runtime = time.perf_counter() - start_time
    return predictions, references, generated_lengths, runtime, total_steps


def evaluate_split_cl("""

new_inference_block = """    model.model.is_inference = False

    # ── SRT ROUTING ACCURACY SUMMARY ─────────────────────────────────
    if srt_routing_stats:
        print()
        print("  ┌" + "─"*66 + "┐")
        print("  │           SRT ROUTING ACCURACY SUMMARY" + " "*(66-44) + "│")
        print("  ├" + "─"*66 + "┤")
        overall_correct = 0
        overall_total = 0
        for gt_key, stats in sorted(srt_routing_stats.items()):
            acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0.0
            overall_correct += stats["correct"]
            overall_total += stats["total"]
            slot_dist = "  ".join([f"{k}={v}" for k, v in sorted(stats["slots"].items())])
            line = f"  │  GT={gt_key[:40]:40s}  acc={acc:5.1f}%  ({stats['correct']:3d}/{stats['total']:3d})  [{slot_dist[:20]}]"
            print(line + " " * max(0, 67 - len(line)) + "│")
        overall_acc = overall_correct / overall_total * 100 if overall_total > 0 else 0.0
        print("  ├" + "─"*66 + "┤")
        summary_line = f"  │  OVERALL ACCURACY: {overall_acc:5.1f}%  ({overall_correct:3d}/{overall_total:3d} samples)"
        print(summary_line + " " * max(0, 67 - len(summary_line)) + "│")
        print("  └" + "─"*66 + "┘")
    else:
        print("  [SRT] No routing data captured.")
    # ── END SRT SUMMARY ───────────────────────────────────────────────

    runtime = time.perf_counter() - start_time
    return predictions, references, generated_lengths, runtime, total_steps


def evaluate_split_cl("""

if old_inference_block in content:
    content = content.replace(old_inference_block, new_inference_block)
    print("✓ run_llama_gainlora_cl.py: routing accuracy summary added")
else:
    print("✗ run_llama_gainlora_cl.py: inference block not found")

# FIX 5: Add routing accuracy tracking inside the SRT debug loop
# Find the current debug print block and add accuracy tracking
# The block starts with "# ── SRT DEBUG LOGGING ─" and ends with "# ── END SRT DEBUG ─"
old_debug_block = """                correct_flag = "✓" if (slot_task == gt_task or pred_task_str == gt_task) else "✗"

                # Format distances compactly
                dist_str = \\"  \\".join([f\\"{t[:20]:20s}:{d:.1f}\\" for t, d in sorted_tasks[:4]])

                # Safe 2nd task access"""

new_debug_block = """                correct_flag = "✓" if (slot_task == gt_task or pred_task_str == gt_task) else "✗"

                # ── Track accuracy per GT task ───────────────────────────
                gt_key = gt_task
                if gt_key not in srt_routing_stats:
                    srt_routing_stats[gt_key] = {"correct": 0, "total": 0, "slots": {}}
                srt_routing_stats[gt_key]["total"] += 1
                if correct_flag == "✓":
                    srt_routing_stats[gt_key]["correct"] += 1
                slot_key = f"slot{pred_slot}"
                srt_routing_stats[gt_key]["slots"][slot_key] = \\
                    srt_routing_stats[gt_key]["slots"].get(slot_key, 0) + 1
                # ─────────────────────────────────────────────────────

                # Format distances compactly
                dist_str = \\"  \\".join([f\\"{t[:20]:20s}:{d:.1f}\\" for t, d in sorted_tasks[:4]])

                # Safe 2nd task access"""

# Since the exact string is complex, let me use a different approach:
# Find the specific section to patch
# The tracking code needs to be inserted right after "correct_flag = ..."
# and before "dist_str = ..."

# Let's try a targeted search-replace
debug_marker_start = 'correct_flag = "✓" if (slot_task == gt_task or pred_task_str == gt_task) else "✗"'
debug_marker_end = '# Format distances compactly'

# Find the position of start marker
pos_start = content.find(debug_marker_start)
if pos_start >= 0:
    # Find the end marker after start
    pos_end = content.find(debug_marker_end, pos_start + len(debug_marker_start))
    if pos_end >= 0:
        old_section = content[pos_start:pos_end]
        new_section = debug_marker_start + """
                # ── Track accuracy per GT task ───────────────────────────
                gt_key = gt_task
                if gt_key not in srt_routing_stats:
                    srt_routing_stats[gt_key] = {"correct": 0, "total": 0, "slots": {}}
                srt_routing_stats[gt_key]["total"] += 1
                if correct_flag == "✓":
                    srt_routing_stats[gt_key]["correct"] += 1
                slot_key = f"slot{pred_slot}"
                srt_routing_stats[gt_key]["slots"][slot_key] = \\
                    srt_routing_stats[gt_key]["slots"].get(slot_key, 0) + 1
                # ─────────────────────────────────────────────────────

"""
        content = content[:pos_start] + new_section + content[pos_end:]
        print("✓ run_llama_gainlora_cl.py: accuracy tracking inserted in debug loop")
    else:
        print("✗ run_llama_gainlora_cl.py: debug end marker not found")
else:
    print("✗ run_llama_gainlora_cl.py: correct_flag marker not found in debug block")

with open(run_path, "w") as f:
    f.write(content)

print()
print("="*60)
print("All fixes applied to run_llama_gainlora_cl.py")
print("="*60)
