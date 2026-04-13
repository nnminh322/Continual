#!/bin/bash
# =============================================================
# cleanup_checkpoints.sh
# Removes redundant checkpoints, keeping only what matters:
#   - COMPLETED task (has saved_weights/srt_signatures.npz): remove ALL checkpoint-*
#   - IN_PROGRESS task (has checkpoint-* but no saved_weights): keep ONLY latest checkpoint-*
# =============================================================

set -e

BASE_DIR="${1:-logs_and_outputs}"

if [ ! -d "$BASE_DIR" ]; then
    echo "ERROR: Directory not found: $BASE_DIR"
    echo "Usage: bash cleanup_checkpoints.sh [base_dir]"
    exit 1
fi

echo "Scanning: $BASE_DIR"
echo "============================================================"
echo ""

total_freed=0
total_tasks=0

# Find all task output directories
for task_dir in "$BASE_DIR"/[0-9]*-*/; do
    [ -d "$task_dir" ] || continue
    task_name=$(basename "$task_dir")

    # Check if task is COMPLETED (has saved_weights marker)
    if [ -f "$task_dir/saved_weights/srt_signatures.npz" ]; then
        # Count checkpoint- directories
        ckpt_count=$(ls -d "$task_dir"/checkpoint-* 2>/dev/null | wc -l)
        if [ "$ckpt_count" -gt 0 ]; then
            freed_bytes=0
            for ckpt in "$task_dir"/checkpoint-*; do
                size=$(du -sb "$ckpt" 2>/dev/null | cut -f1)
                freed_bytes=$((freed_bytes + size))
            done
            freed_mb=$((freed_bytes / 1024 / 1024))
            echo "✓ COMPLETED  $task_name"
            echo "  Removing $ckpt_count checkpoints ($freed_mb MB)..."
            rm -rf "$task_dir"/checkpoint-*
            total_freed=$((total_freed + freed_bytes))
        else
            echo "✓ COMPLETED  $task_name (no checkpoints to clean)"
        fi
        total_tasks=$((total_tasks + 1))

    else
        # Check for partial checkpoints
        ckpt_count=$(ls -d "$task_dir"/checkpoint-* 2>/dev/null | wc -l)
        if [ "$ckpt_count" -gt 1 ]; then
            # Find latest checkpoint
            latest=$(ls -d "$task_dir"/checkpoint-* | sort -V | tail -1)
            latest_name=$(basename "$latest")
            echo "⚠ IN_PROGRESS $task_name"
            echo "  Keeping latest: $latest_name"
            echo "  Removing $((ckpt_count - 1)) older checkpoints..."

            freed_bytes=0
            for ckpt in $(ls -d "$task_dir"/checkpoint-* | sort -V | head -n -1); do
                size=$(du -sb "$ckpt" 2>/dev/null | cut -f1)
                freed_bytes=$((freed_bytes + size))
                rm -rf "$ckpt"
            done
            freed_mb=$((freed_bytes / 1024 / 1024))
            echo "  Freed: $freed_mb MB"
            total_freed=$((total_freed + freed_bytes))
        elif [ "$ckpt_count" -eq 1 ]; then
            echo "⚠ IN_PROGRESS $task_name (1 checkpoint, keeping)"
        else
            echo "  (empty task directory, no checkpoints)"
        fi
    fi
    echo ""
done

echo "============================================================"
total_freed_mb=$((total_freed / 1024 / 1024))
echo "Done. Freed ~${total_freed_mb} MB across $total_tasks completed tasks."
