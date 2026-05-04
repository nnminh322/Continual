#!/bin/bash
# =============================================================================
# run_srt_inflora.sh
#
# Run SRT_InfLoRA on CIFAR-100 (class-incremental learning).
# Defaults: SGWI=true, full LoRA (A+B trainable), no DualFisher/GPM.
#
# Usage:
#   bash run_srt_inflora.sh [DEVICE] [SEED]
#
# Examples:
#   bash run_srt_inflora.sh 0 0           # GPU 0, seed 0
#   bash run_srt_inflora.sh 1,2 0,1,2     # Multi-GPU 1,2, seeds 0,1,2
#   bash run_srt_inflora.sh cpu 0          # CPU (very slow, not recommended)
# =============================================================================

set -e

# ── Defaults ────────────────────────────────────────────────────────────────
DEVICE="${1:-0}"
SEED="${2:-0}"
CONFIG="configs/srt_inflora.json"

# ── Resolve project root ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Check environment ─────────────────────────────────────────────────────
if ! python -c "import timm; import torch; import numpy" 2>/dev/null; then
    echo "[ERROR] Missing dependencies. Run:"
    echo "    conda env create -f environment.yaml"
    echo "    conda activate python3_8"
    exit 1
fi

PY_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "[INFO] Python $PY_VERSION"
echo "[INFO] PyTorch $(python -c 'import torch; print(torch.__version__)')"
echo "[INFO] timm   $(python -c 'import timm; print(timm.__version__)')"
echo "[INFO] Device: $DEVICE"
echo "[INFO] Seed:   $SEED"
echo "[INFO] Config: $CONFIG"

# ── Run ───────────────────────────────────────────────────────────────────
python main.py \
    --config "$CONFIG" \
    --device "$DEVICE"

echo "[DONE] Logs saved to logs/cifar100/*/SRT_InfLoRA/..."
