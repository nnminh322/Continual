#!/bin/bash
# =============================================================================
# run_srt_inflora.sh
#
# Run SRT_InfLoRA on CIFAR-100 or DomainNet.
# Defaults: SGWI=true, full LoRA (A+B trainable), no GPM/DualGPM.
#
# Usage:
#   bash run_srt_inflora.sh [BENCHMARK] [DEVICE] [SEED]
#
# Examples:
#   bash run_srt_inflora.sh cifar100 0 0      # CIFAR-100, GPU 0, seed 0
#   bash run_srt_inflora.sh domainnet 0 0     # DomainNet, GPU 0, seed 0
#   bash run_srt_inflora.sh cifar100 1 2      # GPU 1, seed 2
#
# Benchmark:
#   cifar100    – class-incremental: 10 tasks × 10 classes
#   domainnet   – domain-incremental: 5 tasks × 69 classes
# =============================================================================

set -e

# ── Defaults ────────────────────────────────────────────────────────────────
BENCHMARK="${1:-cifar100}"
DEVICE="${2:-0}"
SEED="${3:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Config mapping ───────────────────────────────────────────────────────────
case "$BENCHMARK" in
    cifar100)
        CONFIG="configs/srt_inflora.json"
        ;;
    domainnet)
        CONFIG="configs/domainnet_srt_inflora.json"
        ;;
    *)
        echo "[ERROR] Unknown benchmark: $BENCHMARK"
        echo "Usage: bash run_srt_inflora.sh [cifar100|domainnet] [DEVICE] [SEED]"
        exit 1
        ;;
esac

# ── Check environment ─────────────────────────────────────────────────────
if ! python -c "import timm; import torch; import numpy" 2>/dev/null; then
    echo "[ERROR] Missing dependencies. Run:"
    echo "    conda env create -f environment.yaml"
    echo "    conda activate python3_8"
    exit 1
fi

PY_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "[INFO] Benchmark: $BENCHMARK"
echo "[INFO] Python $PY_VERSION"
echo "[INFO] PyTorch $(python -c 'import torch; print(torch.__version__)')"
echo "[INFO] timm   $(python -c 'import timm; print(timm.__version__)')"
echo "[INFO] Device: $DEVICE"
echo "[INFO] Seed:   $SEED"
echo "[INFO] Config: $CONFIG"

# ── Validate config exists ─────────────────────────────────────────────────
if [ ! -f "$CONFIG" ]; then
    echo "[ERROR] Config not found: $CONFIG"
    exit 1
fi

# ── Run ───────────────────────────────────────────────────────────────────
python main.py \
    --config "$CONFIG" \
    --device "$DEVICE"

LOGDIR="logs/${BENCHMARK}/"
echo "[DONE] Results saved under $LOGDIR"
