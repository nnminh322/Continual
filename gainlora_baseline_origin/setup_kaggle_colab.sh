#!/usr/bin/env bash
set -euo pipefail

# Conflict-safe setup for Kaggle/Colab notebooks.
# Works with both conda environments and direct pip installs.
# Usage:
#   bash setup_kaggle_colab.sh

echo "=========================================="
echo "SpecRoute Setup for Kaggle/Colab"
echo "=========================================="

# Detect environment
if [[ -d /kaggle/working ]]; then
  ENV_TYPE="kaggle"
elif [[ -d /content ]]; then
  ENV_TYPE="colab"
else
  ENV_TYPE="local"
fi

echo "[Env] Detected: ${ENV_TYPE}"

# Optional conda mode: USE_CONDA=1 bash setup_kaggle_colab.sh
USE_CONDA="${USE_CONDA:-0}"
if command -v conda >/dev/null 2>&1 && [[ "${USE_CONDA}" == "1" ]]; then
  echo "[Setup] Using conda environment (USE_CONDA=1)"
  CONDA_ENV_NAME="specroute"
  conda env remove -n "${CONDA_ENV_NAME}" -y >/dev/null 2>&1 || true
  conda create -n "${CONDA_ENV_NAME}" python=3.10 -y
  PY_CMD="conda run -n ${CONDA_ENV_NAME} python"
  PIP_CMD="conda run -n ${CONDA_ENV_NAME} python -m pip"
  ACTIVATE_CMD="conda activate ${CONDA_ENV_NAME}"
else
  echo "[Setup] Using current Python environment"
  PY_CMD="python"
  PIP_CMD="python -m pip"
fi

# Upgrade pip and core tools
${PIP_CMD} install --upgrade pip setuptools wheel

echo ""
echo "[Remove] Uninstalling conflicting packages..."
# Remove common preinstalled conflicts
${PIP_CMD} uninstall -y torch torchvision torchaudio pytorch-cuda 2>/dev/null || true
${PIP_CMD} uninstall -y triton cupy cupy-cuda12x 2>/dev/null || true
${PIP_CMD} uninstall -y flash-attn xformers bitsandbytes 2>/dev/null || true
${PIP_CMD} uninstall -y transformers datasets accelerate tokenizers 2>/dev/null || true
${PIP_CMD} uninstall -y peft sentence-transformers torchtune 2>/dev/null || true
${PIP_CMD} uninstall -y apex deepspeed 2>/dev/null || true

echo ""
echo "[Install] Core CUDA stack (PyTorch cu121)..."
# Use flexible constraints to survive index churn.
${PIP_CMD} install --no-cache-dir -q \
  'torch>=2.2,<2.6' 'torchvision>=0.17,<0.21' 'torchaudio>=2.2,<2.6' \
  --index-url https://download.pytorch.org/whl/cu121 || \
${PIP_CMD} install --no-cache-dir -q \
  torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
  --index-url https://download.pytorch.org/whl/cu121

echo "[Install] Project dependencies..."
${PIP_CMD} install --no-cache-dir -q \
  'numpy>=1.26,<2.1' 'scipy>=1.11,<1.15' \
  'transformers>=4.30,<4.41' 'datasets>=2.14,<2.22' 'accelerate>=0.24,<0.35' \
  loralib==0.1.2 'sentencepiece>=0.1.99' \
  nltk==3.8.1 scikit-learn==1.5.1 pandas==2.2.2 \
  'pyarrow>=16,<19' 'protobuf>=3.20.3,<5' tqdm==4.66.5 \
  pynvml==11.5.3

echo "[Install] CuPy (with fallback for Python/CUDA compatibility)..."
if ! ${PIP_CMD} install --no-cache-dir -q cupy-cuda12x==13.6.0; then
  ${PIP_CMD} install --no-cache-dir -q 'cupy-cuda12x>=13.0,<15.0'
fi

echo ""
echo "[Check] Verifying installation..."
${PY_CMD} - <<'PY'
import sys
try:
    import torch
    import transformers
    import datasets
    import cupy
    print(f'✓ python {sys.version.split()[0]}')
    print(f'✓ torch {torch.__version__}')
    print(f'✓ cuda available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'✓ gpu count: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'  - GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)')
    print(f'✓ transformers {transformers.__version__}')
    print(f'✓ datasets {datasets.__version__}')
    print(f'✓ cupy {cupy.__version__}')
    print('\n[Success] All packages installed correctly!')
except Exception as e:
    print(f'[Error] {e}', file=sys.stderr)
    sys.exit(1)
PY

echo ""
echo "=========================================="
if [[ -n "${ACTIVATE_CMD:-}" ]]; then
    echo "[Ready] Run: ${ACTIVATE_CMD}"
else
    echo "[Ready] Environment is ready to use"
fi
echo "=========================================="
