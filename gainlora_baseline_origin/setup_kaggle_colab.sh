#!/usr/bin/env bash
set -euo pipefail

# Conflict-safe setup for Kaggle/Colab notebooks.
# Usage:
#   bash setup_kaggle_colab.sh
#   source .venv/bin/activate

if [[ -d /kaggle/working ]]; then
  ROOT_DIR="/kaggle/working"
elif [[ -d /content ]]; then
  ROOT_DIR="/content"
else
  ROOT_DIR="$PWD"
fi

VENV_DIR="${ROOT_DIR}/.venv"

echo "[Setup] Root: ${ROOT_DIR}"
echo "[Setup] Creating venv at: ${VENV_DIR}"
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# Remove common preinstalled conflicts from notebook images.
pip uninstall -y torch torchvision torchaudio triton cupy cupy-cuda12x flash-attn xformers || true

# Core CUDA stack (PyTorch cu121 wheels work well on CUDA 12.x runtime).
pip install --no-cache-dir \
  torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu121

# Project-critical packages.
pip install --no-cache-dir \
  numpy==1.26.4 scipy==1.14.0 \
  transformers==4.30.2 tokenizers==0.13.3 \
  datasets==2.14.6 accelerate==0.24.1 \
  cupy-cuda12x==12.1.0 \
  loralib==0.1.2 sentencepiece==0.2.0 \
  nltk==3.8.1 scikit-learn==1.5.1 pandas==2.2.2 \
  pyarrow==17.0.0 protobuf==3.20.3 tqdm==4.66.5

python - <<'PY'
import torch, transformers, datasets, cupy
print('[Check] torch:', torch.__version__)
print('[Check] cuda available:', torch.cuda.is_available())
print('[Check] gpu count:', torch.cuda.device_count())
print('[Check] transformers:', transformers.__version__)
print('[Check] datasets:', datasets.__version__)
print('[Check] cupy:', cupy.__version__)
PY

echo "[Done] Environment ready. Activate with: source ${VENV_DIR}/bin/activate"
