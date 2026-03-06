#!/usr/bin/env bash
set -e

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

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "[Setup] Using conda environment"
    CONDA_ENV_NAME="specroute"
    
    # Remove if exists
    conda env remove -n "${CONDA_ENV_NAME}" -y 2>/dev/null || true
    
    # Create new environment with Python 3.10
    conda create -n "${CONDA_ENV_NAME}" python=3.10 -y
    
    # Use conda-specific activation
    ACTIVATE_CMD="conda activate ${CONDA_ENV_NAME}"
    echo "[Setup] Created conda env: ${CONDA_ENV_NAME}"
    echo "Activate with: ${ACTIVATE_CMD}"
else
    echo "[Setup] Using system pip (venv not available, installing directly)"
fi

# Upgrade pip and core tools
python -m pip install --upgrade pip setuptools wheel 2>&1 | grep -v "already satisfied" || true

echo ""
echo "[Remove] Uninstalling conflicting packages..."
# Remove common preinstalled conflicts
pip uninstall -y torch torchvision torchaudio pytorch-cuda 2>/dev/null || true
pip uninstall -y triton cupy cupy-cuda12x 2>/dev/null || true  
pip uninstall -y flash-attn xformers 2>/dev/null || true
pip uninstall -y transformers datasets accelerate 2>/dev/null || true
pip uninstall -y apex deepspeed 2>/dev/null || true

echo ""
echo "[Install] Core CUDA stack (PyTorch 2.1.0 cu121)..."
pip install --no-cache-dir -q \
  torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu121

echo "[Install] Project dependencies..."
pip install --no-cache-dir -q \
  numpy==1.26.4 scipy==1.14.0 \
  transformers==4.30.2 tokenizers==0.13.3 \
  datasets==2.14.6 accelerate==0.24.1 \
  cupy-cuda12x==12.1.0 \
  loralib==0.1.2 sentencepiece==0.2.0 \
  nltk==3.8.1 scikit-learn==1.5.1 pandas==2.2.2 \
  pyarrow==17.0.0 protobuf==3.20.3 tqdm==4.66.5 \
  pynvml==11.5.3

echo ""
echo "[Check] Verifying installation..."
python - <<'PY'
import sys
try:
    import torch
    import transformers
    import datasets
    import cupy
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

if [ $? -ne 0 ]; then
    echo "[Error] Installation verification failed!"
    exit 1
fi

echo ""
echo "=========================================="
if [[ -n "${ACTIVATE_CMD:-}" ]]; then
    echo "[Ready] Run: ${ACTIVATE_CMD}"
else
    echo "[Ready] Environment is ready to use"
fi
echo "=========================================="
