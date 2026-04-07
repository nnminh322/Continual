#!/usr/bin/env bash
set -euo pipefail

# Setup script for GPU servers (e.g., RTX 5090 / Blackwell)
# Installs everything INTO THE CURRENTLY ACTIVE environment.
# Does NOT create or manage conda environments.
# Usage:
#   # Make sure your env is activated, then run:
#   cd /path/to/new_gainlora
#   bash setup_server.sh

echo "=========================================="
echo "SpecRoute Setup for GPU Server (RTX 5090)"
echo "=========================================="

# Detect GPU
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")
  echo "[Env] GPU: ${GPU_NAME}"
else
  echo "[Env] No NVIDIA GPU detected — continuing anyway"
fi

# Always use the currently active python/pip — no conda env management.
PY_CMD="python"
PIP_CMD="python -m pip"

echo "[Setup] Using active Python: $(${PY_CMD} --version 2>/dev/null)"

# Upgrade pip and core tools
${PIP_CMD} install --upgrade pip setuptools wheel

echo ""
echo "[Remove] Uninstalling conflicting packages..."
${PIP_CMD} uninstall -y \
  torch torchvision torchaudio pytorch-cuda triton \
  cupy-cuda12x cupy-cuda124 cupy-cuda126 cupy-cuda-default \
  flash-attn xformers bitsandbytes \
  transformers tokenizers datasets accelerate peft \
  sentence-transformers torchtune deepspeed apex \
  cudf-cu12 dask-cudf-cu12 cuml-cu12 cucim-cu12 \
  ydf grain umap-learn hdbscan textblob \
  opentelemetry-proto grpcio-status \
  protobuf \
  jax jaxlib \
  torchmetrics easyocr pytorch-lightning fastai \
  gymnax kaggle-environments dopamine-rl \
  flax optax orbax-checkpoint chex \
  nx-cugraph-cu12 pylibcugraph-cu12 \
  opencv-contrib-python opencv-python opencv-python-headless \
  shap rasterio tobler cesium pytensor \
  tensorflow-decision-forests \
  a2a-sdk bigframes google-adk gcsfs \
  ninja \
  2>/dev/null || true

echo ""
echo "[Install] Core CUDA stack (PyTorch cu124 for RTX 5090 / Blackwell)..."
${PIP_CMD} install --no-cache-dir -q \
  "numpy==1.26.4" "protobuf==5.29.3" 2>&1 | grep -v "which is not installed\|which is incompatible\|dependency resolver" || true

# PyTorch with CUDA 12.4 — RTX 5090 / Blackwell CC 9.0
${PIP_CMD} install --no-cache-dir -q \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124 2>&1 | grep -v "which is not installed\|which is incompatible\|dependency resolver" || true

# Sanity-check
TORCH_VER=$(${PY_CMD} -c "import torch; print(torch.__version__)" 2>/dev/null || echo "NONE")
echo "[Check] torch version: ${TORCH_VER}"
if [[ "${TORCH_VER}" != 2.5.1* ]]; then
  echo "[WARN] torch ${TORCH_VER} != 2.5.1 — forcing uninstall+reinstall..."
  ${PIP_CMD} uninstall -y torch torchvision torchaudio 2>/dev/null || true
  ${PIP_CMD} install --no-cache-dir -q --no-deps \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124
  TORCH_VER=$(${PY_CMD} -c "import torch; print(torch.__version__)" 2>/dev/null || echo "NONE")
  echo "[Check] torch version after force: ${TORCH_VER}"
  if [[ "${TORCH_VER}" != 2.5.1* ]]; then
    echo "[ERROR] Cannot install torch 2.5.1. Aborting."
    exit 1
  fi
fi

echo ""
echo "[Install] Core dependencies..."
${PIP_CMD} install --no-cache-dir -q \
  "numpy==1.26.4" "scipy==1.14.1" \
  transformers==4.40.2 tokenizers==0.19.1 \
  "datasets==2.21.0" accelerate==0.34.2 \
  loralib==0.1.2 sentencepiece==0.2.0 ipdb==0.13.13 \
  nltk==3.9.1 scikit-learn==1.6.1 pandas==2.2.2 \
  "pyarrow==17.0.0" "protobuf==5.29.3" tqdm==4.67.1 \
  "fsspec==2024.6.1" pynvml==11.5.3 \
  2>&1 | grep -v "which is not installed\|which is incompatible\|dependency resolver" || true

echo ""
echo "[Install] Flash Attention 2 (CC 9.0 Blackwell support)..."
${PIP_CMD} install --no-cache-dir -q 'flash-attn>=2.5.0' \
  2>&1 | grep -v "which is not installed\|which is incompatible\|dependency resolver" || true

echo ""
echo "[Install] CuPy CUDA 12.x..."
if ! ${PIP_CMD} install --no-cache-dir -q 'cupy-cuda12x>=13.0,<15.0'; then
  ${PIP_CMD} install --no-cache-dir -q 'cupy-cuda12x'
fi

echo ""
echo "[Symlink] Creating config directory aliases..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [ -d configs/gen_script_long_order3_t5_configs ] && [ ! -e configs/Long_Sequence ]; then
  ln -s gen_script_long_order3_t5_configs configs/Long_Sequence
  echo "  Created: configs/Long_Sequence -> gen_script_long_order3_t5_configs"
fi
if [ -d configs/gen_script_superni_order1_t5_configs ] && [ ! -e configs/SuperNI ]; then
  ln -s gen_script_superni_order1_t5_configs configs/SuperNI
  echo "  Created: configs/SuperNI -> gen_script_superni_order1_t5_configs"
fi

echo ""
echo "[Cache] Clearing stale HuggingFace dataset module cache..."
rm -rf ~/.cache/huggingface/modules/datasets_modules/ 2>/dev/null || true

echo ""
echo "[Check] Verifying installation..."
${PY_CMD} - <<'PY'
import sys
try:
    import torch
    import transformers
    import datasets
    print(f'✓ python {sys.version.split()[0]}')
    print(f'✓ torch {torch.__version__}')
    print(f'✓ cuda available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'✓ gpu count: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f'  - GPU {i}: {props.name} ({props.total_mem/1e9:.1f} GB, CC {props.major}.{props.minor})')
    print(f'✓ transformers {transformers.__version__}')
    print(f'✓ datasets {datasets.__version__}')
    try:
        import cupy
        print(f'✓ cupy {cupy.__version__}')
    except ImportError:
        print('⚠ cupy not installed (optional)')
    try:
        import flash_attn
        print(f'✓ flash-attn installed')
    except ImportError:
        print('⚠ flash-attn not installed (optional)')
    print('\n[Success] All packages installed!')
except Exception as e:
    print(f'[Error] {e}', file=sys.stderr)
    sys.exit(1)
PY

echo ""
echo "=========================================="
echo "[Driver & CUDA]"
nvidia-smi --query-gpu=name,driver_version,cuda_version --format=csv,noheader 2>/dev/null | while read line; do
  echo "  ${line}"
done
echo "=========================================="
echo "[Ready] Env is ready — run your training script."
echo "=========================================="