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
# Remove common preinstalled conflicts (deep clean).
# JAX/jaxlib MUST go because they require numpy>=2.0 which breaks our pin.
# Also remove packages that depend on torch/jax to suppress pip warnings later.
${PIP_CMD} uninstall -y \
  torch torchvision torchaudio pytorch-cuda triton \
  cupy cupy-cuda12x flash-attn xformers bitsandbytes \
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
  2>/dev/null || true

echo ""
echo "[Install] Core CUDA stack (PyTorch cu121)..."
# 1) Pre-pin numpy+protobuf so torch install doesn't drag in wrong versions.
# 2) No --force-reinstall: that flag reinstalls ALL deps (including numpy) → disaster.
#    Since torch was uninstalled above, pip will install 2.5.1 fresh.
# Note: pip may show "ERROR: dependency resolver" warnings from other Kaggle packages.
#       These are harmless — those packages (kaggle-environments, dopamine-rl, etc.)
#       won't be used and don't affect our training. We filter them for readability.
${PIP_CMD} install --no-cache-dir -q \
  "numpy==1.26.4" "protobuf==5.29.3" 2>&1 | grep -v "which is not installed\|which is incompatible\|dependency resolver" || true
${PIP_CMD} install --no-cache-dir -q \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121 2>&1 | grep -v "which is not installed\|which is incompatible\|dependency resolver" || true

# Sanity-check: abort early if wrong torch survived
TORCH_VER=$(${PY_CMD} -c "import torch; print(torch.__version__)" 2>/dev/null || echo "NONE")
echo "[Check] torch version after install: ${TORCH_VER}"
if [[ "${TORCH_VER}" != 2.5.1* ]]; then
  echo "[WARN] torch ${TORCH_VER} != 2.5.1 — forcing pip uninstall+reinstall..."
  ${PIP_CMD} uninstall -y torch torchvision torchaudio 2>/dev/null || true
  ${PIP_CMD} install --no-cache-dir -q --no-deps \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
  TORCH_VER=$(${PY_CMD} -c "import torch; print(torch.__version__)" 2>/dev/null || echo "NONE")
  echo "[Check] torch version after force: ${TORCH_VER}"
  if [[ "${TORCH_VER}" != 2.5.1* ]]; then
    echo "[ERROR] Cannot install torch 2.5.1. Aborting."
    exit 1
  fi
fi

echo "[Install] Project dependencies..."
${PIP_CMD} install --no-cache-dir -q \
  "numpy==1.26.4" "scipy==1.14.1" \
  transformers==4.40.2 tokenizers==0.19.1 \
  "datasets==2.21.0" accelerate==0.34.2 \
  loralib==0.1.2 sentencepiece==0.2.0 ipdb==0.13.13 \
  nltk==3.9.1 scikit-learn==1.6.1 pandas==2.2.2 \
  "pyarrow==17.0.0" "protobuf==5.29.3" tqdm==4.67.1 \
  "fsspec==2024.6.1" pynvml==11.5.3 \
  2>&1 | grep -v "which is not installed\|which is incompatible\|dependency resolver" || true

echo "[Install] CuPy (with fallback for Python/CUDA compatibility)..."
if ! ${PIP_CMD} install --no-cache-dir -q cupy-cuda12x==13.6.0; then
  ${PIP_CMD} install --no-cache-dir -q 'cupy-cuda12x>=13.0,<15.0'
fi

echo "[Cache] Clearing stale HuggingFace dataset module cache..."
rm -rf ~/.cache/huggingface/modules/datasets_modules/ 2>/dev/null || true
echo "[Cache] HF dataset module cache cleared"

echo "[Symlink] Creating config directory aliases..."
# assets.py references configs/Long_Sequence/* and configs/SuperNI/*
if [ -d configs/gen_script_long_order3_t5_configs ] && [ ! -e configs/Long_Sequence ]; then
  ln -s gen_script_long_order3_t5_configs configs/Long_Sequence
  echo "  Created symlink: configs/Long_Sequence -> gen_script_long_order3_t5_configs"
fi
if [ -d configs/gen_script_superni_order1_t5_configs ] && [ ! -e configs/SuperNI ]; then
  ln -s gen_script_superni_order1_t5_configs configs/SuperNI
  echo "  Created symlink: configs/SuperNI -> gen_script_superni_order1_t5_configs"
fi
echo "[Cache] HF dataset module cache cleared"

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
