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

PY_CMD="python"
PIP_CMD="python -m pip"

PY_VER=$(${PY_CMD} -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
echo "[Setup] Using Python ${PY_VER} in active environment"

${PIP_CMD} install --upgrade pip setuptools wheel

echo ""
echo "[Remove] Uninstalling conflicting packages..."
${PIP_CMD} uninstall -y \
  torch torchvision torchaudio pytorch-cuda triton \
  cupy cupy-cuda12x cupy-cuda124 cupy-cuda126 cupy-cuda-default \
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
  absl-py \
  2>/dev/null || true

echo ""
echo "[Install] PyTorch (CUDA 12.4 — RTX 5090 / Blackwell)..."
${PIP_CMD} install --no-cache-dir -q \
  "torch==2.5.1+cu124" "torchvision==0.20.1+cu124" "torchaudio==2.5.1+cu124" \
  --extra-index-url https://download.pytorch.org/whl/cu124 2>&1 | \
  grep -v "which is not installed\|which is incompatible\|dependency resolver\|not a recognized" || true

# Fallback: if +cu124 wheels unavailable, try cu121 (older GPUs)
TORCH_VER=$(${PY_CMD} -c "import torch; print(torch.__version__)" 2>/dev/null || echo "NONE")
if [[ "${TORCH_VER}" == "NONE" ]]; then
  echo "[Fallback] cu124 wheels not available, trying cu121..."
  ${PIP_CMD} install --no-cache-dir -q \
    "torch==2.5.1+cu121" "torchvision==0.20.1+cu121" "torchaudio==2.5.1+cu121" \
    --extra-index-url https://download.pytorch.org/whl/cu121 2>&1 | \
    grep -v "which is not installed\|which is incompatible\|dependency resolver\|not a recognized" || true
fi

TORCH_VER=$(${PY_CMD} -c "import torch; print(torch.__version__)" 2>/dev/null || echo "NONE")
echo "[Check] torch version: ${TORCH_VER}"
if [[ "${TORCH_VER}" == "NONE" ]]; then
  echo "[ERROR] torch not installed. Check CUDA/PyTorch compatibility."
  exit 1
fi

echo ""
echo "[Install] CuPy (install FIRST to avoid numpy upgrade conflict)..."
# cupy-cuda12x from conda-forge-style wheel should NOT drag numpy>=2.0
${PIP_CMD} install --no-cache-dir -q --no-deps cupy-cuda12x \
  2>&1 | grep -v "which is not installed\|which is incompatible\|dependency resolver" || true

echo ""
echo "[Install] Pinned dependencies (numpy=1.26.4 kept intentionally)..."
${PIP_CMD} install --no-cache-dir -q \
  "numpy==1.26.4" \
  "scipy==1.14.1" \
  "protobuf==5.29.3" \
  "pyarrow==17.0.0" \
  "fsspec==2024.6.1" \
  "tqdm==4.67.1" \
  "pynvml==11.5.3" \
  "pandas==2.2.2" \
  "scikit-learn==1.6.1" \
  "sentencepiece==0.2.0" \
  "loralib==0.1.2" \
  "ipdb==0.13.13" \
  "nltk==3.9.1" \
  "accelerate==0.34.2" \
  "absl-py==2.0.0" \
  2>&1 | grep -v "which is not installed\|which is incompatible\|dependency resolver" || true

echo ""
echo "[Install] transformers + tokenizers (EXACT pinned versions)..."
${PIP_CMD} install --no-cache-dir -q \
  "transformers==4.40.2" "tokenizers==0.19.1" \
  2>&1 | grep -v "which is not installed\|which is incompatible\|dependency resolver" || true

echo ""
echo "[Install] datasets (compatible version for Python 3.13)..."
# datasets 2.21.0 needs older deps; use a newer but compatible version
${PIP_CMD} install --no-cache-dir -q \
  "datasets==3.2.0" \
  2>&1 | grep -v "which is not installed\|which is incompatible\|dependency resolver" || true

echo ""
echo "[Install] Flash Attention 2 (optional, Blackwell support)..."
${PIP_CMD} install --no-cache-dir -q 'flash-attn>=2.5.0' \
  2>&1 | grep -v "which is not installed\|which is incompatible\|dependency resolver\|no such option" || true

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
    print(f'✓ numpy {__import__("numpy").__version__}')
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
echo "[Ready] Environment is ready — now remove --load_best_model_at_end from script:"
echo "  sed -i 's/--load_best_model_at_end //g' gen_script_long_order3_t5_srt_hard.sh"
echo "=========================================="