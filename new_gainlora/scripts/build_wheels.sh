#!/usr/bin/env bash
set -euo pipefail

# Usage: ./build_wheels.sh [OUT_DIR] [PYTORCH_WHEEL_DIR]
# Default OUT_DIR: /kaggle/working/srt-sgwi
# Default PYTORCH_WHEEL_DIR: /kaggle/working/specroute_wheels

OUT_DIR=${1:-/kaggle/working/srt-sgwi}
PYTORCH_WHEEL_DIR=${2:-/kaggle/working/specroute_wheels}
PYTHON_PIP=${PYTHON_PIP:-python -m pip}

mkdir -p "$OUT_DIR"
mkdir -p "$PYTORCH_WHEEL_DIR"

echo "Output directory: $OUT_DIR"
echo "PyTorch wheel directory: $PYTORCH_WHEEL_DIR"

# 1) Download PyTorch nightlies (prefer cu128 -> cu130 -> cpu)
echo "\n==> Downloading PyTorch nightlies into $PYTORCH_WHEEL_DIR"
if ${PYTHON_PIP} download --dest "$PYTORCH_WHEEL_DIR" --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128; then
    echo "Downloaded PyTorch (cu128)"
elif ${PYTHON_PIP} download --dest "$PYTORCH_WHEEL_DIR" --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu130; then
    echo "Downloaded PyTorch (cu130)"
else
    echo "Falling back to PyTorch CPU wheels (nightly)"
    ${PYTHON_PIP} download --dest "$PYTORCH_WHEEL_DIR" --pre torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/nightly/cpu || true
fi

# 2) Common pinned/core packages (downloads dependencies as well)
echo "\n==> Downloading core packages into $OUT_DIR"
${PYTHON_PIP} download --dest "$OUT_DIR" \
  "numpy==2.2.2" \
  "protobuf==5.29.3" \
  "pyarrow==17.0.0" \
  "fsspec==2024.6.1" \
  "transformers>=5.0.0" \
  "tokenizers>=0.21.0" \
  "datasets>=3.0.0" \
  "accelerate>=1.3.0" \
  "safetensors>=0.3.0" \
  "loralib==0.1.2" \
  "ipdb==0.13.13" \
  "nltk==3.9.1" \
  "absl-py==2.0.0" \
  "packaging" \
  "sentencepiece" \
  "tqdm" \
  "psutil" \
  "pyyaml" \
  2>&1 | sed -E 's/^/  /'

# 3) Optional/accelerator packages (may fail; we continue)
echo "\n==> Downloading optional accelerator packages (may be CUDA-specific)"
# CuPy: try prebuilt CUDA-12x wheel (cupy-cuda12x), fallback to generic cupy
if ${PYTHON_PIP} download --dest "$OUT_DIR" --no-deps cupy-cuda12x 2>/dev/null; then
  echo "Downloaded cupy-cuda12x"
else
  echo "cupy-cuda12x not available; trying generic cupy"
  ${PYTHON_PIP} download --dest "$OUT_DIR" --no-deps cupy || echo "cupy download failed; skipping"
fi

# flash-attn (best-effort)
${PYTHON_PIP} download --dest "$OUT_DIR" flash-attn 2>/dev/null || echo "flash-attn download failed; skipping"

# bitsandbytes (may have manylinux+cuda wheel or require building)
${PYTHON_PIP} download --dest "$OUT_DIR" bitsandbytes 2>/dev/null || echo "bitsandbytes download failed; skipping"

# 4) Additional useful tools
${PYTHON_PIP} download --dest "$OUT_DIR" tokenizers sentencepiece regex || true

# 5) Summarize
echo "\n==> Summary"
echo "Wheels in $OUT_DIR:"
ls -1 "$OUT_DIR" | sed -E 's/^/  /' || true

echo "Wheels in $PYTORCH_WHEEL_DIR (PyTorch nightlies):"
ls -1 "$PYTORCH_WHEEL_DIR" | sed -E 's/^/  /' || true

cat <<'EOF'

Notes:
- This script tries to download prebuilt binary wheels; some packages (flash-attn, bitsandbytes)
  are CUDA/specialized and may not have wheels for your CUDA version. You can build missing wheels
  locally with `pip wheel --wheel-dir <OUT_DIR> <package>`.
- If you need a fully offline bundle, combine both directories and use `pip install --no-index --find-links <dir>`.
EOF
