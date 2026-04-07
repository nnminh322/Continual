#!/usr/bin/env bash
set -euo pipefail

# Setup script for GPU servers (RTX 5090 / Blackwell sm_120)
# Installs everything INTO THE CURRENTLY ACTIVE environment.
# Usage:
#   source .venv/bin/activate   # hoặc conda activate <env>
#   bash setup_server.sh

echo "=========================================="
echo "SpecRoute Setup for GPU Server (RTX 5090)"
echo "=========================================="

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")
  echo "[GPU] ${GPU_NAME}"
fi

PY_CMD="python"
PIP_CMD="python -m pip"
PY_VER=$(${PY_CMD} -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
echo "[Python] ${PY_VER}"

${PIP_CMD} install --upgrade pip wheel

# ── 1. Uninstall conflicting torch (cu124 blocks cu128/cu130) ──────────────
echo ""
echo "[1/5] Uninstalling old CUDA packages..."
${PIP_CMD} uninstall -y \
  torch torchvision torchaudio triton \
  cupy cupy-cuda12x cuda-pathfinder \
  flash-attn \
  2>/dev/null || true

# ── 2. Install PyTorch nightly (sm_120 / RTX 5090 Blackwell support) ─────────
#    RTX 5090 (CC 12.0) needs CUDA 12.8+ — only available in nightly builds.
#    Try cu128 first, fallback to cu130.
echo ""
echo "[2/5] Installing PyTorch nightly (sm_120 support)..."
if ! ${PIP_CMD} install --no-cache-dir -q --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128; then
  echo "[Fallback] cu128 unavailable, trying cu130..."
  ${PIP_CMD} install --no-cache-dir -q --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu130
fi

TORCH_VER=$(${PY_CMD} -c "import torch; print(torch.__version__)" 2>/dev/null || echo "NONE")
echo "[Check] torch ${TORCH_VER}"

# Check sm_120 support — no warning means compatible
if ${PY_CMD} -c "import torch; torch.cuda.get_device_capability()" 2>&1 | grep -q "not compatible"; then
  echo "[ERROR] PyTorch still doesn't support sm_120. Check PyTorch nightly availability."
  echo "        Try: https://pytorch.org/get-started/locally/"
  exit 1
fi

# ── 3. Core Python dependencies (exact pinned versions) ───────────────────────
echo ""
echo "[3/5] Installing pinned Python dependencies..."
${PIP_CMD} install --no-cache-dir -q \
  "numpy==1.26.4" \
  "protobuf==5.29.3" \
  "pyarrow==17.0.0" \
  "fsspec==2024.6.1" \
  "tqdm==4.67.1" \
  "pynvml==11.5.3" \
  "pandas==2.2.2" \
  "scipy==1.14.1" \
  "scikit-learn==1.6.1" \
  "sentencepiece==0.2.0" \
  "loralib==0.1.2" \
  "ipdb==0.13.13" \
  "nltk==3.9.1" \
  "accelerate==0.34.2" \
  "absl-py==2.0.0" \
  2>&1 | grep -v "which is not installed\|which is incompatible\|dependency resolver" || true

# ── 4. transformers + tokenizers (EXACT — newer versions have breaking changes) ─
echo ""
echo "[4/5] Installing transformers (exact pinned) + datasets..."
${PIP_CMD} install --no-cache-dir -q \
  "transformers==4.40.2" \
  "tokenizers==0.19.1" \
  "datasets==2.21.0" \
  2>&1 | grep -v "which is not installed\|which is incompatible\|dependency resolver" || true

# ── 5. CuPy (optional — compute covariance matrix, not critical for training) ─
#    cupy-cuda12x ships prebuilt; skip if build fails (not blocking).
echo ""
echo "[5/5] Installing CuPy (optional, skip if fails)..."
${PIP_CMD} install --no-cache-dir -q --no-deps cupy-cuda12x \
  2>&1 | grep -v "which is not installed\|which is incompatible\|dependency resolver\|error:" || true

# ── Symlinks ──────────────────────────────────────────────────────────────────
echo ""
echo "[Symlink] Config aliases..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
[ -d configs/gen_script_long_order3_t5_configs ] && \
  [ ! -e configs/Long_Sequence ] && \
  ln -s gen_script_long_order3_t5_configs configs/Long_Sequence
[ -d configs/gen_script_superni_order1_t5_configs ] && \
  [ ! -e configs/SuperNI ] && \
  ln -s gen_script_superni_order1_t5_configs configs/SuperNI

# ── Verify ────────────────────────────────────────────────────────────────────
echo ""
echo "[Verify]"
${PY_CMD} - <<'PY'
import sys
try:
    import torch
    import transformers
    import datasets
    print(f'✓ python {sys.version.split()[0]}')
    print(f'✓ torch {torch.__version__}')
    print(f'✓ numpy {__import__("numpy").__version__}')
    print(f'✓ transformers {transformers.__version__}')
    print(f'✓ datasets {datasets.__version__}')
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability()
        print(f'✓ CUDA available, CC {cc[0]}.{cc[1]}')
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    try:
        import cupy
        print(f'✓ cupy {cupy.__version__}')
    except Exception:
        print('⚠ cupy skipped (optional)')
    print('\n[OK] All good!')
except Exception as e:
    print(f'[ERROR] {e}')
    sys.exit(1)
PY

echo ""
echo "=========================================="
echo "[Next] Xoá --load_best_model_at_end (transformers 4.40.2 validate strict):"
echo "  sed -i 's/--load_best_model_at_end //g' gen_script_long_order3_t5_srt_hard.sh"
echo "=========================================="