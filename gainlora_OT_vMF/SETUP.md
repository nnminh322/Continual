# GainLoRA + OT-SIGN Setup Guide

## Quick Start

### Option 1: Using conda + pip (Recommended for GPU)

```bash
# 1. Create conda environment with CUDA support
conda create -n gainlora \
  python=3.10 \
  pytorch::pytorch \
  pytorch::pytorch-cuda=12.1 \
  pytorch::torchvision \
  pytorch::torchaudio \
  -c pytorch -c nvidia

# 2. Activate environment
conda activate gainlora

# 3. Install pip dependencies
cd gainlora
pip install -r requirements.txt
```

### Option 2: Using only pip (CPU or if CUDA already setup)

```bash
# Install all pip dependencies
pip install -r requirements.txt
```

### Option 3: From conda environment (existing setup)

```bash
# If you already have an environment, just update pip packages
conda activate <your_env>
pip install -r requirements.txt
```

---

## Key Dependencies

| Package | Purpose | Version |
|---------|---------|---------|
| transformers | HuggingFace models (T5, LLaMA, etc.) | 4.30.2 |
| torch/vision/audio | PyTorch + CUDA support | 2.1.0 |
| datasets | Dataset loading & processing | 2.14.6 |
| accelerate | Distributed training | 0.24.1 |
| deepspeed | ZeRO optimization | 0.11.2 |
| loralib | LoRA layers | 0.1.2 |
| cupy-cuda12x | GPU-accelerated arrays (for SVD) | 12.1.0 |
| safetensors | Fast model serialization | 0.4.4 |

---

## Troubleshooting

### CUDA Version Mismatch
If you have a different CUDA version:
```bash
# Change 12.1 to your CUDA version (11.8, 11.6, etc.)
conda create -n gainlora ... pytorch::pytorch-cuda=11.8 ...
```

### CuPy Installation Fails
CuPy requires CUDA dev toolkit. If it fails:
```bash
# Option 1: Install it separately
conda install -c conda-forge cupy cuda-version=12.1

# Option 2: Comment it out in requirements.txt if not using SVD GPM
```

### Out of Memory on 2×T4
Scripts are already optimized (batch_size=8, fp16). If still OOM:
1. Reduce `per_device_train_batch_size` in bash scripts
2. Increase `gradient_accumulation_steps`

### Kaggle/Colab Environment (Pre-built Wheels)

When running on Kaggle or Google Colab, use pre-built wheels to avoid compilation:

```bash
# Clone and install (Colab/Kaggle cell)
! git clone https://github.com/liangyanshuo/gainlora.git
! cd gainlora && pip install -r requirements.txt --no-build-isolation

# Or with uv proxy (faster) - already recommended in Colab/Kaggle
! pip install uv
! uv pip install -r requirements.txt
```

**Note:** `tokenizers>=0.14.1` has pre-built wheels for Python 3.12. If you still get build errors:
- Add GPU dependencies: `! pip install --upgrade setuptools wheel`
- Or downgrade Python: Colab uses 3.10-3.12; ensure compatibility

---

## Running OT-SIGN + GainLoRA

```bash
# Order 1 (SuperNI) - ~10 hours on 2×T4
bash run_ot_sign_order1_t5large.sh 0,1 google/flan-t5-large

# Order 2 (SuperNI) - ~10 hours
bash run_ot_sign_order2_t5large.sh 0,1 google/flan-t5-large

# Order 3 (Long) - ~4 hours  [run this first to test]
bash run_ot_sign_order3_t5large.sh 0,1 google/flan-t5-large

# Order 4 (Long) - ~4 hours
bash run_ot_sign_order4_t5large.sh 0,1 google/flan-t5-large
```

Results and logs:
- Training outputs: `logs_and_outputs/ot_sign_order{1,2,3,4}_t5large/`
- AP/FT scores: Printed in terminal + `ap_ft_result.json`
- Compare with baseline: See `results/comparison_results.md`

---

## Environment Verification

```bash
# Check PyTorch + CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"

# Check transformers
python -c "import transformers; print(f'Transformers {transformers.__version__}')"

# Check other deps
python -c "import cupy; import accelerate; import deepspeed; print('All deps OK')"
```

---

## Notes

- `requirements.yaml` = Full conda environment spec (includes system libs)
- `requirements.txt` = Python pip packages only (easier to share and modify)
- Both methods should give identical Python environment
- CUDA/cuDNN installation separate (via conda in setup)
