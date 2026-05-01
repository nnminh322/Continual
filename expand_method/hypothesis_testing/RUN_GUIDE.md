# SRT Hypothesis Testing — GPU Setup & Run Guide

## Mục lục
1. [Clone & Environment Setup](#1-clone--environment-setup)
2. [Cấu trúc Project](#2-cấu-trúc-project)
3. [Quick Start: Chạy ngay không cần trained checkpoint](#3-quick-start-chạy-ngay-không-cần-trained-checkpoint)
4. [Option A: Routing Accuracy (không cần model)](#4-option-a-routing-accuracy-không-cần-model)
5. [Option B: End-to-End Evaluation (cần trained checkpoint)](#5-option-b-end-to-end-evaluation-cần-trained-checkpoint)
6. [Hướng dẫn chi tiết từng experiment](#6-hướng-dẫn-chi-tiết-từng-experiment)
7. [Đọc kết quả](#7-đọc-kết-quả)
8. [GPU Requirements](#8-gpu-requirements)

---

## 1. Clone & Environment Setup

### 1.1 Clone repository
```bash
# Từ thư mục expand_method (đã có sẵn từ trước)
cd /path/to/expand_method
# hypothesis_testing đã nằm trong expand_method/

# Nếu chưa có, clone:
git clone https://github.com/your-repo/expand_method.git
cd expand_method
```

### 1.2 Tạo Python 3.12 environment (GPU)
```bash
# Tạo conda environment với Python 3.12
conda create -n srt_exp python=3.12 -y
conda activate srt_exp

# Cài PyTorch với CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Cài dependencies cho hypothesis_testing
cd hypothesis_testing
pip install -e .    # hoặc: pip install -r requirements.txt

# Dependencies chính:
#   torch>=2.0.0 (GPU acceleration)
#   transformers>=4.30.0
#   numpy>=1.24.0
#   Pillow>=9.0.0
#   sentence-transformers>=2.2.0
#   tqdm>=4.65.0
#   scikit-learn>=1.3.0
```

### 1.3 Verify GPU access
```bash
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); \
    print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
# Output mong đợi:
#   CUDA available: True
#   GPU: Tesla T4  (hoặc GPU của bạn)
```

### 1.4 Verify packages
```bash
cd hypothesis_testing
python3 -c "
from srt_router.pooled_mahalanobis import PooledMahalanobisRouter
from embedding_extractors.clip_extractor import CLIPVisionExtractor
from srt_router.metrics import compute_routing_accuracy
print('✓ All imports OK — GPU pipeline ready')
"
```

---

## 2. Cấu trúc Project

```
expand_method/
└── hypothesis_testing/
    ├── srt_router/                    # Core SRT router
    │   ├── pooled_mahalanobis.py     # GPU-accelerated Mahalanobis router
    │   ├── metrics.py                # Routing accuracy metrics
    │   └── adapters.py              # Model-specific SRT adapters
    ├── embedding_extractors/
    │   └── clip_extractor.py        # CLIP ViT CLS embeddings (GPU)
    ├── scoring/                      # VQA accuracy scoring
    ├── experiments/
    │   ├── smolora/
    │   │   ├── if_router/            # SMoLoRA IF Router
    │   │   │   ├── routing_accuracy.py  # Option A
    │   │   │   └── end_to_end.py         # Option B
    │   │   ├── vu_router/           # SMoLoRA VU Router
    │   │   │   ├── routing_accuracy.py
    │   │   │   └── end_to_end.py
    │   │   └── dual_router/          # SMoLoRA Dual Router (VU+IF)
    │   │       ├── routing_accuracy.py
    │   │       └── end_to_end.py
    │   └── hide/
    │       └── cosine_router/       # HiDe-LLaVA Cosine → SRT
    │           ├── routing_accuracy.py
    │           └── end_to_end.py
    └── requirements.txt
```

---

## 3. Quick Start: Chạy ngay không cần trained checkpoint

**SMoLoRA IF Router** là experiment nhanh nhất — chỉ cần `ins_emb_single.pkl`:

```bash
cd hypothesis_testing

# Tìm ins_emb pickle (nếu chưa có, tạo dummy)
# ins_emb_single.pkl chứa 7 task embeddings (384-dim) từ Sentence-BERT

# Chạy Option A — IF Router (routing accuracy, ~1 phút)
python experiments/smolora/if_router/routing_accuracy.py \
    --ins_emb path/to/ins_emb_single.pkl \
    --task_names ScienceQA TextVQA GQA VQAv2 Flickr30k ImageNet Place365 \
    --shrinkage ridge \
    --device cuda \
    --n_samples 100 \
    --noise_std 0.1

# Output mẫu:
# [GPU] Tesla T4  15.6GB  === CL Routing [SMoLoRA IF Router] ===
#     Backbone: SMoLoRA (IF Router)
#     Shrinkage: ridge
#     Device: cuda
#
#   [1/7] ScienceQA  (n=7, n/d=0.02, pool=7)
#     Cosine (ORIGINAL)
#     SRT-Mahal_RIDGE (NEW)
#     RESULT Cosine (ORIGINAL)              macro= 71.43%  [100.0%  42.9%]
#     RESULT SRT-Mahal_RIDGE (NEW)        macro= 85.71%  [100.0%  71.4%]
#   ...
#
# ===========================================================================
#   SMoLoRA IF — Final Routing Accuracy (7 tasks, cuda)
# ===========================================================================
#   Method                              Final      Avg  T1     T2     T3  ...
#   ------------------------------------------------------------------------
#   Cosine (ORIGINAL)                   71.43%   79.59%  100.0%  71.4% ...
#   SRT-Mahal_RIDGE (NEW)              85.71%   89.47%  100.0%  85.7% ...
#   Delta (SRT - Original)             +14.29%  +9.88%  +0.0%  +14.3% ...
# ===========================================================================
```

---

## 4. Option A: Routing Accuracy (không cần model)

**Nguyên lý**: Test SRT routing trên embeddings đã extract sẵn, không cần trained model.

### 4.1 SMoLoRA IF Router (nhanh nhất)
```bash
python experiments/smolora/if_router/routing_accuracy.py \
    --ins_emb /path/to/ins_emb_single.pkl \
    --task_names ScienceQA TextVQA GQA VQAv2 \
    --shrinkage ridge \
    --device cuda
```

**Đầu vào**: `ins_emb_single.pkl` — 7 embeddings 384-dim (Sentence-BERT)
**So sánh**: Cosine (ORIGINAL) vs SRT-Mahal (NEW)
**Tốc độ**: ~30 giây

### 4.2 SMoLoRA VU Router (cần ảnh)
```bash
python experiments/smolora/vu_router/routing_accuracy.py \
    --data_root /path/to/task/images \
    --task_names ScienceQA TextVQA GQA VQAv2 \
    --clip_model openai/clip-vit-large-patch14-336 \
    --shrinkage ridge \
    --device cuda \
    --n_train 500 \
    --n_test 200
```

**Đầu vào**: Thư mục chứa ảnh của từng task
**So sánh**: Cosine (ORIGINAL) vs SRT-Mahal (NEW)
**Tốc độ**: ~2-5 phút (tùy số ảnh)

### 4.3 SMoLoRA Dual Router (VU + IF)
```bash
python experiments/smolora/dual_router/routing_accuracy.py \
    --ins_emb /path/to/ins_emb_single.pkl \
    --data_root /path/to/task/images \
    --task_names ScienceQA TextVQA GQA VQAv2 \
    --clip_model openai/clip-vit-large-patch14-336 \
    --shrinkage ridge \
    --alpha 0.5 \
    --device cuda
```

**So sánh**: Cosine VU (ORIGINAL) vs SRT-VU (NEW) vs SRT-Dual (NEW)
**Tốc độ**: ~3-7 phút

### 4.4 HiDe-LLaVA Cosine → SRT
```bash
python experiments/hide/cosine_router/routing_accuracy.py \
    --data_root /path/to/task/images \
    --task_names ScienceQA TextVQA GQA VQAv2 VizWiz TextCaps \
    --clip_model openai/clip-vit-large-patch14-336 \
    --shrinkage ridge \
    --device cuda
```

**So sánh**: Cosine HiDe (ORIGINAL) vs SRT-Mahal (NEW)
**Tốc độ**: ~5-10 phút

---

## 5. Option B: End-to-End Evaluation (cần trained checkpoint)

**Nguyên lý**: Load trained SMoLoRA/HiDe-LLaVA → inject SRT routing → chạy VQA generation → so sánh accuracy.

### 5.1 SMoLoRA IF Router (End-to-End)
```bash
python experiments/smolora/if_router/end_to_end.py \
    --model_path /path/to/smolora/checkpoint \
    --model_base /path/to/vicuna-7b \
    --ins_emb /path/to/ins_emb_single.pkl \
    --task_order ScienceQA TextVQA GQA VQAv2 \
    --routing_mode all \
    --scoring_func vqav2 \
    --device cuda \
    --output_dir results_smolora_if_e2e
```

**Lưu ý**:
- `--routing_mode all` chạy cả 3 modes: `original`, `srt`, `oracle`
- `--scoring_func` có thể là `vqav2`, `science_qa`, hoặc `gqa`

### 5.2 SMoLoRA VU Router (End-to-End)
```bash
python experiments/smolora/vu_router/end_to_end.py \
    --model_path /path/to/smolora/checkpoint \
    --model_base /path/to/vicuna-7b \
    --clip_model openai/clip-vit-large-patch14-336 \
    --task_images_root /path/to/task/images \
    --task_order ScienceQA TextVQA GQA VQAv2 \
    --routing_mode all \
    --device cuda
```

### 5.3 HiDe-LLaVA Cosine → SRT (End-to-End)
```bash
python experiments/hide/cosine_router/end_to_end.py \
    --model_path /path/to/hide/checkpoint \
    --model_base /path/to/vicuna-7b \
    --clip_model openai/clip-vit-large-patch14-336 \
    --task_images_root /path/to/task/images \
    --task_order ScienceQA TextVQA GQA VQAv2 VizWiz TextCaps \
    --routing_mode all \
    --device cuda
```

---

## 6. Hướng dẫn chi tiết từng experiment

### 6.1 SMoLoRA IF Router — Chạy lần đầu

**Bước 1: Tìm ins_emb pickle**
```bash
# Thường nằm ở:
ls expand_method/SMoLoRA/llava/eval/ins_emb_single.pkl
# hoặc:
ls expand_method/SMoLoRA/exp/*/ins_emb_single.pkl
```

**Bước 2: Chạy với 2 shrinkage methods**
```bash
# Ridge (mặc định, recommended)
python experiments/smolora/if_router/routing_accuracy.py \
    --ins_emb /path/to/ins_emb_single.pkl \
    --task_names ScienceQA TextVQA GQA VQAv2 \
    --shrinkage ridge --device cuda

# Ledoit-Wolf
python experiments/smolora/if_router/routing_accuracy.py \
    --ins_emb /path/to/ins_emb_single.pkl \
    --task_names ScienceQA TextVQA GQA VQAv2 \
    --shrinkage lw --device cuda
```

**Bước 3: So sánh kết quả**
```bash
# Kết quả lưu ở:
ls results_smolora_if/smolora_if_routing_ridge_cuda.json
ls results_smolora_if/smolora_if_routing_lw_cuda.json
```

### 6.2 SMoLoRA VU Router — Cần dataset images

**Chuẩn bị data**:
```bash
# Cấu trúc thư mục:
data_root/
├── ScienceQA/
│   ├── image001.jpg
│   ├── image002.png
│   └── ...
├── TextVQA/
│   ├── ...
├── GQA/
│   └── ...
└── VQAv2/
    └── ...
```

**Chạy**:
```bash
python experiments/smolora/vu_router/routing_accuracy.py \
    --data_root /path/to/your/task/images \
    --task_names ScienceQA TextVQA GQA VQAv2 \
    --clip_model openai/clip-vit-large-patch14-336 \
    --shrinkage ridge \
    --device cuda \
    --n_train 500 \
    --n_test 200
```

### 6.3 Dual Router — Kết hợp VU + IF

```bash
python experiments/smolora/dual_router/routing_accuracy.py \
    --ins_emb /path/to/ins_emb_single.pkl \
    --data_root /path/to/task/images \
    --task_names ScienceQA TextVQA GQA VQAv2 \
    --clip_model openai/clip-vit-large-patch14-336 \
    --shrinkage ridge \
    --alpha 0.5 \
    --device cuda

# Thử different alpha values:
for alpha in 0.3 0.5 0.7; do
    python experiments/smolora/dual_router/routing_accuracy.py \
        --ins_emb /path/to/ins_emb_single.pkl \
        --data_root /path/to/task/images \
        --task_names ScienceQA TextVQA GQA VQAv2 \
        --alpha $alpha \
        --device cuda
done
```

---

## 7. Đọc kết quả

### 7.1 JSON output structure
```json
{
  "method": "SMoLoRA IF Router",
  "ins_emb_path": "path/to/ins_emb_single.pkl",
  "n_tasks": 7,
  "task_names": ["ScienceQA", "TextVQA", ...],
  "shrinkage": "ridge",
  "device": "cuda",
  "results": {
    "step_1": {
      "task": "ScienceQA",
      "accuracy_srt": 1.0,
      "accuracy_cosine": 1.0,
      "per_task_srt": {"ScienceQA": 1.0}
    },
    "step_2": {
      "task": "TextVQA",
      "accuracy_srt": 0.857,
      "accuracy_cosine": 0.714,
      "per_task_srt": {"ScienceQA": 1.0, "TextVQA": 0.714}
    }
  }
}
```

### 7.2 In-place print output format
```
[GPU] Tesla T4  15.6GB  === CL Routing [SMoLoRA IF Router] ===
    Backbone: SMoLoRA (IF Router)
    Shrinkage: ridge
    Device: cuda

  [1/7] ScienceQA  (n=7, n/d=0.02, pool=7)
    Cosine (ORIGINAL)
    SRT-Mahal_RIDGE (NEW)
    RESULT Cosine (ORIGINAL)              macro= 71.43%  [100.0%  42.9%]
    RESULT SRT-Mahal_RIDGE (NEW)        macro= 85.71%  [100.0%  71.4%]

  ...

==========================================================================
  SMoLoRA IF — Final Routing Accuracy (7 tasks, cuda)
==========================================================================
  Method                              Final      Avg  T1     T2     T3  ...
  ------------------------------------------------------------------------
  Cosine (ORIGINAL)                   71.43%   79.59%  100.0%  71.4% ...
  SRT-Mahal_RIDGE (NEW)              85.71%   89.47%  100.0%  85.7% ...
  Delta (SRT - Original)             +14.29%   +9.88%  +0.0%  +14.3% ...
==========================================================================
```

### 7.3 Interpretation

| Symbol | Ý nghĩa |
|--------|---------|
| `macro=X%` | Routing accuracy trung bình trên tất cả task đã thấy |
| `T1...TN` | Per-task accuracy breakdown |
| `ORIGINAL` | Method routing gốc (Cosine) |
| `NEW` | SRT Mahalanobis routing |
| `Delta` | Cải thiện = SRT - ORIGINAL |
| `(n=X, n/d=Y)` | Sample size và ratio (n/d < 0.1 → shrinkage quan trọng) |

**Khi nào SRT tốt hơn Cosine?**
- Khi `n/d` ratio nhỏ (ít sample per task, high-dimensional embeddings)
- Khi inter-task overlap cao
- Khi tasks gần nhau trong embedding space

---

## 8. GPU Requirements

| Experiment | VRAM tối thiểu | Model loading |
|---|---|---|
| SMoLoRA IF (Option A) | ~1 GB | Không cần model |
| SMoLoRA VU (Option A) | ~4 GB | CLIP ViT-L/14-336 |
| SMoLoRA Dual (Option A) | ~4 GB | CLIP ViT-L/14-336 |
| HiDe-LLaVA (Option A) | ~4 GB | CLIP ViT-L/14-336 |
| SMoLoRA IF (Option B) | ~16 GB | Vicuna-7B + adapters |
| SMoLoRA VU (Option B) | ~16 GB | Vicuna-7B + adapters |
| HiDe-LLaVA (Option B) | ~16 GB | Vicuna-7B + HiDe adapters |

**Recommended GPU**: NVIDIA A100 (40/80GB) hoặc RTX 3090/4090 (24GB)
**Minimum**: NVIDIA T4 (16GB) — đủ cho Option A

---

## Tóm tắt lệnh chạy nhanh

```bash
cd expand_method/hypothesis_testing

# ── Option A: Routing Accuracy (không cần trained checkpoint) ──

# 1. SMoLoRA IF Router (nhanh nhất)
python experiments/smolora/if_router/routing_accuracy.py \
    --ins_emb path/to/ins_emb_single.pkl \
    --task_names ScienceQA TextVQA GQA VQAv2 \
    --shrinkage ridge --device cuda

# 2. SMoLoRA VU Router (cần ảnh)
python experiments/smolora/vu_router/routing_accuracy.py \
    --data_root path/to/images \
    --task_names ScienceQA TextVQA GQA VQAv2 \
    --clip_model openai/clip-vit-large-patch14-336 \
    --shrinkage ridge --device cuda

# 3. SMoLoRA Dual Router
python experiments/smolora/dual_router/routing_accuracy.py \
    --ins_emb path/to/ins_emb_single.pkl \
    --data_root path/to/images \
    --task_names ScienceQA TextVQA GQA VQAv2 \
    --clip_model openai/clip-vit-large-patch14-336 \
    --alpha 0.5 --device cuda

# 4. HiDe-LLaVA Cosine → SRT
python experiments/hide/cosine_router/routing_accuracy.py \
    --data_root path/to/images \
    --task_names ScienceQA TextVQA GQA VQAv2 VizWiz TextCaps \
    --clip_model openai/clip-vit-large-patch14-336 \
    --device cuda

# ── Option B: End-to-End (cần trained checkpoint) ──

# SMoLoRA IF (all routing modes)
python experiments/smolora/if_router/end_to_end.py \
    --model_path path/to/checkpoint \
    --model_base path/to/vicuna-7b \
    --ins_emb path/to/ins_emb_single.pkl \
    --task_order ScienceQA TextVQA GQA VQAv2 \
    --routing_mode all --scoring_func vqav2

# SMoLoRA VU (all routing modes)
python experiments/smolora/vu_router/end_to_end.py \
    --model_path path/to/checkpoint \
    --model_base path/to/vicuna-7b \
    --clip_model openai/clip-vit-large-patch14-336 \
    --task_images_root path/to/images \
    --task_order ScienceQA TextVQA GQA VQAv2 \
    --routing_mode all

# HiDe-LLaVA (all routing modes)
python experiments/hide/cosine_router/end_to_end.py \
    --model_path path/to/hide/checkpoint \
    --model_base path/to/vicuna-7b \
    --clip_model openai/clip-vit-large-patch14-336 \
    --task_images_root path/to/images \
    --task_order ScienceQA TextVQA GQA VQAv2 VizWiz TextCaps \
    --routing_mode all
```