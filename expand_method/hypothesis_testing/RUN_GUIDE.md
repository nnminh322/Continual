# SRT Hypothesis Testing — GPU Setup & Run Guide

## Mục lục
1. [Setup](#1-setup--environment)
2. [Tổng quan kiểm định](#2-tổng-quan-kiểm-định)
3. [Option A: Routing Accuracy](#3-option-a-routing-accuracy-routing-accuracy-trên-embeddings-thật)
4. [Option B: End-to-End](#4-option-b-end-to-end-cần-trained-checkpoint)
5. [Data paths chuẩn](#5-data-paths-chuẩn-từ-2-repo-gốc)
6. [Scoring metrics đúng](#6-scoring-metrics-đúng-từ-repo-gốc)
7. [Interpret kết quả](#7-interpret-kết-quả)

---

## 1. Setup & Environment

```bash
# Tạo conda environment
conda create -n srt_exp python=3.12 -y
conda activate srt_exp

# Cài PyTorch CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Cài package
cd hypothesis_testing
pip install -e .

# Verify GPU
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 2. Tổng quan kiểm định

### Data sources (từ 2 repo gốc)

| Experiment | Data | Baseline | SRT Method |
|-----------|------|----------|------------|
| **SMoLoRA IF Router** | `ins_emb_single.pkl` (384-dim, Sentence-BERT on real instructions) | Cosine similarity | SRT-Mahal Ridge/LW |
| **SMoLoRA VU Router** | CLIP ViT-L/14 image features (1024-dim) on real images | Cosine similarity | SRT-Mahal Ridge/LW |
| **SMoLoRA Dual Router** | CLIP image features + ins_emb | Cosine VU + Cosine IF | SRT-VU + SRT-IF |
| **HiDe-LLaVA Cosine→SRT** | CLIP ViT-L/14 image features | Cosine (original HiDe) | SRT-Mahal |

### Baseline routing comparison

| Component | Original Method (trong repo gốc) | Baseline trong kiểm định |
|-----------|----------------------------------|--------------------------|
| SMoLoRA IF gate | `nn.Linear(384, 4)` → argmax | Cosine on ins_emb |
| SMoLoRA VU gate | `nn.Linear(hidden_dim, 4)` → argmax | Cosine on CLIP features |
| HiDe-LLaVA router | Cosine similarity | Cosine (ORIGINAL) |

> **Note**: IF router dùng Cosine baseline thay vì learned linear vì Option A không có training. Cosine là upper bound cho learned linear (nếu cosine không phân biệt được, learned linear cũng khó). SRT Mahalanobis test cùng feature space → kết quả có ý nghĩa.

### Scoring metrics (y chang SMoLoRA/HiDe-LLaVA gốc)

| Dataset | Metric | Source |
|---------|--------|--------|
| **VQAv2** | Exact match, case-insensitive | SMoLoRA `eval_vqav2.py` |
| **TextVQA** | 10-way soft matching (EvalAIProcessor) | SMoLoRA `eval_textvqa.py` |
| **ScienceQA** | Letter extraction (A/B/C/D/E), exact match | SMoLoRA `eval_science_qa.py` |
| **GQA** | Balanced accuracy, case-sensitive | SMoLoRA `eval_gqa.py` |
| **Flickr30k** | CIDEr score | SMoLoRA `eval_flickr30k_cider.py` |
| **ImageNet** | Substring match, case-insensitive | SMoLoRA `eval_ImagetNet.py` |
| **Place365** | Substring match, case-insensitive | SMoLoRA `eval_ImagetNet.py` |

---

## 3. Option A: Routing Accuracy (routing accuracy trên embeddings thật)

**Không cần trained checkpoint.** Test trên instruction embeddings và image embeddings thật từ dataset.

### 3.1 Generate ins_emb_single.pkl (1 lần đầu)

```bash
cd hypothesis_testing

python scripts/generate_ins_emb.py \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --output ins_emb_single.pkl
```

File này chứa 7×384 embeddings cho 7 instruction strings từ SMoLoRA.

### 3.2 SMoLoRA IF Router

Test SRT Mahalanobis vs Cosine trên **instruction embeddings thật**.

```bash
python experiments/smolora/if_router/routing_accuracy.py \
    --ins_emb ins_emb_single.pkl \
    --task_order ScienceQA TextVQA GQA VQAv2 Flickr30k ImageNet Place365 \
    --shrinkage ridge \
    --device cuda \
    --n_samples 200 \
    --noise_std 0.1
```

- ins_emb: real 384-dim embeddings từ sentence-transformers (đúng SMoLoRA)
- Baseline: Cosine similarity (proxy cho original IF gate)
- Test: SRT Mahalanobis

### 3.3 SMoLoRA VU Router (cần ảnh thật)

Test SRT Mahalanobis vs Cosine trên **CLIP image features thật**.

```bash
python experiments/smolora/vu_router/routing_accuracy.py \
    --data_root /data/zqwang/moe_cl_data/dataset \
    --task_names ScienceQA TextVQA GQA VQAv2 \
    --clip_model openai/clip-vit-large-patch14-336 \
    --shrinkage ridge \
    --device cuda \
    --n_train 500 \
    --n_test 200
```

- Features: CLIP ViT-L/14-336 image embeddings (1024-dim)
- Baseline: Cosine similarity
- Test: SRT Mahalanobis

### 3.4 SMoLoRA Dual Router (VU + IF)

```bash
python experiments/smolora/dual_router/routing_accuracy.py \
    --ins_emb ins_emb_single.pkl \
    --data_root /data/zqwang/moe_cl_data/dataset \
    --task_names ScienceQA TextVQA GQA VQAv2 \
    --clip_model openai/clip-vit-large-patch14-336 \
    --alpha 0.5 \
    --shrinkage ridge \
    --device cuda
```

### 3.5 HiDe-LLaVA Cosine → SRT

```bash
python experiments/hide/cosine_router/routing_accuracy.py \
    --data_root /data/zqwang/moe_cl_data/dataset \
    --task_names ScienceQA TextVQA GQA VQAv2 VizWiz TextCaps \
    --clip_model openai/clip-vit-large-patch14-336 \
    --shrinkage ridge \
    --device cuda
```

---

## 4. Option B: End-to-End (cần trained checkpoint)

**Cần trained SMoLoRA hoặc HiDe-LLaVA checkpoint.**

Workflow:
1. Train bằng code gốc SMoLoRA/HiDe-LLaVA → get checkpoint
2. Load checkpoint vào framework
3. Inject SRT routing (thay routing decision)
4. Run VQA generation trên test set
5. Score với ground truth bằng metrics y chang repo gốc

### 4.1 SMoLoRA IF (End-to-End)

```bash
python experiments/smolora/if_router/end_to_end.py \
    --model_path /path/to/smolora/checkpoint \
    --model_base /path/to/vicuna-7b \
    --ins_emb ins_emb_single.pkl \
    --task_order ScienceQA TextVQA GQA VQAv2 \
    --routing_mode all \
    --scoring_func vqav2 \
    --device cuda \
    --output_dir results_smolora_if_e2e
```

### 4.2 SMoLoRA VU (End-to-End)

```bash
python experiments/smolora/vu_router/end_to_end.py \
    --model_path /path/to/smolora/checkpoint \
    --model_base /path/to/vicuna-7b \
    --clip_model openai/clip-vit-large-patch14-336 \
    --task_images_root /data/zqwang/moe_cl_data/dataset \
    --task_order ScienceQA TextVQA GQA VQAv2 \
    --routing_mode all \
    --device cuda
```

---

## 5. Data Paths chuẩn từ 2 repo gốc

### SMoLoRA (server: `/data/zqwang/moe_cl_data/`)

```bash
# Images (shared across all datasets)
export SMOLORA_DATA=/data/zqwang/moe_cl_data/dataset

# ins_emb_single.pkl (tạo bằng scripts/generate_ins_emb.py)
# Image subdirs trong $SMOLORA_DATA:
ScienceQA/      # ảnh science
TextVQA/         # ảnh textVQA
GQA/             # ảnh GQA
images/          # COCO val2014 (VQAv2)
flickr30k_images/ # Flickr30k

# Question/annotation JSONs
/data/zqwang/moe_cl_data/CVIT_benchmark/Instructions_Single/
├── ScienceQA/test.json
├── VQAv2/val.json
├── GQA/test.json
├── TextVQA/val.json
└── Flickr30k/val.json

# Ground truth cho ScienceQA
/data/zqwang/moe_cl_data/dataset/ScienceQA/
├── pid_splits.json
└── problems.json
```

### HiDe-LLaVA (placeholder: `/your_path/`)

```bash
export HIDE_DATA=/your_path/datasets

# Question files
/your_path/
├── ScienceQA/test.json
├── VQAv2/test.json
├── GQA/test.json
├── TextVQA/val.json
├── VizWiz/test.json
├── TextCaps/test.json
└── Grounding/test.json
```

---

## 6. Scoring Metrics đúng từ repo gốc

### VQAv2 — Exact match
```python
# scoring/vqav2.py — verbatim from SMoLoRA eval_vqav2.py
if pred.upper() == ground_truth.upper():
    correct += 1
accuracy = correct / total * 100
```

### TextVQA — 10-way soft matching
```python
# scoring/textvqa.py — verbatim from SMoLoRA eval_textvqa.py
# EvalAIAnswerProcessor: normalize text (lowercase, strip, expand contractions)
# For each unique processed answer among 10 reference answers:
#   score = min(1, #matching_other_annotators / 3)
# Final: mean of all prediction scores
```

### ScienceQA — Letter extraction
```python
# scoring/science_qa.py — verbatim from SMoLoRA eval_science_qa.py
# 1. If text IS option letter (A-E) → use it
# 2. If text starts with "X. " → extract X
# 3. Else regex \b(\w)\b → first match uppercased
# 4. Else FAILED
```

### GQA — Balanced accuracy
```python
# scoring/gqa.py — verbatim from SMoLoRA eval_gqa.py
# Only isBalanced=True questions
# case-sensitive string equality
# accuracy = sum(correct) / len(questions) * 100
```

---

## 7. Interpret kết quả

### Khi nào SRT tốt hơn baseline?

| Signal | Ý nghĩa |
|--------|---------|
| `delta > 0` | SRT routing accuracy cao hơn baseline |
| `n/d < 0.1` | High-dimensional regime → shrinkage quan trọng → SRT có lợi thế lớn |
| `delta tăng theo số task` | Tasks càng nhiều, feature overlap càng cao → SRT phân biệt tốt hơn |

### Từ Option A đến thực tế

> **Nếu SRT routing accuracy > Cosine routing accuracy trong Option A,
> thì thực tế apply SRT routing cũng sẽ tốt hơn baseline.**

**Lý do:**
1. IF Router: Cả SRT và Cosine dùng **cùng feature space** (ins_emb 384-dim). Nếu SRT Mahalanobis phân biệt task tốt hơn trong cùng space, thì khi apply vào SMoLoRA (thay cosine bằng SRT), routing sẽ tốt hơn → chọn đúng expert → VQA accuracy cao hơn.

2. VU Router: CLIP features test trên image space. Nếu SRT vượt Cosine, nó capture structure tốt hơn trong feature space đó. Apply vào LLM hidden space sẽ cần Option B để confirm.

3. HiDe-LLaVA: Direct replacement — HiDe gốc dùng cosine, SRT thay cosine → routing accuracy cao hơn → VQA accuracy cao hơn.

### Next steps sau khi có checkpoint

1. Chạy Option B end-to-end với trained checkpoint
2. So sánh VQA accuracy: SRT routing vs original routing vs oracle
3. Nếu Option B confirm: integrate SRT routing vào SMoLoRA/HiDe-LLaVA training loop

---

## Tóm tắt lệnh nhanh

```bash
cd hypothesis_testing

# Generate ins_emb (1 lần)
python scripts/generate_ins_emb.py --output ins_emb_single.pkl

# ── Option A: Routing Accuracy ──

# 1. IF Router (nhanh nhất, không cần ảnh)
python experiments/smolora/if_router/routing_accuracy.py \
    --ins_emb ins_emb_single.pkl \
    --task_order ScienceQA TextVQA GQA VQAv2 \
    --shrinkage ridge --device cpu

# 2. VU Router (cần ảnh)
python experiments/smolora/vu_router/routing_accuracy.py \
    --data_root /data/zqwang/moe_cl_data/dataset \
    --task_names ScienceQA TextVQA GQA VQAv2 \
    --clip_model openai/clip-vit-large-patch14-336 \
    --shrinkage ridge --device cuda

# 3. Dual Router
python experiments/smolora/dual_router/routing_accuracy.py \
    --ins_emb ins_emb_single.pkl \
    --data_root /data/zqwang/moe_cl_data/dataset \
    --task_names ScienceQA TextVQA GQA VQAv2 \
    --clip_model openai/clip-vit-large-patch14-336 \
    --alpha 0.5 --device cuda

# 4. HiDe-LLaVA
python experiments/hide/cosine_router/routing_accuracy.py \
    --data_root /data/zqwang/moe_cl_data/dataset \
    --task_names ScienceQA TextVQA GQA VQAv2 VizWiz TextCaps \
    --clip_model openai/clip-vit-large-patch14-336 \
    --device cuda

# ── Option B: End-to-End (cần checkpoint) ──
python experiments/smolora/if_router/end_to_end.py \
    --model_path /path/to/checkpoint \
    --model_base /path/to/vicuna-7b \
    --ins_emb ins_emb_single.pkl \
    --task_order ScienceQA TextVQA GQA VQAv2 \
    --routing_mode all --scoring_func vqav2
```
