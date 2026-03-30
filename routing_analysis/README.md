# routing_analysis

Phân tích routing trong feature space cho bài toán Continual Learning.

Mục tiêu: khai thác thực tế rằng backbone (T5/LLaMA) **frozen hoàn toàn** → embedding distribution của mỗi task là **cố định**, không phụ thuộc vào LoRA training. Do đó toàn bộ bài toán routing có thể được nghiên cứu độc lập trên raw embeddings, trước khi chạm vào bất kỳ phần training nào.

---

## Cấu trúc

```
routing_analysis/
├── extract_embeddings_t5.py      # trích embedding từ T5 family
├── extract_embeddings_llama.py   # trích embedding từ LLaMA family
├── requirements.txt
├── README.md
└── embeddings/                   # output (tự sinh, không commit)
    └── {model_name}/
        └── {benchmark}/
            └── {task_name}/
                ├── train.npz
                ├── dev.npz
                └── test.npz
```

Mỗi `.npz` chứa:
- `embeddings`: `(N, d_model)` **float32** — avg-pooled encoder hidden state (T5) hoặc last/avg token (LLaMA)
  - T5: full float32 precision (no quantization)
  - LLaMA: bfloat16 native → cast to float32 for consistency
- `labels`: `(N,)` object array — output label strings

---

## Benchmarks & Task Orders

Scripts trích embeddings từ **toàn bộ tập (train + dev + test)** không phân biệt order — embedding cố định bất kể task order nào.

### Hai Benchmarks Chính

| Benchmark | #Tasks | Task Names |
|-----------|--------|-----------|
| `Long_Sequence` | 15 | yelp, amazon, mnli, cb, copa, qqp, rte, imdb, sst2, dbpedia, agnews, yahoo, multirc, boolq, wic |
| `SuperNI` | 15 | task1687_sentiment140_classification, task363_sst2_polarity_classification, task875_emotion_classification, task073_commonsenseqa_answer_generation, task591_sciq_answer_generation, task002_quoref_answer_generation, task1290_xsum_summarization, task1572_samsum_summary, task511_reddit_tifu_long_text_summarization, task181_outcome_extraction, task748_glucose_reverse_cause_event_detection, task1510_evalution_relation_extraction, task639_multi_woz_user_utterance_generation, task1590_diplomacy_text_generation, task1729_personachat_generate_next |

### Task Orders (for later incremental studies)

Khi tôi phát triển **incremental routing** (simulation streaming CL), sẽ cần cụ thể hóa task order:

- **Long Order 3**: yelp → amazon → mnli → cb → copa → qqp → rte → imdb → sst2 → dbpedia → agnews → yahoo → multirc → boolq → wic
- **Long Order 4**: (khác)
- **SuperNI Order 1**: (15 tasks SuperNI, tuần tự)
- **SuperNI Order 2**: (khác)

Các order này được định nghĩa trong codebase: `../configs/gen_script_long_order{3,4}_t5_configs/`, `../configs/gen_script_superni_order{1,2}_*/`

**Lưu ý quan trọng**: 
- **Hiện tại (giai đoạn phân tích embedding)**: order **KHÔNG cần xác định** — embedding của mỗi task là cố định (backbone frozen). Task order là **rời rạc** ở phase này.
- **Khi phát triển incremental routing**: order sẽ **RẤT QUAN TRỌNG** — cần simulate: train task 1 → eval trên task 2 → train task 2 → eval 1-3, ... để kiểm tra ảnh hưởng task order lên routing accuracy và forwarding transfer.

---

## Trích Embeddings

### T5 Family

```bash
# flan-t5-small (d=512, 1 GB VRAM) — quick test
python extract_embeddings_t5.py --model google/flan-t5-small

# flan-t5-large (d=1024, 4 GB VRAM)
python extract_embeddings_t5.py --model google/flan-t5-large

# flan-t5-xl (d=2048, 12 GB VRAM)
python extract_embeddings_t5.py --model google/flan-t5-xl
```

### LLaMA Family

```bash
# Llama-2-7b (d=4096, bfloat16 native, 16 GB VRAM)
python extract_embeddings_llama.py --model meta-llama/Llama-2-7b-hf --token YOUR_HF_TOKEN

# Llama-2-7b-chat
python extract_embeddings_llama.py --model meta-llama/Llama-2-7b-chat-hf --token YOUR_HF_TOKEN

# Llama-3.1-8B (d=4096, 20 GB VRAM)
python extract_embeddings_llama.py --model meta-llama/Llama-3.1-8B --token YOUR_HF_TOKEN

# Llama-2-13b (d=5120, 32 GB VRAM) — requires >= H100 or multi-GPU
python extract_embeddings_llama.py --model meta-llama/Llama-2-13b-hf --token YOUR_HF_TOKEN

# Alternative pooling (last token default, --pool avg for mean)
python extract_embeddings_llama.py --model meta-llama/Llama-2-7b-hf --pool avg --token YOUR_HF_TOKEN
```

### Running on TPU

If you have a TPU runtime (Kaggle: enable Accelerator -> TPU v3-8), use the TPU-ready script:

```bash
# Example (single-XLA-device mode)
python extract_embeddings_llama_tpu.py --model meta-llama/Llama-2-7b-hf --device tpu --token YOUR_HF_TOKEN --batch_size 64
```

- Notes: the script will use `torch-xla` if available and place the model on `xm.xla_device()`.
- For best throughput increase `--batch_size` (TPUs like larger batches). The script runs single-process on TPU; for full multi-core scaling you can extend it with `xmp.spawn`/distributed dataloaders.


### Options chung

| Flag | Default | Mô tả |
|------|---------|-------|
| `--data_root` | `CL_Benchmark` (local folder) | Đường dẫn tới CL_Benchmark/ |
| `--output_dir` | `embeddings` | Thư mục output |
| `--batch_size` | 32 (T5), 4 (LLaMA) | Batch size |
| `--max_length` | 512 | Max token length |
| `--benchmarks` | cả hai | `Long_Sequence` và/hoặc `SuperNI` |

---

## Load Embeddings

```python
import numpy as np

data = np.load("embeddings/flan-t5-large/Long_Sequence/yelp/train.npz", allow_pickle=True)
embeddings = data["embeddings"].astype("float32")  # (N, 1024)
labels     = data["labels"]                         # (N,) strings
```

---

---

## GPU Compatibility

| GPU | CUDA Capability | PyTorch Support | Status |
|-----|-----------------|-----------------|--------|
| **Tesla P100** | 6.0 | ✅ PyTorch 1.12 LTS | ⚠️ New PyTorch (2.1+) NOT supported; need **pytorch/pytorch:1.13-cuda11.8**  |
| **Tesla V100** | 7.0 | ✅ PyTorch 2.1+ | ✅ OK |
| **A100** | 8.0 | ✅ PyTorch 2.1+ | ✅ OK |
| **H100** | 9.0 | ✅ PyTorch 2.1+ | ✅ OK (preferred for 13B+) |
| **RTX 3090** | 8.6 | ✅ PyTorch 2.1+ | ✅ OK |
| **L40** | 8.9 | ✅ PyTorch 2.1+ | ✅ OK |

**For P100 users:**
```bash
# Install PyTorch 1.13 (last version supporting sm_60):
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu118

# Then run script normally:
python extract_embeddings_t5.py --model google/flan-t5-small  # Start small
```

---

## Troubleshooting

### Issue: "not found at CL_Benchmark/..."

**Cause**: CL_Benchmark folder is not in the same directory as the script.

**Solution**: Copy CL_Benchmark into routing_analysis/:
```bash
cd routing_analysis
# If CL_Benchmark is in parent:
cp -r ../CL_Benchmark .
# Or symlink:
ln -s ../CL_Benchmark CL_Benchmark
```

After that, scripts will find it automatically with default `--data_root CL_Benchmark`.

### Issue: "CUDA capability sm_60 is not compatible"

Your GPU is P100. Install PyTorch 1.13 or use CPU (very slow):
```bash
python extract_embeddings_t5.py --model google/flan-t5-small --device cpu
```

---

## Hướng phân tích tiếp theo (Phase 2 onwards)

1. **Visualization** — PCA/UMAP plot tất cả tasks, kiểm tra separability
2. **Nearest Centroid routing** — lưu mean vector per task, đo top-1 accuracy
3. **Contrastive subspace** — $D_t = C_t - \gamma \bar{C}_{<t}$, SVD → routing bases
4. **Same-domain pairs** — yelp/amazon/imdb/sst2: phân tích overlap, tìm threshold
5. **Few-shot stability** — bootstrap sample từ CB (250), Copa (400) → variance của stats
6. **Routing accuracy vs. task order** — khi đó, cần simulate incremental streaming theo order cụ thể, đo routing error per task trên each order
