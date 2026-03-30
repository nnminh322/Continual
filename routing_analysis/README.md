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
- `embeddings`: `(N, d_model)` float16 — avg-pooled encoder hidden state (T5) hoặc last/avg token (LLaMA)
- `labels`: `(N,)` object array — output label strings

---

## Benchmarks

| Benchmark | Tasks | Format |
|-----------|-------|--------|
| `Long_Sequence` | 15 tasks (yelp, amazon, mnli, cb, copa, qqp, rte, imdb, sst2, dbpedia, agnews, yahoo, multirc, boolq, wic) | `Definition + input + Output:` |
| `SuperNI` | 15 tasks | `Definition: ... \n\nNow complete... Input: {0}\nOutput:` |

---

## Trích Embeddings

### T5 Family

```bash
# flan-t5-large (d=1024), cần ~3GB VRAM
python extract_embeddings_t5.py --model google/flan-t5-large

# flan-t5-xl (d=2048), cần ~8GB VRAM
python extract_embeddings_t5.py --model google/flan-t5-xl

# flan-t5-small (d=512) — debug nhanh
python extract_embeddings_t5.py --model google/flan-t5-small

# Chỉ 1 benchmark
python extract_embeddings_t5.py --model google/flan-t5-large --benchmarks Long_Sequence
```

### LLaMA Family

```bash
# Llama-2-7b (d=4096), cần ~14GB VRAM fp16
python extract_embeddings_llama.py --model meta-llama/Llama-2-7b-hf --token YOUR_HF_TOKEN

# Llama-2-7b-chat
python extract_embeddings_llama.py --model meta-llama/Llama-2-7b-chat-hf --token YOUR_HF_TOKEN

# Llama-2-13b với 8-bit quantization (cần ~10GB VRAM)
python extract_embeddings_llama.py --model meta-llama/Llama-2-13b-hf --load_in_8bit --token YOUR_HF_TOKEN

# Llama-3.1-8B (d=4096)
python extract_embeddings_llama.py --model meta-llama/Llama-3.1-8B --token YOUR_HF_TOKEN

# Pooling: avg pool thay vì last token
python extract_embeddings_llama.py --model meta-llama/Llama-2-7b-hf --pool avg --token YOUR_HF_TOKEN
```

### Options chung

| Flag | Default | Mô tả |
|------|---------|-------|
| `--data_root` | `../CL_Benchmark` | Đường dẫn tới CL_Benchmark/ |
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

## Hướng phân tích tiếp theo

1. **Visualization** — PCA/UMAP plot tất cả tasks, kiểm tra separability
2. **Nearest Centroid routing** — lưu mean vector per task, đo top-1 accuracy
3. **Contrastive subspace** — $D_t = C_t - \gamma \bar{C}_{<t}$, SVD → routing bases
4. **Same-domain pairs** — yelp/amazon/imdb/sst2: phân tích overlap, tìm threshold
5. **Few-shot stability** — bootstrap sample từ CB (250), Copa (400) → variance của stats
6. **Routing accuracy vs. task order** — simulate CL streaming, đo routing error per task
