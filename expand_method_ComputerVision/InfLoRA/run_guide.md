# SRT_InfLoRA — Run Guide

## Môi trường

### Cách 1: pipenv (khuyến nghị)

```bash
cd expand_method_ComputerVision/InfLoRA
pip install pipenv
pipenv install torch==1.10.0 timm==0.6.7 numpy==1.23.5 scikit-learn==1.0 scipy==1.7.1 pillow tqdm ipdb pyyaml
pipenv install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html  # CUDA 11.3
pipenv run python main.py --config configs/srt_inflora.json --device 0
```

### Cách 2: pip thuần

```bash
pip install torch==1.10.0 timm==0.6.7 numpy==1.23.5 scikit-learn==1.0 scipy==1.7.1 pillow tqdm ipdb pyyaml

# Hoặc với CUDA
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install timm==0.6.7 numpy==1.23.5 scikit-learn==1.0 scipy==1.7.1 pillow tqdm ipdb pyyaml
```

> **Lưu ý:** PyTorch 1.10 chỉ hỗ trợ Python ≤ 3.9. Nếu máy bạn dùng Python 3.10+, cần dùng PyTorch 1.13+ (nhưng có thể có breaking changes với một số API cũ).

---

## Benchmark 1: CIFAR-100

### Chuẩn bị dữ liệu

```bash
# Tải CIFAR-100 nếu chưa có (mã nguồn sẽ tự tải qua torchvision)
# Dữ liệu mặc định lưu vào: data/
```

### Chạy

```bash
conda activate python3_8
cd expand_method_ComputerVision/InfLoRA

bash run_srt_inflora.sh 0 0
```

**Kết quả** lưu tại `logs/cifar100/10_10_sip/SRT_InfLoRA/adam/rank_lamb_lame_lrate/{seed}/`

### Config chính

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `init_cls` | 10 | Classes/task đầu tiên |
| `increment` | 10 | Classes thêm mỗi task |
| `total_sessions` | 10 | 10 tasks × 10 classes = 100 classes |
| `rank` | 10 | LoRA rank |
| `sgwi` | true | SGWI warm-init enabled |
| `srt_shrinkage` | ridge | Ridge shrinkage cho SRT |
| `init_lr` | 0.0005 | Learning rate task 0 |
| `lrate` | 0.0005 | Learning rate task > 0 |
| `epochs` | 20 | Epochs mỗi task |
| `optim` | adam | Optimizer |
| `batch_size` | 128 | Batch size |

### Điều chỉnh seed

```bash
# Chạy nhiều seeds (khuyến nghị: 0,1,2,3,4)
bash run_srt_inflora.sh 0 0
bash run_srt_inflora.sh 0 1
bash run_srt_inflora.sh 0 2
bash run_srt_inflora.sh 0 3
bash run_srt_inflora.sh 0 4
```

### So sánh với InfLoRA baseline

```bash
# Baseline InfLoRA
python main.py --config configs/cifar100_inflora.json --device 0
```

---

## Benchmark 2: DomainNet

DomainNet là domain-incremental (5 domains × 69 classes = 345 classes).

SRT_InfLoRA hiện tại dùng `eval_task()` cho class-incremental. Để chạy DomainNet, cần thêm method `_compute_accuracy_domain` tương tự `InfLoRA_domain`.

### Tạo config

```json
{
    "prefix": "srt_inflora",
    "dataset": "domainnet",
    "data_path": "data/domainnet",
    "memory_size": 0,
    "memory_per_class": 0,
    "fixed_memory": true,
    "shuffle": false,
    "init_cls": 69,
    "increment": 69,
    "model_name": "SRT_InfLoRA",
    "net_type": "sip",
    "embd_dim": 768,
    "num_heads": 12,
    "total_sessions": 5,
    "seed": [0],
    "EPSILON": 1e-8,
    "init_epoch": 5,
    "optim": "sgd",
    "init_lr": 0.01,
    "init_lr_decay": 0.1,
    "init_weight_decay": 0.0,
    "epochs": 5,
    "fc_lrate": 0.01,
    "lrate": 0.005,
    "lrate_decay": 0.1,
    "batch_size": 128,
    "weight_decay": 0.0,
    "rank": 30,
    "lamb": 0.95,
    "lame": 1.0,
    "num_workers": 16,
    "sgwi": true,
    "srt_shrinkage": "ridge"
}
```

Lưu vào `configs/domainnet_srt_inflora.json`.

### Chạy

```bash
python main.py --config configs/domainnet_srt_inflora.json --device 0
```

---

## Baseline comparison

### CIFAR-100

```bash
# InfLoRA (original baseline)
python main.py --config configs/cifar100_inflora.json --device 0

# InfLoRA + CA (compact augmented)
python main.py --config configs/cifar100_infloraca.json --device 0
```

### DomainNet

```bash
# InfLoRA_domain (original baseline)
python main.py --config configs/domainnet_inflora.json --device 0
```

---

## Tổng hợp kết quả

Checkpoint mỗi task được lưu tại:
```
logs/{dataset}/{init_cls}_{increment}_{net_type}/{model_name}/{optim}/{rank}_{lamb}_{lame}_{lrate}/{seed}/task_{N}.pth
```

Log file: `.log` cùng thư mục với checkpoint.

---

## Troubleshooting

### Lỗi thường gặp

**1. `ModuleNotFoundError: No module named 'timm'`**
```bash
# Kiểm tra đã cài đủ packages
pip list | grep -E "torch|timm|numpy|sklearn"

# Nếu thiếu, cài lại
pip install timm==0.6.7 torch==1.10.0
```

**2. `CUDA out of memory`**
```bash
# Giảm batch_size trong config:
# "batch_size": 64

# Hoặc chạy trên GPU khác
bash run_srt_inflora.sh cifar100 1 0
```

**3. Data not found**
```bash
# Dataset được tải tự động bởi torchvision.
# Kiểm tra:
ls data/
```

**4. Lỗi pretrained weight**
```bash
# timm sẽ tự tải ViT weights từ internet.
# Cache tại: ~/.cache/huggingface/ hoặc ~/.cache/torch/
```

**5. Python version quá cao**
```
# Nếu gặp lỗi với PyTorch 1.10 trên Python > 3.9:
pip install torch --upgrade

# Hoặc tạo virtualenv với Python 3.8
python3.8 -m venv venv38
source venv38/bin/activate
pip install torch==1.10.0 timm==0.6.7 ...
```

---

## Files mới được thêm vào

```
InfLoRA/
├── models/sinet_srt_inflora.py      # SRT_Router + SiNet_SRT model
├── methods/srt_inflora.py           # SRT_InfLoRA learner
├── configs/srt_inflora.json         # CIFAR-100 config
├── configs/domainnet_srt_inflora.json  # DomainNet config
├── Pipfile                         # pipenv dependencies
├── run_srt_inflora.sh             # Bash script chạy nhanh
└── run_guide.md                   # File này
```