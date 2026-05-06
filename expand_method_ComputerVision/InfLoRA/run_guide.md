# SRT_InfLoRA — Run Guide

Guide này dành cho full training thật sự trong `InfLoRA`, không phải nhánh `theory_verify` routing-only.

Trạng thái benchmark hiện tại:

- `cifar100`: benchmark full-training đã được chuẩn hóa trong `InfLoRA`
- `officehome`: benchmark CV chính nên tiếp tục chạy ở nhánh `theory_verify` / `S-Prompts`

## Môi trường

Python 3.10+ | PyTorch 2.x | timm 1.x

### pip thuần

```bash
cd expand_method_ComputerVision/InfLoRA

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm numpy scikit-learn scipy pillow tqdm ipdb pyyaml
```

### conda/mamba

```bash
mamba create -n srt_inflora python=3.12 -y
mamba activate srt_inflora
mamba install pytorch torchvision timm numpy scikit-learn scipy pillow tqdm pyyaml ipdb -c pytorch -c conda-forge
```

## 1. Chạy SRT + SGWI thật sự

### CIFAR-100

```bash
cd expand_method_ComputerVision/InfLoRA

bash run_srt_inflora.sh cifar100 0 0
```

`run_srt_inflora.sh` bây giờ đã thực sự honor `SEED`, nên các lệnh multi-seed dưới đây là hợp lệ:

```bash
bash run_srt_inflora.sh cifar100 0 0
bash run_srt_inflora.sh cifar100 0 1
bash run_srt_inflora.sh cifar100 0 2
bash run_srt_inflora.sh cifar100 0 3
bash run_srt_inflora.sh cifar100 0 4
```

## 2. Chạy baseline để so sánh

### CIFAR-100

```bash
cd expand_method_ComputerVision/InfLoRA

python main.py --config configs/cifar100_inflora.json --device 0 --seed 0
python main.py --config configs/cifar100_infloraca.json --device 0 --seed 0
```

## 3. Kết quả được lưu ở đâu

Checkpoint mỗi task được lưu tại:

```text
logs/{dataset}/{init_cls}_{increment}_{net_type}/{model_name}/{optim}/{rank}/{lamb}_{lame}-{lrate}/{seed}/task_{N}.pth
```

Log file nằm song song dưới dạng `{seed}.log`.

## 4. In bảng kết quả sau khi chạy

Script mới `summarize_logs.py` sẽ parse log và in ra:

- `final_top1`
- `final_with_task`
- `final_task_acc`
- `avg_top1` qua các session
- mean/std nếu có nhiều seed

### Ví dụ: SRT_InfLoRA trên CIFAR-100

```bash
cd expand_method_ComputerVision/InfLoRA

python summarize_logs.py \
    --glob 'logs/cifar100/**/*.log'
```

### Ví dụ: so trực tiếp với baseline hoặc SOTA bạn chọn

```bash
python summarize_logs.py \
    --glob 'logs/cifar100/**/*.log' \
    --reference InfLoRA_domain=PUT_NUMBER_HERE \
    --reference Your_SOTA=PUT_NUMBER_HERE
```

`--reference` so với `final_top1`, tức accuracy cuối cùng sau task cuối.

## 5. Cách đọc log đúng

Đừng chỉ nhìn `final_top1` một mình.

- `final_top1` cho biết performance thật không có task ID
- `final_with_task` cho biết phần classifier/oracle upper bound nếu task biết sẵn
- `final_task_acc` cho biết task predictor hoặc router chọn đúng task được bao nhiêu phần trăm nếu log có ghi metric này

Nếu `final_with_task` cao nhưng `final_top1` thấp, bottleneck là routing chứ không phải classifier.

## 6. Gợi ý so sánh sạch

Nếu mục tiêu là claim SRT + SGWI chạy thật sự và sạch benchmark:

- So trong repo: `SRT_InfLoRA` vs `InfLoRA_domain`
- Với CV benchmark chính: tách riêng `Office-Home` routing-only khỏi full continual training
- Không nên lấy số Office-Home routing-only để đặt ngang hàng với số full continual training

## 7. Office-Home và benchmark khó hơn

Nếu mục tiêu chính hiện tại là `Office-Home`, entrypoint nên là nhánh `theory_verify` / `S-Prompts`, không phải launcher `InfLoRA` trong file này.

Các benchmark khó hơn đang nổi lên trong nhánh DIL/PTM gần đây là:

- `CORe50`: nhiều session hơn, test domain tách khỏi train
- `CDDB-Hard`: khó hơn theo hướng domain shift phi tự nhiên và forgetting mạnh
- `ImageNet-R`, `ImageNet-C`, `ImageNet-Mix`: xuất hiện trong các repo DIL mới hơn như `KA-Prompt`

## 8. Troubleshooting

### `ModuleNotFoundError: No module named 'timm'`

```bash
pip install timm torch torchvision
```

### `CUDA out of memory`

Giảm `batch_size` trong config, hoặc đổi GPU:

```bash
bash run_srt_inflora.sh cifar100 1 0
```

### Office-Home full training chưa có launcher ở đây

Guide này chỉ bao phủ `InfLoRA` full training đã được chuẩn hóa. Với `Office-Home`, hãy dùng pipeline `theory_verify` / `S-Prompts`.

### Muốn chạy đúng seed mà không sửa JSON

```bash
python main.py --config configs/srt_inflora.json --device 0 --seed 2
```
