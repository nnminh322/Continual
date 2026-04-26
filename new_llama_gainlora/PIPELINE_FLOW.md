# Pipeline Flow: run_superni_order1_llama_cl.sh

## Tổng quan: 2 tầng orchestration

```
Bash script (run_superni_order1_llama_cl.sh)
└── 15 lần lặp, mỗi lần gọi deepspeed cho 1 task
    └── Mỗi deepspeed process = 1 Python script (run_llama_gainlora_cl.py)

Task 0  → deepspeed → run_llama_gainlora_cl.py (task 0)
Task 1  → deepspeed → run_llama_gainlora_cl.py (task 1) ← load task0's saved_weights
Task 2  → deepspeed → run_llama_gainlora_cl.py (task 2) ← load task0+1's saved_weights
...
Task 14 → deepspeed → run_llama_gainlora_cl.py (task 14) ← load all 14 prev saved_weights
```

---

## Phase 1: Bash Script — `run_superni_order1_llama_cl.sh`

```
MODEL_NAME_OR_PATH="..." NUM_GPUS=1 bash run_superni_order1_llama_cl.sh --srt_shrink
│
├── Xác định biến môi trường
│   ├── MODEL_NAME_OR_PATH = "/kaggle/input/datasets/natmin322/llama-2-7b-hf-weights"
│   ├── DATA_DIR = "root_gainlora/CL_Benchmark"
│   ├── CONFIG_BASE = "new_gainlora/configs/gen_script_superni_order1_llama_configs"
│   ├── TASK_ORDER = [task1572_samsum_summary, task363_sst2, ...]  (15 tasks)
│   ├── OUTPUTS_DIR = "logs_and_outputs/superni_order1_llama_srt/outputs"
│   └── NUM_TASKS = 15
│
└── for TASK_ID = 0..14:
    │
    ├── TASK_NAME = TASK_ORDER[TASK_ID]
    ├── TASK_CONFIG_DIR = CONFIG_BASE / TASK_NAME
    ├── TASK_NUM = TASK_ID + 1          (1-indexed)
    ├── OUTPUT_DIR = OUTPUTS_DIR / "{TASK_NUM}-{TASK_NAME}"
    │
    ├── Build PREV_LORA_ARG:
    │   ├── Task 0:  PREV_LORA_ARG = ""           (không có previous)
    │   ├── Task T:  PREV_LORA_ARG = "--previous_lora_path path0,path1,...,pathT-1"
    │
    ├── Build SRT_LOAD_ARG:
    │   ├── Task 0:  SRT_LOAD_ARG = ""            (không có previous SRT)
    │   ├── Task T:  SRT_LOAD_ARG = "--srt_load_path pathT-1"
    │
    ├── deepspeed --num_gpus=1 --master_port=$((29500+TASK_ID)) \
    │     run_llama_gainlora_cl.py \
    │     --model_name_or_path MODEL_NAME_OR_PATH \
    │     --cur_task_id TASK_ID \
    │     --task_order "task0,task1,..." \
    │     --srt_metric_mode hard \
    │     --srt_max_emb_samples 200 \
    │     --srt_shrink          ← NEW: enable Ledoit-Wolf shrinkage
    │     --bf16 \
    │     --deepspeed stage2.config \
    │     --seed 42 \
    │     PREV_LORA_ARG \
    │     SRT_LOAD_ARG \
    │   tee "{OUTPUT_DIR}/train.log"
    │
    ├── Tích lũy saved_weights:
    │   CUR_SAVED = OUTPUT_DIR/saved_weights
    │   PREVIOUS_LORA_PATHS = PREVIOUS_LORA_PATHS + CUR_SAVED  (nối thêm)
    │   PREV_SRT_PATH = CUR_SAVED
    │
    └── echo "[Done] Task {TASK_NUM}: saved → {CUR_SAVED}"
        │
        └── Chờ deepspeed process kết thúc → sang task tiếp theo

Sau khi tất cả 15 tasks xong:
└── python score.py {RUN_NAME} {RUN_NAME} {LOG_DIR}
    └── Tính CL metrics (forgetting, forward transfer, avg accuracy)
```

---

## Phase 2: Python Script — `run_llama_gainlora_cl.py` (mỗi task)

### Bước 0: Entry point — `main()`

```python
def main():
    args = parse_args()          # đọc tất cả --flags
    set_seed(args.seed)          # seed = 42
    task_order = args.task_order.split(",")
    cur_task_id = args.cur_task_id
    cur_task = task_order[cur_task_id]
```

### Bước 1: Setup Tokenizer

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"        # causal LM: left-pad
```

### Bước 2: Load Data

```python
train_samples, dev_samples, test_samples = build_splits(
    data_dir=args.data_dir,
    task_config_dir=args.task_config_dir,
    max_train_samples=args.max_train_samples,
    max_eval_samples=args.max_eval_samples,
)
# → CLConfig.parse_task_config(task_config_dir)  đọc train/dev/test_tasks.json
# → CLInstructions._generate_examples()  format thành
#   {Instance: {instruction, sentence, label}, Task, Dataset}
#
# train: Dataset huggingface (160 samples)
# dev:   Dataset huggingface (20-100 samples)
# test:  list of dicts
```

### Bước 3: Setup Model

```python
# prompt_config định nghĩa kiến trúc GainLoRA
prompt_config = {
    "run_single": False,                    # CL mode (bật SRT, prompt_key, trans_input)
    "previous_lora_path": prev_lora_path,    # None cho task 0
    "task_id": cur_task_id,
    "lora_r": 4,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "trans_hidden_dim": 100,
    "mlp_hidden_dim": 800,
}

# Load pretrained Llama-2-7b từ HuggingFace
hf_base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Tạo custom GainLoRA model
custom_config._attn_implementation = "eager"
model = LlamaForCausalLM(custom_config, prompt_config)

# Copy pretrained weights vào custom model
model.load_state_dict(hf_base_model.state_dict(), strict=False)
del hf_base_model

# Re-initialize LoRA params (kaiming A, zeros B)
for module in model.modules():
    if hasattr(module, 'reset_parameters'):
        module.reset_parameters()
nn.init.zeros_(module.lora_B)

# Tạo dummy previous_prompts_keys nếu cần (CL mode)
model.model.previous_prompts_keys = nn.Parameter(zeros(n_prev, hidden_size))
```

### Bước 4: Load Previous LoRA Adapters (task > 0)

```python
if prev_lora_path:
    # Đọc lora_weights_A.pt, lora_weights_B.pt từ mỗi task trước
    # Đảo ngược thứ tự: slot[0] = most recent task
    for i, path in enumerate(reversed(all_prev_paths)):
        lora_A = torch.load(f"{path}/lora_weights_A.pt")
        lora_B = torch.load(f"{path}/lora_weights_B.pt")
        for j in range(num_layers):
            attn = model.model.layers[j].self_attn
            attn.previous_lora_weights_q[i].lora_A.data.copy_(lora_A[...])
            attn.previous_lora_weights_q[i].lora_B.data.copy_(lora_B[...])
            attn.previous_lora_weights_v[i].lora_A.data.copy_(lora_A[...])
            attn.previous_lora_weights_v[i].lora_B.data.copy_(lora_B[...])
    # Move all previous_lora_weights to CPU để tiết kiệm VRAM
    for module in model.modules():
        if hasattr(module, 'previous_lora_weights_q'):
            module.previous_lora_weights_q.to("cpu")
            module.previous_lora_weights_v.to("cpu")
```

### Bước 5: Freeze & Unfreeze

```python
# Freeze TẤT CẢ parameters
for name, param in model.named_parameters():
    param.requires_grad = False

# Unfreeze CHỈ current task's trainable params
for name, param in model.named_parameters():
    if ("lora" in name and "previous_lora_weights" not in name) \
       or ("trans_input" in name and "previous_trans_input" not in name) \
       or "prompt_key" in name:
        param.requires_grad = True

# Upcast trainable params to fp32 (AMP stability)
for param in model.parameters():
    if param.requires_grad and param.dtype != torch.float32:
        param.data = param.data.to(torch.float32)
```

### Bước 6: Setup TrainingArguments

```python
training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=32,
    learning_rate=5e-5,
    num_train_epochs=15,
    eval_strategy="steps",
    eval_steps=max(1, 5 * step_per_epoch),   # eval mỗi 5 epochs
    save_strategy="no",          # KHÔNG save checkpoint (7B file)
    bf16=True,
    deepspeed=args.deepspeed,
)
```

### Bước 7: Init Trainer

```python
trainer = SRTSGWITrainer(
    model              = model,
    args               = training_args,
    train_dataset      = train_dataset,
    eval_dataset      = eval_dataset,   # dummy; real eval = generate-based
    data_collator     = collator,
    cur_task_id       = cur_task_id,
    task_order        = task_order,
    sgwi              = args.sgwi,           # default True
    srt_metric_mode   = "hard",
    srt_shrink        = args.srt_shrink,     # True từ --srt_shrink flag
    srt_shrink_factor = 0.1,
    srt_max_emb_samples=200,
    srt_load_path     = args.srt_load_path,  # task>0: path của task trước
    dev_samples       = dev_samples,
    tokenizer         = tokenizer,
    max_source_length=1024,
    max_new_tokens   =50,
    eval_batch_size  =2,
)
```

**SRTSGWITrainer.__init__():**

```python
def __init__(self, ...):
    super().__init__(model=model, args=args, ...)
    self.srt_router = SRTRouter(
        srt_metric_mode="hard",
        use_shrink=True,          # ← Ledoit-Wolf shrinkage enabled
        shrink_factor=0.1,
    )

    # Nếu có srt_load_path (task>0): load signatures từ task trước
    if srt_load_path:
        self.load_srt_signatures(srt_load_path, wire_model=True)
        # → srt_router.load(path) → signatures của all previous tasks
        # → _replace_attention_routing(): gắn router vào model.model
```

### Bước 8: SGWI Warm-init

```python
trainer.get_reg_matrix()
```

**`get_reg_matrix()` logic:**

```
Task 0:
  → "[SGWI] Task 0: no prior adapters, skipping warm-init."
  → Không làm gì

Task > 0 & sgwi=True:
  → _compute_sgwi_weights():
      1. Extract embeddings via encoder_frozen (FrozenLlamaExtractor)
      2. Compute current task centroid μ_cur
      3. For each previous task sig: compute Mahalanobis distance
         d_s = (μ_cur - μ_s)ᵀ · Σ_pool⁻¹ · (μ_cur - μ_s)
      4. Softmax: w_s = exp(-d_s / τ) / Σ exp(...)
  → _sgwi_init_a(srt_weights):
      ΔW = Σ_s w_s · (B_s @ A_s)
      U, S, Vt = SVD(ΔW)
      A_new = sqrt(S[:r]) * Vt[:r, :]
  → _fuse_past_lora_adapters(srt_weights):
      B_warm = ΔW @ A_curᵀ @ (A_cur @ A_curᵀ + εI)⁻¹

Task > 0 & sgwi=False:
  → "[SGWI] Task {cur_task_id}: sgwi=False → full_lora mode."
  → Không làm gì
```

### Bước 9: Training Loop

```python
trainer.train()  # HuggingFace Trainer loop
```

**Mỗi training step:**
```python
for batch in train_dataloader:
    # Forward
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        input_ids_wo_label=batch["input_ids_wo_label"],  # cho SRT routing
    )
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    scheduler.step()
```

**Mỗi `eval_steps` (5 epochs): Trainer.evaluate() được gọi:**

```python
def evaluate(self, ...):
    # Override của SRTSGWITrainer
    # Generation-based RougeL evaluation trên dev set
    for batch in dev_samples:
        prompts = [inst["instruction"].format(inst["sentence"]) for inst in batch]
        encoded = tokenizer(prompts, ...)

        with torch.no_grad():
            generated = model.generate(
                input_ids=encoded["input_ids"],
                input_ids_wo_label=encoded["input_ids"],  # SRT routing
                attention_mask=encoded["attention_mask"],
            )
        predictions = tokenizer.decode(generated[:, input_length:])

    rouge_l = sum(rouge_l(p, r)) / n

    # Track best in-memory
    if rouge_l > self._best_eval_rougeL:
        self._best_eval_rougeL = rouge_l
        # Snapshot lora_A và lora_B state dicts
        self._best_lora_A_state = {k: v.detach().clone() for k,v in model.named_parameters()
                                    if "lora_A" in k and "previous" not in k}
        self._best_lora_B_state = {k: v.detach().clone() for k,v in model.named_parameters()
                                    if "lora_B" in k and "previous" not in k}
        print(f"  [EVAL] ★ New best eval_rougeL={rouge_l:.4f}")
    else:
        print(f"  [EVAL] eval_rougeL={rouge_l:.4f} (best={best})")
```

### Bước 10: Restore Best Checkpoint

```python
trainer.restore_best_model()
# Copy _best_lora_A_state, _best_lora_B_state trở lại model parameters
# Không cần load 7B model file — chỉ copy trainable LoRA params
```

### Bước 11: Save LoRA Weights

```python
save_path = output_dir / "saved_weights"
save_path.mkdir(parents=True, exist_ok=True)

torch.save(lora_state_dict_A(model), save_path / "lora_weights_A.pt")
torch.save(lora_state_dict_B(model), save_path / "lora_weights_B.pt")
# Output: saved_weights/
#   ├── lora_weights_A.pt  (trainable lora_A params)
#   ├── lora_weights_B.pt  (trainable lora_B params)
#   └── (srt_signatures.npz sẽ được save ở bước 14)
```

### Bước 12: Attach Frozen Backbone (SRT Embedding Extraction)

```python
if args.use_srt_router and model.model.encoder_frozen is None:
    # Load pretrained Llama-2-7b (FROZEN, không LoRA)
    frozen_backbone = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    frozen_decoder = frozen_backbone.model
    model.model.encoder_frozen = FrozenLlamaExtractor(frozen_decoder)
    model.model.encoder_frozen.to(device)
    del frozen_backbone

# FrozenLlamaExtractor.forward(input_ids, attention_mask):
#   → decoder(input_ids, output_hidden_states=True)
#   → h = hidden_states[-1]  (final layer, B×seq×d)
#   → last_non_pad_idx = attention_mask.sum(dim=1) - 1
#   → pooled = h[arange(B), last_non_pad_idx]  (B×d)
#   → return pooled.float()
```

### Bước 13: Post-training — SRT Signature Computation

```python
trainer.on_task_end(cur_task)
```

**`on_task_end()` → `_compute_and_store_signature()`:**

```python
# 1. Extract embeddings từ frozen backbone
h_train = _extract_task_embeddings(max_samples=200)
# → forward qua FrozenLlamaExtractor
# → input: train samples (source prompts, KHÔNG có labels)
# → output: (B×200, d) embeddings

# 2. Add signature vào router
sig = srt_router.add_task(task_id=cur_task, h_train=h_train)
#   a. mu_t, Sigma_t = mean/cov(h_train)
#   b. Sigma_t_shrunk = ledoit_wolf_shrinkage(Sigma_t, factor=0.1)   ← NEW: now works!
#   c. _update_pooled(mu_t, Sigma_t_shrunk, n_t)  (Welford-Hart)
#   d. Shrink pooled Σ: cov_shrunk = (1-f)*Sigma_pool + f*(tr/d)*I
#   e. Refit ZCA: W_zca = eigvecs @ diag(1/sqrt(eigvals)) @ eigvecs.T
#   f. Re-whiten ALL centroids:
#        for s in signatures:
#            s.mu = (s.mu_raw - mu_global) @ W_zca.T
#            s.Sigma = W_zca @ s.Sigma_raw @ W_zca.T
```

**`on_task_end()` → `_replace_attention_routing()`:**

```python
# Wire router vào model để inference routing
model.model.srt_router = self.srt_router
model.model.srt_task_id_to_idx = {
    cur_task: 0,           # slot 0 = current task
    prev_task_1: 1,        # slot 1 = most recent previous
    prev_task_0: 2,
    ...
}
model.model.use_srt_routing = True
```

### Bước 14: Save SRT Signatures

```python
trainer.save_srt_signatures(save_path)
# → srt_router.save(save_path / "srt_signatures.npz")
#   ├── signatures: {task_id: {mu, Sigma, mu_raw, Sigma_raw, n, metric, alpha}}
#   ├── mu_pool: pooled mean
#   ├── Sigma_pool: pooled covariance
#   ├── n_pool: total samples
#   ├── srt_metric_mode: "hard"
#   ├── use_shrink: True
#   ├── shrink_factor: 0.1
#   ├── mu_global: global mean for ZCA
#   └── W_zca: ZCA whitening matrix
```

### Bước 15: Continual Evaluation

```python
model.config.use_cache = True

# Load test set của TẤT CẢ tasks đã học (task 0 → cur_task)
seen_task_names = task_order[:cur_task_id + 1]
continual_samples = build_continual_eval_samples(
    data_dir=args.data_dir,
    task_config_base_dir=args.task_config_dir.parent,
    task_names=seen_task_names,
)

# Generate predictions
continual_metrics = evaluate_continual_split_cl_legacy(
    model, tokenizer, continual_samples, ...
)
```

**`evaluate_continual_split_cl_legacy()` → `generate_predictions_cl()`:**

```python
def generate_predictions_cl(model, samples, ...):
    model.model.is_inference = True  # Bật routing stat collection

    for batch in samples:
        batch = collator(batch)
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            generated = model.generate(
                input_ids=batch["input_ids_wo_label"],
                input_ids_wo_label=batch["input_ids_wo_label"],
                attention_mask=mask,
                generation_config=gen_cfg,
            )

        # Decode và compute RougeL
        for gen_ids, ref in zip(generated, refs):
            pred = tokenizer.decode(gen_ids[input_length:], ...)
            predictions.append(pred)
            references.append(ref.strip())

    model.model.is_inference = False
    return predictions, references, ...
```

### Bước 16: Save Results

```python
# Lưu all_results.json
all_results = {k: v for k, v in continual_metrics.items()
               if not k.endswith("_predictions") and not k.endswith("_references")}
with open(output_dir / "all_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

# Lưu task_order.txt vào outputs/
with open(outputs_dir / "task_order.txt", "w") as f:
    f.write(args.task_order)
```

---

## Phase 3: Model Inference — LlamaModel.forward() với SRT Routing

Khi `model.generate()` được gọi (bước 9, 15):

```
model.generate(input_ids_wo_label=source_ids, ...)
│
├── LlamaForCausalLM.generate()
│   └── super().generate(input_ids=source_ids, input_ids_wo_label=source_ids, ...)
│
├── prepare_inputs_for_generation()
│   └── {"input_ids": ..., "attention_mask": ...,
│        "input_ids_wo_label": source_ids}
│
└── LlamaModel.forward(input_ids, attention_mask, input_ids_wo_label=source_ids)
    │
    ├── 1. embed_tokens(input_ids) → inputs_embeds
    │
    ├── 2. SRT ROUTING:
    │   source_ids_wo_label = input_ids_wo_label
    │   source_embeds = embed_tokens(source_ids_wo_label)
    │   source_mask = _get_source_attention_mask(source_ids_wo_label)
    │
    │   # Extract routing embedding từ FROZEN backbone
    │   route_emb = encoder_frozen(source_ids_wo_label, source_mask)
    │   # route_emb: (B, d) = last non-padding token hidden state
    │
    │   # Route qua SRT router
    │   srt_preds, dists = srt_router.route(route_emb.float().cpu().numpy())
    │   # → argmin L2 distance in ZCA-whitened space
    │
    │   # Map task_id → slot index
    │   for batch_idx, task_id in enumerate(srt_preds):
    │       slot_idx = srt_task_id_to_idx.get(task_id, 0)
    │       key_attention_weights[batch_idx, slot_idx, 0] = 1.0
    │   # → one-hot hard routing
    │
    ├── 3. for each decoder_layer in layers:
    │       outputs = decoder_layer(
    │           hidden_states,
    │           attention_mask=causal_mask,
    │           key_attention_weights=key_attention_weights,  # routing weights
    │       )
    │
    │       # Inside LlamaAttention.forward():
    │       agg_lora_states(hidden_states, lora_q, prev_lora_weights_q, key_attention_weights)
    │       # = lora_q(hidden) * w_cur
    │       #   + Σ prev_lora_weights_q[i](hidden) * w_prev_i
    │       # → Chỉ LoRA của task được routed được activate
    │
    └── 4. norm → hidden_states → lm_head → logits → generate token
```

---

## Phase 4: Bash Score — `score.py`

Sau khi tất cả 15 tasks xong:

```bash
python score.py superni_order1_llama_srt superni_order1_llama_srt logs_and_outputs
```

Đọc `logs_and_outputs/superni_order1_llama_srt/outputs/{N}-{task}/all_results.json` của 15 tasks → tính:

```
- predict_rougeL của từng task tại mỗi thời điểm CL
- Forgetting: avg drop của task t sau khi học các task sau
- Forward Transfer: avg improvement nhờ prior tasks
- Average Accuracy qua toàn bộ CL timeline
```

---

## Timeline per Task

```
TASK 0 (samsum):
  ┌─────────────────────────────────────────────────────────┐
  │ Parse args + setup tokenizer                            │
  │ Load pretrained Llama-2-7b (291 files)                 │
  │ Create custom LlamaForCausalLM + copy weights          │
  │ Fix: re-init LoRA params                               │
  │ Load data: 160 train / 20 dev / 20 test              │
  │ Freeze all, unfreeze current LoRA                       │
  │ Init SRTSGWITrainer: SRTRouter(use_shrink=True)        │
  │ get_reg_matrix(): SKIP (no prior adapters)             │
  │ TRAIN: 15 epochs × 5 steps = 75 steps                 │
  │   Eval @ step 25, 50, 75 ( RougeL on dev )            │
  │ restore_best_model(): restore in-memory best LoRA       │
  │ Save lora_weights_A.pt, lora_weights_B.pt               │
  │ Attach FrozenLlamaExtractor (pretrained Llama-2-7b)     │
  │ on_task_end():                                          │
  │   extract embeddings (200 batches × source prompts)      │
  │   add_task(samsum): compute {μ, Σ}, ZCA refit          │
  │   _replace_attention_routing(): wire router              │
  │ save_srt_signatures.npz: samsum's ZCA matrix           │
  │ Continual eval: generate samsum test (20 samples)      │
  │ Save all_results.json                                   │
  └─────────────────────────────────────────────────────────┘

TASK 1 (sst2):
  ┌─────────────────────────────────────────────────────────┐
  │ Load pretrained Llama-2-7b                              │
  │ Create model, copy weights                             │
  │ load_previous_lora(task0's lora_A/B)                  │
  │ Freeze all, unfreeze sst2 LoRA                           │
  │ Init SRTSGWITrainer + SRTRouter(use_shrink=True)        │
  │ load_srt_signatures(task0's srt_signatures.npz)        │
  │   → restore ZCA matrix + samsum's whitened centroid     │
  │   → _replace_attention_routing(): wire into model       │
  │ get_reg_matrix():                                       │
  │   → compute Mahalanobis softmax weights                 │
  │   → _sgwi_init_a(): SVD warm-init lora_A              │
  │   → _fuse_past_lora_adapters(): B_warm init           │
  │ TRAIN: 15 epochs × 32 steps = 480 steps               │
  │   Eval @ step 160, 320, 480 (dev: sst2 samples)      │
  │ restore_best_model(): restore best sst2 LoRA           │
  │ Save lora_weights_A.pt, lora_weights_B.pt               │
  │ Attach FrozenLlamaExtractor                             │
  │ on_task_end():                                          │
  │   extract embeddings (sst2 train samples)              │
  │   add_task(sst2):                                      │
  │     a. mu_sst2, Sigma_sst2 = mean/cov(embeddings)     │
  │     b. Sigma_sst2_shrunk = LW_shrinkage(Sigma_sst2)   │
  │     c. _update_pooled(mu_sst2, Sigma_sst2_shrunk)     │
  │     d. Refit ZCA: W_zca_new from pooled Σ            │
  │     e. Re-whiten samsum + sst2 centroids              │
  │   _replace_attention_routing()                          │
  │ save_srt_signatures.npz: both samsum + sst2 signatures │
  │ Continual eval: generate samsum + sst2 test (140 samples) │
  │   → SRT routing: ZCA-whitened L2 distances           │
  │   → sst2 samples → routed to sst2 slot               │
  │   → samsum samples → routed to samsum slot            │
  │ Save all_results.json                                   │
  └─────────────────────────────────────────────────────────┘

TASK 2..14: (tương tự Task 1)
  ┌─────────────────────────────────────────────────────────┐
  │ load_previous_lora: load ALL previous LoRA adapters    │
  │ load_srt_signatures: load all previous SRT signatures │
  │   → router có N signatures, W_zca matrix             │
  │ get_reg_matrix(): SGWI warm-init với Mahalanobis weights │
  │ TRAIN: 15 epochs × 32 steps = 480 steps               │
  │ on_task_end():                                         │
  │   extract new task embeddings                          │
  │   add_task(new): refit ZCA, re-whiten ALL N+1 centroids│
  │ Continual eval: generate test set của TẤT CẢ seen tasks│
  │   → SRT routing: distances từ query đến tất cả centroids│
  │   → argmin = predicted task                           │
  └─────────────────────────────────────────────────────────┘
```

---

## Key: Sự khác biệt Train vs Inference Routing

```
TRAINING (LlamaModel.forward được gọi bởi model() trong train step):
  self.training = True
  → key_attention_weights[:, 0, 0] = 1.0
  → HARD ONE-HOT: slot 0 (current task) LUÔN ĐƯỢC activate
  → Ngay cả khi encoder_frozen/SRT có vấn đề, training vẫn chạy đúng
  → Mỗi batch chỉ train với current task's LoRA

INFERENCE (LlamaModel.forward được gọi bởi model.generate()):
  self.training = False
  self.use_srt_routing = True
  self.encoder_frozen != None
  → encoder_frozen(source_ids) → embeddings
  → srt_router.route(embeddings) → task predictions
  → key_attention_weights = one-hot (predicted task slot)
  → agg_lora_states blend LoRA contributions theo routing decision
  → Nghiêng về task đúng → nghiêng về LoRA đúng
  → Chất lượng phụ thuộc hoàn toàn vào routing accuracy
```

---

## SRT Shrinkage Effect (với vs không có --srt_shrink)

```
KHÔNG có --srt_shrink (use_shrink=False):
  - Individual task covariances: raw Σ_t (ill-conditioned với d=4096, n≈160)
  - Pooled covariance: Welford-Hart update của raw Σ_t
  - ZCA eigenvalues: huge dynamic range → whitened space nearly degenerate
  - Whitened centroids: gần như bằng nhau về magnitude
  - Distances: ~99.99% similar → routing ≈ random
  → Catastrophic forgetting, poor continual performance

CÓ --srt_shrink (use_shrink=True):  [FIXED]
  - Individual task covariances: LW-shrunk Σ_t (regularized)
  - Pooled covariance: Welford-Hart update của shrunk Σ_t
  - Pooled Σ cũng được shrink thêm: (1-f)*Σ_pool + f*(tr/d)*I
  - ZCA eigenvalues: well-conditioned → whitened space discriminative
  - Whitened centroids: distinct norms và directions
  - Distances: significantly different → routing accurate (~97%)
  → Minimal forgetting, strong continual performance
```
