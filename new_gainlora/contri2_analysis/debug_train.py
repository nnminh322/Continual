#!/usr/bin/env python3
"""
debug_train.py — Minimal debug script to verify isolated training works.
Tests a single task (CB) with both random init and SFI init.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent
SRC  = ROOT.parent / "src"
sys.path.insert(0, str(SRC))

import torch
from transformers import AutoTokenizer
from contri2_utils import (
    load_task_data, build_model, init_lora_weights,
    format_example, TASK_ORDER,
    RANDOM_INIT, SVD_FUSION_INIT,
)

# ── Step 1: Load data ────────────────────────────────────────────────────────
task = "cb"
train_data = load_task_data(task, "train")
test_data  = load_task_data(task, "test")
print(f"Loaded {len(train_data)} train, {len(test_data)} test for {task}")

# Check label format
inp, lbl = format_example(train_data[0])
print(f"Input sample:  {inp[:80]}...")
print(f"Label sample:  '{lbl}'")
print(f"Label repr:   {repr(lbl)}")

# Check all unique labels
labels = list(set(format_example(s)[1] for s in train_data))
print(f"Unique labels ({len(labels)}): {sorted(labels)}")
random_baseline = 100.0 / len(labels)
print(f"Random baseline: {random_baseline:.2f}%")

# ── Step 2: Build model + random init ──────────────────────────────────────
model = build_model("google/flan-t5-small", adapter_mode=False)
init_lora_weights(model, mode=RANDOM_INIT, router=None, t_name=task,
                  all_lora_paths=None, task_list=TASK_ORDER)
model.train()
device = next(model.parameters()).device
tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
optimizer = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad),
    lr=3e-4,
)

# ── Step 3: Forward pass sanity check ──────────────────────────────────────
print("\n── Forward Pass Sanity Check ──")
inp, lbl = format_example(train_data[0])
inputs = tokenizer(inp, return_tensors="pt", max_length=512, truncation=True).to(device)
targets = tokenizer(lbl, return_tensors="pt", max_length=50).to(device)

# Check LoRA A/B values before training
lora_a_vals = []
for m in model.modules():
    if hasattr(m, 'lora_A'):
        lora_a_vals.append(m.lora_A.data.clone())
print(f"LoRA A param count: {len(lora_a_vals)}")
print(f"LoRA A[0] mean: {lora_a_vals[0].mean():.6f}, std: {lora_a_vals[0].std():.6f}")

# Forward with loss
outputs = model(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    labels=targets["input_ids"],
)
loss_before = outputs.loss.item()
print(f"Loss (before training): {loss_before:.4f}")

# ── Step 4: Gradient check ──────────────────────────────────────────────────
print("\n── Gradient Check ──")
loss_before.backward()
grad_norms = []
for name, p in model.named_parameters():
    if p.grad is not None and p.grad.abs().sum() > 0:
        grad_norms.append((name, p.grad.norm().item()))
print(f"Params with non-zero gradients: {len(grad_norms)}")
for name, gnorm in grad_norms[:5]:
    print(f"  {name}: grad_norm={gnorm:.6f}")
optimizer.step()
optimizer.zero_grad()

# ── Step 5: Quick 10-step training ─────────────────────────────────────────
print("\n── Quick Training (10 steps) ──")
model.train()
losses = []
for i in range(10):
    batch_inp = [format_example(train_data[j])[0] for j in range(i*4, i*4+4)]
    batch_lbl = [format_example(train_data[j])[1] for j in range(i*4, i*4+4)]
    inp_t = tokenizer(batch_inp, return_tensors="pt", padding=True,
                      max_length=128, truncation=True).to(device)
    lbl_t = tokenizer(batch_lbl, return_tensors="pt", padding=True,
                      max_length=20, truncation=True).to(device)
    out = model(input_ids=inp_t["input_ids"],
                attention_mask=inp_t["attention_mask"],
                labels=lbl_t["input_ids"])
    loss = out.loss
    losses.append(loss.item())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
print(f"Loss over 10 steps: {[f'{l:.4f}' for l in losses]}")
print(f"Loss change: {losses[0]:.4f} → {losses[-1]:.4f}")

# ── Step 6: Greedy decode sanity check ─────────────────────────────────────
print("\n── Greedy Decode Check ──")
model.eval()
inp, lbl = format_example(test_data[0])
print(f"Input:  {inp[:100]}...")
print(f"Label:  '{lbl}'")

inputs = tokenizer(inp, return_tensors="pt", max_length=512, truncation=True).to(device)
enc_out = model.encoder(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], head_mask=None)
enc_h = enc_out.last_hidden_state

dec_ids = torch.full((1, 1), tokenizer.pad_token_id or 0, dtype=torch.long, device=device)
for _ in range(30):
    dec_out = model.decoder(input_ids=dec_ids, encoder_hidden_states=enc_h,
                             encoder_attention_mask=inputs["attention_mask"], head_mask=None)
    logits = model.lm_head(dec_out.last_hidden_state)
    next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    dec_ids = torch.cat([dec_ids, next_tok], dim=1)
    if (next_tok == (tokenizer.eos_token_id or 1)).all():
        break

pred = tokenizer.decode(dec_ids[0], skip_special_tokens=True).strip()
pred_clean = pred.replace("Output:", "").replace("output:", "").strip()
print(f"Prediction raw:  '{pred}'")
print(f"Prediction clean: '{pred_clean}'")
print(f"Match: {pred_clean.lower() == lbl.lower()}")
