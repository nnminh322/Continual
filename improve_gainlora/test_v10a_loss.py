import sys
import os
import torch

sys.path.append('src')

from t5_specroute import T5ForConditionalGeneration
from transformers import AutoConfig, AutoTokenizer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

config = AutoConfig.from_pretrained('google/flan-t5-small')
tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-small')

prompt_config = {
    'seq_len': 512,
    'mlp_hidden_dim': 100,
    'attn_temperature': 1.0,
    'routing_mode': 'learned',
    'previous_lora_path': None,
    'previous_prompt_key_path': None,
    'task_id': 0,
    'run_single': False,
    'lora_r': 8,
    'lora_alpha': 16,
    'lora_dropout': 0.1,
    'training_bias': 'none',
    'target_routing_alpha': 0.8,
}

print("Loading model...")
model = T5ForConditionalGeneration.from_pretrained(
    'google/flan-t5-small',
    prompt_config,
    config=config
).to(device)

print("Reinitializing Lora A...")
import math
import torch.nn as nn
for module in model.modules():
    if hasattr(module, 'lora_A') and hasattr(module, 'lora_B') and hasattr(module, 'reset_parameters'):
        nn.init.kaiming_uniform_(module.lora_A, a=math.sqrt(5))

model.train()
# Freeze all except lora_B, trans_input, prompt_key
for name, param in model.named_parameters():
    param.requires_grad = False
    if "lora_B" in name or "trans_input" in name or "prompt_key" in name:
        param.requires_grad = True

input_text = ["This is a test sentence.", "Another test."]
inputs = tokenizer(input_text, return_tensors='pt', padding=True).to(device)
labels = tokenizer(["Positive", "Negative"], return_tensors='pt', padding=True).to(device)

print("Forward pass...")
outputs = model(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    labels=labels.input_ids
)

loss = outputs.loss
print(f"Loss: {loss.item()}")

print("Backward pass...")
loss.backward()

n_grad = 0
n_zero = 0
n_nan = 0
n_none = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        if param.grad is not None:
            norm = param.grad.norm().item()
            if torch.isnan(param.grad).any():
                n_nan += 1
            elif norm > 0:
                n_grad += 1
            else:
                n_zero += 1
        else:
            n_none += 1

print(f"Gradients -> >0: {n_grad}, ==0: {n_zero}, NaN: {n_nan}, None: {n_none}")
