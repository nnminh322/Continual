"""
Reproduce the zero-gradient issue with T5-like LoRA + gradient checkpointing.
Tests with and without gradient checkpointing to isolate the cause.
"""
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class LoRALayer(nn.Module):
    """Exact copy of the LoRALayer from t5_gainlora_inflora.py"""
    def __init__(self, in_features, out_features, r=4, lora_alpha=32, lora_dropout=0.0):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
        self.scaling = self.lora_alpha / self.r
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        result = (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        return result.reshape(x.shape[0], -1, x.shape[-1])


class FakeT5Attention(nn.Module):
    """Mimics T5Attention with LoRA on q and v"""
    def __init__(self, d_model=64, n_heads=4, d_kv=16, r=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_kv = d_kv
        inner_dim = n_heads * d_kv

        self.q = nn.Linear(d_model, inner_dim, bias=False)
        self.k = nn.Linear(d_model, inner_dim, bias=False)
        self.v = nn.Linear(d_model, inner_dim, bias=False)
        self.o = nn.Linear(inner_dim, d_model, bias=False)

        self.lora_q = LoRALayer(d_model, inner_dim, r=r)
        self.lora_v = LoRALayer(d_model, inner_dim, r=r)
        self.gradient_checkpointing = False

    def forward(self, hidden_states, mask=None, position_bias=None,
                key_value_states=None, **kwargs):
        bs, seq_len = hidden_states.shape[:2]

        def shape(states):
            return states.view(bs, -1, self.n_heads, self.d_kv).transpose(1, 2)

        # Exactly like T5Attention when key_attention_weights=None
        query_states = shape(self.q(hidden_states) + self.lora_q(hidden_states))
        key_states = shape(self.k(hidden_states) if key_value_states is None
                          else self.k(key_value_states))
        value_states = shape(self.v(hidden_states) + self.lora_v(hidden_states))

        scores = torch.matmul(query_states, key_states.transpose(3, 2))
        if position_bias is not None:
            scores += position_bias

        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_kv)
        return (self.o(attn_output), None, position_bias)


class FakeT5Block(nn.Module):
    """Mimics T5Block with self-attention + FFN"""
    def __init__(self, d_model=64, n_heads=4, d_kv=16, d_ff=128, r=4):
        super().__init__()
        self.attn = FakeT5Attention(d_model, n_heads, d_kv, r)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False),
        )

    def forward(self, hidden_states, attention_mask=None, position_bias=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                encoder_decoder_position_bias=None, layer_head_mask=None,
                cross_attn_layer_head_mask=None, past_key_value=None,
                use_cache=False, output_attentions=False,
                key_attention_weights=None):
        # Self-attention
        normed = self.norm1(hidden_states)
        attn_out = self.attn(normed, position_bias=position_bias)
        hidden_states = hidden_states + attn_out[0]
        position_bias = attn_out[2]
        # FFN
        normed = self.norm2(hidden_states)
        hidden_states = hidden_states + self.ff(normed)
        return (hidden_states, None, position_bias)


class FakeT5Model(nn.Module):
    """Mimics T5ForConditionalGeneration with encoder + decoder + lm_head"""
    def __init__(self, d_model=64, n_heads=4, d_kv=16, d_ff=128, n_layers=4, vocab_size=100, r=4):
        super().__init__()
        self.d_model = d_model
        self.shared = nn.Embedding(vocab_size, d_model)
        self.encoder_blocks = nn.ModuleList([FakeT5Block(d_model, n_heads, d_kv, d_ff, r) for _ in range(n_layers)])
        self.decoder_blocks = nn.ModuleList([FakeT5Block(d_model, n_heads, d_kv, d_ff, r) for _ in range(n_layers)])
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.gradient_checkpointing = False
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, labels=None):
        # Encoder
        hidden_states = self.dropout(self.shared(input_ids))
        position_bias = None
        for block in self.encoder_blocks:
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache=False, output_attentions=False,
                                           key_attention_weights=None))
                    return custom_forward
                layer_outputs = checkpoint(
                    create_custom_forward(block),
                    hidden_states, None, position_bias,
                    None, None, None, None, None, None,
                    use_reentrant=False,
                )
            else:
                layer_outputs = block(hidden_states, position_bias=position_bias)
            hidden_states = layer_outputs[0]
            position_bias = layer_outputs[2]

        encoder_output = self.encoder_norm(hidden_states)

        # Decoder (simplified: just self-attention, no cross-attention)
        decoder_hidden = self.dropout(self.shared(labels if labels is not None else input_ids))
        position_bias = None
        for block in self.decoder_blocks:
            if self.gradient_checkpointing and self.training:
                def create_custom_forward2(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache=False, output_attentions=False,
                                           key_attention_weights=None))
                    return custom_forward
                layer_outputs = checkpoint(
                    create_custom_forward2(block),
                    decoder_hidden, None, position_bias,
                    None, None, None, None, None, None,
                    use_reentrant=False,
                )
            else:
                layer_outputs = block(decoder_hidden, position_bias=position_bias)
            decoder_hidden = layer_outputs[0]
            position_bias = layer_outputs[2]

        decoder_output = self.decoder_norm(decoder_hidden)
        logits = self.lm_head(decoder_output * (self.d_model ** -0.5))

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss, logits


def test_gradients(model, input_ids, labels, tag):
    model.train()
    model.zero_grad()
    loss, _ = model(input_ids, labels=labels)
    loss.backward()

    n_ok, n_zero, n_none = 0, 0, 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None:
                n_none += 1
            elif p.grad.norm().item() > 0:
                n_ok += 1
            else:
                n_zero += 1
    status = "PASS" if n_ok > 0 else "FAIL"
    print(f"  [{status}] {tag}: loss={loss.item():.4f}, grad>0={n_ok}, grad==0={n_zero}, grad=None={n_none}")
    return n_ok > 0


print("=" * 70)
print("TEST: LoRA gradient flow with/without gradient checkpointing")
print("=" * 70)

torch.manual_seed(42)
model = FakeT5Model(d_model=64, n_heads=4, d_kv=16, d_ff=128, n_layers=4, vocab_size=100, r=4)

# Freeze all except lora_B (InfLoRA style)
for name, p in model.named_parameters():
    if 'lora_B' in name:
        p.requires_grad = True
    else:
        p.requires_grad = False

n_trainable = sum(1 for n, p in model.named_parameters() if p.requires_grad)
print(f"Trainable params: {n_trainable} (all lora_B)")
print(f"All lora_B are zeros: {all((p.data == 0).all().item() for n, p in model.named_parameters() if 'lora_B' in n)}")

input_ids = torch.randint(0, 100, (2, 16))
labels = torch.randint(0, 100, (2, 16))

# Test 1: Without gradient checkpointing
print("\n--- Test 1: No gradient checkpointing ---")
model.gradient_checkpointing = False
passed1 = test_gradients(model, input_ids, labels, "no_checkpoint")

# Test 2: With gradient checkpointing (use_reentrant=False)
print("\n--- Test 2: Gradient checkpointing (use_reentrant=False) ---")
model.gradient_checkpointing = True
passed2 = test_gradients(model, input_ids, labels, "checkpoint_non_reentrant")

# Test 3: With enable_input_require_grads + gradient checkpointing
print("\n--- Test 3: enable_input_require_grads + gradient checkpointing ---")
def enable_input_require_grads_hook(module, input, output):
    output.requires_grad_(True)
hook = model.shared.register_forward_hook(enable_input_require_grads_hook)
model.gradient_checkpointing = True
passed3 = test_gradients(model, input_ids, labels, "checkpoint + require_grads")
hook.remove()

# Test 4: Without checkpointing but WITH enable_input_require_grads
print("\n--- Test 4: enable_input_require_grads, no checkpointing ---")
hook = model.shared.register_forward_hook(enable_input_require_grads_hook)
model.gradient_checkpointing = False
passed4 = test_gradients(model, input_ids, labels, "no_checkpoint + require_grads")
hook.remove()

print("\n" + "=" * 70)
print("SUMMARY:")
print(f"  Test 1 (no checkpoint):         {'PASS' if passed1 else 'FAIL'}")
print(f"  Test 2 (checkpoint+non_reent):  {'PASS' if passed2 else 'FAIL'}")
print(f"  Test 3 (ckpt+req_grads):        {'PASS' if passed3 else 'FAIL'}")
print(f"  Test 4 (no_ckpt+req_grads):     {'PASS' if passed4 else 'FAIL'}")
print("=" * 70)

if all([passed1, passed2, passed3, passed4]):
    print("\nAll tests PASSED. The issue is NOT reproducible with simplified model.")
    print("The zero-gradient bug is likely specific to the actual T5 model interaction.")
    print("Next step: try disabling gradient checkpointing entirely on Colab.")
elif not passed1:
    print("\nTest 1 FAILED: Issue is NOT related to gradient checkpointing!")
    print("The model architecture itself has a gradient problem.")
else:
    print(f"\nGradient checkpointing causes the issue.")
    print(f"Recommendation: disable gradient checkpointing and use BSZ=1 GA=32.")
