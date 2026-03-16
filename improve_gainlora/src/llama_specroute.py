# coding=utf-8
"""
SpecRoute: Llama model with Spectral-Geometric Routing for Continual LoRA Learning.

Replaces GainLoRA's learned gating (Trans_input + prompt_key) with spectral
projection-based routing using SVD signatures from frozen LoRA weights.

Key changes from llama_gainlora_inflora.py:
- Removed: Trans_input, prompt_key, previous_trans_input, previous_prompts_keys
- Removed: All trans_input GPM (get_chunk/get_matrix3 on LlamaModel)
- Removed: memory_replay (no KL loss — routing is parameter-free)
- Added: Spectral signatures + projection-based routing (softmax over weighted projection fits)
- Added: compute_spectral_signatures for Llama architecture
- Added: _thin_svd_low_rank for efficient SVD computation
"""
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from transformers.models.llama.configuration_llama import LlamaConfig

# Prefer magma backend for linalg ops
try:
    torch.backends.cuda.preferred_linalg_library("magma")
except Exception:
    pass

logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = "LlamaConfig"


# ===================== Spectral Routing Functions =====================

def _thin_svd_low_rank(B, A, device=None):
    """
    Compute SVD of delta_W = B @ A efficiently using QR decomposition.
    Since delta_W has rank <= r, we decompose via two small QR factorizations
    and one tiny r×r SVD. Mathematically identical to full SVD, ~2000× faster.
    """
    try:
        Q_B, R_B = torch.linalg.qr(B)
        Q_A, R_A = torch.linalg.qr(A.T)
        small = R_B @ R_A.T
        _, S, Vh_s = torch.linalg.svd(small, full_matrices=False)
        Vt = Vh_s @ Q_A.T
    except RuntimeError:
        B_cpu, A_cpu = B.cpu(), A.cpu()
        Q_B, R_B = torch.linalg.qr(B_cpu)
        Q_A, R_A = torch.linalg.qr(A_cpu.T)
        small = R_B @ R_A.T
        _, S, Vh_s = torch.linalg.svd(small, full_matrices=False)
        Vt = Vh_s @ Q_A.T
        target = device if device is not None else B.device
        S, Vt = S.to(target), Vt.to(target)
    return S, Vt


def compute_spectral_signatures(model, config):
    """
    Compute spectral signatures from all LoRA branches after training.
    For Llama: only self-attention layers (decoder-only, no encoder/cross-attention).
    """
    signatures = {}
    with torch.no_grad():
        for j in range(config.num_hidden_layers):
            attn = model.model.layers[j].self_attn
            for name, lora in [('q', attn.lora_q), ('v', attn.lora_v)]:
                A = lora.lora_A.data.float()
                B = lora.lora_B.data.float()
                r = lora.r
                S, Vt = _thin_svd_low_rank(B, A)
                signatures[f'layer.{j}.{name}'] = {
                    'V': Vt[:r].cpu(),
                    'sigma': S[:r].cpu()
                }
    return signatures


# ===================== Reuse unchanged components from ROOT =====================

def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, r, lora_alpha=1, lora_dropout=0.):
        super(LoRALayer, self).__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.out_features = out_features
        self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
        self.scaling = self.lora_alpha / self.r
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Keep lora_s for compatibility with ROOT weight loading
        self.lora_s = nn.Parameter(torch.randn(1))
        self.s_avg = torch.ones(1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        result = (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        return result.reshape(x.shape[0], -1, self.out_features)


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos.squeeze(1).squeeze(0)
    sin = sin.squeeze(1).squeeze(0)
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ===================== SpecRoute LlamaAttention =====================
# Same agg_lora_states as ROOT, same LoRA on Q and V.
# No GetSubnetFaster gating (SpecRoute doesn't use lora_s masking).

class LlamaAttention(nn.Module):
    """Multi-headed attention with LoRA routing via external key_attention_weights."""

    def __init__(self, config: LlamaConfig, prompt_config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.lora_q = LoRALayer(self.hidden_size, self.num_heads * self.head_dim,
                                r=prompt_config["lora_r"], lora_alpha=prompt_config["lora_alpha"],
                                lora_dropout=prompt_config["lora_dropout"])
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.lora_v = LoRALayer(self.hidden_size, self.num_heads * self.head_dim,
                                r=prompt_config["lora_r"], lora_alpha=prompt_config["lora_alpha"],
                                lora_dropout=prompt_config["lora_dropout"])
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

        self.previous_lora_weights_q, self.previous_lora_weights_v = None, None
        self.prompt_config = prompt_config
        if prompt_config["previous_lora_path"] is not None:
            with torch.no_grad():
                self.previous_lora_weights_q = nn.ModuleList()
                for i in range(prompt_config["task_id"]):
                    layer = LoRALayer(self.hidden_size, self.num_heads * self.head_dim,
                                     r=prompt_config["lora_r"], lora_alpha=prompt_config["lora_alpha"],
                                     lora_dropout=prompt_config["lora_dropout"])
                    self.previous_lora_weights_q.append(layer)
                self.previous_lora_weights_v = nn.ModuleList()
                for i in range(prompt_config["task_id"]):
                    layer = LoRALayer(self.hidden_size, self.num_heads * self.head_dim,
                                     r=prompt_config["lora_r"], lora_alpha=prompt_config["lora_alpha"],
                                     lora_dropout=prompt_config["lora_dropout"])
                    self.previous_lora_weights_v.append(layer)

        # GPM feature collection (same as ROOT)
        self.get_feature = False
        self.stage = 0
        self.matrix = torch.zeros(self.hidden_size, self.hidden_size)
        self.n_matrix = 0

    def get_chunk(self, chunk):
        self.chunk = chunk
        self.index, self.step = chunk, self.hidden_size // chunk
        self.matrix, self.n_matrix = {}, {}
        for index in range(self.index):
            self.matrix[index] = torch.zeros(self.step, self.step).cuda()
            self.n_matrix[index] = 0

    def get_matrix3(self, x):
        for index in range(self.index):
            if self.matrix[index] is None:
                self.matrix[index] = torch.bmm(
                    x[:,:,index*self.step:(index+1)*self.step].detach().permute(0, 2, 1),
                    x[:,:,index*self.step:(index+1)*self.step].detach()
                ).sum(dim=0).float() / (x.shape[0] * x.shape[1])
            else:
                self.matrix[index] = (
                    self.matrix[index] * self.n_matrix[index]
                    + torch.bmm(
                        x[:,:,index*self.step:(index+1)*self.step].detach().permute(0, 2, 1),
                        x[:,:,index*self.step:(index+1)*self.step].detach()
                    ).sum(dim=0).float()
                ) / (self.n_matrix[index] + x.shape[0] * x.shape[1])
            self.n_matrix[index] += x.shape[0] * x.shape[1]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        key_attention_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        def agg_lora_states(hidden_states, lora_layer, pre_lora_layer, key_attention_weights):
            _, num_task, _ = key_attention_weights.size()
            if pre_lora_layer is not None and num_task > 1:
                cur_lora_states = lora_layer(hidden_states).unsqueeze(0)
                with torch.no_grad():
                    pre_lora_states = torch.cat([pre_lora(hidden_states).unsqueeze(0) for pre_lora in pre_lora_layer], dim=0)
                concat_q = torch.cat([cur_lora_states, pre_lora_states], dim=0).transpose(0, 1).reshape(bsz, -1, hidden_states.shape[1]*self.num_heads * self.head_dim)
                agg_lora_states = torch.matmul(key_attention_weights.transpose(1, 2), concat_q).squeeze()
            else:
                cur_lora_states = lora_layer(hidden_states).unsqueeze(0).transpose(0, 1).reshape(bsz, -1, hidden_states.shape[1]*self.num_heads * self.head_dim)
                agg_lora_states = torch.matmul(key_attention_weights.transpose(1, 2), cur_lora_states).squeeze()
            return agg_lora_states.reshape(bsz, -1, self.num_heads * self.head_dim)

        # Collect features for GPM
        if self.get_feature:
            self.get_matrix3(hidden_states)

        if key_attention_weights is not None:
            query_states = (self.q_proj(hidden_states) + agg_lora_states(hidden_states, self.lora_q, self.previous_lora_weights_q, key_attention_weights)).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            query_states = (self.q_proj(hidden_states) + self.lora_q(hidden_states)).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        if key_attention_weights is not None:
            value_states = (self.v_proj(hidden_states) + agg_lora_states(hidden_states, self.lora_v, self.previous_lora_weights_v, key_attention_weights)).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            value_states = (self.v_proj(hidden_states) + self.lora_v(hidden_states)).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device))

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, prompt_config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config, prompt_config=prompt_config)
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        key_attention_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            key_attention_weights=key_attention_weights,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LlamaModel):
            module.gradient_checkpointing = value


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens.
"""


# ===================== SpecRoute LlamaModel =====================

@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Llama decoder with SpecRoute spectral routing instead of learned routing.
    """

    def __init__(self, config: LlamaConfig, prompt_config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, prompt_config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.prompt_config = prompt_config
        self.is_inference = False

        if not prompt_config["run_single"]:
            self.model_dim = config.hidden_size

            # ===== Spectral routing: NO learned parameters =====
            self.spectral_signatures = []  # List[dict] per old task
            self.routing_temperature = prompt_config.get('attn_temperature', 1.0)
            self.training_bias = 1.0  # Boost current task during training

            # For inference logging
            self.all_attn_weights = []

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )
        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
        return combined_attention_mask

    def compute_spectral_routing(self, avg_inputs_embeds):
        """
        Compute routing weights using spectral projection fits.
        For Llama (decoder-only): uses self-attention layers only.

        Args:
            avg_inputs_embeds: (B, 1, d_model) — averaged input token embeddings

        Returns:
            (B, n_tasks, 1) routing weights via softmax over projection fits
        """
        h = avg_inputs_embeds  # (B, 1, d_model)
        h_norm_sq = (h ** 2).sum(dim=-1) + 1e-8  # (B, 1)

        fits = []

        # 1. Current task fit: use A rows directly (cold-start safe proxy)
        current_fits_layers = []
        for layer in self.layers:
            attn = layer.self_attn
            for lora in [attn.lora_q, attn.lora_v]:
                A = lora.lora_A.data.float()
                r = lora.r
                A_h = A.to(h.device, dtype=h.dtype)
                proj = torch.matmul(h, A_h.T)  # (B, 1, r)
                fit = (proj ** 2).sum(dim=-1) / (r * h_norm_sq)  # (B, 1)
                current_fits_layers.append(fit)
        current_fit = torch.stack(current_fits_layers).mean(dim=0)  # (B, 1)
        if self.training and hasattr(self, 'training_bias'):
            current_fit = current_fit + self.training_bias
        fits.append(current_fit)

        # 2. Previous tasks fit: use spectral signatures (V, sigma)
        for sig_dict in self.spectral_signatures:
            task_fits = []
            for key, sig_data in sig_dict.items():
                V = sig_data['V'].to(h.device, dtype=h.dtype)
                sigma = sig_data['sigma'].to(h.device, dtype=h.dtype)
                proj = torch.matmul(h, V.T)  # (B, 1, r)
                sigma_sq = sigma ** 2
                sigma_sq_sum = sigma_sq.sum() + 1e-8
                weighted_proj = (proj ** 2 * sigma_sq.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
                fit = weighted_proj / (sigma_sq_sum * h_norm_sq)
                task_fits.append(fit)

            if task_fits:
                task_fit = torch.stack(task_fits).mean(dim=0)
            else:
                task_fit = torch.zeros_like(current_fit)
            fits.append(task_fit)

        fit_scores = torch.cat(fits, dim=1)  # (B, n_tasks)
        weights = torch.softmax(fit_scores / self.routing_temperature, dim=1)
        return weights.unsqueeze(2)  # (B, n_tasks, 1)

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        input_ids_wo_label: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # ============ SPECTRAL ROUTING ============
        key_attention_weights = None
        if not self.prompt_config["run_single"]:
            if len(self.spectral_signatures) > 0:
                # Multi-task: compute spectral routing using input embeddings
                inputs_embeds_for_query = self.embed_tokens(input_ids_wo_label)
                avg_inputs_embeds = ((input_ids_wo_label != 1).long().unsqueeze(-1) * inputs_embeds_for_query).mean(dim=1, keepdim=True)
                key_attention_weights = self.compute_spectral_routing(avg_inputs_embeds)
                # Detach: routing is parameter-free, detach avoids graph issues with gradient checkpointing
                key_attention_weights = key_attention_weights.detach()

                if self.is_inference:
                    self.all_attn_weights.append(
                        key_attention_weights.squeeze().mean(dim=0).detach().to(torch.float).cpu().numpy()
                    )
            else:
                # First task: single LoRA, weight = 1
                key_attention_weights = torch.ones(
                    batch_size, 1, 1, device=inputs_embeds.device, dtype=inputs_embeds.dtype
                )
                if self.is_inference:
                    self.all_attn_weights.append(
                        key_attention_weights.squeeze(2).mean(dim=0).detach().to(torch.float).cpu().numpy()
                    )
        # ============================================

        # Decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions, None)
                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    key_attention_weights=key_attention_weights,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaForCausalLM(LlamaPreTrainedModel):
    def __init__(self, config, prompt_config):
        super().__init__(config)
        self.model = LlamaModel(config, prompt_config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    # NOTE: No memory_replay method — SpecRoute has no learned routing to distill

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        input_ids_wo_label: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Labels for computing the masked language modeling loss.

        Returns:
            CausalLMOutputWithPast
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            input_ids_wo_label=input_ids_wo_label,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        input_ids_wo_label = kwargs.get("input_ids_wo_label", None)

        model_inputs.update({
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "input_ids_wo_label": input_ids_wo_label,
        })
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past
