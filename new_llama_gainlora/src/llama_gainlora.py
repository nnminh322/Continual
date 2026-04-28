# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import math
from typing import List, Optional, Tuple, Union
try:
    import ipdb
except ImportError:
    ipdb = None

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings

try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DynamicCache = None
from transformers.models.llama.configuration_llama import LlamaConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

class Trans_input(nn.Module):
    """
    Two-layer MLP that maps hidden states to a lower-dimensional space and back.

    IMPORTANT — deferred init:
    When created inside LlamaModel via from_pretrained(low_cpu_mem_usage=True),
    this module is instantiated on META tensors. We defer ALL initialization
    (weight fill + SiLU construction) until the first forward pass, AFTER the
    model has been sharded to real device/dtype by post_init().

    Previously, nn.SiLU() was called in __init__() — this crashes on meta tensors.
    Now it is lazily created on first forward().
    """
    def __init__(self, d_model, hidden_dim=100, n_tasks=1) -> None:
        super().__init__()
        self.input_linear = nn.Parameter(torch.empty((n_tasks, hidden_dim, d_model)))
        self.output_linear = nn.Parameter(torch.empty((n_tasks, d_model, hidden_dim)))
        # NOTE: do NOT call kaiming init or create SiLU here.
        # Both require real tensors (non-meta). See _init_weights() called from
        # LlamaModel.__init__ after post_init() finalizes the model.

    def _init_weights(self):
        """Initialize weights on real tensors. Call from LlamaModel._init_weights()."""
        if self.input_linear.device.type == 'meta':
            return  # not yet materialized — will be called again on first forward
        nn.init.kaiming_uniform_(self.input_linear, a=math.sqrt(3))
        nn.init.kaiming_uniform_(self.output_linear, a=math.sqrt(3))

    def forward(self, x):
        # Lazy init: when first called after meta→real tensor transition,
        # materialize weights and activation.
        if not hasattr(self, '_activation'):
            nn.init.kaiming_uniform_(self.input_linear, a=math.sqrt(3))
            nn.init.kaiming_uniform_(self.output_linear, a=math.sqrt(3))
            self._activation = nn.SiLU()
        x = x.unsqueeze(1)
        x = torch.matmul(x, self.input_linear.permute(0, 2, 1))
        x = self._activation(x)
        x = torch.matmul(x, self.output_linear.permute(0, 2, 1))
        x = self._activation(x)
        return x.squeeze(2)

class LoRALayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        lora_alpha: int = 1,
        lora_dropout: float = 0.
    ):
        super(LoRALayer, self).__init__()
        self.r = r
        self.lora_alpha = lora_alpha

        self.out_features = out_features

        self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r)))

        self.scaling = self.lora_alpha / self.r
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged

        self.reset_parameters()

    def reset_parameters(self):
        # initialize A the same way as the default for nn.Linear and B to zero
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor):
        result = (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        return result.reshape(x.shape[0], -1, self.out_features)


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
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

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

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
        self.lora_q = LoRALayer(self.hidden_size, self.num_heads * self.head_dim, r=prompt_config["lora_r"], lora_alpha=prompt_config["lora_alpha"], lora_dropout=prompt_config["lora_dropout"])
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.lora_v = LoRALayer(self.hidden_size, self.num_heads * self.head_dim, r=prompt_config["lora_r"], lora_alpha=prompt_config["lora_alpha"], lora_dropout=prompt_config["lora_dropout"])
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

        self.previous_lora_weights_q, self.previous_lora_weights_v = None, None
        self.prompt_config = prompt_config
        if prompt_config["previous_lora_path"] is not None:
            with torch.no_grad():
                self.previous_lora_weights_q = nn.ModuleList()
                for i in range(prompt_config["task_id"]):
                    layer = LoRALayer(self.hidden_size, self.num_heads * self.head_dim, r=prompt_config["lora_r"], lora_alpha=prompt_config["lora_alpha"], lora_dropout=prompt_config["lora_dropout"])
                    self.previous_lora_weights_q.append(layer)

                self.previous_lora_weights_v = nn.ModuleList()
                for i in range(prompt_config["task_id"]):
                    layer = LoRALayer(self.hidden_size, self.num_heads * self.head_dim, r=prompt_config["lora_r"], lora_alpha=prompt_config["lora_alpha"], lora_dropout=prompt_config["lora_dropout"])
                    self.previous_lora_weights_v.append(layer)

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
                self.matrix[index] = torch.bmm(x[:,:,index*self.step:(index+1)*self.step].detach().permute(0, 2, 1), x[:,:,index*self.step:(index+1)*self.step].detach()).sum(dim=0).float()/(x.shape[0]*x.shape[1])
            else:
                self.matrix[index] = (self.matrix[index]*self.n_matrix[index] + torch.bmm(x[:,:,index*self.step:(index+1)*self.step].detach().permute(0, 2, 1), x[:,:,index*self.step:(index+1)*self.step].detach()).sum(dim=0).float())/(self.n_matrix[index] + x.shape[0]*x.shape[1])
            self.n_matrix[index] += x.shape[0]*x.shape[1]
        return

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

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
            w_cur = key_attention_weights[:, 0:1, :]
            cur_contribution = lora_layer(hidden_states) * w_cur

            if pre_lora_layer is not None:
                with torch.no_grad():
                    prev_contribution = torch.zeros_like(cur_contribution)
                    for idx, pre_lora in enumerate(pre_lora_layer):
                        w_prev = key_attention_weights[:, idx + 1:idx + 2, :]
                        prev_contribution = prev_contribution + pre_lora(hidden_states) * w_prev
                return cur_contribution + prev_contribution

            return cur_contribution

        # modified
        if self.get_feature:
            self.get_matrix3(hidden_states)

        if key_attention_weights is not None:
            query_states = (self.q_proj(hidden_states)+agg_lora_states(hidden_states, self.lora_q, self.previous_lora_weights_q, key_attention_weights)).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            query_states = (self.q_proj(hidden_states)+self.lora_q(hidden_states)).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        if key_attention_weights is not None:
            value_states = (self.v_proj(hidden_states)+agg_lora_states(hidden_states, self.lora_v, self.previous_lora_weights_v, key_attention_weights)).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            value_states = (self.v_proj(hidden_states)+self.lora_v(hidden_states)).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
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
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            )

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=0.0, neginf=0.0)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

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
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
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

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # Always return a consistent 3-tuple: (hidden_states, self_attn_weights, present_key_value)
        # This is critical for LlamaModel.forward indexing: layer_outputs[2 if output_attentions else 1]
        return (hidden_states, self_attn_weights if output_attentions else None,
                present_key_value if use_cache else None)


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel, GenerationMixin):
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


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
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
        ##########################
        if not prompt_config["run_single"]:

            self.model_dim = config.hidden_size

            self.prompt_key = nn.Parameter(torch.randn((1, config.hidden_size)))
            nn.init.uniform_(self.prompt_key, -1, 1)

            self.trans_input = nn.Sequential(
                nn.Linear(config.hidden_size, prompt_config["trans_hidden_dim"], bias=False),
                nn.SiLU(),
                nn.Linear(prompt_config["trans_hidden_dim"], config.hidden_size, bias=False),
                nn.SiLU(),
            )

            self.get_trans_feature = False
            self.stage_trans = 0
            self.matrix_trans_1 = torch.zeros(config.hidden_size, config.hidden_size)
            self.matrix_trans_2 = torch.zeros(prompt_config["trans_hidden_dim"], prompt_config["trans_hidden_dim"])
            self.n_trans_matrix = 0

            self.srt_router = None
            self.use_srt_routing = False
            self.srt_task_id_to_idx = {}
            self.all_attn_weights = []
            self.key_attention_weights = None
            self.encoder_frozen = None
            # Collect per-batch routing decisions so the caller can build summaries.
            self.srt_debug = True
            self._srt_debug_log = []

            self.previous_prompts_keys = None
            if prompt_config["previous_prompt_key_path"] is not None and prompt_config["task_id"]:
                print("----------Loading Previous Keys----------")
                loaded_previous_keys = torch.load(
                    prompt_config["previous_prompt_key_path"],
                    map_location='cpu',
                    weights_only=True,
                )
                if not isinstance(loaded_previous_keys, torch.Tensor):
                    loaded_previous_keys = torch.tensor(loaded_previous_keys)
                if loaded_previous_keys.ndim == 1:
                    loaded_previous_keys = loaded_previous_keys.unsqueeze(0)
                expected_shape = (prompt_config["task_id"], config.hidden_size)
                if tuple(loaded_previous_keys.shape) != expected_shape:
                    raise ValueError(
                        f"Loaded previous prompt keys have shape {tuple(loaded_previous_keys.shape)}, "
                        f"expected {expected_shape} from {prompt_config['previous_prompt_key_path']}"
                    )
                self.previous_prompts_keys = nn.Parameter(
                    loaded_previous_keys.detach().clone().to(dtype=self.prompt_key.dtype),
                    requires_grad=False,
                )

                self.previous_trans_input = Trans_input(config.hidden_size, prompt_config["trans_hidden_dim"], prompt_config["task_id"])
                for param in self.previous_trans_input.parameters():
                    param.requires_grad = False

        ##########################

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()


    def get_chunk(self, chunk):
        self.chunk_trans = chunk

        self.index_trans, self.step_trans = chunk, self.config.hidden_size // chunk
        self.step, self.index = self.step_trans, self.index_trans
        self.matrix_trans_1, self.matrix_trans_3, self.n_trans_matrix = {}, {}, {}
        for index in range(self.index_trans):
            self.matrix_trans_1[index] = torch.zeros(self.step_trans, self.step_trans).cuda()
            self.matrix_trans_3[index] = torch.zeros(self.step_trans, self.step_trans).cuda()
            self.n_trans_matrix[index] = 0
        # If model was created with meta tensors (init_empty_weights / device mapping),
        # copying a meta tensor to CUDA will raise "Cannot copy out of meta tensor".
        # Create a new CUDA tensor when the stored one is meta.
        try:
            is_meta = getattr(self.matrix_trans_2, "is_meta", False)
        except Exception:
            is_meta = False
        if is_meta:
            shape = tuple(self.matrix_trans_2.shape)
            dtype = getattr(self.matrix_trans_2, "dtype", torch.float32)
            self.matrix_trans_2 = torch.zeros(shape, device="cuda", dtype=dtype)
        else:
            self.matrix_trans_2 = self.matrix_trans_2.cuda()

    def get_matrix3(self, x, medium, x_final):
        for index in range(self.index_trans):
            if self.matrix_trans_1[index] is None:
                assert self.matrix_trans_3[index] is None
                self.matrix_trans_1[index] = torch.bmm(x[:,:,index*self.step_trans:(index+1)*self.step_trans].detach().permute(0, 2, 1), x[:,:,index*self.step_trans:(index+1)*self.step_trans].detach()).sum(dim=0).float()/(x.shape[0]*x.shape[1])

                self.matrix_trans_3[index] = torch.bmm(x_final[:,:,index*self.step_trans:(index+1)*self.step_trans].detach().permute(0, 2, 1), x_final[:,:,index*self.step_trans:(index+1)*self.step_trans].detach()).sum(dim=0).float()/(x_final.shape[0]*x_final.shape[1])
            else:
                self.matrix_trans_1[index] = (self.matrix_trans_1[index]*self.n_trans_matrix[index] + torch.bmm(x[:,:,index*self.step_trans:(index+1)*self.step_trans].detach().permute(0, 2, 1), x[:,:,index*self.step_trans:(index+1)*self.step_trans].detach()).sum(dim=0).float())/(self.n_trans_matrix[index] + x.shape[0]*x.shape[1])

                self.matrix_trans_3[index] = (self.matrix_trans_3[index]*self.n_trans_matrix[index] + torch.bmm(x_final[:,:,index*self.step_trans:(index+1)*self.step_trans].detach().permute(0, 2, 1), x_final[:,:,index*self.step_trans:(index+1)*self.step_trans].detach()).sum(dim=0).float())/(self.n_trans_matrix[index] + x_final.shape[0]*x_final.shape[1])
            self.n_trans_matrix[index] += x.shape[0]*x.shape[1]

        if self.matrix_trans_2 is None:
            self.matrix_trans_2 = torch.bmm(medium.detach().permute(0, 2, 1), medium.detach()).sum(dim=0).float()/(medium.shape[0]*medium.shape[1])
        else:
            self.matrix_trans_2 = (self.matrix_trans_2*self.n_trans_matrix[index] + torch.bmm(medium.detach().permute(0, 2, 1), medium.detach()).sum(dim=0).float())/(self.n_trans_matrix[index] + medium.shape[0]*medium.shape[1])

        return

    def _get_source_attention_mask(self, source_input_ids: torch.LongTensor) -> torch.LongTensor:
        pad_token_id = self.padding_idx if self.padding_idx is not None else 0
        return (source_input_ids != pad_token_id).long()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    # NOTE: cal_attention removed — SRT hard one-hot routing replaces it.
    # Kept as comment for ablation reference only.

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

        _cfg_return_dict = getattr(self.config, "return_dict", None)
        if _cfg_return_dict is None:
            _cfg_return_dict = getattr(self.config, "use_return_dict", True)
        return_dict = return_dict if return_dict is not None else _cfg_return_dict

        # retrieve input_ids and inputs_embeds
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

        # Convert DynamicCache (Transformers 5.x) to legacy tuple-of-tuples
        if DynamicCache is not None and isinstance(past_key_values, DynamicCache):
            if len(past_key_values.key_cache) > 0:
                past_key_values = tuple(
                    (past_key_values.key_cache[i], past_key_values.value_cache[i])
                    for i in range(len(past_key_values.key_cache))
                )
            else:
                past_key_values = None

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            if attention_mask is not None and attention_mask.dim() == 2:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                if past_key_values is not None:
                    position_ids = position_ids[:, -seq_length:]
            else:
                position_ids = torch.arange(
                    past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
                )
                position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
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

        # ═══════════ SRT HARD ONE-HOT ROUTING (replaces cal_attention) ═══════════
        key_attention_weights = None
        self.key_attention_weights = None
        if not self.prompt_config["run_single"]:
            source_input_ids = input_ids_wo_label if input_ids_wo_label is not None else input_ids
            source_attention_mask = self._get_source_attention_mask(source_input_ids)
            source_embeds = self.embed_tokens(source_input_ids)

            if self.get_trans_feature:
                masked_source = source_attention_mask.unsqueeze(-1).to(source_embeds.dtype) * source_embeds
                medium = self.trans_input[1](self.trans_input[0](masked_source.mean(dim=1, keepdim=True)))
                x = self.trans_input[3](self.trans_input[2](medium))
                self.get_matrix3(masked_source.mean(dim=1, keepdim=True), medium, x)

            n_prev = self.previous_prompts_keys.shape[0] if self.previous_prompts_keys is not None else 0
            n_slots = 1 + n_prev
            key_attention_weights = torch.zeros(
                batch_size, n_slots, 1,
                device=inputs_embeds.device, dtype=inputs_embeds.dtype,
            )

            if self.training:
                key_attention_weights[:, 0, 0] = 1.0
            elif self.use_srt_routing and self.srt_router is not None and self.encoder_frozen is not None:
                with torch.no_grad():
                    route_embeddings = self.encoder_frozen(source_input_ids, source_attention_mask)
                    route_inputs = route_embeddings.detach().float().cpu().numpy()

                srt_preds, _ = self.srt_router.route(route_inputs)
                for batch_idx, task_id in enumerate(srt_preds):
                    slot_idx = self.srt_task_id_to_idx.get(task_id, 0)
                    slot_idx = min(slot_idx, n_slots - 1)
                    key_attention_weights[batch_idx, slot_idx, 0] = 1.0

                if self.srt_debug:
                    slot_idxs = [
                        min(self.srt_task_id_to_idx.get(tid, 0), n_slots - 1)
                        for tid in srt_preds
                    ]

                    self._srt_debug_log.append({
                        "batch_size": batch_size,
                        "srt_preds": srt_preds.tolist() if hasattr(srt_preds, "tolist") else list(srt_preds),
                        "slot_idxs": slot_idxs,
                        "task_id_to_idx": dict(self.srt_task_id_to_idx),
                    })
            else:
                key_attention_weights[:, 0, 0] = 1.0

            self.key_attention_weights = key_attention_weights
            if self.is_inference:
                self.all_attn_weights.append(
                    key_attention_weights.squeeze().mean(dim=0, keepdim=True)
                    .detach().to(torch.float).cpu().numpy()
                )
        # ═══════════ END SRT ROUTING ═══════════

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(hidden_states, attention_mask, position_ids):
                        return module(
                            hidden_states,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            past_key_value=None,
                            output_attentions=output_attentions,
                            use_cache=False,
                            key_attention_weights=key_attention_weights,
                        )

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    use_reentrant=False,
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
                next_decoder_cache += (layer_outputs[2],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
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

        # Initialize weights and apply final processing
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

    # ── SRT debug log accessor ─────────────────────────────────
    def get_srt_debug_log(self):
        """Return SRT routing debug log and clear it."""
        log = getattr(self.model, '_srt_debug_log', [])
        self.model._srt_debug_log = []
        return log

    def reset_srt_debug_log(self):
        """Clear SRT routing debug log without returning."""
        self.model._srt_debug_log = []

    # NOTE: memory_replay removed — SRT replaces cal_attention-based replay.

    # ─────────────────────────────────────────────────────────────────────────
    #  OVERRIDE: generate — pass input_ids_wo_label for SRT inference routing
    # ─────────────────────────────────────────────────────────────────────────
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        input_ids_wo_label: Optional[torch.LongTensor] = None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=None,
        assistant_model=None,
        streamer=None,
        negative_prompt_ids=None,
        negative_prompt_attention_mask=None,
        **kwargs,
    ):
        """Pass input_ids_wo_label through to enable SRT routing in decoder layers."""
        # SRT status: print once per generate call
        core = self.model
        if hasattr(core, 'use_srt_routing') and core.use_srt_routing:
            router_ok = core.srt_router is not None
            enc_ok = core.encoder_frozen is not None
            n_tasks = len(getattr(core.srt_router, 'signatures', {})) if router_ok else 0
            if not hasattr(self, '_srt_gen_printed'):
                self._srt_gen_printed = True
                print(f"  [SRT-GEN] use_srt={core.use_srt_routing}  router={'OK' if router_ok else 'NONE'}  "
                      f"encoder={'OK' if enc_ok else 'NONE'}  n_tasks={n_tasks}")
        return super().generate(
            input_ids=input_ids,
            input_ids_wo_label=input_ids_wo_label,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            **kwargs,
        )

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
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        _cfg_return_dict = getattr(self.config, "return_dict", None)
        if _cfg_return_dict is None:
            _cfg_return_dict = getattr(self.config, "use_return_dict", True)
        return_dict = return_dict if return_dict is not None else _cfg_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
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
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            if not torch.isfinite(shift_logits).all():
                nan_count = int(torch.isnan(shift_logits).sum().item())
                inf_count = int(torch.isinf(shift_logits).sum().item())
                raise RuntimeError(
                    "Non-finite logits detected in LlamaForCausalLM.forward before CE loss: "
                    f"nan={nan_count}, inf={inf_count}, dtype={shift_logits.dtype}, "
                    f"shape={tuple(shift_logits.shape)}"
                )
            # Compute CE in fp32 for better stability under mixed precision.
            loss = loss_fct(shift_logits.float(), shift_labels)

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
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        input_ids_wo_label = kwargs.get("input_ids_wo_label", None)

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "input_ids_wo_label": input_ids_wo_label,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
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
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        _cfg_return_dict = getattr(self.config, "return_dict", None)
        if _cfg_return_dict is None:
            _cfg_return_dict = getattr(self.config, "use_return_dict", True)
        return_dict = return_dict if return_dict is not None else _cfg_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
