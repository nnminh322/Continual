# coding=utf-8
"""
SpecRoute: T5 model with Spectral-Geometric Routing for Continual LoRA Learning.

Replaces GainLoRA's learned gating (Trans_input + prompt_key) with spectral
projection-based routing using SVD signatures from frozen LoRA weights.

Key changes from t5_gainlora_inflora.py:
- Removed: Trans_input, prompt_key, previous_trans_input, previous_prompts_keys
- Added: Spectral signatures + projection-based routing (softmax over weighted projection fits)
- Removed: memory_replay (no KL loss on gating since routing is parameter-based)
"""

import copy
import math
import os
import warnings
import numpy as np
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint

from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.utils import (
    is_torch_fx_proxy,
    logging,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.models.t5.configuration_t5 import T5Config

# Import all unchanged classes from the original GainLoRA model
from t5_gainlora_inflora import (
    LoRALayer,
    T5LayerNorm,
    T5DenseActDense,
    T5DenseGatedActDense,
    T5LayerFF,
    T5Attention,
    T5LayerSelfAttention,
    T5LayerCrossAttention,
    T5Block,
    T5PreTrainedModel,
)

logger = logging.get_logger(__name__)


# ===================== Spectral Routing Functions =====================

def compute_spectral_signatures(model, config):
    """
    Compute spectral signatures from all LoRA branches after training.
    For each LoRA layer, computes SVD of B@A and stores top-r right singular
    vectors (input directions) and singular values (importance).

    Returns dict mapping layer keys to {'V': tensor, 'sigma': tensor}.
    """
    signatures = {}
    with torch.no_grad():
        # Encoder layers
        for j in range(config.num_layers):
            attn = model.encoder.block[j].layer[0].SelfAttention
            for name, lora in [('q', attn.lora_q), ('v', attn.lora_v)]:
                A = lora.lora_A.data.float()  # (r, d_model)
                B = lora.lora_B.data.float()  # (inner_dim, r)
                delta_W = B @ A  # (inner_dim, d_model)
                U, S, Vt = torch.linalg.svd(delta_W, full_matrices=False)
                r = lora.r
                signatures[f'enc.{j}.{name}'] = {
                    'V': Vt[:r].cpu(),     # (r, d_model)
                    'sigma': S[:r].cpu()   # (r,)
                }
        # Decoder self-attention layers
        for j in range(config.num_decoder_layers):
            attn = model.decoder.block[j].layer[0].SelfAttention
            for name, lora in [('q', attn.lora_q), ('v', attn.lora_v)]:
                A = lora.lora_A.data.float()
                B = lora.lora_B.data.float()
                delta_W = B @ A
                U, S, Vt = torch.linalg.svd(delta_W, full_matrices=False)
                r = lora.r
                signatures[f'dec.{j}.self.{name}'] = {
                    'V': Vt[:r].cpu(),
                    'sigma': S[:r].cpu()
                }
            # Decoder cross-attention layers
            attn_cross = model.decoder.block[j].layer[1].EncDecAttention
            for name, lora in [('q', attn_cross.lora_q), ('v', attn_cross.lora_v)]:
                A = lora.lora_A.data.float()
                B = lora.lora_B.data.float()
                delta_W = B @ A
                U, S, Vt = torch.linalg.svd(delta_W, full_matrices=False)
                r = lora.r
                signatures[f'dec.{j}.cross.{name}'] = {
                    'V': Vt[:r].cpu(),
                    'sigma': S[:r].cpu()
                }
    return signatures


# ===================== Modified T5Stack with Spectral Routing =====================

class T5Stack(T5PreTrainedModel):
    """
    T5Stack with spectral routing instead of learned gating.

    Instead of Trans_input + prompt_key (GainLoRA), routing weights are computed
    from projection fits between input embeddings and spectral signatures
    (SVD of frozen LoRA weights from previous tasks).
    """

    def __init__(self, config, embed_tokens=None, prompt_config=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        self.model_dim = config.d_model
        self.is_inference = False

        self.block = nn.ModuleList(
            [T5Block(config, prompt_config, has_relative_attention_bias=bool(i == 0))
             for i in range(config.num_layers)]
        )

        self.prompt_config = prompt_config

        if not self.is_decoder and not prompt_config["run_single"]:
            # ===== Spectral routing: NO learned parameters for routing =====
            # Spectral signatures loaded from previous tasks' saved weights
            self.spectral_signatures = []  # List[dict] — one dict per old task
            self.routing_temperature = 1.0

            # For inference logging
            self.all_attn_weights = []
            self.key_attention_weights = None

        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    def compute_spectral_routing(self, avg_inputs_embeds):
        """
        Compute routing weights using spectral projection fits.

        For each task, measures how much of the input's energy falls into that
        task's operating subspace (defined by SVD of its LoRA weights).

        Args:
            avg_inputs_embeds: (B, 1, d_model) — averaged input token embeddings

        Returns:
            (B, n_tasks, 1) routing weights via softmax over projection fits
        """
        h = avg_inputs_embeds  # (B, 1, d_model)
        h_norm_sq = (h ** 2).sum(dim=-1) + 1e-8  # (B, 1)

        fits = []

        # 1. Current task fit: use current LoRA A rows (frozen in InfLoRA)
        # Average over all encoder self-attention LoRA layers
        current_fits_layers = []
        for block in self.block:
            attn = block.layer[0].SelfAttention
            for lora in [attn.lora_q, attn.lora_v]:
                A = lora.lora_A.data  # (r, d_model)
                proj = torch.matmul(h, A.T.to(h.dtype))  # (B, 1, r)
                r = A.shape[0]
                fit = (proj ** 2).sum(dim=-1) / (r * h_norm_sq)  # (B, 1)
                current_fits_layers.append(fit)
        current_fit = torch.stack(current_fits_layers).mean(dim=0)  # (B, 1)
        fits.append(current_fit)

        # 2. Previous tasks fit: use spectral signatures (V, sigma)
        for sig_dict in self.spectral_signatures:
            task_fits = []
            for key, sig_data in sig_dict.items():
                if not key.startswith('enc.'):
                    continue  # Only use encoder signatures for routing
                V = sig_data['V'].to(h.device, dtype=h.dtype)       # (r, d_model)
                sigma = sig_data['sigma'].to(h.device, dtype=h.dtype)  # (r,)

                proj = torch.matmul(h, V.T)  # (B, 1, r)
                sigma_sq = sigma ** 2
                sigma_sq_sum = sigma_sq.sum() + 1e-8
                weighted_proj = (proj ** 2 * sigma_sq.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # (B, 1)
                fit = weighted_proj / (sigma_sq_sum * h_norm_sq)
                task_fits.append(fit)

            if task_fits:
                task_fit = torch.stack(task_fits).mean(dim=0)  # (B, 1)
            else:
                task_fit = torch.zeros_like(current_fit)
            fits.append(task_fit)

        # Stack: (B, n_tasks)
        fit_scores = torch.cat(fits, dim=1)  # (B, n_tasks)

        # Softmax routing with temperature
        weights = torch.softmax(fit_scores / self.routing_temperature, dim=1)  # (B, n_tasks)

        return weights.unsqueeze(2)  # (B, n_tasks, 1)

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.block), range(torch.cuda.device_count()))
            if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.block))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        for k, v in self.device_map.items():
            for layer in v:
                cuda_device = "cuda:" + str(k)
                self.block[layer] = self.block[layer].to(cuda_device)
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)

    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")
        self.embed_tokens = self.embed_tokens.to("cpu")
        self.final_layer_norm = self.final_layer_norm.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        key_attention_weights=None,
    ):
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError("You have to initialize the model with valid token embeddings")
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # ============ SPECTRAL ROUTING ============
        self.key_attention_weights = None
        if not self.is_decoder and not self.prompt_config["run_single"]:
            if len(self.spectral_signatures) > 0:
                # Multi-task: compute spectral routing
                avg_inputs_embeds = (attention_mask.unsqueeze(-1) * inputs_embeds).mean(dim=1, keepdim=True)
                key_attention_weights = self.compute_spectral_routing(avg_inputs_embeds)

                if self.is_inference:
                    self.all_attn_weights.append(
                        key_attention_weights.squeeze().mean(dim=0, keepdim=True).detach().to(torch.float).cpu().numpy()
                    )
            else:
                # First task: single LoRA, weight = 1
                key_attention_weights = torch.ones(
                    batch_size, 1, 1, device=inputs_embeds.device, dtype=inputs_embeds.dtype
                )
                if self.is_inference:
                    self.all_attn_weights.append(
                        key_attention_weights.squeeze(2).mean(dim=0, keepdim=True).detach().to(torch.float).cpu().numpy()
                    )
            self.key_attention_weights = key_attention_weights
        else:
            # Decoder or run_single: use whatever was passed (from encoder)
            key_attention_weights = key_attention_weights

        # ============ REST OF FORWARD PASS (identical to original) ============
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            if not self.is_decoder:
                raise ValueError(f"`use_cache` can only be set to `True` if {self} is used as a decoder")

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))
                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    key_attention_weights=key_attention_weights
                )

            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ] if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


# ===================== Modified T5ForConditionalGeneration =====================

class T5ForConditionalGeneration(T5PreTrainedModel):
    """
    T5ForConditionalGeneration with SpecRoute.
    Same as GainLoRA version except routing is spectral-based (no memory_replay).
    """
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config: T5Config, prompt_config):
        super().__init__(config)
        self.model_dim = config.d_model
        self.prompt_config = prompt_config
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared, prompt_config)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared, prompt_config)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        start_fusion: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        # Encode
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(labels)

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode — pass encoder's routing weights to decoder
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            key_attention_weights=self.encoder.key_attention_weights
        )

        sequence_output = decoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None,
        head_mask=None, decoder_head_mask=None, decoder_attention_mask=None,
        cross_attn_head_mask=None, use_cache=None, encoder_outputs=None, **kwargs,
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past_key_values, beam_idx):
        if past_key_values is None:
            return past_key_values
        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )
            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past
