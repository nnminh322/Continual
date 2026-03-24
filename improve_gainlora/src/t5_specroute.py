# coding=utf-8
"""
SpecRoute V8: T5 model with Spectral-Geometric Routing for Continual LoRA Learning.

Core insight (Routing-Protection Duality): orthogonal subspace protection (GPM)
and task-discriminative routing are dual manifestations of the same spectral structure.
Solving one automatically solves the other.

V8 key design choices (from IDEA_Overall.md):
- C1: A_t rows ARE the spectral signatures — no SVD of B_t A_t needed
- C2: A-row affinity for BOTH training and inference (symmetric metric space)
- C5: Data-informed init via Constrained PCA in null-space (Q C_t Q eigenvectors)
- Removed: Trans_input, prompt_key, memory_replay, prepare_inference_routing()
- Removed: SVD sigma^2-weighted inference (train-inference mismatch source)
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

# Prefer magma backend for linalg ops (cusolver can crash on some GPU configs)
try:
    torch.backends.cuda.preferred_linalg_library("magma")
except Exception:
    pass

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
    Trans_input,
)

logger = logging.get_logger(__name__)


# ===================== Spectral Routing Functions =====================

# NOTE: _thin_svd_low_rank() was REMOVED in V8.
# Theorem 2 (C5 Routing Optimality) proves A_t rows with C5 init are already
# the optimal routing directions. SVD of B_t A_t only adds sigma^2-weighting
# from B optimization artifacts with no theoretical guarantee. See IDEA §3.6.


def compute_spectral_signatures(model, config):
    """
    V8: Store frozen A-row matrices as routing signatures (no SVD needed).

    Theorem 2 (C5 Routing Optimality) proves that A_t rows with C5 init are
    already the optimal routing directions in the null-space. SVD of B_t A_t
    only adds sigma^2-weighting from B optimization artifacts (no theoretical
    guarantee). A-row routing is exact, symmetric, and eliminates prepare_inference_routing().

    Returns dict mapping layer keys to {'A': tensor (r, d_model)}.
    """
    signatures = {}
    
    # V9 FIX 3: Retrieve the calibration scale E[fit] collected during training
    E_fit = getattr(model.encoder, 'current_fit_ema', 1.0)
    if E_fit < 1e-6:
        E_fit = 1.0
        
    with torch.no_grad():
        # Encoder layers
        for j in range(config.num_layers):
            attn = model.encoder.block[j].layer[0].SelfAttention
            for name, lora in [('q', attn.lora_q), ('v', attn.lora_v)]:
                A = lora.lora_A.data.float()
                signatures[f'enc.{j}.{name}'] = {
                    'A': A.cpu(),  # (r, d_model) — frozen routing directions
                    'E_fit': E_fit,
                }
        # Decoder self-attention layers
        for j in range(config.num_decoder_layers):
            attn = model.decoder.block[j].layer[0].SelfAttention
            for name, lora in [('q', attn.lora_q), ('v', attn.lora_v)]:
                A = lora.lora_A.data.float()
                signatures[f'dec.{j}.self.{name}'] = {
                    'A': A.cpu(),
                    'E_fit': E_fit,
                }
            # Decoder cross-attention layers
            attn_cross = model.decoder.block[j].layer[1].EncDecAttention
            for name, lora in [('q', attn_cross.lora_q), ('v', attn_cross.lora_v)]:
                A = lora.lora_A.data.float()
                signatures[f'dec.{j}.cross.{name}'] = {
                    'A': A.cpu(),
                    'E_fit': E_fit,
                }
    return signatures


# ===================== Modified T5Stack with Spectral Routing =====================

class T5Stack(T5PreTrainedModel):
    """
    T5Stack with spectral routing instead of learned gating.

    Instead of Trans_input + prompt_key (GainLoRA), routing weights are computed
    from A-row affinity between input embeddings and spectral signatures
    (A-row matrices of frozen LoRA A from previous tasks — V8, no SVD needed).
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
            self.routing_mode = prompt_config.get("routing_mode", "spectral")
            
            # Common for all spectral/grassmann modes
            self.spectral_signatures = []  # List[dict] — one dict per old task
            
            if self.routing_mode == "learned":
                # V10a: Learned routing matching GainLoRA ROOT exactly
                self.prompt_key = nn.Parameter(torch.randn((1, config.d_model)))
                nn.init.uniform_(self.prompt_key, -1, 1)

                self.trans_input = nn.Sequential(
                    nn.Linear(config.d_model, prompt_config["mlp_hidden_dim"], bias=False),
                    nn.SiLU(),
                    nn.Linear(prompt_config["mlp_hidden_dim"], config.d_model, bias=False),
                    nn.SiLU(),
                )

                self.get_trans_feature = False
                self.stage_trans = 0
                self.matrix_trans_1 = torch.zeros(config.d_model, config.d_model)
                self.matrix_trans_2 = torch.zeros(prompt_config["mlp_hidden_dim"], prompt_config["mlp_hidden_dim"])
                self.n_trans_matrix = 0

                self.previous_prompts_keys = None
                if prompt_config.get("previous_prompt_key_path") is not None and prompt_config.get("task_id", 0):
                    print("----------Loading Previous Keys----------")
                    self.previous_prompts_keys = nn.Parameter(torch.randn((prompt_config["task_id"], config.d_model)))
                    self.previous_prompts_keys.data = torch.load(prompt_config["previous_prompt_key_path"], weights_only=True)
                    self.previous_prompts_keys.requires_grad = False
                    
                    self.previous_trans_input = Trans_input(config.d_model, prompt_config["mlp_hidden_dim"], prompt_config["task_id"])
                    for param in self.previous_trans_input.parameters():
                        param.requires_grad = False
            else:
                # V8/V9/V10b: Spectral routing parameters
                self.routing_temperature = prompt_config.get('attn_temperature', 1.0)
                self._target_routing_alpha = prompt_config.get('target_routing_alpha', 0.8)

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

    # NOTE: _set_gradient_checkpointing removed intentionally.
    # The old format (with 'value' param) causes transformers to silently ignore
    # gradient_checkpointing_kwargs (including use_reentrant=False).

    def get_chunk(self, chunk):
        if self.routing_mode == "learned":
            self.chunk_trans = chunk
            self.index_trans, self.step_trans = chunk, self.config.d_model // chunk
            self.step, self.index = self.step_trans, self.index_trans
            self.matrix_trans_1, self.matrix_trans_3, self.n_trans_matrix = {}, {}, {}
            for idx in range(self.index_trans):
                self.matrix_trans_1[idx] = torch.zeros(self.step_trans, self.step_trans).cuda()
                self.matrix_trans_3[idx] = torch.zeros(self.step_trans, self.step_trans).cuda()
                self.n_trans_matrix[idx] = 0
            self.matrix_trans_2 = self.matrix_trans_2.cuda()

    def get_matrix3(self, x, medium, x_final):
        if self.routing_mode == "learned":
            for idx in range(self.index_trans):
                m1_curr = torch.bmm(x[:,:,idx*self.step_trans:(idx+1)*self.step_trans].detach().permute(0, 2, 1), x[:,:,idx*self.step_trans:(idx+1)*self.step_trans].detach()).sum(dim=0).float()/(x.shape[0]*x.shape[1])
                m3_curr = torch.bmm(x_final[:,:,idx*self.step_trans:(idx+1)*self.step_trans].detach().permute(0, 2, 1), x_final[:,:,idx*self.step_trans:(idx+1)*self.step_trans].detach()).sum(dim=0).float()/(x_final.shape[0]*x_final.shape[1])
                
                if len(self.matrix_trans_1) > 0 and isinstance(self.matrix_trans_1.get(idx), torch.Tensor) and self.matrix_trans_1.get(idx).sum() != 0:
                    self.matrix_trans_1[idx] = (self.matrix_trans_1[idx]*self.n_trans_matrix[idx] + m1_curr)/(self.n_trans_matrix[idx] + x.shape[0]*x.shape[1])
                    self.matrix_trans_3[idx] = (self.matrix_trans_3[idx]*self.n_trans_matrix[idx] + m3_curr)/(self.n_trans_matrix[idx] + x_final.shape[0]*x_final.shape[1])
                else:
                    self.matrix_trans_1[idx] = m1_curr
                    self.matrix_trans_3[idx] = m3_curr
                self.n_trans_matrix[idx] += x.shape[0]*x.shape[1]

            if self.matrix_trans_2.sum() == 0:
                self.matrix_trans_2 = torch.bmm(medium.detach().permute(0, 2, 1), medium.detach()).sum(dim=0).float()/(medium.shape[0]*medium.shape[1])
            else:
                self.matrix_trans_2 = (self.matrix_trans_2*self.n_trans_matrix[0] + torch.bmm(medium.detach().permute(0, 2, 1), medium.detach()).sum(dim=0).float())/(self.n_trans_matrix[0] + medium.shape[0]*medium.shape[1])

    def cal_attention(self, prompt_key, x, return_logits=False):
        # ROOT-style routing similarity
        # Force float32 to prevent underflow/overflow in fp16/bf16 on H100
        x_fp32 = x.float()
        prompt_key_fp32 = prompt_key.float()
        
        # Add epsilon to prevent div-by-zero, clamp to prevent extreme values
        x_norm = x_fp32 / (x_fp32.norm(dim=-1, keepdim=True).clamp_min(1e-12))
        prompt_key_norm = prompt_key_fp32 / (prompt_key_fp32.norm(dim=-1, keepdim=True).clamp_min(1e-12))
        
        attn_scores = (x_norm * prompt_key_norm).sum(dim=-1, keepdim=True)
        # Scaled sigmoid gating exact to ROOT GainLoRA
        weights_fp32 = torch.abs(torch.nn.functional.sigmoid(attn_scores * 4.0) * 2.0 - 1.0)
        
        weights = weights_fp32.to(x.dtype)
        attn_scores = attn_scores.to(x.dtype)
        
        if not return_logits:
            return weights
        else:
            return attn_scores

    def compute_learned_routing(self, avg_inputs_embeds, batch_size):
        """V10a: Learned MLP Routing copying GainLoRA exactly"""
        prompt_key = self.prompt_key
        if self.previous_prompts_keys is not None:
            prompt_key = self.prompt_key.to(prompt_key.device)
            past_prompt_key = torch.cat([prompt_key.repeat(batch_size, 1, 1), self.previous_prompts_keys.repeat(batch_size, 1, 1)], dim=1)

            medium = self.trans_input[1](self.trans_input[0](avg_inputs_embeds))
            x = self.trans_input[3](self.trans_input[2](medium))
            if getattr(self, "get_trans_feature", False):
                self.get_matrix3(avg_inputs_embeds, medium, x)

            past_x = torch.cat([x, self.previous_trans_input(avg_inputs_embeds)], dim=1)
            key_attention_weights = self.cal_attention(past_prompt_key, past_x)
        else:
            medium = self.trans_input[1](self.trans_input[0](avg_inputs_embeds))
            x = self.trans_input[3](self.trans_input[2](medium))
            if getattr(self, "get_trans_feature", False):
                self.get_matrix3(avg_inputs_embeds, medium, x)

            key_attention_weights = self.cal_attention(prompt_key.repeat(batch_size, 1, 1), x)
        return key_attention_weights

    def compute_grassmann_routing(self, h, h_norm_sq):
        """V10b: Grassmann Distance Routing
        Calculates principal angles between batch local subspace and candidate A_t subspaces.
        """
        B, _, d_model = h.shape
        if self.training or B < 8:
            # Fallback to A-row fit for very small batches or training (oracle handles training)
            return self.compute_spectral_routing(h, h_norm_sq)
            
        fits = []
        r = self.block[0].layer[0].SelfAttention.lora_q.r
        
        # Batch PCA to get local subspace U_batch (using SVD)
        # h is (B, 1, d_model) -> reshape to (B, d_model)
        h_flat = h.squeeze(1)
        # torch.linalg.svd returns (U, S, Vh) where Vh = V^T
        # We want right singular vectors V: h_flat = U @ diag(S) @ Vh, so V = Vh.T
        _, _, Vh_batch = torch.linalg.svd(h_flat - h_flat.mean(dim=0, keepdim=True), full_matrices=False)
        U_batch = Vh_batch[:r, :]  # Vh is (min(B,d), d), so first r rows = top-r right sing. vectors, shape (r, d_model)
        
        # Current task Grassmann dist
        current_layer_dists = []
        for block in self.block:
            attn = block.layer[0].SelfAttention
            for lora in [attn.lora_q, attn.lora_v]:
                A = lora.lora_A.data.float().to(h.device) # (r, d_model)
                # SVD of A^T: A^T = U_A @ diag(S_A) @ Vh_A => columns of U_A are right sing vecs of A
                _, _, Vh_A = torch.linalg.svd(A, full_matrices=False)  # A is (r, d_model), Vh_A is (r, d_model)
                U_A = Vh_A[:r, :]  # (r, d_model) — top-r right singular vectors of A, forming the subspace
                
                # Grassmann distance via principal angles
                # cos(theta_i) = singular values of U_batch @ U_A^T
                M = torch.matmul(U_batch, U_A.T)  # (r, r)
                angles = torch.linalg.svdvals(M).clamp(-1.0, 1.0)
                principal_angles = torch.acos(angles)
                dist = torch.sqrt(torch.sum(principal_angles**2))
                current_layer_dists.append(dist)
        
        current_dist = torch.stack(current_layer_dists).mean(dim=0).item()
        fits.append(1.0 / (current_dist + 1e-4)) # Inverse dist as affinity
        
        # Old tasks
        for sig_dict in self.spectral_signatures:
            task_dists = []
            for key, sig_data in sig_dict.items():
                if not key.startswith('enc.'):
                    continue
                A = sig_data['A'].to(h.device, dtype=torch.float32)  # (r, d_model)
                _, _, Vh_A = torch.linalg.svd(A, full_matrices=False)
                U_A = Vh_A[:r, :]
                
                M = torch.matmul(U_batch, U_A.T)
                angles = torch.linalg.svdvals(M).clamp(-1.0, 1.0)
                dist = torch.sqrt(torch.sum(torch.acos(angles)**2))
                task_dists.append(dist)
            
            if task_dists:
                task_dist = torch.stack(task_dists).mean(dim=0).item()
                fits.append(1.0 / (task_dist + 1e-4))
            else:
                fits.append(0.0)
                
        fit_scores = torch.tensor(fits, device=h.device).unsqueeze(0).repeat(B, 1) # (B, n_tasks)
        max_idx = fit_scores.argmax(dim=1, keepdim=True)
        weights = torch.zeros_like(fit_scores).scatter_(1, max_idx, 1.0)
        return weights.unsqueeze(2)

    def compute_spectral_routing(self, h, h_norm_sq):
        """
        V9: Routing with oracle-training / spectral-inference split + calibration.

        Theory (IDEA_Overall.md §3):
        - Lemma 1 (Differential Projection): ||A_t h||^2 = ||A_t Q_{t-1} h||^2
          → routing only sees null-space component, cross-task leakage <= 0.005*tr(C_s)/r
        - Corollary B (Reverse direction): alpha_s(h_t) = ||A_s P_{t-1} h_t||^2/(r||h_t||^2)
          → small when task t has most variance in null-space (different domain)
          → same-domain difficulty is a fundamental limit of ALL task-free CL methods
        - Theorem 2 (C5 Routing Optimality): C5 init maximizes E[alpha_t(h)]
          over all A_t in the restricted Stiefel manifold
        - C5 advantage over random: d'/r * PEV_r(C_tilde) ≈ 44x at task 8 (T5-small)

        Training: Oracle routing (current task always selected, index 0).
          Rationale: GPM-Routing paradox forces fit_current ≈ 0 on current task's data.
          If spectral argmax is used during training, B_t never receives gradients.
          Task ID is available at training time → oracle is valid, not cheating.

        Inference: Hard Top-1 spectral routing with calibration normalization.
          Calibration (FIX 3 EMA) normalizes per-task scale differences for fair argmax.

        Args:
            avg_inputs_embeds: (B, 1, d_model) — averaged input token embeddings

        Returns:
            (B, n_tasks, 1) routing weights: oracle one-hot (training) or top-1 (inference)
        """
        fits = []

        # === CURRENT TASK: A-row fit ===
        # Used for calibration EMA (training) and inference routing score.
        current_fits_layers = []
        for block in self.block:
            attn = block.layer[0].SelfAttention
            for lora in [attn.lora_q, attn.lora_v]:
                A = lora.lora_A.data.float()  # (r, d_model) — frozen after C5 init
                r = lora.r
                A_h = A.to(h.device, dtype=h.dtype)
                proj = torch.matmul(h, A_h.T)  # (B, 1, r)
                fit = (proj ** 2).sum(dim=-1) / (r * h_norm_sq)  # (B, 1)
                current_fits_layers.append(fit)
        current_fit = torch.stack(current_fits_layers).mean(dim=0)  # (B, 1)

        # V9 FIX 3: Calibration Normalization
        # Gather EMA during training to normalize the score
        if self.training:
            batch_mean = current_fit.mean().item()
            if not hasattr(self, 'current_fit_ema'):
                self.current_fit_ema = batch_mean
            else:
                self.current_fit_ema = 0.99 * self.current_fit_ema + 0.01 * batch_mean

        E_fit_current = getattr(self, 'current_fit_ema', 1.0)
        if E_fit_current < 1e-6:
            E_fit_current = 1.0
            
        calibrated_current_fit = current_fit / E_fit_current
        fits.append(calibrated_current_fit)

        # === OLD TASKS: A-row fit using saved A matrices ===
        # Lemma 1 (Differential Projection): ||A_t h||^2 = ||A_t Q_{t-1} h||^2
        # Cross-task leakage <= 0.005 * tr(C_s)/r with tau_GPM = 0.995
        for sig_dict in self.spectral_signatures:
            task_fits = []
            for key, sig_data in sig_dict.items():
                if not key.startswith('enc.'):
                    continue
                A = sig_data['A'].to(h.device, dtype=h.dtype)  # (r, d_model)
                r = A.shape[0]
                proj = torch.matmul(h, A.T)  # (B, 1, r)
                fit = (proj ** 2).sum(dim=-1) / (r * h_norm_sq)
                task_fits.append(fit)
            
            if task_fits:
                task_fit = torch.stack(task_fits).mean(dim=0)
                # V9 FIX 3: Apply Calibrated Normalization scale for the old task
                # Extract E_fit from the first available sig_data (it's the same for all layers)
                E_fit_old = list(sig_dict.values())[0].get('E_fit', 1.0)
                if E_fit_old < 1e-6:
                    E_fit_old = 1.0
                task_fit = task_fit / E_fit_old
            else:
                task_fit = torch.zeros(h.shape[0], 1, device=h.device, dtype=h.dtype)
            
            fits.append(task_fit)

        # Stack: (B, n_tasks) — all tasks use calibrated metric space
        fit_scores = torch.cat(fits, dim=1)  # (B, n_tasks)

        if self.training:
            # Oracle routing during training: always route to current task (index 0).
            # Rationale: during task-t training we have the oracle task label.
            # Using spectral argmax here would zero out B_t gradients whenever
            # fit_current < fit_old (which happens systematically due to GPM-Routing
            # paradox: GPM forces A_t ⊥ h_t, so fit_t ≈ 0 even on task-t data).
            # Oracle routing is NOT cheating — task ID is always available at training
            # time in continual learning (GainLoRA also uses task labels to train routing).
            weights = torch.zeros_like(fit_scores)
            weights[:, 0] = 1.0
        else:
            # Hard Top-1 at inference: no task label available, use calibrated spectral routing.
            # Calibration normalization (FIX 3) ensures fair comparison across tasks
            # despite systematic scale differences in A-row fit magnitudes.
            max_idx = fit_scores.argmax(dim=1, keepdim=True)
            weights = torch.zeros_like(fit_scores).scatter_(1, max_idx, 1.0)

        return weights.unsqueeze(2)  # (B, n_tasks, 1)

    # prepare_inference_routing() REMOVED in V8.
    # Reasoning: Lemma 1 + Theorem 2 prove A_t rows (with C5 init) are already the
    # optimal routing directions. SVD of B_t A_t only adds sigma^2-weighting from
    # B optimization artifacts, causing train-inference mismatch. A_t is the signature.
    # Eliminates O(d*r^2) SVD overhead per task per layer after training.

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
        if not self.is_decoder:
            # Properly masked mean of input embeddings
            _mask_count = attention_mask.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)  # (B,1,1)
            avg_inputs_embeds = (attention_mask.unsqueeze(-1) * inputs_embeds).sum(dim=1, keepdim=True) / _mask_count

        if not self.is_decoder and not self.prompt_config["run_single"]:
            if self.routing_mode == "learned":
                key_attention_weights = self.compute_learned_routing(avg_inputs_embeds, batch_size)
            else:
                if len(self.spectral_signatures) > 0:
                    h_norm_sq = (avg_inputs_embeds ** 2).sum(dim=-1) + 1e-8  # (B, 1)
                    if self.routing_mode == "grassmann":
                        key_attention_weights = self.compute_grassmann_routing(avg_inputs_embeds, h_norm_sq)
                    else:
                        key_attention_weights = self.compute_spectral_routing(avg_inputs_embeds, h_norm_sq)
                else:
                    # First task: expert is always 1.0
                    key_attention_weights = torch.ones(
                        batch_size, 1, 1, device=inputs_embeds.device, dtype=inputs_embeds.dtype
                    )

            # ROUTING SELECTION during training:
            if self.training:
                if self.routing_mode == "learned":
                    # For learned mode, use the network weights so it can update via backprop (ROOT behavior)
                    # We rely on non-zero initialization and fp32 stability to avoid the zero-gradient trap.
                    key_attention_weights = key_attention_weights
                else:
                    # For spectral/grassmann, use Oracle current task (index 0)
                    oracle_weights = torch.zeros_like(key_attention_weights)
                    oracle_weights[:, 0, 0] = 1.0
                    key_attention_weights = key_attention_weights * 0.0 + oracle_weights
            # In inference or non-learned training, we detach to avoid 
            # "backward through graph second time" error with checkpointing.
            if self.routing_mode != "learned":
                key_attention_weights = key_attention_weights.detach()

            # Logging weights during inference
            if self.is_inference:
                if key_attention_weights.shape[1] > 1:
                    self.all_attn_weights.append(key_attention_weights.squeeze().mean(dim=0, keepdim=True).detach().to(torch.float).cpu().numpy())
                else:
                    self.all_attn_weights.append(key_attention_weights.squeeze(2).mean(dim=0, keepdim=True).detach().to(torch.float).cpu().numpy())

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
                # Use _gradient_checkpointing_func (set by new-format
                # gradient_checkpointing_enable) if available, else fallback
                gc_fn = getattr(self, '_gradient_checkpointing_func', None)
                if gc_fn is not None:
                    layer_outputs = gc_fn(
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
                        use_reentrant=False,
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

    # NOTE: _set_gradient_checkpointing removed intentionally.
    # The old format causes transformers to silently ignore gradient_checkpointing_kwargs.

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

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, prompt_config, *model_args, **kwargs):
        """Custom loader to handle T5Stack and prefix-based state_dict mapping."""
        # Force strict=False in kwargs if not present
        kwargs.pop("strict", None)
        
        # 1. Initialize model skeleton
        config = kwargs.get("config", T5Config.from_pretrained(pretrained_model_name_or_path))
        model = cls(config, prompt_config)
        
        # 2. Determine weights file path using transformers utility
        from transformers.utils import cached_file
        import os
        
        try:
            if os.path.isdir(pretrained_model_name_or_path):
                # Local directory
                if os.path.exists(os.path.join(pretrained_model_name_or_path, 'pytorch_model.bin')):
                    weights_path = os.path.join(pretrained_model_name_or_path, 'pytorch_model.bin')
                else:
                    weights_path = os.path.join(pretrained_model_name_or_path, 'model.safetensors')
            else:
                # Hub model
                try:
                    weights_path = cached_file(pretrained_model_name_or_path, 'pytorch_model.bin')
                except:
                    weights_path = cached_file(pretrained_model_name_or_path, 'model.safetensors')
        except Exception as e:
            print(f"[SpecRoute] Error finding weights for {pretrained_model_name_or_path}: {e}")
            return model

        # 3. Load state_dict
        if weights_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(weights_path, device='cpu')
        else:
            state_dict = torch.load(weights_path, weights_only=True, map_location='cpu')
        
        # 4. Correctly map state_dict keys to model members
        # tied weights usually use 'shared.weight'
        if 'shared.weight' in state_dict:
            model.shared.weight.data.copy_(state_dict['shared.weight'])
        elif 'encoder.embed_tokens.weight' in state_dict:
            # Fallback if names are different
            model.shared.weight.data.copy_(state_dict['encoder.embed_tokens.weight'])
            
        if 'lm_head.weight' in state_dict:
            model.lm_head.weight.data.copy_(state_dict['lm_head.weight'])
            
        # 5. Extract and load encoder/decoder stacks after stripping prefixes
        # Standard T5 has 'encoder.block.0...' but we need 'block.0...'
        encoder_state = {k[len('encoder.'):]: v for k, v in state_dict.items() if k.startswith('encoder.')}
        decoder_state = {k[len('decoder.'):]: v for k, v in state_dict.items() if k.startswith('decoder.')}
        
        if not encoder_state: 
            # If state_dict already flat (unlikely for T5-small hub but possible for custom checkpoints)
            encoder_state = state_dict
        if not decoder_state: 
            decoder_state = state_dict
        
        # Load with strict=False to allow for LoRA params missing in base file
        load_info_enc = model.encoder.load_state_dict(encoder_state, strict=False)
        load_info_dec = model.decoder.load_state_dict(decoder_state, strict=False)
        
        print(f"[SpecRoute] Weights loaded from {weights_path}")
        print(f"  Encoder missing keys: {len(load_info_enc.missing_keys)} (expected: {model.prompt_config['lora_r']*2*model.config.num_layers*2 if not model.prompt_config['run_single'] else 0})")
        
        return model

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
