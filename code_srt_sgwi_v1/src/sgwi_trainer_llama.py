"""
SGWI + Dual Fisher Trainer for LLaMA (Contribution 2)
=====================================================
Extends GainLoRA_OLoRA_Trainer (LLaMA) with SRT routing + SGWI warm-init + Dual Fisher.

Key difference from T5 version:
  - Model core is self.model.model (LlamaModel) instead of self.model.encoder (T5Stack)
  - SRT attributes wired to self.model.model.*
  - Embedding extraction uses Case 4 (LlamaForCausalLM → hidden_states[-1])
"""

import os
import math
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Union, Any
from torch.utils.data import DataLoader

from cl_trainer_gainlora_llama import GainLoRA_OLoRA_Trainer
from srt_router import SRTRouter
from cl_trainer_srt import extract_embeddings_from_batch

logger = logging.getLogger(__name__)


class SGWI_DualFisher_LLaMA_Trainer(GainLoRA_OLoRA_Trainer):
    """
    GainLoRA_OLoRA_Trainer (LLaMA GPM) + SRT Router + SGWI + Dual Fisher.
    
    Combines:
      1. GPM gradient projection from GainLoRA_OLoRA_Trainer
      2. SRT non-parametric routing (replaces cal_attention)
      3. SGWI warm initialization of LoRA (from similar past tasks)
      4. Dual Fisher embedding regularization
    """

    SGWI_MODES = ('full_lora', 'sgwi_full', 'sgwi_freeze_a', 'sgwi_train_a', 'inflora', 'random')

    def __init__(
        self,
        model,
        args,
        train_dataset,
        cur_task_id: int,
        task_order: list,
        # SGWI params
        sgwi_mode: str = 'full_lora',
        lambda_emb: float = 0.0,
        # SRT params
        srt_metric_mode: str = 'hard',
        srt_shrink: bool = True,
        srt_shrink_factor: float = 0.1,
        srt_max_emb_samples: int = 500,
        srt_load_path: Optional[str] = None,
        srt_skip_forward: bool = False,
        # Standard trainer params
        data_collator_replay=None,
        replay_dataset_dict=None,
        replay_label_dict=None,
        eval_dataset=None,
        tokenizer=None,
        data_collator=None,
        compute_metrics=None,
        callbacks=None,
        **kwargs,
    ):
        super().__init__(
            model, args, train_dataset, cur_task_id, task_order,
            data_collator_replay, replay_dataset_dict, replay_label_dict,
            eval_dataset, tokenizer, data_collator, compute_metrics, callbacks,
        )

        # ── SRT state ──
        self.srt_metric_mode = srt_metric_mode
        self.srt_shrink = srt_shrink
        self.srt_shrink_factor = srt_shrink_factor
        self.srt_max_emb_samples = srt_max_emb_samples
        self.srt_load_path = srt_load_path
        self.srt_skip_forward = srt_skip_forward
        self.srt_router: Optional[SRTRouter] = None
        self._srt_init()

        # ── SGWI state ──
        self.sgwi_mode = sgwi_mode
        self.lambda_emb = lambda_emb
        self._emb_anchor: Optional[torch.Tensor] = None

        if sgwi_mode not in self.SGWI_MODES:
            raise ValueError(f"sgwi_mode must be one of {self.SGWI_MODES}, got '{sgwi_mode}'")

        print(f"[SGWI-LLaMA] mode={sgwi_mode}, lambda_emb={lambda_emb}, "
              f"srt_mode={srt_metric_mode}, cur_task={cur_task_id}")

    # ─────────────────────────────────────────────────────────────────────────
    #  MODEL CORE ACCESSOR (LLaMA = self.model.model)
    # ─────────────────────────────────────────────────────────────────────────
    def _get_core(self):
        """Return the core model: LlamaModel (decoder)."""
        return self.model.model

    # ─────────────────────────────────────────────────────────────────────────
    #  SRT METHODS (adapted from cl_trainer_srt.py for LLaMA)
    # ─────────────────────────────────────────────────────────────────────────

    def _srt_init(self):
        if self.srt_router is not None:
            return
        self.srt_router = SRTRouter(
            srt_metric_mode=self.srt_metric_mode,
            use_shrink=self.srt_shrink,
            shrink_factor=self.srt_shrink_factor,
        )
        if self.srt_load_path is not None:
            self.load_srt_signatures(self.srt_load_path, wire_model=True)

    def _srt_emb_cache_path(self, task_name: str) -> Optional[str]:
        model_name = getattr(self.args, 'model_name_or_path', '') or ''
        backbone = None
        if 'llama-3' in model_name.lower():
            backbone = 'Meta-Llama-3-8B'
        elif 'llama-2' in model_name.lower():
            backbone = 'Llama-2-7b-hf'
        elif 'flan-t5' in model_name.lower():
            backbone = 'flan-t5-large'
        else:
            return None
        split = 'SuperNI' if task_name.startswith('task') else 'Long_Sequence'
        root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'embeddings')
        return os.path.join(root, backbone, split, task_name, 'train.npz')

    def _extract_task_embeddings(self, max_samples: int = None):
        if max_samples is None:
            max_samples = self.srt_max_emb_samples

        if self.srt_skip_forward:
            cur_task = self.task_order[self.cur_task_id]
            emb_path = self._srt_emb_cache_path(cur_task)
            if emb_path is not None and os.path.exists(emb_path):
                print(f"  [SRT-LLaMA] ★ LOAD FROM CACHE: {emb_path}")
                data = np.load(emb_path, allow_pickle=True)
                embeddings = torch.from_numpy(data['embeddings'])
                if embeddings.shape[0] > max_samples:
                    embeddings = embeddings[:max_samples]
                print(f"  [SRT-LLaMA]   → {embeddings.shape[0]} embeddings, dim={embeddings.shape[1]}")
                return embeddings, []
            else:
                print(f"  [SRT-LLaMA] ✗ Cache miss for '{cur_task}' → forward extraction")

        print(f"  [SRT-LLaMA] → Forward extraction ({max_samples} batches)")
        train_dataloader = self.get_train_dataloader()
        h_list, task_ids = [], []
        max_batches = min(max_samples, len(train_dataloader))
        for step, inputs in enumerate(train_dataloader):
            if step >= max_batches:
                break
            inputs = self._prepare_inputs(inputs)
            h = extract_embeddings_from_batch(self.model, inputs)
            h_list.append(h.cpu())
        if not h_list:
            return torch.empty(0), task_ids
        return torch.cat(h_list, dim=0), task_ids

    def _compute_and_store_signature(self, task_id):
        if self.srt_router is None:
            self._srt_init()
        h_train, _ = self._extract_task_embeddings(self.srt_max_emb_samples)
        if h_train.shape[0] == 0:
            print(f"  [SRT-LLaMA] WARNING: no embeddings for task {task_id}")
            return
        sig = self.srt_router.add_task(task_id=task_id, h_train=h_train.numpy())
        print(f"  [SRT-LLaMA] Task {task_id}: PaR={sig.par:.1f}, metric={sig.metric}, n={sig.n}")

    def _replace_attention_routing(self):
        """Wire SRT router into LlamaModel (self.model.model)."""
        if self.srt_router is None or len(self.srt_router.signatures) == 0:
            return
        core = self._get_core()
        current_task = self.task_order[self.cur_task_id]
        task_id_to_idx = {current_task: 0}
        for prev_idx in range(self.cur_task_id):
            prev_task = self.task_order[prev_idx]
            task_id_to_idx[prev_task] = self.cur_task_id - prev_idx
        core.srt_router = self.srt_router
        core.srt_task_id_to_idx = task_id_to_idx
        core.use_srt_routing = True
        print(f"  [SRT-LLaMA] Wired router: {len(self.srt_router.signatures)} tasks, mapping={task_id_to_idx}")

    def on_task_end(self, task_id):
        self._compute_and_store_signature(task_id)
        self._replace_attention_routing()

    def save_srt_signatures(self, output_dir: str):
        if self.srt_router is None:
            return
        path = os.path.join(output_dir, 'srt_signatures.npz')
        self.srt_router.save(path)
        print(f"  [SRT-LLaMA] Saved signatures to {path}")

    def load_srt_signatures(self, checkpoint_dir: str, wire_model: bool = False):
        path = os.path.join(checkpoint_dir, 'srt_signatures.npz')
        if not os.path.exists(path):
            print(f"  [SRT-LLaMA] No signatures at {path}")
            return
        self.srt_router.load(path)
        print(f"  [SRT-LLaMA] Loaded {len(self.srt_router.signatures)} signatures")
        if wire_model and len(self.srt_router.signatures) > 0:
            self._replace_attention_routing()

    # ─────────────────────────────────────────────────────────────────────────
    #  SGWI WARM INITIALIZATION
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_sgwi_weights(self) -> Optional[Dict[str, float]]:
        """
        Compute softmax weights over SRT distances to past tasks.

        Uses CENTROID distance (consistent with T5 sgwi_trainer.py):
          d = ||μ_cur - μ_s||  (L2 distance of means, NOT mean of per-sample distances)

        The current task's centroid μ_cur is computed from embeddings.
        Each past task's centroid μ_s is stored in srt_router.signatures[task].
        SRT routing already whitens embeddings → L2 distance is valid.
        """
        if self.srt_router is None or len(self.srt_router.signatures) == 0:
            return None
        h_cur, _ = self._extract_task_embeddings(min(200, self.srt_max_emb_samples))
        if h_cur.shape[0] == 0:
            return None

        # Compute centroid of current task
        mu_cur = h_cur.mean(axis=0).numpy()  # (d,)

        # Compute L2 distance from μ_cur to each past task's centroid
        task_ids = list(self.srt_router.signatures.keys())
        centroid_dists = np.zeros(len(task_ids), dtype=np.float64)
        for i, task_id in enumerate(task_ids):
            sig = self.srt_router.signatures[task_id]
            mu_s = sig.mu_raw if hasattr(sig, 'mu_raw') and sig.mu_raw is not None else sig.mu
            centroid_dists[i] = np.linalg.norm(mu_cur - mu_s)

        tau = max(np.median(centroid_dists), 1e-6)
        scores = np.exp(-centroid_dists / tau)
        scores /= scores.sum()
        weights = {task_ids[i]: float(scores[i]) for i in range(len(task_ids))}
        print(f"  [SGWI-LLaMA] weights (τ={tau:.4f}): {weights}")
        return weights

    def _sgwi_init_a(self):
        """Warm-init LoRA A from weighted past tasks' ΔW."""
        if self.cur_task_id == 0:
            return
        weights = self._compute_sgwi_weights()
        if weights is None:
            return
        core = self._get_core()
        for layer in core.layers:
            attn = layer.self_attn
            if attn.previous_lora_weights_q is None:
                continue
            for proj, prev_list in [('q', attn.previous_lora_weights_q), ('v', attn.previous_lora_weights_v)]:
                cur_lora = attn.lora_q if proj == 'q' else attn.lora_v
                fused_dw = torch.zeros_like(cur_lora.lora_B @ cur_lora.lora_A)
                total_w = 0.0
                for past_task_name, w in weights.items():
                    # prev_list index: prev_list[0] = task_order[cur_task_id-1] (most recent),
                    # prev_list[1] = task_order[cur_task_id-2], ..., prev_list[n-1] = task_order[0] (oldest)
                    # → idx = (cur_task_id-1) - position_of(past_task_name in task_order)
                    if isinstance(past_task_name, str):
                        try:
                            pos = self.task_order.index(past_task_name)
                            prev_idx = (self.cur_task_id - 1) - pos
                        except ValueError:
                            print(f"  [SGWI-LLaMA] task '{past_task_name}' not in task_order, skipping")
                            continue
                    else:
                        prev_idx = int(past_task_name)

                    if prev_idx < 0 or prev_idx >= len(prev_list):
                        print(f"  [SGWI-LLaMA] idx={prev_idx} out of range, skipping")
                        continue

                    prev_lora = prev_list[prev_idx]
                    if w < 1e-8:
                        continue
                    dw = prev_lora.lora_B.data @ prev_lora.lora_A.data
                    fused_dw += w * dw
                    total_w += w
                if total_w < 1e-8:
                    continue
                fused_dw /= total_w
                try:
                    U, S, Vh = torch.linalg.svd(fused_dw, full_matrices=False)
                    r = cur_lora.lora_A.shape[0]
                    cur_lora.lora_A.data = (torch.diag(S[:r].sqrt()) @ Vh[:r]).to(cur_lora.lora_A.dtype)
                    cur_lora.lora_B.data = (U[:, :r] @ torch.diag(S[:r].sqrt())).to(cur_lora.lora_B.dtype)
                except Exception as e:
                    print(f"  [SGWI-LLaMA] SVD failed for {proj}: {e}")
        print(f"  [SGWI-LLaMA] LoRA A/B warm-initialized from {len(weights)} past tasks")

    # ─────────────────────────────────────────────────────────────────────────
    #  DUAL FISHER REGULARIZATION
    # ─────────────────────────────────────────────────────────────────────────

    def _snapshot_embeddings(self):
        """Snapshot current embedding state as anchor for Dual Fisher."""
        if self.lambda_emb <= 0:
            return
        core = self._get_core()
        self._emb_anchor = core.embed_tokens.weight.data.clone()
        print(f"  [DualFisher-LLaMA] Snapshot embedding anchor (shape={self._emb_anchor.shape})")

    def _dual_fisher_loss(self) -> torch.Tensor:
        """L2 penalty between current and anchored embeddings."""
        if self._emb_anchor is None or self.lambda_emb <= 0:
            return torch.tensor(0.0)
        core = self._get_core()
        diff = core.embed_tokens.weight - self._emb_anchor.to(core.embed_tokens.weight.device)
        return self.lambda_emb * diff.pow(2).sum()

    # ─────────────────────────────────────────────────────────────────────────
    #  OVERRIDE: get_reg_matrix — apply SGWI before GPM
    # ─────────────────────────────────────────────────────────────────────────

    def get_reg_matrix(self):
        """GPM init + SGWI warm-init + Dual Fisher snapshot."""
        # SGWI: warm-init LoRA before GPM projects
        if self.sgwi_mode.startswith('sgwi'):
            self._sgwi_init_a()

        # Call parent GPM initialization
        super().get_reg_matrix()

        # Dual Fisher: snapshot embeddings after GPM
        self._snapshot_embeddings()

    # ─────────────────────────────────────────────────────────────────────────
    #  OVERRIDE: compute_loss — add Dual Fisher penalty
    # ─────────────────────────────────────────────────────────────────────────

    def compute_loss(self, model, inputs, return_outputs=False):
        """Add Dual Fisher L2 regularization to the standard loss."""
        if return_outputs:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        else:
            loss = super().compute_loss(model, inputs, return_outputs=False)
            outputs = None

        # Add Dual Fisher penalty
        if self.lambda_emb > 0 and self._emb_anchor is not None:
            fisher_loss = self._dual_fisher_loss()
            loss = loss + fisher_loss

        if return_outputs:
            return loss, outputs
        return loss
