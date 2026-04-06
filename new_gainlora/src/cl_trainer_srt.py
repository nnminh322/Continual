"""
SRT Trainer: GainLoRA + SRT Router integration.

Implements CONTRIBUTION_1 (SRT) routing within GainLoRA architecture.
  - Computes {μ_t, Σ_t} from FROZEN backbone embeddings during training
  - Uses SRT metrics for routing at inference
  - Replaces learned MLP router with non-parametric SRT router

Two modes:
  --srt_metric_mode hard    : ZCA whitening + L2 (matches routing_analysis experiment)
  --srt_metric_mode dynamics: SRM metric selection (matches contribution_UNIFIED)

Key SRT features:
  - Zero-drift: no learnable parameters in router
  - Uses FROZEN encoder for embedding extraction (same space at train & inference)
  - Statistical signatures stored (zero-rehearsal compliant)
"""

import copy
import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Union, Any, Tuple, List, Optional

from cl_trainer_gainlora_inflora import GainLoRA_InfLoRA_Trainer

try:
    from srt_router import SRTRouter, TaskSignature
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from srt_router import SRTRouter, TaskSignature


# ─────────────────────────────────────────────────────────────────────────────
#  EMBEDDING EXTRACTION — FROZEN ENCODER
# ─────────────────────────────────────────────────────────────────────────────

def extract_embeddings_from_batch(model, inputs: Dict) -> torch.Tensor:
    """
    Extract mean-pooled encoder hidden states from a FROZEN encoder.

    Theory (contribution_UNIFIED.md): SRT signatures {μ_t, Σ_t} are computed
    on FROZEN backbone embeddings — the embedding space BEFORE any LoRA adaptation.

    Bug fix: previously used `model(**inputs)` which goes through the ADAPTED
    encoder (LoRA fine-tuned). Now uses `model.encoder.encoder_frozen` to match
    the frozen embedding space used during inference routing.

    This matches routing_analysis/extract_embeddings_t5.py which loads a
    pretrained T5EncoderModel and extracts from its frozen encoder.
    """
    input_ids = inputs.get('input_ids')
    attention_mask = inputs.get('attention_mask')

    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids)
    if not isinstance(attention_mask, torch.Tensor):
        attention_mask = torch.tensor(attention_mask)

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        # Use FROZEN encoder — same as routing_analysis experiment
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'encoder_frozen'):
            frozen_enc = model.encoder.encoder_frozen
            enc_out = frozen_enc(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            if hasattr(enc_out, 'last_hidden_state'):
                hidden = enc_out.last_hidden_state
            else:
                hidden = enc_out[0]
        else:
            # Fallback: model is a bare T5EncoderModel (no gainlora wrapper)
            enc_out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            if hasattr(enc_out, 'last_hidden_state'):
                hidden = enc_out.last_hidden_state
            elif hasattr(enc_out, 'encoder_last_hidden_state'):
                hidden = enc_out.encoder_last_hidden_state
            else:
                hidden = enc_out[0]

    # Mean pooling (identical to routing_analysis/extract_embeddings_t5.py)
    if attention_mask is not None:
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    else:
        pooled = hidden.mean(dim=1)

    return pooled.float()   # (B, d)


# ─────────────────────────────────────────────────────────────────────────────
#  SRT TRAINER
# ─────────────────────────────────────────────────────────────────────────────

class SRT_Trainer(GainLoRA_InfLoRA_Trainer):
    """
    GainLoRA + SRT Router.

    Changes from GainLoRA_InfLoRA_Trainer:
      1. After training task t: extract embeddings → compute {μ_t, Σ_t}
      2. Store in SRTRouter (zero-rehearsal compliant)
      3. At inference: SRT router replaces attention-based routing
      4. SRT router persists across tasks (no drift)

    SRT modes:
      srt_metric_mode='hard'     : ZCA whitening + L2 (matches routing_analysis)
      srt_metric_mode='dynamics': SRM metric selection (matches contribution_UNIFIED)
    """

    def __init__(
        self,
        model,
        args,
        train_dataset,
        cur_task_id: int,
        task_order: list,
        data_collator_replay=None,
        replay_dataset_dict=None,
        replay_label_dict=None,
        eval_dataset=None,
        tokenizer=None,
        data_collator=None,
        compute_metrics=None,
        callbacks=None,
        # ── SRT-specific ────────────────────────────────────────────────
        srt_metric_mode: str = 'hard',
        srt_shrink: bool = True,
        srt_shrink_factor: float = 0.1,
        srt_max_emb_samples: int = 500,
        srt_load_path: Optional[str] = None,
    ):
        super().__init__(
            model, args, train_dataset, cur_task_id, task_order,
            data_collator_replay, replay_dataset_dict, replay_label_dict,
            eval_dataset, tokenizer, data_collator, compute_metrics, callbacks,
        )

        self.srt_metric_mode = srt_metric_mode
        self.srt_shrink = srt_shrink
        self.srt_shrink_factor = srt_shrink_factor
        self.srt_max_emb_samples = srt_max_emb_samples
        self.srt_load_path = srt_load_path

        self.srt_router: Optional[SRTRouter] = None
        self._srt_init()

    def _srt_init(self):
        """Initialize SRT router (idempotent — safe to call multiple times)."""
        if self.srt_router is not None:
            return
        self.srt_router = SRTRouter(
            srt_metric_mode=self.srt_metric_mode,
            use_shrink=self.srt_shrink,
            shrink_factor=self.srt_shrink_factor,
        )
        if self.srt_load_path is not None:
            self.load_srt_signatures(self.srt_load_path, wire_model=True)

    # ── 1. EMBEDDING EXTRACTION ────────────────────────────────────────────

    def _extract_task_embeddings(self, max_samples: int = None) -> Tuple[torch.Tensor, List]:
        """Extract embeddings for the current training task from FROZEN encoder."""
        if max_samples is None:
            max_samples = self.srt_max_emb_samples

        train_dataloader = self.get_train_dataloader()
        h_list = []
        task_ids = []

        max_batches = min(max_samples, len(train_dataloader))

        for step, inputs in enumerate(train_dataloader):
            if step >= max_batches:
                break
            inputs = self._prepare_inputs(inputs)
            h = extract_embeddings_from_batch(self.model, inputs)
            h_list.append(h.cpu())
            if 'task_ids' in inputs:
                task_ids.extend(inputs['task_ids'].tolist())

        if not h_list:
            return torch.empty(0), task_ids

        embeddings = torch.cat(h_list, dim=0)  # (n, d)
        return embeddings, task_ids

    # ── 2. SIGMA COMPUTATION ───────────────────────────────────────────────

    def _compute_and_store_signature(self, task_id: int):
        """After training task t: extract embeddings → compute {μ_t, Σ_t} → add to router."""
        if self.srt_router is None:
            self._srt_init()

        h_train, _ = self._extract_task_embeddings(self.srt_max_emb_samples)
        if h_train.shape[0] == 0:
            print(f"  [SRT] WARNING: no embeddings extracted for task {task_id}")
            return

        h_np = h_train.numpy()
        sig = self.srt_router.add_task(task_id=task_id, h_train=h_np)

        summary = self.srt_router.summary()
        print(f"  [SRT] Task {task_id}: PaR={sig.par:.1f}, metric={sig.metric}, "
              f"n={sig.n}, total_tasks={summary['n_tasks']}")

    # ── 3. SRT ROUTING AT INFERENCE ─────────────────────────────────────────

    def _srt_route(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route using SRT router (non-parametric, zero-drift)."""
        if self.srt_router is None or len(self.srt_router.signatures) == 0:
            cur_t = self.task_order[self.cur_task_id]
            B = h.shape[0]
            return torch.full((B,), cur_t, dtype=torch.long), torch.zeros(B, 1)

        h_np = h.cpu().numpy()
        pred, dists = self.srt_router.route(h_np)

        pred_t = torch.tensor(pred, dtype=torch.long, device=h.device)
        dists_t = torch.tensor(dists, dtype=torch.float32, device=h.device)

        return pred_t, dists_t

    # ── 4. ATTENTION-BASED ROUTING → SRT REPLACEMENT ─────────────────────

    def _replace_attention_routing(self):
        """
        Wire SRT router into model encoder for non-parametric routing.

        Position mapping (key_attention_weights shape = (B, 1+N_prev, 1)):
          index 0         = current task
          index 1..N_prev = previous tasks (by task_order)
        """
        if self.srt_router is None or len(self.srt_router.signatures) == 0:
            return

        current_task = self.task_order[self.cur_task_id]
        task_id_to_idx = {current_task: 0}

        for prev_idx in range(self.cur_task_id):
            prev_task = self.task_order[prev_idx]
            task_id_to_idx[prev_task] = prev_idx + 1

        self.model.encoder.srt_router = self.srt_router
        self.model.encoder.srt_task_id_to_idx = task_id_to_idx
        self.model.encoder.use_srt_routing = True
        print(f"  [SRT] Wired router: {len(self.srt_router.signatures)} tasks, "
              f"use_srt_routing=True, cur_task={current_task}, mapping={task_id_to_idx}")

    # ── 5. TASK END HOOK ───────────────────────────────────────────────────

    def on_task_end(self, task_id: Union[int, str]):
        """Called after each task's training finishes."""
        self._compute_and_store_signature(task_id)
        self._replace_attention_routing()

        if self.srt_router:
            summary = self.srt_router.summary()
            print(f"  [SRT] Router summary: {summary['n_tasks']} tasks, "
                  f"avg_PaR={summary['avg_par']:.1f}, metrics={summary['metrics']}")

    # ── 6. EVALUATION ─────────────────────────────────────────────────────

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: bool = False,
        ignore_keys: Optional[List] = None,
        metric_key_prefix: str = "eval",
    ) -> Any:
        """SRT routing handled transparently in model.forward()."""
        return super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

    # ── 7. SAVE / LOAD SIGNATURES ──────────────────────────────────────────

    def save_srt_signatures(self, output_dir: str):
        """Save SRT router signatures to disk."""
        if self.srt_router is None:
            return
        path = os.path.join(output_dir, 'srt_signatures.npz')
        self.srt_router.save(path)
        print(f"  [SRT] Saved signatures to {path}")

    def load_srt_signatures(self, checkpoint_dir: str, wire_model: bool = False):
        """Load SRT router signatures from a previous checkpoint."""
        path = os.path.join(checkpoint_dir, 'srt_signatures.npz')
        if not os.path.exists(path):
            print(f"  [SRT] No signatures found at {path}")
            return
        self.srt_router.load(path)
        print(f"  [SRT] Loaded {len(self.srt_router.signatures)} signatures from {path}")
        if wire_model and len(self.srt_router.signatures) > 0:
            self.model.encoder.srt_router = self.srt_router
            current_task = self.task_order[self.cur_task_id]
            task_id_to_idx = {current_task: 0}
            for prev_idx in range(self.cur_task_id):
                task_id_to_idx[self.task_order[prev_idx]] = prev_idx + 1
            self.model.encoder.srt_task_id_to_idx = task_id_to_idx
            self.model.encoder.use_srt_routing = True
            print(f"  [SRT] Wired {len(self.srt_router.signatures)} signatures to model, "
                  f"cur_task={current_task}, mapping={task_id_to_idx}")
