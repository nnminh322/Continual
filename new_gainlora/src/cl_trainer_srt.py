"""
SRT Trainer: GainLoRA + SRT Router integration.

Implements CONTRIBUTION_1 (SRT) routing within GainLoRA architecture.
  - Computes {μ_t, Σ_t} during training
  - Uses SRT metrics (L2 / Mahalanobis / PSR) for routing at inference
  - Replaces learned MLP router with non-parametric SRT router

Key SRT features:
  - Zero-drift: no learnable parameters in router
  - Metric selection by anisotropy: PaR → metric type
  - Statistical signatures stored (zero-rehearsal compliant)

Architecture modification:
  Original: trans_input → prompt_key attention weights → router
  SRT:      {μ_t, Σ_t} stored → SRT router replaces attention-based routing
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
    from srt_router import SRTRouter, TaskSignature, metric_l2, metric_mahalanobis, metric_psr
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from srt_router import SRTRouter, TaskSignature


# ─────────────────────────────────────────────────────────────────────────────
#  ATTENTION-BASED ROUTING → SRT ROUTING SWAP
# ─────────────────────────────────────────────────────────────────────────────

def extract_embeddings_from_batch(model, inputs: Dict) -> torch.Tensor:
    """
    Extract mean-pooled encoder hidden states from a batch.

    Matches the embedding extraction used in routing_analysis/extract_embeddings_t5.py.
    """
    with torch.no_grad():
        outputs = model(**{k: v for k, v in inputs.items()
                          if k in ('input_ids', 'attention_mask', 'decoder_input_ids')})

    # Get encoder last hidden state
    if hasattr(outputs, 'encoder_last_hidden_state'):
        hidden = outputs.encoder_last_hidden_state   # (B, L, d)
    elif isinstance(outputs, dict) and 'encoder_last_hidden_state' in outputs:
        hidden = outputs['encoder_last_hidden_state']
    else:
        # Fallback: use first output
        hidden = outputs[0]

    # Mean pooling (same as routing_analysis)
    attention_mask = inputs.get('attention_mask')
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

    SRT args:
      srt_whiten:       use ZCA-whitened L2 (equivalen to pooled Mahalanobis)
      srt_shrink:        apply Ledoit-Wolf shrinkage to covariance
      srt_shrink_factor: shrinkage intensity (default 0.1)
      srt_metric:        override metric selection: 'auto' | 'l2' | 'mahalanobis' | 'psr'
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
        srt_whiten: bool = False,
        srt_shrink: bool = True,
        srt_shrink_factor: float = 0.1,
        srt_metric: str = 'auto',
        srt_max_emb_samples: int = 500,
        srt_load_path: Optional[str] = None,   # load signatures from previous checkpoint
    ):
        super().__init__(
            model, args, train_dataset, cur_task_id, task_order,
            data_collator_replay, replay_dataset_dict, replay_label_dict,
            eval_dataset, tokenizer, data_collator, compute_metrics, callbacks,
        )

        self.srt_whiten = srt_whiten
        self.srt_shrink = srt_shrink
        self.srt_shrink_factor = srt_shrink_factor
        self.srt_metric = srt_metric
        self.srt_max_emb_samples = srt_max_emb_samples
        self.srt_load_path = srt_load_path

        # SRT Router: shared across all tasks (no drift)
        self.srt_router: Optional[SRTRouter] = None
        self._srt_init()

    def _srt_init(self):
        """Initialize SRT router (idempotent — safe to call multiple times)."""
        if self.srt_router is not None:
            return   # already initialized
        # BUG F fix: enable SRM metric selection by default
        self.srt_router = SRTRouter(use_srm=True)
        if self.srt_load_path is not None:
            self.load_srt_signatures(self.srt_load_path, wire_model=True)

    # ── 1. EMBEDDING EXTRACTION ────────────────────────────────────────────

    def _extract_task_embeddings(self, max_samples: int = None) -> Tuple[torch.Tensor, List]:
        """
        Extract embeddings for the current training task.

        Uses the frozen backbone forward pass to get encoder hidden states.
        Extracts up to max_samples batches for efficiency.

        Returns:
            embeddings: (n, d) tensor on CPU
            task_ids: list of task indices
        """
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
            # Get task IDs from batch if available
            if 'task_ids' in inputs:
                task_ids.extend(inputs['task_ids'].tolist())

        if not h_list:
            return torch.empty(0), task_ids

        embeddings = torch.cat(h_list, dim=0)  # (n, d)
        return embeddings, task_ids

    # ── 2. SIGMA COMPUTATION ───────────────────────────────────────────────

    def _compute_and_store_signature(self, task_id: int):
        """
        After training task t, compute its statistical signature and store in router.

        Steps:
          1. Extract training embeddings
          2. Compute {μ_t, Σ_t} with optional shrinkage
          3. Add to SRT router
          4. Log SRT statistics
        """
        if self.srt_router is None:
            self._srt_init()

        # Extract embeddings
        h_train, _ = self._extract_task_embeddings(self.srt_max_emb_samples)
        if h_train.shape[0] == 0:
            print(f"  [SRT] WARNING: no embeddings extracted for task {task_id}")
            return

        h_np = h_train.numpy()

        # Override metric if specified
        override_metric = None
        if self.srt_metric != 'auto':
            override_metric = self.srt_metric

        # Compute shrinkage
        shrink = self.srt_shrink

        # Add to SRT router
        sig = self.srt_router.add_task(
            task_id=task_id,
            h_train=h_np,
            use_shrink=shrink,
            shrink_factor=self.srt_shrink_factor,
        )

        if override_metric:
            sig._metric = override_metric

        # Log
        summary = self.srt_router.summary()
        print(f"  [SRT] Task {task_id}: PaR={sig.par:.1f}, metric={sig.metric}, "
              f"n={sig.n}, total_tasks={summary['n_tasks']}")

    # ── 3. SRT ROUTING AT INFERENCE ─────────────────────────────────────────

    def _srt_route(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route using SRT router (non-parametric, zero-drift).

        Replaces the learned attention-based routing from GainLoRA.

        Args:
            h: (B, d) embeddings from frozen backbone

        Returns:
            task_ids: (B,) predicted task ID for each sample
            dists: (B, T) distance to each task
        """
        if self.srt_router is None or len(self.srt_router.signatures) == 0:
            # Fallback: return current task
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

        Sets:
          - model.encoder.srt_router        = self.srt_router
          - model.encoder.srt_task_id_to_idx = mapping from SRT task_id → position in key_attention_weights
          - model.encoder.use_srt_routing   = True

        Position mapping (key_attention_weights shape = (B, 1+N_prev, 1)):
          index 0         = current task  (prompt_key, LoRA branch 0)
          index 1..N_prev = previous tasks (previous_prompts_keys[0..N_prev-1], LoRA branches 1..N_prev)

        BUG B.2 fix: mapping is built from self.task_order, NOT sorted(task_ids).
          previous_prompts_keys[i] corresponds to self.task_order[i], not sorted(tid).
        """
        if self.srt_router is None or len(self.srt_router.signatures) == 0:
            return

        # BUG B.2 fix: build mapping from task_order, not sorted keys.
        # key_attention_weights position i maps to self.task_order[i-1] for i≥1.
        # Position 0 → current task (self.task_order[self.cur_task_id]).
        # Position j (1-indexed) → self.task_order[j-1].
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

    def on_task_end(self, task_id: int):
        """
        Called after each task's training finishes (from run_t5.py).

        Steps:
          1. Compute and store statistical signature {μ_t, Σ_t}
          2. Wire SRT router into model encoder (so inference uses SRT routing)
          3. Log router summary
        """
        # Step 1: compute + store signature
        self._compute_and_store_signature(task_id)

        # Step 2: wire router into model so forward() uses SRT injection
        self._replace_attention_routing()

        # Step 3: log
        if self.srt_router:
            summary = self.srt_router.summary()
            print(f"  [SRT] Router summary: {summary['n_tasks']} tasks, "
                  f"avg_PaR={summary['avg_par']:.1f}, metrics={summary['metrics']}")

    # ── 6. EVALUATION WITH SRT ROUTING ────────────────────────────────────

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: bool = False,
        ignore_keys: Optional[List] = None,
        metric_key_prefix: str = "eval",
    ) -> Any:
        """
        Evaluation loop — SRT routing is handled transparently in model.forward().

        The model's forward() has been pre-wired with srt_router reference and
        use_srt_routing=True, so SRT injection happens automatically during
        generate() / forward() calls. No need to override routing here.
        """
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
            # Build task_id → position mapping from task_order (not sorted keys)
            current_task = self.task_order[self.cur_task_id]
            task_id_to_idx = {current_task: 0}
            for prev_idx in range(self.cur_task_id):
                task_id_to_idx[self.task_order[prev_idx]] = prev_idx + 1
            self.model.encoder.srt_task_id_to_idx = task_id_to_idx
            self.model.encoder.use_srt_routing = True
            print(f"  [SRT] Wired {len(self.srt_router.signatures)} signatures to model, "
                  f"cur_task={current_task}, mapping={task_id_to_idx}")

    # ── 8. LOGGING ──────────────────────────────────────────────────────────

    def _log_srt_metrics(self):
        """Log SRT-specific metrics."""
        if self.srt_router is None:
            return
        summary = self.srt_router.summary()
        logs = {
            'srt/n_tasks': summary['n_tasks'],
            'srt/avg_par': summary['avg_par'],
        }
        self.log(logs)
