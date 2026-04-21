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

from cl_trainer_gainlora import GainLoRATrainer

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
    Extract frozen backbone embeddings for SRT signatures {μ_t, Σ_t}.

    Matches routing_analysis extraction EXACTLY:

      T5 (encoder model):
        Layer: last encoder hidden state  (out.last_hidden_state)
        Pool:  mean over non-padding tokens

      LLaMA (decoder model):
        Layer: last decoder hidden state  (out.hidden_states[-1])
        Pool:  last non-padding token    (NOT mean pooling)

    MUST use model.encoder.encoder_frozen (or equivalent frozen wrapper).
    NEVER use adapted model (LoRA fine-tuned) — embedding space differs.

    References:
      T5:  routing_analysis/extract_embeddings_t5.py (layer="encoder", pool="avg")
      LLaMA: routing_analysis/extract_embeddings_llama.py (pool="last", layer="hidden")
    """
    device = next(model.parameters()).device

    def _to_tensor(value):
        if value is None or isinstance(value, torch.Tensor):
            return value
        return torch.tensor(value)

    input_ids = _to_tensor(inputs.get('input_ids'))
    attention_mask = _to_tensor(inputs.get('attention_mask'))

    if input_ids is not None:
        input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    llama_source_ids = _to_tensor(inputs.get('input_ids_wo_label', inputs.get('input_ids')))
    if llama_source_ids is not None:
        llama_source_ids = llama_source_ids.to(device)

    llama_pad_token_id = getattr(getattr(model, 'model', None), 'padding_idx', None)
    if llama_pad_token_id is None:
        llama_pad_token_id = getattr(getattr(model, 'config', None), 'pad_token_id', 0)
    llama_source_mask = None
    if llama_source_ids is not None:
        llama_source_mask = (llama_source_ids != llama_pad_token_id).long()

    with torch.no_grad():
        # ── Case 1: T5 — frozen encoder (encoder_frozen has last_hidden_state) ──
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'encoder_frozen'):
            frozen_enc = model.encoder.encoder_frozen
            enc_out = frozen_enc(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            # frozen_enc is T5EncoderModel → has last_hidden_state
            hidden = enc_out.last_hidden_state  # (B, L, d)

            # Mean pooling — matches routing_analysis/extract_embeddings_t5.py
            mask = attention_mask.unsqueeze(-1).float()          # (B, L, 1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, d)

        # ── Case 2: LLaMA gainlora wrapper — frozen decoder extractor on source-only inputs ──
        elif hasattr(model, 'model') and hasattr(model.model, 'encoder_frozen') and model.model.encoder_frozen is not None:
            pooled = model.model.encoder_frozen(llama_source_ids, llama_source_mask)

        # ── Case 3: bare T5EncoderModel (no gainlora wrapper) ──
        elif hasattr(model, 'encoder'):
            enc_out = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden = getattr(enc_out, 'last_hidden_state', None) or enc_out[0]
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # ── Case 4: bare LLaMA decoder/causal LM on source-only inputs ──
        elif hasattr(model, 'model'):
            out = model.model(
                input_ids=llama_source_ids,
                attention_mask=llama_source_mask,
                output_hidden_states=True,
            )
            hidden = out.hidden_states[-1]                     # (B, L, d)
            seq_lens = (llama_source_mask.sum(dim=1) - 1).clamp(min=0)  # (B,)
            B = hidden.size(0)
            pooled = hidden[torch.arange(B, device=device), seq_lens]  # (B, d)

        # ── Case 5: bare T5EncoderModel directly ──
        else:
            enc_out = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = getattr(enc_out, 'last_hidden_state', None) or enc_out[0]
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    return pooled  # (B, d), float32


# ─────────────────────────────────────────────────────────────────────────────
#  SRT TRAINER
# ─────────────────────────────────────────────────────────────────────────────

class SRT_Trainer(GainLoRATrainer):
    """
    GainLoRA + SRT Router.

    Changes from GainLoRATrainer:
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
        srt_skip_forward: bool = False,
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
        self.srt_skip_forward = srt_skip_forward

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
        """
        Extract embeddings for the current training task.

        Two modes controlled by self.srt_skip_forward:
          False (default): forward pass through FROZEN backbone (original behavior)
          True          : load pre-extracted embeddings from disk
                          embeddings/{backbone}/{split}/{task}/train.npz
                          Keys: 'embeddings' (n,d), 'labels' (n,)
        """
        if max_samples is None:
            max_samples = self.srt_max_emb_samples

        # ── MODE 1: Load pre-extracted embeddings from disk ───────────────
        if self.srt_skip_forward:
            cur_task = self.task_order[self.cur_task_id]
            emb_path = self._srt_emb_cache_path(cur_task)

            if emb_path is not None and os.path.exists(emb_path):
                print(f"  [SRT] ★ LOAD FROM CACHE: {emb_path}")
                data = np.load(emb_path, allow_pickle=True)
                embeddings = torch.from_numpy(data['embeddings'])  # (n, d)

                # Take at most max_samples (in samples, not batches)
                if embeddings.shape[0] > max_samples:
                    embeddings = embeddings[:max_samples]

                print(f"  [SRT]   → Loaded {embeddings.shape[0]} embeddings, dim={embeddings.shape[1]}")
                return embeddings, []
            else:
                print(f"  [SRT] ✗ SKIP_FORWARD=True but cache MISS for '{cur_task}' → falling back to forward pass")
                if emb_path is not None:
                    print(f"  [SRT]   → Expected path: {emb_path}")
                else:
                    print(f"  [SRT]   → Unknown backbone (model_name='{getattr(self.args, 'model_name_or_path', '')}')")
                # Fall through to forward extraction below

        # ── MODE 2: Forward pass extraction (original behavior) ───────────
        print(f"  [SRT] → Forward pass extraction ({max_samples} batches, backbone)")
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

    # ── helper: resolve embedding cache path ───────────────────────────────

    def _srt_emb_cache_path(self, task_name: str) -> Optional[str]:
        """
        Resolve pre-extracted embedding path.

        Mapping:
          google/flan-t5-xl          → embeddings/flan-t5-xl/
          google/flan-t5-large       → embeddings/flan-t5-large/
          meta-llama/Llama-2-7b-hf   → embeddings/Llama-2-7b-hf/

        Task split:
          SuperNI tasks (task*):     {backbone}/SuperNI/{task}/train.npz
          Long-Sequence tasks:        {backbone}/Long_Sequence/{task}/train.npz
        """
        model_name = getattr(self.args, 'model_name_or_path', '') or ''

        # Map model identifier to embeddings subdirectory
        backbone = None
        if 'flan-t5-xl' in model_name.lower():
            backbone = 'flan-t5-xl'
        elif 'flan-t5-large' in model_name.lower():
            backbone = 'flan-t5-large'
        elif 'llama' in model_name.lower():
            backbone = 'Llama-2-7b-hf'
        else:
            return None

        # Determine split (SuperNI vs Long_Sequence)
        split = 'SuperNI' if task_name.startswith('task') else 'Long_Sequence'

        root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'embeddings')
        return os.path.join(root, backbone, split, task_name, 'train.npz')

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
          index 1..N_prev = previous tasks in REVERSE chronological order
                            (slot 1 = most recent previous, slot N = oldest)

        Why reverse? prompts_keys_till_now.pt saves cat([current, prev...]) and
        previous_lora_list.reverse() loads newest first. So:
          slot 1 = task_order[cur_task_id - 1]  (most recent previous)
          slot 2 = task_order[cur_task_id - 2]
          ...
          slot N = task_order[0]                 (oldest)
        """
        if self.srt_router is None or len(self.srt_router.signatures) == 0:
            return

        current_task = self.task_order[self.cur_task_id]
        task_id_to_idx = {current_task: 0}

        # BUG #31 fix: slots are in REVERSE chronological order
        for prev_idx in range(self.cur_task_id):
            prev_task = self.task_order[prev_idx]
            task_id_to_idx[prev_task] = self.cur_task_id - prev_idx

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
            # BUG #31 fix: slots are in REVERSE chronological order
            for prev_idx in range(self.cur_task_id):
                task_id_to_idx[self.task_order[prev_idx]] = self.cur_task_id - prev_idx
            self.model.encoder.srt_task_id_to_idx = task_id_to_idx
            self.model.encoder.use_srt_routing = True
            print(f"  [SRT] Wired {len(self.srt_router.signatures)} signatures to model, "
                  f"cur_task={current_task}, mapping={task_id_to_idx}")
