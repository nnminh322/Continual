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
        self.theta_stars: Dict[int, Dict[str, torch.Tensor]] = {}
        self._current_task_mu = None

        if sgwi_mode not in self.SGWI_MODES:
            raise ValueError(f"sgwi_mode must be one of {self.SGWI_MODES}, got '{sgwi_mode}'")

        if self.srt_load_path is not None:
            self._load_theta_stars(self.srt_load_path)

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
        model_name = getattr(self.model.config, '_name_or_path', None)
        if not model_name:
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
                embeddings = torch.from_numpy(data['embeddings']).float()
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
            h_list.append(h.float().cpu())
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
        sig = self.srt_router.add_task(task_id=task_id, h_train=h_train.float().cpu().numpy())
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

    def get_reg_matrix(self):
        """Override GainLoRA initialization with T5-aligned SGWI ablation semantics."""
        mode = self.sgwi_mode
        logger.info(f"[SGWI-LLaMA] get_reg_matrix: mode={mode}, task_id={self.cur_task_id}")

        if mode == 'inflora':
            super().get_reg_matrix()
            return

        self._init_gpm_attrs_skip()

        if self.cur_task_id == 0:
            logger.info(f"[SGWI-LLaMA] Task 0, mode={mode} → standard init (no prior tasks)")
            return

        if mode == 'random':
            logger.info("[SGWI-LLaMA] Config 2: random — no warm init")
            return

        if mode == 'full_lora':
            logger.info("[SGWI-LLaMA] Config 3: full_lora — no warm init")
            return

        srt_weights = self._compute_sgwi_weights()
        if not srt_weights:
            logger.warning("[SGWI-LLaMA] No SRT weights. Falling back to random init.")
            return

        logger.info(f"[SGWI-LLaMA] SRT weights: {srt_weights}")

        if mode == 'sgwi_freeze_a':
            logger.info("[SGWI-LLaMA] Config 5: warm-init A only (frozen)")
            self._sgwi_init_a(srt_weights)
            return

        if mode == 'sgwi_train_a':
            logger.info("[SGWI-LLaMA] Config 6: warm-init A only (trainable)")
            self._sgwi_init_a(srt_weights)
            return

        if mode == 'sgwi_full':
            logger.info("[SGWI-LLaMA] Config 4: warm-init both A and B")
            self._sgwi_init_a(srt_weights)
            self._fuse_past_lora_adapters(srt_weights)
            return

        raise ValueError(f"Unknown sgwi_mode: {mode}")

    def _init_gpm_attrs_skip(self):
        self._cur_task = 0
        self.feature_list = []
        self.feature_trans_list = []
        self.feature_mat = []
        self.feature_trans_mat = []
        logger.info("[SGWI-LLaMA] GPM attrs initialized (skip mode): _cur_task=0, feature_list=[]")

    def _compute_sgwi_weights(self) -> Dict[int, float]:
        if self.srt_router is None or len(self.srt_router.signatures) == 0:
            return {}

        h_train, _ = self._extract_task_embeddings(max_samples=self.srt_max_emb_samples)
        if h_train is None or (hasattr(h_train, '__len__') and len(h_train) == 0):
            return {}

        current_mu = h_train.float().mean(dim=0).cpu().numpy()
        self._current_task_mu = current_mu

        distances = {}
        for task_id, sig in self.srt_router.signatures.items():
            diff = current_mu - sig.mu
            if hasattr(self.srt_router, 'pooled_cov') and self.srt_router.pooled_cov is not None:
                try:
                    cov_inv = np.linalg.inv(self.srt_router.pooled_cov + 1e-6 * np.eye(len(diff)))
                    distance = float(diff @ cov_inv @ diff)
                except np.linalg.LinAlgError:
                    distance = float(np.sum(diff ** 2))
            else:
                distance = float(np.sum(diff ** 2))
            distances[task_id] = distance

        if not distances:
            return {}
        if len(distances) == 1:
            return {task_id: 1.0 for task_id in distances}

        tau = float(np.median(list(distances.values()))) + 1e-8
        weights = {task_id: math.exp(-distance / tau) for task_id, distance in distances.items()}
        denom = sum(weights.values()) + 1e-12
        return {task_id: weight / denom for task_id, weight in weights.items()}

    def _sgwi_init_a(self, srt_weights: Dict[int, float]):
        model = self.model
        device = next(model.parameters()).device
        lora_r = self.args.lora_r if hasattr(self.args, 'lora_r') else 8

        fused_count = 0
        skipped_count = 0

        for name, module in model.named_modules():
            if not (hasattr(module, 'lora_q') and hasattr(module, 'lora_v')):
                continue
            if not hasattr(module, 'previous_lora_weights_q'):
                continue

            prev_q = getattr(module, 'previous_lora_weights_q', None)
            prev_v = getattr(module, 'previous_lora_weights_v', None)
            if prev_q is None or len(prev_q) == 0:
                continue

            for lora_tag, lora_cur, prev_list in [
                ('lora_q', module.lora_q, prev_q),
                ('lora_v', module.lora_v, prev_v),
            ]:
                if prev_list is None or len(prev_list) == 0:
                    continue

                delta_w = None
                for past_task_id, weight in srt_weights.items():
                    if isinstance(past_task_id, str):
                        try:
                            pos = self.task_order.index(past_task_id)
                            idx = (self.cur_task_id - 1) - pos
                        except ValueError:
                            skipped_count += 1
                            continue
                    else:
                        idx = int(past_task_id)
                    if idx < 0 or idx >= len(prev_list):
                        skipped_count += 1
                        continue

                    past_lora = prev_list[idx]
                    ba = past_lora.lora_B.data.float() @ past_lora.lora_A.data.float()
                    delta_w = weight * ba if delta_w is None else delta_w + weight * ba

                if delta_w is None or delta_w.norm().item() < 1e-10:
                    skipped_count += 1
                    continue

                try:
                    _, singular_values, vt = torch.linalg.svd(delta_w.to(device), full_matrices=False)
                    rank = min(lora_r, len(singular_values))
                    singular_sqrt = torch.sqrt(singular_values[:rank] + 1e-12)
                    a_new = singular_sqrt.unsqueeze(1) * vt[:rank, :]
                    lora_cur.lora_A.data.copy_(a_new.to(lora_cur.lora_A.data.device))
                    fused_count += 1
                except Exception as exc:
                    logger.warning(f"[SGWI-LLaMA] SVD failed for {name}.{lora_tag}: {exc}")
                    skipped_count += 1

        logger.info(f"[SGWI-LLaMA] Warm-init A for {fused_count} modules, skipped {skipped_count}")

    def _fuse_past_lora_adapters(self, srt_weights: Dict[int, float]):
        model = self.model
        device = next(model.parameters()).device

        fused_count = 0
        skipped_count = 0

        for name, module in model.named_modules():
            if not (hasattr(module, 'lora_q') and hasattr(module, 'lora_v')):
                continue
            if not hasattr(module, 'previous_lora_weights_q'):
                continue

            prev_q = getattr(module, 'previous_lora_weights_q', None)
            prev_v = getattr(module, 'previous_lora_weights_v', None)
            if prev_q is None or len(prev_q) == 0:
                continue

            for lora_tag, lora_cur, prev_list in [
                ('lora_q', module.lora_q, prev_q),
                ('lora_v', module.lora_v, prev_v),
            ]:
                if prev_list is None or len(prev_list) == 0:
                    continue

                delta_w = None
                for past_task_id, weight in srt_weights.items():
                    if isinstance(past_task_id, str):
                        try:
                            pos = self.task_order.index(past_task_id)
                            idx = (self.cur_task_id - 1) - pos
                        except ValueError:
                            skipped_count += 1
                            continue
                    else:
                        idx = int(past_task_id)
                    if idx < 0 or idx >= len(prev_list):
                        skipped_count += 1
                        continue

                    past_lora = prev_list[idx]
                    a_past = past_lora.lora_A.data.float()
                    b_past = past_lora.lora_B.data.float()
                    ba = b_past @ a_past
                    delta_w = weight * ba if delta_w is None else delta_w + weight * ba

                if delta_w is None or delta_w.norm().item() < 1e-10:
                    skipped_count += 1
                    continue

                try:
                    a_cur = lora_cur.lora_A.data.float().to(device)
                    ata = a_cur @ a_cur.T
                    eps = 1e-4 * torch.eye(a_cur.shape[0], device=device)
                    b_warm = delta_w.to(device) @ a_cur.T @ torch.linalg.inv(ata + eps)
                    lora_cur.lora_B.data.copy_(b_warm.to(lora_cur.lora_B.data.device))
                    fused_count += 1
                except Exception as exc:
                    logger.warning(f"[SGWI-LLaMA] lora_B warm init failed for {name}.{lora_tag}: {exc}")

        logger.info(f"[SGWI-LLaMA] Fused {fused_count} LoRA modules, skipped {skipped_count}")

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if self.lambda_emb > 0 and self.cur_task_id > 0 and len(self.theta_stars) > 0:
            loss = loss + self._dual_fisher_penalty()

        return (loss, outputs) if return_outputs else loss

    def _dual_fisher_penalty(self) -> torch.Tensor:
        total = torch.tensor(0.0, device=next(self.model.parameters()).device)

        srt_weights = {}
        if self.srt_router and len(self.srt_router.signatures) > 0:
            for task_id in self.theta_stars:
                if task_id in self.srt_router.signatures:
                    srt_weights[task_id] = 1.0
            if srt_weights:
                normalizer = sum(srt_weights.values())
                srt_weights = {task_id: weight / normalizer for task_id, weight in srt_weights.items()}

        if not srt_weights:
            return total

        for task_id, weight in srt_weights.items():
            if task_id not in self.theta_stars:
                continue

            for name, param in self.model.named_parameters():
                if not param.requires_grad or 'lora_' not in name:
                    continue
                if name not in self.theta_stars[task_id]:
                    continue

                theta_star = self.theta_stars[task_id][name].to(param.device)
                total = total + weight * (param - theta_star).pow(2).sum()

        return self.lambda_emb * total

    def on_task_end(self, task_id):
        logger.info(f"[SGWI-LLaMA] Saving θ* for task {task_id} (Dual Fisher)")
        self.theta_stars[task_id] = {}
        for name, param in self.model.named_parameters():
            if 'lora_' in name and param.requires_grad:
                self.theta_stars[task_id][name] = param.detach().clone().cpu()

        theta_stars_path = os.path.join(self.args.output_dir, 'saved_weights', 'theta_stars.pt')
        os.makedirs(os.path.dirname(theta_stars_path), exist_ok=True)
        torch.save(self.theta_stars, theta_stars_path)

        self._compute_and_store_signature(task_id)
        self._replace_attention_routing()

    def _load_theta_stars(self, srt_load_path: str):
        theta_path = os.path.join(srt_load_path, 'theta_stars.pt')
        if os.path.exists(theta_path):
            self.theta_stars = torch.load(theta_path, map_location='cpu', weights_only=True)
            logger.info(f"[SGWI-LLaMA] Loaded θ* for {len(self.theta_stars)} past tasks from {theta_path}")
        else:
            logger.info(f"[SGWI-LLaMA] No θ* found at {theta_path}")
