"""
SGWI + Dual Fisher Trainer (Contribution 2)
============================================
Inherits from SRT_Trainer. Adds:
  - SGWI (SRT-Guided Warm Initialization): warm-init LoRA from similar past tasks
  - Dual Fisher: SRT-weighted embedding Fisher penalty during training

Usage in run_t5.py:
    from sgwi_trainer import SGWI_DualFisher_Trainer
    trainer = SGWI_DualFisher_Trainer(
        ...,
        sgwi_mode='sgwi',       # 'sgwi', 'inflora', 'random', 'sgwi+inflora'
        lambda_emb=0.01,        # Dual Fisher strength (0 = disabled)
    )
"""

import os
import math
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple
from cl_trainer_srt import SRT_Trainer

logger = logging.getLogger(__name__)


class SGWI_DualFisher_Trainer(SRT_Trainer):
    """SRT_Trainer + SGWI initialization + Dual Fisher regularization."""

    def __init__(
        self,
        model,
        args,
        train_dataset,
        cur_task_id,
        task_order,
        # SGWI params
        sgwi_mode: str = 'sgwi',        # 'sgwi', 'inflora', 'random', 'sgwi+inflora'
        lambda_emb: float = 0.0,         # Dual Fisher λ (0 = disabled)
        # SRT params (passed to SRT_Trainer)
        srt_metric_mode='hard',
        srt_shrink=True,
        srt_shrink_factor=0.1,
        srt_max_emb_samples=500,
        srt_load_path=None,
        srt_skip_forward=False,
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
            model=model,
            args=args,
            train_dataset=train_dataset,
            cur_task_id=cur_task_id,
            task_order=task_order,
            srt_metric_mode=srt_metric_mode,
            srt_shrink=srt_shrink,
            srt_shrink_factor=srt_shrink_factor,
            srt_max_emb_samples=srt_max_emb_samples,
            srt_load_path=srt_load_path,
            srt_skip_forward=srt_skip_forward,
            data_collator_replay=data_collator_replay,
            replay_dataset_dict=replay_dataset_dict,
            replay_label_dict=replay_label_dict,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            **kwargs,
        )
        self.sgwi_mode = sgwi_mode
        self.lambda_emb = lambda_emb
        self.theta_stars: Dict[int, Dict[str, torch.Tensor]] = {}
        self._current_task_mu = None  # cached for Dual Fisher

        logger.info(f"[SGWI] mode={sgwi_mode}, lambda_emb={lambda_emb}, task_id={cur_task_id}")

    # =========================================================================
    # SGWI: Override get_reg_matrix
    # =========================================================================

    def get_reg_matrix(self):
        """Override InfLoRA initialization with SGWI or other modes.

        6 Ablation Configs:
          Config 1 'inflora':       InfLoRA baseline (null-space + GPM)
          Config 2 'random':        No GPM, freeze A(kaiming), B=0
          Config 3 'full_lora':     No GPM, train A(kaiming)+B(0)
          Config 5 'sgwi_freeze_a': No GPM, SGWI→A(frozen), B=0
          Config 6 'sgwi_train_a':  No GPM, SGWI→A(trainable), B=0
          Config 4 'sgwi_full':     No GPM, SGWI→A(trainable)+B(warm)
        """
        mode = self.sgwi_mode
        logger.info(f"[SGWI] get_reg_matrix: mode={mode}, task_id={self.cur_task_id}")

        # ── Config 1: InfLoRA baseline ──────────────────────────────
        if mode == 'inflora':
            super().get_reg_matrix()
            return

        # ── All other configs: NO GPM, NO null-space ────────────────
        self._init_gpm_attrs_skip()

        # Task 0: no prior tasks → nothing to warm-init
        if self.cur_task_id == 0:
            logger.info(f"[SGWI] Task 0, mode={mode} → standard init (no prior tasks)")
            return

        # ── Config 2: random (freeze A kaiming, B=0) ────────────────
        if mode == 'random':
            logger.info("[SGWI] Config 2: random — no warm init")
            return

        # ── Config 3: full_lora (train A kaiming, B=0) ──────────────
        if mode == 'full_lora':
            logger.info("[SGWI] Config 3: full_lora — no warm init, A will be unfrozen by run_t5.py")
            return

        # ── Configs 4,5,6: need SGWI weights ────────────────────────
        srt_weights = self._compute_sgwi_weights()
        if not srt_weights:
            logger.warning("[SGWI] No SRT weights. Falling back to random init.")
            return

        logger.info(f"[SGWI] SRT weights: {srt_weights}")

        # ── Config 5: sgwi_freeze_a (SGWI→A frozen, B=0) ────────────
        if mode == 'sgwi_freeze_a':
            logger.info("[SGWI] Config 5: warm-init A only (frozen)")
            self._sgwi_init_a(srt_weights)
            return

        # ── Config 6: sgwi_train_a (SGWI→A trainable, B=0) ──────────
        if mode == 'sgwi_train_a':
            logger.info("[SGWI] Config 6: warm-init A only (will be unfrozen by run_t5.py)")
            self._sgwi_init_a(srt_weights)
            return

        # ── Config 4: sgwi_full (SGWI→A+B, both trainable) ──────────
        if mode == 'sgwi_full':
            logger.info("[SGWI] Config 4: warm-init both A and B")
            self._sgwi_init_a(srt_weights)
            self._fuse_past_lora_adapters(srt_weights)  # warm-init B
            return

        # Legacy modes
        if mode == 'sgwi':
            self._sgwi_init()
            return
        if mode == 'sgwi+inflora':
            self._sgwi_init()
            super().get_reg_matrix()
            return

        raise ValueError(f"Unknown sgwi_mode: {mode}")

    def _init_gpm_attrs_skip(self):
        """Initialize GPM attributes to safe defaults so _inner_training_loop doesn't crash.
        Sets _cur_task=0 (falsy) so GPM projection is skipped during training.
        Use for sgwi-only and random modes."""
        self._cur_task = 0
        self.feature_list = []
        self.feature_trans_list = []
        self.feature_mat = []
        self.feature_trans_mat = []
        logger.info("[SGWI] GPM attrs initialized (skip mode): _cur_task=0, feature_list=[]")

    def _sgwi_init_a(self, srt_weights: Dict[int, float]):
        """SGWI warm-init for lora_A only, via SVD of weighted past ΔW.
        
        delta_W = Σ w_s * (B_s @ A_s)  [out_dim, in_dim]
        SVD: delta_W = U @ S @ V^T
        A_new = sqrt(S[:r]) * V^T[:r, :]  — captures the important input directions
        """
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

                # Compute delta_W from past tasks
                delta_W = None
                for past_task_id, w_s in srt_weights.items():
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
                    BA = past_lora.lora_B.data.float() @ past_lora.lora_A.data.float()
                    delta_W = w_s * BA if delta_W is None else delta_W + w_s * BA

                if delta_W is None or delta_W.norm().item() < 1e-10:
                    skipped_count += 1
                    continue

                # SVD decomposition → extract top-r input directions for A
                try:
                    U, S, Vt = torch.linalg.svd(delta_W.to(device), full_matrices=False)
                    current_r = lora_cur.lora_A.data.shape[0]
                    r = min(current_r, len(S))
                    S_sqrt = torch.sqrt(S[:r] + 1e-12)
                    A_new = S_sqrt.unsqueeze(1) * Vt[:r, :]  # [r, in_dim]

                    old_norm = lora_cur.lora_A.data.norm().item()
                    lora_cur.lora_A.data[:r, :].copy_(A_new.to(lora_cur.lora_A.data.device))
                    new_norm = lora_cur.lora_A.data.norm().item()

                    if fused_count < 3:
                        logger.info(f"[SGWI-A] {name}.{lora_tag}: A_new norm={new_norm:.4f} (was {old_norm:.4f})")
                    fused_count += 1
                except Exception as e:
                    logger.warning(f"[SGWI-A] SVD failed for {name}.{lora_tag}: {e}")
                    skipped_count += 1

        logger.info(f"[SGWI-A] Warm-init A for {fused_count} modules, skipped {skipped_count}")

    def _sgwi_init(self):
        """SRT-Guided Warm Initialization for current task's LoRA."""
        logger.info(f"[SGWI] Initializing task {self.cur_task_id} from past LoRA adapters...")

        # Step 1: Get SRT weights from router signatures
        srt_weights = self._compute_sgwi_weights()
        if not srt_weights:
            logger.warning("[SGWI] No past task signatures found. Falling back to standard init.")
            super().get_reg_matrix()
            return

        logger.info(f"[SGWI] SRT weights: {srt_weights}")

        # Step 2: Load past LoRA weights and compute weighted fusion
        self._fuse_past_lora_adapters(srt_weights)

    def _compute_sgwi_weights(self) -> Dict[int, float]:
        """Compute softmax weights from SRT distances to past tasks."""
        if self.srt_router is None or len(self.srt_router.signatures) == 0:
            return {}

        # Extract current task embeddings to compute signature
        logger.info("[SGWI] Extracting current task embeddings for distance computation...")
        result = self._extract_task_embeddings(max_samples=self.srt_max_emb_samples)
        # _extract_task_embeddings returns (h_train, task_ids) tuple
        if isinstance(result, tuple):
            h_train, _ = result
        else:
            h_train = result
        if h_train is None or (hasattr(h_train, '__len__') and len(h_train) == 0):
            return {}

        current_mu = h_train.mean(dim=0).cpu().numpy()
        self._current_task_mu = current_mu

        # Compute distances to all past tasks
        distances = {}
        for task_id, sig in self.srt_router.signatures.items():
            diff = current_mu - sig.mu
            # Use pooled covariance if available, else L2
            if hasattr(self.srt_router, 'pooled_cov') and self.srt_router.pooled_cov is not None:
                try:
                    cov_inv = np.linalg.inv(self.srt_router.pooled_cov + 1e-6 * np.eye(len(diff)))
                    d = float(diff @ cov_inv @ diff)
                except np.linalg.LinAlgError:
                    d = float(np.sum(diff ** 2))
            else:
                d = float(np.sum(diff ** 2))
            distances[task_id] = d

        if len(distances) == 0:
            return {}

        # Trivial case: only 1 past task
        if len(distances) == 1:
            return {k: 1.0 for k in distances}

        # Softmax with τ = median heuristic
        d_values = list(distances.values())
        tau = float(np.median(d_values)) + 1e-8

        weights = {}
        for k, d in distances.items():
            weights[k] = math.exp(-d / tau)
        Z = sum(weights.values()) + 1e-12
        weights = {k: w / Z for k, w in weights.items()}

        return weights

    def _fuse_past_lora_adapters(self, srt_weights: Dict[int, float]):
        """Weighted fusion of past LoRA adapters → warm-init current task's lora_B.

        Architecture note (GainLoRA-InfLoRA):
          - lora_A / lora_B: SINGLE nn.Parameter per module = CURRENT task's adapter
          - previous_lora_weights_q[i] / previous_lora_weights_v[i]: past task i's LoRALayer
            with .lora_A (nn.Parameter) and .lora_B (nn.Parameter)

        Strategy:
          1. For each past task s, get B_s and A_s from previous_lora_weights
          2. Compute delta_W = Σ w_s * (B_s @ A_s)
          3. Warm-init current lora_B via least-squares: B_new = delta_W @ A_cur^+ 
        """
        model = self.model
        device = next(model.parameters()).device

        fused_count = 0
        skipped_count = 0

        for name, module in model.named_modules():
            # Look for T5Attention modules with previous_lora_weights
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

                # Compute weighted ΔW = Σ w_s * (B_s @ A_s)
                delta_W = None
                for past_task_id, w_s in srt_weights.items():
                    # past_task_id may be a string task name (from srt_router.signatures)
                    # or an int. Convert to previous_lora_weights index.
                    # In run_t5.py: previous_lora_path is "t1,t2,...,t_{n-1}" in task_order,
                    # then reversed → prev_list[0] = most recent = task_order[cur_task_id-1]
                    if isinstance(past_task_id, str):
                        try:
                            pos = self.task_order.index(past_task_id)
                            idx = (self.cur_task_id - 1) - pos
                        except ValueError:
                            logger.debug(f"[SGWI] task name '{past_task_id}' not in task_order, skipping")
                            skipped_count += 1
                            continue
                    else:
                        idx = int(past_task_id)

                    if idx < 0 or idx >= len(prev_list):
                        logger.debug(f"[SGWI] idx={idx} out of range for prev_list len={len(prev_list)}, skipping")
                        skipped_count += 1
                        continue

                    past_lora = prev_list[idx]
                    A_s = past_lora.lora_A.data.float()  # [r, in_features]
                    B_s = past_lora.lora_B.data.float()  # [out_features, r]

                    BA = B_s @ A_s  # [out_features, in_features]

                    if delta_W is None:
                        delta_W = w_s * BA
                    else:
                        delta_W = delta_W + w_s * BA

                if delta_W is None:
                    continue

                delta_W_norm = delta_W.norm().item()
                if delta_W_norm < 1e-10:
                    logger.debug(f"[SGWI] delta_W ≈ 0 for {name}.{lora_tag} (norm={delta_W_norm:.2e}), skipping")
                    skipped_count += 1
                    continue

                # Warm-init lora_B (trainable) via least-squares given frozen lora_A:
                # B_warm = delta_W @ A_cur^T @ (A_cur @ A_cur^T + εI)^{-1}
                try:
                    A_cur = lora_cur.lora_A.data.float().to(device)  # [r, in_features]
                    AtA = A_cur @ A_cur.T  # [r, r]
                    eps_mat = 1e-4 * torch.eye(A_cur.shape[0], device=device)
                    B_warm = delta_W.to(device) @ A_cur.T @ torch.linalg.inv(AtA + eps_mat)  # [out_features, r]

                    old_norm = lora_cur.lora_B.data.norm().item()
                    lora_cur.lora_B.data.copy_(B_warm.to(lora_cur.lora_B.data.device))
                    new_norm = lora_cur.lora_B.data.norm().item()

                    if fused_count < 3:
                        logger.info(f"[SGWI] {name}.{lora_tag}: delta_W={delta_W_norm:.4f}, "
                                    f"B_warm={new_norm:.4f} (was {old_norm:.4f})")
                    fused_count += 1

                except Exception as e:
                    logger.warning(f"[SGWI] lora_B warm init failed for {name}.{lora_tag}: {e}")
                    continue

        logger.info(f"[SGWI] Fused {fused_count} LoRA modules, skipped {skipped_count}")

    def _get_lora_weight(self, lora_module, attr_name, task_id):
        """Safely get LoRA weight tensor for a given task_id."""
        attr = getattr(lora_module, attr_name, None)
        if attr is None:
            return None

        # Case 1: nn.ModuleDict or dict keyed by task_id
        if isinstance(attr, (nn.ModuleDict, dict)):
            key = str(task_id) if isinstance(attr, nn.ModuleDict) else task_id
            if key in attr:
                m = attr[key]
                return m.weight.data if hasattr(m, 'weight') else m.data
            # Try integer key
            if task_id in attr:
                m = attr[task_id]
                return m.weight.data if hasattr(m, 'weight') else m.data

        # Case 2: nn.ModuleList indexed by task_id
        if isinstance(attr, nn.ModuleList):
            if task_id < len(attr):
                m = attr[task_id]
                return m.weight.data if hasattr(m, 'weight') else m.data

        # Case 3: Single nn.Linear (only for current task)
        if isinstance(attr, nn.Linear):
            return attr.weight.data

        # Case 4: raw tensor
        if isinstance(attr, torch.Tensor):
            return attr

        return None

    def _set_lora_weight(self, lora_module, attr_name, task_id, new_weight):
        """Safely set LoRA weight tensor for current task_id."""
        attr = getattr(lora_module, attr_name, None)
        if attr is None:
            return

        try:
            if isinstance(attr, (nn.ModuleDict, dict)):
                key = str(task_id) if isinstance(attr, nn.ModuleDict) else task_id
                if key in attr:
                    m = attr[key]
                    if hasattr(m, 'weight'):
                        m.weight.data.copy_(new_weight.to(m.weight.device))
                    else:
                        attr[key] = new_weight.to(next(lora_module.parameters()).device)

            elif isinstance(attr, nn.ModuleList):
                if task_id < len(attr):
                    m = attr[task_id]
                    if hasattr(m, 'weight'):
                        m.weight.data.copy_(new_weight.to(m.weight.device))

            elif isinstance(attr, nn.Linear):
                attr.weight.data.copy_(new_weight.to(attr.weight.device))

        except Exception as e:
            logger.debug(f"[SGWI] Could not set weight for {attr_name}[{task_id}]: {e}")

    # =========================================================================
    # Dual Fisher: Override compute_loss
    # =========================================================================

    def compute_loss(self, model, inputs, return_outputs=False):
        """Standard loss + Dual Fisher penalty."""
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if self.lambda_emb > 0 and self.cur_task_id > 0 and len(self.theta_stars) > 0:
            fisher_loss = self._dual_fisher_penalty()
            loss = loss + fisher_loss

        return (loss, outputs) if return_outputs else loss

    def _dual_fisher_penalty(self) -> torch.Tensor:
        """SRT-weighted L2 penalty around past task parameters."""
        total = torch.tensor(0.0, device=next(self.model.parameters()).device)

        # Get SRT weights (cached or recompute)
        srt_weights = {}
        if self.srt_router and len(self.srt_router.signatures) > 0:
            for task_id in self.theta_stars:
                if task_id in self.srt_router.signatures:
                    srt_weights[task_id] = 1.0  # uniform for now
            # Normalize
            if srt_weights:
                Z = sum(srt_weights.values())
                srt_weights = {k: v / Z for k, v in srt_weights.items()}

        if not srt_weights:
            return total

        for task_id, w_s in srt_weights.items():
            if task_id not in self.theta_stars:
                continue

            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if 'lora_' not in name:
                    continue
                if name not in self.theta_stars[task_id]:
                    continue

                theta_star = self.theta_stars[task_id][name].to(param.device)
                delta = param - theta_star
                total = total + w_s * (delta ** 2).sum()

        return self.lambda_emb * total

    # =========================================================================
    # Override on_task_end to save θ* for Dual Fisher
    # =========================================================================

    def on_task_end(self, task_id):
        """Save θ* for Dual Fisher, then do standard SRT task end."""
        # Save current task's final LoRA weights for future Dual Fisher
        logger.info(f"[SGWI] Saving θ* for task {task_id} (Dual Fisher)")
        self.theta_stars[task_id] = {}
        for name, param in self.model.named_parameters():
            if 'lora_' in name and param.requires_grad:
                self.theta_stars[task_id][name] = param.detach().clone().cpu()

        saved_count = len(self.theta_stars[task_id])
        logger.info(f"[SGWI] Saved {saved_count} parameters for task {task_id}")

        # Save theta_stars to disk for persistence
        theta_stars_path = os.path.join(self.args.output_dir, 'saved_weights', 'theta_stars.pt')
        os.makedirs(os.path.dirname(theta_stars_path), exist_ok=True)
        torch.save(self.theta_stars, theta_stars_path)

        # Standard SRT: compute signature, wire router
        super().on_task_end(task_id)

    def _load_theta_stars(self, srt_load_path: str):
        """Load previously saved θ* from checkpoint."""
        theta_path = os.path.join(srt_load_path, 'theta_stars.pt')
        if os.path.exists(theta_path):
            self.theta_stars = torch.load(theta_path, map_location='cpu')
            logger.info(f"[SGWI] Loaded θ* for {len(self.theta_stars)} past tasks from {theta_path}")
        else:
            logger.info(f"[SGWI] No θ* found at {theta_path}")
