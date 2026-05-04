"""
SRT_InfLoRA: InfLoRA with SRT routing + SGWI warm-init.

Replaces InfLoRA's task-ID cumulative LoRA + GPM/DualGPM with:
  - SRT: ZCA whitening + L2 routing on frozen backbone mean-pooled embeddings
  - SGWI: Warm-init for new task LoRA via SVD of weighted ΔW
  - Train both A and B matrices (InfLoRA freezes A)
  - No GPM/DualGPM

Pipeline per task:
  1. Train LoRA A/B + classifier for current task
  2. Extract mean-pooled frozen backbone embeddings → SRT router (ZCA + L2)
  3. SGWI warm-init for next task: ΔW = Σ_s w_s·ΔW_s → SVD → A_new, B_new
"""

import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import logging
import numpy as np
from tqdm import tqdm
import math

from methods.base import BaseLearner
from utils.toolkit import tensor2numpy, accuracy
from models.sinet_srt_inflora import SiNet_SRT, Attention_LoRA_SRT


class SRT_InfLoRA(BaseLearner):

    def __init__(self, args):
        super().__init__(args)

        self._network = SiNet_SRT(args)
        self.args = args

        self.optim = args["optim"]
        self.init_epoch = args["init_epoch"]
        self.init_lr = args["init_lr"]
        self.init_lr_decay = args["init_lr_decay"]
        self.init_weight_decay = args["init_weight_decay"]
        self.epochs = args["epochs"]
        self.lrate = args["lrate"]
        self.lrate_decay = args["lrate_decay"]
        self.batch_size = args["batch_size"]
        self.weight_decay = args["weight_decay"]
        self.num_workers = args["num_workers"]
        self.total_sessions = args["total_sessions"]
        self.dataset = args["dataset"]

        self.topk = 1
        self.class_num = self._network.class_num

        # SRT router lives on the model
        self._srt_router = self._network.srt_router

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info('Zero rehearsal: no exemplar memory')

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)

        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source='train', mode='train')
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers)
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes),
            source='test', mode='test')
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers)

        self._train(self.train_loader, self.test_loader)
        self._update_srt_router()

    def _extract_frozen_embeddings(self, loader):
        """Extract mean-pooled features from frozen backbone."""
        self._network.to(self._device)
        self._network.eval()
        emb_list, tgt_list = [], []

        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                feats = self._network.extract_frozen_vector(inputs)
            emb_list.append(feats.cpu().numpy())
            tgt_list.append(targets.numpy())

        embeddings = np.concatenate(emb_list, axis=0)     # [N, D]
        targets = np.concatenate(tgt_list, axis=0)         # [N]
        return embeddings, targets

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        # ── Freeze all, then unfreeze current task LoRA A/B + classifier ──
        for param in self._network.parameters():
            param.requires_grad_(False)

        cur_task = self._network.numtask - 1

        for name, param in self._network.named_parameters():
            if f"lora_A_k.{cur_task}" in name or f"lora_A_v.{cur_task}" in name or \
               f"lora_B_k.{cur_task}" in name or f"lora_B_v.{cur_task}" in name or \
               f"classifier_pool.{cur_task}" in name:
                param.requires_grad_(True)

        # ── SGWI warm-init for task > 0 ──
        if self._cur_task > 0:
            self._sgwi_warm_init(cur_task)

        # Debug: list trainable params
        trainable = [n for n, p in self._network.named_parameters() if p.requires_grad]
        logging.info('Trainable params for task {}: {}'.format(cur_task, trainable))

        if self._cur_task == 0:
            lr = self.init_lr
            self.run_epoch = self.init_epoch
            if self.optim == 'sgd':
                optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=lr,
                                     weight_decay=self.init_weight_decay)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.init_epoch)
            elif self.optim == 'adam':
                optimizer = optim.Adam(self._network.parameters(), lr=lr,
                                       weight_decay=self.init_weight_decay, betas=(0.9, 0.999))
                scheduler = CosineSchedule(optimizer, K=self.init_epoch)
            else:
                raise ValueError(f'Unknown optim: {self.optim}')
        else:
            lr = self.lrate
            self.run_epoch = self.epochs
            if self.optim == 'sgd':
                optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=lr,
                                     weight_decay=self.weight_decay)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
            elif self.optim == 'adam':
                optimizer = optim.Adam(self._network.parameters(), lr=lr,
                                       weight_decay=self.weight_decay, betas=(0.9, 0.999))
                scheduler = CosineSchedule(optimizer, K=self.epochs)
            else:
                raise ValueError(f'Unknown optim: {self.optim}')

        self.train_function(train_loader, test_loader, optimizer, scheduler)

        # ── Store ΔW for SGWI ──
        self._store_delta_W(cur_task)

    def _sgwi_warm_init(self, new_task: int):
        """
        SGWI warm-init for new_task's lora_A and lora_B.

        Follows the original code_srt_sgwi_v1/src/sgwi_trainer.py exactly:

        1. Extract current task embeddings → current centroid (raw space)
        2. Compute Mahalanobis distance to each past task centroid in pooled space
           (This is what the ORIGINAL uses — NOT ZCA-whitened L2)
        3. Softmax → SGWI weights (median heuristic temperature)
        4. ΔW = Σ_s w_s · ΔW_s → SVD
           - A_new = √S[:r] · V^T[:r, :]   (captures top-r input directions)
           - B_warm = ΔW @ A_cur^T @ (A_cur @ A_cur^T + εI)⁻¹  (least-squares)
        """
        rank = self.args["rank"]
        device = next(self._network.parameters()).device
        router = self._srt_router

        if router.n_tasks == 0:
            logging.info('[SGWI] Task 0: no prior tasks, skipping warm-init.')
            return

        # ── Step 1: Extract current task embeddings → raw centroid ──
        cur_emb, _ = self._extract_frozen_embeddings(self.train_loader)
        cur_centroid = cur_emb.mean(axis=0)  # raw centroid

        # ── Step 2: Compute Mahalanobis distance to each past task ──
        # Use pooled covariance if available (Mahalanobis-like distance)
        # This is exactly what the original _compute_sgwi_weights() does
        distances = []
        for t_id in sorted(router.signatures.keys()):
            sig = router.signatures[t_id]
            diff = cur_centroid - sig.mu_raw  # distance from raw centroid to raw centroid
            if router._Sigma_pool is not None:
                try:
                    cov_inv = np.linalg.inv(router._Sigma_pool + 1e-6 * np.eye(len(diff)))
                    d = float(diff @ cov_inv @ diff)
                except np.linalg.LinAlgError:
                    d = float(np.sum(diff ** 2))
            else:
                d = float(np.sum(diff ** 2))
            distances.append(d)

        # Trivial case: only 1 past task
        if len(distances) == 1:
            weights = [1.0]
        else:
            # Softmax with τ = median heuristic
            tau = float(np.median(distances)) + 1e-8
            weights = [math.exp(-d / tau) for d in distances]
            Z = sum(weights) + 1e-12
            weights = [w / Z for w in weights]

        past_task_ids = sorted(router.signatures.keys())
        srt_weights = {t_id: w for t_id, w in zip(past_task_ids, weights)}
        logging.info(f'[SGWI] Weights for task {new_task}: {[f"{t}:{w:.3f}" for t, w in srt_weights.items()]}')

        # ── Step 3: Apply SGWI to each Attention_LoRA_SRT module ──
        for module in self._network.modules():
            if not isinstance(module, Attention_LoRA_SRT):
                continue

            delta_ws = module.get_all_delta_Ws()
            if not delta_ws:
                continue

            # Weighted ΔW = Σ_s w_s · ΔW_s
            weighted_delta = None
            for s_idx, (t_id, w) in enumerate(srt_weights.items()):
                if s_idx >= len(delta_ws) or delta_ws[s_idx] is None:
                    continue
                delta = delta_ws[s_idx].to(device)
                weighted_delta = w * delta if weighted_delta is None else weighted_delta + w * delta

            if weighted_delta is None:
                continue

            delta_norm = weighted_delta.norm().item()
            if delta_norm < 1e-10:
                logging.debug(f'[SGWI] weighted_delta ≈ 0 (norm={delta_norm:.2e}), skipping')
                continue

            # ── SVD decomposition of ΔW ──
            try:
                U, S, Vt = torch.linalg.svd(weighted_delta.float(), full_matrices=False)
                r = min(rank, len(S))

                # A_new = √S[:r] · V^T[:r, :]  (top-r input directions)
                A_new = torch.sqrt(S[:r] + 1e-12).unsqueeze(1) * Vt[:r, :]

                # ── B_warm via least-squares (given current A is initialized from SVD) ──
                # B_warm = ΔW @ A^T @ (A @ A^T + εI)⁻¹
                # This finds B that best reconstructs ΔW given the A from SVD
                A_cur = A_new.to(device)  # [r, in_dim]
                AtA = A_cur @ A_cur.T    # [r, r]
                eps_mat = 1e-4 * torch.eye(A_cur.shape[0], device=device)
                B_warm = weighted_delta.to(device) @ A_cur.T @ torch.linalg.inv(AtA + eps_mat)  # [out, r]

                with torch.no_grad():
                    module.lora_A_k[new_task].weight.copy_(
                        A_new.to(module.lora_A_k[new_task].weight.device))
                    module.lora_A_v[new_task].weight.copy_(
                        A_new.to(module.lora_A_v[new_task].weight.device))
                    module.lora_B_k[new_task].weight.copy_(
                        B_warm.to(module.lora_B_k[new_task].weight.device))
                    module.lora_B_v[new_task].weight.copy_(
                        B_warm.to(module.lora_B_v[new_task].weight.device))

                logging.info(f'[SGWI] Task {new_task} warm-init: r={r}, ΔW_norm={delta_norm:.4f}, '
                             f'S[0]={S[0].item():.4f}, A_norm={A_new.norm().item():.4f}, '
                             f'B_warm_norm={B_warm.norm().item():.4f}')
            except Exception as e:
                logging.warning(f'[SGWI] Warm-init failed for module: {e}')

    def _store_delta_W(self, task: int):
        """Store ΔW = B @ A for trained task."""
        for module in self._network.modules():
            if isinstance(module, Attention_LoRA_SRT):
                module.store_delta_W(task)
        logging.info(f'[SRT] Stored ΔW for task {task}')

    def _update_srt_router(self):
        """
        Extract mean-pooled frozen backbone embeddings and register with SRT router.
        SRT router stores {μ_t, Σ_t}, fits ZCA whitening, and re-whitens all tasks.
        """
        cur_task = self._network.numtask - 1
        embeddings, _ = self._extract_frozen_embeddings(self.train_loader)
        self._srt_router.add_task(cur_task, embeddings)

        s = self._srt_router.summary()
        logging.info(f'[SRT] Router: n_tasks={s["n_tasks"]}, pool_n={s["pool_n"]}, '
                     f'avg_PaR={s["avg_par"]}, mode={s["mode"]}')

    def train_function(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.run_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.eval()
            losses = 0.
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                # Only current task classes
                mask = (targets >= self._known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                targets = torch.index_select(targets, 0, mask) - self._known_classes

                outputs = self._network(inputs)
                loss = F.cross_entropy(outputs['logits'], targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()
                _, preds = torch.max(outputs['logits'], dim=1)
                correct += preds.eq(targets).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch + 1, self.run_epoch,
                losses / max(len(train_loader), 1), train_acc)
            prog_bar.set_description(info)

        logging.info(info)

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred, y_true, self._known_classes, self.class_num)
        ret['grouped'] = grouped
        ret['top1'] = grouped['total']
        return ret

    def _eval_cnn(self, loader):
        """
        Evaluate using SRT routing on mean-pooled frozen backbone features.
        """
        self._network.eval()
        y_pred, y_true = [], []
        y_pred_with_task = []

        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)
            batch_size = inputs.shape[0]

            with torch.no_grad():
                # Extract mean-pooled frozen backbone features
                frozen_feats = self._network.extract_frozen_vector(inputs)  # [B, D]
                frozen_np = frozen_feats.cpu().numpy()

                # SRT route: argmin L2 in whitened space
                pred_tasks_np = self._srt_router.route(frozen_np)  # [B] numpy
                pred_tasks = torch.from_numpy(pred_tasks_np).to(self._device)  # [B]

                # Compute logits for each predicted task
                outputs = []
                for b in range(batch_size):
                    t = min(pred_tasks[b].item(), self._network.numtask - 1)
                    out = self._network.image_encoder(inputs[b:b+1], task=t)
                    out = self._network.classifier_pool[t](out['features'])
                    outputs.append(out)
                outputs = torch.cat(outputs, dim=0)  # [B, class_num]

                # Global prediction: local_class + task_offset
                task_offset = pred_tasks.cpu() * self.class_num
                task_offset = task_offset.clamp(max=(self._network.numtask - 1) * self.class_num)
                local_pred = outputs.argmax(dim=1)    # [B]
                predicts = (local_pred + task_offset).view(-1)  # [B] global class

                y_pred.append(predicts.cpu().numpy())
                y_true.append(targets.cpu().numpy())

                # With-task evaluation (known task boundary, GT masking)
                outputs_with_task = torch.zeros_like(outputs)
                for idx, i in enumerate(targets // self.class_num):
                    outputs_with_task[idx] = outputs[idx]
                predicts_with_task = outputs_with_task.argmax(dim=1)
                predicts_with_task = predicts_with_task + (targets // self.class_num) * self.class_num
                y_pred_with_task.append(predicts_with_task.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_pred_with_task), np.concatenate(y_true)

    def eval_task(self):
        cnn_accy, cnn_accy_with_task, y_true = self._eval_cnn(self.test_loader)
        cnn_result = self._evaluate(cnn_accy, y_true)
        cnn_result_with_task = self._evaluate(cnn_accy_with_task, y_true)
        return cnn_result, cnn_result_with_task, None, 0.0


class CosineSchedule:
    """Cosine annealing schedule."""
    def __init__(self, optimizer, K: int):
        self.optimizer = optimizer
        self.K = K
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = 0.5 * (1 + math.cos(math.pi * self.current_step / self.K))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr
