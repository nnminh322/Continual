"""
SpecRoute Trainer: Custom Seq2SeqTrainer for SpecRoute continual learning.

Key differences from GainLoRA_InfLoRA_Trainer:
- No trans_input GPM constraints (routing is parameter-free spectral projection)
- No GPM projection on trans_input/prompt_key after optimizer step
- No memory replay (KL loss on gating — routing has no learned parameters)
- Constant threshold for GPM (Elastic Subspace Allocation)
- Simplified optimizer (no special lr for trans_input)
- C4: Preconditioned gradient (AA^T)^{-1/2} + spectral entropy regularization
"""

import gc
import math
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import GenerationConfig
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.trainer import *
from transformers.trainer_pt_utils import (
    nested_truncate, nested_concat, nested_numpify,
    find_batch_size,
)
try:
    from transformers.trainer_pt_utils import denumpify_detensorize
except ImportError:
    from transformers.trainer_utils import denumpify_detensorize
from transformers.trainer_callback import TrainerCallback

# ShardedDDPOption was removed in transformers>=4.38
try:
    from transformers.trainer import ShardedDDPOption
except ImportError:
    class ShardedDDPOption:
        SIMPLE = "simple"
import numpy as np

from cl_collator import SUPPORTED_DECODER_MODELS, check_model
from cl_dataset import ANSWER_PREFIX
import cupy as cp
from torch.utils.dlpack import to_dlpack, from_dlpack
from cupy import fromDlpack
def skip_instructions(model, predictions_ids, tokenizer, ignore_idx=-100):
    predictions_ids = np.where(predictions_ids == ignore_idx, tokenizer.pad_token_id, predictions_ids)
    predictions = tokenizer.batch_decode(
        predictions_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    final_predictions = []
    if check_model(model.config._name_or_path, SUPPORTED_DECODER_MODELS):
        for pred in predictions:
            if ANSWER_PREFIX in pred:
                splits = pred.split(ANSWER_PREFIX)
                final_predictions.append(splits[-1].strip())
            else:
                final_predictions.append('')
    else:
        final_predictions = predictions
    return final_predictions


class DenserEvalCallback(TrainerCallback):
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        log_eval_steps = [1, 50, 100, 200]
        if args.logging_strategy == IntervalStrategy.STEPS and state.global_step in log_eval_steps:
            control.should_log = True
        if args.evaluation_strategy == IntervalStrategy.STEPS and state.global_step in log_eval_steps:
            control.should_evaluate = True
        return control


class PeriodicGCCallback(TrainerCallback):
    """Periodically call gc.collect() to return freed Python memory to OS."""
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % 100 == 0:
            gc.collect()
        return control

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        gc.collect()
        return control


class TransInputGPMCallback(TrainerCallback):
    """V10a: Apply GPM projection to trans_input and prompt_key after optimizer step."""
    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if getattr(self.trainer, "cur_task_id", 0) > 1 and getattr(self.trainer.model.encoder, "routing_mode", "") == "learned":
            from copy import deepcopy
            self.trainer._old_trans_input_0 = deepcopy(self.trainer.model.encoder.trans_input[0].weight.detach())
            self.trainer._old_trans_input_1 = deepcopy(self.trainer.model.encoder.trans_input[2].weight.detach())
            self.trainer._old_prompt_key = deepcopy(self.trainer.model.encoder.prompt_key.detach())

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if getattr(self.trainer, "cur_task_id", 0) > 1 and getattr(self.trainer.model.encoder, "routing_mode", "") == "learned":
            if not hasattr(self.trainer, "feature_trans_mat") or not self.trainer.feature_trans_mat:
                return
            
            from copy import deepcopy
            new_trans_input_0 = deepcopy(self.trainer.model.encoder.trans_input[0].weight.detach())
            new_trans_input_1 = deepcopy(self.trainer.model.encoder.trans_input[2].weight.detach())
            new_trans_input_0norm = new_trans_input_0.norm(dim=1, keepdim=True).clamp(min=1e-12)
            new_trans_input_1norm = new_trans_input_1.norm(dim=1, keepdim=True).clamp(min=1e-12)

            new_prompt_key = deepcopy(self.trainer.model.encoder.prompt_key.detach())
            new_prompt_key_norm = new_prompt_key.norm(dim=1, keepdim=True).clamp(min=1e-12)
            
            old_trans_input_0 = self.trainer._old_trans_input_0
            old_trans_input_1 = self.trainer._old_trans_input_1
            old_prompt_key = self.trainer._old_prompt_key
            
            for index in self.trainer.feature_trans_mat[0].keys():
                new_trans_input_0[:,index*self.trainer.model.encoder.step:(index+1)*self.trainer.model.encoder.step] = self.trainer.model.encoder.trans_input[0].weight.detach()[:,index*self.trainer.model.encoder.step:(index+1)*self.trainer.model.encoder.step] - torch.mm(self.trainer.model.encoder.trans_input[0].weight.detach()[:,index*self.trainer.model.encoder.step:(index+1)*self.trainer.model.encoder.step]-old_trans_input_0[:,index*self.trainer.model.encoder.step:(index+1)*self.trainer.model.encoder.step], self.trainer.feature_trans_mat[0][index])
                new_prompt_key[:,index*self.trainer.model.encoder.step:(index+1)*self.trainer.model.encoder.step] = self.trainer.model.encoder.prompt_key.detach()[:,index*self.trainer.model.encoder.step:(index+1)*self.trainer.model.encoder.step] - torch.mm(self.trainer.model.encoder.prompt_key.detach()[:,index*self.trainer.model.encoder.step:(index+1)*self.trainer.model.encoder.step]-old_prompt_key[:,index*self.trainer.model.encoder.step:(index+1)*self.trainer.model.encoder.step], self.trainer.feature_trans_mat[2][index])
            new_trans_input_1 = self.trainer.model.encoder.trans_input[2].weight.detach() - torch.mm(self.trainer.model.encoder.trans_input[2].weight.detach()-old_trans_input_1, self.trainer.feature_trans_mat[1])

            new_trans_input_0 = torch.nan_to_num(
                new_trans_input_0.float() * new_trans_input_0norm.float() / new_trans_input_0.float().norm(dim=1, keepdim=True).clamp_min(1e-12),
                nan=0.0, posinf=0.0, neginf=0.0).to(new_trans_input_0.dtype)
            new_trans_input_1 = torch.nan_to_num(
                new_trans_input_1.float() * new_trans_input_1norm.float() / new_trans_input_1.float().norm(dim=1, keepdim=True).clamp_min(1e-12),
                nan=0.0, posinf=0.0, neginf=0.0).to(new_trans_input_1.dtype)
            new_prompt_key = torch.nan_to_num(
                new_prompt_key.float() * new_prompt_key_norm.float() / new_prompt_key.float().norm(dim=1, keepdim=True).clamp_min(1e-12),
                nan=0.0, posinf=0.0, neginf=0.0).to(new_prompt_key.dtype)

            self.trainer.model.encoder.trans_input[0].weight.data.copy_(new_trans_input_0)
            self.trainer.model.encoder.trans_input[2].weight.data.copy_(new_trans_input_1)
            self.trainer.model.encoder.prompt_key.data.copy_(new_prompt_key)
        return control


class SpecRoute_Trainer(Seq2SeqTrainer):

    def __init__(self, model, args, train_dataset, cur_task_id, task_order,
                 eval_dataset=None, tokenizer=None, data_collator=None,
                 compute_metrics=None, callbacks=None,
                 lambda_entropy=0.0, use_preconditioning=False,
                 precond_eps=1e-6, entropy_warmup_ratio=0.1,
                 n_batches_c5=100, previous_lora_path=None):
        self.previous_lora_path = previous_lora_path
        if callbacks is None:
            callbacks = []
        callbacks.append(TransInputGPMCallback(self))
        super().__init__(
            model=model, args=args, train_dataset=train_dataset,
            eval_dataset=eval_dataset, tokenizer=tokenizer,
            data_collator=data_collator, compute_metrics=compute_metrics,
            callbacks=callbacks
        )
        self.task_order = task_order
        self.cur_task_id = cur_task_id
        self._grad_check_done = False
        # C4.1: Gradient preconditioning (C4.2 entropy reg removed in V7)
        self.lambda_entropy = 0.0  # disabled in V7 — conflicts with C5 philosophy
        self.use_preconditioning = use_preconditioning
        self.precond_eps = precond_eps
        self.entropy_warmup_ratio = entropy_warmup_ratio
        self._precond_matrices = {}
        # C5: Data-Informed Subspace Initialization
        self.n_batches_c5 = n_batches_c5
        self._task_covariance = []  # list of {chunk_index: cov_tensor} per layer

    def _save(self, output_dir=None, state_dict=None):
        # T5 shared embeddings are incompatible with safetensors; force pytorch format
        old = getattr(self.args, 'save_safetensors', True)
        self.args.save_safetensors = False
        try:
            super()._save(output_dir=output_dir, state_dict=state_dict)
        finally:
            self.args.save_safetensors = old

    # ================================================================
    # C4: Spectrally-Conditioned LoRA Training
    # ================================================================

    def precompute_preconditioners(self):
        """Precompute (AA^T + eps*I)^{-1/2} for each LoRA-B layer.
        Called once after get_reg_matrix() projects A into null-space."""
        if not self.use_preconditioning:
            return
        self._precond_matrices = {}
        for module in self.model.modules():
            if not (hasattr(module, 'lora_q') and hasattr(module, 'lora_v')):
                continue
            for lora in [module.lora_q, module.lora_v]:
                A = lora.lora_A.data.float()          # [r, d_in]
                AAt = A @ A.T                          # [r, r]
                AAt.add_(torch.eye(AAt.size(0), device=AAt.device) * self.precond_eps)
                eigvals, eigvecs = torch.linalg.eigh(AAt)
                inv_sqrt = eigvecs @ torch.diag(eigvals.clamp(min=1e-12).pow(-0.5)) @ eigvecs.T
                self._precond_matrices[id(lora.lora_B)] = inv_sqrt.to(lora.lora_B.data.device)
        print(f"[C4] Precomputed {len(self._precond_matrices)} preconditioner matrices")

    # NOTE: _compute_spectral_entropy_loss() removed in V7.
    # Entropy regularization conflicts with C5 data-informed init philosophy.
    # lambda_entropy is hardcoded to 0.0 and this method was dead code.

    def _apply_preconditioning(self):
        """Apply (AA^T + eps*I)^{-1/2} preconditioner to lora_B gradients after backward."""
        if not self.use_preconditioning:
            return
        for module in self.model.modules():
            if not (hasattr(module, 'lora_q') and hasattr(module, 'lora_v')):
                continue
            for lora in [module.lora_q, module.lora_v]:
                precond = self._precond_matrices.get(id(lora.lora_B))
                if precond is not None and lora.lora_B.grad is not None:
                    lora.lora_B.grad.data = lora.lora_B.grad.data @ precond
                    # Guard against NaN from QR/SVD backward
                    lora.lora_B.grad.data.nan_to_num_(nan=0.0)

    def training_step(self, model, inputs, **kwargs):
        """CE training step + C4.1 gradient preconditioning.
        C4.2 spectral entropy regularization removed in V7 (conflicts with C5)."""
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if self.args.gradient_accumulation_steps > 1 and not self.is_deepspeed_enabled:
            loss = loss / self.args.gradient_accumulation_steps

        if self.is_deepspeed_enabled:
            self.accelerator.backward(loss)
        else:
            loss.backward()

        # C4: Apply spectral preconditioning to lora_B gradients
        self._apply_preconditioning()

        # One-time gradient check after first backward
        if not self._grad_check_done:
            self._grad_check_done = True
            n_with_grad, n_zero_grad, n_none_grad = 0, 0, 0
            sample_name, sample_norm = None, None
            for name, p in self.model.named_parameters():
                if p.requires_grad:
                    if p.grad is not None:
                        gn = p.grad.norm().item()
                        if gn > 0:
                            n_with_grad += 1
                            if sample_name is None:
                                sample_name, sample_norm = name, gn
                        else:
                            n_zero_grad += 1
                    else:
                        n_none_grad += 1
            print("=" * 60)
            print(f"[GRAD CHECK] params with grad>0: {n_with_grad}, "
                  f"grad==0: {n_zero_grad}, grad=None: {n_none_grad}")
            if sample_name:
                print(f"[GRAD CHECK] sample: {sample_name} grad_norm={sample_norm:.6e}")
            else:
                print("[GRAD CHECK] WARNING: NO trainable param has non-zero gradient!")
            print("=" * 60)

        return loss.detach()

    # ================================================================
    # C5: Data-Informed Subspace Initialization
    # ================================================================

    def pre_task_data_collection(self):
        """C5: Collect activation covariance for the current task.
        Must be called BEFORE get_reg_matrix().
        Stores per-layer, per-chunk covariance matrices in self._task_covariance.
        These are used in get_reg_matrix() to initialize A_t optimally.
        """
        # Reset module covariance accumulators and enable collection
        for module in self.model.modules():
            if hasattr(module, 'get_feature'):
                module.get_feature = True
                module.stage = 0
                for index in range(module.index):
                    module.matrix[index] = torch.zeros(
                        module.step, module.step, device='cuda'
                    )
                    module.n_matrix[index] = 0

        print(f'[C5] Collecting activation covariance ({self.n_batches_c5} batches)...')
        train_dataloader = self.get_train_dataloader()
        if isinstance(train_dataloader, DataLoader) and isinstance(
            train_dataloader.sampler, DistributedSampler
        ):
            train_dataloader.sampler.set_epoch(42)

        with torch.no_grad():
            for step, inputs in enumerate(train_dataloader):
                if step >= self.n_batches_c5:
                    break
                inputs = self._prepare_inputs(inputs)
                # Drop labels — we only want activations, not loss
                inputs.pop('labels', None)
                self.model(**inputs)

        # Harvest and store covariance
        self._task_covariance = []
        for module in self.model.modules():
            if hasattr(module, 'get_feature'):
                cov = {}
                for index in range(module.index):
                    cov[index] = module.matrix[index].detach().float().cuda()
                self._task_covariance.append(cov)
                module.get_feature = False
                module.stage = 0

        print(f'[C5] Covariance collected for {len(self._task_covariance)} layers.')

    def load_previous_reg_matrix(self):
        """Load LoRA GPM bases from previous task. Also load trans_input GPM if learned routing."""
        reg_matrix = []
        reg_trans_matrix = []
        log_path = os.path.dirname(self.args.output_dir)
        local_dir = os.path.basename(self.args.output_dir)

        # CRITICAL: Task 1 (cur_task_id=0) has NO previous task. Return empty immediately.
        # Without this guard, stale directories from previous experiment runs
        # (e.g., 0-yelp from a crashed run) can be mistakenly loaded as "previous task".
        if self.cur_task_id == 0:
            print(f"[GPM] Task 1 (cur_task_id=0): no previous task to load.")
            return reg_matrix, reg_trans_matrix, -1

        # If explicit path provided, use it (comma separated for multiple past tasks)
        if hasattr(self, "previous_lora_path") and self.previous_lora_path:
            previous_lora_list = self.previous_lora_path.split(',')
            # InfLoRA GPM bases are cumulative; we only need the bases from the immediately preceding task
            # because InfLoRA's loading logic in ROOT/GainLoRA builds upon them.
            # Actually, the specroute implementation loads reg_{i}.pt from the PREVIOUS task only.
            last_task_path = previous_lora_list[-1]
            
            i = 0
            for module in self.model.modules():
                if hasattr(module, 'get_feature'):
                    path = os.path.join(last_task_path, "reg_{}.pt".format(i))
                    if os.path.exists(path):
                        mat = torch.load(path, map_location='cpu')
                        if torch.isnan(mat).any() or torch.isinf(mat).any():
                            mat = torch.nan_to_num(mat, nan=0.0)
                        reg_matrix.append(mat)
                    i += 1
            if getattr(self.model.encoder, "routing_mode", "") == "learned":
                for name in ['reg_0.pt', 'reg_1.pt', 'reg_2.pt']:
                    path = os.path.join(last_task_path, 'trans_input', name)
                    if os.path.exists(path):
                        mat = torch.load(path, map_location='cpu', weights_only=True)
                        if torch.isnan(mat).any() or torch.isinf(mat).any():
                            mat = torch.nan_to_num(mat, nan=0.0)
                        reg_trans_matrix.append(mat)
            
            print(f"[GPM] Loaded bases from {last_task_path}")
            return reg_matrix, reg_trans_matrix, len(previous_lora_list) - 1

        all_dirs = os.listdir(log_path)
        for all_dir in all_dirs:
            if not os.path.isdir(os.path.join(log_path, all_dir)):
                continue
            if eval(all_dir.split('-')[0]) == eval(local_dir.split('-')[0]) - 1:
                i = 0
                for module in self.model.modules():
                    if hasattr(module, 'get_feature'):
                        path = os.path.join(os.path.join(log_path, all_dir), "reg_{}.pt".format(i))
                        mat = torch.load(path, map_location='cpu')
                        if torch.isnan(mat).any() or torch.isinf(mat).any():
                            print(f'[GPM] WARNING: {path} contains NaN/Inf. Cleaning to 0.')
                            mat = torch.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
                        reg_matrix.append(mat)
                        i += 1
                if getattr(self.model.encoder, "routing_mode", "") == "learned":
                    for name in ['reg_0.pt', 'reg_1.pt', 'reg_2.pt']:
                        path = os.path.join(log_path, all_dir, 'trans_input', name)
                        if os.path.exists(path):
                            mat = torch.load(path, map_location='cpu', weights_only=True)
                            if torch.isnan(mat).any() or torch.isinf(mat).any():
                                print(f'[GPM] WARNING: {path} contains NaN/Inf. Cleaning.')
                                mat = torch.nan_to_num(mat, nan=0.0)
                            reg_trans_matrix.append(mat)

                print(os.path.join(log_path, all_dir))
                print(len(reg_matrix))
                break
        return reg_matrix, reg_trans_matrix, eval(local_dir.split('-')[0]) - 1

    def get_reg_matrix(self):
        """
        Project current LoRA A into null-space of old tasks' GPM bases.
        No prompt_key/trans_input operations.
        """
        self.feature_list, self.feature_trans_list, self._cur_task = self.load_previous_reg_matrix()

        if len(self.feature_list) == 0:
            # First task: no constraints
            return

        if getattr(self.model.encoder, "routing_mode", "") == "learned":
            self.feature_trans_mat = []
            for i in range(len(self.feature_trans_list)):
                if i == 1:
                    self.feature_trans_mat.append(torch.mm(self.feature_trans_list[i], self.feature_trans_list[i].T).to("cuda:0"))
                else:
                    feature_trans_mat = {}
                    for index in self.feature_trans_list[i].keys():
                        feature_trans_mat[index] = torch.mm(self.feature_trans_list[i][index], self.feature_trans_list[i][index].T).to("cuda:0")
                    self.feature_trans_mat.append(feature_trans_mat)

        # Compute projection matrices for LoRA GPM
        self.feature_mat, i = [], 0
        for name, module in self.model.named_modules():
            if hasattr(module, 'get_feature'):
                feature_mat = {}
                for index in self.feature_list[i].keys():
                    feature_mat[index] = torch.mm(
                        self.feature_list[i][index],
                        self.feature_list[i][index].T
                    ).to("cuda:0")
                self.feature_mat.append(feature_mat)

                # C5: Data-Informed Subspace Initialization
                # Replace random A_t with optimal subspace from Constrained PCA.
                # Eigenvectors of Q@C_t@Q are in null(P_old) by construction.
                if self._task_covariance and i < len(self._task_covariance):
                    r = module.lora_q.lora_A.data.shape[0]  # LoRA rank
                    for index in self.feature_list[i].keys():
                        C_t   = self._task_covariance[i][index]       # [step, step]
                        P_old = feature_mat[index]                     # [step, step]
                        Q     = torch.eye(module.step, device=P_old.device) - P_old
                        C_tilde = Q @ C_t.to(P_old.device) @ Q
                        # Enforce symmetry and add tiny jitter for numerical stability
                        C_tilde = (C_tilde + C_tilde.T) * 0.5
                        # Guard against NaNs or Infs that might cause linalg.eigh to fail
                        if torch.isnan(C_tilde).any() or torch.isinf(C_tilde).any():
                            print(f'[C5] WARNING: Layer {i+1} index {index} contains NaN/Inf in C_tilde. Cleaning.')
                            C_tilde = torch.nan_to_num(C_tilde, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        # Add tiny diagonal jitter to prevent ill-conditioned matrix errors (error code 641)
                        # Scaling jitter to mean magnitude of C_tilde
                        jitter_scale = 1e-10 * (C_tilde.abs().mean() + 1e-12)
                        C_tilde += jitter_scale * torch.eye(C_tilde.shape[0], device=C_tilde.device)
                        
                        try:
                            # eigh returns ascending eigenvalues; take last r (largest)
                            eigvals, eigvecs = torch.linalg.eigh(C_tilde.float())
                        except torch._C._LinAlgError as e:
                            print(f'[C5] WARNING: Layer {i+1} index {index} - linalg.eigh failed: {e}. Falling back.')
                            continue

                        # Fallback: if null-space signal is degenerate, keep Kaiming init
                        if eigvals[-1].item() < 1e-6:
                            print(f'[C5] Layer {i+1} index {index}: max_eigval={eigvals[-1].item():.2e} < 1e-6, fallback to Kaiming+InfLoRA')
                            continue
                        top_eigvecs = eigvecs[:, -r:].flip(dims=[1])  # [step, r]
                        A_init = top_eigvecs.T  # [r, step]
                        dtype  = module.lora_q.lora_A.data.dtype
                        sl     = slice(index * module.step, (index + 1) * module.step)
                        module.lora_q.lora_A.data[:, sl].copy_(A_init.to(dtype))
                        module.lora_v.lora_A.data[:, sl].copy_(A_init.to(dtype))
                    print(f'[C5] Layer {i+1}: A_t initialized via Constrained PCA.')

                # InfLoRA null-space projection (near no-op after C5, still applied
                # for numerical correctness and to enforce InfLoRA invariant exactly)
                for index in self.feature_list[i].keys():
                    module.lora_q.lora_A.data[:, index*module.step:(index+1)*module.step].copy_(
                        module.lora_q.lora_A.data[:, index*module.step:(index+1)*module.step]
                        - torch.mm(
                            module.lora_q.lora_A.data[:, index*module.step:(index+1)*module.step],
                            feature_mat[index]
                        )
                    )
                    module.lora_v.lora_A.data[:, index*module.step:(index+1)*module.step].copy_(
                        module.lora_v.lora_A.data[:, index*module.step:(index+1)*module.step]
                        - torch.mm(
                            module.lora_v.lora_A.data[:, index*module.step:(index+1)*module.step],
                            feature_mat[index]
                        )
                    )
                # V8 C5 normalization: scale rows to have norm = sqrt(3) (matching InfLoRA convention)
                # Eigenvectors from C5 are already orthonormal unit vectors in null-space.
                # We scale magnitude to sqrt(3) to match InfLoRA's Kaiming-like initialization scale.
                module.lora_q.lora_A.data *= (math.sqrt(3) / module.lora_q.lora_A.data.norm(dim=1, keepdim=True).clamp(min=1e-8))
                module.lora_v.lora_A.data *= (math.sqrt(3) / module.lora_v.lora_A.data.norm(dim=1, keepdim=True).clamp(min=1e-8))
                i += 1

        # Free GPM bases — no longer needed after projection
        del self.feature_list, self.feature_mat
        self.feature_list, self.feature_mat = [], []
        gc.collect()
        torch.cuda.empty_cache()

        return

    def get_repsentation(self):
        """
        Collect LoRA input covariance and compute GPM bases via SVD.
        For V10a (learned routing), also collect trans_input covariance.
        """
        self.feature_list, self.feature_trans_list, self._cur_task = self.load_previous_reg_matrix()

        train_dataloader = self.get_train_dataloader()
        if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(1998)
        elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
            train_dataloader.dataset.set_epoch(1998)

        for name, module in self.model.named_modules():
            if hasattr(module, 'get_feature'):
                module.get_feature = True
                module.stage = 0

        # V10a: enable trans_input covariance collection
        if getattr(self.model.encoder, "routing_mode", "") == "learned":
            self.model.encoder.get_chunk(self.args.chunk)
            self.model.encoder.get_trans_feature = True

        print('begin get representation')
        with torch.no_grad():
            for step, inputs in enumerate(train_dataloader):
                inputs = self._prepare_inputs(inputs)
                if self.label_smoother is not None and "labels" in inputs:
                    labels = inputs.pop("labels")
                else:
                    labels = None
                outputs = self.model(**inputs)
                if step >= 200:  # 200 batches for stable GPM SVD (IDEA §9)
                    break
        print('end get representation')

        # V10a: disable trans_input collection after forward pass
        if getattr(self.model.encoder, "routing_mode", "") == "learned":
            self.model.encoder.get_trans_feature = False

        # Collect LoRA covariance matrices
        mat_list = []
        for name, module in self.model.named_modules():
            if hasattr(module, 'get_feature'):
                merged_tensor = {}
                for index in range(module.index):
                    merged_tensor[index] = module.matrix[index].cuda().float()
                mat_list.append(merged_tensor)
                module.get_feature = False
                module.stage = 0

        # ESA: importance-weighted dynamic threshold
        # As more tasks accumulate, threshold increases toward 1.0,
        # allocating smaller subspace to later tasks (fair capacity budget)
        total_sessions = len(self.task_order)
        threshold = (1.0 - self.args.threshold) * self._cur_task / total_sessions + self.args.threshold
        print(f'Threshold (ESA dynamic): {threshold:.6f} (task {self._cur_task}/{total_sessions}, base={self.args.threshold})')

        if len(self.feature_list) == 0:
            # First task: compute GPM bases from scratch
            for i in range(len(mat_list)):
                activation = mat_list[i]
                feature = {}
                for index in activation.keys():
                    U, S, Vh = cp.linalg.svd(fromDlpack(to_dlpack(activation[index])), full_matrices=False)
                    U = from_dlpack(U.toDlpack())
                    S = from_dlpack(S.toDlpack())
                    sval_total = (S**2).sum()
                    sval_ratio = (S**2) / sval_total
                    r = torch.sum(torch.cumsum(sval_ratio, dim=0) < threshold)
                    feature[index] = U[:, 0:max(r, 1)]
                self.feature_list.append(feature)
        else:
            # Subsequent tasks: update GPM bases
            for i in range(len(mat_list)):
                activation = mat_list[i]
                for index in activation.keys():
                    U1, S1, Vh1 = cp.linalg.svd(fromDlpack(to_dlpack(activation[index])), full_matrices=False)
                    sval_total = (S1**2).sum()

                    # Projected Representation
                    act_hat = (
                        fromDlpack(to_dlpack(activation[index]))
                        - cp.dot(
                            cp.dot(
                                fromDlpack(to_dlpack(self.feature_list[i][index])),
                                fromDlpack(to_dlpack(self.feature_list[i][index].T))
                            ),
                            fromDlpack(to_dlpack(activation[index]))
                        )
                    )
                    U, S, Vh = cp.linalg.svd(act_hat, full_matrices=False)
                    sval_hat = (S**2).sum()
                    sval_ratio = (S**2) / sval_total
                    accumulated_sval = (sval_total - sval_hat) / sval_total

                    r = 0
                    for ii in range(sval_ratio.shape[0]):
                        if accumulated_sval < threshold:
                            accumulated_sval += sval_ratio[ii]
                            r += 1
                        else:
                            break
                    if r == 0:
                        print('Skip Updating GPM for layer: {}'.format(i + 1))
                        continue

                    # Update GPM
                    Ui = cp.hstack((fromDlpack(to_dlpack(self.feature_list[i][index])), U[:, 0:r]))
                    if Ui.shape[1] > Ui.shape[0]:
                        self.feature_list[i][index] = from_dlpack(Ui[:, 0:Ui.shape[0]].toDlpack())
                    else:
                        self.feature_list[i][index] = from_dlpack(Ui.toDlpack())

        # Collect trans_input GPM bases if learned routing
        if getattr(self.model.encoder, "routing_mode", "") == "learned":
            mat_trans_list = []
            if self.model.encoder.matrix_trans_2.sum() != 0:
                mat_trans_list.append(self.model.encoder.matrix_trans_1)
                mat_trans_list.append(self.model.encoder.matrix_trans_2)
                mat_trans_list.append(self.model.encoder.matrix_trans_3)
                
                self.feature_trans_list, self.feature_trans_mat = [], []
                for i in range(len(mat_trans_list)):
                    if i == 1:
                        U, S, Vh = torch.linalg.svd(mat_trans_list[i].data, full_matrices=False)
                        sval_total = (S**2).sum()
                        sval_ratio = (S**2)/sval_total
                        r = np.sum(np.cumsum(sval_ratio.cpu().numpy()) < self.args.transthreshold) + 1
                        self.feature_trans_list.append(U[:,0:r].float())
                    else:
                        feature_trans_list, feature_trans_mat = {}, {}
                        for index in mat_trans_list[i].keys():
                            U, S, Vh = torch.linalg.svd(mat_trans_list[i][index].data, full_matrices=False)
                            sval_total = (S**2).sum()
                            sval_ratio = (S**2)/sval_total
                            r = np.sum(np.cumsum(sval_ratio.cpu().numpy()) < self.args.transthreshold) + 1
                            feature_trans_list[index] = U[:,0:r].float()
                        self.feature_trans_list.append(feature_trans_list)

        print('-' * 40)
        print('Gradient Constraints Summary')
        print('-' * 40)
        for i in range(len(self.feature_list)):
            for index in range(self.args.chunk):
                print('Layer {} Index {} : {}/{}'.format(
                    i + 1, index + 1,
                    self.feature_list[i][index].shape[1],
                    self.feature_list[i][index].shape[0]
                ))
        print('-' * 40)

        # Save LoRA GPM bases
        for i in range(len(self.feature_list)):
            torch.save(self.feature_list[i], os.path.join(self.args.output_dir, 'reg_{}.pt'.format(i)))

        # Save trans_input GPM bases
        if getattr(self.model.encoder, "routing_mode", "") == "learned" and hasattr(self, "feature_trans_list"):
            os.makedirs(os.path.join(self.args.output_dir, 'trans_input'), exist_ok=True)
            for i in range(len(self.feature_trans_list)):
                torch.save(self.feature_trans_list[i], os.path.join(self.args.output_dir, 'trans_input', 'reg_{}.pt'.format(i)))
                
    # training_step: removed — base Seq2SeqTrainer handles it correctly.
    # SpecRoute has no memory replay or custom training_step logic.

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.create_optimizer()
        if is_sagemaker_mp_enabled():
            _is_post_1_10 = globals().get('IS_SAGEMAKER_MP_POST_1_10', False)
            if _is_post_1_10 and smp.state.cfg.fp16:
                optimizer = self.optimizer.optimizer
            else:
                optimizer = self.optimizer
        else:
            optimizer = self.optimizer
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)

    def create_optimizer(self):
        """
        Simplified optimizer: no special lr for trans_input (doesn't exist in SpecRoute).
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            print("Using Same Learning Rate for All Modules (SpecRoute)")
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            # sharded_ddp was removed in transformers>=4.38; use getattr for compat
            _sharded_ddp = getattr(self, 'sharded_ddp', None)
            if _sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                            logger.info(f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                    logger.info(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        args = self.args
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        if args.deepspeed and not self.is_deepspeed_enabled:
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None,
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size
        logger.info(f"***** Running {description} *****")
        if has_length(dataloader.dataset):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()
        self.callback_handler.eval_dataloader = dataloader
        eval_dataset = dataloader.dataset

        if args.past_index >= 0:
            self._past = None

        losses_host = None
        preds_host = None
        labels_host = None
        all_losses = None
        all_preds = None
        all_labels = None
        observed_num_examples = 0

        for step, inputs in enumerate(dataloader):
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                if batch_size is None:
                    batch_size = observed_batch_size

            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                losses_host, preds_host, labels_host = None, None, None

        if args.past_index and hasattr(self, "_past"):
            delattr(self, "_past")

        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(dataset=eval_dataset, preds=all_preds, save_prefix=metric_key_prefix)
        else:
            metrics = {}

        metrics["global_step"] = self.state.global_step
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            gen_kwargs = {
                "max_new_tokens": 50,
                "num_beams": 1,
                "repetition_penalty": 1.0,
                "decoder_start_token_id": 0,
                "eos_token_id": 1,
                "pad_token_id": 0,
            }
            gen_kwargs["synced_gpus"] = False
        else:
            if inputs.get("input_ids_wo_label", None) is not None:
                gen_kwargs = {
                    "bos_token_id": 1,
                    "max_new_tokens": 50,
                    "num_beams": 1,
                    "temperature": 1.0,
                    "repetition_penalty": 1.0,
                    "eos_token_id": 2,
                    "pad_token_id": 1,
                }
            else:
                gen_kwargs = {
                    "max_new_tokens": 50,
                    "num_beams": 1,
                    "repetition_penalty": 1.0,
                    "decoder_start_token_id": 0,
                    "eos_token_id": 1,
                    "pad_token_id": 0,
                }
            gen_kwargs["synced_gpus"] = False

        attention_mask = inputs.get("attention_mask", None)

        # synced_gpus and attention_mask must be passed to generate(), not GenerationConfig
        _synced_gpus = gen_kwargs.pop("synced_gpus", False)
        _attention_mask = inputs.get("attention_mask", None)  # from inputs, not gen_kwargs

        generation_config = GenerationConfig(**gen_kwargs)

        _generate_extra = {"synced_gpus": _synced_gpus}
        if _attention_mask is not None:
            _generate_extra["attention_mask"] = _attention_mask

        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
            generated_tokens = self.model.generate(
                input_ids=generation_inputs,
                generation_config=generation_config,
                **_generate_extra,
            )
        else:
            generation_inputs = inputs[self.model.main_input_name]
            if inputs.get("input_ids_wo_label", None) is not None:
                generated_tokens = self.model.generate(
                    input_ids=generation_inputs,
                    input_ids_wo_label=inputs["input_ids_wo_label"],
                    generation_config=generation_config,
                    **_generate_extra,
                )
            else:
                generated_tokens = self.model.generate(
                    input_ids=generation_inputs,
                    generation_config=generation_config,
                    **_generate_extra,
                )

        bs, source_len = inputs['input_ids'].shape
        if check_model(self.model.config._name_or_path, SUPPORTED_DECODER_MODELS):
            max_length = source_len + gen_kwargs["max_new_tokens"]
        else:
            max_length = gen_kwargs["max_new_tokens"]

        if generated_tokens.shape[-1] < max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, max_length)

        with torch.no_grad():
            if has_labels:
                with self.autocast_smart_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_new_tokens"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_new_tokens"])
        else:
            labels = None

        return (loss, generated_tokens, labels)

    # _inner_training_loop: removed — base Seq2SeqTrainer handles it correctly.
    # The override was a wholesale copy from old transformers with deprecated attrs
    # (getattr(self, 'do_grad_scaling', False), getattr(self, 'use_apex', False), is_torch_less_than_1_11, etc.)
    # and contained NO SpecRoute-specific logic.
