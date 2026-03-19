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
import torch
import math as _math
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


class SpecRoute_Trainer(Seq2SeqTrainer):

    def __init__(self, model, args, train_dataset, cur_task_id, task_order,
                 eval_dataset=None, tokenizer=None, data_collator=None,
                 compute_metrics=None, callbacks=None,
                 lambda_entropy=0.0, use_preconditioning=False,
                 precond_eps=1e-6, entropy_warmup_ratio=0.1):
        super().__init__(
            model=model, args=args, train_dataset=train_dataset,
            eval_dataset=eval_dataset, tokenizer=tokenizer,
            data_collator=data_collator, compute_metrics=compute_metrics,
            callbacks=callbacks
        )
        self.task_order = task_order
        self.cur_task_id = cur_task_id
        self._grad_check_done = False
        # C4: Spectrally-Conditioned LoRA Training
        self.lambda_entropy = lambda_entropy
        self.use_preconditioning = use_preconditioning
        self.precond_eps = precond_eps
        self.entropy_warmup_ratio = entropy_warmup_ratio
        self._precond_matrices = {}

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

    def _compute_spectral_entropy_loss(self):
        """Compute spectral entropy regularization via efficient QR trick.
        For each LoRA layer: QR(B) and QR(A^T) -> SVD of r×r matrix -> entropy."""
        ent_loss = torch.tensor(0.0, device=self.args.device)
        count = 0
        for module in self.model.modules():
            if not (hasattr(module, 'lora_q') and hasattr(module, 'lora_v')):
                continue
            for lora in [module.lora_q, module.lora_v]:
                B = lora.lora_B.float()    # [d_out, r]
                A = lora.lora_A.float()    # [r, d_in]
                if B.norm() < 1e-8:
                    continue
                _, R_B = torch.linalg.qr(B.T)      # B.T: [r, d_out] -> R_B: [r, d_out]
                _, R_A = torch.linalg.qr(A)         # A:   [r, d_in]  -> R_A: [r, d_in]
                if R_B.shape[1] != R_A.shape[1]:
                    continue  # skip layers where d_out != d_in
                sigma_hat = torch.linalg.svdvals(R_B @ R_A.T)  # [r, d] @ [d, r] -> [r, r] -> [r]
                sigma_hat = sigma_hat / (sigma_hat.sum() + 1e-12)
                ent = -(sigma_hat * torch.log(sigma_hat + 1e-12)).sum()
                max_ent = _math.log(sigma_hat.size(0))
                ent_loss = ent_loss + (max_ent - ent)
                count += 1
        if count > 0:
            ent_loss = ent_loss / count
        return ent_loss

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
        """CE training step + C4 spectral entropy regularization + preconditioning."""
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        # C4: Spectral entropy regularization (after warmup)
        if self.lambda_entropy > 0:
            warmup_steps = int(self.entropy_warmup_ratio * self.state.max_steps)
            if self.state.global_step >= warmup_steps:
                ent_loss = self._compute_spectral_entropy_loss()
                loss = loss + self.lambda_entropy * ent_loss

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

    def load_previous_reg_matrix(self):
        """Load LoRA GPM bases from previous task. No trans_input GPM needed."""
        log_path = os.path.dirname(self.args.output_dir)
        local_dir = os.path.basename(self.args.output_dir)
        print(log_path)

        all_dirs = os.listdir(log_path)
        reg_matrix = []
        for all_dir in all_dirs:
            if not os.path.isdir(os.path.join(log_path, all_dir)):
                continue
            if eval(all_dir.split('-')[0]) == eval(local_dir.split('-')[0]) - 1:
                i = 0
                for module in self.model.modules():
                    if hasattr(module, 'get_feature'):
                        reg_matrix.append(torch.load(
                            os.path.join(os.path.join(log_path, all_dir), "reg_{}.pt".format(i))
                        ))
                        i += 1
                print(os.path.join(log_path, all_dir))
                print(len(reg_matrix))
                break
        return reg_matrix, eval(local_dir.split('-')[0]) - 1

    def get_reg_matrix(self):
        """
        Project current LoRA A into null-space of old tasks' GPM bases.
        No prompt_key/trans_input operations.
        """
        self.feature_list, self._cur_task = self.load_previous_reg_matrix()

        if len(self.feature_list) == 0:
            # First task: no constraints
            return

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

                # Project lora_A into null-space (InfLoRA constraint)
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
                module.lora_q.lora_A.data /= (math.sqrt(3) * module.lora_q.lora_A.data.norm(dim=1, keepdim=True))
                module.lora_v.lora_A.data /= (math.sqrt(3) * module.lora_v.lora_A.data.norm(dim=1, keepdim=True))
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
        ESA: Use constant threshold (no increasing schedule).
        No trans_input features collected.
        """
        self.feature_list, self._cur_task = self.load_previous_reg_matrix()

        train_dataloader = self.get_train_dataloader()
        if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(1998)
        elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
            train_dataloader.dataset.set_epoch(1998)

        for name, module in self.model.named_modules():
            if hasattr(module, 'get_feature'):
                module.get_feature = True
                module.stage = 0

        print('begin get representation')
        with torch.no_grad():
            for step, inputs in enumerate(train_dataloader):
                inputs = self._prepare_inputs(inputs)
                if self.label_smoother is not None and "labels" in inputs:
                    labels = inputs.pop("labels")
                else:
                    labels = None
                outputs = self.model(**inputs)
                if step > 1000:
                    break
        print('end get representation')

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

        # No trans_input GPM to save

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
