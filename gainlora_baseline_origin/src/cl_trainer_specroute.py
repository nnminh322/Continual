"""
SpecRoute Trainer: Custom Seq2SeqTrainer for SpecRoute continual learning.

Key differences from GainLoRA_InfLoRA_Trainer:
- No trans_input GPM constraints (routing is parameter-free spectral projection)
- No GPM projection on trans_input/prompt_key after optimizer step
- No memory replay (KL loss on gating — routing has no learned parameters)
- Constant threshold for GPM (Elastic Subspace Allocation)
- Simplified optimizer (no special lr for trans_input)
"""

import torch
from transformers import GenerationConfig
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.trainer import *
from transformers.trainer_callback import TrainerCallback
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


class SpecRoute_Trainer(Seq2SeqTrainer):

    def __init__(self, model, args, train_dataset, cur_task_id, task_order,
                 eval_dataset=None, tokenizer=None, data_collator=None,
                 compute_metrics=None, callbacks=None):
        super().__init__(
            model=model, args=args, train_dataset=train_dataset,
            eval_dataset=eval_dataset, tokenizer=tokenizer,
            data_collator=data_collator, compute_metrics=compute_metrics,
            callbacks=callbacks
        )
        self.task_order = task_order
        self.cur_task_id = cur_task_id
        # No replay data needed for SpecRoute

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

        # ESA: constant threshold (no increasing schedule)
        threshold = self.args.threshold
        print('Threshold (ESA constant): ', threshold)

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

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Standard training step without memory replay.
        SpecRoute has no learned routing parameters to replay.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if self.args.gradient_accumulation_steps > 1 and not self.is_deepspeed_enabled:
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.is_deepspeed_enabled:
            self.accelerator.backward(loss)
        else:
            loss.backward()

        # No memory replay for SpecRoute

        return loss.detach()

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.create_optimizer()
        if IS_SAGEMAKER_MP_POST_1_10 and smp.state.cfg.fp16:
            optimizer = self.optimizer.optimizer
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

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
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
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if logits is not None:
                logits = self._pad_across_processes(logits)
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

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)

        generation_config = GenerationConfig(**gen_kwargs)

        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
            generated_tokens = self.model.generate(
                input_ids=generation_inputs,
                generation_config=generation_config,
            )
        else:
            generation_inputs = inputs[self.model.main_input_name]
            if inputs.get("input_ids_wo_label", None) is not None:
                generated_tokens = self.model.generate(
                    input_ids=generation_inputs,
                    input_ids_wo_label=inputs["input_ids_wo_label"],
                    generation_config=generation_config,
                )
            else:
                generated_tokens = self.model.generate(
                    input_ids=generation_inputs,
                    generation_config=generation_config,
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

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        """
        Inner training loop. Same as GainLoRA version but WITHOUT:
        - GPM projection on trans_input/prompt_key after optimizer step
        """
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        train_dataloader = self.get_train_dataloader()

        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:
            max_steps = args.max_steps
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if args.logging_steps and args.logging_steps < 1:
            args.logging_steps = math.ceil(max_steps * args.logging_steps)
        if args.eval_steps and args.eval_steps < 1:
            args.eval_steps = math.ceil(max_steps * args.eval_steps)
        if args.save_steps and args.save_steps < 1:
            args.save_steps = math.ceil(max_steps * args.save_steps)

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
            self.sharded_ddp is not None
            and self.sharded_ddp != ShardedDDPOption.SIMPLE
            or is_sagemaker_mp_enabled()
            or self.fsdp is not None
        )

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        if use_accelerator_prepare:
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = model

        if model is not self.model:
            self.model_wrapped = model

        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        if resume_from_checkpoint is not None and self.is_deepspeed_enabled:
            deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint)

        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                if skip_first_batches is None:
                    logger.info(
                        f"  Will skip the first {epochs_trained} epochs then the first"
                        f" {steps_trained_in_current_epoch} batches in the first epoch."
                    )
                else:
                    logger.info(
                        f"  Will skip the first {epochs_trained} epochs then the first"
                        f" {steps_trained_in_current_epoch} batches in the first epoch."
                    )
                if self.is_local_process_zero() and not args.disable_tqdm and skip_first_batches is None:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        tr_loss = torch.tensor(0.0).to(args.device)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if is_torch_less_than_1_11 or not is_random_sampler:
                    for _ in train_dataloader:
                        break
                else:
                    _ = list(train_dataloader.sampler)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if skip_first_batches is not None and steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                with self.accelerator.accumulate(model):
                    tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                if total_batched_samples % args.gradient_accumulation_steps == 0 or (
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # NO GPM projection on trans_input/prompt_key — SpecRoute has none

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        if self.do_grad_scaling:
                            if is_torch_tpu_available():
                                gradients = xm._fetch_gradients(self.optimizer)
                                xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                            self.scaler.unscale_(self.optimizer)

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif hasattr(self.optimizer, "clip_grad_norm"):
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            model.clip_grad_norm_(args.max_grad_norm)
                        elif self.use_apex:
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if is_torch_tpu_available():
                        if self.do_grad_scaling:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            xm.optimizer_step(self.optimizer)
                    elif self.do_grad_scaling:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()
                        optimizer_was_run = not self.accelerator.optimizer_step_was_skipped

                    if optimizer_was_run:
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    # NO trans_input/prompt_key GPM projection after optimizer step

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()
            self._load_best_model()

        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False
        self._memory_tracker.stop_and_update_metrics(metrics)
        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)
