#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed


ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from root_gainlora_bugfix.src.cl_dataset import CLConfig, CLInstructions  # noqa: E402


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0

    previous_row = [0] * (len(right) + 1)
    for left_token in left:
        current_row = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current_row.append(previous_row[index - 1] + 1)
            else:
                current_row.append(max(previous_row[index], current_row[-1]))
        previous_row = current_row
    return previous_row[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = tokenize_words(prediction)
    ref_tokens = tokenize_words(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0

    beta = 1.2
    beta_sq = beta * beta
    return ((1 + beta_sq) * precision * recall) / (recall + beta_sq * precision)


def rouge_1_f1(prediction: str, reference: str) -> float:
    pred_counts = Counter(tokenize_words(prediction))
    ref_counts = Counter(tokenize_words(reference))
    if not pred_counts or not ref_counts:
        return 0.0

    overlap = sum((pred_counts & ref_counts).values())
    precision = overlap / sum(pred_counts.values())
    recall = overlap / sum(ref_counts.values())
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_text(prediction) == normalize_text(reference))


def build_splits(data_dir: Path, task_config_dir: Path, max_train_samples: int | None, max_eval_samples: int | None):
    task_configs = CLConfig.parse_task_config(str(task_config_dir))
    if task_configs is None:
        raise ValueError(f"Invalid task_config_dir: {task_config_dir}")

    builder = CLInstructions()

    def _collect(split_name: str, max_num_instances: int | None):
        samples = []
        for _, sample in builder._generate_examples(
            path=str(data_dir),
            task_config=task_configs[split_name],
            max_num_instances_per_task=max_num_instances,
            subset=split_name,
        ):
            samples.append(sample)
        return samples

    train_samples = _collect("train", max_train_samples)
    dev_samples = _collect("dev", max_eval_samples)
    test_samples = _collect("test", None)
    return train_samples, dev_samples, test_samples


class CausalTaskCollator:
    def __init__(self, tokenizer, max_source_length: int, max_target_length: int):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def __call__(self, features):
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        prepared_examples = []
        max_length = 0

        for feature in features:
            instance = feature["Instance"]
            prompt = instance["instruction"].format(instance["sentence"])
            target = instance["label"] + self.tokenizer.eos_token

            prompt_ids = self.tokenizer(
                prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_source_length,
            )["input_ids"]
            target_ids = self.tokenizer(
                target,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_target_length,
            )["input_ids"]

            input_ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids
            prepared_examples.append((input_ids, labels))
            max_length = max(max_length, len(input_ids))

        for input_ids, labels in prepared_examples:
            pad_length = max_length - len(input_ids)
            batch_input_ids.append([self.pad_token_id] * pad_length + input_ids)
            batch_attention_mask.append([0] * pad_length + [1] * len(input_ids))
            batch_labels.append([-100] * pad_length + labels)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def prepare_prompt(sample: dict) -> str:
    instance = sample["Instance"]
    return instance["instruction"].format(instance["sentence"])


def generate_predictions(model, tokenizer, samples: list[dict], max_source_length: int, max_new_tokens: int, batch_size: int):
    model.eval()
    device = next(model.parameters()).device
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.max_length = None
    generation_config.temperature = None
    generation_config.top_p = None
    generation_config.top_k = None
    generation_config.do_sample = False
    generation_config.num_beams = 1
    generation_config.repetition_penalty = 1.0
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id
    generation_config.bos_token_id = tokenizer.bos_token_id

    predictions: list[str] = []
    references: list[str] = []

    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start:start + batch_size]
        prompts = [prepare_prompt(sample) for sample in batch_samples]
        refs = [sample["Instance"]["label"] for sample in batch_samples]

        encoded = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_source_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        input_length = encoded["input_ids"].shape[1]

        with torch.no_grad():
            generated = model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                generation_config=generation_config,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )

        for generated_ids, reference in zip(generated, refs):
            completion_ids = generated_ids[int(input_length):]
            prediction = tokenizer.decode(
                completion_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
            predictions.append(prediction)
            references.append(reference.strip())

    return predictions, references


def evaluate_split(model, tokenizer, samples: list[dict], max_source_length: int, max_new_tokens: int, batch_size: int, split_name: str):
    predictions, references = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        max_source_length=max_source_length,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )

    rouge_l = sum(rouge_l_f1(pred, ref) for pred, ref in zip(predictions, references)) / max(len(references), 1)
    rouge_1 = sum(rouge_1_f1(pred, ref) for pred, ref in zip(predictions, references)) / max(len(references), 1)
    exact = sum(exact_match(pred, ref) for pred, ref in zip(predictions, references)) / max(len(references), 1)
    empty_predictions = sum(1 for pred in predictions if not pred.strip())

    print(f"=== {split_name.upper()} DEBUG ===")
    for index in range(min(3, len(predictions))):
        print(f"  [{index}] PRED: {predictions[index]!r}")
        print(f"  [{index}] REF : {references[index]!r}")
    print(f"Empty decoded predictions: {empty_predictions}/{len(predictions)} ({split_name})")

    return {
        f"{split_name}_rougeL": rouge_l,
        f"{split_name}_rouge1": rouge_1,
        f"{split_name}_exact_match": exact,
        f"{split_name}_empty_predictions": empty_predictions,
        f"{split_name}_samples": len(references),
        f"{split_name}_predictions": predictions,
        f"{split_name}_references": references,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone task1572 Llama + LoRA fine-tuning baseline.")
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=ROOT_DIR.parent / "root_gainlora" / "CL_Benchmark",
    )
    parser.add_argument(
        "--task_config_dir",
        type=Path,
        default=ROOT_DIR / "root_gainlora_bugfix" / "configs" / "gen_script_superni_order1_llama_configs" / "task1572_samsum_summary",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT_DIR / "logs_and_outputs" / "task1572_llama_lora_simple",
    )
    parser.add_argument("--run_name", type=str, default="task1572_llama_lora_simple")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=float, default=50)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--max_source_length", type=int, default=1024)
    parser.add_argument("--max_target_length", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deepspeed", type=str, default=str(ROOT_DIR / "root_gainlora_bugfix" / "configs" / "ds_configs" / "stage2.config"))
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--use_auth_token", action="store_true")
    parser.add_argument(
        "--local_rank",
        "--local-rank",
        dest="local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (injected by launchers).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    token = True if args.use_auth_token else None
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
        token=token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
    tokenizer.padding_side = "left"

    train_samples, dev_samples, test_samples = build_splits(
        data_dir=args.data_dir,
        task_config_dir=args.task_config_dir,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )

    train_dataset = Dataset.from_list(train_samples)
    print(f"Train/dev/test sizes: {len(train_dataset)}/{len(dev_samples)}/{len(test_samples)}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        token=token,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)

    trainable_params = 0
    total_params = 0
    for parameter in model.parameters():
        total_params += parameter.numel()
        if parameter.requires_grad:
            trainable_params += parameter.numel()
    print(
        f"Trainable parameters: {trainable_params / 1e6:.3f}M / {total_params / 1e6:.3f}M "
        f"({100 * trainable_params / max(total_params, 1):.2f}%)"
    )

    collator = CausalTaskCollator(
        tokenizer=tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant",
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        logging_strategy="steps",
        save_strategy="no",
        report_to=[],
        seed=args.seed,
        bf16=args.bf16,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed=args.deepspeed,
        optim="adamw_torch",
        ddp_find_unused_parameters=False,
        run_name=args.run_name,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    train_result = trainer.train()
    train_metrics = train_result.metrics
    train_metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()

    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    del trainer
    del model
    torch.cuda.empty_cache()

    eval_base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        token=token,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    eval_base_model.config.pad_token_id = tokenizer.pad_token_id
    eval_base_model.config.use_cache = True
    eval_base_model.generation_config.pad_token_id = tokenizer.pad_token_id
    eval_base_model.generation_config.eos_token_id = tokenizer.eos_token_id
    eval_base_model.generation_config.bos_token_id = tokenizer.bos_token_id
    eval_model = PeftModel.from_pretrained(eval_base_model, str(args.output_dir))
    eval_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dev_metrics = evaluate_split(
        model=eval_model,
        tokenizer=tokenizer,
        samples=dev_samples,
        max_source_length=args.max_source_length,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.per_device_eval_batch_size,
        split_name="dev",
    )
    test_metrics = evaluate_split(
        model=eval_model,
        tokenizer=tokenizer,
        samples=test_samples,
        max_source_length=args.max_source_length,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.per_device_eval_batch_size,
        split_name="test",
    )

    final_metrics = {
        "run_name": args.run_name,
        "train_loss": float(train_metrics.get("train_loss", float("nan"))),
        "train_runtime": float(train_metrics.get("train_runtime", float("nan"))),
        "dev_rougeL": dev_metrics["dev_rougeL"],
        "dev_rouge1": dev_metrics["dev_rouge1"],
        "dev_exact_match": dev_metrics["dev_exact_match"],
        "test_rougeL": test_metrics["test_rougeL"],
        "test_rouge1": test_metrics["test_rouge1"],
        "test_exact_match": test_metrics["test_exact_match"],
        "test_empty_predictions": test_metrics["test_empty_predictions"],
    }

    with open(args.output_dir / "final_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(final_metrics, handle, indent=2, ensure_ascii=False)

    with open(args.output_dir / "dev_predictions.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "predictions": dev_metrics["dev_predictions"],
                "references": dev_metrics["dev_references"],
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    with open(args.output_dir / "test_predictions.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "predictions": test_metrics["test_predictions"],
                "references": test_metrics["test_references"],
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(json.dumps(final_metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
