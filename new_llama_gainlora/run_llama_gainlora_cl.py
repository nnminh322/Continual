#!/usr/bin/env python3
"""
LLaMA GainLoRA Continual Learning Training Script
==================================================
Implements the T5 GainLoRA pipeline (SRT + optional SGWI) ported to LLaMA 2.

Design:
  - Model:   llama_gainlora.LlamaForCausalLM (custom LoRA with agg_lora_states)
  - Trainer: SRTSGWITrainer (SRT signatures + SGWI warm-init; NO GPM, NO replay)
  - Routing: hard one-hot during training (slot 0 only);
             SRT ZCA+L2 hard one-hot at inference
  - Save:    lora_weights_A.pt, lora_weights_B.pt, srt_signatures.npz

Usage (single task):
  deepspeed --num_gpus=1 run_llama_gainlora_cl.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --data_dir ../root_gainlora/CL_Benchmark \
    --task_config_dir ../new_gainlora/configs/gen_script_superni_order1_llama_configs/task1572_samsum_summary \
    --output_dir logs_and_outputs/0-task1572_samsum_summary \
    --cur_task_id 0 \
    --task_order task1572_samsum_summary,task363_sst2_polarity_classification,...
    # SGWI is ON by default; pass --no_sgwi for full_lora mode

  # task 1:  add --previous_lora_path <task0_dir>/saved_weights
  #              --srt_load_path       <task0_dir>/saved_weights

See run_superni_order1_llama_cl.sh for full CL orchestration.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    TrainingArguments,
    set_seed,
)

# ── sys.path setup ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent           # new_llama_gainlora/
CONTINUAL_DIR = ROOT_DIR.parent                       # Continual/
SRC_DIR = ROOT_DIR / "src"                            # new_llama_gainlora/src/
NG_SRC = CONTINUAL_DIR / "new_gainlora" / "src"      # new_gainlora/src/ (llama_gainlora, srt_router, cl_dataset)

for _p in [str(SRC_DIR), str(NG_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from llama_gainlora import LlamaForCausalLM                    # noqa: E402
from frozen_extractor import FrozenLlamaExtractor              # noqa: E402
from sgwi_srt_trainer import SRTSGWITrainer                    # noqa: E402
from cl_dataset import CLConfig, CLInstructions                # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Metrics (same as simple baseline)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (same as simple baseline)
# ─────────────────────────────────────────────────────────────────────────────

def build_splits(
    data_dir: Path,
    task_config_dir: Path,
    max_train_samples: int | None,
    max_eval_samples: int | None,
):
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
    dev_samples   = _collect("dev",   max_eval_samples)
    test_samples  = _collect("test",  None)
    return train_samples, dev_samples, test_samples


# ─────────────────────────────────────────────────────────────────────────────
# Collator: CL version also returns input_ids_wo_label for SRT routing
# ─────────────────────────────────────────────────────────────────────────────

class CLCausalTaskCollator:
    """
    Left-pads (prompt + target) for causal LM training.
    Also returns `input_ids_wo_label` (source-only, left-padded separately)
    for SRT routing inside LlamaModel.forward().
    """

    def __init__(self, tokenizer, max_source_length: int, max_target_length: int):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )

    def __call__(self, features):
        prepared = []
        max_full_len = 0
        max_src_len = 0

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

            full_ids = prompt_ids + target_ids
            labels   = [-100] * len(prompt_ids) + target_ids

            prepared.append((prompt_ids, full_ids, labels))
            max_full_len = max(max_full_len, len(full_ids))
            max_src_len  = max(max_src_len,  len(prompt_ids))

        batch_input_ids   = []
        batch_attn_mask   = []
        batch_labels      = []
        batch_src_ids     = []

        for prompt_ids, full_ids, labels in prepared:
            full_pad = max_full_len - len(full_ids)
            src_pad  = max_src_len  - len(prompt_ids)

            batch_input_ids.append([self.pad_id] * full_pad + full_ids)
            batch_attn_mask.append([0] * full_pad + [1] * len(full_ids))
            batch_labels.append(   [-100] * full_pad + labels)
            batch_src_ids.append(  [self.pad_id] * src_pad + prompt_ids)

        return {
            "input_ids":         torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask":    torch.tensor(batch_attn_mask,  dtype=torch.long),
            "labels":            torch.tensor(batch_labels,     dtype=torch.long),
            "input_ids_wo_label":torch.tensor(batch_src_ids,    dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# LoRA weight helpers
# ─────────────────────────────────────────────────────────────────────────────

def lora_state_dict_A(model: nn.Module) -> dict:
    """Current task lora_A weights (exclude previous_lora_weights*)."""
    return {
        k: v.detach().clone().cpu()
        for k, v in model.named_parameters()
        if "lora_A" in k and "previous_lora_weights" not in k
    }


def lora_state_dict_B(model: nn.Module) -> dict:
    """Current task lora_B weights (exclude previous_lora_weights*)."""
    return {
        k: v.detach().clone().cpu()
        for k, v in model.named_parameters()
        if "lora_B" in k and "previous_lora_weights" not in k
    }


def load_previous_lora(
    model: LlamaForCausalLM,
    previous_lora_path: str,
    config,
) -> None:
    """
    Load past LoRA checkpoints into previous_lora_weights_q/v slots.

    previous_lora_path: comma-separated list of saved_weights directories,
                        in task order (oldest first).
    After reverse(): slot[0] = most recent = task_{T-1},
                     slot[i] = task_{T-1-i}.
    """
    paths = [p.strip() for p in previous_lora_path.split(",") if p.strip()]
    paths_reversed = list(reversed(paths))  # slot[0] = most recent

    for i, path in enumerate(paths_reversed):
        lora_A = torch.load(
            os.path.join(path, "lora_weights_A.pt"),
            map_location="cpu",
            weights_only=True,
        )
        lora_B = torch.load(
            os.path.join(path, "lora_weights_B.pt"),
            map_location="cpu",
            weights_only=True,
        )
        for j in range(config.num_hidden_layers):
            attn = model.model.layers[j].self_attn
            attn.previous_lora_weights_q[i].lora_A.data.copy_(
                lora_A[f"model.layers.{j}.self_attn.lora_q.lora_A"]
            )
            attn.previous_lora_weights_q[i].lora_B.data.copy_(
                lora_B[f"model.layers.{j}.self_attn.lora_q.lora_B"]
            )
            attn.previous_lora_weights_v[i].lora_A.data.copy_(
                lora_A[f"model.layers.{j}.self_attn.lora_v.lora_A"]
            )
            attn.previous_lora_weights_v[i].lora_B.data.copy_(
                lora_B[f"model.layers.{j}.self_attn.lora_v.lora_B"]
            )
        print(f"[PREV-LORA] Loaded slot[{i}] ← {path}")

    # Move previous LoRA weights to CPU to save VRAM;
    # agg_lora_states will temporarily move each to GPU during forward.
    for module in model.modules():
        for attr in ("previous_lora_weights_q", "previous_lora_weights_v"):
            prev = getattr(module, attr, None)
            if prev is not None:
                prev.to("cpu")
    print(f"[PREV-LORA] Moved {len(paths)} past adapters to CPU")


# ─────────────────────────────────────────────────────────────────────────────
# Generation / Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_prompt(sample: dict) -> str:
    instance = sample["Instance"]
    return instance["instruction"].format(instance["sentence"])


def generate_predictions_cl(
    model: LlamaForCausalLM,
    tokenizer,
    samples: list[dict],
    max_source_length: int,
    max_new_tokens: int,
    batch_size: int,
) -> tuple[list[str], list[str]]:
    """
    Run generation with the CL model.
    Passes input_ids_wo_label = source prompt to enable SRT inference routing.
    """
    model.eval()
    model.model.is_inference = True  # enable routing stat collection

    device = next(model.parameters()).device

    gen_cfg = copy.deepcopy(model.generation_config)
    gen_cfg.max_length        = None
    gen_cfg.temperature       = None
    gen_cfg.top_p             = None
    gen_cfg.top_k             = None
    gen_cfg.do_sample         = False
    gen_cfg.num_beams         = 1
    gen_cfg.repetition_penalty = 1.0
    gen_cfg.pad_token_id      = tokenizer.pad_token_id
    gen_cfg.eos_token_id      = tokenizer.eos_token_id
    gen_cfg.bos_token_id      = tokenizer.bos_token_id

    predictions: list[str] = []
    references:  list[str] = []

    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start : start + batch_size]
        prompts = [prepare_prompt(s) for s in batch_samples]
        refs    = [s["Instance"]["label"] for s in batch_samples]

        encoded = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_source_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        input_length = encoded["input_ids"].shape[1]

        with torch.no_grad():
            generated = model.generate(
                input_ids            = encoded["input_ids"],
                input_ids_wo_label   = encoded["input_ids"],  # source for SRT routing
                attention_mask       = encoded["attention_mask"],
                generation_config    = gen_cfg,
                max_new_tokens       = max_new_tokens,
            )

        for generated_ids, reference in zip(generated, refs):
            completion_ids = generated_ids[input_length:]
            prediction = tokenizer.decode(
                completion_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
            predictions.append(prediction)
            references.append(reference.strip())

    model.model.is_inference = False
    return predictions, references


def evaluate_split_cl(
    model, tokenizer, samples, max_source_length, max_new_tokens, batch_size, split_name
) -> dict:
    predictions, references = generate_predictions_cl(
        model, tokenizer, samples, max_source_length, max_new_tokens, batch_size
    )
    rouge_l = sum(rouge_l_f1(p, r) for p, r in zip(predictions, references)) / max(len(references), 1)
    rouge_1 = sum(rouge_1_f1(p, r) for p, r in zip(predictions, references)) / max(len(references), 1)
    exact   = sum(exact_match(p, r) for p, r in zip(predictions, references)) / max(len(references), 1)
    empty   = sum(1 for p in predictions if not p.strip())

    print(f"=== {split_name.upper()} ===")
    for idx in range(min(3, len(predictions))):
        print(f"  [{idx}] PRED: {predictions[idx]!r}")
        print(f"  [{idx}] REF : {references[idx]!r}")
    print(f"Empty predictions: {empty}/{len(predictions)}")

    return {
        f"{split_name}_rougeL":            rouge_l,
        f"{split_name}_rouge1":            rouge_1,
        f"{split_name}_exact_match":       exact,
        f"{split_name}_empty_predictions": empty,
        f"{split_name}_samples":           len(references),
        f"{split_name}_predictions":       predictions,
        f"{split_name}_references":        references,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLaMA GainLoRA Continual Learning (SRT + SGWI, no GPM)"
    )

    # ── Model ──────────────────────────────────────────────────────────────
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--lora_r",         type=int,   default=4)
    parser.add_argument("--lora_alpha",     type=int,   default=32)
    parser.add_argument("--lora_dropout",   type=float, default=0.0)

    # ── Paths ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--data_dir", type=Path,
        default=CONTINUAL_DIR / "root_gainlora" / "CL_Benchmark",
    )
    parser.add_argument("--task_config_dir", type=Path, required=True)
    parser.add_argument(
        "--output_dir", type=Path,
        default=ROOT_DIR / "logs_and_outputs" / "llama_gainlora_cl",
    )

    # ── CL ─────────────────────────────────────────────────────────────────
    parser.add_argument("--cur_task_id", type=int, required=True,
                        help="0-based index of current task in task_order")
    parser.add_argument("--task_order",  type=str, required=True,
                        help="Comma-separated full task sequence")
    parser.add_argument("--previous_lora_path", type=str, default=None,
                        help="Comma-separated saved_weights dirs (oldest→newest)")
    parser.add_argument("--srt_load_path", type=str, default=None,
                        help="Dir with srt_signatures.npz from previous task")

    # ── SRT ────────────────────────────────────────────────────────────────
    parser.add_argument("--use_srt_router",     action="store_true", default=True)
    parser.add_argument("--no_srt_router",      dest="use_srt_router", action="store_false")
    parser.add_argument("--srt_metric_mode",    type=str, default="hard",
                        choices=["hard", "dynamics"])
    parser.add_argument("--srt_shrink",         action="store_true", default=False)
    parser.add_argument("--srt_shrink_factor",  type=float, default=0.1)
    parser.add_argument("--srt_max_emb_samples",type=int,   default=500)
    parser.add_argument("--srt_skip_forward",   action="store_true", default=False)

    # ── SGWI ───────────────────────────────────────────────────────────────
    parser.add_argument("--sgwi", action="store_true", default=True,
                        help="Enable SGWI warm-init (default: True, matching T5 gold standard)")
    parser.add_argument("--no_sgwi", dest="sgwi", action="store_false",
                        help="Disable SGWI warm-init (full_lora mode)")

    # ── Data ───────────────────────────────────────────────────────────────
    parser.add_argument("--max_train_samples",     type=int, default=None)
    parser.add_argument("--max_eval_samples",      type=int, default=None)
    parser.add_argument("--max_source_length",     type=int, default=1024)
    parser.add_argument("--max_target_length",     type=int, default=50)
    parser.add_argument("--max_new_tokens",        type=int, default=50)

    # ── Training ───────────────────────────────────────────────────────────
    parser.add_argument("--per_device_train_batch_size", type=int,   default=1)
    parser.add_argument("--per_device_eval_batch_size",  type=int,   default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int,   default=32)
    parser.add_argument("--learning_rate",               type=float, default=5e-5)
    parser.add_argument("--num_train_epochs",            type=float, default=100)
    parser.add_argument("--warmup_steps",                type=int,   default=0)
    parser.add_argument("--logging_steps",               type=int,   default=10)
    parser.add_argument("--seed",                        type=int,   default=42)
    parser.add_argument("--bf16",        action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument(
        "--deepspeed", type=str,
        default=str(CONTINUAL_DIR / "new_gainlora" / "configs" / "ds_configs" / "stage2.config"),
    )
    parser.add_argument("--use_auth_token", action="store_true")
    parser.add_argument(
        "--local_rank", "--local-rank",
        dest="local_rank", type=int, default=-1,
        help="Local rank injected by DeepSpeed launcher",
    )

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    task_order   = args.task_order.split(",")
    cur_task_id  = args.cur_task_id
    cur_task     = task_order[cur_task_id]
    run_name     = f"{cur_task_id}-{cur_task}"

    print(f"\n{'='*60}")
    print(f"Task {cur_task_id}: {cur_task}")
    print(f"Task order ({len(task_order)} tasks): {task_order}")
    print(f"previous_lora_path: {args.previous_lora_path}")
    print(f"srt_load_path:      {args.srt_load_path}")
    print(f"sgwi:               {args.sgwi}")
    print(f"{'='*60}\n")

    token = True if args.use_auth_token else None
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    # ── Tokenizer ──────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=False, token=token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
    tokenizer.padding_side = "left"

    # ── AutoConfig (for num_hidden_layers etc.) ────────────────────────────
    config = AutoConfig.from_pretrained(args.model_name_or_path, token=token)
    config.pad_token_id = tokenizer.pad_token_id
    config.use_cache    = False

    # ── Data ───────────────────────────────────────────────────────────────
    train_samples, dev_samples, test_samples = build_splits(
        data_dir          = args.data_dir,
        task_config_dir   = args.task_config_dir,
        max_train_samples = args.max_train_samples,
        max_eval_samples  = args.max_eval_samples,
    )
    print(f"Train/dev/test: {len(train_samples)}/{len(dev_samples)}/{len(test_samples)}")
    train_dataset = Dataset.from_list(train_samples)
    # Dummy eval_dataset: keeps Trainer happy when eval_strategy='steps';
    # actual generation-based eval happens inside SRTSGWITrainer.evaluate() override.
    eval_dataset  = Dataset.from_list(dev_samples)
    collator = CLCausalTaskCollator(
        tokenizer         = tokenizer,
        max_source_length = args.max_source_length,
        max_target_length = args.max_target_length,
    )

    # ── prompt_config for LlamaForCausalLM ────────────────────────────────
    prev_lora_path = args.previous_lora_path  # None for task 0
    prompt_config = {
        "run_single":               False,         # CL mode always
        "previous_lora_path":       prev_lora_path,
        "previous_prompt_key_path": None,          # not used (dead weight)
        "task_id":                  cur_task_id,
        "lora_r":                   args.lora_r,
        "lora_alpha":               args.lora_alpha,
        "lora_dropout":             args.lora_dropout,
        "trans_hidden_dim":         100,            # dead weight; needed for model init
        "attn_temperature":         1,
        "mlp_hidden_dim":           800,
        "seq_len":                  args.max_source_length,
        "load_checkpoint_from":     None,
    }

    # ── Load model ─────────────────────────────────────────────────────────
    model = LlamaForCausalLM.from_pretrained(
        args.model_name_or_path,
        prompt_config,
        config          = config,
        token           = token,
        torch_dtype     = dtype,
        low_cpu_mem_usage = True,
        use_safetensors = True,
    )
    model.config.use_cache = False
    model.resize_token_embeddings(len(tokenizer))

    # ── FIX: re-init lora_A/B after no_init_weights from_pretrained ────────
    _n_reinit = 0
    for module in model.modules():
        if (
            hasattr(module, "lora_A")
            and hasattr(module, "lora_B")
            and hasattr(module, "reset_parameters")
            and "previous_lora_weights" not in type(module).__module__
        ):
            module.reset_parameters()   # kaiming lora_A, zeros lora_B
            nn.init.zeros_(module.lora_B)
            _n_reinit += 1
    print(f"[FIX] Re-initialized lora_A/B in {_n_reinit} LoRA layers")

    # ── FIX: re-init trans_input linears and prompt_key (dead weight) ──────
    if hasattr(model.model, "trans_input"):
        for layer in model.model.trans_input:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
    if hasattr(model.model, "prompt_key"):
        nn.init.uniform_(model.model.prompt_key.data, -1.0, 1.0)
        print("[FIX] Re-initialized prompt_key ~ U(-1,1)")

    # ── FIX: n_slots bug — set dummy previous_prompts_keys to correct size ─
    # LlamaModel.forward() uses previous_prompts_keys.shape[0] for n_prev,
    # but we never load previous_prompt_key_path → previous_prompts_keys=None.
    # Fix: create a dummy parameter with the right shape so n_slots = 1+n_prev.
    if model.model.previous_prompts_keys is None and prev_lora_path:
        n_prev = len([p for p in prev_lora_path.split(",") if p.strip()])
        dummy_dtype = torch.float32
        model.model.previous_prompts_keys = nn.Parameter(
            torch.zeros(n_prev, config.hidden_size, dtype=dummy_dtype),
            requires_grad=False,
        )
        print(
            f"[FIX] n_slots: dummy previous_prompts_keys shape=({n_prev}, {config.hidden_size})"
        )

    # ── Attach frozen backbone for SRT embedding extraction ────────────────
    if args.use_srt_router:
        print(f"[SRT] Loading frozen backbone from {args.model_name_or_path} …")
        frozen_backbone = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            token           = token,
            torch_dtype     = dtype,
            low_cpu_mem_usage = True,
            use_safetensors = True,
        )
        frozen_backbone.resize_token_embeddings(len(tokenizer))
        frozen_decoder = (
            frozen_backbone.model
            if hasattr(frozen_backbone, "model")
            else frozen_backbone
        )
        model.model.encoder_frozen = FrozenLlamaExtractor(frozen_decoder)
        print("[SRT] Frozen backbone attached.")

    # ── Load previous LoRA adapters ────────────────────────────────────────
    if prev_lora_path:
        load_previous_lora(model, prev_lora_path, config)

    # ── Load SRT signatures from previous task ─────────────────────────────
    # (SRTSGWITrainer.__init__ handles this via srt_load_path)

    # ── Freeze all; unfreeze current LoRA, trans_input, prompt_key ─────────
    for name, param in model.named_parameters():
        param.requires_grad = False
        if (
            ("lora" in name and "previous_lora_weights" not in name)
            or ("trans_input" in name and "previous_trans_input" not in name)
            or "prompt_key" in name
        ):
            param.requires_grad = True

    # ── Upcast trainable params to fp32 for AMP stability ──────────────────
    _n_upcast = 0
    for param in model.parameters():
        if param.requires_grad and param.dtype != torch.float32:
            param.data = param.data.to(torch.float32)
            _n_upcast += 1
    if _n_upcast:
        print(f"[FIX] Upcast {_n_upcast} trainable tensors to fp32")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable/1e6:.3f}M / {total/1e6:.3f}M ({100*trainable/total:.2f}%)")

    # ── Generation config ──────────────────────────────────────────────────
    if not hasattr(model, "generation_config") or model.generation_config is None:
        model.generation_config = GenerationConfig.from_model_config(model.config)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id

    # ── TrainingArguments ──────────────────────────────────────────────────
    # eval_steps = 5 * step_per_epoch  (same cadence as T5 gold standard)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    step_per_epoch = math.ceil(
        len(train_dataset)
        / (args.per_device_train_batch_size * world_size * args.gradient_accumulation_steps)
    )
    eval_steps = max(1, 5 * step_per_epoch)
    print(f"[TRAIN] step_per_epoch={step_per_epoch}, eval_steps={eval_steps}")

    training_args = TrainingArguments(
        output_dir                  = str(args.output_dir),
        per_device_train_batch_size = args.per_device_train_batch_size,
        per_device_eval_batch_size  = args.per_device_eval_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        learning_rate               = args.learning_rate,
        lr_scheduler_type           = "constant",
        warmup_steps                = args.warmup_steps,
        num_train_epochs            = args.num_train_epochs,
        logging_steps               = args.logging_steps,
        logging_strategy            = "steps",
        # Eval every 5 epochs (mirrors T5 gold: eval_steps = 5 * step_per_epoch)
        eval_strategy               = "steps",
        eval_steps                  = eval_steps,
        # No checkpoint saves (avoid writing 7B files); best model tracked in-memory
        save_strategy               = "no",
        report_to                   = [],
        seed                        = args.seed,
        bf16                        = args.bf16,
        remove_unused_columns       = False,
        dataloader_pin_memory       = False,
        gradient_checkpointing      = args.gradient_checkpointing,
        deepspeed                   = args.deepspeed,
        optim                       = "adamw_torch",
        ddp_find_unused_parameters  = False,
        run_name                    = run_name,
    )

    # ── SRTSGWITrainer ─────────────────────────────────────────────────────
    trainer = SRTSGWITrainer(
        model              = model,
        args               = training_args,
        train_dataset      = train_dataset,
        eval_dataset       = eval_dataset,   # dummy; real eval via evaluate() override
        data_collator      = collator,
        cur_task_id        = cur_task_id,
        task_order         = task_order,
        sgwi               = args.sgwi,
        srt_metric_mode    = args.srt_metric_mode,
        srt_shrink         = args.srt_shrink,
        srt_shrink_factor  = args.srt_shrink_factor,
        srt_max_emb_samples= args.srt_max_emb_samples,
        srt_load_path      = args.srt_load_path,
        srt_skip_forward   = args.srt_skip_forward,
        # in-training eval for best-model selection (mirrors T5 load_best_model_at_end)
        dev_samples        = dev_samples,
        tokenizer          = tokenizer,
        max_source_length  = args.max_source_length,
        max_new_tokens     = args.max_new_tokens,
        eval_batch_size    = args.per_device_eval_batch_size,
    )

    # ── Pre-training: SGWI warm-init ───────────────────────────────────────
    trainer.get_reg_matrix()

    # ── Training ───────────────────────────────────────────────────────────
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()

    train_result  = trainer.train()
    train_metrics = train_result.metrics
    train_metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()

    # ── Restore best checkpoint (mirrors load_best_model_at_end=True in T5) ───
    best_rougeL = trainer.restore_best_model()

    # ── Save LoRA weights (from best checkpoint) ───────────────────────────
    save_path = args.output_dir / "saved_weights"
    save_path.mkdir(parents=True, exist_ok=True)

    torch.save(lora_state_dict_A(model), str(save_path / "lora_weights_A.pt"))
    torch.save(lora_state_dict_B(model), str(save_path / "lora_weights_B.pt"))
    tokenizer.save_pretrained(str(save_path))
    print(f"[SAVE] LoRA weights → {save_path}")

    # ── Post-training: SRT signature ───────────────────────────────────────
    trainer.on_task_end(cur_task)
    trainer.save_srt_signatures(str(save_path))

    # ── Evaluation ─────────────────────────────────────────────────────────
    model.config.use_cache = True

    dev_metrics  = evaluate_split_cl(
        model, tokenizer, dev_samples,
        args.max_source_length, args.max_new_tokens,
        args.per_device_eval_batch_size, "dev",
    )
    test_metrics = evaluate_split_cl(
        model, tokenizer, test_samples,
        args.max_source_length, args.max_new_tokens,
        args.per_device_eval_batch_size, "test",
    )

    # ── Save final metrics ─────────────────────────────────────────────────
    final_metrics = {
        "run_name":         run_name,
        "task_id":          cur_task_id,
        "task":             cur_task,
        "train_loss":       float(train_metrics.get("train_loss", float("nan"))),
        "best_eval_rougeL": float(best_rougeL),   # from best in-training checkpoint
        **{
            k: v
            for k, v in {**dev_metrics, **test_metrics}.items()
            if not k.endswith("_predictions") and not k.endswith("_references")
        },
    }

    metrics_path = args.output_dir / "final_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)

    print("\n=== FINAL METRICS ===")
    for k, v in final_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"\nMetrics saved → {metrics_path}")


if __name__ == "__main__":
    main()
