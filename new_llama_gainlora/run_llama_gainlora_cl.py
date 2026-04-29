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
import importlib.util
import json
import math
import os
import time
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from datasets import Dataset
from tqdm.auto import tqdm
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
DATASET_MODULE_PATH = CONTINUAL_DIR / "new_gainlora" / "src" / "cl_dataset.py"

for _p in [str(SRC_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_module_from_file(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_dataset_module = _load_module_from_file("new_llama_cl_dataset", DATASET_MODULE_PATH)
CLConfig = _dataset_module.CLConfig
CLInstructions = _dataset_module.CLInstructions

from llama_gainlora import LlamaForCausalLM                    # noqa: E402
from frozen_extractor import FrozenLlamaExtractor              # noqa: E402
from sgwi_srt_trainer import SRTSGWITrainer                    # noqa: E402
from baseline_metrics import (                                  # noqa: E402
    compute_grouped_metrics as compute_grouped_metrics_local,
    compute_metrics as compute_metrics_local,
)

import logging
import pickle
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics (same as simple baseline)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (same as simple baseline)
# ─────────────────────────────────────────────────────────────────────────────

def build_splits(
    data_dir: Path,
    task_config_dir: Path,
    max_train_samples: int | None,
    max_eval_samples: int | None,
):
    train_samples = build_split_samples(data_dir, task_config_dir, "train", max_train_samples)
    dev_samples   = build_split_samples(data_dir, task_config_dir, "dev",   max_eval_samples)
    test_samples  = build_split_samples(data_dir, task_config_dir, "test",  None)
    return train_samples, dev_samples, test_samples


def build_split_samples(
    data_dir: Path,
    task_config_dir: Path,
    split_name: str,
    max_num_instances: int | None,
) -> list[dict]:
    task_configs = CLConfig.parse_task_config(str(task_config_dir))
    if task_configs is None:
        raise ValueError(f"Invalid task_config_dir: {task_config_dir}")

    builder = CLInstructions()
    samples = []
    for _, sample in builder._generate_examples(
        path=str(data_dir),
        task_config=task_configs[split_name],
        max_num_instances_per_task=max_num_instances,
        subset=split_name,
    ):
        samples.append(sample)
    return samples


def build_continual_eval_samples(
    data_dir: Path,
    task_config_base_dir: Path,
    task_names: list[str],
) -> list[dict]:
    samples: list[dict] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for task_name in task_names:
        task_config_dir = task_config_base_dir / task_name
        task_samples = build_split_samples(data_dir, task_config_dir, "test", None)
        for sample in task_samples:
            dataset_name = str(sample.get("Dataset", ""))
            instance = sample.get("Instance", {}) or {}
            instance_id = str(instance.get("id", ""))
            sentence = str(instance.get("sentence", ""))
            key = (dataset_name, instance_id, sentence)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            samples.append(sample)
    return samples


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


def _is_expected_custom_missing_key(name: str) -> bool:
    return (
        name.endswith("rotary_emb.inv_freq")
        or ".lora_q.lora_" in name
        or ".lora_v.lora_" in name
        or "prompt_key" in name
        or "trans_input" in name
        or "previous_lora_weights" in name
        or "previous_prompts_keys" in name
        or "previous_trans_input" in name
    )


def load_custom_llama_from_hf_checkpoint(
    model_name_or_path: str,
    config,
    prompt_config: dict,
    token,
    dtype: torch.dtype,
) -> tuple[LlamaForCausalLM, nn.Module]:
    """
    Load the HF checkpoint through the official AutoModel path, then copy the
    shared backbone weights into the custom GainLoRA model.

    This avoids silent weight drift observed when instantiating the custom
    class via from_pretrained() under Transformers 5.x.
    """
    hf_base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        token=token,
        dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )

    custom_config = copy.deepcopy(config)
    custom_config._attn_implementation = "eager"
    model = LlamaForCausalLM(custom_config, prompt_config)
    model = model.to(dtype=dtype)

    incompatible = model.load_state_dict(hf_base_model.state_dict(), strict=False)
    unexpected_missing = [
        name for name in incompatible.missing_keys
        if not _is_expected_custom_missing_key(name)
    ]
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Custom LLaMA weight load mismatch. "
            f"unexpected_missing={unexpected_missing}, "
            f"unexpected_keys={incompatible.unexpected_keys}"
        )

    print(
        "[LOAD] HF checkpoint copied into custom LLaMA "
        f"(allowed_missing={len(incompatible.missing_keys)})"
    )
    return model, hf_base_model


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
    max_target_length: int,
    max_new_tokens: int,
    batch_size: int,
    desc: str = "Generating",
) -> tuple[list[str], list[str], list[int], float, int]:
    """
    Run generation with the CL model.
    Passes input_ids_wo_label = source prompt to enable SRT inference routing.

    Returns: (predictions, references, generated_lengths, runtime_seconds, total_steps)
    """
    model.eval()
    model.model.is_inference = True  # enable routing stat collection
    # Attach tokenizer so LlamaModel.forward() can decode tokens for SRT debug log
    model.model._tokenizer = tokenizer

    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    collator = CLCausalTaskCollator(tokenizer, max_source_length, max_target_length)

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
    gen_cfg.max_new_tokens    = max_new_tokens

    predictions: list[str] = []
    references:  list[str] = []
    generated_lengths: list[int] = []
    total_steps = 0
    start_time = time.perf_counter()

    n_batches = math.ceil(len(samples) / batch_size)
    # SRT routing accuracy tracker (per GT task, per slot)
    srt_routing_stats: dict = {}

    # SRT router status check (print once)
    if hasattr(model.model, 'srt_router') and model.model.srt_router is not None:
        router = model.model.srt_router
        print(f"  [SRT-INIT] router tasks={list(router.signatures.keys())}")
        print(f"  [SRT-INIT] task_id_to_idx={getattr(model.model, 'srt_task_id_to_idx', {})}")
        print(f"  [SRT-INIT] use_srt_routing={getattr(model.model, 'use_srt_routing', False)}")
        print(f"  [SRT-INIT] encoder_frozen={type(model.model.encoder_frozen).__name__ if model.model.encoder_frozen is not None else 'None'}")
    else:
        print(f"  [SRT-INIT] NO ROUTER FOUND!")

    for start in tqdm(range(0, len(samples), batch_size), total=n_batches, desc=desc):
        batch_samples = samples[start : start + batch_size]
        batch = collator(batch_samples)
        batch = {k: v.to(device) for k, v in batch.items()}
        refs = [s["Instance"]["label"] for s in batch_samples]
        input_length = batch["input_ids_wo_label"].shape[1]

        model.reset_srt_debug_log()
        with torch.no_grad():
            generated = model.generate(
                input_ids          = batch["input_ids_wo_label"],
                input_ids_wo_label = batch["input_ids_wo_label"],  # source for SRT routing
                attention_mask     = (batch["input_ids_wo_label"] != pad_id).long(),
                generation_config  = gen_cfg,
            )

        total_steps += 1

        # ── SRT DEBUG LOGGING ─────────────────────────────────────────────
        srt_log = model.get_srt_debug_log()
        if srt_log:
            entry = srt_log[0]
            if len(srt_log) > 1:
                print(
                    f"  [SRT-D] Collapsed {len(srt_log)} routing log steps to the first "
                    f"source-prompt decision for batch starting at idx={start}."
                )

            n_tasks = entry.get("n_tasks", 0)
            task_list = entry.get("task_id_list", [])
            sample_count = min(len(batch_samples), int(entry.get("batch_size", len(batch_samples))))

            for sample_in_batch in range(sample_count):
                global_idx = start + sample_in_batch
                sample = batch_samples[sample_in_batch]
                gt_task = sample.get("Dataset", sample.get("Instance", {}).get("task", "unknown"))

                # Predicted routing decision
                srt_preds = entry["srt_preds"]
                slot_idxs = entry["slot_idxs"]
                pred_task_str = srt_preds[sample_in_batch]
                pred_slot = slot_idxs[sample_in_batch]
                tid_map = entry["task_id_to_idx"]
                idx_to_tid = {v: k for k, v in tid_map.items()}
                slot_task = idx_to_tid.get(pred_slot, f"unknown_slot{pred_slot}")

                # Distances
                dists = entry.get("dists", {})
                best_d = entry["best_dist"][sample_in_batch]
                conf = entry["confidence"][sample_in_batch]

                # Print top-2 nearest tasks
                sorted_tasks = sorted(
                    [(tid, dists.get(tid, [0])[sample_in_batch]) for tid in task_list],
                    key=lambda x: x[1]
                )
                top2 = sorted_tasks[:2]

                label_correct = (pred_task_str == gt_task)
                slot_correct = (slot_task == gt_task)
                if slot_correct and label_correct:
                    correct_flag = "slot✓ label✓"
                elif slot_correct:
                    correct_flag = "slot✓ label✗"
                elif label_correct:
                    correct_flag = "slot✗ label✓"
                else:
                    correct_flag = "slot✗ label✗"
                # ── Track accuracy per GT task ───────────────────────────
                gt_key = gt_task
                if gt_key not in srt_routing_stats:
                    srt_routing_stats[gt_key] = {
                        "slot_correct": 0,
                        "label_correct": 0,
                        "disagree": 0,
                        "total": 0,
                        "slots": {},
                    }
                srt_routing_stats[gt_key]["total"] += 1
                if slot_correct:
                    srt_routing_stats[gt_key]["slot_correct"] += 1
                if label_correct:
                    srt_routing_stats[gt_key]["label_correct"] += 1
                if label_correct != slot_correct:
                    srt_routing_stats[gt_key]["disagree"] += 1
                slot_key = f"slot{pred_slot}"
                srt_routing_stats[gt_key]["slots"][slot_key] = \
                    srt_routing_stats[gt_key]["slots"].get(slot_key, 0) + 1
                # ─────────────────────────────────────────────────────

                dist_str = "  ".join([f"{t[:20]:20s}:{d:.1f}" for t, d in sorted_tasks[:4]])

                if len(top2) > 1:
                    second_name = top2[1][0][:25]
                    second_disp = f"{second_name:25s}(d={top2[1][1]:.2f})"
                else:
                    second_disp = f"{'N/A':25s}(d=0.00)"

                print(
                    f"  [SRT-D] idx={global_idx:3d}  "
                    f"n_tasks={n_tasks}  "
                    f"slot={pred_slot}  "
                    f"1st={pred_task_str[:25]:25s}(d={best_d:.2f})  "
                    f"2nd={second_disp}  "
                    f"conf={conf:.3f}  "
                    f"GT={gt_task[:25]:25s}  "
                    f"{correct_flag}  "
                    f"txt={entry['decoded_first'][:60]!r}"
                )
                print(f"        dists: {dist_str}")
        # ── END SRT DEBUG ─────────────────────────────────────────────

        for generated_ids, reference in zip(generated, refs):
            completion_ids = generated_ids[input_length:]
            generated_lengths.append(int((completion_ids != pad_id).sum().item()))
            prediction = tokenizer.decode(
                completion_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
            predictions.append(prediction)
            references.append(reference.strip())

    model.model.is_inference = False

    # ── SRT ROUTING ACCURACY SUMMARY ─────────────────────────────────
    if srt_routing_stats:
        box_width = 98
        print()
        print("  ┌" + "─"*box_width + "┐")
        print("  │           SRT ROUTING ACCURACY SUMMARY" + " "*(box_width-44) + "│")
        print("  ├" + "─"*box_width + "┤")
        overall_slot_correct = 0
        overall_label_correct = 0
        overall_disagree = 0
        overall_total = 0
        for gt_key, stats in sorted(srt_routing_stats.items()):
            slot_acc = stats["slot_correct"] / stats["total"] * 100 if stats["total"] > 0 else 0.0
            label_acc = stats["label_correct"] / stats["total"] * 100 if stats["total"] > 0 else 0.0
            overall_slot_correct += stats["slot_correct"]
            overall_label_correct += stats["label_correct"]
            overall_disagree += stats["disagree"]
            overall_total += stats["total"]
            slot_dist = "  ".join([f"{k}={v}" for k, v in sorted(stats["slots"].items())])
            line = (
                f"  │  GT={gt_key[:32]:32s}  "
                f"slot_acc={slot_acc:5.1f}%  "
                f"label_acc={label_acc:5.1f}%  "
                f"disagree={stats['disagree']:3d}  "
                f"[{slot_dist[:18]}]"
            )
            print(line + " " * max(0, box_width + 3 - len(line)) + "│")
        overall_slot_acc = overall_slot_correct / overall_total * 100 if overall_total > 0 else 0.0
        overall_label_acc = overall_label_correct / overall_total * 100 if overall_total > 0 else 0.0
        print("  ├" + "─"*box_width + "┤")
        summary_line = (
            f"  │  OVERALL: slot_acc={overall_slot_acc:5.1f}% ({overall_slot_correct:3d}/{overall_total:3d})  "
            f"label_acc={overall_label_acc:5.1f}% ({overall_label_correct:3d}/{overall_total:3d})  "
            f"disagree={overall_disagree:3d}"
        )
        print(summary_line + " " * max(0, box_width + 3 - len(summary_line)) + "│")
        print("  └" + "─"*box_width + "┘")
    else:
        print("  [SRT] No routing data captured.")
    # ── END SRT SUMMARY ───────────────────────────────────────────────

    runtime = time.perf_counter() - start_time
    return predictions, references, generated_lengths, runtime, total_steps


def evaluate_split_cl(
    model,
    tokenizer,
    samples,
    max_source_length,
    max_target_length,
    max_new_tokens,
    batch_size,
    split_name,
    group_names: list[str] | None = None,
    display_name: str | None = None,
) -> dict:
    predictions, references, generated_lengths, _, _ = generate_predictions_cl(
        model, tokenizer, samples, max_source_length, max_target_length, max_new_tokens, batch_size,
        desc=display_name or f"Eval {split_name}",
    )
    legacy_metrics = compute_metrics_local(predictions, references)

    metrics = {
        f"{split_name}_rougeL":            legacy_metrics["eval_rougeL"],
        f"{split_name}_rouge1":            legacy_metrics["rouge1"],
        f"{split_name}_exact_match":       legacy_metrics["exact_match"],
        f"{split_name}_gen_len":           round(float(np.mean(generated_lengths)) if generated_lengths else 0.0, 4),
        f"{split_name}_samples":           len(references),
        f"{split_name}_predictions":       predictions,
        f"{split_name}_references":        references,
    }

    if group_names is not None:
        if len(group_names) != len(references):
            raise ValueError(
                "group_names length must match samples length for continual evaluation"
            )
        grouped = compute_grouped_metrics_local(predictions, references, group_names)
        group_counts = Counter(group_names)
        metrics.update(
            {
                f"{split_name}_{metric}": value
                for metric, value in grouped.items()
            }
        )

    return metrics


def evaluate_continual_split_cl_legacy(
    model,
    tokenizer,
    samples,
    max_source_length,
    max_target_length,
    max_new_tokens,
    batch_size,
    seen_task_names: list[str],
    epoch: float | None = None,
    global_step: int | None = None,
) -> dict:
    """Match the old `predict_*` metric naming and grouping exactly."""
    predictions, references, generated_lengths, runtime, total_steps = generate_predictions_cl(
        model, tokenizer, samples, max_source_length, max_target_length, max_new_tokens, batch_size,
        desc="Continual predict",
    )

    metrics = compute_metrics_local(predictions, references)
    metrics = {f"predict_{key}": value for key, value in metrics.items()}
    metrics["predict_gen_len"] = round(float(np.mean(generated_lengths)) if generated_lengths else 0.0, 4)
    metrics["predict_runtime"] = float(runtime)
    metrics["predict_samples"] = len(references)
    metrics["predict_samples_per_second"] = round(len(references) / max(runtime, 1e-9), 3)
    metrics["predict_steps_per_second"] = round(total_steps / max(runtime, 1e-9), 3)
    if epoch is not None:
        metrics["epoch"] = float(epoch)
    if global_step is not None:
        metrics["predict_global_step"] = int(global_step)

    task_groups = [sample["Task"] for sample in samples]
    dataset_groups = [sample["Dataset"] for sample in samples]
    dataset_counts = Counter(dataset_groups)
    metrics.update({f"predict_{key}": value for key, value in compute_grouped_metrics_local(predictions, references, task_groups).items()})
    metrics.update({f"predict_{key}": value for key, value in compute_grouped_metrics_local(predictions, references, dataset_groups).items()})

    return metrics


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
    parser.add_argument("--srt_shrinkage",      type=str, default="ridge",
                        choices=["ridge", "oas", "lw", "none"],
                        help="PooledMahalanobis shrinkage method (default: ridge)")
    parser.add_argument("--srt_max_emb_samples",type=int,   default=500)
    parser.add_argument("--srt_pca_components", type=int, default=None,
                        help="PCA dims before Mahalanobis (e.g. 128). Reduces 4096→128 for stable Σ estimate.")
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
    model, hf_base_model = load_custom_llama_from_hf_checkpoint(
        model_name_or_path = args.model_name_or_path,
        config             = config,
        prompt_config      = prompt_config,
        token              = token,
        dtype              = dtype,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    # ── FIX: ensure task-local params start from clean init ────────────────
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

    # The frozen SRT backbone is loaded lazily after training. Task 0 should
    # train as a plain single-task LoRA run without carrying an extra 7B model.
    del hf_base_model

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
        srt_shrinkage      = args.srt_shrinkage,
        srt_max_emb_samples= args.srt_max_emb_samples,
        srt_load_path      = args.srt_load_path,
        srt_skip_forward   = args.srt_skip_forward,
        srt_pca_components = getattr(args, 'srt_pca_components', None),
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
    # NOTE: skip trainer.save_metrics() / trainer.save_state() to save disk —
    # everything we need ends up in all_results.json at end of task.

    # ── Restore best checkpoint (mirrors load_best_model_at_end=True in T5) ───
    best_rougeL = trainer.restore_best_model()

    # ── Save LoRA weights (from best checkpoint) ───────────────────────────
    save_path = args.output_dir / "saved_weights"
    save_path.mkdir(parents=True, exist_ok=True)

    torch.save(lora_state_dict_A(model), str(save_path / "lora_weights_A.pt"))
    torch.save(lora_state_dict_B(model), str(save_path / "lora_weights_B.pt"))
    print(f"[SAVE] LoRA weights → {save_path}")

    # ── Post-training: SRT signature ───────────────────────────────────────
    if args.use_srt_router and getattr(model.model, "encoder_frozen", None) is None:
        print(f"[SRT] Loading frozen backbone from {args.model_name_or_path} …")
        frozen_backbone = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            token=token,
            dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        frozen_decoder = (
            frozen_backbone.model
            if hasattr(frozen_backbone, "model")
            else frozen_backbone
        )
        model.model.encoder_frozen = FrozenLlamaExtractor(frozen_decoder)
        # Ensure frozen extractor is on same device as model parameters (handles DeepSpeed/GPU)
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        model.model.encoder_frozen.to(device)
        del frozen_backbone
        print(f"[SRT] Frozen backbone attached (moved to {device}).")

    trainer.on_task_end(cur_task)
    trainer.save_srt_signatures(str(save_path))

    # ── Evaluation ─────────────────────────────────────────────────────────
    model.config.use_cache = True

    print("*** Prediction ***")
    logger.info("*** Prediction ***")

    # Continual_eval: rerun all seen tasks with the current SRT router active.
    seen_task_names = task_order[: cur_task_id + 1]
    continual_samples = build_continual_eval_samples(
        args.data_dir,
        args.task_config_dir.parent,
        seen_task_names,
    )
    continual_metrics = evaluate_continual_split_cl_legacy(
        model,
        tokenizer,
        continual_samples,
        args.max_source_length,
        args.max_target_length,
        args.max_new_tokens,
        args.per_device_eval_batch_size,
        seen_task_names,
        epoch=float(getattr(trainer.state, "epoch", 0.0) or 0.0),
        global_step=int(getattr(trainer.state, "global_step", 0) or 0),
    )

    if not prompt_config["run_single"]:
        save_path = args.output_dir / "saved_weights"
        attention_weights = getattr(getattr(trainer.model, "encoder", None), "all_attn_weights", None)
        if attention_weights is not None:
            with open(os.path.join(save_path, "attention_weights.pkl"), 'wb') as f:
                all_2d = [x for x in attention_weights if getattr(x, "ndim", 0) == 2]
                if all_2d:
                    attn_w = np.array(np.concatenate(all_2d)).mean(axis=0)
                    print(f"{'*'*20} Saving Attention Weights {'*'*20}")
                    print(attn_w)
                    pickle.dump(attn_w, f)
                else:
                    print(f"{'*'*20} No valid 2D Attention Weights — saving empty dict {'*'*20}")
                    pickle.dump({}, f)

    trainer.log(continual_metrics)
    trainer.log_metrics("predict", continual_metrics)

    all_results_path = args.output_dir / "all_results.json"
    all_results = {
        k: v
        for k, v in continual_metrics.items()
        if not k.endswith("_predictions") and not k.endswith("_references")
    }
    if train_metrics:
        for key, value in train_metrics.items():
            if key.startswith("train_") or key == "total_flos":
                all_results[key] = value
    with open(all_results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    outputs_dir = args.output_dir.parent
    with open(os.path.join(outputs_dir, "task_order.txt"), 'w') as f:
        f.write(args.task_order)


if __name__ == "__main__":
    main()

