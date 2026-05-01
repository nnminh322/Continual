#!/usr/bin/env python3
"""
Master script: Run all SRT hypothesis testing experiments.

Usage:
    # Option A: Routing accuracy (no trained model needed)
    python experiments/run_all.py --option a --exp smolora_if

    # Option A: All experiments
    python experiments/run_all.py --option a --exp all

    # Option B: End-to-end (requires trained checkpoints)
    python experiments/run_all.py --option b --exp all --checkpoint /path/to/checkpoint

Arguments:
    --exp: smolora_if | smolora_vu | smolora_dual | hide | all
    --option: a (routing accuracy) | b (end-to-end)
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

EXPERIMENTS = {
    "smolora_if": {
        "name": "SMoLoRA IF Router",
        "option_a": {
            "script": "experiments/smolora/if_router/routing_accuracy.py",
            "args": ["--ins_emb", "path/to/ins_emb_single.pkl",
                     "--task_names", "ScienceQA", "TextVQA", "GQA", "VQAv2",
                     "Flickr30k", "ImageNet", "Place365"],
        },
        "option_b": {
            "script": "experiments/smolora/if_router/end_to_end.py",
            "args": ["--model_path", "path/to/checkpoint",
                     "--model_base", "path/to/vicuna-7b",
                     "--ins_emb", "path/to/ins_emb_single.pkl",
                     "--task_order", "ScienceQA", "TextVQA", "GQA", "VQAv2",
                     "--routing_mode", "all",
                     "--scoring_func", "vqav2"],
        },
    },
    "smolora_vu": {
        "name": "SMoLoRA VU Router",
        "option_a": {
            "script": "experiments/smolora/vu_router/routing_accuracy.py",
            "args": ["--data_root", "path/to/images",
                     "--task_names", "ScienceQA", "TextVQA", "GQA", "VQAv2",
                     "Flickr30k", "ImageNet", "Place365",
                     "--clip_model", "openai/clip-vit-large-patch14-336"],
        },
        "option_b": {
            "script": "experiments/smolora/vu_router/end_to_end.py",
            "args": ["--model_path", "path/to/checkpoint",
                     "--model_base", "path/to/vicuna-7b",
                     "--clip_model", "openai/clip-vit-large-patch14-336",
                     "--task_images_root", "path/to/images",
                     "--task_order", "ScienceQA", "TextVQA", "GQA", "VQAv2",
                     "--routing_mode", "all"],
        },
    },
    "smolora_dual": {
        "name": "SMoLoRA Dual Router",
        "option_a": {
            "script": "experiments/smolora/dual_router/routing_accuracy.py",
            "args": ["--ins_emb", "path/to/ins_emb_single.pkl",
                     "--data_root", "path/to/images",
                     "--task_names", "ScienceQA", "TextVQA", "GQA", "VQAv2",
                     "--clip_model", "openai/clip-vit-large-patch14-336",
                     "--alpha", "0.5"],
        },
        "option_b": {
            "script": "experiments/smolora/dual_router/end_to_end.py",
            "args": ["--model_path", "path/to/checkpoint",
                     "--model_base", "path/to/vicuna-7b",
                     "--ins_emb", "path/to/ins_emb_single.pkl",
                     "--clip_model", "openai/clip-vit-large-patch14-336",
                     "--task_images_root", "path/to/images",
                     "--task_order", "ScienceQA", "TextVQA", "GQA", "VQAv2",
                     "--routing_mode", "all",
                     "--alpha", "0.5"],
        },
    },
    "hide": {
        "name": "HiDe-LLaVA Cosine → SRT",
        "option_a": {
            "script": "experiments/hide/cosine_router/routing_accuracy.py",
            "args": ["--data_root", "path/to/images",
                     "--task_names", "ScienceQA", "TextVQA", "GQA", "VQAv2",
                     "VizWiz", "TextCaps",
                     "--clip_model", "openai/clip-vit-large-patch14-336"],
        },
        "option_b": {
            "script": "experiments/hide/cosine_router/end_to_end.py",
            "args": ["--model_path", "path/to/checkpoint",
                     "--model_base", "path/to/vicuna-7b",
                     "--clip_model", "openai/clip-vit-large-patch14-336",
                     "--task_images_root", "path/to/images",
                     "--task_order", "ScienceQA", "TextVQA", "GQA", "VQAv2",
                     "VizWiz", "TextCaps",
                     "--routing_mode", "all"],
        },
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="SRT Hypothesis Testing — Master Runner")
    parser.add_argument(
        "--option",
        type=str,
        choices=["a", "b"],
        required=True,
        help="'a' = routing accuracy, 'b' = end-to-end",
    )
    parser.add_argument(
        "--exp",
        type=str,
        default="all",
        choices=list(EXPERIMENTS.keys()) + ["all"],
        help="Which experiment to run",
    )
    parser.add_argument(
        "--shrinkage",
        type=str,
        default="ridge",
        choices=["ridge", "oas", "lw", "none"],
        help="SRT shrinkage method",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit samples (for debugging)",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help="Suffix for output directory",
    )
    return parser.parse_args()


def run_experiment(exp_key, option, shrinkage, max_samples, suffix, base_args):
    exp = EXPERIMENTS[exp_key]
    opt_key = f"option_{option}"
    config = exp[opt_key]

    script = config["script"]
    args = list(config["args"])

    # Override shrinkage
    args = [a for a in args if a != "--shrinkage"]
    shrink_idx = None
    for i, a in enumerate(args):
        if a == "--shrinkage" and i + 1 < len(args):
            args[i + 1] = shrinkage
            shrink_idx = i
            break
    if shrink_idx is None:
        args.extend(["--shrinkage", shrinkage])

    # Override output_dir with suffix
    for i, a in enumerate(args):
        if "--output_dir" in a and i + 1 < len(args):
            args[i + 1] = args[i + 1] + suffix

    # Add max_samples limit if debugging
    if max_samples is not None:
        args.extend(["--max_samples", str(max_samples)])

    # Merge base_args overrides
    for i, a in enumerate(args):
        for j, b in enumerate(base_args):
            if a == b and i < len(args) - 1:
                args[i + 1] = base_args[j + 1]

    cmd = [sys.executable, script] + args
    print(f"\n{'='*70}")
    print(f"  Running: {exp['name']} ({opt_key})")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*70}")

    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    args = parse_args()

    print(f"\n{'#'*70}")
    print(f"  SRT Hypothesis Testing Framework — Master Runner")
    print(f"  Option: {'A' if args.option == 'a' else 'B'} ({'Routing Accuracy' if args.option == 'a' else 'End-to-End'})")
    print(f"  Shrinkage: {args.shrinkage}")
    print(f"  Max samples: {args.max_samples or 'unlimited'}")
    print(f"{'#'*70}")

    # Determine which experiments to run
    if args.exp == "all":
        exp_keys = list(EXPERIMENTS.keys())
    else:
        exp_keys = [args.exp]

    suffix = f"_{args.output_suffix}" if args.output_suffix else ""

    success_count = 0
    fail_count = 0

    for exp_key in exp_keys:
        ok = run_experiment(
            exp_key, args.option, args.shrinkage,
            args.max_samples, suffix, []
        )
        if ok:
            success_count += 1
        else:
            fail_count += 1

    print(f"\n{'='*70}")
    print(f"  Summary: {success_count} succeeded, {fail_count} failed")
    print(f"{'='*70}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
