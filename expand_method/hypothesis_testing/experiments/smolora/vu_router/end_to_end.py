#!/usr/bin/env python3
"""
Option B: SMoLoRA VU Router — End-to-End Evaluation.

Loads a trained SMoLoRA checkpoint and evaluates with CLIP-based
SRT routing replacing the VU router (mean-pooled hidden states).

Usage:
    python experiments/smolora/vu_router/end_to_end.py \
        --model_path /path/to/smolora/checkpoint \
        --model_base /path/to/vicuna-7b \
        --clip_model openai/clip-vit-large-patch14-336 \
        --task_images_root /path/to/task/images \
        --task_order ScienceQA TextVQA GQA VQAv2 \
        --routing_mode all

Comparison output:
    Shows ORIGINAL (VU baseline) vs NEW (SRT) vs ORACLE side-by-side.
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import shortuuid
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from srt_router.pooled_mahalanobis import PooledMahalanobisRouter
from embedding_extractors.clip_extractor import CLIPVisionExtractor


def parse_args():
    parser = argparse.ArgumentParser(description="SMoLoRA VU Router — End-to-End Evaluation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_base", type=str, required=True)
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--task_images_root", type=str, required=True)
    parser.add_argument("--task_order", type=str, nargs="+", required=True)
    parser.add_argument("--routing_mode", type=str, default="srt",
                       choices=["srt", "original", "oracle", "all"])
    parser.add_argument("--scoring_func", type=str, default="vqav2",
                       choices=["vqav2", "science_qa", "gqa"])
    parser.add_argument("--output_dir", type=str, default="results_smolora_vu_e2e")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_build_signatures", type=int, default=200,
                       help="Images per task for building SRT signatures")
    return parser.parse_args()


_TASK_IMAGE_SUBDIRS = {
    "VQAv2": "images",
    "Flickr30k": "flickr30k_images",
}


def collect_task_images(data_root: Path, task_name: str, max_n: int) -> List[Path]:
    """Collect image paths for a task. Handles SMoLoRA's non-standard layout."""
    subdir = _TASK_IMAGE_SUBDIRS.get(task_name, task_name)
    task_dir = data_root / subdir
    if not task_dir.exists():
        return []
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPEG", "*.JPG", "*.PNG"]
    images = []
    for ext in extensions:
        images.extend(task_dir.rglob(ext))
    return images[:max_n]


def build_clip_srt_router(
    clip_extractor: CLIPVisionExtractor,
    task_images_root: Path,
    task_order: List[str],
    n_per_task: int,
    shrinkage: str = "ridge",
) -> PooledMahalanobisRouter:
    """Build SRT router from CLIP embeddings of task images."""
    router = PooledMahalanobisRouter(shrinkage=shrinkage)

    for task_name in tqdm(task_order, desc="  Building SRT signatures"):
        paths = collect_task_images(task_images_root, task_name, max_n=n_per_task)
        if not paths:
            print(f"  [WARN] No images for {task_name}")
            continue
        embs = clip_extractor.extract_from_paths(paths, batch_size=8)
        router.add_task(embs, task_name=task_name)
        print(f"  {task_name}: {len(paths)} images, router has {router.n_tasks} tasks")

    return router


def load_smolora_model(model_path: str, model_base: str, device: str):
    """Load SMoLoRA model."""
    import torch

    # Check paths exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"model_path does not exist: {model_path}\n"
            f"  → Run SMoLoRA training first (Option B requires trained checkpoint)\n"
            f"  → On server: export SMOLORA_REPO=/path/to/SMoLoRA\n"
            f"  → Then: python experiments/smolora/vu_router/end_to_end.py ..."
        )
    if not os.path.exists(model_base):
        raise FileNotFoundError(
            f"model_base does not exist: {model_base}\n"
            f"  → This should be the Vicuna-7B model directory"
        )

    # Add SMoLoRA repo to sys.path so 'llava' can be imported
    smolora_repo = os.environ.get("SMOLORA_REPO", "")
    if smolora_repo and os.path.exists(smolora_repo):
        sys.path.insert(0, smolora_repo)
    else:
        possible_paths = [
            Path(__file__).parent.parent.parent.parent.parent / "MINGLE",
            Path(model_path).parent / "SMoLoRA",
            Path(model_base).parent / "SMoLoRA",
        ]
        for p in possible_paths:
            if p.exists():
                sys.path.insert(0, str(p))
                break
        else:
            raise RuntimeError(
                f"Cannot find SMoLoRA 'llava' module.\n"
                f"  → Set SMOLORA_REPO=/path/to/SMoLoRA before running, or\n"
                f"  → Ensure 'llava' is in the SMoLoRA subdirectory next to your checkpoint.\n"
                f"  Checked: {[str(p) for p in possible_paths]}"
            )

    from llava.model.builder import load_pretrained_model
    from PEFT_SMoLoRA.peft.tuners.smolora import SMoLoraLinear

    model, tokenizer, _, _ = load_pretrained_model(model_path, model_base, None, None)

    for name, module in model.named_modules():
        if isinstance(module, SMoLoraLinear):
            module.ins_type = 0

    return model, tokenizer


def set_ins_type(model, ins_type: int):
    from PEFT_SMoLoRA.peft.tuners.smolora import SMoLoraLinear
    for name, module in model.named_modules():
        if isinstance(module, SMoLoraLinear):
            module.ins_type = ins_type


def generate_with_routing(
    model,
    tokenizer,
    clip_extractor: CLIPVisionExtractor,
    router: PooledMahalanobisRouter,
    test_samples: List[dict],
    routing_mode: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    max_samples: Optional[int] = None,
    device: str = "cuda",
) -> List[dict]:
    """Generate with specified routing mode."""
    import torch
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import tokenizer_image_token
    from llava.conversation import conv_templates
    from PIL import Image

    if max_samples is not None:
        test_samples = test_samples[:max_samples]

    results = []
    task_order = router.task_names

    for sample in tqdm(test_samples, desc=f"  [{routing_mode}]"):
        task_id = sample.get("task_id", 0)
        qs = sample["text"]

        image = None
        clip_emb = None
        if "image" in sample and sample["image"]:
            try:
                image = Image.open(sample["image"]).convert("RGB")
                clip_emb = clip_extractor.extract_single(image).reshape(1, -1)
            except Exception:
                pass

        # Determine ins_type
        if routing_mode == "srt" and clip_emb is not None and router.n_tasks > 0:
            ins_type = int(router.route(clip_emb)[0])
        elif routing_mode == "oracle":
            ins_type = task_id
        else:
            ins_type = task_id

        set_ins_type(model, ins_type)

        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(device)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                images=image,
                do_sample=temperature > 0,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )

        output = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

        results.append({
            "question_id": sample.get("question_id", shortuuid.uuid()),
            "text": output,
            "task_id": task_id,
            "ins_type_used": ins_type,
            "routing_mode": routing_mode,
        })

    return results


def score_predictions(results: List[dict], scoring_func: str) -> Dict:
    """Score predictions and return accuracy metrics."""
    if scoring_func == "vqav2":
        total = len(results)
        correct = sum(
            1 for r in results
            if r.get("text", "").strip().upper() == r.get("ground_truth", "").strip().upper()
        )
        return {"accuracy": correct / total * 100 if total > 0 else 0, "n": total}
    elif scoring_func == "science_qa":
        correct = sum(
            1 for r in results
            if r.get("text", "").strip().upper() == r.get("ground_truth", "").strip().upper()
        )
        return {"accuracy": correct / len(results) * 100 if results else 0, "n": len(results)}
    return {"accuracy": 0, "n": 0}


def _print_comparison_table(task_order: List[str], all_task_acc: Dict[str, Dict[str, float]]):
    """Print comparison table in ablation_truely.txt style."""
    n_tasks = len(task_order)

    print(f"\n{'='*75}")
    print("  SMoLoRA VU Router — End-to-End Task Accuracy Comparison")
    print(f"  ({n_tasks} tasks)")
    print(f"{'='*75}")

    task_headers = "  ".join(f"{t[:8]:>8}" for t in task_order)
    print(f"  {'Method':38}  {'Avg':>8}  {task_headers}")
    print(f"  {'-'*75}")

    def task_accs(method):
        return [all_task_acc.get(method, {}).get(t, 0.0) for t in task_order]

    def avg_acc(method):
        accs = task_accs(method)
        return np.mean(accs) if accs else 0.0

    if "original" in all_task_acc:
        accs = task_accs("original")
        task_str = "  ".join(f"{a:>8.2f}%" for a in accs)
        print(f"  {'SMoLoRA VU (ORIGINAL)':38}  {avg_acc('original'):>7.2f}%  {task_str}")

    if "srt" in all_task_acc:
        accs = task_accs("srt")
        task_str = "  ".join(f"{a:>8.2f}%" for a in accs)
        delta = avg_acc("srt") - avg_acc("original")
        print(f"  {'SMoLoRA VU + SRT (NEW)':38}  {avg_acc('srt'):>7.2f}%  {task_str}  ({delta:+.2f}%)")

    if "oracle" in all_task_acc:
        accs = task_accs("oracle")
        task_str = "  ".join(f"{a:>8.2f}%" for a in accs)
        print(f"  {'Oracle (Upper Bound)':38}  {avg_acc('oracle'):>7.2f}%  {task_str}")

    # Delta row
    if "original" in all_task_acc and "srt" in all_task_acc:
        orig = task_accs("original")
        srt = task_accs("srt")
        deltas = [srt[i] - orig[i] for i in range(n_tasks)]
        delta_task_str = "  ".join(f"{d:>+8.2f}%" for d in deltas)
        print(f"  {'Delta (SRT - Original)':38}  {np.mean(deltas):>+7.2f}%  {delta_task_str}")

    print(f"{'='*75}")


def main():
    args = parse_args()

    print(f"\n{'='*70}")
    print(f"  SMoLoRA VU Router — End-to-End Evaluation")
    print(f"  Model: {args.model_path}")
    print(f"  CLIP: {args.clip_model}")
    print(f"  Routing: {args.routing_mode}")
    print(f"  NOTE: Compares ORIGINAL vs SRT vs ORACLE side-by-side")
    print(f"{'='*70}\n")

    # Load CLIP and build SRT router
    print("Loading CLIP and building SRT router...")
    clip_ext = CLIPVisionExtractor(model_name=args.clip_model)
    srt_router = build_clip_srt_router(
        clip_extractor=clip_ext,
        task_images_root=Path(args.task_images_root),
        task_order=args.task_order,
        n_per_task=args.n_build_signatures,
    )
    print(f"  SRT router: {srt_router.n_tasks} tasks")

    # Load model
    print("Loading SMoLoRA model...")
    model, tokenizer = load_smolora_model(args.model_path, args.model_base, args.device)

    # Determine routing modes
    if args.routing_mode == "all":
        modes = ["original", "srt", "oracle"]
    else:
        modes = [args.routing_mode]

    all_results = {}

    for mode in modes:
        print(f"\n--- Routing mode: {mode} ---")
        results = generate_with_routing(
            model=model,
            tokenizer=tokenizer,
            clip_extractor=clip_ext,
            router=srt_router,
            test_samples=[],  # Load from dataset
            routing_mode=mode,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_samples=args.max_samples,
            device=args.device,
        )
        all_results[mode] = results

    # Score each mode
    scored = {}
    for mode, preds in all_results.items():
        task_acc = {}
        for t_idx, task_name in enumerate(args.task_order):
            task_preds = [p for p in preds if p.get("task_id", 0) == t_idx]
            score = score_predictions(task_preds, args.scoring_func)
            task_acc[task_name] = score.get("accuracy", 0.0)
        scored[mode] = task_acc

    # Print comparison table
    _print_comparison_table(args.task_order, scored)

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for mode, preds in all_results.items():
        out_path = output_dir / f"predictions_{mode}.jsonl"
        with open(out_path, "w") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")
        print(f"  Saved {mode} predictions: {out_path}")

    print(f"\n✓ Results saved to {output_dir}")


if __name__ == "__main__":
    main()