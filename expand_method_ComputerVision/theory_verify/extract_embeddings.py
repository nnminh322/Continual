from __future__ import annotations

import argparse
import importlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import (
    THIS_DIR,
    build_run_name,
    ensure_inflora_imports,
    get_task_bounds,
    get_task_name,
    load_config,
    resolve_torch_device,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frozen vision embeddings under the original InfLoRA continual protocol.")
    parser.add_argument("--config", required=True,
                        help="Path to an InfLoRA JSON config, e.g. configs/domainnet_srt_inflora.json")
    parser.add_argument("--output_root", default=str(THIS_DIR / "embeddings"),
                        help="Root directory for saved embedding shards")
    parser.add_argument("--descriptor", default="cls",
                        choices=["cls", "mean_patch", "cls_mean_concat"],
                        help="Frozen descriptor to save. 'cls' matches the current runtime path.")
    parser.add_argument("--device", default=None,
                        help="Torch device. Examples: 0, cuda:0, cpu. Defaults to cuda:0 when available.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Optional override for config batch_size")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Optional override for config num_workers")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override the first config seed for deterministic extraction")
    parser.add_argument("--max_tasks", type=int, default=None,
                        help="Only extract the first N tasks")
    parser.add_argument("--limit_per_split", type=int, default=None,
                        help="Debug option: cap saved samples per train/test split")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing run directory")
    return parser.parse_args()


def build_descriptor(hidden, descriptor):
    cls_token = hidden[:, 0]
    patch_mean = hidden[:, 1:].mean(dim=1)

    if descriptor == "cls":
        return cls_token
    if descriptor == "mean_patch":
        return patch_mean
    if descriptor == "cls_mean_concat":
        return torch.cat([cls_token, patch_mean], dim=-1)
    raise ValueError(f"Unknown descriptor: {descriptor}")


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def build_loader(dataset, batch_size: int, num_workers: int, seed: int):
    max_workers = min(4, os.cpu_count() or 1)
    safe_num_workers = min(max(num_workers, 0), max_workers)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": safe_num_workers,
        "pin_memory": torch.cuda.is_available(),
        "generator": generator,
    }
    if safe_num_workers > 0:
        loader_kwargs["worker_init_fn"] = worker_init_fn
        loader_kwargs["persistent_workers"] = True
    return DataLoader(**loader_kwargs)


def extract_split_embeddings(loader, backbone, descriptor: str, device: str, limit_per_split: int | None):
    embeddings = []
    labels = []
    sample_indices = []
    seen = 0

    backbone.eval()
    with torch.no_grad():
        for batch_indices, images, batch_labels in loader:
            if limit_per_split is not None and seen >= limit_per_split:
                break

            images = images.to(device, non_blocking=True)
            hidden = backbone.vit.forward_features(images, task=-1)
            feats = build_descriptor(hidden, descriptor)

            if limit_per_split is not None:
                take = min(len(batch_labels), limit_per_split - seen)
                feats = feats[:take]
                batch_labels = batch_labels[:take]
                batch_indices = batch_indices[:take]
            else:
                take = len(batch_labels)

            embeddings.append(feats.cpu().float().numpy())
            labels.append(batch_labels.cpu().numpy().astype(np.int64))
            sample_indices.append(batch_indices.cpu().numpy().astype(np.int64))
            seen += take

    if not embeddings:
        raise RuntimeError("No embeddings were extracted for the requested split.")

    return {
        "embeddings": np.concatenate(embeddings, axis=0).astype(np.float32),
        "labels": np.concatenate(labels, axis=0).astype(np.int64),
        "sample_indices": np.concatenate(sample_indices, axis=0).astype(np.int64),
    }


def save_split(task_dir: Path, split: str, payload: dict, metadata: dict) -> None:
    np.savez_compressed(
        task_dir / f"{split}.npz",
        embeddings=payload["embeddings"],
        labels=payload["labels"],
        sample_indices=payload["sample_indices"],
        metadata_json=json.dumps(metadata, ensure_ascii=True),
    )


def main() -> None:
    args = parse_args()
    ensure_inflora_imports()

    DataManager = importlib.import_module("utils.data_manager").DataManager
    ViT_Frozen = importlib.import_module("models.sinet_srt_inflora").ViT_Frozen

    config, config_path = load_config(args.config)
    seed = int(args.seed if args.seed is not None else config["seed"][0])
    batch_size = int(args.batch_size if args.batch_size is not None else config["batch_size"])
    requested_num_workers = int(args.num_workers if args.num_workers is not None else config["num_workers"])
    num_workers = min(max(requested_num_workers, 0), min(4, os.cpu_count() or 1))
    device = resolve_torch_device(args.device)

    set_seed(seed)

    data_args = dict(config)
    data_args["seed"] = seed
    data_manager = DataManager(
        data_args["dataset"],
        data_args["shuffle"],
        seed,
        data_args["init_cls"],
        data_args["increment"],
        data_args,
    )
    data_args["class_order"] = data_manager._class_order

    backbone = ViT_Frozen(data_args).to(device)
    backbone.eval()

    print(f"[setup] device={device} requested_num_workers={requested_num_workers} effective_num_workers={num_workers}")

    output_root = Path(args.output_root)
    run_dir = output_root / build_run_name(data_args, args.descriptor)
    if run_dir.exists() and not args.force:
        raise FileExistsError(f"{run_dir} already exists. Use --force to overwrite.")
    run_dir.mkdir(parents=True, exist_ok=True)

    max_tasks = data_manager.nb_tasks if args.max_tasks is None else min(args.max_tasks, data_manager.nb_tasks)
    task_specs = []

    for task_id in range(max_tasks):
        start_class, end_class = get_task_bounds(data_manager._increments, task_id)
        task_name = get_task_name(task_id, start_class, end_class)
        task_dir = run_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        class_indices = np.arange(start_class, end_class)
        train_dataset = data_manager.get_dataset(class_indices, source="train", mode="train")
        test_dataset = data_manager.get_dataset(class_indices, source="test", mode="test")

        train_loader = build_loader(train_dataset, batch_size, num_workers, seed + task_id * 17)
        test_loader = build_loader(test_dataset, batch_size, num_workers, seed + task_id * 17 + 1)

        train_payload = extract_split_embeddings(
            train_loader, backbone, args.descriptor, device, args.limit_per_split)
        test_payload = extract_split_embeddings(
            test_loader, backbone, args.descriptor, device, args.limit_per_split)

        split_common = {
            "dataset": data_args["dataset"],
            "task_id": task_id,
            "task_name": task_name,
            "start_class": start_class,
            "end_class_exclusive": end_class,
            "descriptor": args.descriptor,
            "backbone": "vit_base_patch16_224_in21k",
            "seed": seed,
            "config_path": str(config_path),
            "class_order": [int(x) for x in data_manager._class_order],
            "train_protocol": "source=train,mode=train,shuffle=False,zero-rehearsal",
            "test_protocol": "source=test,mode=test,eval-over-current-task-only",
            "limit_per_split": args.limit_per_split,
        }
        save_split(task_dir, "train", train_payload, {**split_common, "split": "train"})
        save_split(task_dir, "test", test_payload, {**split_common, "split": "test"})

        task_specs.append({
            "task_id": task_id,
            "task_name": task_name,
            "start_class": start_class,
            "end_class_exclusive": end_class,
            "train_count": int(train_payload["embeddings"].shape[0]),
            "test_count": int(test_payload["embeddings"].shape[0]),
        })

        print(
            f"[extract] {task_name}: train={train_payload['embeddings'].shape} "
            f"test={test_payload['embeddings'].shape}")

    metadata = {
        "dataset": data_args["dataset"],
        "config_path": str(config_path),
        "descriptor": args.descriptor,
        "backbone": "vit_base_patch16_224_in21k",
        "seed": seed,
        "device": device,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "shuffle": bool(data_args["shuffle"]),
        "init_cls": int(data_args["init_cls"]),
        "increment": int(data_args["increment"]),
        "increments": [int(x) for x in data_manager._increments],
        "class_order": [int(x) for x in data_manager._class_order],
        "task_specs": task_specs,
        "embedding_dim": int(train_payload["embeddings"].shape[1]) if task_specs else 0,
        "zero_rehearsal_protocol": True,
        "notes": [
            "Train embeddings are extracted from the original current-task train split.",
            "Train split uses the same InfLoRA train transform path (mode=train).",
            "Test embeddings are extracted task-by-task and evaluated cumulatively offline.",
        ],
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"[done] saved embeddings to {run_dir}")


if __name__ == "__main__":
    main()
