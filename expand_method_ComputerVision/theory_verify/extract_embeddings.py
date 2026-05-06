from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from common import (
    THIS_DIR,
    build_run_name,
    ensure_inflora_imports,
    get_task_bounds,
    get_task_name,
    load_config,
    resolve_torch_device,
    set_seed,
    log,
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
    parser.add_argument("--data_path", default=None,
                        help="Optional override for config['data_path'] when the dataset is mounted elsewhere.")
    parser.add_argument("--device", default=None,
                        help="Torch device. Examples: 0, cuda:0, cpu. Defaults to cuda:0 when available.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Optional override for config batch_size")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Optional override for config num_workers")
    parser.add_argument("--use_amp", action="store_true",
                        help="Use mixed precision (autocast) on CUDA to speed inference")
    parser.add_argument("--fast_mode", action="store_true",
                        help="Favor speed over strict determinism (enables cuDNN benchmark and TF32 on CUDA)")
    parser.add_argument("--prefetch_factor", type=int, default=4,
                        help="DataLoader prefetch factor per worker (num_workers > 0)")
    parser.add_argument("--progress_log_every", type=int, default=20,
                        help="Emit a heartbeat progress log every N batches")
    parser.add_argument("--save_uncompressed", action="store_true",
                        help="Use np.savez (faster write, larger files) instead of np.savez_compressed")
    parser.add_argument("--domainnet_verify", default="sample", choices=["none", "sample", "full"],
                        help="How strictly to verify resolved DomainNet file paths during DataManager setup")
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


def build_loader(dataset, batch_size: int, num_workers: int, seed: int, prefetch_factor: int = 4):
    max_workers = os.cpu_count() or 1
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
        # tune prefetch to keep workers busy (requires num_workers>0)
        try:
            loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
        except Exception:
            pass
    return DataLoader(**loader_kwargs)


def extract_split_embeddings(
    loader,
    backbone,
    descriptor: str,
    device: str,
    limit_per_split: int | None,
    desc: str,
    use_amp: bool = False,
    progress_log_every: int = 20,
):
    embeddings = []
    labels = []
    sample_indices = []
    seen = 0

    dataset_size = None
    try:
        dataset_size = int(len(loader.dataset))
    except Exception:
        dataset_size = None

    target_total = int(min(dataset_size, limit_per_split)) if (dataset_size is not None and limit_per_split is not None) else (
        int(dataset_size) if dataset_size is not None else limit_per_split
    )

    backbone.eval()
    split_start = time.time()
    last_batch_end = split_start

    with torch.inference_mode():
        progress = tqdm(loader, desc=desc, unit="batch", dynamic_ncols=True, leave=False)
        for batch_idx, (batch_indices, images, batch_labels) in enumerate(progress, start=1):
            if limit_per_split is not None and seen >= limit_per_split:
                break

            data_wait = time.time() - last_batch_end
            batch_start = time.time()
            images = images.to(device, non_blocking=True)

            # optionally use mixed precision on CUDA for faster inference
            if use_amp and torch.cuda.is_available() and str(device).startswith("cuda"):
                with torch.cuda.amp.autocast():
                    hidden = backbone.vit.forward_features(images, task=-1)
            else:
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

            # batch timing and throughput
            batch_time = time.time() - batch_start
            fps = float(take) / batch_time if batch_time > 0 else 0.0
            postfix = {"fps": f"{fps:.1f}", "seen": seen}
            if limit_per_split is not None:
                postfix["limit"] = limit_per_split
            progress.set_postfix(postfix)

            elapsed = time.time() - split_start
            avg_fps = float(seen) / elapsed if elapsed > 0 else 0.0
            if target_total is not None and avg_fps > 0:
                remaining = max(target_total - seen, 0) / avg_fps
            else:
                remaining = None

            if batch_idx == 1 or (progress_log_every > 0 and batch_idx % progress_log_every == 0):
                if remaining is None:
                    log(
                        f"[heartbeat] {desc}: batches={batch_idx} seen={seen} "
                        f"avg_fps={avg_fps:.1f} wait={data_wait:.3f}s step={batch_time:.3f}s")
                else:
                    pct = (100.0 * seen / max(target_total, 1))
                    log(
                        f"[heartbeat] {desc}: {seen}/{target_total} ({pct:.1f}%) "
                        f"avg_fps={avg_fps:.1f} eta={remaining/60:.1f}m wait={data_wait:.3f}s step={batch_time:.3f}s")

            last_batch_end = time.time()

    if not embeddings:
        raise RuntimeError("No embeddings were extracted for the requested split.")

    return {
        "embeddings": np.concatenate(embeddings, axis=0).astype(np.float32),
        "labels": np.concatenate(labels, axis=0).astype(np.int64),
        "sample_indices": np.concatenate(sample_indices, axis=0).astype(np.int64),
    }


def save_split(task_dir: Path, split: str, payload: dict, metadata: dict, compressed: bool = True) -> None:
    saver = np.savez_compressed if compressed else np.savez
    saver(
        task_dir / f"{split}.npz",
        embeddings=payload["embeddings"],
        labels=payload["labels"],
        sample_indices=payload["sample_indices"],
        metadata_json=json.dumps(metadata, ensure_ascii=True),
    )


def run_extraction(args: argparse.Namespace) -> Path:
    ensure_inflora_imports()

    DataManager = importlib.import_module("utils.data_manager").DataManager
    ViT_Frozen = importlib.import_module("models.sinet_srt_inflora").ViT_Frozen
    data_utils = importlib.import_module("utils.data")

    config, config_path = load_config(args.config)
    seed = int(args.seed if args.seed is not None else config["seed"][0])
    batch_size = int(args.batch_size if args.batch_size is not None else config["batch_size"])
    requested_num_workers = int(args.num_workers if args.num_workers is not None else config["num_workers"])
    num_workers = min(max(requested_num_workers, 0), os.cpu_count() or 1)
    device = resolve_torch_device(args.device)

    set_seed(seed)
    if args.fast_mode and torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        log("[setup] fast_mode enabled: cudnn.benchmark=True, tf32=True, deterministic=False")

    log(f"[setup] starting extraction config={args.config} seed={seed} device={device} batch_size={batch_size} num_workers={num_workers}")
    setup_t0 = time.time()

    data_args = dict(config)
    data_args["seed"] = seed
    if args.data_path is not None:
        data_args["data_path"] = args.data_path
    data_args["domainnet_verify"] = getattr(args, "domainnet_verify", "sample")
    if data_args["dataset"].lower() == "domainnet":
        # Skip expensive auto-resolution when a valid path already exists.
        candidate_root = data_args.get("data_path")
        candidate_path = Path(candidate_root).expanduser() if candidate_root else None
        if candidate_path is not None and candidate_path.exists():
            data_args["data_path"] = str(candidate_path)
            log(f"[setup] using provided DomainNet path directly: {data_args['data_path']}")
        else:
            resolve_t0 = time.time()
            resolved_data_path = data_utils.resolve_domainnet_root(candidate_root)
            if resolved_data_path is not None:
                data_args["data_path"] = resolved_data_path
            log(f"[setup] resolve_domainnet_root done in {time.time() - resolve_t0:.1f}s -> {data_args.get('data_path')}")
    log("[setup] building DataManager (this can be slow on large datasets)...")
    dm_t0 = time.time()
    try:
        data_manager = DataManager(
            data_args["dataset"],
            data_args["shuffle"],
            seed,
            data_args["init_cls"],
            data_args["increment"],
            data_args,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{exc}\n"
            f"The DomainNet dataset is not mounted where the config expects it. "
            f"Resolved data_path was {data_args.get('data_path')!r}. "
            f"Pass --data_path to the actual root, or set DOMAINNET_ROOT to the parent directory that contains DomainNet/domainnet/data/DomainNet.") from exc
    data_args["class_order"] = data_manager._class_order
    log(f"[setup] DataManager ready in {time.time() - dm_t0:.1f}s with nb_tasks={data_manager.nb_tasks}")

    model_t0 = time.time()
    backbone = ViT_Frozen(data_args).to(device)
    backbone.eval()
    log(f"[setup] backbone loaded in {time.time() - model_t0:.1f}s")

    model_device = next(backbone.parameters()).device
    log(f"[setup] device={device} model_device={model_device} cuda_available={torch.cuda.is_available()}")
    log(f"[setup] backbone=ViT_Frozen vit={backbone.vit.__class__.__name__}")
    log(f"[setup] requested_num_workers={requested_num_workers} effective_num_workers={num_workers}")
    log(f"[setup] data_path={data_args.get('data_path')}")

    output_root = Path(args.output_root)
    run_dir = output_root / build_run_name(data_args, args.descriptor)
    if run_dir.exists() and not args.force:
        raise FileExistsError(f"{run_dir} already exists. Use --force to overwrite.")
    run_dir.mkdir(parents=True, exist_ok=True)

    max_tasks = data_manager.nb_tasks if args.max_tasks is None else min(args.max_tasks, data_manager.nb_tasks)
    log(f"[setup] extraction setup finished in {time.time() - setup_t0:.1f}s; scheduled_tasks={max_tasks}")
    task_specs = []

    overall_start = time.time()
    task_durations: list[float] = []

    for task_id in range(max_tasks):
        task_start = time.time()
        start_class, end_class = get_task_bounds(data_manager._increments, task_id)
        task_name = get_task_name(task_id, start_class, end_class)
        task_dir = run_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        log(f"[task {task_id + 1}/{max_tasks}] {task_name} classes={start_class}-{end_class - 1}")

        class_indices = np.arange(start_class, end_class)
        train_dataset = data_manager.get_dataset(class_indices, source="train", mode="train")
        test_dataset = data_manager.get_dataset(class_indices, source="test", mode="test")

        train_loader = build_loader(
            train_dataset,
            batch_size,
            num_workers,
            seed + task_id * 17,
            prefetch_factor=args.prefetch_factor,
        )
        test_loader = build_loader(
            test_dataset,
            batch_size,
            num_workers,
            seed + task_id * 17 + 1,
            prefetch_factor=args.prefetch_factor,
        )

        train_payload = extract_split_embeddings(
            train_loader,
            backbone,
            args.descriptor,
            device,
            args.limit_per_split,
            desc=f"{task_name} train",
            use_amp=getattr(args, "use_amp", False),
            progress_log_every=max(1, int(getattr(args, "progress_log_every", 20))),
        )
        test_payload = extract_split_embeddings(
            test_loader,
            backbone,
            args.descriptor,
            device,
            args.limit_per_split,
            desc=f"{task_name} test",
            use_amp=getattr(args, "use_amp", False),
            progress_log_every=max(1, int(getattr(args, "progress_log_every", 20))),
        )

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
        save_t0 = time.time()
        use_compression = not bool(getattr(args, "save_uncompressed", False))
        save_split(task_dir, "train", train_payload, {**split_common, "split": "train"}, compressed=use_compression)
        save_split(task_dir, "test", test_payload, {**split_common, "split": "test"}, compressed=use_compression)
        log(f"[save] {task_name} persisted in {time.time() - save_t0:.1f}s compression={'on' if use_compression else 'off'}")

        task_specs.append({
            "task_id": task_id,
            "task_name": task_name,
            "start_class": start_class,
            "end_class_exclusive": end_class,
            "train_count": int(train_payload["embeddings"].shape[0]),
            "test_count": int(test_payload["embeddings"].shape[0]),
        })

        log(
            f"[extract] {task_name}: train={train_payload['embeddings'].shape} "
            f"test={test_payload['embeddings'].shape}")

        # timing info for task
        task_time = time.time() - task_start
        task_durations.append(task_time)
        avg = sum(task_durations) / len(task_durations)
        remaining = avg * (max_tasks - (task_id + 1))
        log(f"[timing] task {task_id + 1}/{max_tasks} took {task_time:.1f}s; avg {avg:.1f}s; est remaining {remaining/60:.1f}min")

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

    total_elapsed = time.time() - overall_start
    log(f"[done] saved embeddings to {run_dir} total_elapsed={total_elapsed:.1f}s")
    return run_dir


def main() -> None:
    run_extraction(parse_args())


if __name__ == "__main__":
    main()
