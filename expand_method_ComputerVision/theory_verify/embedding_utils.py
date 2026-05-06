from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from common import log


def should_enable_multi_gpu(mode: str, device: str) -> bool:
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        return False
    gpu_count = torch.cuda.device_count()
    if mode == "on":
        return gpu_count > 1
    if mode == "off":
        return False
    return gpu_count > 1


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
        try:
            loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
        except Exception:
            pass
    return DataLoader(**loader_kwargs)


def extract_split_embeddings(
    loader,
    feature_model,
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

    feature_model.eval()
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

            if use_amp and torch.cuda.is_available() and str(device).startswith("cuda"):
                with torch.cuda.amp.autocast():
                    hidden = feature_model(images)
            else:
                hidden = feature_model(images)

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