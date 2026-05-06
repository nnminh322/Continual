from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset

from common import THIS_DIR, SPROMPTS_ROOT, ensure_sprompts_imports, log, resolve_torch_device, set_seed
from extract_embeddings import build_loader, extract_split_embeddings, save_split, should_enable_multi_gpu


OFFICEHOME_DOMAINS = ("Art", "Clipart", "Product", "Real_World")
OFFICEHOME_DOMAIN_ALIASES = {
    "Art": ("Art", "art"),
    "Clipart": ("Clipart", "clipart"),
    "Product": ("Product", "product"),
    "Real_World": ("Real_World", "Real World", "RealWorld", "real_world", "realworld"),
}
OFFICEHOME_ROOT_NAMES = (
    "OfficeHome",
    "officehome",
    "Office-Home",
    "office-home",
    "OfficeHomeDataset_10072016",
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OFFICEHOME_DOWNLOAD_URL = "https://www.hemanthdv.org/officeHomeDataset.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frozen Office-Home embeddings with the CLIP backbone shipped in S-Prompts.")
    parser.add_argument("--output_root", default=str(THIS_DIR / "embeddings"),
                        help="Root directory for saved embedding shards")
    parser.add_argument("--descriptor", default="cls",
                        choices=["cls", "mean_patch", "cls_mean_concat"],
                        help="Frozen descriptor to save from CLIP token outputs")
    parser.add_argument("--data_path", default=None,
                        help="Optional Office-Home root override. Can point to the dataset root or its parent.")
    parser.add_argument("--device", default=None,
                        help="Torch device. Examples: 0, cuda:0, cpu. Defaults to cuda:0 when available.")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size used during feature extraction")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of DataLoader workers")
    parser.add_argument("--use_amp", action="store_true",
                        help="Use mixed precision (autocast) on CUDA to speed inference")
    parser.add_argument("--multi_gpu", default="auto", choices=["auto", "off", "on"],
                        help="Use DataParallel across visible GPUs for extraction")
    parser.add_argument("--fast_mode", action="store_true",
                        help="Favor speed over strict determinism (enables cuDNN benchmark and TF32 on CUDA)")
    parser.add_argument("--prefetch_factor", type=int, default=4,
                        help="DataLoader prefetch factor per worker (num_workers > 0)")
    parser.add_argument("--progress_log_every", type=int, default=20,
                        help="Emit a heartbeat progress log every N batches")
    parser.add_argument("--save_uncompressed", action="store_true",
                        help="Use np.savez (faster write, larger files) instead of np.savez_compressed")
    parser.add_argument("--seed", type=int, default=1993,
                        help="Seed used for deterministic train/test splitting and feature extraction")
    parser.add_argument("--max_tasks", type=int, default=None,
                        help="Only extract the first N domains in the requested order")
    parser.add_argument("--limit_per_split", type=int, default=None,
                        help="Debug option: cap saved samples per train/test split")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing run directory")
    parser.add_argument("--backbone", default="ViT-B/16",
                        help="CLIP backbone name supported by S-Prompts")
    parser.add_argument("--task_order", default=",".join(OFFICEHOME_DOMAINS),
                        help="Comma-separated Office-Home domain order. Defaults to Art,Clipart,Product,Real_World")
    parser.add_argument("--train_split_ratio", type=float, default=0.8,
                        help="Per-class ratio assigned to the train split inside each domain")
    return parser.parse_args()


def sanitize_name(raw: str) -> str:
    return raw.lower().replace("/", "-").replace(" ", "").replace("_", "")


def build_sprompt_clip_run_name(descriptor: str, backbone: str, task_order: list[str]) -> str:
    order_tag = "-".join(sanitize_name(domain) for domain in task_order)
    return f"officehome__spromptclip__{sanitize_name(backbone)}__{descriptor}__{order_tag}"


def _iter_officehome_root_candidates(data_path: str | None):
    if data_path:
        raw = Path(data_path).expanduser()
        yield raw
        for root_name in OFFICEHOME_ROOT_NAMES:
            yield raw / root_name

        if raw.exists() and raw.is_dir():
            try:
                child_dirs = [child for child in raw.iterdir() if child.is_dir()]
            except OSError:
                child_dirs = []

            for child in child_dirs:
                yield child
                for root_name in OFFICEHOME_ROOT_NAMES:
                    yield child / root_name

    common_roots = [
        SPROMPTS_ROOT / "data",
        SPROMPTS_ROOT.parent / "data",
        SPROMPTS_ROOT.parent.parent / "data",
        Path.cwd(),
        Path("/kaggle/input"),
        Path("/content"),
    ]
    for base in common_roots:
        yield base
        for root_name in OFFICEHOME_ROOT_NAMES:
            yield base / root_name

        if not base.exists() or not base.is_dir():
            continue

        try:
            child_dirs = [child for child in base.iterdir() if child.is_dir()]
        except OSError:
            child_dirs = []

        for child in child_dirs:
            yield child
            for root_name in OFFICEHOME_ROOT_NAMES:
                yield child / root_name


def _resolve_domain_dirs(root: Path) -> dict[str, Path] | None:
    resolved = {}
    for canonical, aliases in OFFICEHOME_DOMAIN_ALIASES.items():
        match = None
        for alias in aliases:
            candidate = root / alias
            if candidate.is_dir():
                match = candidate
                break
        if match is None:
            return None
        resolved[canonical] = match
    return resolved


def resolve_officehome_root(data_path: str | None) -> tuple[Path, dict[str, Path]]:
    checked = []
    for candidate in _iter_officehome_root_candidates(data_path):
        candidate = candidate.expanduser().resolve()
        checked.append(str(candidate))
        if not candidate.exists() or not candidate.is_dir():
            continue
        domain_dirs = _resolve_domain_dirs(candidate)
        if domain_dirs is not None:
            return candidate, domain_dirs

    raise FileNotFoundError(
        "Could not locate Office-Home. Checked: {}. "
        "Download the dataset from {} and either mount it under /kaggle/input or pass "
        "--data_path to the dataset root or its parent directory, e.g. --data_path /kaggle/input/<dataset-slug>.".format(
            ", ".join(checked[:20]) + (" ..." if len(checked) > 20 else ""),
            OFFICEHOME_DOWNLOAD_URL,
        ))


def parse_task_order(raw: str) -> list[str]:
    tokens = [item.strip() for item in raw.split(",") if item.strip()]
    if not tokens:
        raise ValueError("task_order must contain at least one Office-Home domain.")

    normalized = []
    used = set()
    alias_lookup = {}
    for canonical, aliases in OFFICEHOME_DOMAIN_ALIASES.items():
        for alias in aliases:
            alias_lookup[sanitize_name(alias)] = canonical

    for token in tokens:
        key = sanitize_name(token)
        if key not in alias_lookup:
            raise ValueError("Unknown Office-Home domain in task_order: {}".format(token))
        canonical = alias_lookup[key]
        if canonical in used:
            raise ValueError("Duplicate Office-Home domain in task_order: {}".format(token))
        normalized.append(canonical)
        used.add(canonical)

    return normalized


def discover_shared_class_names(domain_dirs: dict[str, Path]) -> list[str]:
    class_sets = []
    for domain_name in OFFICEHOME_DOMAINS:
        classes = {entry.name for entry in domain_dirs[domain_name].iterdir() if entry.is_dir()}
        if not classes:
            raise RuntimeError("No class folders found under {}".format(domain_dirs[domain_name]))
        class_sets.append(classes)

    shared = sorted(set.intersection(*class_sets))
    if not shared:
        raise RuntimeError("No shared Office-Home class names found across domains.")
    return shared


def _split_index(count: int, train_ratio: float) -> int:
    if count <= 1:
        return count
    split = int(round(count * train_ratio))
    return min(max(split, 1), count - 1)


def _stable_sort_key(path: Path, seed: int, domain_name: str, class_name: str) -> str:
    digest = hashlib.sha1(f"{seed}:{domain_name}:{class_name}:{path.name}".encode("utf-8")).hexdigest()
    return digest


def collect_class_images(class_dir: Path) -> list[Path]:
    return sorted(
        [path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: path.name,
    )


class OfficeHomeDomainDataset(Dataset):
    def __init__(
        self,
        domain_name: str,
        domain_root: Path,
        class_names: list[str],
        split: str,
        train_ratio: float,
        seed: int,
        transform,
    ):
        if split not in {"train", "test"}:
            raise ValueError("Unknown split: {}".format(split))

        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        for label, class_name in enumerate(class_names):
            class_dir = domain_root / class_name
            if not class_dir.is_dir():
                continue

            images = collect_class_images(class_dir)
            images = sorted(images, key=lambda path: _stable_sort_key(path, seed, domain_name, class_name))
            split_at = _split_index(len(images), train_ratio)
            chosen = images[:split_at] if split == "train" else images[split_at:]
            self.samples.extend((path, label) for path in chosen)

        if not self.samples:
            raise RuntimeError("No Office-Home samples found for {} split={} under {}".format(domain_name, split, domain_root))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, label = self.samples[idx]
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return idx, tensor, label


class FrozenClipTokenExtractor(nn.Module):
    def __init__(self, visual: nn.Module, dtype: torch.dtype):
        super().__init__()
        self.visual = visual
        self.dtype = dtype

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.visual.conv1(image.type(self.dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        cls = self.visual.class_embedding.to(x.dtype)
        cls = cls + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.visual.ln_post(x)
        if self.visual.proj is not None:
            x = x @ self.visual.proj
        return x


def run_extraction(args: argparse.Namespace) -> Path:
    ensure_sprompts_imports()

    from models.clip.clip import _transform
    from models.clip.prompt_learner import cfgc, load_clip_to_cpu

    device = resolve_torch_device(args.device)
    set_seed(int(args.seed))
    if args.fast_mode and torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    officehome_root, domain_dirs = resolve_officehome_root(args.data_path)
    task_order = parse_task_order(args.task_order)
    if args.max_tasks is not None:
        task_order = task_order[:args.max_tasks]

    class_names = discover_shared_class_names(domain_dirs)
    run_dir = Path(args.output_root) / build_sprompt_clip_run_name(args.descriptor, args.backbone, task_order)
    if run_dir.exists():
        if not args.force:
            raise FileExistsError(f"{run_dir} already exists. Use --force to overwrite.")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    clip_cfg = cfgc()
    clip_cfg.backbonename = args.backbone
    clip_model = load_clip_to_cpu(clip_cfg)
    feature_model = FrozenClipTokenExtractor(clip_model.visual, clip_model.dtype)
    for parameter in feature_model.parameters():
        parameter.requires_grad_(False)
    feature_model = feature_model.eval().to(device)

    if should_enable_multi_gpu(args.multi_gpu, device):
        feature_model = nn.DataParallel(feature_model)

    transform = _transform(clip_model.visual.input_resolution)

    log(f"[setup] benchmark=officehome_sprompt_clip root={officehome_root}")
    log(f"[setup] backbone={args.backbone} descriptor={args.descriptor} class_count={len(class_names)}")
    log(f"[setup] task_order={task_order}")
    log(f"[setup] device={device} multi_gpu={isinstance(feature_model, nn.DataParallel)}")

    task_specs = []
    total_start = time.time()
    for task_id, domain_name in enumerate(task_order):
        domain_root = domain_dirs[domain_name]
        task_start = time.time()
        task_dir = run_dir / f"task_{task_id:02d}_{sanitize_name(domain_name)}"
        task_dir.mkdir(parents=True, exist_ok=True)

        train_dataset = OfficeHomeDomainDataset(
            domain_name=domain_name,
            domain_root=domain_root,
            class_names=class_names,
            split="train",
            train_ratio=float(args.train_split_ratio),
            seed=int(args.seed),
            transform=transform,
        )
        test_dataset = OfficeHomeDomainDataset(
            domain_name=domain_name,
            domain_root=domain_root,
            class_names=class_names,
            split="test",
            train_ratio=float(args.train_split_ratio),
            seed=int(args.seed),
            transform=transform,
        )

        train_loader = build_loader(
            train_dataset,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            seed=int(args.seed),
            prefetch_factor=int(args.prefetch_factor),
        )
        test_loader = build_loader(
            test_dataset,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            seed=int(args.seed),
            prefetch_factor=int(args.prefetch_factor),
        )

        log(f"[task {task_id + 1}/{len(task_order)}] domain={domain_name} train={len(train_dataset)} test={len(test_dataset)}")
        train_payload = extract_split_embeddings(
            train_loader,
            feature_model,
            args.descriptor,
            device,
            args.limit_per_split,
            desc=f"{domain_name}:train",
            use_amp=bool(args.use_amp),
            progress_log_every=int(args.progress_log_every),
        )
        test_payload = extract_split_embeddings(
            test_loader,
            feature_model,
            args.descriptor,
            device,
            args.limit_per_split,
            desc=f"{domain_name}:test",
            use_amp=bool(args.use_amp),
            progress_log_every=int(args.progress_log_every),
        )

        split_metadata = {
            "domain_name": domain_name,
            "descriptor": args.descriptor,
            "backbone": args.backbone,
            "train_split_ratio": float(args.train_split_ratio),
            "class_count": len(class_names),
        }
        save_split(task_dir, "train", train_payload, split_metadata, compressed=not args.save_uncompressed)
        save_split(task_dir, "test", test_payload, split_metadata, compressed=not args.save_uncompressed)
        task_specs.append({
            "task_id": task_id,
            "task_name": task_dir.name,
            "domain_name": domain_name,
            "class_count": len(class_names),
            "train_count": int(train_payload["embeddings"].shape[0]),
            "test_count": int(test_payload["embeddings"].shape[0]),
        })
        log(f"[timing] task {domain_name} saved in {(time.time() - task_start)/60:.1f}m")

    metadata = {
        "dataset": "officehome",
        "benchmark": "officehome_sprompt_clip",
        "source_repo": str(SPROMPTS_ROOT),
        "backbone": args.backbone,
        "descriptor": args.descriptor,
        "train_split_ratio": float(args.train_split_ratio),
        "seed": int(args.seed),
        "task_order": task_order,
        "class_names": class_names,
        "task_specs": task_specs,
        "officehome_root": str(officehome_root),
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    log(f"[done] wrote Office-Home CLIP embeddings to {run_dir} total_elapsed={(time.time() - total_start)/60:.1f}m")
    return run_dir


if __name__ == "__main__":
    run_extraction(parse_args())
