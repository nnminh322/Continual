from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn

from common import (
    SOYO_ROOT,
    THIS_DIR,
    WORKSPACE_ROOT,
    ensure_soyo_imports,
    get_task_bounds,
    log,
    resolve_torch_device,
    set_seed,
)
from embedding_utils import build_loader, extract_split_embeddings, save_split, should_enable_multi_gpu


SOYO_DATASET_ROOT_NAMES = {
    "core50": ("CORe50", "core50"),
    "cddb": ("CDDB", "cddb", "CDDB-Hard", "cddb-hard", "CDDB_Hard"),
}
CORE50_REQUIRED_FILES = ("labels.pkl", "LUP.pkl", "paths.pkl")
CDDB_DEFAULT_TASK_NAMES = ("gaugan", "biggan", "wild", "whichfaceisreal", "san")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frozen SOYO CLIP embeddings for zero-rehearsal SRT routing checks.")
    parser.add_argument(
        "--config",
        default=str(SOYO_ROOT / "configs" / "core50_soyo_clip.yaml"),
        help="Path to a SOYO YAML config, e.g. configs/core50_soyo_clip.yaml",
    )
    parser.add_argument("--output_root", default=str(THIS_DIR / "embeddings"),
                        help="Root directory for saved embedding shards")
    parser.add_argument("--descriptor", default="cls", choices=["cls"],
                        help="SOYO exposes pooled CLIP embeddings only, so descriptor=cls is required")
    parser.add_argument("--data_path", default=None,
                        help="Optional dataset root override. Can point to the dataset root or to a parent directory containing CORe50/CDDB.")
    parser.add_argument("--device", default=None,
                        help="Torch device. Examples: 0, cuda:0, cpu. Defaults to cuda:0 when available.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Optional batch size override. Defaults to the SOYO config value.")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Optional DataLoader worker override. Defaults to the SOYO config value.")
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
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional seed override. Defaults to the first seed in the SOYO config.")
    parser.add_argument("--max_tasks", type=int, default=None,
                        help="Only extract the first N continual tasks")
    parser.add_argument("--limit_per_split", type=int, default=None,
                        help="Debug option: cap saved samples per train/test split")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing run directory")
    return parser.parse_args()


def sanitize_name(raw: str) -> str:
    return raw.lower().replace("/", "-").replace(" ", "").replace("_", "")


def build_soyo_clip_run_name(config: dict, descriptor: str) -> str:
    dataset = str(config["dataset"]).lower()
    init_cls = int(config["init_cls"])
    increment = int(config["increment"])
    return f"{dataset}__soyoclip__init{init_cls}__inc{increment}__{descriptor}"


def load_soyo_config(config_path: str) -> tuple[dict, Path]:
    import yaml

    raw_path = Path(config_path)
    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend([
            SOYO_ROOT / raw_path,
            WORKSPACE_ROOT / raw_path,
            Path.cwd() / raw_path,
        ])

    resolved = None
    for candidate in candidates:
        if candidate.exists():
            resolved = candidate.resolve()
            break

    if resolved is None:
        resolved = (Path.cwd() / raw_path).resolve()

    with open(resolved, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return config, resolved


def _expand_root_candidate(base: Path, root_names: tuple[str, ...]):
    yield base
    for root_name in root_names:
        yield base / root_name

    if not base.exists() or not base.is_dir():
        return

    try:
        children = [child for child in base.iterdir() if child.is_dir()]
    except OSError:
        children = []

    for child in children:
        yield child
        for root_name in root_names:
            yield child / root_name


def _resolve_user_path_variants(raw_path: str) -> list[Path]:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return [path]

    return [
        WORKSPACE_ROOT / path,
        Path.cwd() / path,
        path,
    ]


def _iter_dataset_root_candidates(dataset: str, data_path: str | None, config_data_path: str | None):
    root_names = SOYO_DATASET_ROOT_NAMES[dataset]
    bases = []

    if data_path:
        bases.extend(_resolve_user_path_variants(data_path))

    if config_data_path:
        config_root = Path(config_data_path).expanduser()
        if config_root.is_absolute():
            bases.append(config_root)
        else:
            bases.extend([
                SOYO_ROOT / config_root,
                WORKSPACE_ROOT / config_root,
                Path.cwd() / config_root,
            ])

    bases.extend([
        SOYO_ROOT / "dil_dataset",
        SOYO_ROOT.parent / "data",
        SOYO_ROOT.parent.parent / "data",
        WORKSPACE_ROOT,
        Path.cwd(),
        Path("/kaggle/input"),
        Path("/content"),
    ])

    seen = set()
    for base in bases:
        for candidate in _expand_root_candidate(base, root_names):
            candidate = candidate.expanduser()
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            yield candidate


def _is_core50_root(candidate: Path) -> bool:
    if not candidate.is_dir():
        return False
    if not all((candidate / name).exists() for name in CORE50_REQUIRED_FILES):
        return False
    return (candidate / "core50_imgs.npz").exists() or (candidate / "core50_imgs.bin").exists()


def _is_cddb_root(candidate: Path, task_names: list[str]) -> bool:
    if not candidate.is_dir():
        return False
    return all((candidate / task_name / "train").is_dir() and (candidate / task_name / "val").is_dir()
               for task_name in task_names)


def resolve_soyo_dataset_root(
    dataset: str,
    data_path: str | None,
    config_data_path: str | None,
    task_names: list[str] | tuple[str, ...] | None,
) -> Path:
    dataset = dataset.lower()
    checked = []
    resolved_task_names = [str(name) for name in (task_names or CDDB_DEFAULT_TASK_NAMES)]

    for candidate in _iter_dataset_root_candidates(dataset, data_path, config_data_path):
        checked.append(str(candidate))
        if dataset == "core50" and _is_core50_root(candidate):
            return candidate.resolve()
        if dataset == "cddb" and _is_cddb_root(candidate, resolved_task_names):
            return candidate.resolve()

    example_root = "CORe50" if dataset == "core50" else "CDDB"
    raise FileNotFoundError(
        "Could not locate {} for SOYO. Checked: {}. Pass --data_path to the dataset root itself or to a parent "
        "directory containing {}.".format(
            dataset,
            ", ".join(checked[:20]) + (" ..." if len(checked) > 20 else ""),
            example_root,
        ))


def resolve_seed(seed_value) -> int:
    if isinstance(seed_value, list):
        if not seed_value:
            raise ValueError("SOYO config seed list is empty.")
        return int(seed_value[0])
    return int(seed_value)


def describe_task(config: dict, task_id: int, start_class: int, end_class: int) -> tuple[str, str | None]:
    dataset = str(config["dataset"]).lower()
    if dataset == "cddb":
        task_names = [str(name) for name in config.get("task_name", CDDB_DEFAULT_TASK_NAMES)]
        if task_id < len(task_names):
            domain_name = task_names[task_id]
            return f"task_{task_id:02d}_{sanitize_name(domain_name)}", domain_name
    return f"task_{task_id:02d}_c{start_class:03d}-{end_class - 1:03d}", None


class FrozenSoyoClipExtractor(nn.Module):
    def __init__(self, visual: nn.Module, dtype: torch.dtype):
        super().__init__()
        self.visual = visual
        self.dtype = dtype

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.visual(image.type(self.dtype))
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
        return features.unsqueeze(1)


def run_extraction(args: argparse.Namespace) -> Path:
    if args.descriptor != "cls":
        raise ValueError("SOYO CLIP extraction only supports descriptor=cls because SOYO exposes pooled image embeddings.")

    ensure_soyo_imports()

    DataManager = importlib.import_module("utils.data_manager").DataManager
    prompt_learner = importlib.import_module("models.clip.prompt_learner")
    cfgc = prompt_learner.cfgc
    load_clip_to_cpu = prompt_learner.load_clip_to_cpu

    config, config_path = load_soyo_config(args.config)
    if str(config.get("net_type", "")).lower() != "soyo_clip":
        raise ValueError(f"Expected a soyo_clip config, got net_type={config.get('net_type')!r} from {config_path}")

    seed = int(args.seed if args.seed is not None else resolve_seed(config.get("seed", 0)))
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

    data_args = dict(config)
    data_args["seed"] = seed
    resolved_data_path = resolve_soyo_dataset_root(
        str(data_args["dataset"]),
        args.data_path,
        data_args.get("data_path"),
        data_args.get("task_name"),
    )
    data_args["data_path"] = str(resolved_data_path)

    log(
        f"[setup] starting SOYO extraction config={config_path} dataset={data_args['dataset']} "
        f"seed={seed} device={device} batch_size={batch_size} num_workers={num_workers}")

    dm_t0 = time.time()
    data_manager = DataManager(
        data_args["dataset"],
        data_args["shuffle"],
        seed,
        data_args["init_cls"],
        data_args["increment"],
        data_args,
    )
    data_args["class_order"] = data_manager._class_order
    log(f"[setup] DataManager ready in {time.time() - dm_t0:.1f}s with nb_tasks={data_manager.nb_tasks}")

    clip_cfg = cfgc()
    clip_model = load_clip_to_cpu(clip_cfg)
    feature_model: nn.Module = FrozenSoyoClipExtractor(clip_model.visual, clip_model.dtype)
    for parameter in feature_model.parameters():
        parameter.requires_grad_(False)
    feature_model = feature_model.eval().to(device)

    multi_gpu_active = should_enable_multi_gpu(args.multi_gpu, device)
    if multi_gpu_active:
        device_ids = list(range(torch.cuda.device_count()))
        feature_model = nn.DataParallel(feature_model, device_ids=device_ids)
        log(f"[setup] multi_gpu=on DataParallel device_ids={device_ids} primary={device_ids[0]}")
    else:
        log(f"[setup] multi_gpu=off mode={args.multi_gpu} visible_gpus={torch.cuda.device_count()}")

    output_root = Path(args.output_root)
    run_dir = output_root / build_soyo_clip_run_name(data_args, args.descriptor)
    if run_dir.exists():
        if not args.force:
            raise FileExistsError(f"{run_dir} already exists. Use --force to overwrite.")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    max_tasks = data_manager.nb_tasks if args.max_tasks is None else min(int(args.max_tasks), data_manager.nb_tasks)
    log(f"[setup] benchmark={data_args['dataset']}_soyo_clip root={resolved_data_path}")
    log(f"[setup] backbone={clip_cfg.backbonename} descriptor={args.descriptor} max_tasks={max_tasks}")

    task_specs = []
    embedding_dim = 0
    total_start = time.time()

    for task_id in range(max_tasks):
        start_class, end_class = get_task_bounds(data_manager._increments, task_id)
        task_name, domain_name = describe_task(data_args, task_id, start_class, end_class)
        task_dir = run_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        class_indices = torch.arange(start_class, end_class).cpu().numpy()
        train_dataset = data_manager.get_dataset(class_indices, source="train", mode="train")
        test_dataset = data_manager.get_dataset(class_indices, source="test", mode="test")

        if len(train_dataset) == 0 or len(test_dataset) == 0:
            raise RuntimeError(
                f"Empty split for {task_name}: train={len(train_dataset)} test={len(test_dataset)}. "
                "This usually means the dataset root is wrong or the continual label remap does not match SOYO's expected layout.")

        train_loader = build_loader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed + task_id * 17,
            prefetch_factor=int(args.prefetch_factor),
        )
        test_loader = build_loader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed + task_id * 17 + 1,
            prefetch_factor=int(args.prefetch_factor),
        )

        log(
            f"[task {task_id + 1}/{max_tasks}] {task_name} classes={start_class}-{end_class - 1} "
            f"train={len(train_dataset)} test={len(test_dataset)}")
        train_payload = extract_split_embeddings(
            train_loader,
            feature_model,
            args.descriptor,
            device,
            args.limit_per_split,
            desc=f"{task_name} train",
            use_amp=bool(args.use_amp),
            progress_log_every=max(1, int(args.progress_log_every)),
        )
        test_payload = extract_split_embeddings(
            test_loader,
            feature_model,
            args.descriptor,
            device,
            args.limit_per_split,
            desc=f"{task_name} test",
            use_amp=bool(args.use_amp),
            progress_log_every=max(1, int(args.progress_log_every)),
        )
        embedding_dim = int(train_payload["embeddings"].shape[1])

        split_metadata = {
            "dataset": data_args["dataset"],
            "task_id": task_id,
            "task_name": task_name,
            "start_class": start_class,
            "end_class_exclusive": end_class,
            "descriptor": args.descriptor,
            "backbone": clip_cfg.backbonename,
            "seed": seed,
            "config_path": str(config_path),
            "class_order": [int(x) for x in data_manager._class_order],
            "train_protocol": "source=train,mode=train,shuffle=False,zero-rehearsal,soyo_datamanager",
            "test_protocol": "source=test,mode=test,eval-over-current-task-only,soyo_datamanager",
            "limit_per_split": args.limit_per_split,
        }
        if domain_name is not None:
            split_metadata["domain_name"] = domain_name

        use_compression = not bool(args.save_uncompressed)
        save_split(task_dir, "train", train_payload, {**split_metadata, "split": "train"}, compressed=use_compression)
        save_split(task_dir, "test", test_payload, {**split_metadata, "split": "test"}, compressed=use_compression)

        task_spec = {
            "task_id": task_id,
            "task_name": task_name,
            "start_class": start_class,
            "end_class_exclusive": end_class,
            "train_count": int(train_payload["embeddings"].shape[0]),
            "test_count": int(test_payload["embeddings"].shape[0]),
        }
        if domain_name is not None:
            task_spec["domain_name"] = domain_name
        task_specs.append(task_spec)

    metadata = {
        "dataset": data_args["dataset"],
        "benchmark": f"{str(data_args['dataset']).lower()}_soyo_clip",
        "source_repo": str(SOYO_ROOT),
        "config_path": str(config_path),
        "data_path": str(resolved_data_path),
        "descriptor": args.descriptor,
        "backbone": clip_cfg.backbonename,
        "seed": seed,
        "device": device,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "init_cls": int(data_args["init_cls"]),
        "increment": int(data_args["increment"]),
        "increments": [int(x) for x in data_manager._increments],
        "class_order": [int(x) for x in data_manager._class_order],
        "task_specs": task_specs,
        "embedding_dim": embedding_dim,
        "zero_rehearsal_protocol": True,
        "notes": [
            "Backbone is SOYO's frozen CLIP visual encoder only; prompt pools and the SOYO selector are intentionally bypassed.",
            "Train/test task streams come directly from SOYO's DataManager for the chosen benchmark.",
            "SOYO CLIP extraction currently supports descriptor=cls only because SOYO exposes pooled image embeddings.",
        ],
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    log(f"[done] wrote SOYO CLIP embeddings to {run_dir} total_elapsed={(time.time() - total_start)/60:.1f}m")
    return run_dir


if __name__ == "__main__":
    run_extraction(parse_args())
