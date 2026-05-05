from __future__ import annotations

import json
import os
import random
import sys
import types
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None


THIS_DIR = Path(__file__).resolve().parent
EXPAND_CV_ROOT = THIS_DIR.parent
INFLORA_ROOT = EXPAND_CV_ROOT / "InfLoRA"
WORKSPACE_ROOT = EXPAND_CV_ROOT.parent


def _noop(*_args, **_kwargs):
    return None


def ensure_inflora_imports() -> None:
    if "ipdb" not in sys.modules:
        try:
            import ipdb  # noqa: F401
        except ImportError:
            stub = types.ModuleType("ipdb")
            stub.set_trace = _noop
            sys.modules["ipdb"] = stub

    inflora_path = str(INFLORA_ROOT)
    if inflora_path not in sys.path:
        sys.path.insert(0, inflora_path)

    os.chdir(INFLORA_ROOT)


def load_config(config_path: str) -> tuple[dict, Path]:
    raw_path = Path(config_path)
    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend([
            INFLORA_ROOT / raw_path,
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
        config = json.load(handle)
    return config, resolved


def resolve_torch_device(device_arg: str | None) -> str:
    if torch is None:
        return "cpu"
    if device_arg is None or device_arg == "":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device_arg == "cpu":
        return "cpu"
    if device_arg.startswith("cuda"):
        return device_arg
    return f"cuda:{device_arg}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    if torch is None:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_task_bounds(increments: list[int], task_id: int) -> tuple[int, int]:
    start = int(sum(increments[:task_id]))
    end = start + int(increments[task_id])
    return start, end


def get_task_name(task_id: int, start_class: int, end_class: int) -> str:
    return f"task_{task_id:02d}_c{start_class:03d}-{end_class - 1:03d}"


def build_run_name(config: dict, descriptor: str) -> str:
    dataset = config["dataset"]
    init_cls = int(config["init_cls"])
    increment = int(config["increment"])
    return f"{dataset}__init{init_cls}__inc{increment}__{descriptor}"
