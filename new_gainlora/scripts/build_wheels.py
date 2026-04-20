#!/usr/bin/env python3
"""Download an offline wheel bundle for new_gainlora without installing anything.

This script is notebook-friendly: copy the full file into a single Python cell on
Kaggle, or run it as a normal Python script.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


OUT_DIR = Path(os.environ.get("GAINLORA_WHEEL_DIR", "/kaggle/working/gainlora_wheels"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_PYTHON = os.environ.get("GAINLORA_TARGET_PYTHON", "3.12")
TARGET_ABI = os.environ.get("GAINLORA_TARGET_ABI", "cp312")

PIP = [sys.executable, "-m", "pip"]
TORCH_PACKAGES = [
    "torch",
    "torchvision",
    "torchaudio",
]
TORCH_INDEXES = [
    ("cu128", "https://download.pytorch.org/whl/nightly/cu128"),
    ("cu130", "https://download.pytorch.org/whl/nightly/cu130"),
    ("cpu", "https://download.pytorch.org/whl/nightly/cpu"),
]

# These versions follow setup_server.sh, which targets Python 3.12 and the
# newer Transformers/datasets stack used in the current migration work.
CORE_PACKAGES = [
    "numpy==2.2.2",
    "protobuf==5.29.3",
    "pyarrow==17.0.0",
    "fsspec==2024.6.1",
    "tqdm==4.67.1",
    "pynvml==11.5.3",
    "pandas==2.2.2",
    "scipy==1.14.1",
    "scikit-learn==1.6.1",
    "loralib==0.1.2",
    "sentencepiece==0.2.0",
    "ipdb==0.13.13",
    "nltk==3.9.1",
    "absl-py==2.0.0",
    "transformers>=5.0.0",
    "tokenizers>=0.21.0",
    "accelerate>=1.3.0",
    "datasets>=3.0.0",
]

# Optional packages are downloaded one-by-one so a missing wheel does not abort
# the full bundle generation.
OPTIONAL_PACKAGES = [
    "deepspeed==0.11.2",
    "flash-attn==2.5.8",
    "bitsandbytes",
]


def run(cmd: list[str], allow_fail: bool = False) -> bool:
    print("\n>>>", shlex.join(cmd))
    completed = subprocess.run(cmd, check=False)
    ok = completed.returncode == 0
    print("returncode:", completed.returncode)
    if not ok and not allow_fail:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {shlex.join(cmd)}")
    return ok


def pip_download(packages: list[str], extra_args: list[str] | None = None, allow_fail: bool = False) -> bool:
    cmd = PIP + ["download", "--dest", str(OUT_DIR), "--progress-bar", "off"]
    cmd.extend([
        "--python-version",
        TARGET_PYTHON,
        "--implementation",
        "cp",
        "--abi",
        TARGET_ABI,
    ])
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(packages)
    return run(cmd, allow_fail=allow_fail)


def download_torch() -> str:
    for label, index_url in TORCH_INDEXES:
        ok = pip_download(
            TORCH_PACKAGES,
            extra_args=["--index-url", index_url, "--pre", "--only-binary=:all:"],
            allow_fail=True,
        )
        if ok:
            return label
    raise RuntimeError("Could not download torch/torchvision/torchaudio wheels from any configured index.")


def download_core_packages() -> list[str]:
    print("\n==> Downloading core wheels")
    ok = pip_download(CORE_PACKAGES, extra_args=["--only-binary=:all:"], allow_fail=True)
    if ok:
        return []

    print("Core batch download failed. Retrying one package at a time to isolate missing wheels...")
    missing = []
    for package in CORE_PACKAGES:
        if not pip_download([package], extra_args=["--only-binary=:all:"], allow_fail=True):
            missing.append(package)
    return missing


def download_optional_packages() -> list[str]:
    print("\n==> Downloading optional wheels")
    missing = []

    if not pip_download(["cupy-cuda12x"], extra_args=["--only-binary=:all:"], allow_fail=True):
        print("cupy-cuda12x unavailable for cp312, trying generic cupy...")
        if not pip_download(["cupy"], extra_args=["--only-binary=:all:"], allow_fail=True):
            missing.append("cupy-cuda12x")
            missing.append("cupy")

    for package in OPTIONAL_PACKAGES:
        if not pip_download([package], extra_args=["--only-binary=:all:"], allow_fail=True):
            missing.append(package)
    return missing


def main() -> None:
    print(f"Python executable: {sys.executable}")
    print(f"Python version   : {sys.version.split()[0]}")
    print(f"Target python    : {TARGET_PYTHON} ({TARGET_ABI})")
    print(f"Wheel output dir : {OUT_DIR}")

    torch_variant = download_torch()
    missing_core = download_core_packages()
    missing_optional = download_optional_packages()

    files = sorted(path.name for path in OUT_DIR.iterdir())
    manifest = {
        "python": sys.version.split()[0],
        "target_python": TARGET_PYTHON,
        "target_abi": TARGET_ABI,
        "output_dir": str(OUT_DIR),
        "torch_variant": torch_variant,
        "missing_core": missing_core,
        "missing_optional": missing_optional,
        "file_count": len(files),
        "files": files,
    }
    manifest_path = OUT_DIR / "wheel_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n==> Summary")
    print("Torch index used:", torch_variant)
    print("Core packages missing wheels:", missing_core or "none")
    print("Optional packages missing wheels:", missing_optional or "none")
    print("Total downloaded files:", len(files))
    print("Manifest:", manifest_path)
    for name in files:
        print(" ", name)

    print(
        "\nOffline install example:\n"
        f"{sys.executable} -m pip install --no-index --find-links {OUT_DIR} "
        "torch torchvision torchaudio transformers datasets accelerate"
    )


if __name__ == "__main__":
    main()
