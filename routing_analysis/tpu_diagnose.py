#!/usr/bin/env python3
"""Simple runtime diagnostics for TPU/XLA and VFIO/IOMMU issues.

Run this inside your runtime to collect helpful checks that explain
why `torch_xla.device()` may fail (e.g., `/dev/vfio/0` busy).

It prints a short summary of devices, kernel modules, dmesg snippets,
torch-xla import/device diagnostics, and suggestions.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import os


def run(cmd: str) -> str:
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, timeout=10)
        return out.decode("utf-8", errors="replace").strip()
    except Exception as e:
        return f"ERROR: {e}"


def main() -> None:
    print("TPU/XLA Diagnose\n" + "=" * 60)
    print("Python:", sys.version.replace('\n', ' '))
    print("Cwd:", os.getcwd())
    print()

    print("1) /dev/vfio* nodes")
    print(run("ls -l /dev/vfio* 2>/dev/null || true"))
    print()

    print("2) kernel modules (vfio/iommu)")
    print(run("lsmod | egrep 'vfio|iommu' || true"))
    print()

    print("3) dmesg (recent; vfio/iommu/tpu/xla)")
    print(run("dmesg | tail -n 200 | egrep -i 'vfio|iommu|tpu|xla' || true"))
    print()

    print("4) torch_xla import/device diagnostics")
    try:
        spec = importlib.util.find_spec("torch_xla")
        if spec is None:
            print("  torch_xla not importable (find_spec returned None).")
        else:
            print("  torch_xla appears importable.")
            try:
                import torch_xla
                print("  torch_xla.__version__:", getattr(torch_xla, "__version__", "<unknown>"))
                try:
                    dev = torch_xla.device()
                    print("  torch_xla.device():", dev)
                except Exception as e:
                    print("  torch_xla.device() raised:", repr(e))
            except Exception as e:
                print("  Importing torch_xla raised:", repr(e))
    except Exception as e:
        print("  Unexpected error while probing torch_xla:", repr(e))
    print()

    print("5) PyTorch info")
    print(run("python -c \"import torch,sys; print(torch.__version__, torch.cuda.is_available(), getattr(torch.version,'cuda',None))\""))
    print()

    print("6) Processes / locks referencing /dev/vfio/0")
    print(run("ps aux | egrep 'vfio' || true"))
    print(run("sudo lsof /dev/vfio/0 2>/dev/null || true"))
    print()

    print("Suggestions:")
    print(" - Restart or disconnect/delete the TPU runtime (Colab: Runtime → Restart runtime or Manage sessions → terminate)")
    print(" - Ensure TPU accelerator is enabled for this session")
    print(" - If you control the VM, ensure IOMMU and VFIO are configured and kernel modules are loaded")
    print(" - If immediate progress is needed and a GPU exists, re-run the extraction with --device cuda or --allow-fallback")


if __name__ == '__main__':
    main()
