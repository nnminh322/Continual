#!/usr/bin/env python3
"""Dry-import all Python modules under new_gainlora/src to surface ImportError/AttributeError

Run this in the target environment (Kaggle) to collect real import-time failures
caused by mismatched library versions (e.g., transformers v5 vs v4 helpers).

Usage:
  python new_gainlora/scripts/dry_import_transformers_check.py

Outputs a simple summary and traceback per failing file.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# Add src/ to sys.path so intra-package imports (cl_collator, rouge, etc.) work
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def try_import(path: Path) -> tuple[bool, str]:
    name = "drycheck." + ".".join(path.with_suffix("").relative_to(SRC).parts)
    print(f"\n=== Importing {path} as {name} ===")
    try:
        spec = importlib.util.spec_from_file_location(name, str(path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        # execute module (may run top-level code) — that's intended for diagnosis
        spec.loader.exec_module(module)  # type: ignore
        print("OK")
        return True, ""
    except Exception:
        tb = traceback.format_exc()
        print("FAILED:\n", tb)
        return False, tb


def main() -> int:
    if not SRC.exists():
        print("Source dir not found:", SRC)
        return 2

    py_files = sorted(SRC.rglob("*.py"))
    failures = {}
    print(f"Found {len(py_files)} python files under {SRC}")
    for p in py_files:
        ok, tb = try_import(p)
        if not ok:
            failures[str(p.relative_to(ROOT))] = tb

    print("\n\n== Summary ==")
    print("Total files:", len(py_files))
    print("Failures:", len(failures))
    if failures:
        for f, tb in failures.items():
            print("\n---", f)
            print(tb)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
