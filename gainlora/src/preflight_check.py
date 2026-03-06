#!/usr/bin/env python3
"""Fast preflight checks for GainLoRA training compatibility.

This script catches common breakages in seconds before launching long runs:
- Python syntax/import issues in critical source files
- transformers/accelerate API mismatches
- Trainer init signature mismatches (processing_class vs tokenizer)
- Known bad patterns that caused recent runtime failures
"""

import glob
import importlib
import inspect
import os
import py_compile
import sys
from typing import List


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "src")

CRITICAL_FILES = [
    "run_t5.py",
    "run_llama.py",
    "compat_transformers.py",
    "cl_trainer_gainlora_inflora.py",
    "cl_trainer_gainlora_inflora_llama.py",
    "cl_trainer_gainlora_olora.py",
    "cl_trainer_gainlora_olora_llama.py",
]

GAINLORA_TRAINERS = [
    "cl_trainer_gainlora_inflora.py",
    "cl_trainer_gainlora_inflora_llama.py",
    "cl_trainer_gainlora_olora.py",
    "cl_trainer_gainlora_olora_llama.py",
]


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _fail(msg: str, failures: List[str]) -> None:
    print(f"[FAIL] {msg}")
    failures.append(msg)


def check_syntax(failures: List[str]) -> None:
    for rel in CRITICAL_FILES:
        path = os.path.join(SRC_DIR, rel)
        if not os.path.exists(path):
            _fail(f"Missing file: {rel}", failures)
            continue
        try:
            py_compile.compile(path, doraise=True)
            _ok(f"Syntax compile: {rel}")
        except Exception as exc:
            _fail(f"Syntax error in {rel}: {exc}", failures)


def check_core_api(failures: List[str]) -> None:
    try:
        import transformers
        from transformers import Trainer

        sig = inspect.signature(Trainer.__init__)
        params = sig.parameters
        version_str = transformers.__version__
        major = int(version_str.split(".")[0])

        if major >= 5:
            if "processing_class" not in params:
                _fail("transformers>=5 requires Trainer.__init__.processing_class", failures)
            else:
                _ok("Trainer.__init__ has processing_class (transformers>=5)")
        else:
            if "tokenizer" not in params:
                _fail("transformers<5 expected Trainer.__init__.tokenizer", failures)
            else:
                _ok("Trainer.__init__ has tokenizer (transformers<5)")

        _ok(f"transformers version: {version_str}")
    except Exception as exc:
        _fail(f"transformers API check failed: {exc}", failures)

    try:
        import accelerate
        from accelerate import Accelerator

        clip_sig = inspect.signature(Accelerator.clip_grad_norm_)
        clip_params = set(clip_sig.parameters.keys())
        if "check_grad_overflow" in clip_params:
            _warn("Accelerator.clip_grad_norm_ supports check_grad_overflow")
        else:
            _ok("Accelerator.clip_grad_norm_ does not support check_grad_overflow")
        _ok(f"accelerate version: {accelerate.__version__}")
    except Exception as exc:
        _fail(f"accelerate API check failed: {exc}", failures)


def check_text_patterns(failures: List[str]) -> None:
    # Patterns that recently caused failures in this codebase.
    bad_patterns = {
        "check_grad_overflow=": "Unsupported kwarg for accelerate.clip_grad_norm_ in many envs",
    }

    for rel in GAINLORA_TRAINERS:
        path = os.path.join(SRC_DIR, rel)
        try:
            text = open(path, "r", encoding="utf-8").read()
        except Exception as exc:
            _fail(f"Cannot read {rel}: {exc}", failures)
            continue

        for pattern, reason in bad_patterns.items():
            if pattern in text:
                _fail(f"{rel} contains '{pattern}' ({reason})", failures)

        if "self.accelerator.clip_grad_norm_(" in text:
            _fail(
                f"{rel} still uses self.accelerator.clip_grad_norm_ (can trigger double unscale)",
                failures,
            )
        elif "nn.utils.clip_grad_norm_(" in text:
            _ok(f"{rel} uses nn.utils.clip_grad_norm_")

    # Check Trainer API migration in run scripts.
    for rel in ["run_t5.py", "run_llama.py"]:
        path = os.path.join(SRC_DIR, rel)
        try:
            text = open(path, "r", encoding="utf-8").read()
        except Exception as exc:
            _fail(f"Cannot read {rel}: {exc}", failures)
            continue

        if "tokenizer=tokenizer" in text:
            _fail(f"{rel} still contains tokenizer=tokenizer (should be processing_class=tokenizer)", failures)
        if "processing_class=tokenizer" in text:
            _ok(f"{rel} uses processing_class=tokenizer")


def check_custom_trainer_signatures(failures: List[str]) -> None:
    sys.path.insert(0, SRC_DIR)
    trainer_mods = sorted(glob.glob(os.path.join(SRC_DIR, "cl_trainer_*.py")))

    for mod_path in trainer_mods:
        mod_name = os.path.splitext(os.path.basename(mod_path))[0]
        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:
            _fail(f"Import failed for {mod_name}: {exc}", failures)
            continue

        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if obj.__module__ != mod_name:
                continue
            if not hasattr(obj, "__mro__"):
                continue
            if not any(base.__name__ == "Seq2SeqTrainer" for base in obj.__mro__):
                continue

            init_sig = inspect.signature(obj.__init__)
            params = init_sig.parameters
            if "processing_class" not in params:
                _fail(f"{mod_name}.{obj.__name__}.__init__ missing processing_class", failures)
            elif "tokenizer" in params:
                _fail(f"{mod_name}.{obj.__name__}.__init__ still has tokenizer", failures)
            else:
                _ok(f"{mod_name}.{obj.__name__}.__init__ signature is compatible")


def main() -> int:
    failures: List[str] = []

    print("== GainLoRA Preflight Check ==")
    print(f"Root: {ROOT}")

    check_syntax(failures)
    check_core_api(failures)
    check_text_patterns(failures)
    check_custom_trainer_signatures(failures)

    print("\n== Result ==")
    if failures:
        print(f"FAILED with {len(failures)} issue(s)")
        return 1

    print("PASS: no blocking issues found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
