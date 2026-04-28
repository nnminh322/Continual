#!/usr/bin/env python3
"""
Backfill metadata_json into legacy embedding .npz files without re-running inference.

This upgrades old archives created before extract_embeddings_llama.py started
writing extraction metadata. It does not recompute embeddings; it only rewrites
the .npz file with an inferred metadata_json field.

Examples:
    python routing_analysis/backfill_embedding_metadata.py \
      --emb_dir /kaggle/working/embeddings/Llama-2-7b-hf \
      --benchmark SuperNI
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def infer_profile(emb_dir: str, benchmark: str, npz_path: Path):
    model_dir = Path(emb_dir).name
    layer = "embedding" if model_dir.endswith("_wordemb") else "hidden"
    pool = "avg" if "_poolavg" in model_dir else "last"

    benchmark_dir = npz_path.parent.parent.name
    task_name = npz_path.parent.name
    split = npz_path.stem

    profile = {
        "benchmark": benchmark_dir,
        "task_name": task_name,
        "split": split,
        "layer": layer,
        "pool": pool,
        "metadata_backfilled": True,
        "metadata_source": "legacy_dir_inference",
    }

    if benchmark == "SuperNI":
        profile.update({
            "superni_prompt_style": "runtime_cl",
            "add_special_tokens": False,
            "max_length": 1024,
            "padding_side": "left",
            "runtime_aligned": layer == "hidden" and pool == "last",
        })
    else:
        profile.update({
            "superni_prompt_style": None,
            "add_special_tokens": False,
            "max_length": 1024,
            "padding_side": "left",
            "runtime_aligned": False,
        })

    return profile


def iter_npz_files(emb_dir: str, benchmark: str):
    root = Path(emb_dir) / benchmark
    if not root.exists():
        return []
    return sorted(root.glob("*/*.npz"))


def rewrite_npz(npz_path: Path, metadata: dict, overwrite: bool):
    with np.load(str(npz_path), allow_pickle=True) as data:
        if "metadata_json" in data.files and not overwrite:
            return "skipped"
        payload = {key: data[key] for key in data.files if overwrite or key != "metadata_json"}

    payload["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))

    tmp_path = npz_path.with_suffix(".tmp.npz")
    np.savez_compressed(str(tmp_path), **payload)
    os.replace(tmp_path, npz_path)
    return "updated"


def main():
    parser = argparse.ArgumentParser(description="Backfill metadata_json into legacy embedding .npz files")
    parser.add_argument("--emb_dir", required=True, help="Embedding root, e.g. /kaggle/working/embeddings/Llama-2-7b-hf")
    parser.add_argument("--benchmark", nargs="+", default=["SuperNI"], choices=["SuperNI", "Long_Sequence"],
                        help="Benchmarks to patch in place")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite metadata_json even if already present")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be patched without rewriting files")
    args = parser.parse_args()

    total = 0
    updated = 0
    skipped = 0

    for benchmark in args.benchmark:
        npz_files = iter_npz_files(args.emb_dir, benchmark)
        print(f"[{benchmark}] found {len(npz_files)} npz files")

        for npz_path in npz_files:
            total += 1
            metadata = infer_profile(args.emb_dir, benchmark, npz_path)
            if args.dry_run:
                print(f"  [DRY] {npz_path} -> {json.dumps(metadata, sort_keys=True)}")
                continue

            status = rewrite_npz(npz_path, metadata, overwrite=args.overwrite)
            if status == "updated":
                updated += 1
            else:
                skipped += 1

        if not npz_files:
            print(f"  [WARN] no files found under {Path(args.emb_dir) / benchmark}")

    if args.dry_run:
        print(f"\nDry run complete. Examined {total} files.")
    else:
        print(f"\nDone. Updated {updated} files, skipped {skipped} files.")


if __name__ == "__main__":
    main()