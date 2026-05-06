from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

from common import INFLORA_ROOT, THIS_DIR, build_run_name, load_config, log
from extract_embeddings import run_extraction as run_inflora_extraction
from extract_embeddings_sprompt_clip import (
    build_sprompt_clip_run_name,
    run_extraction as run_sprompt_clip_extraction,
)
from routing_class import run_routing


EXPERIMENT_SPECS = {
    "cifar100_smoke": {
        "backend": "inflora",
        "config": INFLORA_ROOT / "configs" / "srt_inflora.json",
        "hypothesis_role": "artificial_class_block_smoke_test",
    },
    "domainnet_natural": {
        "backend": "inflora",
        "config": INFLORA_ROOT / "configs" / "domainnet_srt_inflora.json",
        "hypothesis_role": "natural_domain_incremental_main_test",
    },
    "officehome_sprompt_clip": {
        "backend": "sprompt_clip",
        "hypothesis_role": "primary_clip_domain_incremental_test",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the benchmark-shift hypothesis check for SRT inside the InfLoRA repo.")
    parser.add_argument(
        "--experiment",
        default="domainnet_natural",
        choices=["cifar100_smoke", "domainnet_natural", "officehome_sprompt_clip", "both"],
        help="Which benchmark setting to run. Defaults to the DomainNet main experiment; 'both' also adds the old CIFAR control.",
    )
    parser.add_argument(
        "--descriptor",
        default="cls",
        choices=["cls", "mean_patch", "cls_mean_concat"],
        help="Frozen descriptor used for offline routing.",
    )
    parser.add_argument(
        "--data_path",
        default=None,
        help="Optional data root override. For 'both', this is applied to DomainNet only.",
    )
    parser.add_argument(
        "--output_root",
        default=str(THIS_DIR / "embeddings"),
        help="Root directory for extracted embeddings.",
    )
    parser.add_argument(
        "--results_root",
        default=str(THIS_DIR / "results"),
        help="Root directory for routing reports and the combined hypothesis report.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device passed to extraction, e.g. 0, cuda:0, cpu.",
    )
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Optional batch size override shared by all runs.")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Optional num_workers override shared by all runs.")
    parser.add_argument("--use_amp", action="store_true",
                        help="Enable mixed precision during extraction (autocast on CUDA).")
    parser.add_argument("--multi_gpu", default="auto", choices=["auto", "off", "on"],
                        help="Use DataParallel across visible GPUs during extraction.")
    parser.add_argument("--fast_mode", action="store_true",
                        help="Favor speed over strict determinism during extraction.")
    parser.add_argument("--prefetch_factor", type=int, default=4,
                        help="DataLoader prefetch factor used in extraction.")
    parser.add_argument("--progress_log_every", type=int, default=20,
                        help="Emit extraction heartbeat every N batches.")
    parser.add_argument("--save_uncompressed", action="store_true",
                        help="Store npz without compression for faster writes.")
    parser.add_argument("--domainnet_verify", default="sample", choices=["none", "sample", "full"],
                        help="How strictly to verify DomainNet file paths during DataManager setup.")
    parser.add_argument("--routing_device", default="auto",
                        help="Routing compute device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--embed_dtype", default="float32", choices=["float32", "float64"],
                        help="Embedding dtype used in routing computations.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional seed override shared by all runs.")
    parser.add_argument("--max_tasks", type=int, default=None,
                        help="Only use the first N tasks for quick checks.")
    parser.add_argument("--limit_per_split", type=int, default=None,
                        help="Only extract the first N samples per split for quick checks.")
    parser.add_argument(
        "--routers",
        default="all",
        help="Comma-separated subset. Choices: nearest, cosine, online_zca, maha_ridge, maha_oas",
    )
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing routing reports and re-extract embeddings.")
    parser.add_argument(
        "--reuse_embeddings",
        action="store_true",
        help="Reuse an existing embedding directory if it is already present.",
    )
    parser.add_argument("--backbone", default="ViT-B/16",
                        help="CLIP backbone used by the S-Prompts Office-Home extractor.")
    parser.add_argument("--task_order", default="Art,Clipart,Product,Real_World",
                        help="Comma-separated Office-Home task order for the CLIP benchmark.")
    parser.add_argument("--train_split_ratio", type=float, default=0.8,
                        help="Per-class train split ratio used by the Office-Home CLIP extractor.")
    return parser.parse_args()


def selected_experiments(selection: str) -> list[str]:
    if selection == "both":
        return ["cifar100_smoke", "domainnet_natural"]
    return [selection]


def resolve_data_override(experiment_name: str, args: argparse.Namespace) -> str | None:
    if args.data_path is None:
        return None
    if experiment_name in {"domainnet_natural", "officehome_sprompt_clip"}:
        return args.data_path
    if args.experiment != "both":
        return args.data_path
    return None


def build_extract_args(experiment_name: str, args: argparse.Namespace) -> SimpleNamespace:
    spec = EXPERIMENT_SPECS[experiment_name]
    if spec["backend"] == "sprompt_clip":
        return SimpleNamespace(
            output_root=args.output_root,
            descriptor=args.descriptor,
            data_path=resolve_data_override(experiment_name, args),
            device=args.device,
            batch_size=(args.batch_size if args.batch_size is not None else 128),
            num_workers=(args.num_workers if args.num_workers is not None else 8),
            use_amp=args.use_amp,
            multi_gpu=args.multi_gpu,
            fast_mode=args.fast_mode,
            prefetch_factor=args.prefetch_factor,
            progress_log_every=args.progress_log_every,
            save_uncompressed=args.save_uncompressed,
            seed=(args.seed if args.seed is not None else 1993),
            max_tasks=args.max_tasks,
            limit_per_split=args.limit_per_split,
            force=args.force,
            backbone=args.backbone,
            task_order=args.task_order,
            train_split_ratio=args.train_split_ratio,
        )

    return SimpleNamespace(
        config=str(spec["config"]),
        output_root=args.output_root,
        descriptor=args.descriptor,
        data_path=resolve_data_override(experiment_name, args),
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_amp=args.use_amp,
        multi_gpu=args.multi_gpu,
        fast_mode=args.fast_mode,
        prefetch_factor=args.prefetch_factor,
        progress_log_every=args.progress_log_every,
        save_uncompressed=args.save_uncompressed,
        domainnet_verify=args.domainnet_verify,
        seed=args.seed,
        max_tasks=args.max_tasks,
        limit_per_split=args.limit_per_split,
        force=args.force,
    )


def build_routing_args(emb_dir: Path, args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        emb_dir=str(emb_dir),
        out_dir=args.results_root,
        max_tasks=args.max_tasks,
        routers=args.routers,
        device=args.routing_device,
        embed_dtype=args.embed_dtype,
        force=args.force,
    )


def maybe_reuse_embedding_dir(experiment_name: str, args: argparse.Namespace) -> Path | None:
    if not args.reuse_embeddings or args.force:
        return None

    spec = EXPERIMENT_SPECS[experiment_name]
    if spec["backend"] == "sprompt_clip":
        run_dir = Path(args.output_root) / build_sprompt_clip_run_name(
            args.descriptor,
            args.backbone,
            [item.strip() for item in args.task_order.split(",") if item.strip()][:args.max_tasks],
        )
    else:
        config, _ = load_config(str(spec["config"]))
        run_dir = Path(args.output_root) / build_run_name(config, args.descriptor)
    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        log(f"[reuse] using existing embeddings at {run_dir}")
        return run_dir
    return None


def load_summary(report_path: Path) -> dict:
    with open(report_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_comparison(experiment_reports: dict[str, dict]) -> dict:
    if not {"cifar100_smoke", "domainnet_natural"}.issubset(experiment_reports):
        return {}

    cifar_summary = experiment_reports["cifar100_smoke"]["summary"]
    domainnet_summary = experiment_reports["domainnet_natural"]["summary"]
    comparison = {}

    for router_name in sorted(set(cifar_summary) & set(domainnet_summary)):
        cifar_macro = float(cifar_summary[router_name]["final_macro_accuracy"])
        domainnet_macro = float(domainnet_summary[router_name]["final_macro_accuracy"])
        comparison[router_name] = {
            "cifar100_final_macro_accuracy": cifar_macro,
            "domainnet_final_macro_accuracy": domainnet_macro,
            "delta_domainnet_minus_cifar100": domainnet_macro - cifar_macro,
            "supports_natural_task_geometry_hypothesis": domainnet_macro > cifar_macro,
        }

    return comparison


def main() -> None:
    args = parse_args()
    total_start = time.time()
    experiments = selected_experiments(args.experiment)
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    experiment_reports = {}
    report_paths = {}

    for exp_idx, experiment_name in enumerate(experiments, start=1):
        exp_start = time.time()
        log(f"[stage {exp_idx}/{len(experiments)}] {experiment_name}: preparing embeddings")
        reused_dir = maybe_reuse_embedding_dir(experiment_name, args)
        if reused_dir is not None:
            emb_dir = reused_dir
        else:
            extract_args = build_extract_args(experiment_name, args)
            if EXPERIMENT_SPECS[experiment_name]["backend"] == "sprompt_clip":
                emb_dir = run_sprompt_clip_extraction(extract_args)
            else:
                emb_dir = run_inflora_extraction(extract_args)
        log(f"[stage {exp_idx}/{len(experiments)}] {experiment_name}: running routing evaluation")
        report_path = run_routing(build_routing_args(emb_dir, args))
        report_paths[experiment_name] = str(report_path)
        experiment_reports[experiment_name] = load_summary(report_path)

        exp_elapsed = time.time() - exp_start
        avg = (time.time() - total_start) / exp_idx
        remaining = avg * (len(experiments) - exp_idx)
        log(f"[timing] experiment {experiment_name} done in {exp_elapsed/60:.1f}m; est remaining {remaining/60:.1f}m")

    combined_report = {
        "hypothesis": "SRT routing should behave more cleanly on natural domain-incremental tasks than on artificial class-block tasks.",
        "descriptor": args.descriptor,
        "experiments": {
            name: {
                "config": str(EXPERIMENT_SPECS[name].get("config", "sprompt_clip_builtin")),
                "hypothesis_role": EXPERIMENT_SPECS[name]["hypothesis_role"],
                "routing_report": report_paths[name],
                "summary": experiment_reports[name]["summary"],
            }
            for name in experiments
        },
        "comparison": build_comparison(experiment_reports),
    }

    summary_name = f"hypothesis__{args.experiment}__{args.descriptor}.json"
    summary_path = results_root / summary_name
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(combined_report, handle, indent=2)

    log(f"[done] wrote hypothesis report to {summary_path} total_elapsed={(time.time() - total_start)/60:.1f}m")


if __name__ == "__main__":
    main()
