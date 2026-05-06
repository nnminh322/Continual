from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from common import INFLORA_ROOT, THIS_DIR, build_run_name, load_config
from extract_embeddings import run_extraction
from routing_class import run_routing


EXPERIMENT_SPECS = {
    "cifar100_smoke": {
        "config": INFLORA_ROOT / "configs" / "srt_inflora.json",
        "hypothesis_role": "artificial_class_block_smoke_test",
    },
    "domainnet_natural": {
        "config": INFLORA_ROOT / "configs" / "domainnet_srt_inflora.json",
        "hypothesis_role": "natural_domain_incremental_main_test",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the benchmark-shift hypothesis check for SRT inside the InfLoRA repo.")
    parser.add_argument(
        "--experiment",
        default="domainnet_natural",
        choices=["cifar100_smoke", "domainnet_natural", "both"],
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
    return parser.parse_args()


def selected_experiments(selection: str) -> list[str]:
    if selection == "both":
        return ["cifar100_smoke", "domainnet_natural"]
    return [selection]


def resolve_data_override(experiment_name: str, args: argparse.Namespace) -> str | None:
    if args.data_path is None:
        return None
    if experiment_name == "domainnet_natural":
        return args.data_path
    if args.experiment != "both":
        return args.data_path
    return None


def build_extract_args(experiment_name: str, args: argparse.Namespace) -> SimpleNamespace:
    spec = EXPERIMENT_SPECS[experiment_name]
    return SimpleNamespace(
        config=str(spec["config"]),
        output_root=args.output_root,
        descriptor=args.descriptor,
        data_path=resolve_data_override(experiment_name, args),
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
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
        force=args.force,
    )


def maybe_reuse_embedding_dir(experiment_name: str, args: argparse.Namespace) -> Path | None:
    if not args.reuse_embeddings or args.force:
        return None

    spec = EXPERIMENT_SPECS[experiment_name]
    config, _ = load_config(str(spec["config"]))
    run_dir = Path(args.output_root) / build_run_name(config, args.descriptor)
    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        print(f"[reuse] using existing embeddings at {run_dir}")
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
    experiments = selected_experiments(args.experiment)
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    experiment_reports = {}
    report_paths = {}

    for experiment_name in experiments:
        print(f"[stage] {experiment_name}: preparing embeddings", flush=True)
        reused_dir = maybe_reuse_embedding_dir(experiment_name, args)
        emb_dir = reused_dir if reused_dir is not None else run_extraction(build_extract_args(experiment_name, args))
        print(f"[stage] {experiment_name}: running routing evaluation", flush=True)
        report_path = run_routing(build_routing_args(emb_dir, args))
        report_paths[experiment_name] = str(report_path)
        experiment_reports[experiment_name] = load_summary(report_path)

    combined_report = {
        "hypothesis": "SRT routing should behave more cleanly on natural domain-incremental tasks than on artificial class-block tasks.",
        "descriptor": args.descriptor,
        "experiments": {
            name: {
                "config": str(EXPERIMENT_SPECS[name]["config"]),
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

    print(f"[done] wrote hypothesis report to {summary_path}")


if __name__ == "__main__":
    main()
