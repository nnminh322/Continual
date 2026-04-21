#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"Expected exactly one occurrence of {label!r}, found {count}")
    return text.replace(old, new)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an order-1 LLaMA script with a different epoch count.")
    parser.add_argument("--epochs", type=int, required=True, help="Replacement value for --num_train_epochs.")
    parser.add_argument(
        "--source-script",
        type=Path,
        default=None,
        help="Optional path to the source shell script. Defaults to new_gainlora/gen_script_superni_order1_llama2_srt.sh",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to place the generated script. Defaults to llama_epoch_ablation/generated_scripts",
    )
    args = parser.parse_args()

    ablation_dir = Path(__file__).resolve().parent
    workspace_dir = ablation_dir.parent
    source_script = args.source_script or workspace_dir / "new_gainlora" / "gen_script_superni_order1_llama2_srt.sh"
    output_dir = args.output_dir or ablation_dir / "generated_scripts"
    output_dir.mkdir(parents=True, exist_ok=True)

    text = source_script.read_text(encoding="utf-8")

    text = replace_once(text, "#SBATCH -J srt-llama2", f"#SBATCH -J srt-llama2-e{args.epochs}", "SBATCH job name")
    text = replace_once(text, "#SBATCH -o srt-llama2-%j.out", f"#SBATCH -o srt-llama2-e{args.epochs}-%j.out", "SBATCH log path")
    text = replace_once(
        text,
        'RUN_NAME="superni_order1_llama2_srt"',
        f'RUN_NAME="superni_order1_llama2_srt_epochs{args.epochs}"',
        "RUN_NAME",
    )
    text = replace_once(
        text,
        'BASE_OUT="logs_and_outputs/$RUN_NAME"',
        'BASE_OUT="../llama_epoch_ablation/logs_and_outputs/$RUN_NAME"',
        "BASE_OUT",
    )

    epoch_old = "       --num_train_epochs 100 \\\n"
    epoch_new = f"       --num_train_epochs {args.epochs} \\\n"
    epoch_count = text.count(epoch_old)
    if epoch_count == 0:
        raise ValueError("No '--num_train_epochs 100' blocks were found in the source script")
    text = text.replace(epoch_old, epoch_new)

    generated_notice = (
        "# GENERATED FILE. DO NOT EDIT.\n"
        f"# Source: {source_script.relative_to(workspace_dir)}\n"
        f"# Changes: num_train_epochs={args.epochs}, RUN_NAME redirected, outputs redirected to llama_epoch_ablation.\n"
    )
    shebang = "#!/bin/bash\n"
    if not text.startswith(shebang):
        raise ValueError("Unexpected source script format: missing #!/bin/bash shebang")
    text = shebang + generated_notice + text[len(shebang):]

    output_path = output_dir / f"gen_script_superni_order1_llama2_srt_epochs{args.epochs}.sh"
    output_path.write_text(text, encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
