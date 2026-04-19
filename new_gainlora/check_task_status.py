#!/usr/bin/env python3
"""
Smart Task Status Checker for Continual Learning Pipeline.
Auto-detects each task's state: train_done, eval_done, checkpoint_available, etc.
"""

import os
import json
import sys
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class TaskState:
    task_id: int
    task_name: str
    output_dir: str

    # Checkpoint detection
    has_checkpoint: bool = False
    latest_checkpoint: str = ""
    checkpoint_count: int = 0

    # Saved weights (LoRA, prompts)
    has_lora_weights: bool = False
    has_prompt_keys: bool = False

    # SRT signatures
    has_srt_signatures: bool = False

    # Training done (HuggingFace trainer_state.json)
    train_done: bool = False
    train_steps: int = 0
    train_epochs: float = 0.0

    # Eval done (all_results.json exists)
    eval_done: bool = False
    eval_results: dict = field(default_factory=dict)

    # Predict done (all_results.json populated with predict_* keys)
    predict_done: bool = False

    # Overall phase (UNKNOWN / TRAINING / EVALUATING / TRAINED / EVALUATED / INFERRED / KILLED)
    phase: str = "UNKNOWN"

    # Warnings
    warnings: list = field(default_factory=list)

    def summarize(self) -> str:
        parts = []
        if self.train_done:
            parts.append(f"train✓({self.train_epochs:.1f}ep)")
        else:
            parts.append(f"train✗({self.train_steps}st)")

        if self.predict_done:
            parts.append("predict✓")
        elif self.eval_done:
            parts.append("eval✓")
        else:
            parts.append("eval✗")

        if self.has_checkpoint:
            parts.append(f"ckpt({self.checkpoint_count})")
        if self.has_srt_signatures:
            parts.append("srt✓")
        if self.warnings:
            parts.append(f"⚠{len(self.warnings)}")

        return " | ".join(parts) if parts else "???"


def check_dir(base: str, run_name: str):
    """Find the output directory for a run."""
    candidates = [
        os.path.join(base, run_name),
        os.path.join(base, run_name, "outputs"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


def get_task_order(output_dir: str) -> list:
    """Read task order from task_order.txt."""
    f = os.path.join(output_dir, "task_order.txt")
    if os.path.exists(f):
        with open(f) as fh:
            return fh.read().strip().split(',')
    return []


def detect_checkpoints(task_dir: str) -> tuple:
    """Return (has_ckpt, latest_ckpt, count)."""
    if not os.path.isdir(task_dir):
        return False, "", 0
    subdirs = [d for d in os.listdir(task_dir) if d.startswith("checkpoint-")]
    if not subdirs:
        return False, "", 0
    # Sort by modification time (latest first)
    subdirs.sort(key=lambda d: os.path.getmtime(os.path.join(task_dir, d)), reverse=True)
    return True, subdirs[0], len(subdirs)


def check_train_done(task_dir: str) -> tuple:
    """Check trainer_state.json to see if training finished."""
    trainer_state_path = os.path.join(task_dir, "trainer_state.json")
    if not os.path.exists(trainer_state_path):
        return False, 0, 0.0

    try:
        with open(trainer_state_path) as f:
            state = json.load(f)
    except (json.JSONDecodeError, IOError):
        return False, 0, 0.0

    # Check if training actually completed
    # HuggingFace trainer: epoch is a dict with 'epoch' key, or a float
    epoch_info = state.get("epoch", 0)
    if isinstance(epoch_info, dict):
        epoch = epoch_info.get("epoch", 0.0)
    else:
        epoch = float(epoch_info) if epoch_info is not None else 0.0

    max_steps = state.get("max_steps", 0)
    log_history = state.get("log_history", [])

    # Best metric saved?
    best_metric = state.get("best_model_metric", None)
    last_step = state.get("last_step", 0)

    # Detect killed: has steps logged but no final epoch recorded
    if epoch == 0.0 and last_step > 0 and best_metric is None:
        return False, last_step, epoch

    # Normal: epoch > 0 or best_metric saved
    return True, last_step, epoch


def check_eval_predict_done(task_dir: str, task_name: str, run_name: str) -> tuple:
    """Check all_results.json for eval/predict status."""
    all_results_path = os.path.join(task_dir, "all_results.json")
    if not os.path.exists(all_results_path):
        return False, False, {}

    try:
        with open(all_results_path) as f:
            results = json.load(f)
    except (json.JSONDecodeError, IOError):
        return False, False, {}

    if not results:
        return False, False, {}

    # Predict done = has predict_* keys
    predict_keys = [k for k in results if k.startswith("predict_")]
    predict_done = len(predict_keys) > 0

    # Eval done = has eval_* keys
    eval_keys = [k for k in results if k.startswith("eval_")]
    eval_done = len(eval_keys) > 0

    return eval_done, predict_done, results


def check_saved_weights(task_dir: str) -> tuple:
    """Check if LoRA weights and prompt keys exist."""
    saved = os.path.join(task_dir, "saved_weights")
    has_lora = False
    has_prompts = False

    if os.path.isdir(saved):
        lora_a = os.path.join(saved, "lora_weights_A.pt")
        lora_b = os.path.join(saved, "lora_weights_B.pt")
        has_lora = os.path.exists(lora_a) and os.path.exists(lora_b)

        prompt_keys = os.path.join(saved, "prompts_keys_till_now.pt")
        has_prompts = os.path.exists(prompt_keys)

    return has_lora, has_prompts


def check_srt_signatures(task_dir: str) -> bool:
    """Check if SRT signatures exist."""
    saved = os.path.join(task_dir, "saved_weights")
    srt_path = os.path.join(saved, "srt_signatures.npz")
    return os.path.exists(srt_path)


def infer_phase(state: TaskState) -> str:
    """Infer the phase of this task based on its state."""
    if state.predict_done:
        return "INFERRED"
    if state.eval_done and state.train_done:
        return "EVALUATED"
    if state.train_done:
        # Train done but eval not done
        if state.eval_done is False:
            # Check if it's still running or was killed
            if state.warnings and any("KILLED" in w or "killed" in w for w in state.warnings):
                return "KILLED"
            return "TRAINED"
    if state.train_steps > 0 and not state.train_done:
        return "TRAINING"
    return "UNKNOWN"


def diagnose_task_state(task_dir: str, task_id: int, task_name: str, run_name: str) -> TaskState:
    """Full diagnostic of a single task."""
    s = TaskState(task_id=task_id, task_name=task_name, output_dir=task_dir)

    # Checkpoints
    s.has_checkpoint, s.latest_checkpoint, s.checkpoint_count = detect_checkpoints(task_dir)

    # Saved weights
    s.has_lora_weights, s.has_prompt_keys = check_saved_weights(task_dir)

    # SRT signatures
    s.has_srt_signatures = check_srt_signatures(task_dir)

    # Training status
    s.train_done, s.train_steps, s.train_epochs = check_train_done(task_dir)

    # Eval / Predict status
    s.eval_done, s.predict_done, s.eval_results = check_eval_predict_done(
        task_dir, task_name, run_name
    )

    # Warnings
    if not os.path.isdir(task_dir):
        s.warnings.append("OUTPUT_DIR_NOT_FOUND")
    elif not s.has_checkpoint and not s.train_done:
        s.warnings.append("NO_CHECKPOINT")
    if s.train_steps > 0 and not s.train_done:
        s.warnings.append("TRAINING_KILLED")
    if s.train_done and not s.has_lora_weights:
        s.warnings.append("NO_LORA_WEIGHTS")
    if s.train_done and not s.has_srt_signatures:
        s.warnings.append("NO_SRT_SIGNATURES")
    if s.predict_done and len(s.eval_results) == 0:
        s.warnings.append("ALL_RESULTS_EMPTY")

    # Infer phase
    s.phase = infer_phase(s)

    return s


def print_task_table(states: list):
    """Print a formatted table of all tasks."""
    print(f"\n{'─'*110}")
    header = f"  {'TASK':<8} {'DIR':<5} {'PHASE':<12} {'STATUS':<45} {'WARNINGS'}"
    print(header)
    print(f"{'─'*110}")

    phases = {"UNKNOWN": "🔴", "TRAINING": "🟡", "TRAINED": "🟡", "EVALUATING": "🟡",
              "EVALUATED": "🟢", "INFERRED": "🟢", "KILLED": "🔴"}

    for s in states:
        icon = phases.get(s.phase, "⚪")
        warn_str = " ".join(s.warnings) if s.warnings else ""
        row = (f"  [{s.task_id:>2}] {s.task_name:<15} {icon} {s.phase:<12} "
               f"{s.summarize():<45} {warn_str}")
        print(row)

    print(f"{'─'*110}")


def summarize_pipeline(states: list):
    """Summarize the overall pipeline state."""
    total = len(states)
    phases_count = {}
    for s in states:
        phases_count[s.phase] = phases_count.get(s.phase, 0) + 1

    print(f"\n{'═'*60}")
    print("  PIPELINE SUMMARY")
    print(f"{'═'*60}")
    for phase, count in sorted(phases_count.items()):
        print(f"  {phase:<12}: {count}/{total} tasks")

    killed_tasks = [s for s in states if s.phase == "KILLED"]
    trained_only = [s for s in states if s.phase == "TRAINED"]
    evaluated = [s for s in states if s.phase in ("EVALUATED", "INFERRED")]
    unknown = [s for s in states if s.phase == "UNKNOWN"]

    if killed_tasks:
        print(f"\n  🔴 KILLED tasks ({len(killed_tasks)}):")
        for s in killed_tasks:
            print(f"     - {s.task_id}. {s.task_name} (steps={s.train_steps})")

    if trained_only:
        print(f"\n  🟡 Train-done but not eval'd ({len(trained_only)}):")
        for s in trained_only:
            print(f"     - {s.task_id}. {s.task_name}")

    if evaluated:
        print(f"\n  🟢 Fully evaluated ({len(evaluated)}):")
        for s in evaluated:
            eval_key = None
            for k in s.eval_results:
                if k.startswith("predict_"):
                    eval_key = k
                    break
            val = s.eval_results.get(eval_key, "N/A") if eval_key else "N/A"
            print(f"     - {s.task_id}. {s.task_name}: {eval_key} = {val}")

    if unknown:
        print(f"\n  ⚪ Unknown / Not started ({len(unknown)}):")
        for s in unknown:
            print(f"     - {s.task_id}. {s.task_name}")

    # Recovery suggestions
    print(f"\n{'─'*60}")
    print("  RECOVERY SUGGESTIONS")
    print(f"{'─'*60}")

    if trained_only:
        print(f"\n  1. Re-run prediction for these tasks (train done, eval missing):")
        for s in trained_only:
            print(f"     bash run_predict.sh {s.task_id} {s.task_name}")
        print(f"\n     Or manually re-run with --do_predict flag:")

    # Find the last fully evaluated task
    last_evalued = None
    for s in reversed(states):
        if s.phase == "INFERRED":
            last_evalued = s
            break

    if last_evalued:
        print(f"\n  2. Last fully evaluated task: {last_evalued.task_id}. {last_evalued.task_name}")
        next_task = states[last_evalued.task_id] if last_evalued.task_id < len(states) else None
        if next_task and next_task.phase == "UNKNOWN":
            print(f"     Next task to resume: {next_task.task_id}. {next_task.task_name}")
        if next_task and next_task.phase == "KILLED":
            print(f"     Next task (KILLED): {next_task.task_id}. {next_task.task_name}")
            print(f"     → Resume from checkpoint: {next_task.latest_checkpoint}")

    if killed_tasks:
        print(f"\n  3. For KILLED tasks — find checkpoint and resume:")
        for s in killed_tasks:
            if s.latest_checkpoint:
                print(f"     {s.task_id}. {s.task_name}: checkpoint={s.latest_checkpoint}")
            elif s.checkpoint_count > 0:
                print(f"     {s.task_id}. {s.task_name}: {s.checkpoint_count} checkpoints available")

    print(f"\n{'═'*60}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_task_status.py <run_name> [base_dir]")
        print("  base_dir defaults to 'logs_and_outputs'")
        sys.exit(1)

    run_name = sys.argv[1]
    base_dir = sys.argv[2] if len(sys.argv) >= 3 else "logs_and_outputs"

    output_dir = check_dir(base_dir, run_name)
    if output_dir is None:
        print(f"[ERROR] Cannot find output dir for run_name='{run_name}' in base_dir='{base_dir}'")
        print(f"  Searched: {base_dir}/{run_name}")
        sys.exit(1)

    print(f"[INFO] Run: {run_name}")
    print(f"[INFO] Base: {output_dir}")

    # Get task order
    task_order = get_task_order(output_dir)
    if not task_order:
        print("[ERROR] task_order.txt not found in output directory.")
        sys.exit(1)

    print(f"[INFO] Task order ({len(task_order)} tasks): {task_order}")

    # Diagnose each task
    states = []
    for i, task_name in enumerate(task_order):
        task_id = i + 1
        task_dir = os.path.join(output_dir, f"{task_id}-{task_name}")
        state = diagnose_task_state(task_dir, task_id, task_name, run_name)
        states.append(state)

    # Print results
    print_task_table(states)
    summarize_pipeline(states)


if __name__ == "__main__":
    main()
