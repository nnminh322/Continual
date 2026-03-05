"""
OT-SIGN Run Logger
==================
Logs each training run with config, vMF signatures, and per-task scores.
Computes AP (Average Performance) and FT (Forgetting) directly comparable
to GainLoRA paper Table 1 / Table 2.

Output files:
  run_log.jsonl          — machine-readable full log
  scores_matrix.json     — R[i][j] = score of task j after training task i
  ap_ft_summary.md       — final AP/FT table ready for paper comparison
"""

import json
import os
from datetime import datetime


# Keys used for score extraction from HuggingFace Trainer metrics
SCORE_KEY_TEMPLATES = {
    "superni": ["predict_rougeL_for_{task}", "eval_rougeL_for_{task}", "predict_rougeL"],
    "long":    ["predict_exact_match_for_{task}", "eval_exact_match_for_{task}", "predict_exact_match"],
}


def _extract_score(metrics_dict, task_name, benchmark_type="superni"):
    """Extract the primary score for a given task from a metrics dict."""
    templates = SCORE_KEY_TEMPLATES.get(benchmark_type, SCORE_KEY_TEMPLATES["superni"])
    for tmpl in templates:
        key = tmpl.format(task=task_name)
        if key in metrics_dict:
            v = metrics_dict[key]
            # rougeL is typically 0–1 from compute_metrics; multiply to get percentage
            if v <= 1.0 and "rouge" in key.lower():
                v = v * 100.0
            return round(v, 2)
    # fallback: look for any key containing the task name
    for k, v in metrics_dict.items():
        if task_name in k and isinstance(v, (int, float)):
            if v <= 1.0 and "rouge" in k.lower():
                v = v * 100.0
            return round(v, 2)
    return None


class RunLogger:
    """
    Logs every task training run and builds the result matrix R[i][j].
    
    Result matrix semantics (matching GainLoRA paper Table 1):
      R[i][j] = score on task j after model has been trained on tasks 0..i
      AP       = (1/T) * sum_j R[T-1][j]           (final row average)
      FT       = (1/(T-1)) * sum_j (R[j][j] - R[T-1][j])  (diagonal drop)
    """

    def __init__(self, log_dir="logs_and_outputs/ot_sign_logs", benchmark_type="superni"):
        self.log_dir = log_dir
        self.benchmark_type = benchmark_type  # "superni" or "long"
        os.makedirs(log_dir, exist_ok=True)
        self.log_file     = os.path.join(log_dir, "run_log.jsonl")
        self.matrix_file  = os.path.join(log_dir, "scores_matrix.json")
        self.summary_file = os.path.join(log_dir, "ap_ft_summary.md")

        # In-memory score matrix: {trained_on_task_id: {eval_task_name: score}}
        self._scores = self._load_scores_matrix()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_run(self, task_name, task_id, config, vmf_signature=None, metrics=None, extra=None):
        """Log a single task's training metadata (config, vMF, train metrics)."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "task_name": task_name,
            "task_id": task_id,
            "config": config,
            "vmf_signature": vmf_signature,
            "metrics": metrics,
            "extra": extra,
        }
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry, default=str) + "\n")
        self._update_md_log(entry)
        print(f"[RunLogger] Logged task {task_id}: {task_name}")

    def log_predict_scores(self, trained_on_task_id, predict_metrics_dict, task_order_list):
        """
        After training on task `trained_on_task_id`, log prediction scores for all
        tasks evaluated so far.
        
        Args:
            trained_on_task_id: index of the just-trained task (0-based)
            predict_metrics_dict: HF Trainer predict_results.metrics
            task_order_list: full list of task names in sequence order
        """
        row = {}
        for j, tname in enumerate(task_order_list[:trained_on_task_id + 1]):
            score = _extract_score(predict_metrics_dict, tname, self.benchmark_type)
            if score is not None:
                row[tname] = score

        self._scores[trained_on_task_id] = row
        self._save_scores_matrix()

        # Print progress table
        self._print_current_scores(trained_on_task_id, task_order_list)

    def print_ap_ft(self, task_order_list, method_name="OT-SIGN+GainLoRA"):
        """
        Compute and print AP / FT from the accumulated score matrix.
        Prints a table directly comparable to GainLoRA paper.
        
        Returns:
            (ap, ft) as floats
        """
        T = len(task_order_list)
        if T not in self._scores and (T - 1) not in self._scores:
            print("[RunLogger] Not enough data to compute AP/FT yet.")
            return None, None

        final_row_id = T - 1
        if final_row_id not in self._scores:
            print("[RunLogger] Final task scores not found.")
            return None, None

        final_scores = [self._scores[final_row_id].get(t) for t in task_order_list]
        final_scores_valid = [s for s in final_scores if s is not None]
        ap = round(sum(final_scores_valid) / len(final_scores_valid), 2) if final_scores_valid else 0.0

        # FT = mean of (diagonal - final row), skip first task
        ft_values = []
        for j in range(T - 1):
            tname = task_order_list[j]
            diag = self._scores.get(j, {}).get(tname)
            final = self._scores.get(final_row_id, {}).get(tname)
            if diag is not None and final is not None:
                ft_values.append(diag - final)
        ft = round(sum(ft_values) / len(ft_values), 2) if ft_values else 0.0

        # ---- Console output ----
        sep = "-" * 60
        print(f"\n{sep}")
        print(f"  FINAL RESULTS — {method_name}")
        print(sep)
        print(f"  {'Task':<45} {'Peak':>6}  {'Final':>6}  {'Drop':>6}")
        print(sep)
        for j, tname in enumerate(task_order_list):
            diag  = self._scores.get(j, {}).get(tname, "-")
            final = self._scores.get(final_row_id, {}).get(tname, "-")
            drop  = f"{diag - final:.2f}" if isinstance(diag, float) and isinstance(final, float) else "-"
            diag_s  = f"{diag:.2f}"  if isinstance(diag,  float) else str(diag)
            final_s = f"{final:.2f}" if isinstance(final, float) else str(final)
            print(f"  {tname:<45} {diag_s:>6}  {final_s:>6}  {drop:>6}")
        print(sep)
        print(f"  AP  = {ap:.2f}   |   FT = {ft:.2f}")
        print(f"{sep}\n")

        # ---- File output ----
        self._write_ap_ft_summary(task_order_list, method_name, ap, ft)
        return ap, ft

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_scores_matrix(self):
        if os.path.exists(self.matrix_file):
            with open(self.matrix_file) as f:
                raw = json.load(f)
            return {int(k): v for k, v in raw.items()}
        return {}

    def _save_scores_matrix(self):
        with open(self.matrix_file, 'w') as f:
            json.dump({str(k): v for k, v in self._scores.items()}, f, indent=2)

    def _print_current_scores(self, trained_on_task_id, task_order_list):
        row = self._scores.get(trained_on_task_id, {})
        tname = task_order_list[trained_on_task_id] if trained_on_task_id < len(task_order_list) else "?"
        print(f"\n[RunLogger] After task {trained_on_task_id} ({tname}) — predict scores:")
        for t, s in row.items():
            print(f"  {t:<50} {s:.2f}")

    def _update_md_log(self, entry):
        """Append human-readable training info only (not the score matrix)."""
        with open(self.log_file.replace(".jsonl", "_detail.md"), 'a') as f:
            f.write(f"\n## Task {entry['task_id']}: {entry['task_name']}\n")
            f.write(f"*{entry['timestamp']}*\n\n")
            if entry.get('vmf_signature'):
                f.write(f"- vMF kappa: `{entry['vmf_signature'].get('kappa', '?')}`  "
                        f"mu_norm: `{entry['vmf_signature'].get('mu_norm', '?')}`\n")
            if entry.get('metrics'):
                f.write("| Metric | Value |\n|--------|-------|\n")
                for k, v in entry['metrics'].items():
                    val = f"{v:.4f}" if isinstance(v, float) else str(v)
                    f.write(f"| `{k}` | {val} |\n")
            f.write("\n")

    def _write_ap_ft_summary(self, task_order_list, method_name, ap, ft):
        """Write a markdown file ready to paste into the comparison table."""
        T = len(task_order_list)
        final_row_id = T - 1
        with open(self.summary_file, 'a') as f:
            f.write(f"\n## {method_name}\n")
            f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n")
            f.write("| Task | Peak Score (R[i,i]) | Final Score (R[T,i]) | Forgetting |\n")
            f.write("|------|---------------------|----------------------|------------|\n")
            for j, tname in enumerate(task_order_list):
                diag  = self._scores.get(j, {}).get(tname, "-")
                final = self._scores.get(final_row_id, {}).get(tname, "-")
                drop  = f"{diag - final:.2f}" if isinstance(diag, float) and isinstance(final, float) else "-"
                f.write(f"| {tname} | {diag} | {final} | {drop} |\n")
            f.write(f"\n**AP = {ap:.2f}** | **FT = {ft:.2f}**\n\n")

    # ---- Legacy compat ----
    def load_all_runs(self):
        runs = []
        if os.path.exists(self.log_file):
            with open(self.log_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        runs.append(json.loads(line))
        return runs

    def get_latest_run(self):
        runs = self.load_all_runs()
        return runs[-1] if runs else None
