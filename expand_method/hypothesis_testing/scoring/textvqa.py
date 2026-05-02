"""
TextVQA scoring — adapted from SMoLoRA eval_textvqa.py + eval_vqav2.py.

Uses the standard TextVQA soft accuracy metric (TextVQAAccuracyEvaluator):
  - Each question has 10 human annotator answers
  - Prediction is processed via EvalAIAnswerProcessor (normalize, lowercase, strip)
  - Score = mean over 10 unique answers of min(1, #matching_annotators / 3)
  - Final accuracy = mean(scores) over all questions

NOT exact match — uses soft matching like the original SMoLoRA evaluation.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional

from .m4c_evaluator import TextVQAAccuracyEvaluator


def score_textvqa(
    result_file: str,
    annotation_file: str,
    output_file: Optional[str] = None,
) -> Dict:
    """
    Score TextVQA predictions using the standard soft accuracy metric.

    Args:
        result_file: Path to model predictions (.jsonl, one JSON per line).
            Each line: {"question_id": ..., "text": generated_answer}
        annotation_file: Path to TextVQA annotation (.json).
            Each entry: {"question_id": ..., "answers": [10 reference answers]}
        output_file: Optional path to write results.

    Returns:
        Dict with accuracy (%), n_total.
    """
    annotations_raw = json.load(open(annotation_file))
    # TextVQA annotations are wrapped in a "data" key
    if isinstance(annotations_raw, dict) and "data" in annotations_raw:
        annotations = {a["question_id"]: a for a in annotations_raw["data"]}
    else:
        annotations = {a["question_id"]: a for a in annotations_raw}

    predictions = [json.loads(line) for line in open(result_file)]

    evaluator = TextVQAAccuracyEvaluator()
    pred_list = []
    for pred in predictions:
        qid = str(pred.get("question_id", ""))
        annotation = annotations.get(qid, {})
        pred_list.append({
            "pred_answer": pred.get("text", ""),
            "gt_answers": annotation.get("answers", []),
        })

    accuracy = evaluator.eval_pred_list(pred_list) * 100
    n_total = len(pred_list)

    result = {
        "accuracy": float(accuracy),
        "n_total": n_total,
    }

    print(f"TextVQA — Samples: {n_total}, Accuracy: {accuracy:.2f}%")

    if output_file:
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

    return result