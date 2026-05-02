"""
VQAv2 scoring — adapted verbatim from SMoLoRA eval_vqav2.py.

Metric: Exact match accuracy (case-insensitive).
Logic: pred.upper() == ground_truth.upper()

This is the SMoLoRA metric — NOT the TextVQA soft metric
(which uses EvalAIAnswerProcessor + 10-way human annotator matching).
For TextVQA scoring, use scoring/textvqa.py instead.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Optional


def score_vqav2(
    result_file: str,
    annotation_file: str,
    output_file: Optional[str] = None,
) -> Dict:
    """
    Score VQAv2 predictions (exact match, case-insensitive).

    Matches SMoLoRA eval_vqav2.py exactly:
        if pred.upper() == ground_truth.upper():
            right += 1

    Args:
        result_file: Path to model predictions (.jsonl, one JSON per line).
            Each line: {"question_id": ..., "text": generated_answer}
        annotation_file: Path to VQAv2 annotations (.json).
            Each entry: {"question_id": ..., "answer": "..."}
        output_file: Optional path to write results.

    Returns:
        Dict with accuracy (%), n_correct, n_total.
    """
    annotations = json.load(open(annotation_file))
    annotations = {
        str(a["question_id"]): a["answer"]
        for a in annotations
    }

    predictions = [json.loads(line) for line in open(result_file)]

    total = len(predictions)
    correct = 0

    for pred in predictions:
        qid = str(pred.get("question_id", ""))
        generated = pred.get("text", "").strip()
        gt = annotations.get(qid, "")

        if generated.upper() == gt.upper():
            correct += 1

    accuracy = correct / total * 100 if total > 0 else 0.0

    result = {
        "accuracy": accuracy,
        "n_correct": correct,
        "n_total": total,
    }

    print(f"VQAv2 — Samples: {total}, Correct: {correct}, Accuracy: {accuracy:.2f}%")

    if output_file:
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

    return result
