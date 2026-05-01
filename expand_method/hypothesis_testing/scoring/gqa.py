"""
GQA scoring — adapted from SMoLoRA eval_gqa.py.

Metrics: Accuracy on balanced test set, plus breakdowns by type/length.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def score_gqa(
    result_file: str,
    question_dir: str,
    tier: str = "val",
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Score GQA predictions.

    Args:
        result_file: Path to predictions JSON with format:
            [{"questionId": str, "prediction": str}, ...]
        question_dir: Path to GQA dataset directory.
        tier: 'train', 'val', or 'test'.
        output_dir: Optional directory to write result files.

    Returns:
        Dict with accuracy (%), binary (%), open (%), per-type breakdowns.
    """
    question_dir = Path(question_dir)

    # Load questions
    questions_path = question_dir / f"{tier}_questions.json"
    if not questions_path.exists():
        # Try chunked format
        questions_path = question_dir / tier

    questions = json.load(open(questions_path))

    # Load predictions
    predictions_raw = json.load(open(result_file))
    if isinstance(predictions_raw, list):
        predictions = {p["questionId"]: p["prediction"] for p in predictions_raw}
    else:
        predictions = predictions_raw

    # Initialize score buckets
    scores = {
        "accuracy": [],
        "binary": [],
        "open": [],
        "accuracyPerStructuralType": defaultdict(list),
        "accuracyPerSemanticType": defaultdict(list),
        "accuracyPerLength": defaultdict(list),
        "accuracyPerSteps": defaultdict(list),
    }

    correct_count = 0
    total_balanced = 0

    for qid, question in questions.items():
        if not question.get("isBalanced", False):
            continue

        gold = question["answer"]
        pred = predictions.get(qid, "FAILED")
        is_correct = pred == gold

        if is_correct:
            correct_count += 1
        total_balanced += 1

        scores["accuracy"].append(1 if is_correct else 0)

        # Structural type
        struct_type = question.get("types", {}).get("structural", "query")
        scores["accuracyPerStructuralType"][struct_type].append(1 if is_correct else 0)

        # Semantic type
        sem_type = question.get("types", {}).get("semantic", "global")
        scores["accuracyPerSemanticType"][sem_type].append(1 if is_correct else 0)

        # Binary vs open
        answer_type = "open" if struct_type == "query" else "binary"
        scores[answer_type].append(1 if is_correct else 0)

        # Question length
        words = len(question.get("question", "").split())
        scores["accuracyPerLength"][words].append(1 if is_correct else 0)

        # Reasoning steps
        sem = question.get("semantic", [])
        steps = len([c for c in sem if not any(
            o in "{}: {}".format(c.get("operation", ""), c.get("argument", ""))
            for o in ["exist", "query: name", "choose name"]
        )])
        scores["accuracyPerSteps"][steps].append(1 if is_correct else 0)

    # Average scores
    def avg(lst):
        return sum(lst) / len(lst) * 100 if lst else 0.0

    def avg_dict(d):
        return {k: (avg(v), len(v)) for k, v in d.items()}

    result = {
        "accuracy": avg(scores["accuracy"]),
        "binary": avg(scores["binary"]),
        "open": avg(scores["open"]),
        "n_total": total_balanced,
        "n_correct": correct_count,
        "accuracyPerStructuralType": avg_dict(scores["accuracyPerStructuralType"]),
        "accuracyPerSemanticType": avg_dict(scores["accuracyPerSemanticType"]),
        "accuracyPerLength": avg_dict(scores["accuracyPerLength"]),
        "accuracyPerSteps": avg_dict(scores["accuracyPerSteps"]),
    }

    print(f"GQA — Balanced: {total_balanced}, Correct: {correct_count}, "
          f"Accuracy: {result['accuracy']:.2f}%")
    print(f"  Binary: {result['binary']:.2f}%, Open: {result['open']:.2f}%")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result_file_out = output_dir / "gqa_scores.json"
        with open(result_file_out, "w") as f:
            json.dump(result, f, indent=2)

    return result
