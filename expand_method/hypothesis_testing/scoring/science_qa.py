"""
ScienceQA scoring — adapted from SMoLoRA eval_science_qa.py.

Metric: Exact match accuracy.
Logic: Extract answer letter (A/B/C/D/E) from generated text,
       compare with ground truth index.
"""
from __future__ import annotations
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


OPTIONS = ["A", "B", "C", "D", "E"]


def extract_answer_letter(
    generated_text: str,
    options: List[str] = OPTIONS,
) -> str:
    """
    Extract the answer letter (A/B/C/D/E) from generated text.

    Logic (from SMoLoRA eval_science_qa.py):
        1. If the text IS an option letter, return it
        2. If text starts with "X. " (e.g., "C. Something"), extract X
        3. Otherwise, search for the first option letter in text
        4. If none found, return "FAILED"
    """
    generated_text = generated_text.strip()

    # Direct match
    if generated_text in options:
        return generated_text

    # "X. " prefix pattern (e.g., "C. The answer is ...")
    pattern = re.compile(r"^([A-E])\.\s*")
    match = pattern.match(generated_text)
    if match:
        return match.group(1)

    # "The answer is X." pattern
    pattern2 = re.compile(r"\b([A-E])\b")
    matches = pattern2.findall(generated_text)
    if matches:
        # Return the first match that appears early in the text
        for m in matches:
            pos = generated_text.find(m)
            if pos < 30:  # First 30 chars
                return m

    # Search for any option letter in text
    for opt in options:
        if opt in generated_text:
            return opt

    return "FAILED"


def get_pred_idx(
    prediction: str,
    choices: List[str],
    options: List[str] = OPTIONS,
) -> int:
    """
    Get the index (0-based) of the predicted answer.

    Returns -1 if the prediction doesn't match any choice.
    """
    pred_letter = extract_answer_letter(prediction, options)
    if pred_letter not in options[:len(choices)]:
        return -1
    return options.index(pred_letter)


def score_science_qa_from_results(
    predictions: List[Dict],
    ground_truth: Dict,  # problem_id -> {'choices': [...], 'answer': int}
    split_indices: List[str],
    options: List[str] = OPTIONS,
) -> Dict:
    """
    Score ScienceQA results from a list of predictions.

    Args:
        predictions: List of dicts with 'question_id' and 'text' (generated).
        ground_truth: Dict mapping problem_id -> {choices, answer}.
        split_indices: List of problem IDs in this split.
        options: Answer options (default: A, B, C, D, E).

    Returns:
        Dict with acc (%), n_correct, n_total, per-sample results.
    """
    pred_dict = {p["question_id"]: p["text"] for p in predictions}

    results = {"correct": [], "incorrect": []}
    per_sample = {}

    correct = 0
    for prob_id in split_indices:
        prob = ground_truth[prob_id]
        gt_letter = options[prob["answer"]]

        pred_text = pred_dict.get(prob_id, "FAILED")
        pred_letter = extract_answer_letter(pred_text, options)
        pred_idx = get_pred_idx(pred_text, prob["choices"], options)

        is_correct = pred_idx == prob["answer"]
        if is_correct:
            correct += 1
            results["correct"].append({
                "question_id": prob_id,
                "parsed_ans": pred_letter,
                "ground_truth": gt_letter,
                "prediction": pred_text,
            })
        else:
            results["incorrect"].append({
                "question_id": prob_id,
                "parsed_ans": pred_letter,
                "ground_truth": gt_letter,
                "prediction": pred_text,
            })

        per_sample[prob_id] = {
            "parsed": pred_letter,
            "gt": gt_letter,
            "correct": is_correct,
            "prediction": pred_text,
        }

    total = len(split_indices)
    acc = correct / total * 100 if total > 0 else 0.0

    return {
        "acc": acc,
        "n_correct": correct,
        "n_total": total,
        "per_sample": per_sample,
    }


def score_science_qa(
    result_file: str,
    base_dir: str,
    split: str = "test",
    output_file: Optional[str] = None,
    output_result_file: Optional[str] = None,
    options: List[str] = OPTIONS,
) -> Dict:
    """
    Full ScienceQA scoring from result file and ground truth files.

    Args:
        result_file: Path to model predictions (.jsonl, one JSON per line).
        base_dir: Path to ScienceQA dataset directory.
        split: 'train', 'val', or 'test'.
        output_file: Optional path to write per-sample results.
        output_result_file: Optional path to write summary.
        options: Answer options.

    Returns:
        Dict with scoring results.
    """
    base_dir = Path(base_dir)

    # Load ground truth
    split_indices = json.load(open(base_dir / "pid_splits.json"))[split]
    problems = json.load(open(base_dir / "problems.json"))

    # Load predictions
    predictions = [json.loads(line) for line in open(result_file)]

    results = score_science_qa_from_results(
        predictions=predictions,
        ground_truth=problems,
        split_indices=split_indices,
        options=options,
    )

    # Write outputs
    if output_file:
        with open(output_file, "w") as f:
            json.dump(
                {
                    "correct": results["correct"][:100],  # limit for size
                    "incorrect": results["incorrect"][:100],
                },
                f,
                indent=2,
            )

    if output_result_file:
        with open(output_result_file, "w") as f:
            json.dump(
                {
                    "acc": results["acc"],
                    "n_correct": results["n_correct"],
                    "n_total": results["n_total"],
                },
                f,
                indent=2,
            )

    print(f"Total: {results['n_total']}, Correct: {results['n_correct']}, "
          f"Accuracy: {results['acc']:.2f}%")

    return results


def parse_results_jsonl(result_file: str) -> List[Dict]:
    """Parse a .jsonl result file into a list of dicts."""
    return [json.loads(line) for line in open(result_file)]
