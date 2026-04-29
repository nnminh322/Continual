import string
import sys
from pathlib import Path

try:
    from rouge import rouge_scorer
except ImportError:  # pragma: no cover - compatibility fallback
    vendored_rouge_parent = Path(__file__).resolve().parents[2] / "new_gainlora" / "src"
    if str(vendored_rouge_parent) not in sys.path:
        sys.path.insert(0, str(vendored_rouge_parent))
    from rouge import rouge_scorer


def normalize_answer(text: str) -> str:
    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(char for char in value if char not in exclude)

    def lower(value: str) -> str:
        return value.lower()

    return white_space_fix(remove_punc(lower(text)))


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def rouge1_score(prediction: str, ground_truth: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rouge1"].fmeasure


def rougeL_score(prediction: str, ground_truth: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rougeL"].fmeasure


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: list[str]) -> float:
    return max(metric_fn(prediction, ground_truth) for ground_truth in ground_truths)


def compute_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    if len(predictions) != len(references):
        raise ValueError(
            f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
        )
    if not references:
        return {"exact_match": 0.0, "rouge1": 0.0, "eval_rougeL": 0.0}

    exact_match = 0.0
    rouge1 = 0.0
    rougeL = 0.0
    for prediction, reference in zip(predictions, references):
        ground_truths = [reference]
        exact_match += metric_max_over_ground_truths(
            exact_match_score,
            prediction=prediction,
            ground_truths=ground_truths,
        )
        rouge1 += metric_max_over_ground_truths(
            rouge1_score,
            prediction=prediction,
            ground_truths=ground_truths,
        )
        rougeL += metric_max_over_ground_truths(
            rougeL_score,
            prediction=prediction,
            ground_truths=ground_truths,
        )

    scale = 100.0 / len(references)
    return {
        "exact_match": round(exact_match * scale, 4),
        "rouge1": round(rouge1 * scale, 4),
        "eval_rougeL": round(rougeL * scale, 4),
    }


def compute_grouped_metrics(
    predictions: list[str],
    references: list[str],
    groups: list[str],
) -> dict[str, float]:
    if not (len(predictions) == len(references) == len(groups)):
        raise ValueError("predictions, references, and groups must have the same length")

    grouped_examples: dict[str, list[tuple[str, str]]] = {}
    for prediction, reference, group in zip(predictions, references, groups):
        grouped_examples.setdefault(group, []).append((prediction, reference))

    results: dict[str, float] = {}
    for group, group_examples in grouped_examples.items():
        group_predictions, group_references = zip(*group_examples)
        group_metrics = compute_metrics(list(group_predictions), list(group_references))
        for metric_name, metric_value in group_metrics.items():
            results[f"{metric_name}_for_{group}"] = metric_value
    return results