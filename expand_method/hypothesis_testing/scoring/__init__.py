"""
Scoring functions adapted from SMoLoRA and HiDe-LLaVA original evaluation code.
"""
from .science_qa import score_science_qa, score_science_qa_from_results
from .vqav2 import score_vqav2
from .gqa import score_gqa
from .textvqa import score_textvqa
from .m4c_evaluator import TextVQAAccuracyEvaluator, EvalAIAnswerProcessor

__all__ = [
    "score_science_qa",
    "score_science_qa_from_results",
    "score_vqav2",
    "score_gqa",
    "score_textvqa",
    "TextVQAAccuracyEvaluator",
    "EvalAIAnswerProcessor",
]
