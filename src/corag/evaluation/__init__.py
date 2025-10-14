"""Evaluation framework for multi-hop QA."""

from corag.evaluation.datasets import DatasetLoader, EvalExample
from corag.evaluation.evaluator import EvaluationReport, EvaluationResult, Evaluator
from corag.evaluation.metrics import (
    exact_match,
    f1_score,
    max_em_over_ground_truths,
    max_f1_over_ground_truths,
)

__all__ = [
    "DatasetLoader",
    "EvalExample",
    "Evaluator",
    "EvaluationReport",
    "EvaluationResult",
    "exact_match",
    "f1_score",
    "max_em_over_ground_truths",
    "max_f1_over_ground_truths",
]
