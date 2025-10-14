"""Evaluation metrics for QA and retrieval."""

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison."""
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Normalize whitespace
    text = " ".join(text.split())
    return text


def exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact match score.

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)
    return 1.0 if pred_norm == gt_norm else 0.0


def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute F1 score.

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        F1 score between 0 and 1
    """
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0 if len(pred_tokens) != len(gt_tokens) else 1.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def max_f1_over_ground_truths(prediction: str, ground_truths: list[str]) -> float:
    """Compute max F1 over multiple ground truths.

    Args:
        prediction: Predicted answer
        ground_truths: List of acceptable ground truth answers

    Returns:
        Maximum F1 score
    """
    if not ground_truths:
        return 0.0
    return max(f1_score(prediction, gt) for gt in ground_truths)


def max_em_over_ground_truths(prediction: str, ground_truths: list[str]) -> float:
    """Compute max EM over multiple ground truths.

    Args:
        prediction: Predicted answer
        ground_truths: List of acceptable ground truth answers

    Returns:
        1.0 if any match, 0.0 otherwise
    """
    if not ground_truths:
        return 0.0
    return max(exact_match(prediction, gt) for gt in ground_truths)


def compute_retrieval_recall(
    retrieved_doc_ids: set[str], gold_doc_ids: set[str]
) -> float:
    """Compute retrieval recall.

    Args:
        retrieved_doc_ids: Set of retrieved document IDs
        gold_doc_ids: Set of gold document IDs

    Returns:
        Recall score
    """
    if not gold_doc_ids:
        return 0.0

    overlap = len(retrieved_doc_ids & gold_doc_ids)
    return overlap / len(gold_doc_ids)


def compute_retrieval_precision(
    retrieved_doc_ids: set[str], gold_doc_ids: set[str]
) -> float:
    """Compute retrieval precision.

    Args:
        retrieved_doc_ids: Set of retrieved document IDs
        gold_doc_ids: Set of gold document IDs

    Returns:
        Precision score
    """
    if not retrieved_doc_ids:
        return 0.0

    overlap = len(retrieved_doc_ids & gold_doc_ids)
    return overlap / len(retrieved_doc_ids)
