"""Tests for evaluation module."""

from corag.evaluation.metrics import (
    compute_retrieval_precision,
    compute_retrieval_recall,
    exact_match,
    f1_score,
    max_em_over_ground_truths,
    max_f1_over_ground_truths,
    normalize_answer,
)


def test_normalize_answer():
    """Test answer normalization."""
    assert normalize_answer("The Quick Brown Fox") == "quick brown fox"
    assert normalize_answer("  Extra   spaces  ") == "extra spaces"
    assert normalize_answer("Hello, World!") == "hello world"


def test_exact_match():
    """Test exact match metric."""
    assert exact_match("Paris", "Paris") == 1.0
    assert exact_match("paris", "Paris") == 1.0
    assert exact_match("Paris", "London") == 0.0
    assert exact_match("The capital is Paris", "Paris") == 0.0


def test_f1_score():
    """Test F1 score metric."""
    assert f1_score("Paris", "Paris") == 1.0
    assert f1_score("Paris, France", "Paris") > 0.0
    assert f1_score("Paris", "London") == 0.0


def test_max_f1_over_ground_truths():
    """Test max F1 over multiple ground truths."""
    prediction = "Paris"
    ground_truths = ["London", "Paris", "Berlin"]
    score = max_f1_over_ground_truths(prediction, ground_truths)
    assert score == 1.0


def test_max_em_over_ground_truths():
    """Test max EM over multiple ground truths."""
    prediction = "Paris"
    ground_truths = ["London", "Paris", "Berlin"]
    score = max_em_over_ground_truths(prediction, ground_truths)
    assert score == 1.0


def test_retrieval_recall():
    """Test retrieval recall."""
    retrieved = {"doc1", "doc2", "doc3"}
    gold = {"doc2", "doc3", "doc4"}
    recall = compute_retrieval_recall(retrieved, gold)
    assert recall == 2.0 / 3.0  # 2 out of 3 gold docs retrieved


def test_retrieval_precision():
    """Test retrieval precision."""
    retrieved = {"doc1", "doc2", "doc3"}
    gold = {"doc2", "doc3", "doc4"}
    precision = compute_retrieval_precision(retrieved, gold)
    assert precision == 2.0 / 3.0  # 2 out of 3 retrieved docs are gold
