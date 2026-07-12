"""Tests for evaluation module."""

from corag.corpus.document import Chunk
from corag.evaluation.datasets import EvalExample
from corag.evaluation.evaluator import Evaluator
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


class FakePipeline:
    """Returns a pre-built retrieval state for any question."""

    def __init__(self, state):
        self.state = state

    def run(self, question):
        return self.state


class FakeSynthesizer:
    """Returns a fixed answer citing only source [1]."""

    def __init__(self, answer, citations):
        self.answer = answer
        self.citations = citations

    def synthesize(self, state):
        return self.answer, self.citations


class FakeLoader:
    """Yields a fixed list of examples for any dataset name."""

    def __init__(self, examples):
        self.examples = examples

    def load_dataset(self, dataset_name, split="validation", max_examples=None):
        yield from self.examples


def _make_state(question, doc_titles):
    from corag.retrieval.state import RetrievalState, RetrievalStep

    chunks = [
        Chunk(
            chunk_id=f"chunk-{i}",
            doc_id=f"doc-{i}",
            text=f"Text from {title}.",
            start_char=0,
            end_char=20,
            tokens=4,
            doc_title=title,
        )
        for i, title in enumerate(doc_titles)
    ]
    state = RetrievalState(original_question=question)
    state.add_step(
        RetrievalStep(
            step_num=1, query=question, chunks=chunks, scores=[1.0] * len(chunks)
        )
    )
    return state


def test_evaluator_computes_grounding_metrics():
    """End-to-end Evaluator run over stub components."""
    question = "What is the capital of France?"
    state = _make_state(question, ["Gold Doc", "Noise Doc"])
    citations = [
        {"id": "1", "title": "Gold Doc", "url": "", "chunk_id": "chunk-0"},
        {"id": "2", "title": "Noise Doc", "url": "", "chunk_id": "chunk-1"},
    ]
    examples = [
        EvalExample(
            id="ex1",
            question=question,
            answer="Paris",
            supporting_facts=["Gold Doc", "Missing Doc"],
        ),
        # No gold supporting facts: grounding metrics must stay None
        EvalExample(id="ex2", question=question, answer="Paris"),
    ]

    evaluator = Evaluator(
        retrieval_pipeline=FakePipeline(state),
        synthesizer=FakeSynthesizer("Paris [1].", citations),
        dataset_loader=FakeLoader(examples),
    )

    report = evaluator.evaluate("hotpotqa")

    assert report.num_examples == 2
    assert report.avg_em == 1.0

    ex1 = report.results[0]
    # Retrieved {Gold Doc, Noise Doc} vs gold {Gold Doc, Missing Doc}
    assert ex1.retrieval_precision == 0.5
    assert ex1.retrieval_recall == 0.5
    # Answer cites only [1] = Gold Doc
    assert ex1.citation_precision == 1.0
    assert ex1.citation_recall == 0.5

    ex2 = report.results[1]
    assert ex2.retrieval_precision is None
    assert ex2.citation_recall is None

    # Averages ignore examples without gold supporting facts
    assert report.avg_retrieval_precision == 0.5
    assert report.avg_citation_precision == 1.0

    metrics = report.to_dict()["metrics"]
    assert metrics["citation_recall"] == 0.5
    assert metrics["retrieval_recall"] == 0.5


def test_evaluator_report_grounding_none_when_no_gold_facts():
    question = "Anything?"
    state = _make_state(question, ["Some Doc"])
    evaluator = Evaluator(
        retrieval_pipeline=FakePipeline(state),
        synthesizer=FakeSynthesizer("An answer [1].", []),
        dataset_loader=FakeLoader(
            [EvalExample(id="ex1", question=question, answer="An answer")]
        ),
    )

    report = evaluator.evaluate("hotpotqa")

    assert report.avg_retrieval_precision is None
    assert report.avg_citation_recall is None
