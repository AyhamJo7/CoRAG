"""Tests for the multi-step retrieval pipeline."""

import json

import pytest

from corag.corpus.document import Chunk
from corag.indexing.index import FAISSIndex
from corag.retrieval.pipeline import RetrievalPipeline
from tests.stubs import StubController, StubEmbedder

CRITIQUE_SUFFICIENT = json.dumps({"is_sufficient": True, "confidence": 0.9})
CRITIQUE_INSUFFICIENT = json.dumps({"is_sufficient": False, "confidence": 0.2})


@pytest.fixture
def corpus_chunks():
    return [
        Chunk(
            chunk_id=f"chunk{i}",
            doc_id=f"doc{i}",
            text=f"Fact number {i} about the corpus.",
            start_char=0,
            end_char=30,
            tokens=6,
            doc_title=f"Document {i}",
        )
        for i in range(6)
    ]


@pytest.fixture
def embedder():
    return StubEmbedder(dimension=32)


@pytest.fixture
def index(corpus_chunks, embedder):
    idx = FAISSIndex(dimension=32, index_type="Flat")
    embeddings = embedder.embed_texts([c.text for c in corpus_chunks])
    idx.build(embeddings, corpus_chunks)
    return idx


def make_pipeline(index, embedder, responses, k=2, max_steps=6):
    controller = StubController(responses)
    pipeline = RetrievalPipeline(
        index=index,
        embedder=embedder,
        controller=controller,
        k=k,
        max_steps=max_steps,
    )
    return pipeline, controller


def test_decomposition_executes_all_sub_queries(index, embedder):
    decomposition = json.dumps({"sub_queries": ["who founded ACME", "where is ACME"]})
    pipeline, controller = make_pipeline(
        index, embedder, [decomposition, CRITIQUE_SUFFICIENT]
    )

    state = pipeline.run("Who founded ACME and where is it based?")

    assert len(state.steps) == 2
    assert state.executed_queries == ["who founded ACME", "where is ACME"]
    assert state.is_sufficient is True
    assert all(len(step.chunks) == 2 for step in state.steps)
    # decomposition + self-critique, no gap analysis needed
    assert len(controller.calls) == 2


def test_malformed_decomposition_falls_back_to_original_question(index, embedder):
    pipeline, _ = make_pipeline(
        index, embedder, ["this is not JSON", CRITIQUE_SUFFICIENT]
    )

    state = pipeline.run("What is the answer?")

    assert len(state.steps) == 1
    assert state.executed_queries == ["What is the answer?"]


def test_gap_analysis_adds_follow_up_step(index, embedder):
    decomposition = json.dumps({"sub_queries": ["first query"]})
    gap_analysis = json.dumps(
        {
            "answered_aspects": ["the first aspect"],
            "gaps": ["missing founding date"],
            "contradictions": [],
            "follow_up_queries": ["when was ACME founded"],
        }
    )
    pipeline, _ = make_pipeline(
        index,
        embedder,
        [decomposition, CRITIQUE_INSUFFICIENT, gap_analysis, CRITIQUE_SUFFICIENT],
    )

    state = pipeline.run("Tell me about ACME")

    assert len(state.steps) == 2
    assert state.steps[1].query == "when was ACME founded"
    assert state.steps[1].rationale == "Gap-filling follow-up query"
    assert state.gaps == ["missing founding date"]
    assert state.is_sufficient is True


def test_max_steps_caps_sub_queries(index, embedder):
    decomposition = json.dumps({"sub_queries": [f"query {i}" for i in range(5)]})
    pipeline, controller = make_pipeline(index, embedder, [decomposition], max_steps=2)

    state = pipeline.run("A very complex question")

    assert len(state.steps) == 2
    # Loop exits at max_steps without invoking critique or gap analysis
    assert len(controller.calls) == 1
    assert state.is_sufficient is False


def test_duplicate_sub_queries_are_skipped(index, embedder):
    decomposition = json.dumps(
        {"sub_queries": ["repeated query", "repeated query", "distinct query"]}
    )
    pipeline, _ = make_pipeline(index, embedder, [decomposition, CRITIQUE_SUFFICIENT])

    state = pipeline.run("Question with redundant decomposition")

    assert len(state.steps) == 2
    assert state.executed_queries == ["repeated query", "distinct query"]


def test_stops_when_no_gaps_identified(index, embedder):
    decomposition = json.dumps({"sub_queries": ["only query"]})
    no_gaps = json.dumps(
        {
            "answered_aspects": [],
            "gaps": [],
            "contradictions": [],
            "follow_up_queries": [],
        }
    )
    pipeline, _ = make_pipeline(
        index, embedder, [decomposition, CRITIQUE_INSUFFICIENT, no_gaps]
    )

    state = pipeline.run("Question")

    assert len(state.steps) == 1
    assert state.is_sufficient is False
