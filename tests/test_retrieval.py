"""Tests for retrieval module."""

import pytest

from corag.retrieval.state import RetrievalState, RetrievalStep


def test_retrieval_state_add_step(sample_chunks):
    """Test adding retrieval step."""
    state = RetrievalState(original_question="Test question")

    step = RetrievalStep(
        step_num=1,
        query="test query",
        chunks=sample_chunks[:3],
        scores=[0.9, 0.8, 0.7],
    )

    state.add_step(step)

    assert len(state.steps) == 1
    assert len(state.executed_queries) == 1
    assert state.total_chunks_retrieved == 3


def test_retrieval_state_get_unique_chunks(sample_chunks):
    """Test getting unique chunks."""
    state = RetrievalState(original_question="Test question")

    # Add same chunks in two steps
    step1 = RetrievalStep(
        step_num=1,
        query="query 1",
        chunks=sample_chunks[:3],
        scores=[0.9, 0.8, 0.7],
    )
    step2 = RetrievalStep(
        step_num=2,
        query="query 2",
        chunks=sample_chunks[1:4],  # Overlaps with step1
        scores=[0.9, 0.8, 0.7],
    )

    state.add_step(step1)
    state.add_step(step2)

    unique = state.get_unique_chunks()
    unique_ids = {c.chunk_id for c in unique}

    # Should have 4 unique chunks (0, 1, 2, 3)
    assert len(unique_ids) == 4


def test_retrieval_state_to_dict(sample_chunks):
    """Test state serialization."""
    state = RetrievalState(original_question="Test question")

    step = RetrievalStep(
        step_num=1,
        query="test query",
        chunks=sample_chunks[:2],
        scores=[0.9, 0.8],
    )

    state.add_step(step)
    state_dict = state.to_dict()

    assert state_dict["original_question"] == "Test question"
    assert len(state_dict["steps"]) == 1
    assert state_dict["steps"][0]["step_num"] == 1
