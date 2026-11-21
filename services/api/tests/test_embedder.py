"""Unit tests for the OpenAI embedder (fake client, no network)."""

from types import SimpleNamespace

import numpy as np
import pytest

from corag_cloud.retrieval.openai_embedder import OpenAIEmbedder


class FakeEmbeddings:
    def __init__(self):
        self.calls: list[list[str]] = []

    def create(self, model, input, dimensions):
        self.calls.append(list(input))
        data = [
            SimpleNamespace(
                embedding=[float(len(text)), 1.0] + [0.0] * (dimensions - 2)
            )
            for text in input
        ]
        return SimpleNamespace(data=data)


@pytest.fixture
def embedder():
    e = OpenAIEmbedder(api_key="test-key", dimensions=8, batch_size=2)
    fake = FakeEmbeddings()
    e.client = SimpleNamespace(embeddings=fake)  # type: ignore[assignment]
    return e, fake


def test_embed_texts_batches(embedder):
    e, fake = embedder
    result = e.embed_texts(["a", "bb", "ccc", "dddd", "eeeee"])

    assert result.shape == (5, 8)
    assert [len(call) for call in fake.calls] == [2, 2, 1]


def test_embeddings_are_normalized(embedder):
    e, _ = embedder
    result = e.embed_texts(["hello", "world"])

    norms = np.linalg.norm(result, axis=1)
    assert np.allclose(norms, 1.0)


def test_empty_strings_are_padded(embedder):
    e, fake = embedder
    e.embed_texts(["", "  "])

    assert fake.calls == [[" ", " "]]


def test_embed_query_caches(embedder):
    e, fake = embedder
    first = e.embed_query("what is corag?")
    second = e.embed_query("what is corag?")

    assert np.array_equal(first, second)
    assert len(fake.calls) == 1


def test_embed_texts_empty_input(embedder):
    e, fake = embedder
    result = e.embed_texts([])

    assert result.shape == (0, 8)
    assert fake.calls == []
