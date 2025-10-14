"""Tests for indexing module."""

import tempfile
from pathlib import Path

from corag.indexing.index import FAISSIndex
from corag.utils.text import clean_text, count_tokens_approximate


def test_faiss_index_build(faiss_index, mock_embeddings, sample_chunks):
    """Test building FAISS index."""
    # Use only as many chunks as embeddings
    chunks = sample_chunks[: len(mock_embeddings)]
    embeddings = mock_embeddings[: len(chunks)]

    faiss_index.build(embeddings, chunks)

    assert faiss_index.index.ntotal == len(chunks)
    assert len(faiss_index.docstore) == len(chunks)


def test_faiss_index_search(faiss_index, mock_embeddings, sample_chunks):
    """Test FAISS search."""
    chunks = sample_chunks[: len(mock_embeddings)]
    embeddings = mock_embeddings[: len(chunks)]

    faiss_index.build(embeddings, chunks)

    # Search with first embedding
    query = mock_embeddings[0]
    results, scores = faiss_index.search(query, k=3)

    assert len(results) <= 3
    assert len(results) == len(scores)
    assert all(isinstance(s, float) for s in scores)


def test_faiss_index_save_load(faiss_index, mock_embeddings, sample_chunks):
    """Test saving and loading index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_dir = Path(tmpdir) / "index"

        chunks = sample_chunks[: len(mock_embeddings)]
        embeddings = mock_embeddings[: len(chunks)]

        # Build and save
        faiss_index.build(embeddings, chunks)
        faiss_index.save(index_dir)

        # Load
        loaded_index = FAISSIndex.load(index_dir)

        assert loaded_index.index.ntotal == faiss_index.index.ntotal
        assert len(loaded_index.docstore) == len(faiss_index.docstore)


def test_clean_text():
    """Test text cleaning."""
    text = "  This   has   extra   spaces.  "
    cleaned = clean_text(text)
    assert cleaned == "This has extra spaces."


def test_count_tokens_approximate():
    """Test token counting."""
    text = "This is a test sentence."
    count = count_tokens_approximate(text)
    assert count > 0
    assert isinstance(count, int)
