"""Pytest configuration and fixtures."""

import pytest
import numpy as np
from pathlib import Path
from corag.corpus.document import Document, Chunk
from corag.corpus.chunker import Chunker
from corag.indexing.embedder import Embedder
from corag.indexing.index import FAISSIndex


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    return Document(
        id="doc1",
        title="Test Document",
        text="This is a test document. It has multiple sentences. We use it for testing.",
        url="https://example.com/doc1",
        sections=["Introduction"],
    )


@pytest.fixture
def sample_documents():
    """Create multiple sample documents."""
    return [
        Document(
            id="doc1",
            title="Document One",
            text="This is the first document. " * 50,  # Make it long enough to chunk
        ),
        Document(
            id="doc2",
            title="Document Two",
            text="This is the second document. " * 50,
        ),
    ]


@pytest.fixture
def sample_chunk():
    """Create a sample chunk for testing."""
    return Chunk(
        chunk_id="chunk1",
        doc_id="doc1",
        text="This is a test chunk.",
        start_char=0,
        end_char=21,
        tokens=5,
        doc_title="Test Doc",
        doc_url="https://example.com",
    )


@pytest.fixture
def sample_chunks():
    """Create multiple sample chunks."""
    return [
        Chunk(
            chunk_id=f"chunk{i}",
            doc_id="doc1",
            text=f"This is chunk number {i}.",
            start_char=i * 30,
            end_char=(i + 1) * 30,
            tokens=6,
            doc_title="Test Doc",
        )
        for i in range(5)
    ]


@pytest.fixture
def chunker():
    """Create a chunker instance."""
    return Chunker(chunk_size=20, chunk_overlap=5, min_chunk_size=5)


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings."""
    return np.random.rand(10, 128).astype(np.float32)


@pytest.fixture
def faiss_index():
    """Create a FAISS index for testing."""
    return FAISSIndex(dimension=128, index_type="Flat")
