"""Tests for corpus module."""

import pytest
from pathlib import Path
import json
import tempfile

from corag.corpus.document import Document, Chunk
from corag.corpus.chunker import Chunker
from corag.corpus.ingest import CorpusIngestor


def test_document_to_dict(sample_document):
    """Test document serialization."""
    doc_dict = sample_document.to_dict()
    assert doc_dict["id"] == "doc1"
    assert doc_dict["title"] == "Test Document"
    assert "test document" in doc_dict["text"]


def test_document_from_dict(sample_document):
    """Test document deserialization."""
    doc_dict = sample_document.to_dict()
    doc = Document.from_dict(doc_dict)
    assert doc.id == sample_document.id
    assert doc.title == sample_document.title


def test_chunk_to_dict(sample_chunk):
    """Test chunk serialization."""
    chunk_dict = sample_chunk.to_dict()
    assert chunk_dict["chunk_id"] == "chunk1"
    assert chunk_dict["doc_id"] == "doc1"


def test_chunker_basic(chunker, sample_document):
    """Test basic chunking."""
    chunks = chunker.chunk_document(sample_document)
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.doc_id == sample_document.id for c in chunks)


def test_chunker_multiple_documents(chunker, sample_documents):
    """Test chunking multiple documents."""
    chunks = chunker.chunk_documents(sample_documents)
    assert len(chunks) > 0
    doc_ids = {c.doc_id for c in chunks}
    assert "doc1" in doc_ids
    assert "doc2" in doc_ids


def test_corpus_ingestor_jsonl():
    """Test JSONL ingestion."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test JSONL file
        jsonl_path = Path(tmpdir) / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({"id": "1", "title": "Doc 1", "text": "Text 1"}) + "\n")
            f.write(json.dumps({"id": "2", "title": "Doc 2", "text": "Text 2"}) + "\n")

        ingestor = CorpusIngestor()
        docs = list(ingestor.ingest_jsonl(jsonl_path))

        assert len(docs) == 2
        assert docs[0].id == "1"
        assert docs[1].id == "2"


def test_corpus_ingestor_max_docs():
    """Test max docs limit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = Path(tmpdir) / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for i in range(10):
                f.write(json.dumps({"id": str(i), "title": f"Doc {i}", "text": f"Text {i}"}) + "\n")

        ingestor = CorpusIngestor()
        docs = list(ingestor.ingest_jsonl(jsonl_path, max_docs=3))

        assert len(docs) == 3
