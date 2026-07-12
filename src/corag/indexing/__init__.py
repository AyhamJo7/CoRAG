"""Dense indexing and retrieval with FAISS."""

try:
    from corag.indexing.embedder import Embedder
    from corag.indexing.index import FAISSIndex
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "The FAISS/sentence-transformers research stack is not installed. "
        'Install it with: pip install "corag[research]"'
    ) from e

__all__ = ["Embedder", "FAISSIndex"]
