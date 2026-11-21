"""Structural interfaces for pluggable retrieval backends.

The research stack (FAISSIndex + sentence-transformers Embedder) and any
hosted backend (e.g. pgvector + API embeddings) satisfy these protocols
structurally — no inheritance required.
"""

from typing import Protocol

import numpy as np

from corag.corpus.document import Chunk


class QueryEmbedder(Protocol):
    """Embeds a query string into a normalized vector."""

    def embed_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """Embed a single query.

        Args:
            query: Query text
            normalize: Normalize embedding to unit length

        Returns:
            Embedding vector
        """
        ...


class SearchIndex(Protocol):
    """Nearest-neighbour search over embedded chunks."""

    def search(
        self, query_embedding: np.ndarray, k: int = 10
    ) -> tuple[list[Chunk], list[float]]:
        """Search for similar chunks.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return

        Returns:
            Tuple of (chunks, scores)
        """
        ...
