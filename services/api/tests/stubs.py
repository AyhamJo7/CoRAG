"""Deterministic test doubles for corag_cloud tests."""

import hashlib

import numpy as np


class StubEmbedder:
    """Deterministic unit vectors, same contract as OpenAIEmbedder.

    Identical texts map to identical vectors so duplicate detection and
    similarity search behave like a real model.
    """

    def __init__(self, dimensions: int = 1536):
        self.dimensions = dimensions
        self._cache: dict[str, np.ndarray] = {}

    def _vector(self, text: str) -> np.ndarray:
        if text not in self._cache:
            seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            vector = rng.standard_normal(self.dimensions).astype(np.float32)
            self._cache[text] = vector / np.linalg.norm(vector)
        return self._cache[text]

    def embed_texts(
        self,
        texts: list[str],
        show_progress: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        return np.stack([self._vector(t) for t in texts])

    def embed_query(self, query: str, normalize: bool = True) -> np.ndarray:
        return self._vector(query)


class FailingEmbedder:
    """Always raises — exercises the worker's failure/retry path."""

    def embed_texts(self, texts, show_progress=False, normalize=True):
        raise RuntimeError("embedding backend unavailable")

    def embed_query(self, query, normalize=True):
        raise RuntimeError("embedding backend unavailable")
