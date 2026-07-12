"""Deterministic test doubles for pipeline and controller tests."""

import hashlib

import numpy as np

from corag.controller.base import Controller, GenerationConfig


class StubController(Controller):
    """Controller that replays a scripted queue of responses."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[dict[str, str | None]] = []

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        config: GenerationConfig | None = None,
    ) -> str:
        self.calls.append({"prompt": prompt, "system": system})
        if not self.responses:
            raise AssertionError("StubController ran out of scripted responses")
        return self.responses.pop(0)

    def count_tokens(self, text: str) -> int:
        return len(text.split())


class StubEmbedder:
    """Embedder producing deterministic unit vectors per distinct text.

    Identical texts map to identical vectors (cosine similarity 1.0), so
    duplicate-query detection behaves exactly as with a real model, while
    distinct random unit vectors in 32 dimensions stay far below the
    similarity threshold.
    """

    def __init__(self, dimension: int = 32):
        self.dimension = dimension
        self._cache: dict[str, np.ndarray] = {}

    def _vector(self, text: str) -> np.ndarray:
        if text not in self._cache:
            seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            vector = rng.standard_normal(self.dimension).astype(np.float32)
            self._cache[text] = vector / np.linalg.norm(vector)
        return self._cache[text]

    def embed_query(self, query: str, normalize: bool = True) -> np.ndarray:
        return self._vector(query)

    def embed_texts(
        self,
        texts: list[str],
        show_progress: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        return np.stack([self._vector(t) for t in texts])
