"""Query/document embedding via the OpenAI embeddings API.

Satisfies the ``corag.retrieval.interfaces.QueryEmbedder`` protocol. The
per-instance LRU cache on ``embed_query`` matters: the pipeline re-embeds
every executed query during duplicate detection on each step, which would
otherwise turn one question into O(steps²) API calls.
"""

import logging
import time
from collections import OrderedDict
from typing import Any

import numpy as np
from openai import (
    APIConnectionError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30.0
RETRYABLE_ERRORS = (APIConnectionError, RateLimitError, InternalServerError)


class OpenAIEmbedder:
    """Embeds text with text-embedding-3-small (1536 dims), L2-normalized."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        batch_size: int = 256,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = DEFAULT_TIMEOUT,
        query_cache_size: int = 256,
    ):
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = OpenAI(api_key=api_key, timeout=timeout, max_retries=0)
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._query_cache_size = query_cache_size

    def embed_texts(
        self,
        texts: list[str],
        show_progress: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """Embed a list of texts, batched.

        Args:
            texts: Texts to embed
            show_progress: Unused (API is fast); kept for protocol parity
            normalize: Normalize embeddings to unit length

        Returns:
            Array of embeddings with shape (n_texts, dimensions)
        """
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            # The API rejects empty strings; a lone space embeds harmlessly.
            batch = [t if t.strip() else " " for t in batch]
            response = self._create_with_retry(batch)
            vectors.extend(item.embedding for item in response.data)

        result = np.asarray(vectors, dtype=np.float32)
        if normalize:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            result = result / norms
        return result

    def embed_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """Embed a single query with LRU caching."""
        cached = self._query_cache.get(query)
        if cached is not None:
            self._query_cache.move_to_end(query)
            return cached

        embedding: np.ndarray = self.embed_texts([query], normalize=normalize)[0]
        self._query_cache[query] = embedding
        if len(self._query_cache) > self._query_cache_size:
            self._query_cache.popitem(last=False)
        return embedding

    def _create_with_retry(self, batch: list[str]) -> Any:
        for attempt in range(self.max_retries):
            try:
                return self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    dimensions=self.dimensions,
                )
            except RETRYABLE_ERRORS as e:
                logger.warning("Embedding call failed (attempt %d): %s", attempt + 1, e)
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise
        raise RuntimeError("unreachable")
