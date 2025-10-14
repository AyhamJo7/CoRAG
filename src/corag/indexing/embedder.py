"""Text embedding using sentence transformers."""

import logging

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """Embedds text using sentence transformers."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/msmarco-distilbert-base-v4",
        device: str | None = None,
        batch_size: int = 32,
    ):
        """Initialize embedder.

        Args:
            model_name: HuggingFace model name
            device: Device to use (cpu/cuda/mps), auto-detect if None
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.batch_size = batch_size

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        logger.info(f"Loading embedding model {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.dimension}")

    def embed_texts(
        self,
        texts: list[str],
        show_progress: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        """Embed a list of texts.

        Args:
            texts: Texts to embed
            show_progress: Show progress bar
            normalize: Normalize embeddings to unit length

        Returns:
            Array of embeddings with shape (n_texts, dimension)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        return embeddings

    def embed_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """Embed a single query.

        Args:
            query: Query text
            normalize: Normalize embedding to unit length

        Returns:
            Embedding vector
        """
        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        return embedding
