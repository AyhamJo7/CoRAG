"""FAISS-based dense index for retrieval."""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from corag.corpus.document import Chunk

logger = logging.getLogger(__name__)


class FAISSIndex:
    """FAISS index for dense retrieval."""

    def __init__(
        self,
        dimension: int,
        index_type: str = "Flat",
        nlist: int = 100,
        metric: str = "inner_product",
    ):
        """Initialize FAISS index.

        Args:
            dimension: Embedding dimension
            index_type: Index type (Flat, IVF, HNSW)
            nlist: Number of clusters for IVF
            metric: Distance metric (inner_product or l2)
        """
        self.dimension = dimension
        self.index_type = index_type
        self.nlist = nlist
        self.metric = metric

        self.index: Optional[faiss.Index] = None
        self.docstore: Dict[int, Chunk] = {}
        self.is_trained = False

        self._initialize_index()

    def _initialize_index(self) -> None:
        """Initialize the FAISS index."""
        if self.metric == "inner_product":
            if self.index_type == "Flat":
                self.index = faiss.IndexFlatIP(self.dimension)
            elif self.index_type == "IVF":
                quantizer = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIVFFlat(
                    quantizer, self.dimension, self.nlist, faiss.METRIC_INNER_PRODUCT
                )
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
        elif self.metric == "l2":
            if self.index_type == "Flat":
                self.index = faiss.IndexFlatL2(self.dimension)
            elif self.index_type == "IVF":
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.IndexIVFFlat(
                    quantizer, self.dimension, self.nlist, faiss.METRIC_L2
                )
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

        logger.info(f"Initialized {self.index_type} index with {self.metric} metric")

    def build(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        """Build index from embeddings and chunks.

        Args:
            embeddings: Array of embeddings (n_chunks, dimension)
            chunks: List of Chunk objects corresponding to embeddings
        """
        if len(embeddings) != len(chunks):
            raise ValueError("Number of embeddings and chunks must match")

        logger.info(f"Building index with {len(embeddings)} vectors")

        # Train index if needed
        if self.index_type == "IVF" and not self.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings.astype(np.float32))
            self.is_trained = True

        # Add vectors to index
        self.index.add(embeddings.astype(np.float32))

        # Build docstore
        for idx, chunk in enumerate(chunks):
            self.docstore[idx] = chunk

        logger.info(f"Index built with {self.index.ntotal} vectors")

    def search(
        self, query_embedding: np.ndarray, k: int = 10
    ) -> Tuple[List[Chunk], List[float]]:
        """Search for similar chunks.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return

        Returns:
            Tuple of (chunks, scores)
        """
        if self.index.ntotal == 0:
            return [], []

        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Search
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding.astype(np.float32), k)

        # Retrieve chunks
        chunks = []
        valid_scores = []
        for idx, score in zip(indices[0], scores[0]):
            if idx in self.docstore:
                chunks.append(self.docstore[idx])
                valid_scores.append(float(score))

        return chunks, valid_scores

    def save(self, index_dir: Path) -> None:
        """Save index and docstore to disk.

        Args:
            index_dir: Directory to save index files
        """
        index_dir.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = index_dir / "index.faiss"
        faiss.write_index(self.index, str(index_path))

        # Save docstore
        docstore_path = index_dir / "docstore.pkl"
        with open(docstore_path, "wb") as f:
            pickle.dump(self.docstore, f)

        # Save metadata
        metadata = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "nlist": self.nlist,
            "metric": self.metric,
            "num_vectors": self.index.ntotal,
            "is_trained": self.is_trained,
        }
        metadata_path = index_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Index saved to {index_dir}")

    @classmethod
    def load(cls, index_dir: Path) -> "FAISSIndex":
        """Load index from disk.

        Args:
            index_dir: Directory containing index files

        Returns:
            Loaded FAISSIndex
        """
        # Load metadata
        metadata_path = index_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Create instance
        instance = cls(
            dimension=metadata["dimension"],
            index_type=metadata["index_type"],
            nlist=metadata["nlist"],
            metric=metadata["metric"],
        )

        # Load FAISS index
        index_path = index_dir / "index.faiss"
        instance.index = faiss.read_index(str(index_path))
        instance.is_trained = metadata["is_trained"]

        # Load docstore
        docstore_path = index_dir / "docstore.pkl"
        with open(docstore_path, "rb") as f:
            instance.docstore = pickle.load(f)

        logger.info(f"Loaded index with {instance.index.ntotal} vectors from {index_dir}")
        return instance
