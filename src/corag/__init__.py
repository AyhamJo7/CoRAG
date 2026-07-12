"""CoRAG: Adaptive Multi-Step Retrieval for Complex Queries."""

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"
__author__ = "AyhamJo77"
__email__ = "mhd.ayham.joumran@studium.uni-hamburg.de"

from corag.controller.base import Controller
from corag.corpus.chunker import Chunker
from corag.corpus.document import Document
from corag.generation.synthesizer import Synthesizer
from corag.retrieval.interfaces import QueryEmbedder, SearchIndex
from corag.retrieval.pipeline import RetrievalPipeline

if TYPE_CHECKING:
    from corag.indexing.embedder import Embedder
    from corag.indexing.index import FAISSIndex

# Embedder/FAISSIndex need the heavy research extra (torch, faiss); they are
# loaded lazily so the base install stays import-safe without them.
_RESEARCH_EXPORTS = ("Embedder", "FAISSIndex")


def __getattr__(name: str) -> Any:
    if name in _RESEARCH_EXPORTS:
        from corag import indexing

        return getattr(indexing, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Document",
    "Chunker",
    "Embedder",
    "FAISSIndex",
    "Controller",
    "QueryEmbedder",
    "SearchIndex",
    "RetrievalPipeline",
    "Synthesizer",
]
