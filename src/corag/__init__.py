"""CoRAG: Adaptive Multi-Step Retrieval for Complex Queries."""

__version__ = "0.1.0"
__author__ = "AyhamJo77"
__email__ = "mhd.ayham.joumran@studium.uni-hamburg.de"

from corag.corpus.document import Document
from corag.corpus.chunker import Chunker
from corag.indexing.embedder import Embedder
from corag.indexing.index import FAISSIndex
from corag.controller.base import Controller
from corag.retrieval.pipeline import RetrievalPipeline
from corag.generation.synthesizer import Synthesizer

__all__ = [
    "Document",
    "Chunker",
    "Embedder",
    "FAISSIndex",
    "Controller",
    "RetrievalPipeline",
    "Synthesizer",
]
