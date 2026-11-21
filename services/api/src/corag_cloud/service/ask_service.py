"""Runs the CoRAG pipeline over a tenant's pgvector corpus.

Synchronous — the SSE endpoint dispatches it to a worker thread. LLM and
embedding clients are cached per process; the retriever is per-request
(it carries the tenant id).
"""

import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from uuid import UUID

from corag.controller.base import GenerationConfig
from corag.controller.openai_controller import OpenAIController
from corag.generation.synthesizer import Synthesizer
from corag.retrieval.pipeline import RetrievalPipeline
from corag_cloud.config import Settings, get_settings
from corag_cloud.retrieval.openai_embedder import OpenAIEmbedder
from corag_cloud.retrieval.pgvector_index import PgVectorRetriever

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AskResult:
    answer: str
    citations: list[dict[str, str]]
    num_steps: int
    num_chunks: int
    latency_ms: int


@lru_cache
def _controller(api_key: str, model: str) -> OpenAIController:
    return OpenAIController(api_key=api_key, model=model)


@lru_cache
def _embedder(api_key: str, model: str, dimensions: int) -> OpenAIEmbedder:
    return OpenAIEmbedder(api_key=api_key, model=model, dimensions=dimensions)


def run_ask(
    tenant_id: UUID, question: str, settings: Settings | None = None
) -> AskResult:
    """Decompose, retrieve iteratively, and synthesize a cited answer."""
    settings = settings or get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    started = time.time()
    controller = _controller(settings.openai_api_key, settings.llm_model)
    embedder = _embedder(
        settings.openai_api_key,
        settings.embedding_model,
        settings.embedding_dimensions,
    )
    retriever = PgVectorRetriever(settings.database_url, tenant_id)
    config = GenerationConfig(temperature=0.2, max_tokens=settings.ask_max_tokens)

    pipeline = RetrievalPipeline(
        index=retriever,
        embedder=embedder,
        controller=controller,
        k=settings.ask_k,
        max_steps=settings.ask_max_steps,
        config=config,
    )
    state = pipeline.run(question)

    synthesizer = Synthesizer(controller=controller, config=config)
    answer, citations = synthesizer.synthesize(state)

    return AskResult(
        answer=answer,
        citations=citations,
        num_steps=len(state.steps),
        num_chunks=state.total_chunks_retrieved,
        latency_ms=int((time.time() - started) * 1000),
    )
