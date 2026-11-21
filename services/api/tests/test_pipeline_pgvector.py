"""End-to-end proof: RetrievalPipeline over pgvector with zero OpenAI calls.

Seeds two tenants' chunks with deterministic stub embeddings and drives the
full decompose → retrieve → critique loop through the Protocol seam.
"""

import json
import uuid

import asyncpg

from corag.controller.base import Controller, GenerationConfig
from corag.retrieval.pipeline import RetrievalPipeline
from corag_cloud.provision import provision_tenant_with_owner
from corag_cloud.retrieval.pgvector_index import PgVectorRetriever, _vector_literal

from .conftest import APP_URL, requires_db
from .stubs import StubEmbedder

pytestmark = requires_db


class ScriptedController(Controller):
    def __init__(self, responses: list[str]):
        self.responses = list(responses)

    def generate(self, prompt, system=None, config=None):
        return self.responses.pop(0)

    def count_tokens(self, text: str) -> int:
        return len(text.split())


FACTS = {
    "founding": "CoRAG GmbH was founded in Hamburg in 2026 by a retrieval team.",
    "product": "The flagship product answers multi-hop questions with citations.",
    "office": "The Hamburg office sits near the Elbe river in Altona.",
}


async def _seed_tenant(admin_dsn: str, embedder: StubEmbedder, label: str):
    result = await provision_tenant_with_owner(
        admin_dsn,
        workspace=f"pipe-{label}",
        email=f"pipe-{label}-{uuid.uuid4().hex[:8]}@example.com",
        name="Pipeline Seeder",
        password="a-long-password",
    )
    conn = await asyncpg.connect(admin_dsn)
    try:
        for key, text in FACTS.items():
            doc_id = await conn.fetchval(
                "INSERT INTO document (tenant_id, title, filename, mime, "
                "size_bytes, status, text) "
                "VALUES ($1, $2, $3, 'text/plain', 1, 'indexed', $4) RETURNING id",
                result.tenant_id,
                f"{label}-{key}",
                f"{key}.txt",
                text,
            )
            await conn.execute(
                "INSERT INTO chunk (tenant_id, document_id, chunk_index, text, "
                "tokens, start_char, end_char, embedding) "
                "VALUES ($1, $2, 0, $3, 10, 0, 100, $4::vector)",
                result.tenant_id,
                doc_id,
                text,
                _vector_literal(embedder.embed_query(text)),
            )
    finally:
        await conn.close()
    return result.tenant_id


async def test_pipeline_retrieves_seeded_chunks(migrated_db):
    embedder = StubEmbedder()
    tenant_id = await _seed_tenant(migrated_db, embedder, "a")

    decomposition = json.dumps({"sub_queries": [FACTS["founding"], FACTS["office"]]})
    critique = json.dumps({"is_sufficient": True, "confidence": 0.9})
    pipeline = RetrievalPipeline(
        index=PgVectorRetriever(APP_URL, tenant_id),
        embedder=embedder,
        controller=ScriptedController([decomposition, critique]),
        k=2,
        max_steps=4,
        config=GenerationConfig(),
    )

    state = pipeline.run("Where and when was CoRAG founded?")

    assert state.is_sufficient is True
    assert len(state.steps) == 2
    # Stub embeddings are exact for identical text: top hit must match.
    assert state.steps[0].chunks[0].text == FACTS["founding"]
    assert state.steps[1].chunks[0].text == FACTS["office"]
    assert state.steps[0].chunks[0].doc_title == "a-founding"
    assert all(s.scores[0] > 0.99 for s in state.steps)


async def test_pipeline_sees_nothing_across_tenants(migrated_db):
    embedder = StubEmbedder()
    await _seed_tenant(migrated_db, embedder, "rich")
    empty_tenant = await provision_tenant_with_owner(
        migrated_db,
        workspace="empty-ws",
        email=f"empty-{uuid.uuid4().hex[:8]}@example.com",
        name="Empty Tenant",
        password="a-long-password",
    )

    retriever = PgVectorRetriever(APP_URL, empty_tenant.tenant_id)
    chunks, scores = retriever.search(embedder.embed_query(FACTS["founding"]), k=5)

    assert chunks == []
    assert scores == []
