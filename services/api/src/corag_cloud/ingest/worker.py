"""Ingestion worker: claims queued jobs, chunks, embeds, stores vectors.

Runs as its own process (compose service ``worker``). Job pickup uses
``FOR UPDATE SKIP LOCKED`` on the ingest_job table — exactly-once claiming
with no extra infrastructure. All tenant-data writes happen inside a
tenant-scoped transaction; the job row itself is not RLS'd (ids/status only).
"""

import asyncio
import logging
import signal
from typing import Any
from uuid import UUID

from corag.corpus.chunker import Chunker
from corag.corpus.document import Document
from corag_cloud.config import get_settings
from corag_cloud.db.pool import close_pool, open_pool, plain_connection
from corag_cloud.retrieval.openai_embedder import OpenAIEmbedder

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 2.0
MAX_ATTEMPTS = 3

CLAIM_SQL = """
UPDATE ingest_job
SET status = 'running', attempts = attempts + 1, updated_at = now()
WHERE id = (
    SELECT id FROM ingest_job
    WHERE status = 'queued'
    ORDER BY id
    LIMIT 1
    FOR UPDATE SKIP LOCKED
)
RETURNING id, tenant_id, document_id, attempts
"""


class IngestWorker:
    """Claims and processes ingest jobs until stopped."""

    def __init__(self, embedder: Any, chunker: Chunker | None = None):
        self.embedder = embedder
        self.chunker = chunker or Chunker(
            chunk_size=512, chunk_overlap=64, min_chunk_size=20
        )
        self._stop = asyncio.Event()

    def request_stop(self) -> None:
        self._stop.set()

    async def run_forever(self) -> None:
        logger.info("Ingest worker started")
        while not self._stop.is_set():
            processed = await self.process_one()
            if not processed:
                try:
                    await asyncio.wait_for(
                        self._stop.wait(), timeout=POLL_INTERVAL_SECONDS
                    )
                except TimeoutError:
                    pass
        logger.info("Ingest worker stopped")

    async def process_one(self) -> bool:
        """Claim and process a single job. Returns False when queue is empty."""
        async with plain_connection() as conn:
            job = await conn.fetchrow(CLAIM_SQL)
        if job is None:
            return False

        job_id: int = job["id"]
        tenant_id: UUID = job["tenant_id"]
        document_id: UUID = job["document_id"]

        try:
            await self._index_document(tenant_id, document_id)
        except Exception as e:
            logger.exception("Job %d failed (attempt %d)", job_id, job["attempts"])
            await self._mark_failure(job_id, tenant_id, document_id, str(e), job)
        else:
            async with plain_connection() as conn:
                await conn.execute(
                    "UPDATE ingest_job SET status = 'done', updated_at = now() "
                    "WHERE id = $1",
                    job_id,
                )
            logger.info("Job %d done (document %s)", job_id, document_id)
        return True

    async def _index_document(self, tenant_id: UUID, document_id: UUID) -> None:
        from corag_cloud.db.pool import tenant_connection

        async with tenant_connection(tenant_id) as conn:
            row = await conn.fetchrow(
                "SELECT id, title, text FROM document WHERE id = $1", document_id
            )
            if row is None:
                raise RuntimeError(f"document {document_id} not found")
            await conn.execute(
                "UPDATE document SET status = 'processing', error = NULL WHERE id = $1",
                document_id,
            )

        # Chunk + embed outside the transaction (API calls can be slow).
        doc = Document(id=str(document_id), title=row["title"], text=row["text"])
        chunks = self.chunker.chunk_document(doc)
        embeddings = (
            self.embedder.embed_texts([c.text for c in chunks]) if chunks else None
        )

        async with tenant_connection(tenant_id) as conn:
            await conn.execute("DELETE FROM chunk WHERE document_id = $1", document_id)
            if chunks and embeddings is not None:
                await conn.executemany(
                    "INSERT INTO chunk (tenant_id, document_id, chunk_index, text, "
                    "tokens, start_char, end_char, embedding) "
                    "VALUES ($1, $2, $3, $4, $5, $6, $7, $8::vector)",
                    [
                        (
                            tenant_id,
                            document_id,
                            i,
                            c.text,
                            c.tokens,
                            c.start_char,
                            c.end_char,
                            "[" + ",".join(f"{x:.7f}" for x in emb) + "]",
                        )
                        for i, (c, emb) in enumerate(
                            zip(chunks, embeddings, strict=True)
                        )
                    ],
                )
            await conn.execute(
                "UPDATE document SET status = 'indexed' WHERE id = $1", document_id
            )

    async def _mark_failure(
        self,
        job_id: int,
        tenant_id: UUID,
        document_id: UUID,
        error: str,
        job: Any,
    ) -> None:
        from corag_cloud.db.pool import tenant_connection

        exhausted = job["attempts"] >= MAX_ATTEMPTS
        status = "failed" if exhausted else "queued"
        async with plain_connection() as conn:
            await conn.execute(
                "UPDATE ingest_job SET status = $2, last_error = $3, "
                "updated_at = now() WHERE id = $1",
                job_id,
                status,
                error[:2000],
            )
        if exhausted:
            async with tenant_connection(tenant_id) as conn:
                await conn.execute(
                    "UPDATE document SET status = 'failed', error = $2 WHERE id = $1",
                    document_id,
                    error[:500],
                )


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    settings = get_settings()
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is required for the ingest worker")
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for the ingest worker")

    await open_pool(settings.database_url, max_size=2)
    embedder = OpenAIEmbedder(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
    )
    worker = IngestWorker(embedder)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, worker.request_stop)

    try:
        await worker.run_forever()
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
