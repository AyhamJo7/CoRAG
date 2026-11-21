"""Tenant-scoped pgvector retrieval satisfying corag's SearchIndex protocol.

Synchronous by design: RetrievalPipeline is sync and runs inside a worker
thread. Each search opens a short-lived psycopg connection, binds the RLS
tenant context transaction-locally, and queries the HNSW index. RLS is the
real tenant fence; the WHERE clause is for planner efficiency.
"""

import logging
from uuid import UUID

import numpy as np
import psycopg

from corag.corpus.document import Chunk

logger = logging.getLogger(__name__)


def _vector_literal(embedding: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.7f}" for x in embedding.tolist()) + "]"


class PgVectorRetriever:
    """Nearest-neighbour search over one tenant's chunks."""

    def __init__(self, dsn: str, tenant_id: UUID):
        self.dsn = dsn
        self.tenant_id = tenant_id

    def search(
        self, query_embedding: np.ndarray, k: int = 10
    ) -> tuple[list[Chunk], list[float]]:
        """Return the k most similar chunks for a query embedding."""
        if query_embedding.ndim > 1:
            query_embedding = query_embedding[0]
        vector = _vector_literal(query_embedding)

        with psycopg.connect(self.dsn) as conn, conn.transaction():
            conn.execute(
                "SELECT set_config('app.tenant_id', %s, true)",
                (str(self.tenant_id),),
            )
            rows = conn.execute(
                """
                SELECT c.id, c.document_id, c.text, c.tokens,
                       c.start_char, c.end_char, d.title,
                       1 - (c.embedding <=> %s::vector) AS score
                FROM chunk c
                JOIN document d ON d.id = c.document_id
                WHERE c.tenant_id = %s AND c.embedding IS NOT NULL
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
                """,
                (vector, self.tenant_id, vector, k),
            ).fetchall()

        chunks = [
            Chunk(
                chunk_id=f"chunk-{row[0]}",
                doc_id=str(row[1]),
                text=row[2],
                start_char=row[4],
                end_char=row[5],
                tokens=row[3],
                doc_title=row[6],
            )
            for row in rows
        ]
        scores = [float(row[7]) for row in rows]
        return chunks, scores
