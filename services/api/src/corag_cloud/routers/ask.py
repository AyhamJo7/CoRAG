"""The /ask endpoint: quota-gated, SSE-streamed cited answers.

``ask_response`` is shared with the developer API (/v1/ask) — same gates,
rate limit, streaming shape, and usage accounting for both entry points.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Annotated
from uuid import UUID

import anyio.to_thread
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from corag_cloud.billing.access import require_active_access, require_question_quota
from corag_cloud.config import get_settings
from corag_cloud.db.pool import tenant_connection
from corag_cloud.deps import RequestContext, get_request_context
from corag_cloud.rate_limit import FixedWindowLimiter
from corag_cloud.service.ask_service import run_ask

logger = logging.getLogger(__name__)

router = APIRouter()

ContextDep = Annotated[RequestContext, Depends(get_request_context)]

MAX_QUESTION_LENGTH = 2000
KEEPALIVE_SECONDS = 10.0


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)


def _event(name: str, payload: object) -> str:
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"


@lru_cache
def _limiter() -> FixedWindowLimiter:
    return FixedWindowLimiter(get_settings().ask_rate_limit_per_minute)


async def ask_response(
    tenant_id: UUID, actor_id: UUID, question: str
) -> StreamingResponse:
    """Gate, run the pipeline in a worker thread, and stream SSE events.

    ``actor_id`` lands in question_log.user_id: the user id for dashboard
    asks, the api_key id for developer-API asks.
    """
    # Gate before streaming starts so limit errors surface as clean 4xx.
    _limiter().check(tenant_id)
    async with tenant_connection(tenant_id) as conn:
        tenant = await conn.fetchrow("SELECT * FROM tenant WHERE id = $1", tenant_id)
        assert tenant is not None
        require_active_access(tenant)
        require_question_quota(tenant)

    async def stream() -> AsyncIterator[str]:
        yield _event("status", {"stage": "retrieving"})
        task = asyncio.create_task(
            anyio.to_thread.run_sync(run_ask, tenant_id, question)
        )
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=KEEPALIVE_SECONDS)
                if done:
                    break
                yield ": keepalive\n\n"
            result = task.result()
        except Exception:
            logger.exception("ask failed for tenant %s", tenant_id)
            yield _event(
                "error", {"message": "Something went wrong answering this question"}
            )
            return

        async with tenant_connection(tenant_id) as conn:
            await conn.execute(
                "UPDATE tenant SET questions_used = questions_used + 1 WHERE id = $1",
                tenant_id,
            )
            await conn.execute(
                "INSERT INTO question_log (tenant_id, user_id, question, answer, "
                "citations, num_steps, num_chunks, latency_ms) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
                tenant_id,
                actor_id,
                question,
                result.answer,
                json.dumps(result.citations),
                result.num_steps,
                result.num_chunks,
                result.latency_ms,
            )

        yield _event("answer", {"text": result.answer})
        yield _event("citations", {"items": result.citations})
        yield _event(
            "done",
            {
                "num_steps": result.num_steps,
                "num_chunks": result.num_chunks,
                "latency_ms": result.latency_ms,
            },
        )

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
    )


@router.post("/ask")
async def ask(ctx: ContextDep, body: AskRequest) -> StreamingResponse:
    """Answer a question over the tenant's indexed documents."""
    return await ask_response(ctx.tenant_id, ctx.user_id, body.question)
