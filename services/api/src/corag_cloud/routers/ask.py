"""The /ask endpoint: quota-gated, SSE-streamed cited answers."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Annotated

import anyio.to_thread
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from corag_cloud.billing.access import require_active_access, require_question_quota
from corag_cloud.db.pool import tenant_connection
from corag_cloud.deps import RequestContext, get_request_context
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


@router.post("/ask")
async def ask(ctx: ContextDep, body: AskRequest) -> StreamingResponse:
    """Answer a question over the tenant's indexed documents."""
    # Gate before streaming starts so quota errors surface as clean 402s.
    async with tenant_connection(ctx.tenant_id) as conn:
        tenant = await conn.fetchrow(
            "SELECT * FROM tenant WHERE id = $1", ctx.tenant_id
        )
        assert tenant is not None
        require_active_access(tenant)
        require_question_quota(tenant)

    async def stream() -> AsyncIterator[str]:
        yield _event("status", {"stage": "retrieving"})
        task = asyncio.create_task(
            anyio.to_thread.run_sync(run_ask, ctx.tenant_id, body.question)
        )
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=KEEPALIVE_SECONDS)
                if done:
                    break
                yield ": keepalive\n\n"
            result = task.result()
        except Exception:
            logger.exception("ask failed for tenant %s", ctx.tenant_id)
            yield _event(
                "error", {"message": "Something went wrong answering this question"}
            )
            return

        async with tenant_connection(ctx.tenant_id) as conn:
            await conn.execute(
                "UPDATE tenant SET questions_used = questions_used + 1 WHERE id = $1",
                ctx.tenant_id,
            )
            await conn.execute(
                "INSERT INTO question_log (tenant_id, user_id, question, answer, "
                "citations, num_steps, num_chunks, latency_ms) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
                ctx.tenant_id,
                ctx.user_id,
                body.question,
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
