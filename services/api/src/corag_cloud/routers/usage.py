"""Usage and plan-limit reporting for the dashboard."""

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from corag_cloud.billing.plans import limits_for
from corag_cloud.db.pool import tenant_connection
from corag_cloud.deps import RequestContext, get_request_context

router = APIRouter()

ContextDep = Annotated[RequestContext, Depends(get_request_context)]


class UsageOut(BaseModel):
    plan: str
    subscription_status: str | None
    trial_ends_at: datetime | None
    current_period_end: datetime | None
    questions_used: int
    questions_limit: int
    docs_count: int
    docs_limit: int
    storage_bytes_used: int
    storage_bytes_limit: int


@router.get("/usage", response_model=UsageOut)
async def usage(ctx: ContextDep) -> UsageOut:
    async with tenant_connection(ctx.tenant_id) as conn:
        tenant = await conn.fetchrow(
            "SELECT * FROM tenant WHERE id = $1", ctx.tenant_id
        )
    assert tenant is not None
    limits = limits_for(tenant["plan"])
    return UsageOut(
        plan=tenant["plan"],
        subscription_status=tenant["subscription_status"],
        trial_ends_at=tenant["trial_ends_at"],
        current_period_end=tenant["current_period_end"],
        questions_used=tenant["questions_used"],
        questions_limit=limits.questions_per_month,
        docs_count=tenant["docs_count"],
        docs_limit=limits.max_documents,
        storage_bytes_used=tenant["storage_bytes_used"],
        storage_bytes_limit=limits.max_storage_bytes,
    )
