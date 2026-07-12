"""API-key management for the dashboard (via BFF)."""

import logging
from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from corag_cloud.api_keys import MAX_KEYS_PER_TENANT, generate_api_key
from corag_cloud.config import Settings, get_settings
from corag_cloud.db.pool import tenant_connection
from corag_cloud.deps import RequestContext, get_request_context

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api-keys")

ContextDep = Annotated[RequestContext, Depends(get_request_context)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


class ApiKeyOut(BaseModel):
    id: UUID
    name: str
    key_prefix: str
    created_at: datetime
    last_used_at: datetime | None
    revoked_at: datetime | None


class ApiKeyCreated(ApiKeyOut):
    # Present exactly once, at creation.
    key: str


class CreateKeyRequest(BaseModel):
    name: str = Field(default="", max_length=100)


@router.post("", response_model=ApiKeyCreated, status_code=201)
async def create_key(
    ctx: ContextDep, settings: SettingsDep, body: CreateKeyRequest
) -> ApiKeyCreated:
    generated = generate_api_key(settings.api_key_salt)
    async with tenant_connection(ctx.tenant_id) as conn:
        active = await conn.fetchval(
            "SELECT count(*) FROM api_key WHERE tenant_id = $1 AND revoked_at IS NULL",
            ctx.tenant_id,
        )
        if active >= MAX_KEYS_PER_TENANT:
            raise HTTPException(
                status_code=409,
                detail=f"At most {MAX_KEYS_PER_TENANT} active keys per workspace",
            )
        row = await conn.fetchrow(
            "INSERT INTO api_key (tenant_id, name, key_hash, key_prefix) "
            "VALUES ($1, $2, $3, $4) "
            "RETURNING id, name, key_prefix, created_at, last_used_at, revoked_at",
            ctx.tenant_id,
            body.name,
            generated.key_hash,
            generated.key_prefix,
        )
    assert row is not None
    logger.info("API key %s created for tenant %s", row["id"], ctx.tenant_id)
    return ApiKeyCreated(**dict(row), key=generated.plaintext)


@router.get("", response_model=list[ApiKeyOut])
async def list_keys(ctx: ContextDep) -> list[ApiKeyOut]:
    async with tenant_connection(ctx.tenant_id) as conn:
        rows = await conn.fetch(
            "SELECT id, name, key_prefix, created_at, last_used_at, revoked_at "
            "FROM api_key WHERE tenant_id = $1 ORDER BY created_at DESC",
            ctx.tenant_id,
        )
    return [ApiKeyOut(**dict(r)) for r in rows]


@router.delete("/{key_id}", status_code=204)
async def revoke_key(ctx: ContextDep, key_id: UUID) -> None:
    async with tenant_connection(ctx.tenant_id) as conn:
        row = await conn.fetchrow(
            "UPDATE api_key SET revoked_at = now() "
            "WHERE id = $1 AND tenant_id = $2 AND revoked_at IS NULL RETURNING id",
            key_id,
            ctx.tenant_id,
        )
    if row is None:
        raise HTTPException(status_code=404, detail="Key not found")
