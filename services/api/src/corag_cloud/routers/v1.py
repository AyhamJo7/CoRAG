"""Public developer API (exposed at corag.iquantum.co/v1 via tunnel ingress).

Authenticated by ``Authorization: Bearer corag_live_...`` — never by the
BFF internal token. Only this router is reachable from outside; the tunnel
routes ``^/v1/.*`` here and everything else to the web app.
"""

import logging
from dataclasses import dataclass
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse

from corag_cloud.api_keys import KEY_PREFIX, hash_api_key
from corag_cloud.config import Settings, get_settings
from corag_cloud.db.pool import plain_connection
from corag_cloud.routers.ask import AskRequest, ask_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1")


@dataclass(frozen=True)
class ApiKeyContext:
    key_id: UUID
    tenant_id: UUID


async def get_api_key_context(
    settings: Annotated[Settings, Depends(get_settings)],
    authorization: Annotated[str, Header()] = "",
) -> ApiKeyContext:
    """Resolve a bearer API key to its tenant; 401 on any mismatch."""
    scheme, _, credential = authorization.partition(" ")
    if scheme.lower() != "bearer" or not credential.startswith(KEY_PREFIX):
        raise HTTPException(status_code=401, detail="Invalid API key")

    key_hash = hash_api_key(credential.strip(), settings.api_key_salt)
    async with plain_connection() as conn:
        row = await conn.fetchrow(
            "UPDATE api_key SET last_used_at = now() "
            "WHERE key_hash = $1 AND revoked_at IS NULL "
            "RETURNING id, tenant_id",
            key_hash,
        )
    if row is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return ApiKeyContext(key_id=row["id"], tenant_id=row["tenant_id"])


KeyDep = Annotated[ApiKeyContext, Depends(get_api_key_context)]


@router.post("/ask")
async def v1_ask(key: KeyDep, body: AskRequest) -> StreamingResponse:
    """Ask a question over the workspace's documents (SSE response)."""
    return await ask_response(key.tenant_id, key.key_id, body.question)
