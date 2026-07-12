"""Request dependencies: BFF trust boundary and request context."""

import hmac
from dataclasses import dataclass
from typing import Annotated
from uuid import UUID

from fastapi import Depends, Header, HTTPException

from corag_cloud.config import Settings, get_settings


@dataclass(frozen=True)
class RequestContext:
    """Identity forwarded by the Next.js BFF for an authenticated request."""

    user_id: UUID
    tenant_id: UUID
    request_id: str | None = None


def require_internal(
    settings: Annotated[Settings, Depends(get_settings)],
    x_internal_token: Annotated[str, Header()] = "",
) -> None:
    """Reject requests that do not carry the BFF shared secret."""
    if not hmac.compare_digest(x_internal_token, settings.internal_service_token):
        raise HTTPException(status_code=401, detail="Invalid internal token")


def get_request_context(
    _: Annotated[None, Depends(require_internal)],
    x_user_id: Annotated[str, Header()] = "",
    x_tenant_id: Annotated[str, Header()] = "",
    x_request_id: Annotated[str | None, Header()] = None,
) -> RequestContext:
    """Build the per-request identity from BFF-forwarded headers."""
    try:
        return RequestContext(
            user_id=UUID(x_user_id),
            tenant_id=UUID(x_tenant_id),
            request_id=x_request_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid identity headers") from e
