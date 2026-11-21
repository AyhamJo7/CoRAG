"""Internal endpoints for the Next.js BFF (never exposed to browsers).

The BFF's catch-all proxy blocks `/internal/*`, and every route here
additionally requires the shared internal token.
"""

import logging
from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr, Field

from corag_cloud.billing.plans import is_valid_plan
from corag_cloud.billing.sync import apply_stripe_event
from corag_cloud.config import Settings, get_settings
from corag_cloud.db.pool import plain_connection, tenant_connection
from corag_cloud.deps import require_internal
from corag_cloud.provision import (
    EmailAlreadyExists,
    provision_tenant_with_owner,
)
from corag_cloud.security import verify_password

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/internal", dependencies=[Depends(require_internal)])

MAX_PASSWORD_LENGTH = 256


class ProvisionRequest(BaseModel):
    workspace: str = Field(default="", max_length=120)
    name: str = Field(min_length=1, max_length=120)
    email: EmailStr
    password: str = Field(min_length=10, max_length=MAX_PASSWORD_LENGTH)


class ProvisionResponse(BaseModel):
    user_id: UUID
    tenant_id: UUID


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=MAX_PASSWORD_LENGTH)


class TenantMembership(BaseModel):
    id: UUID
    name: str
    role: str
    plan: str


class LoginResponse(BaseModel):
    user_id: UUID
    email: str
    name: str
    session_version: int
    tenants: list[TenantMembership]


@router.post("/provision", response_model=ProvisionResponse, status_code=201)
async def provision(
    body: ProvisionRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> ProvisionResponse:
    """Create a workspace + owner atomically (admin connection)."""
    if not settings.database_admin_url:
        raise HTTPException(status_code=503, detail="Provisioning is not configured")
    try:
        result = await provision_tenant_with_owner(
            settings.database_admin_url,
            workspace=body.workspace,
            email=body.email,
            name=body.name,
            password=body.password,
        )
    except EmailAlreadyExists:
        raise HTTPException(
            status_code=409, detail="This email is already registered"
        ) from None
    logger.info("Provisioned tenant %s", result.tenant_id)
    return ProvisionResponse(user_id=result.user_id, tenant_id=result.tenant_id)


@router.post("/login", response_model=LoginResponse)
async def login(body: LoginRequest) -> LoginResponse:
    """Verify credentials and return the user's tenant memberships."""
    async with plain_connection() as conn:
        user = await conn.fetchrow(
            "SELECT id, email, name, password_hash, session_version "
            "FROM app_user WHERE email = $1",
            body.email.strip().lower(),
        )
        if user is None or not verify_password(body.password, user["password_hash"]):
            # Same response for unknown email and wrong password.
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # SECURITY DEFINER helper: the tenant table is FORCE-RLS'd and no
        # tenant context exists yet at login.
        rows = await conn.fetch("SELECT * FROM user_memberships($1)", user["id"])

    return LoginResponse(
        user_id=user["id"],
        email=user["email"],
        name=user["name"],
        session_version=user["session_version"],
        tenants=[
            TenantMembership(id=r["id"], name=r["name"], role=r["role"], plan=r["plan"])
            for r in rows
        ],
    )


class BillingSyncRequest(BaseModel):
    event_id: str = Field(min_length=1, max_length=255)
    event_type: str = Field(min_length=1, max_length=100)
    tenant_id: UUID
    plan: str | None = None
    stripe_customer_id: str | None = None
    stripe_subscription_id: str | None = None
    subscription_status: str | None = None
    period_end: datetime | None = None
    cancel: bool = False


class BillingSyncResponse(BaseModel):
    duplicate: bool
    updated: bool


@router.post("/billing/sync", response_model=BillingSyncResponse)
async def billing_sync(body: BillingSyncRequest) -> BillingSyncResponse:
    """Apply a verified Stripe event to tenant billing state (idempotent)."""
    if body.plan is not None and not is_valid_plan(body.plan):
        raise HTTPException(status_code=422, detail="Unknown plan")
    outcome = await apply_stripe_event(
        event_id=body.event_id,
        event_type=body.event_type,
        tenant_id=body.tenant_id,
        plan=body.plan,
        stripe_customer_id=body.stripe_customer_id,
        stripe_subscription_id=body.stripe_subscription_id,
        subscription_status=body.subscription_status,
        period_end=body.period_end,
        cancel=body.cancel,
    )
    return BillingSyncResponse(duplicate=outcome.duplicate, updated=outcome.updated)


class BillingStateResponse(BaseModel):
    plan: str
    subscription_status: str | None
    stripe_customer_id: str | None


@router.get("/billing/state", response_model=BillingStateResponse)
async def billing_state(tenant_id: UUID) -> BillingStateResponse:
    """Billing state for the portal route (server-side only)."""
    async with tenant_connection(tenant_id) as conn:
        tenant = await conn.fetchrow(
            "SELECT plan, subscription_status, stripe_customer_id "
            "FROM tenant WHERE id = $1",
            tenant_id,
        )
    if tenant is None:
        raise HTTPException(status_code=404, detail="Unknown tenant")
    return BillingStateResponse(**dict(tenant))
