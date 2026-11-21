"""Self-serve workspace provisioning.

Creates a workspace and its owner in one atomic transaction: ``tenant`` +
``app_user`` + an ``owner`` membership. Runs on a dedicated ADMIN connection —
creating a ``tenant`` row needs privileges the restricted app role
deliberately lacks. Only the internal-token-guarded provisioning endpoint may
call this.
"""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

import asyncpg

from corag_cloud.security import hash_password

TRIAL_DAYS = 14


class EmailAlreadyExists(Exception):
    """Raised when the email is already registered — surfaced as HTTP 409."""


@dataclass(frozen=True)
class ProvisionResult:
    user_id: UUID
    tenant_id: UUID


async def provision_tenant_with_owner(
    admin_dsn: str,
    *,
    workspace: str,
    email: str,
    name: str,
    password: str,
) -> ProvisionResult:
    """Atomically create a workspace + its owner.

    Raises ``EmailAlreadyExists`` if the email is taken (checked inside the
    transaction; the unique index on ``app_user.email`` is the race-safe
    backstop).
    """
    workspace = workspace.strip()
    email = email.strip().lower()
    name = name.strip()
    if not email or not name:
        raise ValueError("email and name are required")
    if not workspace:
        workspace = f"{name}'s workspace"

    conn = await asyncpg.connect(admin_dsn)
    try:
        async with conn.transaction():
            existing = await conn.fetchrow(
                "SELECT 1 FROM app_user WHERE email = $1", email
            )
            if existing is not None:
                raise EmailAlreadyExists(email)

            tenant_id = uuid.uuid4()
            trial_ends_at = datetime.now(UTC) + timedelta(days=TRIAL_DAYS)
            await conn.execute(
                "INSERT INTO tenant (id, name, plan, trial_ends_at) "
                "VALUES ($1, $2, 'trial', $3)",
                tenant_id,
                workspace,
                trial_ends_at,
            )

            user_id = uuid.uuid4()
            try:
                await conn.execute(
                    "INSERT INTO app_user (id, email, name, password_hash) "
                    "VALUES ($1, $2, $3, $4)",
                    user_id,
                    email,
                    name,
                    hash_password(password),
                )
            except asyncpg.UniqueViolationError as e:
                raise EmailAlreadyExists(email) from e
            await conn.execute(
                "INSERT INTO user_tenant (user_id, tenant_id, role) "
                "VALUES ($1, $2, 'owner')",
                user_id,
                tenant_id,
            )
    finally:
        await conn.close()

    return ProvisionResult(user_id=user_id, tenant_id=tenant_id)
