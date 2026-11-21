"""FORCE-RLS isolation proof, run under the restricted app role."""

import uuid

import asyncpg
import pytest

from corag_cloud.provision import provision_tenant_with_owner

from .conftest import requires_db

pytestmark = requires_db


async def _provision(admin_dsn: str, label: str) -> uuid.UUID:
    result = await provision_tenant_with_owner(
        admin_dsn,
        workspace=f"ws-{label}",
        email=f"{label}-{uuid.uuid4().hex[:8]}@example.com",
        name=f"User {label}",
        password="correct-horse-battery",
    )
    return result.tenant_id


async def _set_tenant(conn: asyncpg.Connection, tenant_id: uuid.UUID) -> None:
    await conn.execute("SELECT set_config('app.tenant_id', $1, true)", str(tenant_id))


async def test_tenant_rows_are_isolated(migrated_db, app_conn):
    tenant_a = await _provision(migrated_db, "iso-a")
    tenant_b = await _provision(migrated_db, "iso-b")

    # Insert a document as tenant A.
    async with app_conn.transaction():
        await _set_tenant(app_conn, tenant_a)
        await app_conn.execute(
            "INSERT INTO document (tenant_id, title, filename, mime, size_bytes) "
            "VALUES ($1, 'secret plans', 'plans.pdf', 'application/pdf', 123)",
            tenant_a,
        )

    # Tenant B sees nothing.
    async with app_conn.transaction():
        await _set_tenant(app_conn, tenant_b)
        rows = await app_conn.fetch("SELECT * FROM document")
        assert rows == []
        visible_tenants = await app_conn.fetch("SELECT id FROM tenant")
        assert [r["id"] for r in visible_tenants] == [tenant_b]

    # Tenant A sees exactly its own row.
    async with app_conn.transaction():
        await _set_tenant(app_conn, tenant_a)
        rows = await app_conn.fetch("SELECT title FROM document")
        assert [r["title"] for r in rows] == ["secret plans"]


async def test_cross_tenant_insert_is_rejected(migrated_db, app_conn):
    tenant_a = await _provision(migrated_db, "chk-a")
    tenant_b = await _provision(migrated_db, "chk-b")

    with pytest.raises(asyncpg.InsufficientPrivilegeError):
        async with app_conn.transaction():
            await _set_tenant(app_conn, tenant_a)
            # WITH CHECK must reject a row stamped for another tenant.
            await app_conn.execute(
                "INSERT INTO document (tenant_id, title, filename, mime, size_bytes) "
                "VALUES ($1, 'smuggled', 'x.txt', 'text/plain', 1)",
                tenant_b,
            )


async def test_no_tenant_context_sees_nothing(migrated_db, app_conn):
    await _provision(migrated_db, "noctx")

    # Without app.tenant_id set, current_setting() errors inside the policy,
    # so any access to a FORCE-RLS table fails outright for the app role.
    with pytest.raises(asyncpg.PostgresError):
        await app_conn.fetch("SELECT * FROM document")


async def test_app_role_cannot_bypass_rls(app_conn):
    role = await app_conn.fetchrow(
        "SELECT rolbypassrls, rolsuper FROM pg_roles WHERE rolname = current_user"
    )
    assert role["rolbypassrls"] is False
    assert role["rolsuper"] is False
