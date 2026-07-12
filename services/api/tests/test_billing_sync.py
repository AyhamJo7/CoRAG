"""Stripe event sync: idempotency, entitlement SET, period-reset semantics."""

import uuid
from datetime import UTC, datetime, timedelta

import asyncpg
import pytest
from fastapi.testclient import TestClient

from corag_cloud.config import INSECURE_INTERNAL_TOKEN, get_settings
from corag_cloud.main import create_app
from corag_cloud.provision import provision_tenant_with_owner

from .conftest import ADMIN_URL, APP_URL, requires_db

pytestmark = requires_db

AUTH = {"x-internal-token": INSECURE_INTERNAL_TOKEN}


@pytest.fixture
def client(migrated_db, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", APP_URL)
    monkeypatch.setenv("DATABASE_ADMIN_URL", ADMIN_URL)
    get_settings.cache_clear()
    with TestClient(create_app()) as c:
        yield c
    get_settings.cache_clear()


@pytest.fixture
async def tenant_id(migrated_db):
    result = await provision_tenant_with_owner(
        migrated_db,
        workspace="bill-ws",
        email=f"bill-{uuid.uuid4().hex[:10]}@example.com",
        name="Payer",
        password="a-long-password",
    )
    return result.tenant_id


async def _tenant_row(tenant_id):
    admin = await asyncpg.connect(ADMIN_URL)
    try:
        return await admin.fetchrow("SELECT * FROM tenant WHERE id = $1", tenant_id)
    finally:
        await admin.close()


def _evt() -> str:
    return f"evt_{uuid.uuid4().hex}"


async def test_checkout_event_activates_plan(client, tenant_id):
    response = client.post(
        "/internal/billing/sync",
        headers=AUTH,
        json={
            "event_id": _evt(),
            "event_type": "checkout.session.completed",
            "tenant_id": str(tenant_id),
            "plan": "starter",
            "stripe_customer_id": "cus_123",
            "stripe_subscription_id": "sub_123",
            "subscription_status": "active",
        },
    )
    assert response.status_code == 200
    assert response.json() == {"duplicate": False, "updated": True}

    tenant = await _tenant_row(tenant_id)
    assert tenant["plan"] == "starter"
    assert tenant["subscription_status"] == "active"
    assert tenant["stripe_customer_id"] == "cus_123"


async def test_replayed_event_is_a_noop(client, tenant_id):
    event_id = _evt()
    payload = {
        "event_id": event_id,
        "event_type": "checkout.session.completed",
        "tenant_id": str(tenant_id),
        "plan": "pro",
        "subscription_status": "active",
    }
    first = client.post("/internal/billing/sync", headers=AUTH, json=payload)
    assert first.json()["duplicate"] is False

    # Mutate state between the two deliveries; the replay must not re-apply.
    admin = await asyncpg.connect(ADMIN_URL)
    try:
        await admin.execute("UPDATE tenant SET plan = 'team' WHERE id = $1", tenant_id)
    finally:
        await admin.close()

    replay = client.post("/internal/billing/sync", headers=AUTH, json=payload)
    assert replay.json()["duplicate"] is True

    tenant = await _tenant_row(tenant_id)
    assert tenant["plan"] == "team"


async def test_period_advance_resets_usage(client, tenant_id):
    admin = await asyncpg.connect(ADMIN_URL)
    try:
        await admin.execute(
            "UPDATE tenant SET questions_used = 17 WHERE id = $1", tenant_id
        )
    finally:
        await admin.close()

    period_end = (datetime.now(UTC) + timedelta(days=30)).isoformat()
    client.post(
        "/internal/billing/sync",
        headers=AUTH,
        json={
            "event_id": _evt(),
            "event_type": "customer.subscription.updated",
            "tenant_id": str(tenant_id),
            "plan": "starter",
            "subscription_status": "active",
            "period_end": period_end,
        },
    )
    tenant = await _tenant_row(tenant_id)
    assert tenant["questions_used"] == 0

    # Same period again (e.g. a plan-quantity change) must NOT reset usage.
    admin = await asyncpg.connect(ADMIN_URL)
    try:
        await admin.execute(
            "UPDATE tenant SET questions_used = 9 WHERE id = $1", tenant_id
        )
    finally:
        await admin.close()
    client.post(
        "/internal/billing/sync",
        headers=AUTH,
        json={
            "event_id": _evt(),
            "event_type": "customer.subscription.updated",
            "tenant_id": str(tenant_id),
            "plan": "starter",
            "subscription_status": "active",
            "period_end": period_end,
        },
    )
    tenant = await _tenant_row(tenant_id)
    assert tenant["questions_used"] == 9


async def test_cancel_blocks_processing_access(client, tenant_id):
    client.post(
        "/internal/billing/sync",
        headers=AUTH,
        json={
            "event_id": _evt(),
            "event_type": "checkout.session.completed",
            "tenant_id": str(tenant_id),
            "plan": "starter",
            "subscription_status": "active",
        },
    )
    client.post(
        "/internal/billing/sync",
        headers=AUTH,
        json={
            "event_id": _evt(),
            "event_type": "customer.subscription.deleted",
            "tenant_id": str(tenant_id),
            "cancel": True,
        },
    )

    tenant = await _tenant_row(tenant_id)
    assert tenant["subscription_status"] == "canceled"

    # A canceled paid plan may no longer ask questions.
    headers = {
        **AUTH,
        "x-user-id": str(uuid.uuid4()),
        "x-tenant-id": str(tenant_id),
    }
    response = client.post("/ask", headers=headers, json={"question": "Still there?"})
    assert response.status_code == 402
    assert response.json()["detail"]["code"] == "subscription_inactive"


def test_unknown_plan_rejected(client, tenant_id):
    response = client.post(
        "/internal/billing/sync",
        headers=AUTH,
        json={
            "event_id": _evt(),
            "event_type": "checkout.session.completed",
            "tenant_id": str(tenant_id),
            "plan": "platinum",
        },
    )
    assert response.status_code == 422


async def test_billing_state_endpoint(client, tenant_id):
    client.post(
        "/internal/billing/sync",
        headers=AUTH,
        json={
            "event_id": _evt(),
            "event_type": "checkout.session.completed",
            "tenant_id": str(tenant_id),
            "plan": "pro",
            "stripe_customer_id": "cus_state",
            "subscription_status": "active",
        },
    )
    response = client.get(
        f"/internal/billing/state?tenant_id={tenant_id}", headers=AUTH
    )
    assert response.status_code == 200
    body = response.json()
    assert body["plan"] == "pro"
    assert body["stripe_customer_id"] == "cus_state"
