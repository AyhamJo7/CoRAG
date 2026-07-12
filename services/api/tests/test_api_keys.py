"""Developer API keys: lifecycle, /v1 auth, and rate limiting."""

import uuid

import pytest
from fastapi.testclient import TestClient

from corag_cloud.api_keys import KEY_PREFIX, generate_api_key, hash_api_key
from corag_cloud.config import INSECURE_INTERNAL_TOKEN, get_settings
from corag_cloud.main import create_app
from corag_cloud.provision import provision_tenant_with_owner
from corag_cloud.rate_limit import FixedWindowLimiter
from corag_cloud.routers import ask as ask_router
from corag_cloud.service.ask_service import AskResult

from .conftest import ADMIN_URL, APP_URL, requires_db

STUB_RESULT = AskResult(
    answer="An answer [1].",
    citations=[{"id": "1", "title": "doc", "url": "", "chunk_id": "c1"}],
    num_steps=1,
    num_chunks=2,
    latency_ms=10,
)


def test_generated_keys_have_expected_shape():
    generated = generate_api_key("salt-value")

    assert generated.plaintext.startswith(KEY_PREFIX)
    assert len(generated.plaintext) == len(KEY_PREFIX) + 48
    assert generated.key_prefix == generated.plaintext[:16]
    assert generated.key_hash == hash_api_key(generated.plaintext, "salt-value")
    # Same key, different salt: different hash.
    assert generated.key_hash != hash_api_key(generated.plaintext, "other-salt")


def test_rate_limiter_blocks_after_limit():
    from fastapi import HTTPException

    limiter = FixedWindowLimiter(limit_per_minute=2)
    tenant = uuid.uuid4()

    limiter.check(tenant)
    limiter.check(tenant)
    with pytest.raises(HTTPException) as excinfo:
        limiter.check(tenant)
    assert excinfo.value.status_code == 429
    # Another tenant is unaffected.
    limiter.check(uuid.uuid4())


@pytest.fixture
def client(migrated_db, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", APP_URL)
    monkeypatch.setenv("DATABASE_ADMIN_URL", ADMIN_URL)
    get_settings.cache_clear()
    ask_router._limiter.cache_clear()
    monkeypatch.setattr(ask_router, "run_ask", lambda tenant_id, question: STUB_RESULT)
    with TestClient(create_app()) as c:
        yield c
    get_settings.cache_clear()
    ask_router._limiter.cache_clear()


@pytest.fixture
async def tenant_headers(migrated_db):
    result = await provision_tenant_with_owner(
        migrated_db,
        workspace="keys-ws",
        email=f"keys-{uuid.uuid4().hex[:10]}@example.com",
        name="Key Owner",
        password="a-long-password",
    )
    return {
        "x-internal-token": INSECURE_INTERNAL_TOKEN,
        "x-user-id": str(result.user_id),
        "x-tenant-id": str(result.tenant_id),
    }


@requires_db
def test_key_lifecycle_and_v1_ask(client, tenant_headers):
    created = client.post("/api-keys", headers=tenant_headers, json={"name": "prod"})
    assert created.status_code == 201
    body = created.json()
    plaintext = body["key"]
    assert plaintext.startswith(KEY_PREFIX)

    # Listing never exposes the key again — only the display prefix.
    listed = client.get("/api-keys", headers=tenant_headers).json()
    assert listed[0]["key_prefix"] == plaintext[:16]
    assert "key" not in listed[0]
    assert listed[0]["last_used_at"] is None

    # The key answers on /v1/ask and shares the quota accounting.
    asked = client.post(
        "/v1/ask",
        headers={"authorization": f"Bearer {plaintext}"},
        json={"question": "What does the doc say?"},
    )
    assert asked.status_code == 200
    assert "event: answer" in asked.text

    listed = client.get("/api-keys", headers=tenant_headers).json()
    assert listed[0]["last_used_at"] is not None

    # Revocation kills the key immediately.
    revoked = client.delete(f"/api-keys/{body['id']}", headers=tenant_headers)
    assert revoked.status_code == 204
    denied = client.post(
        "/v1/ask",
        headers={"authorization": f"Bearer {plaintext}"},
        json={"question": "Still?"},
    )
    assert denied.status_code == 401


@requires_db
def test_v1_rejects_garbage_credentials(client, tenant_headers):
    no_header = client.post("/v1/ask", json={"question": "hi"})
    wrong_scheme = client.post(
        "/v1/ask",
        headers={"authorization": "Basic corag_live_x"},
        json={"question": "hi"},
    )
    unknown_key = client.post(
        "/v1/ask",
        headers={"authorization": f"Bearer {KEY_PREFIX}{'0' * 48}"},
        json={"question": "hi"},
    )
    assert no_header.status_code == 401
    assert wrong_scheme.status_code == 401
    assert unknown_key.status_code == 401


@requires_db
def test_key_cap_per_tenant(client, tenant_headers):
    for i in range(5):
        assert (
            client.post(
                "/api-keys", headers=tenant_headers, json={"name": f"k{i}"}
            ).status_code
            == 201
        )
    sixth = client.post("/api-keys", headers=tenant_headers, json={"name": "k5"})
    assert sixth.status_code == 409


@requires_db
def test_ask_rate_limit_enforced(migrated_db, monkeypatch, tenant_headers):
    monkeypatch.setenv("DATABASE_URL", APP_URL)
    monkeypatch.setenv("DATABASE_ADMIN_URL", ADMIN_URL)
    monkeypatch.setenv("ASK_RATE_LIMIT_PER_MINUTE", "2")
    get_settings.cache_clear()
    ask_router._limiter.cache_clear()
    monkeypatch.setattr(ask_router, "run_ask", lambda tenant_id, question: STUB_RESULT)
    try:
        with TestClient(create_app()) as limited:
            first = limited.post(
                "/ask", headers=tenant_headers, json={"question": "1?"}
            )
            second = limited.post(
                "/ask", headers=tenant_headers, json={"question": "2?"}
            )
            third = limited.post(
                "/ask", headers=tenant_headers, json={"question": "3?"}
            )
        assert first.status_code == 200
        assert second.status_code == 200
        assert third.status_code == 429
    finally:
        get_settings.cache_clear()
        ask_router._limiter.cache_clear()
