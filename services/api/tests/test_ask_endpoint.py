"""SSE /ask endpoint and /usage: quotas, streaming events, usage accounting."""

import uuid

import asyncpg
import pytest
from fastapi.testclient import TestClient

from corag_cloud.config import INSECURE_INTERNAL_TOKEN, get_settings
from corag_cloud.main import create_app
from corag_cloud.provision import provision_tenant_with_owner
from corag_cloud.routers import ask as ask_router
from corag_cloud.service.ask_service import AskResult

from .conftest import ADMIN_URL, APP_URL, requires_db

pytestmark = requires_db

STUB_RESULT = AskResult(
    answer="CoRAG was founded in Hamburg in 2026 [1].",
    citations=[{"id": "1", "title": "founding", "url": "", "chunk_id": "chunk-1"}],
    num_steps=2,
    num_chunks=4,
    latency_ms=1234,
)


@pytest.fixture
def client(migrated_db, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", APP_URL)
    monkeypatch.setenv("DATABASE_ADMIN_URL", ADMIN_URL)
    get_settings.cache_clear()
    monkeypatch.setattr(ask_router, "run_ask", lambda tenant_id, question: STUB_RESULT)
    with TestClient(create_app()) as c:
        yield c
    get_settings.cache_clear()


@pytest.fixture
async def tenant_headers(migrated_db):
    result = await provision_tenant_with_owner(
        migrated_db,
        workspace="ask-ws",
        email=f"ask-{uuid.uuid4().hex[:10]}@example.com",
        name="Asker",
        password="a-long-password",
    )
    return {
        "x-internal-token": INSECURE_INTERNAL_TOKEN,
        "x-user-id": str(result.user_id),
        "x-tenant-id": str(result.tenant_id),
    }, result.tenant_id


def _events(sse_body: str) -> dict[str, str]:
    events: dict[str, str] = {}
    for block in sse_body.split("\n\n"):
        name, data = None, ""
        for line in block.split("\n"):
            if line.startswith("event: "):
                name = line[7:]
            if line.startswith("data: "):
                data = line[6:]
        if name:
            events[name] = data
    return events


async def test_ask_streams_answer_and_accounts_usage(
    client, tenant_headers, migrated_db
):
    headers, tenant_id = tenant_headers

    response = client.post(
        "/ask", headers=headers, json={"question": "Where was CoRAG founded?"}
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _events(response.text)
    assert "Hamburg" in events["answer"]
    assert "founding" in events["citations"]
    assert "latency_ms" in events["done"]

    admin = await asyncpg.connect(migrated_db)
    try:
        used = await admin.fetchval(
            "SELECT questions_used FROM tenant WHERE id = $1", tenant_id
        )
        logged = await admin.fetchrow(
            "SELECT question, answer, num_steps FROM question_log WHERE tenant_id = $1",
            tenant_id,
        )
    finally:
        await admin.close()
    assert used == 1
    assert logged["question"] == "Where was CoRAG founded?"
    assert logged["num_steps"] == 2


async def test_ask_quota_exhausted_returns_402(client, tenant_headers, migrated_db):
    headers, tenant_id = tenant_headers

    admin = await asyncpg.connect(migrated_db)
    try:
        await admin.execute(
            "UPDATE tenant SET questions_used = 25 WHERE id = $1", tenant_id
        )
    finally:
        await admin.close()

    response = client.post("/ask", headers=headers, json={"question": "One more?"})

    assert response.status_code == 402
    assert response.json()["detail"]["code"] == "question_quota"


async def test_ask_expired_trial_returns_402(client, tenant_headers, migrated_db):
    headers, tenant_id = tenant_headers

    admin = await asyncpg.connect(migrated_db)
    try:
        await admin.execute(
            "UPDATE tenant SET trial_ends_at = now() - interval '1 day' WHERE id = $1",
            tenant_id,
        )
    finally:
        await admin.close()

    response = client.post("/ask", headers=headers, json={"question": "Still ok?"})

    assert response.status_code == 402
    assert response.json()["detail"]["code"] == "trial_expired"


def test_ask_validates_question_length(client, tenant_headers):
    headers, _ = tenant_headers
    assert (
        client.post("/ask", headers=headers, json={"question": ""}).status_code == 422
    )
    assert (
        client.post("/ask", headers=headers, json={"question": "x" * 2001}).status_code
        == 422
    )


def test_usage_reports_limits(client, tenant_headers):
    headers, _ = tenant_headers

    response = client.get("/usage", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["plan"] == "trial"
    assert body["questions_limit"] == 25
    assert body["docs_limit"] == 10
