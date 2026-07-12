"""Upload → queue → worker indexing flow against a real database."""

import uuid

import asyncpg
import pytest
from fastapi.testclient import TestClient

from corag_cloud.config import INSECURE_INTERNAL_TOKEN, get_settings
from corag_cloud.db.pool import close_pool, open_pool
from corag_cloud.ingest.worker import IngestWorker
from corag_cloud.main import create_app
from corag_cloud.provision import provision_tenant_with_owner

from .conftest import ADMIN_URL, APP_URL, requires_db
from .stubs import FailingEmbedder, StubEmbedder

pytestmark = requires_db


@pytest.fixture
def client(migrated_db, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", APP_URL)
    monkeypatch.setenv("DATABASE_ADMIN_URL", ADMIN_URL)
    get_settings.cache_clear()
    with TestClient(create_app()) as c:
        yield c
    get_settings.cache_clear()


@pytest.fixture
async def tenant_headers(migrated_db):
    result = await provision_tenant_with_owner(
        migrated_db,
        workspace="docs-ws",
        email=f"docs-{uuid.uuid4().hex[:10]}@example.com",
        name="Doc Owner",
        password="a-long-password",
    )
    return {
        "x-internal-token": INSECURE_INTERNAL_TOKEN,
        "x-user-id": str(result.user_id),
        "x-tenant-id": str(result.tenant_id),
    }, result.tenant_id


def _upload(client, headers, filename="notes.txt", content=b"", mime="text/plain"):
    content = content or b"CoRAG was founded in 2026. " * 100
    return client.post(
        "/documents",
        headers=headers,
        files={"file": (filename, content, mime)},
    )


async def _run_worker_once(embedder) -> bool:
    # The TestClient app's pool lives on the client thread's event loop; the
    # worker needs its own pool on this loop. Save/restore the module global.
    from corag_cloud.db import pool as pool_mod

    saved = pool_mod._pool
    await open_pool(APP_URL, max_size=2)
    try:
        return await IngestWorker(embedder).process_one()
    finally:
        await close_pool()
        pool_mod._pool = saved


async def test_upload_queue_index_roundtrip(client, tenant_headers, migrated_db):
    headers, tenant_id = tenant_headers

    created = _upload(client, headers)
    assert created.status_code == 201, created.text
    doc = created.json()
    assert doc["status"] == "uploaded"

    processed = await _run_worker_once(StubEmbedder())
    assert processed is True

    listed = client.get("/documents", headers=headers).json()
    target = next(d for d in listed if d["id"] == doc["id"])
    assert target["status"] == "indexed"

    admin = await asyncpg.connect(migrated_db)
    try:
        chunk_count = await admin.fetchval(
            "SELECT count(*) FROM chunk WHERE document_id = $1", uuid.UUID(doc["id"])
        )
        tenant = await admin.fetchrow(
            "SELECT docs_count, storage_bytes_used FROM tenant WHERE id = $1",
            tenant_id,
        )
        job_status = await admin.fetchval(
            "SELECT status FROM ingest_job WHERE document_id = $1",
            uuid.UUID(doc["id"]),
        )
    finally:
        await admin.close()
    assert chunk_count > 0
    assert tenant["docs_count"] == 1
    assert tenant["storage_bytes_used"] == doc["size_bytes"]
    assert job_status == "done"


async def test_worker_failure_marks_document_failed(
    client, tenant_headers, migrated_db
):
    headers, _ = tenant_headers
    doc = _upload(client, headers).json()

    # MAX_ATTEMPTS=3: two failures requeue, the third marks failed.
    for _ in range(3):
        assert await _run_worker_once(FailingEmbedder()) is True

    listed = client.get("/documents", headers=headers).json()
    target = next(d for d in listed if d["id"] == doc["id"])
    assert target["status"] == "failed"
    assert "unavailable" in (target["error"] or "")

    admin = await asyncpg.connect(migrated_db)
    try:
        job = await admin.fetchrow(
            "SELECT status, attempts FROM ingest_job WHERE document_id = $1",
            uuid.UUID(doc["id"]),
        )
    finally:
        await admin.close()
    assert job["status"] == "failed"
    assert job["attempts"] == 3


async def test_document_cap_enforced(client, tenant_headers, migrated_db):
    headers, tenant_id = tenant_headers

    admin = await asyncpg.connect(migrated_db)
    try:
        await admin.execute(
            "UPDATE ingest_job SET status = 'done' WHERE tenant_id = $1", tenant_id
        )
        await admin.execute(
            "SELECT set_config('app.tenant_id', $1, false)", str(tenant_id)
        )
        await admin.execute(
            "UPDATE tenant SET docs_count = 10 WHERE id = $1", tenant_id
        )
    finally:
        await admin.close()

    response = _upload(client, headers)
    assert response.status_code == 402
    assert response.json()["detail"]["code"] == "document_limit"


async def test_storage_cap_enforced(client, tenant_headers, migrated_db):
    headers, tenant_id = tenant_headers

    admin = await asyncpg.connect(migrated_db)
    try:
        await admin.execute(
            "SELECT set_config('app.tenant_id', $1, false)", str(tenant_id)
        )
        await admin.execute(
            "UPDATE tenant SET storage_bytes_used = 20 * 1024 * 1024 WHERE id = $1",
            tenant_id,
        )
    finally:
        await admin.close()

    response = _upload(client, headers)
    assert response.status_code == 402
    assert response.json()["detail"]["code"] == "storage_limit"


def test_oversize_upload_rejected(client, tenant_headers):
    headers, _ = tenant_headers
    response = _upload(client, headers, content=b"x" * (10 * 1024 * 1024 + 1))
    assert response.status_code == 413


def test_unsupported_type_rejected(client, tenant_headers):
    headers, _ = tenant_headers
    response = _upload(
        client, headers, filename="data.csv", content=b"a,b", mime="text/csv"
    )
    assert response.status_code == 415


def test_delete_restores_capacity(client, tenant_headers):
    headers, _ = tenant_headers
    doc = _upload(client, headers).json()

    deleted = client.delete(f"/documents/{doc['id']}", headers=headers)
    assert deleted.status_code == 204

    listed = client.get("/documents", headers=headers).json()
    assert all(d["id"] != doc["id"] for d in listed)
