"""Smoke tests for the CoRAG Cloud app skeleton."""

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from corag_cloud.config import INSECURE_INTERNAL_TOKEN, Settings
from corag_cloud.deps import RequestContext, get_request_context
from corag_cloud.main import create_app


def test_healthz() -> None:
    client = TestClient(create_app())

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_openapi_docs_disabled() -> None:
    client = TestClient(create_app())

    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_settings_reject_insecure_defaults_outside_development() -> None:
    with pytest.raises(ValueError, match="insecure default"):
        Settings(environment="production")


def test_settings_reject_short_token_outside_development() -> None:
    with pytest.raises(ValueError, match="at least 32 characters"):
        Settings(environment="production", internal_service_token="short")


def test_settings_allow_defaults_in_development() -> None:
    settings = Settings(environment="development")

    assert settings.internal_service_token == INSECURE_INTERNAL_TOKEN


def _protected_app() -> FastAPI:
    app = create_app()

    @app.get("/whoami")
    def whoami(ctx: RequestContext = Depends(get_request_context)) -> dict[str, str]:
        return {"user_id": str(ctx.user_id), "tenant_id": str(ctx.tenant_id)}

    return app


def test_internal_token_required() -> None:
    client = TestClient(_protected_app())

    assert client.get("/whoami").status_code == 401
    assert (
        client.get("/whoami", headers={"x-internal-token": "wrong"}).status_code == 401
    )


def test_internal_token_and_identity_headers_accepted() -> None:
    client = TestClient(_protected_app())
    headers = {
        "x-internal-token": INSECURE_INTERNAL_TOKEN,
        "x-user-id": "6f1e0863-95a9-4a6c-9d61-3f1a1a1a1a1a",
        "x-tenant-id": "aa1e0863-95a9-4a6c-9d61-3f1a1a1a1a1a",
    }

    response = client.get("/whoami", headers=headers)

    assert response.status_code == 200
    assert response.json()["tenant_id"] == "aa1e0863-95a9-4a6c-9d61-3f1a1a1a1a1a"


def test_invalid_identity_headers_rejected() -> None:
    client = TestClient(_protected_app())
    headers = {
        "x-internal-token": INSECURE_INTERNAL_TOKEN,
        "x-user-id": "not-a-uuid",
        "x-tenant-id": "also-not-a-uuid",
    }

    assert client.get("/whoami", headers=headers).status_code == 400
