"""Provisioning and login flow against a real database."""

import uuid

import pytest
from fastapi.testclient import TestClient

from corag_cloud.config import INSECURE_INTERNAL_TOKEN, get_settings
from corag_cloud.main import create_app
from corag_cloud.provision import EmailAlreadyExists, provision_tenant_with_owner

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


def _email() -> str:
    return f"user-{uuid.uuid4().hex[:10]}@example.com"


async def test_provision_is_atomic_and_duplicate_safe(migrated_db):
    email = _email()
    result = await provision_tenant_with_owner(
        migrated_db,
        workspace="",
        email=email,
        name="Ada",
        password="a-long-password",
    )
    assert result.tenant_id and result.user_id

    with pytest.raises(EmailAlreadyExists):
        await provision_tenant_with_owner(
            migrated_db,
            workspace="other",
            email=email.upper(),  # citext: case-insensitive duplicate
            name="Ada Again",
            password="another-password",
        )


def test_signup_then_login_roundtrip(client):
    email = _email()

    created = client.post(
        "/internal/provision",
        headers=AUTH,
        json={"name": "Grace", "email": email, "password": "hopper-passw0rd"},
    )
    assert created.status_code == 201
    tenant_id = created.json()["tenant_id"]

    ok = client.post(
        "/internal/login",
        headers=AUTH,
        json={"email": email, "password": "hopper-passw0rd"},
    )
    assert ok.status_code == 200
    body = ok.json()
    assert body["email"] == email
    assert body["tenants"][0]["id"] == tenant_id
    assert body["tenants"][0]["role"] == "owner"
    assert body["tenants"][0]["plan"] == "trial"


def test_login_rejects_bad_credentials_uniformly(client):
    email = _email()
    client.post(
        "/internal/provision",
        headers=AUTH,
        json={"name": "Alan", "email": email, "password": "turing-passw0rd"},
    )

    wrong_pw = client.post(
        "/internal/login",
        headers=AUTH,
        json={"email": email, "password": "wrong-password"},
    )
    unknown = client.post(
        "/internal/login",
        headers=AUTH,
        json={"email": _email(), "password": "whatever-password"},
    )
    assert wrong_pw.status_code == unknown.status_code == 401
    assert wrong_pw.json() == unknown.json()


def test_duplicate_email_maps_to_409(client):
    email = _email()
    first = client.post(
        "/internal/provision",
        headers=AUTH,
        json={"name": "Edsger", "email": email, "password": "dijkstra-pass1"},
    )
    assert first.status_code == 201

    second = client.post(
        "/internal/provision",
        headers=AUTH,
        json={"name": "Edsger", "email": email, "password": "dijkstra-pass2"},
    )
    assert second.status_code == 409


def test_provision_requires_internal_token(client):
    response = client.post(
        "/internal/provision",
        json={"name": "Eve", "email": _email(), "password": "sniffing-pass1"},
    )
    assert response.status_code == 401
