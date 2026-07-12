"""Shared fixtures for corag_cloud tests.

Database-backed tests need a pgvector Postgres and two DSNs:

- ``TEST_DATABASE_ADMIN_URL`` — superuser (migrations, provisioning)
- ``TEST_DATABASE_URL``       — restricted ``corag_app`` role (runtime path)

When they are unset, those tests are skipped (unit tests still run).
Locally: ``docker run --rm -d -p 5433:5432 -e POSTGRES_PASSWORD=test
-e POSTGRES_DB=corag_test pgvector/pgvector:pg16`` and export
TEST_DATABASE_ADMIN_URL=postgresql://postgres:test@localhost:5433/corag_test
TEST_DATABASE_URL=postgresql://corag_app:corag_app@localhost:5433/corag_test
"""

import os
from pathlib import Path

import asyncpg
import pytest
from alembic import command
from alembic.config import Config

ADMIN_URL = os.environ.get("TEST_DATABASE_ADMIN_URL", "")
APP_URL = os.environ.get("TEST_DATABASE_URL", "")

requires_db = pytest.mark.skipif(
    not (ADMIN_URL and APP_URL),
    reason="TEST_DATABASE_ADMIN_URL / TEST_DATABASE_URL not set",
)

_API_DIR = Path(__file__).resolve().parents[1]

_CREATE_APP_ROLE = """
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'corag_app') THEN
        CREATE ROLE corag_app LOGIN PASSWORD 'corag_app';
    END IF;
END
$$
"""


@pytest.fixture(scope="session")
def migrated_db() -> str:
    """Create the app role, run migrations, and return the admin DSN."""
    if not (ADMIN_URL and APP_URL):
        pytest.skip("database not configured")

    import asyncio

    async def prepare() -> None:
        conn = await asyncpg.connect(ADMIN_URL)
        try:
            await conn.execute(_CREATE_APP_ROLE)
            await conn.execute(
                "GRANT CONNECT ON DATABASE "
                + ADMIN_URL.rsplit("/", 1)[1].split("?")[0]
                + " TO corag_app"
            )
        finally:
            await conn.close()

    asyncio.run(prepare())

    config = Config(str(_API_DIR / "alembic.ini"))
    config.set_main_option("script_location", str(_API_DIR / "alembic"))
    os.environ["DATABASE_ADMIN_URL"] = ADMIN_URL
    command.upgrade(config, "head")
    return ADMIN_URL


@pytest.fixture
async def admin_conn(migrated_db: str) -> asyncpg.Connection:
    conn = await asyncpg.connect(migrated_db)
    yield conn
    await conn.close()


@pytest.fixture
async def app_conn(migrated_db: str) -> asyncpg.Connection:
    """Connection under the restricted corag_app role (RLS enforced)."""
    conn = await asyncpg.connect(APP_URL)
    yield conn
    await conn.close()
