"""Alembic environment: migrations run under the ADMIN database role."""

import os

from alembic import context
from sqlalchemy import create_engine


def _migration_url() -> str:
    url = os.environ.get("DATABASE_ADMIN_URL") or os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_ADMIN_URL (or DATABASE_URL) must be set")
    # asyncpg-style DSNs are rewritten for the sync psycopg driver.
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def run_migrations_online() -> None:
    engine = create_engine(_migration_url())
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=None)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    raise RuntimeError("Offline migrations are not supported")
run_migrations_online()
