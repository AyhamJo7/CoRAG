"""asyncpg pool management and tenant-scoped connections.

``tenant_connection`` is the only sanctioned way to touch tenant data: it
opens a transaction and binds ``app.tenant_id`` transaction-locally so the
FORCE-RLS policies apply. The pool runs under the restricted ``corag_app``
role (no BYPASSRLS).
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

import asyncpg

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None


async def open_pool(dsn: str, min_size: int = 1, max_size: int = 5) -> asyncpg.Pool:
    """Create the process-wide pool (called from the app lifespan)."""
    global _pool
    _pool = await asyncpg.create_pool(dsn, min_size=min_size, max_size=max_size)
    logger.info("Database pool opened (max_size=%d)", max_size)
    return _pool


async def close_pool() -> None:
    """Dispose the process-wide pool."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("Database pool closed")


def get_pool() -> asyncpg.Pool:
    """Return the open pool; raises if the app started without a database."""
    if _pool is None:
        raise RuntimeError("Database pool is not open")
    return _pool


@asynccontextmanager
async def tenant_connection(tenant_id: UUID) -> AsyncIterator[asyncpg.Connection]:
    """Yield a connection inside a transaction scoped to one tenant."""
    pool = get_pool()
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute(
            "SELECT set_config('app.tenant_id', $1, true)", str(tenant_id)
        )
        yield conn


@asynccontextmanager
async def plain_connection() -> AsyncIterator[asyncpg.Connection]:
    """Yield a connection without tenant context (identity tables only)."""
    pool = get_pool()
    async with pool.acquire() as conn:
        yield conn
