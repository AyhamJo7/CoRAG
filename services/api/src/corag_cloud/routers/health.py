"""Health and version endpoints."""

import asyncio

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from corag_cloud import __version__
from corag_cloud.config import get_settings
from corag_cloud.db.pool import get_pool

router = APIRouter()

DB_PING_TIMEOUT_SECONDS = 2.0


@router.get("/healthz")
async def healthz() -> JSONResponse:
    """Liveness + readiness: pings Postgres when a database is configured."""
    settings = get_settings()
    checks: dict[str, str] = {}
    healthy = True

    if settings.database_url:
        try:
            pool = get_pool()
            async with asyncio.timeout(DB_PING_TIMEOUT_SECONDS):
                async with pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
            checks["postgres"] = "ok"
        except Exception:
            checks["postgres"] = "unreachable"
            healthy = False

    return JSONResponse(
        status_code=200 if healthy else 503,
        content={
            "status": "ok" if healthy else "degraded",
            "version": __version__,
            "checks": checks,
        },
    )
