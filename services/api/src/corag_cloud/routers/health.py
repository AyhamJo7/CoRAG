"""Health and version endpoints."""

from fastapi import APIRouter

from corag_cloud import __version__

router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness probe; Phase 2 adds dependency pings (Postgres) with 503."""
    return {"status": "ok", "version": __version__}
