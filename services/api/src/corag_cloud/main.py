"""FastAPI application factory for CoRAG Cloud."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from corag_cloud.config import get_settings
from corag_cloud.db.pool import close_pool, open_pool
from corag_cloud.routers import (
    api_keys_router,
    ask,
    documents,
    health,
    internal,
    usage,
    v1,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    if settings.database_url:
        await open_pool(settings.database_url)
    yield
    if settings.database_url:
        await close_pool()


def create_app() -> FastAPI:
    """Build the application with all routers registered."""
    get_settings()  # fail fast on invalid configuration
    app = FastAPI(
        title="CoRAG Cloud API",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(internal.router)
    app.include_router(documents.router)
    app.include_router(ask.router)
    app.include_router(usage.router)
    app.include_router(api_keys_router.router)
    app.include_router(v1.router)
    return app


app = create_app()
