"""FastAPI application factory for CoRAG Cloud."""

import logging

from fastapi import FastAPI

from corag_cloud.config import get_settings
from corag_cloud.routers import health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def create_app() -> FastAPI:
    """Build the application with all routers registered."""
    get_settings()  # fail fast on invalid configuration
    app = FastAPI(
        title="CoRAG Cloud API",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.include_router(health.router)
    return app


app = create_app()
