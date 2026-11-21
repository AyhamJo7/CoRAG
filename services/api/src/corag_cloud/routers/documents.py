"""Document upload, listing, and deletion."""

import logging
from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from pydantic import BaseModel

from corag_cloud.billing.access import (
    require_active_access,
    require_document_capacity,
)
from corag_cloud.billing.plans import MAX_UPLOAD_BYTES
from corag_cloud.db.pool import tenant_connection
from corag_cloud.deps import RequestContext, get_request_context
from corag_cloud.ingest.extract import (
    ExtractionFailed,
    UnsupportedFileType,
    extract_text,
    resolve_mime,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents")

ContextDep = Annotated[RequestContext, Depends(get_request_context)]


class DocumentOut(BaseModel):
    id: UUID
    title: str
    filename: str
    mime: str
    size_bytes: int
    status: str
    error: str | None
    created_at: datetime


@router.post("", response_model=DocumentOut, status_code=201)
async def upload_document(ctx: ContextDep, file: UploadFile) -> DocumentOut:
    """Accept a txt/md/pdf file, extract its text, and queue indexing."""
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
        )
    if not data:
        raise HTTPException(status_code=422, detail="File is empty")

    filename = file.filename or "upload"
    try:
        mime = resolve_mime(filename, file.content_type)
        text = extract_text(data, mime)
    except UnsupportedFileType:
        raise HTTPException(
            status_code=415, detail="Only .txt, .md and .pdf files are supported"
        ) from None
    except ExtractionFailed:
        raise HTTPException(
            status_code=422, detail="Could not extract text from this file"
        ) from None
    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail="No extractable text found (is this a scanned PDF?)",
        )

    title = filename.rsplit(".", 1)[0][:200]

    async with tenant_connection(ctx.tenant_id) as conn:
        # FOR UPDATE serializes concurrent uploads against the caps.
        tenant = await conn.fetchrow(
            "SELECT * FROM tenant WHERE id = $1 FOR UPDATE", ctx.tenant_id
        )
        assert tenant is not None  # RLS guarantees it is ours
        require_active_access(tenant)
        require_document_capacity(tenant, len(data))

        row = await conn.fetchrow(
            "INSERT INTO document (tenant_id, title, filename, mime, size_bytes, "
            "text) VALUES ($1, $2, $3, $4, $5, $6) "
            "RETURNING id, title, filename, mime, size_bytes, status, error, "
            "created_at",
            ctx.tenant_id,
            title,
            filename[:255],
            mime,
            len(data),
            text,
        )
        assert row is not None
        await conn.execute(
            "UPDATE tenant SET docs_count = docs_count + 1, "
            "storage_bytes_used = storage_bytes_used + $2 WHERE id = $1",
            ctx.tenant_id,
            len(data),
        )
        await conn.execute(
            "INSERT INTO ingest_job (tenant_id, document_id) VALUES ($1, $2)",
            ctx.tenant_id,
            row["id"],
        )

    logger.info("Queued document %s for tenant %s", row["id"], ctx.tenant_id)
    return DocumentOut(**dict(row))


@router.get("", response_model=list[DocumentOut])
async def list_documents(ctx: ContextDep) -> list[DocumentOut]:
    async with tenant_connection(ctx.tenant_id) as conn:
        rows = await conn.fetch(
            "SELECT id, title, filename, mime, size_bytes, status, error, "
            "created_at FROM document ORDER BY created_at DESC"
        )
    return [DocumentOut(**dict(r)) for r in rows]


@router.delete("/{document_id}", status_code=204)
async def delete_document(ctx: ContextDep, document_id: UUID) -> None:
    async with tenant_connection(ctx.tenant_id) as conn:
        row = await conn.fetchrow(
            "DELETE FROM document WHERE id = $1 RETURNING size_bytes", document_id
        )
        if row is None:
            raise HTTPException(status_code=404, detail="Document not found")
        await conn.execute(
            "UPDATE tenant SET docs_count = GREATEST(docs_count - 1, 0), "
            "storage_bytes_used = GREATEST(storage_bytes_used - $2, 0) "
            "WHERE id = $1",
            ctx.tenant_id,
            row["size_bytes"],
        )
