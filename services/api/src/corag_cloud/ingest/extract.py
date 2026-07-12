"""Text extraction from uploaded files (txt/md/pdf)."""

import io
import logging

from pypdf import PdfReader

logger = logging.getLogger(__name__)

TEXT_MIMES = {"text/plain", "text/markdown"}
PDF_MIME = "application/pdf"
SUPPORTED_MIMES = TEXT_MIMES | {PDF_MIME}

SUFFIX_MIME = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".pdf": PDF_MIME,
}


class UnsupportedFileType(Exception):
    """Raised for files we cannot extract text from."""


class ExtractionFailed(Exception):
    """Raised when a supported file cannot be parsed."""


def resolve_mime(filename: str, declared_mime: str | None) -> str:
    """Resolve the effective MIME type, preferring the file suffix."""
    lowered = filename.lower()
    for suffix, mime in SUFFIX_MIME.items():
        if lowered.endswith(suffix):
            return mime
    if declared_mime in SUPPORTED_MIMES:
        return declared_mime
    raise UnsupportedFileType(filename)


def extract_text(data: bytes, mime: str) -> str:
    """Extract plain text from file bytes.

    Args:
        data: Raw file content
        mime: Resolved MIME type (see resolve_mime)

    Returns:
        Extracted text (may be empty for image-only PDFs)
    """
    if mime in TEXT_MIMES:
        return data.decode("utf-8", errors="replace")
    if mime == PDF_MIME:
        try:
            reader = PdfReader(io.BytesIO(data))
            pages = [page.extract_text() or "" for page in reader.pages]
        except Exception as e:
            raise ExtractionFailed(str(e)) from e
        return "\n\n".join(pages)
    raise UnsupportedFileType(mime)
