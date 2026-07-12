"""Unit tests for upload text extraction."""

import io

import pytest
from pypdf import PdfWriter

from corag_cloud.ingest.extract import (
    ExtractionFailed,
    UnsupportedFileType,
    extract_text,
    resolve_mime,
)


def test_resolve_mime_prefers_suffix():
    assert resolve_mime("notes.txt", "application/octet-stream") == "text/plain"
    assert resolve_mime("README.MD", None) == "text/markdown"
    assert resolve_mime("paper.pdf", "text/plain") == "application/pdf"


def test_resolve_mime_falls_back_to_declared():
    assert resolve_mime("upload", "text/plain") == "text/plain"


def test_resolve_mime_rejects_unknown():
    with pytest.raises(UnsupportedFileType):
        resolve_mime("archive.zip", "application/zip")


def test_extract_plain_text_handles_bad_utf8():
    text = extract_text(b"caf\xc3\xa9 \xff invalid", "text/plain")
    assert "café" in text


def test_extract_empty_pdf_yields_no_text():
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buffer = io.BytesIO()
    writer.write(buffer)

    text = extract_text(buffer.getvalue(), "application/pdf")

    assert text.strip() == ""


def test_extract_corrupt_pdf_raises():
    with pytest.raises(ExtractionFailed):
        extract_text(b"%PDF-1.4 garbage without structure", "application/pdf")
