"""Document schema and processing."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Document:
    """Represents a document in the knowledge corpus."""

    id: str
    title: str
    text: str
    url: str | None = None
    sections: list[str] = field(default_factory=list)
    timestamp: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert document to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "text": self.text,
            "url": self.url,
            "sections": self.sections,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        """Create document from dictionary."""
        timestamp = None
        if data.get("timestamp"):
            timestamp = datetime.fromisoformat(data["timestamp"])

        return cls(
            id=data["id"],
            title=data["title"],
            text=data["text"],
            url=data.get("url"),
            sections=data.get("sections", []),
            timestamp=timestamp,
            metadata=data.get("metadata", {}),
        )


@dataclass
class Chunk:
    """Represents a chunk of text from a document."""

    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    tokens: int
    doc_title: str
    doc_url: str | None = None
    section: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert chunk to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "tokens": self.tokens,
            "doc_title": self.doc_title,
            "doc_url": self.doc_url,
            "section": self.section,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        """Create chunk from dictionary."""
        return cls(
            chunk_id=data["chunk_id"],
            doc_id=data["doc_id"],
            text=data["text"],
            start_char=data["start_char"],
            end_char=data["end_char"],
            tokens=data["tokens"],
            doc_title=data["doc_title"],
            doc_url=data.get("doc_url"),
            section=data.get("section"),
            metadata=data.get("metadata", {}),
        )
