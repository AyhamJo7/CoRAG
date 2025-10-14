"""Corpus ingestion from various formats."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional

from corag.corpus.document import Document

logger = logging.getLogger(__name__)


class CorpusIngestor:
    """Ingests documents from various formats."""

    def ingest_jsonl(
        self, path: Path, max_docs: Optional[int] = None
    ) -> Iterator[Document]:
        """Ingest documents from JSONL file.

        Args:
            path: Path to JSONL file
            max_docs: Maximum number of documents to ingest

        Yields:
            Document objects
        """
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if max_docs and count >= max_docs:
                    break

                try:
                    data = json.loads(line)
                    doc = self._parse_document(data, line_num)
                    if doc:
                        yield doc
                        count += 1
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")

        logger.info(f"Ingested {count} documents from {path}")

    def _parse_document(self, data: dict, line_num: int) -> Optional[Document]:
        """Parse document from dictionary.

        Args:
            data: Document data
            line_num: Line number for error reporting

        Returns:
            Document object or None if invalid
        """
        # Required fields
        if not all(k in data for k in ["id", "title", "text"]):
            logger.warning(f"Line {line_num}: Missing required fields")
            return None

        # Parse timestamp if present
        timestamp = None
        if "timestamp" in data:
            try:
                timestamp = datetime.fromisoformat(data["timestamp"])
            except (ValueError, TypeError):
                logger.warning(f"Line {line_num}: Invalid timestamp format")

        return Document(
            id=str(data["id"]),
            title=str(data["title"]),
            text=str(data["text"]),
            url=data.get("url"),
            sections=data.get("sections", []),
            timestamp=timestamp,
            metadata=data.get("metadata", {}),
        )

    def ingest_directory(
        self, directory: Path, pattern: str = "*.jsonl", max_docs: Optional[int] = None
    ) -> Iterator[Document]:
        """Ingest documents from directory of JSONL files.

        Args:
            directory: Directory containing JSONL files
            pattern: File pattern to match
            max_docs: Maximum total documents to ingest

        Yields:
            Document objects
        """
        count = 0
        for file_path in sorted(directory.glob(pattern)):
            remaining = max_docs - count if max_docs else None
            for doc in self.ingest_jsonl(file_path, remaining):
                yield doc
                count += 1
                if max_docs and count >= max_docs:
                    return

    def save_documents(self, documents: List[Document], output_path: Path) -> None:
        """Save documents to JSONL file.

        Args:
            documents: Documents to save
            output_path: Output file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for doc in documents:
                json_line = json.dumps(doc.to_dict(), ensure_ascii=False)
                f.write(json_line + "\n")

        logger.info(f"Saved {len(documents)} documents to {output_path}")
