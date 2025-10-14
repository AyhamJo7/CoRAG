"""Document chunking with token-based splitting and overlap."""

import hashlib
from typing import List, Optional

from corag.corpus.document import Chunk, Document
from corag.utils.text import clean_text, count_tokens_approximate


class Chunker:
    """Chunks documents into smaller pieces with overlap."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 50,
    ):
        """Initialize chunker.

        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Number of tokens to overlap between chunks
            min_chunk_size: Minimum chunk size in tokens
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_document(self, document: Document) -> List[Chunk]:
        """Chunk a document into overlapping pieces.

        Args:
            document: Document to chunk

        Returns:
            List of Chunk objects
        """
        text = clean_text(document.text)
        chunks = []

        # Split by words to handle tokens
        words = text.split()
        current_pos = 0
        start_char = 0
        chunk_idx = 0

        while current_pos < len(words):
            # Take chunk_size words (approximate tokens)
            end_pos = min(current_pos + self.chunk_size, len(words))
            chunk_words = words[current_pos:end_pos]
            chunk_text = " ".join(chunk_words)

            # Calculate character positions
            end_char = start_char + len(chunk_text)

            # Create chunk
            token_count = count_tokens_approximate(chunk_text)

            if token_count >= self.min_chunk_size or end_pos == len(words):
                chunk_id = self._generate_chunk_id(document.id, chunk_idx)
                chunk = Chunk(
                    chunk_id=chunk_id,
                    doc_id=document.id,
                    text=chunk_text,
                    start_char=start_char,
                    end_char=end_char,
                    tokens=token_count,
                    doc_title=document.title,
                    doc_url=document.url,
                    metadata={
                        "doc_sections": document.sections,
                        "doc_timestamp": (
                            document.timestamp.isoformat() if document.timestamp else None
                        ),
                    },
                )
                chunks.append(chunk)
                chunk_idx += 1

            # Move forward with overlap
            if end_pos < len(words):
                current_pos += self.chunk_size - self.chunk_overlap
                # Update start_char position
                overlap_text = " ".join(words[current_pos:end_pos])
                start_char = end_char - len(overlap_text)
            else:
                break

        return chunks

    def _generate_chunk_id(self, doc_id: str, chunk_idx: int) -> str:
        """Generate unique chunk ID."""
        content = f"{doc_id}_{chunk_idx}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Chunk multiple documents.

        Args:
            documents: List of documents to chunk

        Returns:
            List of all chunks
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        return all_chunks
