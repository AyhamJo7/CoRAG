"""Answer synthesis with citation support."""

import logging
import re

from corag.controller.base import Controller, GenerationConfig
from corag.corpus.document import Chunk
from corag.retrieval.state import RetrievalState

logger = logging.getLogger(__name__)


SYNTHESIS_PROMPT = """You are a helpful assistant that synthesizes information from multiple sources to answer complex questions.

Question: {question}

Retrieved information from {num_sources} sources:

{sources}

Instructions:
1. Provide a comprehensive answer to the question
2. Use ONLY information from the provided sources
3. Cite sources using inline markers like [1], [2], etc.
4. If sources contradict, acknowledge both perspectives
5. If information is insufficient, state what is missing
6. Structure the answer clearly with sections if needed

Answer:"""


class Synthesizer:
    """Synthesizes answers from retrieved chunks with citations."""

    def __init__(
        self,
        controller: Controller,
        config: GenerationConfig | None = None,
        max_context_chunks: int = 20,
        max_chunk_length: int = 500,
    ):
        """Initialize synthesizer.

        Args:
            controller: LLM controller
            config: Generation configuration
            max_context_chunks: Maximum chunks to include in context
            max_chunk_length: Maximum characters per chunk in context
        """
        self.controller = controller
        self.config = config or GenerationConfig(temperature=0.3, max_tokens=2048)
        self.max_context_chunks = max_context_chunks
        self.max_chunk_length = max_chunk_length

    def synthesize(self, state: RetrievalState) -> tuple[str, list[dict[str, str]]]:
        """Synthesize answer from retrieval state.

        Args:
            state: Retrieval state with chunks

        Returns:
            Tuple of (answer text, citation list)
        """
        # Get unique chunks and select top ones
        unique_chunks = state.get_unique_chunks()

        # Prioritize chunks from later steps (more refined queries)
        if len(unique_chunks) > self.max_context_chunks:
            # Take from most recent steps first
            recent_chunks = []
            for step in reversed(state.steps):
                for chunk in step.chunks:
                    if chunk not in recent_chunks:
                        recent_chunks.append(chunk)
                        if len(recent_chunks) >= self.max_context_chunks:
                            break
                if len(recent_chunks) >= self.max_context_chunks:
                    break
            unique_chunks = recent_chunks

        logger.info(f"Synthesizing answer from {len(unique_chunks)} chunks")

        # Build sources text with citations
        sources_text, citations = self._format_sources(unique_chunks)

        # Generate answer
        prompt = SYNTHESIS_PROMPT.format(
            question=state.original_question,
            num_sources=len(unique_chunks),
            sources=sources_text,
        )

        answer = self.controller.generate(
            prompt=prompt,
            config=self.config,
        )

        logger.info(f"Generated answer: {len(answer)} characters")

        return answer, citations

    def _format_sources(self, chunks: list[Chunk]) -> tuple[str, list[dict[str, str]]]:
        """Format chunks as numbered sources.

        Args:
            chunks: List of chunks

        Returns:
            Tuple of (formatted text, citation list)
        """
        sources = []
        citations = []

        for idx, chunk in enumerate(chunks, 1):
            # Truncate chunk text if needed
            text = chunk.text
            if len(text) > self.max_chunk_length:
                text = text[: self.max_chunk_length] + "..."

            source_text = f"[{idx}] {text}\nSource: {chunk.doc_title}"
            if chunk.doc_url:
                source_text += f" ({chunk.doc_url})"

            sources.append(source_text)

            # Build citation entry
            citation = {
                "id": str(idx),
                "title": chunk.doc_title,
                "url": chunk.doc_url or "",
                "chunk_id": chunk.chunk_id,
            }
            citations.append(citation)

        sources_text = "\n\n".join(sources)
        return sources_text, citations

    def verify_citations(self, answer: str, citations: list[dict[str, str]]) -> bool:
        """Verify that all citations in answer are valid.

        Args:
            answer: Generated answer
            citations: List of available citations

        Returns:
            True if all citations are valid
        """
        # Find all citation markers in answer
        citation_pattern = r"\[(\d+)\]"
        cited_ids = set(re.findall(citation_pattern, answer))

        # Check if all cited IDs are valid
        valid_ids = {c["id"] for c in citations}

        invalid_ids = cited_ids - valid_ids
        if invalid_ids:
            logger.warning(f"Invalid citation IDs found: {invalid_ids}")
            return False

        return True

    def format_answer_with_citations(
        self, answer: str, citations: list[dict[str, str]]
    ) -> str:
        """Format answer with citation list appended.

        Args:
            answer: Generated answer
            citations: List of citations

        Returns:
            Formatted answer with references
        """
        output = answer + "\n\n## References\n\n"

        for citation in citations:
            line = f"[{citation['id']}] {citation['title']}"
            if citation.get("url"):
                line += f" - {citation['url']}"
            output += line + "\n"

        return output
