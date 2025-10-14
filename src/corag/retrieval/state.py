"""Retrieval state tracking."""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from corag.corpus.document import Chunk


@dataclass
class RetrievalStep:
    """Represents one step in the retrieval loop."""

    step_num: int
    query: str
    chunks: List[Chunk]
    scores: List[float]
    rationale: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalState:
    """Tracks state across multi-step retrieval."""

    original_question: str
    steps: List[RetrievalStep] = field(default_factory=list)
    executed_queries: List[str] = field(default_factory=list)
    viewed_chunk_ids: set[str] = field(default_factory=set)
    answered_aspects: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    is_sufficient: bool = False
    total_chunks_retrieved: int = 0
    total_tokens_used: int = 0

    def add_step(self, step: RetrievalStep) -> None:
        """Add a retrieval step."""
        self.steps.append(step)
        self.executed_queries.append(step.query)
        for chunk in step.chunks:
            self.viewed_chunk_ids.add(chunk.chunk_id)
        self.total_chunks_retrieved += len(step.chunks)

    def get_all_chunks(self) -> List[Chunk]:
        """Get all retrieved chunks across steps."""
        all_chunks = []
        for step in self.steps:
            all_chunks.extend(step.chunks)
        return all_chunks

    def get_unique_chunks(self) -> List[Chunk]:
        """Get unique chunks (deduplicated by chunk_id)."""
        seen = set()
        unique = []
        for step in self.steps:
            for chunk in step.chunks:
                if chunk.chunk_id not in seen:
                    seen.add(chunk.chunk_id)
                    unique.append(chunk)
        return unique

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "original_question": self.original_question,
            "steps": [
                {
                    "step_num": s.step_num,
                    "query": s.query,
                    "num_chunks": len(s.chunks),
                    "chunks": [c.to_dict() for c in s.chunks],
                    "scores": s.scores,
                    "rationale": s.rationale,
                    "metadata": s.metadata,
                }
                for s in self.steps
            ],
            "executed_queries": self.executed_queries,
            "viewed_chunk_ids": list(self.viewed_chunk_ids),
            "answered_aspects": self.answered_aspects,
            "gaps": self.gaps,
            "contradictions": self.contradictions,
            "is_sufficient": self.is_sufficient,
            "total_chunks_retrieved": self.total_chunks_retrieved,
            "total_tokens_used": self.total_tokens_used,
        }
