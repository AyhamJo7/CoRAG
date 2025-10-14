"""Multi-step retrieval pipeline with adaptive refinement."""

import json
import logging
from typing import List, Optional, Tuple

from corag.controller.base import Controller, GenerationConfig
from corag.controller.prompts import (
    DECOMPOSITION_PROMPT,
    GAP_ANALYSIS_PROMPT,
    SELF_CRITIQUE_PROMPT,
)
from corag.indexing.embedder import Embedder
from corag.indexing.index import FAISSIndex
from corag.retrieval.state import RetrievalState, RetrievalStep
from corag.utils.text import truncate_text

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """Multi-step retrieval with query decomposition and gap analysis."""

    def __init__(
        self,
        index: FAISSIndex,
        embedder: Embedder,
        controller: Controller,
        k: int = 8,
        max_steps: int = 6,
        similarity_threshold: float = 0.85,
        config: Optional[GenerationConfig] = None,
    ):
        """Initialize retrieval pipeline.

        Args:
            index: FAISS index for retrieval
            embedder: Text embedder
            controller: LLM controller
            k: Number of chunks to retrieve per query
            max_steps: Maximum retrieval steps
            similarity_threshold: Threshold for query deduplication
            config: Generation configuration
        """
        self.index = index
        self.embedder = embedder
        self.controller = controller
        self.k = k
        self.max_steps = max_steps
        self.similarity_threshold = similarity_threshold
        self.config = config or GenerationConfig()

    def run(self, question: str) -> RetrievalState:
        """Run multi-step retrieval loop.

        Args:
            question: Complex query

        Returns:
            RetrievalState with all steps
        """
        state = RetrievalState(original_question=question)

        # Step 1: Decompose query
        logger.info(f"Starting retrieval for: {question}")
        sub_queries = self._decompose_query(question)

        logger.info(f"Decomposed into {len(sub_queries)} sub-queries")

        # Execute sub-queries
        for step_num, query in enumerate(sub_queries, 1):
            if step_num > self.max_steps:
                logger.info(f"Reached max steps ({self.max_steps})")
                break

            # Check for duplicate queries
            if self._is_duplicate_query(query, state.executed_queries):
                logger.info(f"Skipping duplicate query: {query}")
                continue

            # Retrieve
            chunks, scores = self._retrieve(query)
            step = RetrievalStep(
                step_num=step_num,
                query=query,
                chunks=chunks,
                scores=scores,
                rationale=f"Sub-query {step_num} from decomposition",
            )
            state.add_step(step)

            logger.info(
                f"Step {step_num}: Retrieved {len(chunks)} chunks for '{query[:50]}...'"
            )

        # Iterative refinement loop
        while len(state.steps) < self.max_steps:
            # Check if we have enough information
            is_sufficient = self._self_critique(state)

            if is_sufficient:
                logger.info("Self-critique: Sufficient information gathered")
                state.is_sufficient = True
                break

            # Analyze gaps
            gaps = self._analyze_gaps(state)

            if not gaps:
                logger.info("No more gaps identified")
                break

            # Generate follow-up query
            follow_up = gaps[0]  # Take first gap
            if self._is_duplicate_query(follow_up, state.executed_queries):
                logger.info("Follow-up query is duplicate, stopping")
                break

            # Retrieve with follow-up
            chunks, scores = self._retrieve(follow_up)
            step = RetrievalStep(
                step_num=len(state.steps) + 1,
                query=follow_up,
                chunks=chunks,
                scores=scores,
                rationale="Gap-filling follow-up query",
            )
            state.add_step(step)

            logger.info(
                f"Step {step.step_num}: Follow-up retrieved {len(chunks)} chunks"
            )

        logger.info(
            f"Retrieval complete: {len(state.steps)} steps, "
            f"{state.total_chunks_retrieved} total chunks, "
            f"{len(state.get_unique_chunks())} unique chunks"
        )

        return state

    def _decompose_query(self, question: str) -> List[str]:
        """Decompose complex query into sub-queries.

        Args:
            question: Original question

        Returns:
            List of sub-queries
        """
        prompt = f"Question: {question}\n\nDecompose this into sub-queries."

        try:
            response = self.controller.generate(
                prompt=prompt,
                system=DECOMPOSITION_PROMPT,
                config=self.config,
            )

            # Parse JSON response
            data = json.loads(response)
            sub_queries = data.get("sub_queries", [question])

            if not sub_queries:
                return [question]

            return sub_queries

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse decomposition: {e}. Using original query.")
            return [question]

    def _retrieve(self, query: str) -> Tuple[List, List[float]]:
        """Retrieve chunks for a query.

        Args:
            query: Search query

        Returns:
            Tuple of (chunks, scores)
        """
        query_embedding = self.embedder.embed_query(query)
        chunks, scores = self.index.search(query_embedding, k=self.k)
        return chunks, scores

    def _is_duplicate_query(self, query: str, executed_queries: List[str]) -> bool:
        """Check if query is too similar to executed queries.

        Args:
            query: Query to check
            executed_queries: Previously executed queries

        Returns:
            True if duplicate
        """
        if not executed_queries:
            return False

        query_emb = self.embedder.embed_query(query)

        for prev_query in executed_queries:
            prev_emb = self.embedder.embed_query(prev_query)

            # Cosine similarity (embeddings are normalized)
            similarity = float(query_emb.dot(prev_emb))

            if similarity > self.similarity_threshold:
                logger.debug(
                    f"Query similarity {similarity:.3f} > threshold "
                    f"{self.similarity_threshold}"
                )
                return True

        return False

    def _analyze_gaps(self, state: RetrievalState) -> List[str]:
        """Analyze gaps in retrieved information.

        Args:
            state: Current retrieval state

        Returns:
            List of follow-up queries
        """
        # Build summary of retrieved information
        unique_chunks = state.get_unique_chunks()
        info_summary = "\n\n".join(
            [f"- {truncate_text(c.text, 200)}" for c in unique_chunks[:10]]
        )

        prompt = f"""Executed queries: {', '.join(state.executed_queries)}

Information retrieved:
{info_summary}
"""

        try:
            response = self.controller.generate(
                prompt=GAP_ANALYSIS_PROMPT.format(
                    question=state.original_question,
                    executed_queries=", ".join(state.executed_queries),
                    retrieved_info=info_summary,
                ),
                config=self.config,
            )

            data = json.loads(response)
            state.answered_aspects = data.get("answered_aspects", [])
            state.gaps = data.get("gaps", [])
            state.contradictions = data.get("contradictions", [])

            follow_ups = data.get("follow_up_queries", [])
            return follow_ups

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse gap analysis: {e}")
            return []

    def _self_critique(self, state: RetrievalState) -> bool:
        """Check if enough information has been gathered.

        Args:
            state: Current retrieval state

        Returns:
            True if sufficient
        """
        unique_chunks = state.get_unique_chunks()
        info_summary = "\n\n".join(
            [f"- {truncate_text(c.text, 150)}" for c in unique_chunks[:15]]
        )

        prompt = SELF_CRITIQUE_PROMPT.format(
            question=state.original_question,
            num_steps=len(state.steps),
            information_summary=info_summary,
        )

        try:
            response = self.controller.generate(
                prompt=prompt,
                config=self.config,
            )

            data = json.loads(response)
            is_sufficient = data.get("is_sufficient", False)
            confidence = data.get("confidence", 0.0)

            logger.info(
                f"Self-critique: sufficient={is_sufficient}, confidence={confidence:.2f}"
            )

            return is_sufficient

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse self-critique: {e}")
            return False
