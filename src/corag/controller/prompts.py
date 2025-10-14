"""System prompts for the controller agent."""

DECOMPOSITION_PROMPT = """You are a query analysis expert. Given a complex question, break it down into ordered sub-queries that need to be answered to fully address the original question.

Each sub-query should:
- Be self-contained and specific
- Focus on one aspect of the original question
- Be answerable from a knowledge base

Output format (JSON):
{
  "sub_queries": ["query1", "query2", ...],
  "rationale": "Brief explanation of the decomposition strategy"
}"""

GAP_ANALYSIS_PROMPT = """You are analyzing retrieval results for a complex question.

Original question: {question}

Sub-queries executed: {executed_queries}

Retrieved information summary: {retrieved_info}

Analyze what has been answered and what gaps remain. Identify:
1. Aspects of the original question that have been addressed
2. Missing information or unanswered aspects
3. Contradictions or inconsistencies in the retrieved information
4. Follow-up queries needed to fill gaps

Output format (JSON):
{
  "answered_aspects": ["aspect1", ...],
  "gaps": ["gap1", ...],
  "contradictions": ["contradiction1", ...],
  "follow_up_queries": ["query1", ...]
}"""

SELF_CRITIQUE_PROMPT = """You are evaluating whether enough information has been gathered to answer a complex question.

Original question: {question}

Information gathered across {num_steps} retrieval steps:
{information_summary}

Determine if:
1. All aspects of the question have been addressed
2. There is sufficient evidence for each claim
3. Contradictions have been resolved or acknowledged
4. No significant gaps remain

Output format (JSON):
{
  "is_sufficient": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "Explanation of the decision",
  "missing_aspects": ["aspect1", ...] (if not sufficient)
}"""

REFINEMENT_PROMPT = """You are refining a search query based on previous results.

Original query: {original_query}

Previous results summary: {results_summary}

Issue: {issue}

Generate a refined query that:
- Avoids redundancy with previous queries
- Targets the specific gap or issue
- Uses different phrasing or keywords

Output the refined query only (no JSON, just the query text)."""
