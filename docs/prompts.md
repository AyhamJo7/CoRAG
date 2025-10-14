# Prompt Specifications

## Overview

CoRAG uses specialized prompts for each stage of the retrieval loop. All prompts request JSON-structured output for reliable parsing.

## Query Decomposition

**Purpose**: Break complex queries into ordered sub-queries

**Input**:
- Original complex question

**Output** (JSON):
```json
{
  "sub_queries": ["sub-query 1", "sub-query 2", ...],
  "rationale": "Explanation of decomposition strategy"
}
```

**Example**:
```
Question: "Who was the first person to walk on the moon and when did they return to Earth?"

Output:
{
  "sub_queries": [
    "Who was the first person to walk on the moon?",
    "When did the first moon walker return to Earth?"
  ],
  "rationale": "First identify the person, then find their return date"
}
```

## Gap Analysis

**Purpose**: Identify answered aspects and information gaps

**Input**:
- Original question
- Executed queries
- Retrieved information summary

**Output** (JSON):
```json
{
  "answered_aspects": ["aspect 1", ...],
  "gaps": ["gap 1", ...],
  "contradictions": ["contradiction 1", ...],
  "follow_up_queries": ["follow-up 1", ...]
}
```

## Self-Critique

**Purpose**: Determine if sufficient information has been gathered

**Input**:
- Original question
- Number of steps taken
- Information summary across all steps

**Output** (JSON):
```json
{
  "is_sufficient": true/false,
  "confidence": 0.8,
  "reasoning": "Explanation of decision",
  "missing_aspects": ["aspect 1", ...] 
}
```

**Heuristics**:
- All question aspects addressed
- Evidence sufficient for claims
- Contradictions resolved
- No significant gaps

## Synthesis

**Purpose**: Generate comprehensive answer with citations

**Input**:
- Original question
- Retrieved passages with [1], [2], ... markers
- Source metadata (titles, URLs)

**Output**: Free-form answer with inline citations

**Requirements**:
- Use ONLY provided sources
- Cite with [1], [2], etc.
- Acknowledge contradictions
- State if information insufficient
- Structure clearly (sections if needed)

**Example**:
```
Paris is the capital and most populous city of France [1]. It has an estimated population of 2.2 million residents [2].
```

## Tuning Tips

- **Temperature**: Lower (0.2) for decomposition/gap analysis, moderate (0.3) for synthesis
- **Max Tokens**: 2048 typically sufficient for all stages
- **JSON Parsing**: Always wrap in try-except; fall back to sensible defaults on parse failure
- **Prompt Length**: Keep context under ~3000 tokens to avoid truncation
