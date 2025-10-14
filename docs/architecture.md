# CoRAG Architecture

## Overview

CoRAG implements a multi-stage pipeline for answering complex queries through adaptive retrieval and synthesis.

## Components

### 1. Corpus & Indexing Layer

**Document Schema**
- Normalized document format with ID, title, text, URL, sections, timestamp, and metadata
- Flexible enough for Wikipedia, arXiv, or custom corpora

**Chunking Strategy**
- Token-based splitting with configurable chunk size (default: 512 tokens)
- Overlapping chunks (default: 64 tokens) to preserve context across boundaries
- Maintains provenance: each chunk tracks its source document and position

**Dense Index**
- FAISS-based similarity search with configurable index types (Flat, IVF)
- Supports inner product or L2 distance metrics
- Persistent storage with integrity checks

### 2. Controller Layer

**LLM Orchestration**
- Abstract interface supporting multiple providers (OpenAI, Anthropic, local models)
- Rate limiting and retry logic
- Token counting for budget tracking

**Prompt Engineering**
- Specialized prompts for each stage: decomposition, gap analysis, self-critique, synthesis
- JSON-structured outputs for reliable parsing
- Temperature and max_tokens configurable per use case

### 3. Retrieval Layer

**Multi-Step Loop**
1. **Decomposition**: Break complex query into sub-queries
2. **Execution**: Retrieve top-k chunks for each sub-query
3. **Deduplication**: Skip queries semantically similar to previous ones (cosine > 0.85)
4. **Gap Analysis**: Identify answered aspects and missing information
5. **Self-Critique**: Determine if sufficient information gathered
6. **Iteration**: Continue until stopping criteria met or max steps reached

**State Tracking**
- Tracks all executed queries, retrieved chunks, viewed document IDs
- Records answered aspects, gaps, contradictions
- Maintains full trace for reproducibility and debugging

### 4. Generation Layer

**Citation-Aware Synthesis**
- Aggregates evidence from unique chunks across all steps
- Generates answer with inline citations [1], [2], etc.
- Maps citations to source titles and URLs
- Verifies all citation IDs are valid

**Quality Checks**
- Every claim must be traceable to a retrieved passage
- Contradictions acknowledged explicitly
- Insufficient information flagged

## Data Flow

```
User Query
    │
    ▼
┌───────────────────────────────────────────┐
│ Controller: Query Decomposition           │
│ Input: Complex query                      │
│ Output: List of sub-queries               │
└───────────────┬───────────────────────────┘
                │
                ▼
    ┌───────────────────────┐
    │ For each sub-query:   │
    │   1. Embed query      │
    │   2. Search FAISS     │
    │   3. Retrieve top-k   │
    └───────┬───────────────┘
            │
            ▼
┌───────────────────────────────────────────┐
│ Controller: Gap Analysis                  │
│ Input: Original query + retrieved info    │
│ Output: Answered aspects + gaps           │
└───────────────┬───────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────┐
│ Controller: Self-Critique                 │
│ Input: All retrieved info + question      │
│ Output: Sufficient? (yes/no)              │
└───────────────┬───────────────────────────┘
                │
                ├─→ No: Generate follow-up query → Loop
                │
                ▼ Yes
┌───────────────────────────────────────────┐
│ Synthesis: Generate Answer                │
│ Input: Unique chunks + question           │
│ Output: Answer with citations             │
└───────────────┬───────────────────────────┘
                │
                ▼
            Final Answer
```

## Design Decisions

### Why Dense Retrieval?

- Better semantic matching than BM25/TF-IDF for complex queries
- Handles paraphrasing and synonym variations
- Enables query deduplication via embedding similarity

### Why Multi-Step?

Complex queries like "Who was the first person to walk on the moon and when did they return to Earth?" require:
1. Identifying who walked on the moon (Neil Armstrong)
2. Finding when he returned (July 24, 1969)

A single retrieval may retrieve passages about the moon landing but miss the return date. Multi-step retrieval allows targeted follow-up queries.

### Stopping Criteria

The loop stops when:
1. **Self-critique satisfied**: LLM determines sufficient information gathered
2. **Max steps reached**: Prevents infinite loops (default: 6 steps)
3. **No new queries**: Follow-up queries are duplicates of previous queries

### Citation Strategy

Inline citations `[1]`, `[2]` rather than footnotes for:
- Clear mapping between claims and sources
- Easy verification of cited information
- Standard format familiar to users

## Scalability Considerations

- **Index size**: FAISS IVF for large corpora (>1M chunks)
- **Embedding batch size**: Tune based on available memory
- **LLM calls**: ~3-6 calls per question (can be expensive)
- **Latency**: 5-15 seconds typical for 6-step retrieval + generation
