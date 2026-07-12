# CLAUDE.md — CoRAG

## Research Problem

Standard RAG retrieves once and hopes for the best. **CoRAG** handles complex, multi-hop questions that require iterative evidence gathering — it decomposes queries, identifies what's still missing, retrieves again, and stops only when it has enough to answer with citations.

## Architecture

```
Question
  → Controller (LLM) — decomposes query, identifies gaps, applies self-critique
  → Retriever — FAISS dense retrieval (DPR / E5 embeddings)
  → Generator — citation-aware answer synthesis
```

Key pipeline: `query decomposition → iterative retrieval with gap analysis → self-critique stopping → cited generation`

## Package Layout

```
src/corag/
  controller/    LLM controller: query decomposition, gap analysis, self-critique
  retrieval/     Dense retriever (FAISS Flat + IVF), embedder (sentence-transformers)
  indexing/      FAISS index builder, document chunker, JSONL ingestion
  corpus/        Document schema and corpus management
  generation/    Citation-aware answer synthesis
  evaluation/    HotpotQA + 2WikiMultihopQA benchmark harness
  cli/           CLI entrypoints
```

## Setup & Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[research,dev]"   # research stack + dev tools

make install     # same as above
make lint        # ruff check + format
make typecheck   # mypy
make test        # pytest tests/
make all         # lint + typecheck + test

# CLI entrypoints (after install)
corag-ingest          # parse + chunk documents into JSONL
corag-build-index     # build FAISS index from corpus
corag-search          # ad-hoc retrieval query
corag-answer          # run full CoRAG pipeline on a question
corag-eval            # run benchmark evaluation
corag-serve           # start FastAPI service
```

Config: all CLIs are flag-driven (see `--help`); `configs/config.yaml` and `configs/prompts.yaml` document the reference defaults. Env vars via `.env` (see `.env.example`).

## Evaluation

Benchmarks: **HotpotQA** (multi-hop reasoning), **2WikiMultihopQA** (cross-document multi-hop)

Primary metrics: Exact Match (EM), F1 score, retrieval precision/recall and citation precision/recall against gold supporting documents

Run evaluation: `corag-eval --dataset hotpotqa --index-dir data/index --output reports/hotpotqa_results`

Results land in `reports/`.

## Implementation Status

All modules complete — see `PROJECT_STATUS.md` for module-by-module checklist.

## Key Dependencies

- `faiss-cpu` / `faiss-gpu` — vector index
- `sentence-transformers` — DPR, E5 embeddings
- `openai` — LLM controller (GPT-4 / GPT-3.5)
- `fastapi` + `uvicorn` — serving
- `datasets` (HuggingFace) — benchmark datasets

## Research Notes

The self-critique stopping policy is the critical mechanism — it decides when retrieved evidence is sufficient without over-retrieving. If you're changing retrieval logic, always test against both benchmarks since HotpotQA (bridge) and 2WikiMultihopQA (comparison) stress different failure modes.
