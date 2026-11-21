# CoRAG Agent Notes

CoRAG is a research portfolio repo for adaptive multi-step retrieval over
complex, multi-hop questions.

Read `CLAUDE.md` when present; in this workspace it may be local/untracked.

## Architecture

- `src/corag/controller/`: LLM controller for query decomposition, gap analysis,
  and self-critique stopping.
- `src/corag/retrieval/`: retrieval state and iterative retrieval loop.
- `src/corag/indexing/`: FAISS indexing and embedding.
- `src/corag/corpus/`: document schemas, chunking, JSONL ingestion.
- `src/corag/generation/`: citation-aware answer synthesis.
- `src/corag/evaluation/`: HotpotQA / 2WikiMultihopQA evaluation.
- `src/corag/cli/`: CLI entrypoints.

The research point is not "RAG with a vector DB"; it is the controller loop:
decompose -> retrieve -> identify gaps -> retrieve again -> stop only when
evidence is sufficient -> generate with citations.

## Commands

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[research,dev]"
make lint
make typecheck
make test
make all
```

CLI surfaces after install include `corag-ingest`, `corag-build-index`,
`corag-search`, `corag-answer`, `corag-eval`, and `corag-serve`.

## Editing Rules

- Preserve citation traceability from generated claims back to retrieved chunks.
- If changing retrieval logic, test against both HotpotQA and 2WikiMultihopQA
  because they stress different multi-hop failure modes.
- Keep benchmark data, generated indexes, reports, and secrets out of git.
- This repo uses Python research conventions; add type hints and keep ruff/mypy
  clean before considering work complete.
