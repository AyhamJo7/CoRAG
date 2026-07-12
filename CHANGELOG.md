# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Gap-analysis and self-critique prompt templates crashed on `.format()` due to
  unescaped JSON braces, breaking the iterative refinement loop
- HotpotQA loader extracted the first character of each supporting-fact title
  instead of the full title
- `/ask` no longer returns internal exception details in 500 responses and no
  longer blocks the event loop during retrieval

### Added
- Retrieval and citation precision/recall against gold supporting documents in
  the evaluation report, CSV output, and `corag-eval` summary
- Request validation bounds on `/ask` (question length, `max_steps`, `k`,
  `temperature`) and a readiness-aware `/healthz` (503 until loaded)
- Explicit OpenAI request timeout with retries limited to transient errors
- Tests for the retrieval pipeline loop, dataset loaders, controller retry
  policy, and the FastAPI service

### Removed
- Unused dependencies (`anthropic`, `transformers`, `pydantic-settings`,
  `pyyaml`, `requests`, `black`) and empty placeholder scripts

## [0.1.0] - 2025-10-14

### Added
- Initial release of CoRAG
- Document ingestion and chunking for Wikipedia corpora
- Dense indexing with FAISS (Flat and IVF modes)
- DPR and E5 embedding support via Hugging Face
- Controller agent with pluggable LLM providers (OpenAI, Anthropic, local)
- Multi-step retrieval loop with query decomposition
- Gap analysis and iterative refinement
- Self-critique stopping policy
- Answer synthesis with inline citations
- Evaluation harness for HotpotQA and 2WikiMultihopQA
- CLI tools: ingest, build-index, search, answer, eval
- FastAPI REST API with /ask endpoint
- Comprehensive test suite
- Type hints and mypy configuration
- CI/CD with GitHub Actions
- Full documentation and examples
