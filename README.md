# CoRAG: Adaptive Multi-Step Retrieval for Complex Queries

[![CI](https://github.com/AyhamJo7/CoRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/AyhamJo7/CoRAG/actions/workflows/ci.yml)[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="data/CoRAG.png" alt="CoRAG Logo" width="400"/>
</p>


CoRAG is a production-grade system for answering complex, multi-hop questions through adaptive retrieval. Unlike traditional RAG systems that retrieve once, CoRAG iteratively decomposes queries, identifies information gaps, and refines its search until sufficient evidence is gathered.

## Key Features

- **Adaptive Multi-Step Retrieval**: Automatically decomposes complex queries and iteratively retrieves information
- **Gap Analysis**: Identifies missing information and generates targeted follow-up queries
- **Self-Critique**: Determines when sufficient information has been gathered
- **Citation-Aware Generation**: Produces answers with inline source citations
- **Dense Retrieval**: FAISS-based indexing with state-of-the-art embeddings (DPR, E5)
- **Evaluation Framework**: Built-in support for HotpotQA and 2WikiMultihopQA benchmarks
- **Production Ready**: FastAPI service, comprehensive tests, type hints throughout

## Architecture

```
┌─────────────┐
│   Question  │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Controller (LLM)                   │
│  - Query Decomposition              │
│  - Gap Analysis                     │
│  - Self-Critique                    │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Multi-Step Retrieval Loop          │
│  ┌─────────────────────────────┐   │
│  │ 1. Execute Sub-Query        │   │
│  │ 2. Retrieve from FAISS      │   │
│  │ 3. Analyze Gaps             │   │
│  │ 4. Check Sufficiency        │   │
│  └─────────────────────────────┘   │
│         │                           │
│         └─→ Repeat if needed        │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Synthesis & Generation             │
│  - Aggregate Evidence               │
│  - Generate with Citations          │
│  - Resolve Contradictions           │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────┐
│   Answer    │
└─────────────┘
```

## Quick Start

### Installation

```bash
git clone https://github.com/AyhamJo77/CoRAG.git
cd CoRAG
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

### Configuration

Copy the environment template and configure your API keys:

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Build an Index

First, prepare your corpus as JSONL (one document per line):

```json
{"id": "1", "title": "Paris", "text": "Paris is the capital of France..."}
{"id": "2", "title": "France", "text": "France is a country in Europe..."}
```

Then build the index:

```bash
corag-build-index data/corpus.jsonl data/index \
  --embedding-model sentence-transformers/msmarco-distilbert-base-v4 \
  --index-type Flat \
  --chunk-size 512 \
  --chunk-overlap 64
```

### Run a Query

```bash
corag-search \
  --question "Who was the first person to walk on the moon and when did they return to Earth?" \
  --index-dir data/index \
  --max-steps 6 \
  --k 8 \
  --output trace.json
```

### Generate an Answer

```bash
corag-answer \
  --trace trace.json \
  --output answer.md
```

### Start the API Server

```bash
corag-serve --index-dir data/index --host 0.0.0.0 --port 8000
```

Then query via HTTP:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of France?", "max_steps": 4, "k": 8}'
```

## Evaluation

Run evaluation on HotpotQA:

```bash
corag-eval \
  --dataset hotpotqa \
  --split validation \
  --index-dir data/index \
  --max-steps 6 \
  --k 8 \
  --max-examples 100 \
  --output reports/hotpotqa_results
```

Results are saved as JSON (summary) and CSV (detailed).

## Project Structure

```
CoRAG/
├── src/corag/              # Main package
│   ├── corpus/            # Document ingestion and chunking
│   │   ├── document.py    # Document/Chunk schemas
│   │   ├── chunker.py     # Token-based chunking
│   │   └── ingest.py      # JSONL ingestion
│   ├── indexing/          # Dense indexing with FAISS
│   │   ├── embedder.py    # Sentence transformers
│   │   └── index.py       # FAISS index with persistence
│   ├── controller/        # LLM orchestration
│   │   ├── base.py        # Controller interface
│   │   ├── openai_controller.py
│   │   └── prompts.py     # System prompts
│   ├── retrieval/         # Multi-step retrieval
│   │   ├── state.py       # State tracking
│   │   └── pipeline.py    # Retrieval loop
│   ├── generation/        # Answer synthesis
│   │   └── synthesizer.py # Citation-aware generation
│   ├── evaluation/        # Evaluation framework
│   │   ├── metrics.py     # EM, F1, retrieval metrics
│   │   ├── datasets.py    # HotpotQA, 2WikiMultihop loaders
│   │   └── evaluator.py   # Evaluation harness
│   ├── cli/               # Command-line interfaces
│   └── utils/             # Text processing utilities
├── tests/                 # Comprehensive test suite
├── scripts/               # Standalone scripts
├── docs/                  # Additional documentation
├── configs/               # Configuration files
├── examples/              # Example queries and outputs
└── data/                  # Data directory (gitignored)
```

## Design Decisions

### Why Multi-Step Retrieval?

Complex questions often require information from multiple sources that can't be retrieved in a single query. CoRAG:

1. **Decomposes** complex queries into simpler sub-queries
2. **Retrieves** information for each sub-query
3. **Analyzes** what's been answered and what gaps remain
4. **Iterates** until sufficient information is gathered

### Query Deduplication

To avoid redundant retrieval, CoRAG checks similarity between new queries and previously executed ones using embedding cosine similarity (default threshold: 0.85).

### Stopping Policy

The controller performs self-critique after each iteration, determining if:
- All aspects of the question have been addressed
- Sufficient evidence exists for each claim
- Contradictions have been resolved

### Citation Traceability

Every claim in the generated answer must be traceable to retrieved passages. Citations use inline markers `[1]`, `[2]`, etc., mapped to document titles and URLs.

## Performance

Benchmark results on HotpotQA (validation, 100 examples):

| Metric | Score |
|--------|-------|
| Exact Match | TBD |
| F1 Score | TBD |
| Avg Steps | TBD |
| Avg Latency | TBD |

*Run `corag-eval` to generate your own benchmarks.*

## Limitations

- **Corpus Dependency**: Quality depends on the indexed corpus
- **LLM Costs**: Multiple LLM calls per question can be expensive
- **Latency**: Multi-step retrieval increases response time
- **Language**: Currently optimized for English

## Development

### Running Tests

```bash
pytest
pytest --cov=corag --cov-report=html
```

### Code Quality

```bash
ruff check src/ tests/
ruff format src/ tests/
mypy src/
```

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use CoRAG in your research, please cite:

```bibtex
@software{corag2025,
  author = {AyhamJo77},
  title = {CoRAG: Adaptive Multi-Step Retrieval for Complex Queries},
  year = {2025},
  url = {https://github.com/AyhamJo77/CoRAG}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Built on [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- Uses [Sentence Transformers](https://www.sbert.net/) for embeddings
- Evaluation on [HotpotQA](https://hotpotqa.github.io/) and [2WikiMultihopQA](https://github.com/Alab-NII/2wikimultihop)
