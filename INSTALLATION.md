# Installation Guide

## Quick Install

```bash
# Clone repository
git clone https://github.com/AyhamJo77/CoRAG.git
cd CoRAG

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package with dev dependencies
pip install -e ".[dev]"

# Verify installation
corag-search --help
```

## Dependencies

CoRAG requires Python 3.10 or higher.

### Core Dependencies
- numpy: Numerical computations
- torch: PyTorch for model inference
- transformers: HuggingFace transformers
- sentence-transformers: Text embeddings
- faiss-cpu: Dense retrieval (or faiss-gpu for GPU support)
- openai: OpenAI API client
- pydantic: Data validation
- fastapi: API server
- click: CLI framework

### Development Dependencies
- pytest: Testing framework
- ruff: Linting and formatting
- mypy: Type checking
- black: Code formatting

## GPU Support

For GPU acceleration of embeddings:

```bash
# Install GPU version of FAISS
pip install faiss-gpu

# Or build from source for optimal performance
# See: https://github.com/facebookresearch/faiss/blob/main/INSTALL.md
```

## Verification

Run tests to verify installation:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=corag --cov-report=html

# Run specific test module
pytest tests/test_corpus.py -v
```

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`, ensure the virtual environment is activated:

```bash
source venv/bin/activate
which python  # Should point to venv/bin/python
```

### FAISS Installation Issues

On some systems, FAISS may require additional build tools:

```bash
# Ubuntu/Debian
sudo apt-get install build-essential

# macOS
xcode-select --install
```

### OpenAI API Key

Don't forget to set your API key:

```bash
export OPENAI_API_KEY="sk-..."
# Or add to .env file
echo "OPENAI_API_KEY=sk-..." >> .env
```

## Next Steps

After installation:

1. Prepare your corpus as JSONL
2. Build an index with `corag-build-index`
3. Run queries with `corag-search`
4. Generate answers with `corag-answer`

See [README.md](README.md) for detailed usage instructions.
