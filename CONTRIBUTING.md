# Contributing to CoRAG

Thank you for your interest in contributing to CoRAG! This document provides
guidelines for contributing to the project.

## Development Setup

### Prerequisites

* Python 3.10 or higher
* Git

### Setting Up Your Environment

1. Fork and clone the repository:
```bash
git clone https://github.com/AyhamJo77/CoRAG.git
cd CoRAG
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e ".[research,dev]"
```

4. Copy the environment template:
```bash
cp .env.example .env
```

5. Configure your API keys in `.env`

## Code Style

We use the following tools to maintain code quality:

* **ruff** - Linting and formatting
* **mypy** - Type checking
* **black** - Code formatting (via ruff)

Run checks before committing:
```bash
ruff check src/ tests/
ruff format src/ tests/
mypy src/
```

### Style Guidelines

* Use type hints for all function signatures
* Maximum line length: 100 characters
* Use double quotes for strings
* Follow PEP 8 conventions
* Write docstrings for all public functions and classes

## Testing

We use pytest for testing. Run tests with:

```bash
pytest
pytest --cov=corag  # With coverage report
```

### Writing Tests

* Place tests in the `tests/` directory
* Name test files `test_*.py`
* Use descriptive test function names: `test_<what>_<condition>_<expected>`
* Mock external dependencies (API calls, file I/O when appropriate)

## Commit Message Conventions

Follow these guidelines for commit messages:

* Use present tense ("add feature" not "added feature")
* Use imperative mood ("move cursor to..." not "moves cursor to...")
* Start with a lowercase verb
* Keep the first line under 72 characters
* Reference issues and PRs when applicable

Examples:
```
add query deduplication to retrieval loop
fix FAISS index loading with IVF configuration
update evaluation metrics to include precision@k
docs: clarify chunking parameters in README
```

## Pull Request Process

1. Create a feature branch from `main`:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and commit them with descriptive messages

3. Push your branch and create a pull request:
```bash
git push origin feature/your-feature-name
```

4. Ensure all CI checks pass

5. Request review from maintainers

### PR Requirements

* All tests must pass
* Code coverage should not decrease
* Include tests for new functionality
* Update documentation as needed
* Add entry to CHANGELOG.md under "Unreleased"

## Project Structure

```
CoRAG/
```
