.PHONY: help install test lint format clean

help:
	@echo "CoRAG Development Commands"
	@echo ""
	@echo "  install     Install package and dependencies"
	@echo "  test        Run test suite with coverage"
	@echo "  lint        Run linting checks"
	@echo "  format      Format code with ruff"
	@echo "  typecheck   Run mypy type checking"
	@echo "  clean       Remove build artifacts and caches"
	@echo "  all         Run lint, typecheck, and test"

install:
	pip install -e ".[dev]"

test:
	pytest --cov=corag --cov-report=term-missing --cov-report=html

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

typecheck:
	mypy src/

clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

all: lint typecheck test
	@echo "All checks passed!"
