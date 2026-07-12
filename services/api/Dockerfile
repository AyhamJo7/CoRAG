# Build from the repo root: docker build -f services/api/Dockerfile .
# Installs the corag BASE package (no torch/faiss research extra) + corag_cloud.
FROM python:3.11-slim

RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY services/api/pyproject.toml services/api/alembic.ini ./services/api/
COPY services/api/src ./services/api/src
COPY services/api/alembic ./services/api/alembic

RUN pip install --no-cache-dir ./ ./services/api

USER app
EXPOSE 8000
CMD ["uvicorn", "corag_cloud.main:app", "--host", "0.0.0.0", "--port", "8000"]
