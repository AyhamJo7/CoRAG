"""Tests for the FastAPI service."""

import json

import pytest
from fastapi.testclient import TestClient

from corag.cli import serve
from corag.corpus.document import Chunk
from corag.indexing.index import FAISSIndex
from corag.retrieval.pipeline import RetrievalPipeline
from tests.stubs import StubController, StubEmbedder


class StubSynthesizer:
    def __init__(self, fail: bool = False):
        self.fail = fail

    def synthesize(self, state):
        if self.fail:
            raise RuntimeError("secret internal failure detail")
        citations = [{"id": "1", "title": "Document 0", "url": "", "chunk_id": "c0"}]
        return "The answer [1].", citations


@pytest.fixture
def client():
    serve.app_state.clear()
    yield TestClient(serve.app)
    serve.app_state.clear()


def populate_app_state(fail_synthesis: bool = False) -> None:
    embedder = StubEmbedder(dimension=32)
    chunks = [
        Chunk(
            chunk_id=f"c{i}",
            doc_id=f"doc{i}",
            text=f"Passage {i} content.",
            start_char=0,
            end_char=20,
            tokens=4,
            doc_title=f"Document {i}",
        )
        for i in range(4)
    ]
    index = FAISSIndex(dimension=32, index_type="Flat")
    index.build(embedder.embed_texts([c.text for c in chunks]), chunks)

    controller = StubController(
        [
            json.dumps({"sub_queries": ["single sub-query"]}),
            json.dumps({"is_sufficient": True, "confidence": 0.9}),
        ]
    )
    pipeline = RetrievalPipeline(
        index=index, embedder=embedder, controller=controller, k=2, max_steps=3
    )

    serve.app_state["pipeline"] = pipeline
    serve.app_state["synthesizer"] = StubSynthesizer(fail=fail_synthesis)
    serve.app_state["model_name"] = "stub-model"


def test_healthz_unavailable_before_startup(client):
    response = client.get("/healthz")

    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"


def test_healthz_healthy_when_loaded(client):
    populate_app_state()

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_ask_returns_503_when_not_ready(client):
    response = client.post("/ask", json={"question": "Who?"})

    assert response.status_code == 503


@pytest.mark.parametrize(
    "payload",
    [
        {},  # missing question
        {"question": ""},  # empty question
        {"question": "q" * 3000},  # question too long
        {"question": "valid", "max_steps": 0},
        {"question": "valid", "max_steps": 99},
        {"question": "valid", "k": 0},
        {"question": "valid", "k": 999},
        {"question": "valid", "temperature": -0.5},
        {"question": "valid", "temperature": 3.0},
    ],
)
def test_ask_rejects_invalid_payloads(client, payload):
    populate_app_state()

    response = client.post("/ask", json=payload)

    assert response.status_code == 422


def test_ask_happy_path(client):
    populate_app_state()

    response = client.post(
        "/ask", json={"question": "What is in the corpus?", "max_steps": 3, "k": 2}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "What is in the corpus?"
    assert body["answer"] == "The answer [1]."
    assert body["citations"] == [
        {"id": "1", "title": "Document 0", "url": "", "chunk_id": "c0"}
    ]
    assert body["num_steps"] == 1
    assert body["num_chunks"] == 2


def test_ask_internal_error_is_not_leaked(client):
    populate_app_state(fail_synthesis=True)

    response = client.post("/ask", json={"question": "Trigger a failure"})

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail == "Internal error while processing the question"
    assert "secret internal failure detail" not in response.text
