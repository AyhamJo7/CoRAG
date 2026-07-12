"""Tests for the OpenAI controller retry policy."""

from types import SimpleNamespace

import httpx
import pytest
from openai import AuthenticationError, RateLimitError

from corag.controller.openai_controller import OpenAIController


def make_response(content: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=None,
    )


def make_api_error(error_cls, status_code: int):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return error_cls("simulated error", response=response, body=None)


class FakeCompletions:
    """Replays a queue of responses; raises entries that are exceptions."""

    def __init__(self, results: list):
        self.results = list(results)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture
def controller():
    return OpenAIController(api_key="test-key", max_retries=3, retry_delay=0.0)


def install(controller: OpenAIController, results: list) -> FakeCompletions:
    fake = FakeCompletions(results)
    controller.client = SimpleNamespace(chat=SimpleNamespace(completions=fake))  # type: ignore[assignment]
    return fake


def test_generate_returns_text(controller):
    fake = install(controller, [make_response("hello world")])

    assert controller.generate("prompt") == "hello world"
    assert fake.calls == 1


def test_rate_limit_is_retried(controller):
    fake = install(
        controller,
        [make_api_error(RateLimitError, 429), make_response("recovered")],
    )

    assert controller.generate("prompt") == "recovered"
    assert fake.calls == 2


def test_empty_completion_is_retried(controller):
    fake = install(controller, [make_response(None), make_response("second try")])

    assert controller.generate("prompt") == "second try"
    assert fake.calls == 2


def test_auth_error_is_not_retried(controller):
    fake = install(controller, [make_api_error(AuthenticationError, 401)])

    with pytest.raises(AuthenticationError):
        controller.generate("prompt")
    assert fake.calls == 1


def test_raises_after_exhausting_retries(controller):
    fake = install(
        controller,
        [make_api_error(RateLimitError, 429) for _ in range(3)],
    )

    with pytest.raises(RateLimitError):
        controller.generate("prompt")
    assert fake.calls == 3
