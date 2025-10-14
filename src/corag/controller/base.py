"""Base controller interface for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    temperature: float = 0.2
    max_tokens: int = 2048
    top_p: float = 1.0
    stop: list[str] | None = None
    seed: int | None = None


class Controller(ABC):
    """Abstract base class for LLM controllers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: str | None = None,
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: User prompt
            system: System prompt
            config: Generation configuration

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens

        Returns:
            Number of tokens
        """
        pass
