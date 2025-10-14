"""Base controller interface for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    temperature: float = 0.2
    max_tokens: int = 2048
    top_p: float = 1.0
    stop: Optional[list[str]] = None
    seed: Optional[int] = None


class Controller(ABC):
    """Abstract base class for LLM controllers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
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
