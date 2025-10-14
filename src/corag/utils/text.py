"""Text processing utilities."""

import re
from typing import List


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    # Normalize unicode quotes
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    return text


def count_tokens_approximate(text: str) -> int:
    """Approximate token count (words + punctuation)."""
    # Simple approximation: split on whitespace and count
    # More accurate would use tiktoken but this is faster
    tokens = text.split()
    # Add some overhead for punctuation
    return int(len(tokens) * 1.3)


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Simple sentence splitter
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def truncate_text(text: str, max_chars: int) -> str:
    """Truncate text to max characters at word boundary."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(' ', 1)[0]
    return truncated + '...'
