"""Dataset loaders for evaluation."""

import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

logger = logging.getLogger(__name__)


@dataclass
class EvalExample:
    """Evaluation example."""

    id: str
    question: str
    answer: str
    supporting_facts: list[str] | None = None  # Document titles or IDs
    type: str = ""  # Question type (e.g., bridge, comparison)
    level: str = ""  # Difficulty level


def extract_supporting_titles(supporting_facts: Any) -> list[str]:
    """Extract unique gold document titles from a supporting_facts field.

    Handles both encodings used by the HuggingFace multi-hop QA datasets:
    a columnar mapping ``{"title": [...], "sent_id": [...]}`` (HotpotQA
    distractor) and a row-wise list of ``[title, sent_id]`` pairs
    (2WikiMultihopQA raw JSON).

    Args:
        supporting_facts: Raw supporting_facts value from a dataset item

    Returns:
        Unique document titles, insertion-ordered
    """
    titles: list[str] = []
    if isinstance(supporting_facts, Mapping):
        titles = [str(t) for t in supporting_facts.get("title", [])]
    elif isinstance(supporting_facts, list):
        titles = [
            str(fact[0])
            for fact in supporting_facts
            if isinstance(fact, (list, tuple)) and fact
        ]
    return list(dict.fromkeys(titles))


def example_from_hotpotqa(item: Mapping[str, Any]) -> EvalExample:
    """Convert a raw HotpotQA dataset row to an EvalExample.

    Args:
        item: Raw dataset row

    Returns:
        EvalExample with gold supporting-document titles
    """
    return EvalExample(
        id=item["id"],
        question=item["question"],
        answer=item["answer"],
        supporting_facts=extract_supporting_titles(item.get("supporting_facts")),
        type=item.get("type", ""),
        level=item.get("level", ""),
    )


def example_from_2wiki(item: Mapping[str, Any]) -> EvalExample:
    """Convert a raw 2WikiMultihopQA dataset row to an EvalExample.

    Args:
        item: Raw dataset row

    Returns:
        EvalExample with gold supporting-document titles
    """
    return EvalExample(
        id=item["_id"],
        question=item["question"],
        answer=item["answer"],
        supporting_facts=extract_supporting_titles(item.get("supporting_facts")),
        type=item.get("type", ""),
        level="",
    )


class DatasetLoader:
    """Loads evaluation datasets."""

    def load_hotpotqa(
        self, split: str = "validation", max_examples: int | None = None
    ) -> Iterator[EvalExample]:
        """Load HotpotQA dataset.

        Args:
            split: Dataset split (train/validation)
            max_examples: Maximum examples to load

        Yields:
            EvalExample objects
        """
        logger.info(f"Loading HotpotQA {split} split...")

        try:
            dataset = load_dataset("hotpot_qa", "distractor", split=split)

            count = 0
            for item in dataset:
                if max_examples and count >= max_examples:
                    break

                yield example_from_hotpotqa(item)
                count += 1

            logger.info(f"Loaded {count} examples from HotpotQA")

        except Exception as e:
            logger.error(f"Failed to load HotpotQA: {e}")
            raise

    def load_2wikimultihopqa(
        self, split: str = "validation", max_examples: int | None = None
    ) -> Iterator[EvalExample]:
        """Load 2WikiMultihopQA dataset.

        Args:
            split: Dataset split (train/dev/test)
            max_examples: Maximum examples to load

        Yields:
            EvalExample objects
        """
        logger.info(f"Loading 2WikiMultihopQA {split} split...")

        # Map split name
        split_map = {"validation": "dev", "train": "train", "test": "test"}
        split = split_map.get(split, split)

        try:
            dataset = load_dataset("THUDM/2WikiMultihopQA", split=split)

            count = 0
            for item in dataset:
                if max_examples and count >= max_examples:
                    break

                yield example_from_2wiki(item)
                count += 1

            logger.info(f"Loaded {count} examples from 2WikiMultihopQA")

        except Exception as e:
            logger.error(f"Failed to load 2WikiMultihopQA: {e}")
            raise

    def load_dataset(
        self,
        dataset_name: str,
        split: str = "validation",
        max_examples: int | None = None,
    ) -> Iterator[EvalExample]:
        """Load a dataset by name.

        Args:
            dataset_name: Name of dataset (hotpotqa, 2wikimultihopqa)
            split: Dataset split
            max_examples: Maximum examples to load

        Yields:
            EvalExample objects
        """
        if dataset_name.lower() == "hotpotqa":
            yield from self.load_hotpotqa(split, max_examples)
        elif dataset_name.lower() == "2wikimultihopqa":
            yield from self.load_2wikimultihopqa(split, max_examples)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
