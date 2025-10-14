"""Dataset loaders for evaluation."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

from datasets import load_dataset

logger = logging.getLogger(__name__)


@dataclass
class EvalExample:
    """Evaluation example."""

    id: str
    question: str
    answer: str
    supporting_facts: List[str] = None  # Document titles or IDs
    type: str = ""  # Question type (e.g., bridge, comparison)
    level: str = ""  # Difficulty level


class DatasetLoader:
    """Loads evaluation datasets."""

    def load_hotpotqa(
        self, split: str = "validation", max_examples: Optional[int] = None
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

                # Extract supporting fact document titles
                supporting_facts = []
                if "supporting_facts" in item and item["supporting_facts"]:
                    supporting_facts = [
                        fact[0] for fact in item["supporting_facts"]["title"]
                    ]

                example = EvalExample(
                    id=item["id"],
                    question=item["question"],
                    answer=item["answer"],
                    supporting_facts=supporting_facts,
                    type=item.get("type", ""),
                    level=item.get("level", ""),
                )

                yield example
                count += 1

            logger.info(f"Loaded {count} examples from HotpotQA")

        except Exception as e:
            logger.error(f"Failed to load HotpotQA: {e}")
            raise

    def load_2wikimultihopqa(
        self, split: str = "validation", max_examples: Optional[int] = None
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

                # Extract supporting facts
                supporting_facts = []
                if "supporting_facts" in item:
                    supporting_facts = [
                        fact[0] for fact in item["supporting_facts"]
                    ]

                example = EvalExample(
                    id=item["_id"],
                    question=item["question"],
                    answer=item["answer"],
                    supporting_facts=supporting_facts,
                    type=item.get("type", ""),
                    level="",
                )

                yield example
                count += 1

            logger.info(f"Loaded {count} examples from 2WikiMultihopQA")

        except Exception as e:
            logger.error(f"Failed to load 2WikiMultihopQA: {e}")
            raise

    def load_dataset(
        self,
        dataset_name: str,
        split: str = "validation",
        max_examples: Optional[int] = None,
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
