"""Evaluation harness for CoRAG."""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

from corag.evaluation.datasets import DatasetLoader, EvalExample
from corag.evaluation.metrics import (
    max_em_over_ground_truths,
    max_f1_over_ground_truths,
)
from corag.generation.synthesizer import Synthesizer
from corag.retrieval.pipeline import RetrievalPipeline
from corag.retrieval.state import RetrievalState

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result for a single evaluation example."""

    example_id: str
    question: str
    prediction: str
    ground_truth: str
    em: float
    f1: float
    num_steps: int
    num_chunks: int
    num_unique_chunks: int
    latency: float
    retrieval_state: dict[str, Any] | None = None


@dataclass
class EvaluationReport:
    """Aggregated evaluation report."""

    dataset: str
    split: str
    num_examples: int
    avg_em: float
    avg_f1: float
    avg_steps: float
    avg_chunks: float
    avg_unique_chunks: float
    avg_latency: float
    results: list[EvaluationResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dataset": self.dataset,
            "split": self.split,
            "num_examples": self.num_examples,
            "metrics": {
                "exact_match": self.avg_em,
                "f1": self.avg_f1,
            },
            "retrieval_stats": {
                "avg_steps": self.avg_steps,
                "avg_chunks": self.avg_chunks,
                "avg_unique_chunks": self.avg_unique_chunks,
            },
            "efficiency": {
                "avg_latency_seconds": self.avg_latency,
            },
        }


class Evaluator:
    """Evaluates CoRAG on multi-hop QA datasets."""

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline,
        synthesizer: Synthesizer,
        dataset_loader: DatasetLoader | None = None,
    ):
        """Initialize evaluator.

        Args:
            retrieval_pipeline: Retrieval pipeline
            synthesizer: Answer synthesizer
            dataset_loader: Dataset loader (creates default if None)
        """
        self.retrieval_pipeline = retrieval_pipeline
        self.synthesizer = synthesizer
        self.dataset_loader = dataset_loader or DatasetLoader()

    def evaluate(
        self,
        dataset_name: str,
        split: str = "validation",
        max_examples: int | None = None,
        save_results: Path | None = None,
    ) -> EvaluationReport:
        """Evaluate on a dataset.

        Args:
            dataset_name: Name of dataset
            split: Dataset split
            max_examples: Maximum examples to evaluate
            save_results: Path to save detailed results

        Returns:
            EvaluationReport
        """
        logger.info(f"Evaluating on {dataset_name} {split}...")

        results = []
        examples = self.dataset_loader.load_dataset(dataset_name, split, max_examples)

        for example in tqdm(list(examples), desc="Evaluating"):
            result = self._evaluate_example(example)
            results.append(result)

        # Aggregate metrics
        report = self._aggregate_results(dataset_name, split, results)

        logger.info(
            f"Evaluation complete: EM={report.avg_em:.3f}, F1={report.avg_f1:.3f}"
        )

        # Save results if requested
        if save_results:
            self._save_results(report, save_results)

        return report

    def _evaluate_example(self, example: EvalExample) -> EvaluationResult:
        """Evaluate a single example.

        Args:
            example: Evaluation example

        Returns:
            EvaluationResult
        """
        start_time = time.time()

        try:
            # Run retrieval
            state = self.retrieval_pipeline.run(example.question)

            # Synthesize answer
            answer, citations = self.synthesizer.synthesize(state)

            # Extract answer text (remove citation markers and references)
            prediction = self._extract_answer_text(answer)

        except Exception as e:
            logger.error(f"Error evaluating example {example.id}: {e}")
            prediction = ""
            state = RetrievalState(original_question=example.question)

        latency = time.time() - start_time

        # Compute metrics
        em = max_em_over_ground_truths(prediction, [example.answer])
        f1 = max_f1_over_ground_truths(prediction, [example.answer])

        result = EvaluationResult(
            example_id=example.id,
            question=example.question,
            prediction=prediction,
            ground_truth=example.answer,
            em=em,
            f1=f1,
            num_steps=len(state.steps),
            num_chunks=state.total_chunks_retrieved,
            num_unique_chunks=len(state.get_unique_chunks()),
            latency=latency,
            retrieval_state=state.to_dict(),
        )

        return result

    def _extract_answer_text(self, answer: str) -> str:
        """Extract answer text, removing references section.

        Args:
            answer: Full answer with citations

        Returns:
            Clean answer text
        """
        # Remove references section
        if "## References" in answer:
            answer = answer.split("## References")[0]

        # Remove citation markers for fair comparison
        # Keep the actual text, just remove [1], [2], etc.
        import re

        answer = re.sub(r"\[\d+\]", "", answer)

        return answer.strip()

    def _aggregate_results(
        self, dataset: str, split: str, results: list[EvaluationResult]
    ) -> EvaluationReport:
        """Aggregate results into a report.

        Args:
            dataset: Dataset name
            split: Split name
            results: List of results

        Returns:
            EvaluationReport
        """
        num_examples = len(results)

        if num_examples == 0:
            return EvaluationReport(
                dataset=dataset,
                split=split,
                num_examples=0,
                avg_em=0.0,
                avg_f1=0.0,
                avg_steps=0.0,
                avg_chunks=0.0,
                avg_unique_chunks=0.0,
                avg_latency=0.0,
            )

        avg_em = sum(r.em for r in results) / num_examples
        avg_f1 = sum(r.f1 for r in results) / num_examples
        avg_steps = sum(r.num_steps for r in results) / num_examples
        avg_chunks = sum(r.num_chunks for r in results) / num_examples
        avg_unique_chunks = sum(r.num_unique_chunks for r in results) / num_examples
        avg_latency = sum(r.latency for r in results) / num_examples

        return EvaluationReport(
            dataset=dataset,
            split=split,
            num_examples=num_examples,
            avg_em=avg_em,
            avg_f1=avg_f1,
            avg_steps=avg_steps,
            avg_chunks=avg_chunks,
            avg_unique_chunks=avg_unique_chunks,
            avg_latency=avg_latency,
            results=results,
        )

    def _save_results(self, report: EvaluationReport, output_path: Path) -> None:
        """Save evaluation results to file.

        Args:
            report: Evaluation report
            output_path: Output file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save summary as JSON
        summary_path = output_path.with_suffix(".json")
        with open(summary_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        logger.info(f"Saved evaluation summary to {summary_path}")

        # Save detailed results as CSV
        csv_path = output_path.with_suffix(".csv")
        with open(csv_path, "w") as f:
            # Header
            f.write(
                "id,question,prediction,ground_truth,em,f1,steps,chunks,unique_chunks,latency\n"
            )

            # Rows
            for r in report.results:
                # Escape commas and quotes
                question = r.question.replace('"', '""')
                prediction = r.prediction.replace('"', '""')
                ground_truth = r.ground_truth.replace('"', '""')

                f.write(
                    f'{r.example_id},"{question}","{prediction}","{ground_truth}",'
                    f"{r.em},{r.f1},{r.num_steps},{r.num_chunks},"
                    f"{r.num_unique_chunks},{r.latency:.2f}\n"
                )

        logger.info(f"Saved detailed results to {csv_path}")
