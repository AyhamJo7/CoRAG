"""Evaluation harness for CoRAG."""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

from corag.evaluation.datasets import DatasetLoader, EvalExample
from corag.evaluation.metrics import (
    compute_retrieval_precision,
    compute_retrieval_recall,
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
    # Grounding metrics are None when the example carries no gold
    # supporting-document titles.
    retrieval_precision: float | None = None
    retrieval_recall: float | None = None
    citation_precision: float | None = None
    citation_recall: float | None = None
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
    avg_retrieval_precision: float | None = None
    avg_retrieval_recall: float | None = None
    avg_citation_precision: float | None = None
    avg_citation_recall: float | None = None
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
                "retrieval_precision": self.avg_retrieval_precision,
                "retrieval_recall": self.avg_retrieval_recall,
                "citation_precision": self.avg_citation_precision,
                "citation_recall": self.avg_citation_recall,
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
            answer = ""
            citations = []
            state = RetrievalState(original_question=example.question)

        latency = time.time() - start_time

        # Compute answer metrics
        em = max_em_over_ground_truths(prediction, [example.answer])
        f1 = max_f1_over_ground_truths(prediction, [example.answer])

        # Compute grounding metrics against gold supporting-document titles
        retrieval_precision = retrieval_recall = None
        citation_precision = citation_recall = None
        gold_titles = set(example.supporting_facts or [])
        if gold_titles:
            retrieved_titles = {
                c.doc_title for c in state.get_unique_chunks() if c.doc_title
            }
            retrieval_precision = compute_retrieval_precision(
                retrieved_titles, gold_titles
            )
            retrieval_recall = compute_retrieval_recall(retrieved_titles, gold_titles)

            cited_titles = self._extract_cited_titles(answer, citations)
            citation_precision = compute_retrieval_precision(cited_titles, gold_titles)
            citation_recall = compute_retrieval_recall(cited_titles, gold_titles)

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
            retrieval_precision=retrieval_precision,
            retrieval_recall=retrieval_recall,
            citation_precision=citation_precision,
            citation_recall=citation_recall,
            retrieval_state=state.to_dict(),
        )

        return result

    def _extract_cited_titles(
        self, answer: str, citations: list[dict[str, str]]
    ) -> set[str]:
        """Extract titles of documents actually cited in the answer.

        Args:
            answer: Full answer text with [n] citation markers
            citations: Citation entries produced by the synthesizer

        Returns:
            Set of document titles whose citation marker appears in the answer
        """
        cited_ids = set(re.findall(r"\[(\d+)\]", answer))
        return {
            c["title"] for c in citations if c.get("id") in cited_ids and c.get("title")
        }

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
            avg_retrieval_precision=self._mean_of_present(
                [r.retrieval_precision for r in results]
            ),
            avg_retrieval_recall=self._mean_of_present(
                [r.retrieval_recall for r in results]
            ),
            avg_citation_precision=self._mean_of_present(
                [r.citation_precision for r in results]
            ),
            avg_citation_recall=self._mean_of_present(
                [r.citation_recall for r in results]
            ),
            results=results,
        )

    @staticmethod
    def _mean_of_present(values: list[float | None]) -> float | None:
        """Average the non-None values, or None if all are missing.

        Args:
            values: Per-example metric values, None where not computable

        Returns:
            Mean over examples that have the metric, or None
        """
        present = [v for v in values if v is not None]
        if not present:
            return None
        return sum(present) / len(present)

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
                "id,question,prediction,ground_truth,em,f1,steps,chunks,"
                "unique_chunks,latency,retrieval_precision,retrieval_recall,"
                "citation_precision,citation_recall\n"
            )

            # Rows
            for r in report.results:
                # Escape commas and quotes
                question = r.question.replace('"', '""')
                prediction = r.prediction.replace('"', '""')
                ground_truth = r.ground_truth.replace('"', '""')

                def fmt(value: float | None) -> str:
                    return "" if value is None else f"{value:.4f}"

                f.write(
                    f'{r.example_id},"{question}","{prediction}","{ground_truth}",'
                    f"{r.em},{r.f1},{r.num_steps},{r.num_chunks},"
                    f"{r.num_unique_chunks},{r.latency:.2f},"
                    f"{fmt(r.retrieval_precision)},{fmt(r.retrieval_recall)},"
                    f"{fmt(r.citation_precision)},{fmt(r.citation_recall)}\n"
                )

        logger.info(f"Saved detailed results to {csv_path}")
