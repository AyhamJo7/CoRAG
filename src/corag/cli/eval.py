"""CLI for running evaluation."""

import logging
import os
from pathlib import Path

import click
from dotenv import load_dotenv

from corag.controller.base import GenerationConfig
from corag.controller.openai_controller import OpenAIController
from corag.evaluation.evaluator import Evaluator
from corag.generation.synthesizer import Synthesizer
from corag.indexing.embedder import Embedder
from corag.indexing.index import FAISSIndex
from corag.retrieval.pipeline import RetrievalPipeline

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--dataset",
    required=True,
    type=click.Choice(["hotpotqa", "2wikimultihopqa"], case_sensitive=False),
    help="Dataset name",
)
@click.option("--split", default="validation", help="Dataset split")
@click.option(
    "--index-dir", required=True, type=click.Path(exists=True), help="Index directory"
)
@click.option(
    "--embedding-model",
    default="sentence-transformers/msmarco-distilbert-base-v4",
    help="Embedding model",
)
@click.option("--device", default="cpu", help="Device")
@click.option("--provider", default="openai", help="LLM provider")
@click.option(
    "--model", default=os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini"), help="LLM model"
)
@click.option("--temperature", default=0.2, type=float, help="Temperature")
@click.option("--max-steps", default=6, type=int, help="Max retrieval steps")
@click.option("--k", default=8, type=int, help="Chunks per query")
@click.option("--max-examples", type=int, help="Maximum examples to evaluate")
@click.option("--output", type=click.Path(), help="Output path for results")
def main(
    dataset: str,
    split: str,
    index_dir: str,
    embedding_model: str,
    device: str,
    provider: str,
    model: str,
    temperature: float,
    max_steps: int,
    k: int,
    max_examples: int | None,
    output: str | None,
) -> None:
    """Run evaluation on a multi-hop QA dataset."""
    index_dir_obj = Path(index_dir)

    # Load index
    logger.info(f"Loading index from {index_dir_obj}")
    index = FAISSIndex.load(index_dir_obj)

    # Load embedder
    logger.info(f"Loading embedder: {embedding_model}")
    embedder = Embedder(model_name=embedding_model, device=device)

    # Initialize controller
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        controller = OpenAIController(api_key=api_key, model=model)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    # Create pipeline and synthesizer
    config = GenerationConfig(temperature=temperature)
    pipeline = RetrievalPipeline(
        index=index,
        embedder=embedder,
        controller=controller,
        k=k,
        max_steps=max_steps,
        config=config,
    )
    synthesizer = Synthesizer(controller=controller, config=config)

    # Create evaluator
    evaluator = Evaluator(
        retrieval_pipeline=pipeline,
        synthesizer=synthesizer,
    )

    # Run evaluation
    output_path = Path(output) if output else None
    report = evaluator.evaluate(
        dataset_name=dataset,
        split=split,
        max_examples=max_examples,
        save_results=output_path,
    )

    # Print summary
    print("\n" + "=" * 80)
    print(f"Evaluation Results: {dataset} ({split})")
    print("=" * 80)
    print(f"\nDataset: {report.dataset}")
    print(f"Split: {report.split}")
    print(f"Examples: {report.num_examples}")
    print("\nMetrics:")
    print(f"  Exact Match: {report.avg_em:.4f}")
    print(f"  F1 Score:    {report.avg_f1:.4f}")
    print("\nRetrieval Stats:")
    print(f"  Avg Steps:         {report.avg_steps:.2f}")
    print(f"  Avg Chunks:        {report.avg_chunks:.2f}")
    print(f"  Avg Unique Chunks: {report.avg_unique_chunks:.2f}")
    print("\nEfficiency:")
    print(f"  Avg Latency: {report.avg_latency:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
