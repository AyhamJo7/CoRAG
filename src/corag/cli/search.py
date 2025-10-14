"""CLI for running CoRAG multi-step search."""

import json
import logging
import os
from pathlib import Path

import click
from dotenv import load_dotenv

from corag.controller.base import GenerationConfig
from corag.controller.openai_controller import OpenAIController
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
@click.option("--question", required=True, help="Question to answer")
@click.option(
    "--index-dir", required=True, type=click.Path(exists=True), help="Index directory"
)
@click.option(
    "--embedding-model",
    default="sentence-transformers/msmarco-distilbert-base-v4",
    help="Embedding model",
)
@click.option("--device", default="cpu", help="Device (cpu/cuda/mps)")
@click.option("--provider", default="openai", help="LLM provider (openai)")
@click.option(
    "--model",
    default=os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini"),
    help="LLM model name",
)
@click.option(
    "--temperature",
    default=float(os.getenv("DEFAULT_TEMPERATURE", "0.2")),
    type=float,
    help="Generation temperature",
)
@click.option(
    "--max-tokens",
    default=int(os.getenv("DEFAULT_MAX_TOKENS", "2048")),
    type=int,
    help="Max tokens",
)
@click.option(
    "--max-steps",
    default=int(os.getenv("DEFAULT_MAX_STEPS", "6")),
    type=int,
    help="Max retrieval steps",
)
@click.option(
    "--k",
    default=int(os.getenv("DEFAULT_TOP_K", "8")),
    type=int,
    help="Chunks per query",
)
@click.option("--output", type=click.Path(), help="Output file for trace JSON")
def main(
    question: str,
    index_dir: str,
    embedding_model: str,
    device: str,
    provider: str,
    model: str,
    temperature: float,
    max_tokens: int,
    max_steps: int,
    k: int,
    output: str | None,
) -> None:
    """Run CoRAG multi-step retrieval search."""
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
            raise ValueError("OPENAI_API_KEY environment variable not set")

        controller = OpenAIController(
            api_key=api_key,
            model=model,
            api_base=os.getenv("OPENAI_API_BASE"),
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    # Create pipeline
    config = GenerationConfig(temperature=temperature, max_tokens=max_tokens)
    pipeline = RetrievalPipeline(
        index=index,
        embedder=embedder,
        controller=controller,
        k=k,
        max_steps=max_steps,
        config=config,
    )

    # Run search
    logger.info(f"Running search for: {question}")
    state = pipeline.run(question)

    # Print summary
    print("\n" + "=" * 80)
    print(f"Question: {question}")
    print("=" * 80)
    print(
        f"\nRetrieved {state.total_chunks_retrieved} chunks across {len(state.steps)} steps"
    )
    print(f"Unique chunks: {len(state.get_unique_chunks())}")
    print(f"Is sufficient: {state.is_sufficient}")
    print("\nExecuted queries:")
    for i, query in enumerate(state.executed_queries, 1):
        print(f"  {i}. {query}")

    # Save trace if output specified
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(state.to_dict(), f, indent=2)

        logger.info(f"Saved trace to {output_path}")


if __name__ == "__main__":
    main()
