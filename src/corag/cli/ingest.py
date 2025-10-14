"""CLI for corpus ingestion."""

import logging
from pathlib import Path

import click

from corag.corpus.ingest import CorpusIngestor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--max-docs", type=int, default=None, help="Maximum documents to ingest")
@click.option("--pattern", default="*.jsonl", help="File pattern for directory input")
def main(input_path: str, output_path: str, max_docs: int | None, pattern: str) -> None:
    """Ingest documents from JSONL files.

    INPUT_PATH: Path to JSONL file or directory
    OUTPUT_PATH: Path for output JSONL file
    """
    input_path_obj = Path(input_path)
    output_path_obj = Path(output_path)

    ingestor = CorpusIngestor()

    # Load documents
    if input_path_obj.is_file():
        documents = list(ingestor.ingest_jsonl(input_path_obj, max_docs))
    elif input_path_obj.is_dir():
        documents = list(ingestor.ingest_directory(input_path_obj, pattern, max_docs))
    else:
        raise ValueError(f"Invalid input path: {input_path_obj}")

    # Save
    ingestor.save_documents(documents, output_path_obj)

    logger.info(f"Successfully ingested {len(documents)} documents")


if __name__ == "__main__":
    main()
