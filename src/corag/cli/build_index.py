"""CLI for building FAISS index."""

import logging
from pathlib import Path

import click

from corag.corpus.chunker import Chunker
from corag.corpus.ingest import CorpusIngestor
from corag.indexing.embedder import Embedder
from corag.indexing.index import FAISSIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@click.command()
@click.argument("corpus_path", type=click.Path(exists=True))
@click.argument("index_dir", type=click.Path())
@click.option(
    "--embedding-model",
    default="sentence-transformers/msmarco-distilbert-base-v4",
    help="Embedding model name",
)
@click.option("--device", default="cpu", help="Device (cpu/cuda/mps)")
@click.option("--index-type", default="Flat", help="FAISS index type (Flat/IVF)")
@click.option("--nlist", default=100, type=int, help="Number of clusters for IVF")
@click.option("--chunk-size", default=512, type=int, help="Chunk size in tokens")
@click.option("--chunk-overlap", default=64, type=int, help="Chunk overlap in tokens")
@click.option("--batch-size", default=32, type=int, help="Embedding batch size")
@click.option("--max-docs", type=int, default=None, help="Maximum documents to process")
def main(
    corpus_path: str,
    index_dir: str,
    embedding_model: str,
    device: str,
    index_type: str,
    nlist: int,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
    max_docs: int | None,
) -> None:
    """Build FAISS index from document corpus.

    CORPUS_PATH: Path to corpus JSONL file
    INDEX_DIR: Directory to save index
    """
    corpus_path_obj = Path(corpus_path)
    index_dir_obj = Path(index_dir)

    # Load documents
    logger.info(f"Loading documents from {corpus_path_obj}")
    ingestor = CorpusIngestor()
    documents = list(ingestor.ingest_jsonl(corpus_path_obj, max_docs))
    logger.info(f"Loaded {len(documents)} documents")

    # Chunk documents
    logger.info("Chunking documents...")
    chunker = Chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")

    # Initialize embedder
    logger.info(f"Loading embedding model: {embedding_model}")
    embedder = Embedder(
        model_name=embedding_model, device=device, batch_size=batch_size
    )

    # Embed chunks
    logger.info("Embedding chunks...")
    chunk_texts = [c.text for c in chunks]
    embeddings = embedder.embed_texts(chunk_texts, show_progress=True)
    logger.info(f"Generated embeddings with shape {embeddings.shape}")

    # Build index
    logger.info(f"Building {index_type} index...")
    dimension = embedder.dimension
    if dimension is None:
        raise ValueError("Embedder dimension is not set")
    index = FAISSIndex(
        dimension=dimension,
        index_type=index_type,
        nlist=nlist,
    )
    index.build(embeddings, chunks)

    # Save index
    logger.info(f"Saving index to {index_dir_obj}")
    index.save(index_dir_obj)

    logger.info("Index building complete!")


if __name__ == "__main__":
    main()
