# src/embedder.py

import logging
import os
import sys
from functools import lru_cache
from typing import Optional
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

import src.book_recommender.core.config as config
from src.book_recommender.core.exceptions import DataNotFoundError, ModelLoadError

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_model(model_name: str = config.EMBEDDING_MODEL) -> SentenceTransformer:
    """
    Loads and caches the sentence-transformer model.
    
    Checks for a local cache at 'data/processed/model_cache'. 
    If not found or incomplete, downloads and saves it.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        SentenceTransformer: The loaded model instance.

    Raises:
        ModelLoadError: If the model cannot be loaded.
    """
    cache_path = config.PROCESSED_DATA_DIR / "model_cache"
    
    try:
        # Robust check: A valid model directory must contain 'modules.json'
        if cache_path.exists() and (cache_path / "modules.json").exists():
            logger.info(f"Loading model from local project cache: {cache_path}")
            model = SentenceTransformer(str(cache_path), device=config.EMBEDDING_DEVICE)
            logger.info("Model loaded successfully from cache.")
            return model
        
        logger.info(f"Model not found in project cache (or incomplete). Downloading {model_name}...")
        model = SentenceTransformer(model_name, device=config.EMBEDDING_DEVICE)
        
        logger.info(f"Saving model to project cache: {cache_path}")
        # Ensure parent directory exists
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(cache_path))
        logger.info("Model saved to cache.")
        
        return model
    except Exception as e:
        logger.error(f"Failed to load sentence-transformer model '{model_name}': {e}")
        raise ModelLoadError(f"Failed to load sentence-transformer model '{model_name}': {e}")


def generate_embeddings(
    df: pd.DataFrame,
    model_name: str = config.EMBEDDING_MODEL,
    show_progress_bar: bool = True,
    batch_size: int = config.DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """
    Generates sentence embeddings for the 'combined_text' of a DataFrame.

    This function loads a specified sentence-transformer model and uses it to
    encode the `combined_text` column of the provided DataFrame into
    high-dimensional vectors (embeddings).

    Args:
        df (pd.DataFrame): The DataFrame containing book data with a 'combined_text' column.
        model_name (str): The name of the sentence-transformer model to use.
        show_progress_bar (bool): Whether to display a progress bar during encoding.
        batch_size (int): The batch size for the encoding process to manage memory.

    Returns:
        np.ndarray: A 2D NumPy array containing the generated embeddings.
    """
    model = load_model(model_name)

    logger.info(f"Generating embeddings for {len(df)} books...")
    try:
        embeddings = model.encode(
            df["combined_text"].tolist(), show_progress_bar=show_progress_bar, batch_size=batch_size
        )
        logger.info(f"Embeddings generated with shape: {embeddings.shape}")
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        raise

    return np.asarray(embeddings)


def generate_embedding_for_query(
    query: str, model_name: str = config.EMBEDDING_MODEL, model: Optional[SentenceTransformer] = None
) -> np.ndarray:
    """
    Generates a sentence embedding for a single text query.

    Args:
        query (str): The text query to embed.
        model_name (str): The name of the sentence-transformer model to use (if model not provided).
        model (SentenceTransformer, optional): A pre-loaded model instance to use.

    Returns:
        np.ndarray: A 1D NumPy array representing the query embedding.
    """
    if model is None:
        model = load_model(model_name)
    
    logger.info(f"Generating embedding for query: '{query[:50]}...'")
    embedding = model.encode(query, show_progress_bar=False)
    return np.asarray(embedding)


if __name__ == "__main__":
    import argparse
    import json
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.book_recommender.core import config as config_main
    from src.book_recommender.core.exceptions import DataNotFoundError
    from src.book_recommender.utils import ensure_dir_exists as ensure_dir_exists_main

    # When running as a script, basic logging is not configured by default.
    # To see log output, set the LOG_LEVEL environment variable,
    # e.g., `export LOG_LEVEL=INFO` or run with `python -m logging ...`
    if os.getenv("LOG_LEVEL"):
        logging.basicConfig(level=os.getenv("LOG_LEVEL"))
    else:
        # Provide a default configuration if the script is run directly
        # and no environment variable is set, to ensure messages are not lost.
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Generate and save book embeddings.")
    parser.add_argument(
        "--processed-path",
        type=str,
        default=config_main.PROCESSED_DATA_PATH,
        help="Path to the processed Parquet data file.",
    )
    parser.add_argument(
        "--embeddings-path",
        type=str,
        default=config_main.EMBEDDINGS_PATH,
        help="Path to save the embeddings .npy file.",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=config_main.EMBEDDING_METADATA_PATH,
        help="Path to save the embedding metadata JSON file.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=config_main.EMBEDDING_MODEL,
        help="Name of the sentence-transformer model to use.",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_false",
        dest="show_progress_bar",
        help="Disable the progress bar during embedding generation.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=config_main.DEFAULT_BATCH_SIZE, help="The batch size to use for encoding."
    )
    args = parser.parse_args()

    logger.info("--- Starting Embedding Generation Standalone Script ---")

    if not os.path.exists(args.processed_path):
        logger.error(f"Processed data file not found at: {args.processed_path}")
        raise DataNotFoundError(f"Processed data file not found at: {args.processed_path}")

    logger.info(f"Loading processed data from {args.processed_path}...")
    processed_df = pd.read_parquet(args.processed_path)

    embeddings_array = generate_embeddings(
        df=processed_df,
        model_name=args.model_name,
        show_progress_bar=args.show_progress_bar,
        batch_size=args.batch_size,
    )

    try:
        ensure_dir_exists_main(args.embeddings_path)
        logger.info(f"Saving embeddings to {args.embeddings_path}...")
        np.save(args.embeddings_path, embeddings_array)
        logger.info("Embeddings saved successfully.")

        metadata = {
            "model_name": args.model_name,
            "embedding_dimension": config_main.EMBEDDING_DIMENSION,
            "num_books": len(processed_df),
            "batch_size": args.batch_size,
            "created_at": pd.Timestamp.now().isoformat(),
        }
        ensure_dir_exists_main(args.metadata_path)
        
        # Save a simple metadata file to indicate the embedding version/date
        with open(args.metadata_path, "w", encoding="utf-8") as f:
            import json
            json.dump(metadata, f, indent=4)
        
        logger.info(f"Metadata saved to {args.metadata_path}")

    except Exception as e:
        logger.error(f"Failed to save embeddings or metadata: {e}")
        raise

    logger.info("--- Embedding Generation Finished ---")
