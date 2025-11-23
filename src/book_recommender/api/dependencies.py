import logging
from functools import lru_cache
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

from src.book_recommender.ml.recommender import BookRecommender
import src.book_recommender.core.config as config
from src.book_recommender.core.exceptions import DataNotFoundError
from src.book_recommender.ml.embedder import (
    _load_model as embedder_load_model,
)  # Alias to avoid name conflict
from src.book_recommender.ml.clustering import (
    cluster_books,
    get_cluster_names,
)  # New import for clustering
from slowapi import Limiter
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])


@lru_cache(maxsize=1)
def get_recommender() -> BookRecommender:
    """
    Loads book data and embeddings, then initializes and returns a cached BookRecommender instance.
    This function is designed to be a FastAPI dependency.
    """
    try:
        logger.info(f"Loading book metadata from {config.PROCESSED_DATA_PATH}...")
        book_data_df = pd.read_parquet(config.PROCESSED_DATA_PATH)
        
        logger.info(f"Loading book embeddings from {config.EMBEDDINGS_PATH}...")
        embeddings_arr = np.load(config.EMBEDDINGS_PATH)
        
        recommender = BookRecommender(book_data=book_data_df, embeddings=embeddings_arr)
        logger.info("BookRecommender initialized and cached.")
        return recommender
    except FileNotFoundError as e:
        logger.error(f"Required data file not found: {e}")
        raise DataNotFoundError(f"Could not find processed data or embeddings. Please ensure '{config.PROCESSED_DATA_PATH}' and '{config.EMBEDDINGS_PATH}' exist.")
    except Exception as e:
        logger.error(f"Error initializing BookRecommender: {e}")
        raise

@lru_cache(maxsize=1)
def get_sentence_transformer_model() -> SentenceTransformer:
    """
    Loads and returns a cached SentenceTransformer model instance.
    This function is designed to be a FastAPI dependency.
    """
    logger.info("Loading sentence transformer model for API use...")
    model = embedder_load_model(config.EMBEDDING_MODEL) # Use the aliased function
    logger.info("Sentence Transformer model loaded and cached.")
    return model

@lru_cache(maxsize=1)
def get_clusters_data() -> tuple[np.ndarray, dict, pd.DataFrame]:
    """
    Generates and returns cached book clusters, cluster names, and book data with cluster IDs.
    This function is designed to be a FastAPI dependency.
    """
    logger.info("Generating/Loading cluster data...")
    # This will trigger get_recommender if not already cached
    recommender = get_recommender() 
    book_data_df = recommender.book_data.copy() # Work on a copy to add cluster_id
    embeddings_arr = recommender.embeddings # Use embeddings from recommender

    # Generate clusters
    clusters_arr, _ = cluster_books(embeddings_arr, n_clusters=config.NUM_CLUSTERS)
    book_data_df['cluster_id'] = clusters_arr

    # Generate cluster names
    names = get_cluster_names(book_data_df, clusters_arr)
    
    logger.info("Cluster data generated/loaded and cached.")
    return clusters_arr, names, book_data_df
