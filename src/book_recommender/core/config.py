"""Centralized configuration for the book recommender application."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_PATH = RAW_DATA_DIR / "books_prepared.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "books_cleaned.parquet"
EMBEDDINGS_PATH = PROCESSED_DATA_DIR / "book_embeddings.npy"
EMBEDDING_METADATA_PATH = PROCESSED_DATA_DIR / "embedding_metadata.json"
CLUSTERS_CACHE_PATH = PROCESSED_DATA_DIR / "cluster_cache.pkl"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

EMBEDDING_DIMENSION = 384

EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

DEFAULT_BATCH_SIZE = 32

DEFAULT_TOP_K = 10
MIN_SIMILARITY_THRESHOLD = 0.3
NUM_CLUSTERS = int(os.getenv("NUM_CLUSTERS", "20"))


APP_VERSION = "0.1.0"
FALLBACK_COVER_URL = "https://placehold.co/200x300/667eea/white?text=No+Cover"


# --- Data/Model Versioning (Future Consideration)
# For production systems, consider implementing a robust data and model
# versioning system (e.g., DVC - Data Version Control) to track changes
# to processed data and generated embeddings. For this MVP, manual
# management or timestamping of files is suggested if versioning is critical.
