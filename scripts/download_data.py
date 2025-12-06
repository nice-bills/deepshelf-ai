import logging
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define paths directly to avoid importing from src (which isn't copied yet in Docker build)
PROCESSED_DATA_DIR = Path("data/processed")
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "books_with_embeddings.parquet"
EMBEDDINGS_PATH = PROCESSED_DATA_DIR / "embeddings.npy"
CLUSTERS_CACHE_PATH = PROCESSED_DATA_DIR / "clusters_cache.pkl"

def download_processed_data(repo_id: str):
    """
    Downloads processed data files (parquet, npy, pkl) from a private Hugging Face Dataset.
    
    Args:
        repo_id (str): The Hugging Face dataset ID (e.g., 'username/dataset-name').
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN environment variable not found. If the dataset is private, download will fail.")

    logger.info(f"Starting download from Hugging Face Dataset: {repo_id}")
    logger.info(f"Target directory: {PROCESSED_DATA_DIR}")

    try:
        # Ensure directory exists
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Download only specific files to avoid clutter
        allow_patterns = [
            "*.parquet",
            "*.npy",
            "*.pkl",
            "*.json"
        ]

        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=PROCESSED_DATA_DIR,
            local_dir_use_symlinks=False, # Important for Docker/Deployment
            allow_patterns=allow_patterns,
            token=hf_token
        )
        
        logger.info("Successfully downloaded all data files.")
        
        # Verify files
        expected_files = [
            PROCESSED_DATA_PATH,
            EMBEDDINGS_PATH,
            CLUSTERS_CACHE_PATH
        ]
        
        missing = [f.name for f in expected_files if not f.exists()]
        if missing:
            logger.error(f"Warning: The following expected files are still missing after download: {missing}")
        else:
            logger.info("Verification successful: All core data files are present.")

    except Exception as e:
        logger.error(f"Failed to download data from Hugging Face: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Default repo ID - WILL BE OVERRIDDEN by environment variable in production
    DEFAULT_REPO_ID = "nice-bill/book-recommender-data"
    
    repo_id = os.getenv("HF_DATASET_ID", DEFAULT_REPO_ID)
    
    if repo_id == "PLACEHOLDER_USERNAME/PLACEHOLDER_DATASET":
        logger.error("Please set the HF_DATASET_ID environment variable or update the script.")
        sys.exit(1)
        
    download_processed_data(repo_id)
