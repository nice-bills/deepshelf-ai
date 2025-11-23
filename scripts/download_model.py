# scripts/download_model.py
import os
import sys

from sentence_transformers import SentenceTransformer

# Add src to path to import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import config


def download_model():
    """
    Downloads the sentence-transformer model specified in the config file.
    This is useful for pre-downloading the model in a Docker build.
    """
    print(f"Downloading model: {config.EMBEDDING_MODEL}...")
    _ = SentenceTransformer(config.EMBEDDING_MODEL)
    print("Model downloaded successfully.")


if __name__ == "__main__":
    download_model()
