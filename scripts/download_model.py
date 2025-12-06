import os
import sys

from sentence_transformers import SentenceTransformer

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# from src.book_recommender.core import config

MODEL_NAME = "all-MiniLM-L6-v2"

def download_model():
    """
    Downloads the sentence-transformer model specified in the config file.
    This is useful for pre-downloading the model in a Docker build.
    """
    print(f"Downloading model: {MODEL_NAME}...")
    _ = SentenceTransformer(MODEL_NAME)
    print("Model downloaded successfully.")


if __name__ == "__main__":
    download_model()
