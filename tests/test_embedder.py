# tests/test_embedder.py

import os
import tempfile
import unittest

import numpy as np  # Changed from pickle
import pandas as pd

import src.book_recommender.core.config as config
from src.book_recommender.data.processor import clean_and_prepare_data
from src.book_recommender.ml.embedder import generate_embeddings


class TestEmbedder(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.raw_path = os.path.join(self.test_dir.name, "test_books.csv")
        self.processed_path = os.path.join(self.test_dir.name, "test_books_cleaned.parquet")  # Changed to .parquet
        self.embeddings_path = os.path.join(self.test_dir.name, "test_embeddings.npy")  # Changed to .npy

        # Create dummy data and process it
        sample_data = {
            "title": ["Book A", "Book B", "Book C"],
            "authors": ["Author A", "Author B", "Author C"],
            "genres": ["Fiction", "Sci-Fi", "History"],
            "description": ["Desc A", "Desc B", "Desc C"],
            "tags": ["tag1", "tag2", "tag3"],
        }
        pd.DataFrame(sample_data).to_csv(self.raw_path, index=False)
        clean_and_prepare_data(self.raw_path, self.processed_path)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_generate_embeddings(self):
        # Load the processed DataFrame first
        processed_df = pd.read_parquet(self.processed_path)  # Load DataFrame

        # Run the function with correct signature
        embeddings = generate_embeddings(processed_df, show_progress_bar=False)  # Pass DataFrame

        # Save embeddings
        np.save(self.embeddings_path, embeddings)

        # 1. Test that the output file is created
        self.assertTrue(os.path.exists(self.embeddings_path))

        # 2. Test that the embeddings shape is correct
        expected_rows = 3
        expected_dims = config.EMBEDDING_DIMENSION
        self.assertEqual(embeddings.shape, (expected_rows, expected_dims))

        # 3. Test that the saved file can be loaded and has the same shape
        loaded_embeddings = np.load(self.embeddings_path)  # Use np.load instead of pickle
        self.assertEqual(loaded_embeddings.shape, (expected_rows, expected_dims))


if __name__ == "__main__":
    unittest.main()
