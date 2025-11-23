# tests/test_integration.py

import os
import tempfile
import unittest

import pandas as pd

# FastAPI related imports
import src.book_recommender.core.config as config
from src.book_recommender.data.processor import process_dataframe
from src.book_recommender.ml.embedder import generate_embeddings
from src.book_recommender.ml.recommender import BookRecommender


class TestIntegration(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()

        self.raw_csv_path = os.path.join(self.test_dir.name, "test_books.csv")

        # Create a dummy CSV file for the test
        sample_data = {
            "title": ["The Sun Also Rises", "A Farewell to Arms", "For Whom the Bell Tolls", "The Old Man and the Sea"],
            "authors": ["Ernest Hemingway", "Ernest Hemingway", "Ernest Hemingway", "Ernest Hemingway"],
            "genres": ["Fiction", "War, Fiction", "War, Fiction", "Fiction"],
            "description": [
                "A story of American and British expatriates in Paris.",
                "A love story during World War I.",
                "An American in the Spanish Civil War.",
                "An old fisherman struggles with a giant marlin.",
            ],
            "tags": ["lost generation", "war", "spain", "cuba"],
        }
        pd.DataFrame(sample_data).to_csv(self.raw_csv_path, index=False)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_full_pipeline(self):
        """
        Tests the full data processing and recommendation pipeline end-to-end.
        """
        # --- 1. Load Raw Data ---
        raw_df = pd.read_csv(self.raw_csv_path)
        self.assertEqual(len(raw_df), 4)

        # --- 2. Process DataFrame ---
        processed_df = process_dataframe(raw_df)
        self.assertEqual(len(processed_df), 4)
        self.assertIn("combined_text", processed_df.columns)

        # --- 3. Generate Embeddings ---
        embeddings = generate_embeddings(df=processed_df, model_name=config.EMBEDDING_MODEL)
        self.assertEqual(embeddings.shape, (4, config.EMBEDDING_DIMENSION))

        # --- 4. Initialize Recommender ---
        recommender = BookRecommender(book_data=processed_df, embeddings=embeddings)
        self.assertIsNotNone(recommender)

        # --- 5. Get Recommendations ---
        # "A Farewell to Arms" and "For Whom the Bell Tolls" are both war novels by the same author,
        # so they should be highly similar.
        recommendations = recommender.get_recommendations("A Farewell to Arms", top_k=1)

        # --- 6. Assert Results ---
        self.assertEqual(len(recommendations), 1)

        top_recommendation = recommendations[0]

        # The exact top book can vary with model updates, so we check for reasonableness.
        # It should be another Hemingway novel from our list.
        expected_titles = ["The Sun Also Rises", "For Whom the Bell Tolls", "The Old Man and the Sea"]
        self.assertIn(top_recommendation["title"], expected_titles)

        # The similarity should be high, indicating a strong match.
        # NOTE: The threshold is set lower (e.g., > 0.3) because with normalized
        # text and the 'all-MiniLM-L6-v2' model, scores for even closely
        # related documents might not be extremely high. This ensures the test
        # is robust to minor model variations.
        self.assertTrue(
            top_recommendation["similarity"] > 0.3, f"Similarity score {top_recommendation['similarity']} is not > 0.3"
        )


if __name__ == "__main__":
    unittest.main()
