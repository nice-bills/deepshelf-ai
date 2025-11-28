import logging
import os
import sys
from typing import Dict, List, Optional

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import faiss
import numpy as np
import pandas as pd

import src.book_recommender.core.config as config

logger = logging.getLogger(__name__)


class BookRecommender:
    """
    A content-based book recommender system that uses a FAISS index for
    efficient similarity search.

    This class encapsulates the logic for building a searchable index of book
    embeddings and retrieving recommendations based on semantic similarity.
    """

    def __init__(self, book_data: pd.DataFrame, embeddings: np.ndarray):
        """
        Initializes the recommender, builds the FAISS index, and prepares data.

        Args:
            book_data (pd.DataFrame): DataFrame containing book metadata.
                                      Must include 'title_lower' column for indexing.
            embeddings (np.ndarray): A 2D NumPy array of book embeddings.
        """
        if len(book_data) != len(embeddings):
            raise ValueError("Mismatch between number of books and number of embeddings.")

        self.book_data = book_data
        self.embeddings = embeddings.astype("float32")

        faiss.normalize_L2(self.embeddings)

        self.index = faiss.IndexFlatL2(config.EMBEDDING_DIMENSION)
        self.index.add(self.embeddings)

        self.title_to_index = pd.Series(self.book_data.index, index=self.book_data["title_lower"]).to_dict()
        logger.info(f"Recommender initialized with FAISS index containing {self.index.ntotal} vectors.")

    def get_recommendations_from_vector(
        self,
        vector: np.ndarray,
        top_k: int = config.DEFAULT_TOP_K,
        similarity_threshold: float = config.MIN_SIMILARITY_THRESHOLD,
        ignore_index: Optional[int] = None,
    ) -> List[Dict]:
        """
        Finds and returns top_k book recommendations for a given embedding vector.

        Args:
            vector (np.ndarray): The embedding vector to find recommendations for.
            top_k (int): The number of recommendations to return.
            similarity_threshold (float): The minimum similarity score.
            ignore_index (int, optional): An index to ignore in the results (e.g., the query book itself).

        Returns:
            A list of dictionaries, each containing details of a recommended book.
        """
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        faiss.normalize_L2(vector)

        distances, indices = self.index.search(vector, top_k + (1 if ignore_index is not None else 0))

        # Filter out the ignore_index if present
        valid_mask = indices[0] != ignore_index
        valid_indices = indices[0][valid_mask]
        valid_distances = distances[0][valid_mask]

        # Calculate similarity scores
        # cosine_sim = 1 - (L2_dist^2) / 2
        similarity_scores = 1 - (valid_distances**2) / 2

        # Filter by threshold
        threshold_mask = similarity_scores >= similarity_threshold
        final_indices = valid_indices[threshold_mask]
        final_scores = similarity_scores[threshold_mask]

        if len(final_indices) == 0:
            return []

        # Batch retrieve book data
        # We use iloc[final_indices] to get all rows at once
        recommended_books_df = self.book_data.iloc[final_indices]

        recommendations = []
        # Iterate over the subset DataFrame and the corresponding scores
        for (idx, row), score in zip(recommended_books_df.iterrows(), final_scores):
            recommendations.append(
                {
                    "id": row["id"],
                    "title": row["title"],
                    "authors": row.get("authors", "N/A"),
                    "description": row.get("description", ""),
                    "genres": row.get("genres", ""),
                    "tags": row.get("tags", ""),
                    "rating": row.get("rating", "N/A"),
                    "similarity": float(score),
                }
            )

        return recommendations

    def get_recommendations(
        self,
        title: str,
        top_k: int = 5,
        similarity_threshold: float = config.MIN_SIMILARITY_THRESHOLD,
    ) -> List[Dict]:
        """
        Finds and returns top_k book recommendations for a given title using FAISS.

        Args:
            title (str): The title of the book to get recommendations for.
            top_k (int): The number of recommendations to return.
            similarity_threshold (float): The minimum similarity score for a book
                                          to be considered a recommendation.

        Returns:
            A list of dictionaries, where each dictionary contains the details
            of a recommended book (title, authors, similarity, etc.).
            Returns an empty list if the title is not found or no books meet
            the similarity threshold.
        """
        book_index = self.title_to_index.get(title.lower())

        if book_index is None:
            logger.warning(f"Title '{title}' not found in the dataset.")
            return []

        book_vector = self.embeddings[book_index]

        recommendations = self.get_recommendations_from_vector(
            book_vector, top_k, similarity_threshold, ignore_index=book_index
        )

        logger.info(f"Found {len(recommendations)} recommendations for '{title}'.")
        return recommendations


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.book_recommender.core import config as config_main

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if not (os.path.exists(config_main.PROCESSED_DATA_PATH) and os.path.exists(config_main.EMBEDDINGS_PATH)):
        print("Processed data or embeddings not found.")
        print("Please run 'python src/data_processor.py' and 'python src/embedder.py' first.")
    else:
        logger.info(f"Loading book metadata from {config_main.PROCESSED_DATA_PATH}...")
        book_data_df = pd.read_parquet(config_main.PROCESSED_DATA_PATH)

        logger.info(f"Loading book embeddings from {config_main.EMBEDDINGS_PATH}...")
        embeddings_arr = np.load(config_main.EMBEDDINGS_PATH)

        recommender = BookRecommender(book_data=book_data_df, embeddings=embeddings_arr)

        book_titles = recommender.book_data["title"].tolist()
        if book_titles:
            test_title = book_titles[0]
            print(f"--- Getting recommendations for: '{test_title}' ---")
            recs = recommender.get_recommendations(test_title, top_k=5)

            if recs:
                for i, rec in enumerate(recs):
                    print(f"{i+1}. {rec['title']} by {rec['authors']} (Similarity: {rec['similarity']:.2f})")
            else:
                print("Could not find any recommendations.")
        else:
            print("No book titles found in the dataset.")
