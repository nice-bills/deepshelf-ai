# src/recommender.py

import faiss
from typing import List, Dict
import os
import logging
import pandas as pd
import numpy as np
from src.book_recommender.core.exceptions import DataNotFoundError
import src.book_recommender.core.config as config

# Use a module-specific logger
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
        self.embeddings = embeddings.astype('float32') # FAISS requires float32
        
        # Normalize embeddings for cosine similarity search with FAISS
        faiss.normalize_L2(self.embeddings)
        
        # Build the FAISS index for fast similarity search
        self.index = faiss.IndexFlatL2(config.EMBEDDING_DIMENSION)
        self.index.add(self.embeddings)
        
        # Create a title-to-index mapping for fast lookups
        self.title_to_index = pd.Series(self.book_data.index, index=self.book_data['title_lower']).to_dict()
        logger.info(f"Recommender initialized with FAISS index containing {self.index.ntotal} vectors.")

    def get_recommendations_from_vector(
        self,
        vector: np.ndarray,
        top_k: int = config.DEFAULT_TOP_K,
        similarity_threshold: float = config.MIN_SIMILARITY_THRESHOLD,
        ignore_index: int = None
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
        
        recommendations = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            dist = distances[0][i]
            
            if idx == ignore_index:
                continue

            similarity_score = 1 - (dist**2) / 2
            
            if similarity_score >= similarity_threshold:
                rec = self.book_data.iloc[idx]
                recommendations.append({
                    'id': rec['id'],
                    'title': rec['title'],
                    'authors': rec.get('authors', 'N/A'),
                    'description': rec.get('description', ''),
                    'genres': rec.get('genres', ''),
                    'tags': rec.get('tags', ''),
                    'rating': rec.get('rating', 'N/A'),
                    'similarity': similarity_score
                })
        
        return recommendations

    def get_recommendations(
        self,
        title: str,
        top_k: int = config.DEFAULT_TOP_K,
        similarity_threshold: float = config.MIN_SIMILARITY_THRESHOLD
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

if __name__ == '__main__':
    import sys

    # Add project root to path to allow absolute imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.book_recommender.core import config as config_main
    from src.book_recommender.core.exceptions import DataNotFoundError

    # Example usage of the recommender
    
    # Configure logging only when run as a script
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Ensure data exists. In a real app, you'd run these scripts in order.
    if not (os.path.exists(config_main.PROCESSED_DATA_PATH) and os.path.exists(config_main.EMBEDDINGS_PATH)):
        print("Processed data or embeddings not found.")
        print("Please run 'python src/data_processor.py' and 'python src/embedder.py' first.")
    else:
        # Load the data from disk
        logger.info(f"Loading book metadata from {config_main.PROCESSED_DATA_PATH}...")
        book_data_df = pd.read_parquet(config_main.PROCESSED_DATA_PATH)
        
        logger.info(f"Loading book embeddings from {config_main.EMBEDDINGS_PATH}...")
        embeddings_arr = np.load(config_main.EMBEDDINGS_PATH)
        
        # Initialize the recommender with the loaded data
        recommender = BookRecommender(book_data=book_data_df, embeddings=embeddings_arr)
        
        # Get a book title to test with (use the first one from the dataset)
        book_titles = recommender.book_data['title'].tolist()
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
