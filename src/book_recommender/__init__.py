"""BookFinder-AI - Semantic Book Recommendation Engine"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Expose main classes at package level
from src.book_recommender.ml.clustering import cluster_books, get_cluster_names
from src.book_recommender.ml.embedder import generate_embedding_for_query
from src.book_recommender.ml.recommender import BookRecommender

__all__ = [
    "BookRecommender",
    "generate_embedding_for_query",
    "cluster_books",
    "get_cluster_names",
]
