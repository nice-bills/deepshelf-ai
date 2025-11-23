"""Machine learning modules for BookFinder-AI"""

from src.book_recommender.ml.recommender import BookRecommender
from src.book_recommender.ml.embedder import generate_embedding_for_query, generate_embeddings
from src.book_recommender.ml.clustering import cluster_books, get_cluster_names
from src.book_recommender.ml.explainability import explain_recommendation
from src.book_recommender.ml.feedback import save_feedback, get_all_feedback

__all__ = [
    "BookRecommender",
    "generate_embedding_for_query",
    "generate_embeddings",
    "cluster_books",
    "get_cluster_names",
    "explain_recommendation",
    "save_feedback",
    "get_all_feedback",
]