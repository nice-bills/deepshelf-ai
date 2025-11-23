import os
import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

# FastAPI related imports
from fastapi.testclient import TestClient

import src.book_recommender.core.config as config
from src.book_recommender.api.dependencies import (  # Import actual dependencies to override
    get_clusters_data,
    get_recommender,
    get_sentence_transformer_model,
)
from src.book_recommender.api.main import app as fastapi_app  # Import the FastAPI app instance
from src.book_recommender.ml.recommender import BookRecommender


class TestFastAPIEndpoints(unittest.TestCase):

    dummy_processed_data = pd.DataFrame(
        {
            "id": ["1", "2", "3", "4", "5"],
            "title": ["Test Book 1", "Test Book 2", "Related Test Book 1", "Unrelated Book", "Sci-Fi Classic"],
            "authors": ["Author A", "Author B", "Author A", "Author C", "Author D"],
            "genres": ["Fiction", "Science Fiction", "Fiction", "Fantasy", "Science Fiction"],
            "description": [
                "Description for book 1",
                "Description for book 2 with sci-fi themes",
                "Another book by Author A",
                "A magical adventure",
                "Deep space exploration",
            ],
            "title_lower": ["test book 1", "test book 2", "related test book 1", "unrelated book", "sci-fi classic"],
            "authors_lower": ["author a", "author b", "author a", "author c", "author d"],
            "combined_text": [
                "test book 1 by author a. genres: fiction. description: description for book 1. tags: ",
                "test book 2 by author b. genres: science fiction. description: description for book 2 with sci-fi themes. tags: ",
                "related test book 1 by author a. genres: fiction. description: another book by author a. tags: ",
                "unrelated book by author c. genres: fantasy. description: a magical adventure. tags: ",
                "sci-fi classic by author d. genres: science fiction. description: deep space exploration. tags: ",
            ],
            "cluster_id": [0, 1, 0, 2, 1],
        }
    )
    dummy_embeddings = np.random.rand(len(dummy_processed_data), config.EMBEDDING_DIMENSION).astype("float32")

    dummy_cluster_names = {0: "Fiction Collection", 1: "Sci-Fi Collection", 2: "Fantasy Collection"}

    def setUp(self):
        super().setUp()

        # Clear lru_cache for dependencies
        get_recommender.cache_clear()
        get_sentence_transformer_model.cache_clear()
        get_clusters_data.cache_clear()

        # 1. Create mock objects for dependencies
        self.mock_model = MagicMock()
        self.mock_recommender_instance = MagicMock(
            spec=BookRecommender
        )  # Note: this is the instance returned by get_recommender
        self.mock_clusters_data_value = (  # Note: this is the actual value returned by get_clusters_data
            self.dummy_processed_data["cluster_id"].values,
            self.dummy_cluster_names,
            self.dummy_processed_data,
        )

        # 2. Configure mock behaviors
        self.mock_model.encode.side_effect = lambda x, **kwargs: (
            np.array([np.random.rand(config.EMBEDDING_DIMENSION)]).astype("float32")
            if isinstance(x, str)
            else np.random.rand(len(x), config.EMBEDDING_DIMENSION).astype("float32")
        )

        self.mock_recommender_instance.book_data = self.dummy_processed_data
        self.mock_recommender_instance.embeddings = self.dummy_embeddings
        self.mock_recommender_instance.get_recommendations.return_value = [
            {
                "id": "3",
                "title": "Related Test Book 1",
                "authors": "Author A",
                "description": "Another book by Author A",
                "genres": "Fiction",
                "similarity": 0.8,
            },
            {
                "id": "1",
                "title": "Test Book 1",
                "authors": "Author A",
                "description": "Description for book 1",
                "genres": "Fiction",
                "similarity": 0.7,
            },
        ]
        self.mock_recommender_instance.get_recommendations_from_vector.return_value = [
            {
                "id": "5",
                "title": "Sci-Fi Classic",
                "authors": "Author D",
                "description": "Deep space exploration",
                "genres": "Science Fiction",
                "similarity": 0.9,
            },
            {
                "id": "2",
                "title": "Test Book 2",
                "authors": "Author B",
                "description": "Description for book 2 with sci-fi themes",
                "genres": "Science Fiction",
                "similarity": 0.8,
            },
        ]

        # 3. Override dependencies for the FastAPI app
        fastapi_app.dependency_overrides[get_sentence_transformer_model] = lambda: self.mock_model
        fastapi_app.dependency_overrides[get_recommender] = (
            lambda: self.mock_recommender_instance
        )  # Return the instance
        fastapi_app.dependency_overrides[get_clusters_data] = (
            lambda: self.mock_clusters_data_value
        )  # Return the actual value

        # self.client needs to be created *after* the dependencies are mocked
        self.client = TestClient(fastapi_app)

        # Ensure feedback file is clean before each test
        feedback_file_path = os.path.join(config.BASE_DIR, "data", "feedback", "user_feedback.jsonl")
        if os.path.exists(feedback_file_path):
            os.remove(feedback_file_path)

    def tearDown(self):
        # Clear dependency overrides after each test
        fastapi_app.dependency_overrides = {}
        super().tearDown()

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(), {"status": "OK", "message": "BookFinder API is healthy and core services are loaded."}
        )

    def test_recommend_by_query(self):
        response = self.client.post("/recommend/query", json={"query": "sci-fi books", "top_k": 2})
        self.assertEqual(response.status_code, 200)
        recommendations = response.json()
        self.assertEqual(len(recommendations), 2)
        self.assertEqual(recommendations[0]["book"]["title"], "Sci-Fi Classic")
        self.assertEqual(recommendations[1]["book"]["title"], "Test Book 2")

    def test_recommend_by_title(self):
        response = self.client.post("/recommend/title", json={"title": "Test Book 1", "top_k": 2})
        self.assertEqual(response.status_code, 200)
        recommendations = response.json()
        self.assertEqual(len(recommendations), 2)
        self.assertEqual(recommendations[0]["book"]["title"], "Related Test Book 1")
        self.assertEqual(recommendations[1]["book"]["title"], "Test Book 1")

    def test_list_books(self):
        response = self.client.get("/books?page=1&page_size=3")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total"], 5)
        self.assertEqual(len(data["books"]), 3)
        self.assertEqual(data["books"][0]["title"], "Test Book 1")

    def test_search_books(self):
        response = self.client.get("/books/search?query=author A&page=1&page_size=5")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total"], 2)
        self.assertEqual(len(data["books"]), 2)
        self.assertEqual(data["books"][0]["title"], "Test Book 1")
        self.assertEqual(data["books"][1]["title"], "Related Test Book 1")

    def test_get_stats(self):
        response = self.client.get("/stats")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_books"], 5)
        self.assertIn("fiction", data["genres_count"])
        self.assertIn("author a", data["authors_count"])

    def test_list_clusters(self):
        response = self.client.get("/clusters")
        self.assertEqual(response.status_code, 200)
        clusters = response.json()
        self.assertEqual(len(clusters), 3)
        self.assertEqual(clusters[0]["name"], "Fiction Collection")
        self.assertEqual(clusters[0]["size"], 2)  # Test Book 1, Related Test Book 1
        self.assertEqual(len(clusters[0]["top_books"]), 2)  # Should sample up to 3, but there are only 2

    def test_get_books_in_cluster(self):
        response = self.client.get("/clusters/0?page=1&page_size=10")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total"], 2)
        self.assertEqual(len(data["books"]), 2)
        self.assertEqual(data["books"][0]["title"], "Test Book 1")

    def test_get_cluster_sample(self):
        response = self.client.get("/clusters/1/sample?sample_size=1")
        self.assertEqual(response.status_code, 200)
        books = response.json()
        self.assertEqual(len(books), 1)
        self.assertEqual(books[0]["genres"], ["Science Fiction"])  # Only Sci-Fi books in cluster 1

    def test_explain_recommendation_endpoint(self):
        sample_book = {
            "id": "1",
            "title": "Test Book 1",
            "authors": ["Author A"],
            "description": "Description for book 1",
            "genres": ["Fiction"],
            "cover_image_url": None,
        }
        response = self.client.post(
            "/explain",
            json={"query_text": "A book about fiction", "recommended_book": sample_book, "similarity_score": 0.75},
        )
        self.assertEqual(response.status_code, 200)
        explanation = response.json()
        self.assertIn("match_score", explanation)
        self.assertIn("confidence", explanation)
        self.assertIn("summary", explanation)
        self.assertIn("details", explanation)

    def test_submit_feedback_and_get_stats(self):
        # Submit feedback
        feedback_payload = {
            "query": "fantasy adventure",
            "book_id": "4",  # Unrelated Book
            "feedback_type": "positive",
            "session_id": "test_session_1",
        }
        response = self.client.post("/feedback", json=feedback_payload)
        self.assertEqual(response.status_code, 204)  # No Content

        feedback_payload_2 = {
            "query": "sci-fi classic",
            "book_id": "5",  # Sci-Fi Classic
            "feedback_type": "negative",
            "session_id": "test_session_1",
        }
        response = self.client.post("/feedback", json=feedback_payload_2)
        self.assertEqual(response.status_code, 204)  # No Content

        # Get stats
        response = self.client.get("/feedback/stats")
        self.assertEqual(response.status_code, 200)
        stats = response.json()
        self.assertEqual(stats["total_feedback"], 2)
        self.assertEqual(stats["positive_feedback"], 1)
        self.assertEqual(stats["negative_feedback"], 1)
        self.assertIn("Unrelated Book", stats["feedback_by_book_title"])
        self.assertIn("fantasy adventure", stats["feedback_by_query"])
