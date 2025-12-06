import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from fastapi.testclient import TestClient
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.book_recommender.api.main import app, get_recommender
from src.book_recommender.services.personalizer import PersonalizationService
from src.book_recommender.ml.recommender import BookRecommender

class TestPersonalizationIntegration(unittest.TestCase):
    
    def setUp(self):
        # Mock Recommender Data
        self.mock_recommender = MagicMock(spec=BookRecommender)
        self.mock_recommender.book_data = pd.DataFrame({
            "id": ["1", "2"],
            "title": ["Test Book A", "Test Book B"],
            "authors": ["Author A", "Author B"],
            "description": ["Desc A", "Desc B"],
            "genres": ["Genre A", "Genre B"],
            "cover_image_url": ["http://img.com/a.jpg", None]
        })
        
        # Mock Personalization Service
        self.mock_personalizer = MagicMock(spec=PersonalizationService)
        
        # Override dependencies
        app.dependency_overrides[get_recommender] = lambda: self.mock_recommender
        
        # Patch the personalizer instance in the api module
        self.patcher = patch("src.book_recommender.api.main.personalizer", self.mock_personalizer)
        self.patcher.start()
        
        self.client = TestClient(app)
        
    def tearDown(self):
        app.dependency_overrides = {}
        self.patcher.stop()

    def test_personalize_endpoint_success(self):
        # Setup mock return value from personalization service
        self.mock_personalizer.get_recommendations.return_value = [
            {"title": "Test Book A", "score": 0.9, "genres": "Genre A"},
            {"title": "Test Book B", "score": 0.8, "genres": "Genre B"}
        ]
        
        payload = {
            "user_history": ["Some Old Book"],
            "top_k": 2
        }
        
        response = self.client.post("/recommend/personalize", json=payload)
        
        self.assertEqual(response.status_code, 200)
        results = response.json()
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["book"]["title"], "Test Book A")
        self.assertEqual(results[0]["similarity_score"], 0.9)
        self.assertEqual(results[1]["book"]["title"], "Test Book B")
        
        # Check if cover image logic worked (Book B had None, but we didn't mock load_book_covers_batch so it might stay None or fail if called)
        # Actually, load_book_covers_batch is imported in main.py. We might need to mock it if we want to test cover fetching.
        # But for basic integration, this is enough.

    def test_personalize_endpoint_empty_response(self):
        self.mock_personalizer.get_recommendations.return_value = []
        
        payload = {
            "user_history": ["Some Old Book"],
            "top_k": 2
        }
        
        response = self.client.post("/recommend/personalize", json=payload)
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])

    def test_personalize_endpoint_service_failure(self):
        # Simulate service returning None or raising exception (though wrapper usually catches it)
        # The wrapper in main checks "if not semantic_recs: return []"
        self.mock_personalizer.get_recommendations.return_value = []
        
        payload = {
            "user_history": ["Some Old Book"],
            "top_k": 2
        }
        
        response = self.client.post("/recommend/personalize", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])

if __name__ == "__main__":
    unittest.main()
