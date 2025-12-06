import requests
import os
from typing import List, Dict, Optional

class PersonalizationService:
    def __init__(self):
        # Use env var for Docker compatibility, default to localhost
        # The Service B (Personalizer) runs on Port 8001
        self.base_url = os.getenv("PERSONALIZER_URL", "http://localhost:8001")
        self.timeout = 60.0  # Increased timeout for CPU inference

    def get_recommendations(self, user_history: List[str], top_k: int = 5) -> List[Dict]:
        """
        Call the external personalization API.
        Returns a list of dicts: {'title': str, 'score': float, 'genres': str}
        """
        if not user_history:
            return []

        try:
            payload = {
                "user_history": user_history,
                "top_k": top_k
            }
            response = requests.post(
                f"{self.base_url}/personalize/recommend", 
                json=payload, 
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                # Log error but don't crash
                print(f"Personalizer Error: {response.status_code}")
                return []
                
        except requests.exceptions.RequestException:
            # Circuit Breaker pattern: If service is down, return empty list
            # so the main app doesn't crash.
            print("Personalizer Service Unreachable")
            return []

    def semantic_search(self, query: str) -> List[Dict]:
        """
        Search for books by plot/vibe.
        """
        try:
            response = requests.post(
                f"{self.base_url}/search",
                json={"query": query, "top_k": 10},
                timeout=self.timeout
            )
            return response.json() if response.status_code == 200 else []
        except:
            return []
