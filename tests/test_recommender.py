# tests/test_recommender.py

import unittest
import os
import pandas as pd
import numpy as np
import tempfile
from src.recommender import BookRecommender
import src.config as config # Import config for EMBEDDING_DIMENSION

class TestRecommender(unittest.TestCase):

    def setUp(self):
        # Create dummy data directly in memory
        self.book_data = pd.DataFrame({
            'title': ['Alpha Book', 'Beta Book', 'Gamma Book', 'Delta Book', 'Epsilon Book'],
            'title_lower': ['alpha book', 'beta book', 'gamma book', 'delta book', 'epsilon book'],
            'authors': ['Auth1', 'Auth2', 'Auth3', 'Auth1', 'Auth5'],
            'authors_lower': ['auth1', 'auth2', 'auth3', 'auth1', 'auth5'],
            'description': ['Desc1', 'Desc2', 'Desc3', 'Desc1', 'Desc5'],
            'genres': ['Fiction', 'Sci-Fi', 'History', 'Fiction', 'Fantasy'],
            'tags': ['tag1', 'tag2', 'tag3', 'tag4', 'tag5'],
            'rating': [4.5, 4.0, 4.8, 4.3, 4.6],
            'combined_text': ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
        })

        # Create dummy embeddings (simple, not from a real model)
        # Make 'alpha book' very similar to 'delta book'
        # Make 'beta book' very similar to 'epsilon book'
        self.embeddings = np.array([
            [0.9, 0.1, 0.1, 0.1, 0.1],  # alpha
            [0.1, 0.9, 0.1, 0.1, 0.1],  # beta
            [0.1, 0.1, 0.9, 0.1, 0.1],  # gamma
            [0.85, 0.15, 0.1, 0.1, 0.1], # delta (similar to alpha)
            [0.1, 0.8, 0.1, 0.1, 0.1],  # epsilon (similar to beta)
        ])
        # Pad embeddings to match expected dimension if necessary
        if self.embeddings.shape[1] < config.EMBEDDING_DIMENSION:
            padding = np.zeros((self.embeddings.shape[0], config.EMBEDDING_DIMENSION - self.embeddings.shape[1]))
            self.embeddings = np.hstack((self.embeddings, padding))

        # Initialize the recommender with in-memory data
        self.recommender = BookRecommender(book_data=self.book_data, embeddings=self.embeddings)

    def test_get_recommendations(self):
        # Test with a book that exists
        recs = self.recommender.get_recommendations('alpha book', top_k=2)

        # 1. Should return top_k recommendations
        self.assertEqual(len(recs), 2)

        # 2. The input book itself should not be in the recommendations
        rec_titles = [r['title'] for r in recs]
        self.assertNotIn('alpha book', rec_titles)

        # 3. The most similar book ('delta book') should be first
        self.assertEqual(recs[0]['title'], 'Delta Book')
        
        # 4. Check the structure of the output
        self.assertIsInstance(recs[0], dict)
        self.assertIn('title', recs[0])
        self.assertIn('authors', recs[0])
        self.assertIn('description', recs[0])
        self.assertIn('genres', recs[0]) 
        self.assertIn('tags', recs[0])
        self.assertIn('rating', recs[0])
        self.assertIn('similarity', recs[0])

    def test_get_recommendations_non_existent_title(self):
        # Test with a book title that does not exist
        recs = self.recommender.get_recommendations('non existent book')
        self.assertEqual(len(recs), 0)

    def test_recommender_accuracy(self):
        """
        Tests if the recommender returns expected similar books based on dummy embeddings.
        """
        # Test for 'beta book', expecting 'epsilon book' as the most similar
        recs = self.recommender.get_recommendations('beta book', top_k=1)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]['title'], 'Epsilon Book')
        self.assertTrue(recs[0]['similarity'] > 0.7) # Expect a high similarity for a known similar book

if __name__ == '__main__':
    unittest.main()
