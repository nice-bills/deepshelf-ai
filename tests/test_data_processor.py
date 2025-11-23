# tests/test_data_processor.py

import os
import tempfile
import unittest

import pandas as pd

from src.book_recommender.core.exceptions import DataNotFoundError, FileProcessingError
from src.book_recommender.data.processor import clean_and_prepare_data


class TestDataProcessor(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for test artifacts
        self.test_dir = tempfile.TemporaryDirectory()
        self.raw_path = os.path.join(self.test_dir.name, "test_books.csv")
        self.processed_path = os.path.join(self.test_dir.name, "test_books_cleaned.parquet")

        # Create a dummy CSV file
        self.sample_data = {
            "title": ["Book A", "Book B", None, "Book D"],
            "authors": ["Author 1", "Author 2", "Author 3", "Author 4"],
            "genres": ["['Fiction']", "['Sci-Fi', 'Fantasy']", "['History']", float("nan")],
            "description": ["Desc A", "Desc B", "Desc C", "Desc D"],
            "tags": ["['tag1']", "['tag2']", "['tag3']", "['tag4']"],
        }
        pd.DataFrame(self.sample_data).to_csv(self.raw_path, index=False)

    def tearDown(self):
        # Clean up the temporary directory
        self.test_dir.cleanup()

    def test_clean_and_prepare_data(self):
        # Run the function to be tested
        processed_df = clean_and_prepare_data(self.raw_path, self.processed_path)

        # 1. Test that the output file is created
        self.assertTrue(os.path.exists(self.processed_path))

        # 2. Test that rows with no title are dropped
        self.assertEqual(len(processed_df), 3)
        self.assertNotIn(None, processed_df["title_lower"])

        # 3. Test that NaN values are filled and list-strings are parsed
        self.assertFalse(processed_df["genres"].isnull().any())
        self.assertEqual(processed_df.iloc[0]["genres"], "fiction")

        # 4. Test that 'combined_text' column is created and is correct
        self.assertIn("combined_text", processed_df.columns)

        # Check the content for the first valid book
        expected_text = "book a book a book a by author 1. genres: fiction. description: desc a. tags: tag1"
        self.assertEqual(processed_df.iloc[0]["combined_text"], expected_text)

    def test_missing_raw_data_file(self):
        """Test that DataNotFoundError is raised if the raw data file is missing."""
        with self.assertRaises(DataNotFoundError) as cm:
            clean_and_prepare_data(raw_path="non_existent_file.csv", processed_path=self.processed_path)
        self.assertIsInstance(cm.exception, DataNotFoundError)

    def test_empty_dataframe(self):
        """Test that a ValueError is raised if the dataframe is empty after cleaning."""
        # Create an empty csv
        empty_df = pd.DataFrame(columns=["title", "authors", "genres", "description", "tags"])
        empty_df.to_csv(self.raw_path, index=False)

        with self.assertRaises(ValueError) as cm:
            clean_and_prepare_data(raw_path=self.raw_path, processed_path=self.processed_path)
        self.assertIsInstance(cm.exception, ValueError)

    def test_malformed_csv(self):
        """Test that FileProcessingError is raised for a malformed CSV."""
        # Create a malformed CSV file
        with open(self.raw_path, "w") as f:
            f.write('title,authors\n"Book","Author"\n"Another Book')  # Missing closing quote

        with self.assertRaises(FileProcessingError) as cm:
            clean_and_prepare_data(raw_path=self.raw_path, processed_path=self.processed_path)
        self.assertIsInstance(cm.exception, FileProcessingError)


if __name__ == "__main__":
    unittest.main()
