# src/utils.py

import os
import logging
import requests
from functools import lru_cache

# A placeholder image for books where no cover is found
PLACEHOLDER_IMAGE_URL = "https://via.placeholder.com/180x250.png?text=No+Cover"

def ensure_dir_exists(file_path: str):
    """
    Ensures that the directory for a given file path exists.

    Args:
        file_path (str): The path to the file.
    """
    try:
        output_dir = os.path.dirname(file_path)
        if output_dir: # Ensure it's not an empty string
            os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        logging.error(f"Error creating directory for {file_path}: {e}")
        raise

@lru_cache(maxsize=128)
def get_cover_url(title: str, author: str) -> str:
    """
    Fetches the book cover URL from the Open Library API based on title and author.
    Uses LRU cache to avoid repeated API calls for the same book.

    Args:
        title (str): The title of the book.
        author (str): The author of the book.

    Returns:
        str: The URL of the book cover image, or a placeholder if not found.
    """
    search_url = f"https://openlibrary.org/search.json?title={title}&author={author}"
    logging.info(f"Searching for cover with URL: {search_url}")
    try:
        response = requests.get(search_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get('numFound', 0) > 0:
            docs = data.get('docs', [])
            if docs:
                for doc in docs:
                    if 'isbn' in doc and doc['isbn']:
                        isbn = doc['isbn'][0]
                        cover_url = f"https://covers.openlibrary.org/b/isbn/{isbn}-M.jpg"
                        logging.info(f"Found ISBN {isbn} for '{title}'. Cover URL: {cover_url}")
                        return cover_url
        
        logging.warning(f"No cover found for '{title}' by {author}. Using placeholder.")
        return PLACEHOLDER_IMAGE_URL
        
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed for '{title}': {e}")
        return PLACEHOLDER_IMAGE_URL
    except Exception as e:
        logging.error(f"An unexpected error occurred while fetching cover for '{title}': {e}")
        return PLACEHOLDER_IMAGE_URL
