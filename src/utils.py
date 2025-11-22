# src/utils.py - IMPROVED

import os
import logging
import requests
from functools import lru_cache
import urllib.parse

logger = logging.getLogger(__name__)

# Multiple placeholder options
PLACEHOLDER_IMAGES = [
    "https://images.unsplash.com/photo-1543002588-bfa74002ed7e?w=300&h=450&fit=crop",  # Book stack
    "https://images.unsplash.com/photo-1512820790803-83ca734da794?w=300&h=450&fit=crop",  # Open book
    "https://images.unsplash.com/photo-1495446815901-a7297e633e8d?w=300&h=450&fit=crop",  # Books
]

def ensure_dir_exists(file_path: str):
    """Ensures that the directory for a given file path exists."""
    try:
        output_dir = os.path.dirname(file_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Error creating directory for {file_path}: {e}")
        raise

@lru_cache(maxsize=256)
def get_cover_url_multi_source(title: str, author: str) -> str:
    """
    Fetch book cover from multiple sources with fallback chain.
    
    Priority order:
    1. Google Books API (best quality, most reliable)
    2. Open Library API
    3. Beautiful placeholder from Unsplash
    """
    # Try Google Books first
    cover = _get_cover_from_google_books(title, author)
    if cover:
        return cover
    
    # Try Open Library
    cover = _get_cover_from_openlibrary(title, author)
    if cover:
        return cover
    
    # Return beautiful placeholder
    import random
    return random.choice(PLACEHOLDER_IMAGES)

def _get_cover_from_google_books(title: str, author: str) -> str:
    """Fetch cover from Google Books API."""
    try:
        query = f"{title} {author}".strip()
        encoded_query = urllib.parse.quote(query)
        url = f"https://www.googleapis.com/books/v1/volumes?q={encoded_query}&maxResults=1"
        
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data.get('totalItems', 0) > 0:
            items = data.get('items', [])
            if items and 'volumeInfo' in items[0]:
                volume_info = items[0]['volumeInfo']
                image_links = volume_info.get('imageLinks', {})
                
                # Try to get the largest available image
                for size in ['large', 'medium', 'small', 'thumbnail', 'smallThumbnail']:
                    if size in image_links:
                        cover_url = image_links[size]
                        # Force HTTPS
                        cover_url = cover_url.replace('http://', 'https://')
                        logger.info(f"Found Google Books cover for '{title}'")
                        return cover_url
        
        return None
    except Exception as e:
        logger.debug(f"Google Books API failed for '{title}': {e}")
        return None

def _get_cover_from_openlibrary(title: str, author: str) -> str:
    """Fetch cover from Open Library API."""
    try:
        search_url = f"https://openlibrary.org/search.json?title={urllib.parse.quote(title)}&author={urllib.parse.quote(author)}"
        
        response = requests.get(search_url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data.get('numFound', 0) > 0:
            docs = data.get('docs', [])
            if docs:
                for doc in docs:
                    if 'isbn' in doc and doc['isbn']:
                        isbn = doc['isbn'][0]
                        cover_url = f"https://covers.openlibrary.org/b/isbn/{isbn}-L.jpg"
                        logger.info(f"Found Open Library cover for '{title}'")
                        return cover_url
        
        return None
    except Exception as e:
        logger.debug(f"Open Library API failed for '{title}': {e}")
        return None

from concurrent.futures import ThreadPoolExecutor
import streamlit as st

@st.cache_data
def load_book_covers_batch(books):
    """Pre-fetch covers in batch for faster display"""
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(get_cover_url_multi_source, book['title'], book.get('authors', '')): book
            for book in books
        }
        
        results = {}
        for future in futures:
            book = futures[future]
            try:
                # Use a unique key for each book, like its title
                results[book['title']] = future.result()
            except Exception as e:
                logger.error(f"Error loading cover for {book['title']}: {e}")
                # Assign a placeholder if fetching fails
                results[book['title']] = PLACEHOLDER_IMAGES[0]
        
        return results