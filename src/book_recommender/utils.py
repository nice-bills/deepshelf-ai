import logging
import os
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Optional
import difflib

import requests


logger = logging.getLogger(__name__)

PLACEHOLDER_IMAGES = [
    "https://images.unsplash.com/photo-1543002588-bfa74002ed7e?w=300&h=450&fit=crop",
    "https://images.unsplash.com/photo-1512820790803-83ca734da794?w=300&h=450&fit=crop",
    "https://images.unsplash.com/photo-1495446815901-a7297e633e8d?w=300&h=450&fit=crop",
]


def _strings_are_similar(s1: str, s2: str, threshold: float = 0.6) -> bool:
    """Check if two strings are similar using sequence matching or containment."""
    if not s1 or not s2:
        return False
    s1, s2 = s1.lower(), s2.lower()
    # Check for containment (handles substrings like "Harry Potter" in "Harry Potter and the...")
    if s1 in s2 or s2 in s1:
        return True
    return difflib.SequenceMatcher(None, s1, s2).ratio() > threshold


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
    cover = _get_cover_from_google_books(title, author)
    if cover:
        return cover

    cover = _get_cover_from_openlibrary(title, author)
    if cover:
        return cover

    import random

    return random.choice(PLACEHOLDER_IMAGES)


def _get_cover_from_google_books(title: str, author: str) -> Optional[str]:
    """Fetch cover from Google Books API."""
    try:
        query = f"{title} {author}".strip()
        encoded_query = urllib.parse.quote(query)
        
        base_url = "https://www.googleapis.com/books/v1/volumes"
        url = f"{base_url}?q={encoded_query}&maxResults=1"
        
        # Add API key if available to avoid rate limiting
        api_key = os.getenv("GOOGLE_BOOKS_API_KEY")
        if api_key:
            url += f"&key={api_key}"

        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()

        if data.get("totalItems", 0) > 0:
            items = data.get("items", [])
            if items and "volumeInfo" in items[0]:
                volume_info = items[0]["volumeInfo"]
                
                # Validate match to avoid false positives
                found_title = volume_info.get("title", "")
                if not _strings_are_similar(title, found_title):
                    logger.info(f"Google Books mismatch: queried '{title}', got '{found_title}'. Skipping.")
                    return None

                image_links = volume_info.get("imageLinks", {})

                for size in ["large", "medium", "small", "thumbnail", "smallThumbnail"]:
                    if size in image_links:
                        cover_url: str = image_links[size]
                        cover_url = cover_url.replace("http://", "https://")
                        logger.info(f"Found Google Books cover for '{title}'")
                        return cover_url

        return None
    except Exception as e:
        logger.debug(f"Google Books API failed for '{title}': {e}")
        return None


def _get_cover_from_openlibrary(title: str, author: str) -> Optional[str]:
    """Fetch cover from Open Library API."""
    try:
        search_url = (
            f"https://openlibrary.org/search.json?title={urllib.parse.quote(title)}&author={urllib.parse.quote(author)}"
        )

        response = requests.get(search_url, timeout=5)
        response.raise_for_status()
        data = response.json()

        if data.get("numFound", 0) > 0:
            docs = data.get("docs", [])
            if docs:
                for doc in docs:
                    if "isbn" in doc and doc["isbn"]:
                        isbn = doc["isbn"][0]
                        cover_url = f"https://covers.openlibrary.org/b/isbn/{isbn}-L.jpg"
                        logger.info(f"Found Open Library cover for '{title}'")
                        return cover_url

        return None
    except Exception as e:
        logger.debug(f"Open Library API failed for '{title}': {e}")
        return None


def load_book_covers_batch(books):
    """Pre-fetch covers in batch, using existing URLs if available."""
    results = {}
    books_to_fetch = []

    for book in books:
        # Check if we already have a valid URL from our enriched dataset
        existing_url = book.get("cover_image_url")
        if existing_url and isinstance(existing_url, str) and len(existing_url) > 10:
             results[book["title"]] = existing_url
        else:
             books_to_fetch.append(book)

    if not books_to_fetch:
        return results

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(get_cover_url_multi_source, book["title"], book.get("authors", "")): book for book in books_to_fetch
        }

        for future in futures:
            book = futures[future]
            try:
                results[book["title"]] = future.result()
            except Exception as e:
                logger.error(f"Error loading cover for {book['title']}: {e}")
                results[book["title"]] = PLACEHOLDER_IMAGES[0]

        return results


def fetch_book_cover(title: str, author: str) -> Optional[str]:
    """
    Fetch book cover from multiple sources with fallback chain.

    Priority order:
    1. Google Books API (best quality, most reliable)
    2. Open Library API
    3. Placeholder image

    Args:
        title (str): Book title
        author (str): Book author

    Returns:
        Optional[str]: URL to book cover or None if not found
    """
    return get_cover_url_multi_source(title, author)
