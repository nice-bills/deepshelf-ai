import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from book_recommender.utils import ensure_dir_exists

logger = logging.getLogger(__name__)

FEEDBACK_DIR = "data/feedback"
FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, "user_feedback.jsonl")


def save_feedback(
    query: str, book_details: Dict[str, Any], feedback_type: str, session_id: Optional[str] = None
) -> None:
    """
    Saves user feedback to a JSONL file.

    Args:
        query (str): The original query text that led to the recommendation.
        book_details (Dict[str, Any]): Details of the book for which feedback is given.
                                      Should include 'id', 'title', 'authors'.
        feedback_type (str): Type of feedback, e.g., "positive", "negative".
        session_id (Optional[str]): Unique identifier for the user session.
    """
    ensure_dir_exists(FEEDBACK_FILE)

    feedback_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "book_id": book_details.get("id"),
        "book_title": book_details.get("title"),
        "book_authors": book_details.get("authors"),
        "feedback": feedback_type,
        "session_id": session_id,
    }

    try:
        with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_entry) + "\n")
        logger.info(f"Feedback saved: {feedback_type} for '{book_details.get('title')}'")
    except Exception as e:
        logger.error(f"Failed to save feedback to {FEEDBACK_FILE}: {e}")


def get_all_feedback() -> List[Dict[str, Any]]:
    """
    Loads all saved user feedback from the JSONL file.

    Returns:
        List[Dict[str, Any]]: A list of feedback entries.
    """
    if not os.path.exists(FEEDBACK_FILE):
        return []

    feedback_data = []
    try:
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            for line in f:
                feedback_data.append(json.loads(line))
    except Exception as e:
        logger.error(f"Failed to load feedback from {FEEDBACK_FILE}: {e}")

    return feedback_data


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger.info("--- Testing feedback system ---")

    sample_book_1 = {
        "id": "book1",
        "title": "The Great Novel",
        "authors": ["Author A"],
        "description": "...",
        "genres": "Fiction",
    }
    sample_book_2 = {
        "id": "book2",
        "title": "Another Story",
        "authors": ["Author B"],
        "description": "...",
        "genres": "Fantasy",
    }

    save_feedback("sci-fi book", sample_book_1, "positive", "session123")
    save_feedback("fantasy epic", sample_book_2, "negative", "session123")
    save_feedback("classic literature", sample_book_1, "positive", "session456")

    all_feedback = get_all_feedback()
    print(f"\nTotal feedback entries: {len(all_feedback)}")
    for entry in all_feedback:
        print(entry)
