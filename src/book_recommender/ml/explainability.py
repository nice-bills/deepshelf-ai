import logging
import os
from typing import Any, Dict, List, Union

try:
    from groq import Groq
except ImportError:
    Groq = None

logger = logging.getLogger(__name__)


def _normalize_to_list(field_value: Union[str, List[str], None]) -> List[str]:
    """Normalize a string (comma-separated) or list of strings into a list of strings."""
    if not field_value:
        return []
    if isinstance(field_value, list):
        return [str(item).strip() for item in field_value]
    if isinstance(field_value, str):
        return [item.strip() for item in field_value.split(",") if item.strip()]
    return []


def _generate_rule_based_summary(
    query_text: str, recommended_book: Dict[str, Any], contribution_scores: Dict[str, float]
) -> str:
    """Fallback: Generates a simple rule-based summary."""
    summary = f"Recommended because it's a good match for your interest in '{query_text}'. "

    matching_features = []
    if contribution_scores.get("genres", 0) > 0.1:
        book_genres_list = _normalize_to_list(recommended_book.get("genres"))
        if book_genres_list:
            matching_features.append(f"shares genres like {', '.join(book_genres_list[:3])}")

    if contribution_scores.get("description_keywords", 0) > 0.1:
        matching_features.append("has matching keywords in the description")

    if contribution_scores.get("authors", 0) > 0.1:
        book_authors_list = _normalize_to_list(recommended_book.get("authors"))
        if book_authors_list and book_authors_list != ["Unknown Author"]:
            matching_features.append(f"is by author(s) {', '.join(book_authors_list)}")

    if matching_features:
        summary += "Specifically, it " + ", and ".join(matching_features) + "."
    else:
        summary += "Its content aligns well with your query."

    return summary


def _generate_llm_summary(query_text: str, recommended_book: Dict[str, Any]) -> str | None:
    """Generates a personalized explanation using Groq (Llama 3)."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or not Groq:
        return None

    try:
        client = Groq(api_key=api_key)
        
        # System Prompt: Sets the persona and constraints
        system_content = (
            "You are an objective literary analyst. "
            "Your goal is to explain the specific thematic connection between a user's request and a book recommendation. "
            "Be insightful, concise, and direct. Do not shy away from mature themes if they are relevant to the book's content."
        )

        # User Prompt: Provides the data and specific task
        user_content = f"""
        User Request: "{query_text}"
        
        Book Metadata:
        - Title: {recommended_book.get('title')}
        - Author: {recommended_book.get('authors')}
        - Synopsis: {recommended_book.get('description')}
        
        Task: Write a 2-sentence explanation of why this book fits the user's request. 
        
        Guidelines:
        1. Explicitly link the specific themes or plot points in the synopsis to the user's keywords.
        2. Do NOT simply summarize the book. Focus on the *match*.
        3. Avoid generic phrases like "This book is a great choice." Start directly with the insight.
        4. If the match is loose, focus on the genre or mood overlap.
        """

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ],
            temperature=0.6, # Slightly lower for more focused results
            max_tokens=120,
            top_p=1,
            stop=None,
            stream=False,
        )

        return completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq API failed: {e}")
        return None


def get_contribution_scores(query_text: str, recommended_book: Dict[str, Any]) -> Dict[str, float]:
    """Calculates hypothetical contribution scores for different features."""
    scores = {"genres": 0.0, "description_keywords": 0.0, "authors": 0.0}
    # Simple logic kept for metadata details
    query_words = set(query_text.lower().split())
    book_desc = recommended_book.get("description", "").lower()
    
    if any(w in book_desc for w in query_words):
        scores["description_keywords"] = 0.8
        
    return scores


def explain_recommendation(
    query_text: str, recommended_book: Dict[str, Any], similarity_score: float
) -> Dict[str, Any]:
    """
    Generates an explanation for why a particular book was recommended.
    Tries LLM first, falls back to rules.
    """
    logger.info(f"Generating explanation for '{recommended_book.get('title')}'")

    # 1. Try LLM
    summary = _generate_llm_summary(query_text, recommended_book)
    
    # 2. Fallback to Rule-Based
    contribution_scores = get_contribution_scores(query_text, recommended_book)
    if not summary:
        summary = _generate_rule_based_summary(query_text, recommended_book, contribution_scores)

    confidence = "HIGH" if similarity_score > 0.7 else "MEDIUM"

    return {
        "match_score": round(similarity_score * 100),
        "confidence": confidence,
        "summary": summary,
        "details": {
            "genres_contribution": round(contribution_scores["genres"] * 100),
            "description_keywords_contribution": round(contribution_scores["description_keywords"] * 100),
            "authors_contribution": round(contribution_scores["authors"] * 100),
        },
    }
