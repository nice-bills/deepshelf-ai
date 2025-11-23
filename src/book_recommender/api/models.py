from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class RecommendByQueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=255,
        strip_whitespace=True,
        json_schema_extra={"example": "science fiction about space travel"},
    )
    top_k: int = Field(5, gt=0, le=100, json_schema_extra={"example": 5})


class RecommendByTitleRequest(BaseModel):
    title: str = Field(
        ...,
        min_length=1,
        max_length=255,
        strip_whitespace=True,
        json_schema_extra={"example": "Dune"},
    )
    top_k: int = Field(5, gt=0, le=100, json_schema_extra={"example": 5})


class Book(BaseModel):
    id: str
    title: str
    authors: List[str]
    description: Optional[str] = None
    genres: List[str]
    cover_image_url: Optional[str] = None


class RecommendationResult(BaseModel):
    book: Book
    similarity_score: float


class BookStats(BaseModel):
    total_books: int
    genres_count: dict
    authors_count: dict


class BookSearchResult(BaseModel):
    books: List[Book]
    total: int
    page: int
    page_size: int


class BookCluster(BaseModel):
    id: int
    name: str
    size: int
    top_books: List[Book]  # Sample top books in the cluster


class ExplainRecommendationRequest(BaseModel):
    query_text: str = Field(..., json_schema_extra={"example": "A fantasy novel with dragons"})
    recommended_book: Book  # Reuse the Book model
    similarity_score: float = Field(..., ge=0.0, le=1.0)


class ExplanationResponse(BaseModel):
    match_score: int
    confidence: str
    summary: str
    details: Dict[str, int]  # Assuming details are contribution percentages


class FeedbackRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=255,
        strip_whitespace=True,
        json_schema_extra={"example": "books about dragons"},
    )
    book_id: str = Field(
        ...,
        min_length=1,
        max_length=36,
        strip_whitespace=True,
        json_schema_extra={"example": "b1b2b3b4-b5b6-b7b8-b9b0-b1b2b3b4b5b6"},
    )
    feedback_type: str = Field(..., pattern="^(positive|negative)$")  # "positive" or "negative"
    session_id: Optional[str] = Field(
        None,
        min_length=1,
        max_length=128,
        strip_whitespace=True,
        json_schema_extra={"example": "user-session-12345"},
    )


class FeedbackStatsResponse(BaseModel):
    total_feedback: int
    positive_feedback: int
    negative_feedback: int
    feedback_by_book_title: Dict[str, Dict[str, int]]
    feedback_by_query: Dict[str, Dict[str, int]]
