import logging
import os # Import os to get environment variables for log level
import random
from fastapi import FastAPI, Depends, HTTPException, status, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

from src.book_recommender.api.models import (
    RecommendByQueryRequest,
    RecommendByTitleRequest,
    RecommendationResult,
    Book,
    BookStats,
    BookSearchResult,
    BookCluster,
    ExplanationResponse,
    ExplainRecommendationRequest,
    FeedbackRequest,
    FeedbackStatsResponse,
)
from src.book_recommender.api.dependencies import (
    get_recommender,
    get_sentence_transformer_model,
    get_clusters_data,
    limiter,
)
from src.book_recommender.ml.recommender import BookRecommender
from src.book_recommender.core.exceptions import DataNotFoundError
from src.book_recommender.core.logging_config import configure_logging
import src.book_recommender.core.config as config
from src.book_recommender.ml.explainability import explain_recommendation
from src.book_recommender.ml.feedback import save_feedback, get_all_feedback
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from contextlib import asynccontextmanager


# Configure logging at the very beginning
configure_logging(log_file="api.log", log_level=os.getenv("LOG_LEVEL", "INFO"))

logger = logging.getLogger(__name__) # Now this logger will use the configured settings

IS_TESTING = os.getenv("TESTING_ENV", "False").lower() == "true"

def log_exception(e: Exception):
    """Logs an exception with traceback if in DEBUG mode, otherwise logs a generic message."""
    if logger.isEnabledFor(logging.DEBUG):
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
    else:
        logger.error("An unexpected error occurred.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML models
    if not IS_TESTING:
        logger.info("Application startup: Pre-loading recommender and embedding model...")
        try:
            get_recommender()
            get_sentence_transformer_model()
            get_clusters_data() # Also pre-load cluster data
            logger.info("Recommender, embedding model, and cluster data pre-loaded successfully.")
        except Exception as e:
            log_exception(e)
            raise
    else:
        logger.info("TESTING_ENV is true. Skipping startup model loading.")
    yield
    # Clean up the ML models and release the resources
    logger.info("Application shutdown: Cleaning up resources.")

app = FastAPI(
    title="BookFinder API",
    description="API for content-based book recommendations and book management.",
    version="0.1.0",
    lifespan=lifespan
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS Middleware for allowing cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    return response


@app.get(
    "/health",
    summary="Perform a health check",
    response_description="Return HTTP Status Code 200 (OK)",
)
@limiter.limit("10/minute")
async def health_check(request: Request):
    """
    Checks the health of the API and its core components.
    """
    try:
        # Attempt to get the recommender to ensure it's loaded and functional
        recommender = get_recommender()
        # You could add a more robust check here, e.g., recommender.index.is_ready()
        
        # Attempt to get the model to ensure it's loaded and functional
        model = get_sentence_transformer_model()
        # You could add a more robust check here, e.g., model.is_ready()

        # Attempt to get cluster data to ensure it's loaded and functional
        get_clusters_data()

        return {"status": "OK", "message": "BookFinder API is healthy and core services are loaded."}
    except Exception as e:
        log_exception(e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Core services (recommender/embedding model/clusters) are not available: {e}"
        )


@app.post(
    "/recommend/query",
    response_model=List[RecommendationResult],
    summary="Get book recommendations based on a natural language query",
)
@limiter.limit("10/minute")
async def recommend_by_query(
    request: Request,
    body: RecommendByQueryRequest,
    recommender: BookRecommender = Depends(get_recommender),
    model: SentenceTransformer = Depends(get_sentence_transformer_model),
):
    """
    Provides book recommendations by semantically comparing a natural language query
    against the book embedding database.
    """
    try:
        query_embedding = model.encode(body.query, show_progress_bar=False)
        recommendations = recommender.get_recommendations_from_vector(
            query_embedding, top_k=body.top_k
        )
        # Map recommender output to RecommendationResult model
        results = []
        for rec in recommendations:
            book = Book(
                id=str(rec["id"]),
                title=rec["title"],
                authors=(
                    rec.get("authors", "").split(", ")
                    if isinstance(rec.get("authors"), str)
                    else []
                ),
                description=rec.get("description"),
                genres=(
                    rec.get("genres", "").split(", ")
                    if isinstance(rec.get("genres"), str)
                    else []
                ),
                cover_image_url=rec.get(
                    "cover_image_url"
                ),  # Assuming cover_image_url might be in recommender output
            )
            results.append(
                RecommendationResult(book=book, similarity_score=rec["similarity"])
            )
        return results
    except DataNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        log_exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during recommendation.",
        )


@app.post(
    "/recommend/title",
    response_model=List[RecommendationResult],
    summary="Get similar books based on a book title",
)
@limiter.limit("10/minute")
async def recommend_by_title(
    request: Request,
    body: RecommendByTitleRequest,
    recommender: BookRecommender = Depends(get_recommender),
):
    """
    Provides recommendations for books similar to a given title.
    """
    try:
        recommendations = recommender.get_recommendations(
            body.title, top_k=body.top_k
        )
        if not recommendations:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Book with title '{body.title}' not found or no recommendations met the similarity threshold.",
            )

        # Map recommender output to RecommendationResult model
        results = []
        for rec in recommendations:
            book = Book(
                id=str(rec["id"]),
                title=rec["title"],
                authors=(
                    rec.get("authors", "").split(", ")
                    if isinstance(rec.get("authors"), str)
                    else []
                ),
                description=rec.get("description"),
                genres=(
                    rec.get("genres", "").split(", ")
                    if isinstance(rec.get("genres"), str)
                    else []
                ),
                cover_image_url=rec.get(
                    "cover_image_url"
                ),  # Assuming cover_image_url might be in recommender output
            )
            results.append(
                RecommendationResult(book=book, similarity_score=rec["similarity"])
            )
        return results
    except DataNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except HTTPException:  # Re-raise HTTPExceptions created above
        raise
    except Exception as e:
        log_exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during recommendation.",
        )


@app.get(
    "/books",
    response_model=BookSearchResult,
    summary="List all books with pagination",
)
@limiter.limit("10/minute")
async def list_books(
    request: Request,
    recommender: BookRecommender = Depends(get_recommender),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Number of items per page"),
):
    """
    Retrieves a paginated list of all books in the catalog.
    """
    try:
        all_books_df = recommender.book_data
        total_books = len(all_books_df)

        start_index = (page - 1) * page_size
        end_index = start_index + page_size
        paginated_books_df = all_books_df.iloc[start_index:end_index]

        books = []
        for _, rec in paginated_books_df.iterrows():
            book = Book(
                id=str(rec["id"]),
                title=rec["title"],
                authors=(
                    rec.get("authors", "").split(", ")
                    if isinstance(rec.get("authors"), str)
                    else []
                ),
                description=rec.get("description"),
                genres=(
                    rec.get("genres", "").split(", ")
                    if isinstance(rec.get("genres"), str)
                    else []
                ),
                cover_image_url=rec.get("cover_image_url"),
            )
            books.append(book)

        return BookSearchResult(
            books=books, total=total_books, page=page, page_size=page_size
        )
    except Exception as e:
        log_exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while listing books.",
        )


@app.get(
    "/books/search",
    response_model=BookSearchResult,
    summary="Search books by title or author with pagination",
)
@limiter.limit("10/minute")
async def search_books(
    request: Request,
    recommender: BookRecommender = Depends(get_recommender),
    query: str = Query(
        ...,
        min_length=2,
        max_length=255,
        description="Search query for title or author",
    ),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Number of items per page"),
):
    """
    Searches for books by matching the query against book titles or author names.
    The search is case-insensitive.
    """
    try:
        # Sanitize query
        sanitized_query = query.strip()
        if not sanitized_query:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty or just whitespace.",
            )

        all_books_df = recommender.book_data

        # Case-insensitive search
        mask = (
            all_books_df["title_lower"].str.contains(sanitized_query.lower(), na=False)
        ) | (
            all_books_df["authors_lower"].str.contains(
                sanitized_query.lower(), na=False
            )
        )

        filtered_books_df = all_books_df[mask]
        total_books = len(filtered_books_df)

        start_index = (page - 1) * page_size
        end_index = start_index + page_size
        paginated_books_df = filtered_books_df.iloc[start_index:end_index]

        books = []
        for _, rec in paginated_books_df.iterrows():
            book = Book(
                id=str(rec["id"]),
                title=rec["title"],
                authors=(
                    rec.get("authors", "").split(", ")
                    if isinstance(rec.get("authors"), str)
                    else []
                ),
                description=rec.get("description"),
                genres=(
                    rec.get("genres", "").split(", ")
                    if isinstance(rec.get("genres"), str)
                    else []
                ),
                cover_image_url=rec.get("cover_image_url"),
            )
            books.append(book)

        return BookSearchResult(
            books=books, total=total_books, page=page, page_size=page_size
        )
    except Exception as e:
        log_exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while searching books.",
        )


@app.get(
    "/stats",
    response_model=BookStats,
    summary="Get database statistics",
)
@limiter.limit("10/minute")
async def get_stats(
    request: Request, recommender: BookRecommender = Depends(get_recommender)
):
    """
    Provides various statistics about the book dataset, including total book count,
    genre distribution, and author distribution.
    """
    try:
        all_books_df = recommender.book_data
        total_books = len(all_books_df)

        # Genre counts
        # Assuming genres are comma-separated strings
        all_genres = (
            all_books_df["genres"].str.lower().str.split(", ").explode().dropna()
        )
        genres_count = all_genres.value_counts().to_dict()

        # Author counts
        # Assuming authors are comma-separated strings
        all_authors = (
            all_books_df["authors"].str.lower().str.split(", ").explode().dropna()
        )
        authors_count = all_authors.value_counts().to_dict()

        return BookStats(
            total_books=total_books,
            genres_count=genres_count,
            authors_count=authors_count,
        )
    except Exception as e:
        log_exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while fetching statistics.",
        )

# --- Cluster Endpoints ---

@app.get(
    "/clusters",
    response_model=List[BookCluster],
    summary="List all book clusters",
)
@limiter.limit("10/minute")
async def list_clusters(
    request: Request,
    clusters_data: tuple[np.ndarray, dict, pd.DataFrame] = Depends(get_clusters_data),
):
    """
    Retrieves a list of all identified book clusters, including their names, sizes,
    and a sample of top books from each cluster.
    """
    try:
        clusters_arr, cluster_names, book_data_with_clusters = clusters_data

        all_clusters = []
        for cluster_id, name in cluster_names.items():
            cluster_books_df = book_data_with_clusters[
                book_data_with_clusters["cluster_id"] == cluster_id
            ]

            # Get a sample of top books (e.g., 3 random books)
            sample_books = []
            if not cluster_books_df.empty:
                # Take a random sample or top 3 books by some criteria
                # For simplicity, let's take up to 3 random books
                sample_recs = cluster_books_df.sample(
                    min(len(cluster_books_df), 3)
                ).to_dict(orient="records")
                for rec in sample_recs:
                    sample_books.append(
                        Book(
                            id=str(rec["id"]),
                            title=rec["title"],
                            authors=(
                                rec.get("authors", "").split(", ")
                                if isinstance(rec.get("authors"), str)
                                else []
                            ),
                            description=rec.get("description"),
                            genres=(
                                rec.get("genres", "").split(", ")
                                if isinstance(rec.get("genres"), str)
                                else []
                            ),
                            cover_image_url=rec.get("cover_image_url"),
                        )
                    )

            all_clusters.append(
                BookCluster(
                    id=cluster_id,
                    name=name,
                    size=len(cluster_books_df),
                    top_books=sample_books,
                )
            )

        return all_clusters
    except Exception as e:
        log_exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while listing clusters.",
        )


@app.get(
    "/clusters/{cluster_id}",
    response_model=BookSearchResult,
    summary="Get books in a specific cluster with pagination",
)
@limiter.limit("10/minute")
async def get_books_in_cluster(
    request: Request,
    cluster_id: int,
    clusters_data: tuple[np.ndarray, dict, pd.DataFrame] = Depends(get_clusters_data),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Number of items per page"),
):
    """
    Retrieves a paginated list of books belonging to a specific cluster.
    """
    try:
        clusters_arr, cluster_names, book_data_with_clusters = clusters_data

        if cluster_id not in cluster_names:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Cluster with ID {cluster_id} not found.",
            )

        cluster_books_df = book_data_with_clusters[
            book_data_with_clusters["cluster_id"] == cluster_id
        ]
        total_books = len(cluster_books_df)

        start_index = (page - 1) * page_size
        end_index = start_index + page_size
        paginated_books_df = cluster_books_df.iloc[start_index:end_index]

        books = []
        for _, rec in paginated_books_df.iterrows():
            book = Book(
                id=str(rec["id"]),
                title=rec["title"],
                authors=(
                    rec.get("authors", "").split(", ")
                    if isinstance(rec.get("authors"), str)
                    else []
                ),
                description=rec.get("description"),
                genres=(
                    rec.get("genres", "").split(", ")
                    if isinstance(rec.get("genres"), str)
                    else []
                ),
                cover_image_url=rec.get("cover_image_url"),
            )
            books.append(book)

        return BookSearchResult(
            books=books, total=total_books, page=page, page_size=page_size
        )
    except HTTPException:
        raise  # Re-raise HTTPExceptions
    except Exception as e:
        log_exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while fetching books in cluster.",
        )


@app.get(
    "/clusters/{cluster_id}/sample",
    response_model=List[Book],
    summary="Get a random sample of books from a specific cluster",
)
@limiter.limit("10/minute")
async def get_cluster_sample(
    request: Request,
    cluster_id: int,
    clusters_data: tuple[np.ndarray, dict, pd.DataFrame] = Depends(get_clusters_data),
    sample_size: int = Query(
        5, ge=1, le=20, description="Number of sample books to return"
    ),
):
    """
    Retrieves a random sample of books from a specified cluster.
    """
    try:
        clusters_arr, cluster_names, book_data_with_clusters = clusters_data

        if cluster_id not in cluster_names:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Cluster with ID {cluster_id} not found.",
            )

        cluster_books_df = book_data_with_clusters[
            book_data_with_clusters["cluster_id"] == cluster_id
        ]

        if cluster_books_df.empty:
            return []  # Return empty list if cluster is empty

        # Take a random sample
        sample_df = cluster_books_df.sample(min(len(cluster_books_df), sample_size))

        books = []
        for _, rec in sample_df.iterrows():
            book = Book(
                id=str(rec["id"]),
                title=rec["title"],
                authors=(
                    rec.get("authors", "").split(", ")
                    if isinstance(rec.get("authors"), str)
                    else []
                ),
                description=rec.get("description"),
                genres=(
                    rec.get("genres", "").split(", ")
                    if isinstance(rec.get("genres"), str)
                    else []
                ),
                cover_image_url=rec.get("cover_image_url"),
            )
            books.append(book)

        return books
    except HTTPException:
        raise  # Re-raise HTTPExceptions
    except Exception as e:
        log_exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while fetching cluster sample.",
        )

@app.post(
    "/explain",
    response_model=ExplanationResponse,
    summary="Get an explanation for a book recommendation",
)
@limiter.limit("10/minute")
async def explain_recommendation_endpoint(
    request: Request, body: ExplainRecommendationRequest
):
    """
    Generates a human-readable explanation for why a specific book was recommended
    based on a user query and the book's attributes.
    """
    try:
        # The explain_recommendation function expects a dictionary for the book,
        # so we convert the Pydantic model to a dictionary.
        book_dict = body.recommended_book.model_dump()
        
        explanation = explain_recommendation(
            query_text=body.query_text,
            recommended_book=book_dict,
            similarity_score=body.similarity_score
        )
        return ExplanationResponse(**explanation)
    except Exception as e:
        log_exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during explanation generation.",
        )


# --- Feedback Endpoints ---


@app.post(
    "/feedback",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Submit user feedback on a recommendation",
)
@limiter.limit("10/minute")
async def submit_feedback(
    request: Request,
    body: FeedbackRequest,
    recommender: BookRecommender = Depends(
        get_recommender
    ),  # Use recommender to get full book details if needed
):
    """
    Allows users to submit positive or negative feedback on a book recommendation.
    """
    try:
        # Fetch book details using book_id from the recommender's book_data
        book_details_df = recommender.book_data[
            recommender.book_data["id"] == body.book_id
        ]
        if book_details_df.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Book with ID {body.book_id} not found.",
            )

        book_details = book_details_df.iloc[0].to_dict()

        save_feedback(
            query=body.query,
            book_details=book_details,
            feedback_type=body.feedback_type,
            session_id=body.session_id,
        )
        return {"message": "Feedback submitted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        log_exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while submitting feedback.",
        )


@app.get(
    "/feedback/stats",
    response_model=FeedbackStatsResponse,
    summary="Get aggregate feedback statistics",
)
@limiter.limit("10/minute")
async def get_feedback_stats(request: Request):
    """
    Retrieves aggregated statistics about the collected user feedback.
    """
    try:
        all_feedback = get_all_feedback()

        total_feedback = len(all_feedback)
        positive_feedback = sum(1 for f in all_feedback if f["feedback"] == "positive")
        negative_feedback = sum(1 for f in all_feedback if f["feedback"] == "negative")

        feedback_by_book_title: Dict[str, Dict[str, int]] = {}
        feedback_by_query: Dict[str, Dict[str, int]] = {}

        for entry in all_feedback:
            book_title = entry.get("book_title", "Unknown Book")
            query = entry.get("query", "Unknown Query")
            feedback_type = entry["feedback"]

            feedback_by_book_title.setdefault(book_title, {"positive": 0, "negative": 0})[
                feedback_type
            ] += 1
            feedback_by_query.setdefault(query, {"positive": 0, "negative": 0})[
                feedback_type
            ] += 1

        return FeedbackStatsResponse(
            total_feedback=total_feedback,
            positive_feedback=positive_feedback,
            negative_feedback=negative_feedback,
            feedback_by_book_title=feedback_by_book_title,
            feedback_by_query=feedback_by_query,
        )
    except Exception as e:
        log_exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while fetching feedback statistics.",
        )


def main():
    """Entry point for uvicorn"""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()