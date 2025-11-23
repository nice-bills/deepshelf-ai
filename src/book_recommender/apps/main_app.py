# app.py - REDESIGNED

import json
import logging  # Keep logging import for logger instance
import os
import uuid  # New import for session ID
from datetime import datetime
from typing import Optional  # New import

import numpy as np
import pandas as pd
import streamlit as st

import src.book_recommender.core.config as config
from src.book_recommender.core.exceptions import (
    DataNotFoundError,
)
from src.book_recommender.core.logging_config import (
    configure_logging,
)  # Import logging configuration
from src.book_recommender.data.processor import clean_and_prepare_data
from src.book_recommender.ml.clustering import (
    cluster_books,
    get_cluster_names,
)  # New import
from src.book_recommender.ml.embedder import generate_embedding_for_query
from src.book_recommender.ml.explainability import explain_recommendation  # New import
from src.book_recommender.ml.feedback import save_feedback  # New import
from src.book_recommender.ml.recommender import BookRecommender
from src.book_recommender.utils import get_cover_url_multi_source, load_book_covers_batch

# Configure logging at the very beginning of the app
configure_logging(log_file="app.log", log_level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)  # Initialize logger for this module

# --- Page Configuration ---
st.set_page_config(
    page_title="BookFinder - AI Book Recommendations", page_icon="📚", layout="wide", initial_sidebar_state="collapsed"
)

# --- Custom CSS for Modern Design ---
st.markdown(
    """
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }

    /* Content wrapper */
    .stApp {
        background: transparent;
    }

    /* Enforce flexbox for Streamlit columns */
    [data-testid="column"] {
        display: flex !important;
        flex-direction: column !important;
    }

    [data-testid="column"] > div {
        flex: 1;
        display: flex;
        flex-direction: column;
    }

    /* Header section */
    .header-section {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 2rem 3rem;
        border-radius: 0 0 30px 30px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }

    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        padding: 0;
    }

    .subtitle {
        font-size: 1.3rem;
        color: #6b7280;
        margin-top: 0.5rem;
    }

    /* Search section */
    .search-container {
        background: white;
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.15);
        margin: 2rem auto;
        max-width: 900px;
    }

    /* Book Card wrapper and structure */
    .book-card-wrapper {
        height: 100%;
        display: flex;
        flex-direction: column;
    }

    .book-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        display: flex;
        flex-direction: column;
        height: 100%;
        flex: 1;
    }

    .book-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }

    /* Book cover with fixed aspect ratio */
    .book-cover-container {
        position: relative;
        width: 100%;
        padding-top: 150%; /* 2:3 book aspect ratio */
        overflow: hidden;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        flex-shrink: 0;
    }

    .book-cover {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
    }

    /* Content wrapper */
    .book-content {
        display: flex;
        flex-direction: column;
        flex: 1;
    }

    /* Info section that grows */
    .book-info-section {
        flex: 1;
        display: flex;
        flex-direction: column;
    }

    .similarity-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 1rem;
        width: fit-content;
    }

    /* Fixed height elements for alignment */
    .book-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.5rem;
        line-height: 1.4;
        height: 2.8em; /* Exactly 2 lines */
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }

    .book-author {
        font-size: 0.9rem;
        color: #6b7280;
        margin-bottom: 0.8rem;
        height: 1.4em;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .rating-stars {
        color: #fbbf24;
        font-size: 0.95rem;
        margin-bottom: 0.8rem;
        height: 1.4em;
        display: flex;
        align-items: center;
    }

    .genre-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-bottom: 1rem;
        min-height: 2.2rem;
        max-height: 2.2rem;
        overflow: hidden;
        align-content: flex-start;
    }

    .genre-pill {
        display: inline-block;
        background: #f3f4f6;
        color: #4b5563;
        padding: 0.25rem 0.7rem;
        border-radius: 12px;
        font-size: 0.7rem;
        white-space: nowrap;
        height: fit-content;
    }

    /* Actions pushed to bottom */
    .card-actions {
        margin-top: auto;
        padding-top: 1rem;
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.7rem 1.5rem;
        border-radius: 25px;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
    }

    /* Text input */
    .stTextArea > div > div > textarea {
        border-radius: 15px;
        border: 2px solid #e5e7eb;
        padding: 1rem;
        font-size: 1rem;
        transition: border-color 0.3s ease;
    }

    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }

    /* Results section */
    .results-header {
        text-align: center;
        font-size: 2rem;
        font-weight: 600;
        color: white;
        margin: 3rem 0 2rem 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }

    /* Loading states */
    .stSpinner > div {
        border-color: #667eea !important;
    }

    /* Info/Warning boxes */
    .stAlert {
        border-radius: 15px;
        border-left: 4px solid #667eea;
    }

/* Mobile Responsive */
@media (max-width: 768px) {
    .main-title {
        font-size: 2rem;
    }

    .subtitle {
        font-size: 1rem;
    }

    .search-container {
        padding: 1.5rem;
    }

    .book-card {
        padding: 1rem;
    }

    .book-cover-container {
        padding-top: 140%;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


# --- Caching Functions ---
@st.cache_resource(show_spinner=False)
def load_recommender() -> BookRecommender:
    """Load recommender without cluttering UI with status messages."""
    files_exist = (
        os.path.exists(config.PROCESSED_DATA_PATH)
        and os.path.exists(config.EMBEDDINGS_PATH)
        and os.path.exists(config.EMBEDDING_METADATA_PATH)
    )

    model_changed = False
    if files_exist:
        with open(config.EMBEDDING_METADATA_PATH, "r") as f:
            metadata = json.load(f)
        if metadata.get("model_name") != config.EMBEDDING_MODEL:
            model_changed = True

    if files_exist and not model_changed:
        book_data = pd.read_parquet(config.PROCESSED_DATA_PATH)
        embeddings = np.load(config.EMBEDDINGS_PATH)
    else:
        from src.book_recommender.ml.embedder import generate_embeddings

        if not os.path.exists(config.RAW_DATA_PATH):
            raise DataNotFoundError(f"Raw data file not found at: {config.RAW_DATA_PATH}")

        book_data = clean_and_prepare_data(str(config.RAW_DATA_PATH), str(config.PROCESSED_DATA_PATH))
        embeddings = generate_embeddings(book_data, model_name=config.EMBEDDING_MODEL, show_progress_bar=False)
        np.save(config.EMBEDDINGS_PATH, embeddings)

        metadata = {"model_name": config.EMBEDDING_MODEL}
        with open(config.EMBEDDING_METADATA_PATH, "w") as f:
            json.dump(metadata, f)

    recommender = BookRecommender(book_data=book_data, embeddings=embeddings)
    return recommender


@st.cache_resource(show_spinner=False)
def load_cluster_data() -> tuple[np.ndarray, dict, pd.DataFrame]:
    """
    Generates and returns cached book clusters, cluster names, and book data with cluster IDs.
    """
    logger.info("Generating/Loading cluster data for Streamlit app...")
    recommender = load_recommender()  # Use the existing cached recommender
    book_data_df = recommender.book_data.copy()  # Work on a copy to add cluster_id
    embeddings_arr = recommender.embeddings  # Use embeddings from recommender

    # Generate clusters
    clusters_arr, _ = cluster_books(embeddings_arr, n_clusters=config.NUM_CLUSTERS)
    book_data_df["cluster_id"] = clusters_arr

    # Generate cluster names
    names = get_cluster_names(book_data_df, clusters_arr)

    logger.info("Cluster data generated/loaded and cached for Streamlit app.")
    return clusters_arr, names, book_data_df


def render_header():
    """Modern header with gradient background."""
    st.markdown(
        """
        <div class="header-section">
            <h1 class="main-title">📚 BookFinder</h1>
            <p class="subtitle">Discover your next favorite book</p>
        </div>
    """,
        unsafe_allow_html=True,
    )


def render_search_section():
    """Enhanced search interface with examples."""
    st.markdown('<div class="search-container">', unsafe_allow_html=True)

    st.markdown("### What kind of book are you looking for?")

    # Example queries as clickable chips
    col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 1])
    with col2:
        if st.button("🧙 Fantasy", key="example1", width="stretch"):
            st.session_state.query = "A fantasy adventure with magic and dragons"
    with col3:
        if st.button("🔪 Mystery", key="example2", width="stretch"):
            st.session_state.query = "A psychological thriller with unexpected twists"
    with col4:
        if st.button("❤️ Romance", key="example3", width="stretch"):
            st.session_state.query = "A heartwarming romance set in a small town"

    # Main search input
    query = st.text_area(
        "Describe the book you want",
        value=st.session_state.get("query", ""),
        height=100,
        placeholder="Example: A science fiction novel about time travel with complex characters and philosophical themes...",
        label_visibility="collapsed",
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        search_button = st.button("Find My Perfect Books", type="primary", width="stretch")

    st.markdown("</div>", unsafe_allow_html=True)

    return query, search_button


def render_book_card(rec, index, cover_url, query_text: Optional[str] = None):
    """Render a beautiful book card with all details."""

    # Wrapper div
    st.markdown('<div class="book-card-wrapper">', unsafe_allow_html=True)
    st.markdown('<div class="book-card">', unsafe_allow_html=True)

    # Cover (fixed aspect ratio)
    st.markdown(
        f"""
        <div class="book-cover-container">
            <img class="book-cover" src="{cover_url}" alt="{rec['title']}">
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Content wrapper
    st.markdown('<div class="book-content">', unsafe_allow_html=True)

    # Info section (grows to fill space)
    st.markdown('<div class="book-info-section">', unsafe_allow_html=True)

    # Similarity badge
    if "similarity" in rec:
        similarity_percent = rec["similarity"] * 100
        st.markdown(f'<div class="similarity-badge">{similarity_percent:.0f}% Match</div>', unsafe_allow_html=True)

    # Title (clamped to 2 lines)
    st.markdown(f'<div class="book-title">{rec["title"]}</div>', unsafe_allow_html=True)

    # Author (1 line)
    author = rec.get("authors", "Unknown Author")
    st.markdown(f'<div class="book-author">by {author}</div>', unsafe_allow_html=True)

    # Rating (fixed height)
    if rec.get("rating") and rec.get("rating") != "N/A" and pd.notna(rec["rating"]):
        try:
            rating = float(rec["rating"])
            stars = "⭐" * int(rating)
            st.markdown(f"{stars} **{rating:.1f}/5**")
        except Exception:
            # optionally log: st.write("Rating parse error")
            pass
    else:
        st.markdown('<div class="rating-stars">&nbsp;</div>', unsafe_allow_html=True)

    # Genres (max 3, fixed height)
    if rec.get("genres") and isinstance(rec.get("genres"), str):
        genres = [g.strip() for g in rec["genres"].split(",")[:3]]
        genre_html = '<div class="genre-container">'
        for g in genres:
            genre_html += f'<span class="genre-pill">{g.title()}</span>'
        genre_html += "</div>"
        st.markdown(genre_html, unsafe_allow_html=True)
    else:
        st.markdown('<div class="genre-container">&nbsp;</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # Close book-info-section

    # Actions section (pushed to bottom with margin-top: auto)
    st.markdown('<div class="card-actions">', unsafe_allow_html=True)

    # Description expander
    if rec.get("description"):
        with st.expander("📖 Read More"):
            desc = rec["description"][:250] + "..." if len(rec["description"]) > 250 else rec["description"]
            st.write(desc)

    # Explanation expander (only for search results)
    if query_text and "similarity" in rec:  # Only show explanation if it's a search result and has similarity
        with st.expander("💡 Why this recommendation?"):
            from src.book_recommender.ml.explainability import (  # Re-import if not already
                explain_recommendation,
            )

            explanation = explain_recommendation(
                query_text=query_text, recommended_book=rec, similarity_score=rec["similarity"]
            )

            st.info(explanation["summary"])

            # Show matching features
            if explanation.get("matching_features"):
                st.markdown("**Matching features:**")
                for feature in explanation["matching_features"]:
                    st.markdown(f"• {feature}")

    # Action buttons
    if query_text:  # Only show feedback and details button for search results
        col_a, col_b, col_c = st.columns([1, 1, 3])

        with col_a:
            if st.button("👍", key=f"like_{index}", help="Good recommendation", use_container_width=True):
                save_feedback(query_text, rec, "positive", st.session_state.session_id)
                st.toast(f"Thank you for your feedback on '{rec['title']}'!", icon="👍")
        with col_b:
            if st.button("👎", key=f"dislike_{index}", help="Not relevant", use_container_width=True):
                save_feedback(query_text, rec, "negative", st.session_state.session_id)
                st.toast(f"Thank you for your feedback on '{rec['title']}'!", icon="👎")
        with col_c:
            if st.button("View Details", key=f"details_{index}", width="stretch"):
                show_book_details(rec, query_text=query_text)
    else:  # For non-search results (e.g., cluster browsing), only show "View Details"
        if st.button("View Details", key=f"details_{index}", width="stretch"):
            show_book_details(rec, query_text=query_text)

    st.markdown("</div>", unsafe_allow_html=True)  # Close card-actions
    st.markdown("</div>", unsafe_allow_html=True)  # Close book-content
    st.markdown("</div></div>", unsafe_allow_html=True)  # Close card and wrapper


@st.dialog("Book Details")
def show_book_details(book, query_text: Optional[str] = None):
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(get_cover_url_multi_source(book["title"], book.get("authors", "")))

    with col2:
        st.markdown(f"## {book['title']}")
        st.markdown(f"**by {book.get('authors', 'Unknown Author')}**")

        if query_text and "similarity" in book:  # Explanation in details dialog too
            explanation = explain_recommendation(
                query_text=query_text, recommended_book=book, similarity_score=book["similarity"]
            )
            st.markdown(f"**Match Score:** {explanation['match_score']}% (Confidence: {explanation['confidence']})")
            st.write(explanation["summary"])
            st.divider()  # Add divider after explanation

        similarity_percent = book["similarity"] * 100
        st.markdown(f"**{similarity_percent:.0f}% Match**")

        if book.get("rating") and pd.notna(book["rating"]):
            try:
                rating = float(book["rating"])
                stars = "⭐" * int(rating)
                st.markdown(f"{stars} **{rating:.1f}/5**")
            except Exception:
                # optionally log: st.write("Rating parse error")
                pass

        if book.get("genres"):
            st.markdown(f"**Genres:** {book['genres']}")

        st.divider()

        if book.get("description"):
            st.markdown("### Description")
            st.write(book["description"])

        # Action buttons
        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            st.link_button(
                "Search on Google", f"https://www.google.com/search?q={book['title']}+{book.get('authors', '')}"
            )
        with col_b:
            st.link_button("View on Goodreads", f"https://www.goodreads.com/search?q={book['title']}")


def render_sidebar(recommender):
    """Renders the sidebar with filters and search history."""
    with st.sidebar:
        st.markdown("## Recent Searches")
        if not st.session_state.search_history:
            st.info("Your recent searches will appear here.")
        else:
            for item in reversed(st.session_state.search_history):
                if st.button(f"'{item['query'][:20]}...' ({item['results_count']} found)", key=item["timestamp"]):
                    st.session_state.query = item["query"]
                    st.rerun()

        st.markdown("---")
        st.markdown("## Filters")

        if "recommendations" in st.session_state and st.session_state.recommendations:
            st.session_state.min_rating = st.slider(
                "Minimum Rating", 0.0, 5.0, st.session_state.get("min_rating", 0.0), 0.5
            )

            all_genres = set()
            for rec in st.session_state.recommendations:
                if rec.get("genres"):
                    genres = [g.strip() for g in rec["genres"].split(",")]
                    all_genres.update(genres)

            st.session_state.selected_genres = st.multiselect(
                "Genres", sorted(list(all_genres)), default=st.session_state.get("selected_genres", [])
            )
        else:
            st.info("Perform a search to see filters.")


def main():
    """Main application with modern UI."""
    render_header()

    # Initialize session state
    if "query" not in st.session_state:
        st.session_state.query = ""
    if "search_history" not in st.session_state:
        st.session_state.search_history = []
    if "recommendations" not in st.session_state:
        st.session_state.recommendations = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())  # Generate a unique session ID

    try:
        recommender = load_recommender()
        render_sidebar(recommender)

        tab1, tab2 = st.tabs(["Search", "Browse Collections"])

        with tab1:
            # Search section (existing code moved here)
            query, search_button = render_search_section()

            # Process search if button is clicked or if a history item was clicked
            if (search_button and query.strip()) or (query and query != st.session_state.get("last_query", "")):
                st.session_state.last_query = query
                with st.spinner("Finding the perfect books for you..."):
                    query_embedding = generate_embedding_for_query(query)
                    # Store raw recommendations in session state
                    st.session_state.recommendations = recommender.get_recommendations_from_vector(
                        query_embedding,
                        top_k=50,  # Fetch more results to allow for effective filtering
                        similarity_threshold=0.25,
                    )

                if st.session_state.recommendations:
                    # Add to search history
                    st.session_state.search_history.append(
                        {
                            "query": query,
                            "timestamp": datetime.now(),
                            "results_count": len(st.session_state.recommendations),
                        }
                    )
                    st.session_state.search_history = st.session_state.search_history[-5:]
                    # Rerun to apply filters on the new recommendations
                    st.rerun()
                else:
                    st.warning(
                        "No books found matching your description. Try being more specific or use different keywords!"
                    )
                    st.session_state.recommendations = []  # Clear old results

            elif search_button:
                st.warning("Please describe what kind of book you're looking for!")

            # --- Display Results (runs on every interaction) ---
            if st.session_state.recommendations:
                # Apply filters
                filtered = st.session_state.recommendations
                min_rating = st.session_state.get("min_rating", 0.0)
                selected_genres = st.session_state.get("selected_genres", [])

                if min_rating > 0.0:
                    filtered = [
                        rec
                        for rec in filtered
                        if rec.get("rating") and pd.notna(rec["rating"]) and float(rec.get("rating", 0)) >= min_rating
                    ]

                if selected_genres:
                    filtered = [
                        rec
                        for rec in filtered
                        if rec.get("genres") and any(g in rec.get("genres", "") for g in selected_genres)
                    ]

                # Results header
                st.markdown(
                    f'<div class="results-header">✨ Found {len(filtered)} Perfect Books For You</div>',
                    unsafe_allow_html=True,
                )

                if not filtered:
                    st.info("😕 No books match your current filters. Try adjusting them!")
                else:
                    # --- Batch-fetch covers for visible books ---
                    visible_recs = filtered[:12]
                    with st.spinner("Loading book covers..."):
                        covers_dict = load_book_covers_batch(visible_recs)

                    # Display in rows of 4
                    for row_idx in range(0, min(len(filtered), 12), 4):
                        cols = st.columns(4, gap="medium")

                        for col_idx, rec in enumerate(filtered[row_idx : row_idx + 4]):
                            with cols[col_idx]:
                                # Generate unique index for Streamlit keys
                                unique_idx = row_idx + col_idx
                                cover_url = covers_dict.get(rec["title"], config.FALLBACK_COVER_URL)
                                render_book_card(rec, unique_idx, cover_url, query_text=query)

        with tab2:
            st.markdown('<div class="search-container">', unsafe_allow_html=True)
            st.markdown("### Browse Books by Collection")

            # Load cluster data
            clusters_arr, cluster_names, book_data_with_clusters = load_cluster_data()

            # Prepare options for selectbox: cluster name (size)
            cluster_options = [
                f"{name} ({np.sum(clusters_arr == cluster_id)} books)" for cluster_id, name in cluster_names.items()
            ]

            selected_cluster_option = st.selectbox(
                "Select a Collection", options=cluster_options, index=0  # Default to first cluster
            )

            # Extract cluster_id from selected option
            selected_cluster_id = int(list(cluster_names.keys())[cluster_options.index(selected_cluster_option)])

            # Filter books for the selected cluster
            cluster_books_df = book_data_with_clusters[book_data_with_clusters["cluster_id"] == selected_cluster_id]

            st.markdown(
                f'<div class="results-header">📚 Books in {cluster_names[selected_cluster_id]}</div>',
                unsafe_allow_html=True,
            )

            if not cluster_books_df.empty:
                # Convert DataFrame rows to list of dicts for render_book_card
                cluster_recs = cluster_books_df.to_dict(orient="records")

                # --- Batch-fetch covers for visible books ---
                visible_recs = cluster_recs[:12]
                with st.spinner("Loading book covers..."):
                    covers_dict = load_book_covers_batch(visible_recs)

                # Display in rows of 4
                for row_idx in range(0, min(len(cluster_recs), 12), 4):
                    cols = st.columns(4, gap="medium")

                    for col_idx, rec in enumerate(cluster_recs[row_idx : row_idx + 4]):
                        with cols[col_idx]:
                            unique_idx = (
                                f"cluster_{selected_cluster_id}_{row_idx + col_idx}"  # Unique key for cluster books
                            )
                            cover_url = covers_dict.get(rec["title"], config.FALLBACK_COVER_URL)
                            # For clustered books, similarity is not directly applicable, so pass a dummy
                            render_book_card(
                                {**rec, "similarity": 1.0}, unique_idx, cover_url
                            )  # Pass 1.0 similarity for display

            else:
                st.info("No books found in this collection.")
            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Something went wrong: {str(e)}")
        logger.error(f"Application error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
