import json
import logging
import os
import uuid
import pickle
from datetime import datetime
from typing import Optional
from contextlib import contextmanager

import numpy as np
import pandas as pd
import streamlit as st
import sys
import os
from sentence_transformers import SentenceTransformer

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import src.book_recommender.core.config as config
from src.book_recommender.core.exceptions import (
    DataNotFoundError,
)
from src.book_recommender.core.logging_config import (
    configure_logging,
)
from src.book_recommender.data.processor import clean_and_prepare_data
from src.book_recommender.ml.clustering import (
    cluster_books,
    get_cluster_names,
)
from src.book_recommender.ml.embedder import generate_embedding_for_query
from src.book_recommender.ml.explainability import explain_recommendation
from src.book_recommender.ml.feedback import save_feedback
from src.book_recommender.ml.recommender import BookRecommender
from src.book_recommender.utils import get_cover_url_multi_source, load_book_covers_batch

configure_logging(log_file="app.log", log_level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)
st.set_page_config(
    page_title="BookFinder - AI Book Recommendations", page_icon="📚", layout="wide", initial_sidebar_state="collapsed"
)


@st.cache_resource(show_spinner=False)
def load_embedding_model() -> SentenceTransformer:
    """
    Load and cache the sentence-transformer model.
    This ensures the model is loaded only once per session/process.
    """
    logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
    return SentenceTransformer(config.EMBEDDING_MODEL)


@contextmanager
def custom_spinner(text="Loading..."):
    """
    A custom spinner that uses CSS/HTML for a better loading animation.
    """
    placeholder = st.empty()
    placeholder.markdown(
        f"""
        <div style="
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            margin: 2rem auto; /* Centered with consistent spacing */
        ">
            <div class="book-loader-small">
                <div class="inner">
                    <div class="left"></div>
                    <div class="middle"></div>
                    <div class="right"></div>
                </div>
                <ul>
                    <li></li>
                    <li></li>
                    <li></li>
                    <li></li>
                    <li></li>
                    <li></li>
                    <li></li>
                    <li></li>
                    <li></li>
                    <li></li>
                    <li></li>
                    <li></li>
                </ul>
            </div>
            <p style="
                margin-top: 0.8rem;
                font-size: 0.85rem;
                color: #6b7280;
                font-weight: 500;
                font-family: 'Inter', sans-serif;
                letter-spacing: 0.3px;
                opacity: 0.8;
                animation: text-pulse 2s infinite ease-in-out;
            ">{text}</p>
        </div>
        <style>
            .book-loader-small {{
                --color: #667eea;
                --duration: 6.8s;
                width: 24px; /* Reduced from 32px */
                height: 9px; /* Reduced from 12px */
                position: relative;
                margin: 0;
            }}
            .book-loader-small .inner {{
                width: 24px;
                height: 9px;
                position: relative;
                transform-origin: 1.5px 1.5px; /* Adjusted origin */
                transform: rotateZ(-90deg);
                animation: book var(--duration) ease infinite;
            }}
            .book-loader-small .inner .left,
            .book-loader-small .inner .right {{
                width: 45px; /* Reduced from 60px */
                height: 3px; /* Reduced from 4px */
                top: 0;
                border-radius: 1.5px;
                background: var(--color);
                position: absolute;
            }}
            .book-loader-small .inner .left {{
                right: 21px; /* Adjusted */
                transform-origin: 43.5px 1.5px; /* Adjusted */
                transform: rotateZ(90deg);
                animation: left var(--duration) ease infinite;
            }}
            .book-loader-small .inner .right {{
                left: 21px; /* Adjusted */
                transform-origin: 1.5px 1.5px; /* Adjusted */
                transform: rotateZ(-90deg);
                animation: right var(--duration) ease infinite;
            }}
            .book-loader-small .inner .middle {{
                width: 24px;
                height: 9px;
                border: 3px solid var(--color); /* Reduced border */
                border-top: 0;
                border-radius: 0 0 7px 7px;
                transform: translateY(1.5px);
            }}
            .book-loader-small ul {{
                margin: 0;
                padding: 0;
                list-style: none;
                position: absolute;
                left: 50%;
                top: 0;
            }}
            .book-loader-small ul li {{
                height: 3px; /* Reduced from 4px */
                border-radius: 1.5px;
                transform-origin: 100% 1.5px; /* Adjusted */
                width: 36px; /* Reduced from 48px */
                right: 0;
                top: -7.5px; /* Adjusted */
                position: absolute;
                background: var(--color);
                transform: rotateZ(0deg) translateX(-13.5px); /* Adjusted */
                animation-duration: var(--duration);
                animation-timing-function: ease;
                animation-iteration-count: infinite;
            }}
            .book-loader-small ul li:nth-child(1) {{ animation-name: page-1; }}
            .book-loader-small ul li:nth-child(2) {{ animation-name: page-2; }}
            .book-loader-small ul li:nth-child(3) {{ animation-name: page-3; }}
            .book-loader-small ul li:nth-child(4) {{ animation-name: page-4; }}
            .book-loader-small ul li:nth-child(5) {{ animation-name: page-5; }}
            .book-loader-small ul li:nth-child(6) {{ animation-name: page-6; }}
            .book-loader-small ul li:nth-child(7) {{ animation-name: page-7; }}
            .book-loader-small ul li:nth-child(8) {{ animation-name: page-8; }}
            .book-loader-small ul li:nth-child(9) {{ animation-name: page-9; }}
            .book-loader-small ul li:nth-child(10) {{ animation-name: page-10; }}
            .book-loader-small ul li:nth-child(11) {{ animation-name: page-11; }}
            .book-loader-small ul li:nth-child(12) {{ animation-name: page-12; }}
            
            @keyframes page-1 {{ 4% {{ transform: rotateZ(0deg) translateX(-13.5px); }} 13%, 54% {{ transform: rotateZ(180deg) translateX(-13.5px); }} 63% {{ transform: rotateZ(0deg) translateX(-13.5px); }} }}
            @keyframes page-2 {{ 9% {{ transform: rotateZ(0deg) translateX(-13.5px); }} 18%, 59% {{ transform: rotateZ(180deg) translateX(-13.5px); }} 68% {{ transform: rotateZ(0deg) translateX(-13.5px); }} }}
            @keyframes page-3 {{ 14% {{ transform: rotateZ(0deg) translateX(-13.5px); }} 23%, 64% {{ transform: rotateZ(180deg) translateX(-13.5px); }} 73% {{ transform: rotateZ(0deg) translateX(-13.5px); }} }}
            @keyframes page-4 {{ 19% {{ transform: rotateZ(0deg) translateX(-13.5px); }} 28%, 69% {{ transform: rotateZ(180deg) translateX(-13.5px); }} 78% {{ transform: rotateZ(0deg) translateX(-13.5px); }} }}
            @keyframes page-5 {{ 24% {{ transform: rotateZ(0deg) translateX(-13.5px); }} 33%, 74% {{ transform: rotateZ(180deg) translateX(-13.5px); }} 83% {{ transform: rotateZ(0deg) translateX(-13.5px); }} }}
            @keyframes page-6 {{ 29% {{ transform: rotateZ(0deg) translateX(-13.5px); }} 38%, 79% {{ transform: rotateZ(180deg) translateX(-13.5px); }} 88% {{ transform: rotateZ(0deg) translateX(-13.5px); }} }}
            @keyframes page-7 {{ 34% {{ transform: rotateZ(0deg) translateX(-13.5px); }} 43%, 84% {{ transform: rotateZ(180deg) translateX(-13.5px); }} 93% {{ transform: rotateZ(0deg) translateX(-13.5px); }} }}
            @keyframes page-8 {{ 39% {{ transform: rotateZ(0deg) translateX(-13.5px); }} 48%, 89% {{ transform: rotateZ(180deg) translateX(-13.5px); }} 98% {{ transform: rotateZ(0deg) translateX(-13.5px); }} }}
            @keyframes page-9 {{ 44% {{ transform: rotateZ(0deg) translateX(-13.5px); }} 53%, 94% {{ transform: rotateZ(180deg) translateX(-13.5px); }} 100% {{ transform: rotateZ(0deg) translateX(-13.5px); }} }}
            @keyframes page-10 {{ 49% {{ transform: rotateZ(0deg) translateX(-13.5px); }} 58%, 99% {{ transform: rotateZ(180deg) translateX(-13.5px); }} 100% {{ transform: rotateZ(0deg) translateX(-13.5px); }} }}
            @keyframes page-11 {{ 54% {{ transform: rotateZ(0deg) translateX(-13.5px); }} 63% {{ transform: rotateZ(180deg) translateX(-13.5px); }} 100% {{ transform: rotateZ(0deg) translateX(-13.5px); }} }}
            @keyframes page-12 {{ 59% {{ transform: rotateZ(0deg) translateX(-13.5px); }} 68% {{ transform: rotateZ(180deg) translateX(-13.5px); }} 100% {{ transform: rotateZ(0deg) translateX(-13.5px); }} }}
            
            @keyframes left {{ 4% {{ transform: rotateZ(90deg); }} 10%, 40% {{ transform: rotateZ(0deg); }} 46%, 54% {{ transform: rotateZ(90deg); }} 60%, 90% {{ transform: rotateZ(0deg); }} 96% {{ transform: rotateZ(90deg); }} }}
            @keyframes right {{ 4% {{ transform: rotateZ(-90deg); }} 10%, 40% {{ transform: rotateZ(0deg); }} 46%, 54% {{ transform: rotateZ(-90deg); }} 60%, 90% {{ transform: rotateZ(0deg); }} 96% {{ transform: rotateZ(-90deg); }} }}
            @keyframes book {{ 4% {{ transform: rotateZ(-90deg); }} 10%, 40% {{ transform: rotateZ(0deg); transform-origin: 1.5px 1.5px; }} 40.01%, 59.99% {{ transform-origin: 22.5px 1.5px; }} 46%, 54% {{ transform: rotateZ(90deg); }} 60%, 90% {{ transform: rotateZ(0deg); transform-origin: 1.5px 1.5px; }} 96% {{ transform: rotateZ(-90deg); }} }}
            @keyframes text-pulse {{ 0%, 100% {{ opacity: 0.5; }} 50% {{ opacity: 1; }} }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    try:
        yield
    finally:
        placeholder.empty()


st.markdown(
    """
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }

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
    .book-card-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        display: flex;
        flex-direction: column;
        height: 100%;
        position: relative;
        overflow: hidden;
    }

    .book-card-container:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }

    /* Book cover with fixed aspect ratio */
    .book-cover-wrapper {
        position: relative;
        width: 100%;
        padding-top: 150%; /* 2:3 aspect ratio standard for books */
        overflow: hidden;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        background-color: #f3f4f6; /* Placeholder background */
    }

    .book-cover-wrapper img {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        object-fit: cover; /* Ensures image covers the area without distortion */
        object-position: center;
        transition: transform 0.3s ease;
    }
    
    .book-card-container:hover .book-cover-wrapper img {
        transform: scale(1.05);
    }

    /* Info section */
    .book-info {
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
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
        width: fit-content;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }

    /* Typography */
    .book-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.4rem;
        line-height: 1.3;
        /* Clamp to 2 lines */
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        height: 2.8em; 
    }

    .book-author {
        font-size: 0.9rem;
        color: #6b7280;
        margin-bottom: 0.8rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .rating-stars {
        color: #fbbf24;
        font-size: 0.9rem;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .rating-stars strong {
        color: #4b5563;
        font-weight: 600;
    }

    .genre-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-bottom: 1rem;
        height: 2.2rem; /* Fixed height for consistency */
        overflow: hidden;
    }

    .genre-pill {
        background: #f3f4f6;
        color: #4b5563;
        padding: 0.25rem 0.7rem;
        border-radius: 12px;
        font-size: 0.7rem;
        white-space: nowrap;
    }

    /* Actions area */
    .card-actions {
        margin-top: auto;
        padding-top: 0.5rem;
        border-top: 1px solid #f3f4f6;
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
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
    Tries to load from disk first, then falls back to computation and saves to disk.
    """
    logger.info("Generating/Loading cluster data for Streamlit app...")
    recommender = load_recommender()
    book_data_df = recommender.book_data.copy()
    
    # Try loading from disk
    if os.path.exists(config.CLUSTERS_CACHE_PATH):
        try:
            logger.info(f"Loading cached clusters from {config.CLUSTERS_CACHE_PATH}")
            with open(config.CLUSTERS_CACHE_PATH, "rb") as f:
                cached_data = pickle.load(f)
            
            clusters_arr = cached_data["clusters_arr"]
            names = cached_data["names"]
            
            # Verify consistency
            if len(clusters_arr) == len(book_data_df):
                book_data_df["cluster_id"] = clusters_arr
                logger.info("Cluster data loaded from disk cache.")
                return clusters_arr, names, book_data_df
            else:
                logger.warning("Cached cluster data size mismatch. Recomputing...")
        except Exception as e:
            logger.error(f"Failed to load cluster cache: {e}. Recomputing...")

    # Recompute if not found or failed
    embeddings_arr = recommender.embeddings
    clusters_arr, _ = cluster_books(embeddings_arr, n_clusters=config.NUM_CLUSTERS)
    book_data_df["cluster_id"] = clusters_arr

    names = get_cluster_names(book_data_df, clusters_arr)

    # Save to disk
    try:
        with open(config.CLUSTERS_CACHE_PATH, "wb") as f:
            pickle.dump({"clusters_arr": clusters_arr, "names": names}, f)
        logger.info(f"Cluster data saved to {config.CLUSTERS_CACHE_PATH}")
    except Exception as e:
        logger.error(f"Failed to save cluster cache: {e}")

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

    col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 1])
    with col2:
        if st.button("Fantasy", key="example1", width="stretch"):
            st.session_state.query = "A fantasy adventure with magic and dragons"
    with col3:
        if st.button("Mystery", key="example2", width="stretch"):
            st.session_state.query = "A psychological thriller with unexpected twists"
    with col4:
        if st.button("Romance", key="example3", width="stretch"):
            st.session_state.query = "A heartwarming romance set in a small town"

    # Main search input
    query = st.text_area(
        "Describe the book you want",
        value=st.session_state.get("query", ""),
        height=100,
        placeholder=(
            "Example: A science fiction novel about time travel with " "complex characters and philosophical themes..."
        ),
        label_visibility="collapsed",
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        search_button = st.button("Find My Perfect Books", type="primary", width="stretch")

    st.markdown("</div>", unsafe_allow_html=True)

    return query, search_button


def render_book_card(rec, index, cover_url, query_text: Optional[str] = None):
    """Render a beautiful book card with all details."""
    
    # Generate stars string
    rating_html = '<div class="rating-stars">&nbsp;</div>'
    if rec.get("rating") and rec.get("rating") != "N/A" and pd.notna(rec["rating"]):
        try:
            rating = float(rec["rating"])
            stars = "⭐" * int(rating)
            rating_html = f'<div class="rating-stars">{stars} <strong>{rating:.1f}/5</strong></div>'
        except Exception:
            pass

    # Generate genres HTML
    genres_html = '<div class="genre-container">&nbsp;</div>'
    if rec.get("genres") and isinstance(rec.get("genres"), str):
        genres = [g.strip() for g in rec["genres"].split(",")[:3]]
        pills = "".join([f'<span class="genre-pill">{g.title()}</span>' for g in genres])
        genres_html = f'<div class="genre-container">{pills}</div>'

    # Generate match badge
    match_html = ""
    if "similarity" in rec:
        similarity_percent = rec["similarity"] * 100
        match_html = f'<div class="similarity-badge">{similarity_percent:.0f}% Match</div>'

    # Single HTML block for the card content
    card_html = f"""
    <div class="book-card-container">
        <div class="book-cover-wrapper">
            <img src="{cover_url}" alt="{rec['title']}">
        </div>
        <div class="book-info">
            {match_html}
            <div class="book-title" title="{rec['title']}">{rec['title']}</div>
            <div class="book-author">by {rec.get("authors", "Unknown Author")}</div>
            {rating_html}
            {genres_html}
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

    # Interactive elements (outside the HTML block)
    with st.container():
        if rec.get("description"):
            with st.expander("Read More"):
                desc = rec["description"][:250] + "..." if len(rec["description"]) > 250 else rec["description"]
                st.write(desc)

        if query_text and "similarity" in rec:
            with st.expander("Why this recommendation?"):
                from src.book_recommender.ml.explainability import (
                    explain_recommendation,
                )
                explanation = explain_recommendation(
                    query_text=query_text, recommended_book=rec, similarity_score=rec["similarity"]
                )
                st.info(explanation["summary"])
                if explanation.get("matching_features"):
                    st.markdown("**Matching features:**")
                    for feature in explanation["matching_features"]:
                        st.markdown(f"• {feature}")

        st.markdown('<div class="card-actions">', unsafe_allow_html=True)
        if query_text:
            col_a, col_b, col_c = st.columns([1, 1, 3])
            with col_a:
                if st.button("👍", key=f"like_{index}", help="Good recommendation", use_container_width=True):
                    save_feedback(query_text, rec, "positive", st.session_state.session_id)
                    st.toast(f"Feedback saved!", icon="👍")
            with col_b:
                if st.button("👎", key=f"dislike_{index}", help="Not relevant", use_container_width=True):
                    save_feedback(query_text, rec, "negative", st.session_state.session_id)
                    st.toast(f"Feedback saved!", icon="👎")
            with col_c:
                if st.button("View Details", key=f"details_{index}", use_container_width=True):
                    show_book_details(rec, query_text=query_text)
        else:
            if st.button("View Details", key=f"details_{index}", use_container_width=True):
                show_book_details(rec, query_text=query_text)
        st.markdown('</div>', unsafe_allow_html=True)


@st.dialog("Book Details")
def show_book_details(book, query_text: Optional[str] = None):
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(get_cover_url_multi_source(book["title"], book.get("authors", "")))

    with col2:
        st.markdown(f"## {book['title']}")
        st.markdown(f"**by {book.get('authors', 'Unknown Author')}**")

        if query_text and "similarity" in book:
            explanation = explain_recommendation(
                query_text=query_text, recommended_book=book, similarity_score=book["similarity"]
            )
            st.markdown(f"**Match Score:** {explanation['match_score']}% (Confidence: {explanation['confidence']})")
            st.write(explanation["summary"])
            st.divider()

        similarity_percent = book["similarity"] * 100
        st.markdown(f"**{similarity_percent:.0f}% Match**")

        if book.get("rating") and pd.notna(book["rating"]):
            try:
                rating = float(book["rating"])
                stars = "⭐" * int(rating)
                st.markdown(f"{stars} **{rating:.1f}/5**")
            except Exception:
                pass

        if book.get("genres"):
            st.markdown(f"**Genres:** {book['genres']}")

        st.divider()

        if book.get("description"):
            st.markdown("### Description")
            st.write(book["description"])

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
    """Main application with UI."""
    render_header()

    if "query" not in st.session_state:
        st.session_state.query = ""
    if "search_history" not in st.session_state:
        st.session_state.search_history = []
    if "recommendations" not in st.session_state:
        st.session_state.recommendations = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    try:
        recommender = load_recommender()
        render_sidebar(recommender)

        tab1, tab2 = st.tabs(["Search", "Browse Collections"])

        with tab1:
            query, search_button = render_search_section()

            if (search_button and query.strip()) or (query and query != st.session_state.get("last_query", "")):
                st.session_state.last_query = query
                with custom_spinner("Finding the perfect books for you..."):
                    # Use the cached model instance
                    embedding_model = load_embedding_model()
                    query_embedding = generate_embedding_for_query(query, model=embedding_model)
                    
                    st.session_state.recommendations = recommender.get_recommendations_from_vector(
                        query_embedding,
                        top_k=10,
                        similarity_threshold=0.25,
                    )

                if st.session_state.recommendations:
                    st.session_state.search_history.append(
                        {
                            "query": query,
                            "timestamp": datetime.now(),
                            "results_count": len(st.session_state.recommendations),
                        }
                    )
                    st.session_state.search_history = st.session_state.search_history[-5:]
                    st.rerun()
                else:
                    st.warning(
                        "No books found matching your description. Try being more specific or use different keywords!"
                    )
                    st.session_state.recommendations = []

            elif search_button:
                st.warning("Please describe what kind of book you're looking for!")

            if st.session_state.recommendations:
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

                st.markdown(
                    f'<div class="results-header">Found {len(filtered)} Perfect Books For You</div>',
                    unsafe_allow_html=True,
                )

                if not filtered:
                    st.info("No books match your current filters. Try adjusting them!")
                else:
                    visible_recs = filtered[:12]
                    with custom_spinner("Loading book covers..."):
                        covers_dict = load_book_covers_batch(visible_recs)

                    for row_idx in range(0, min(len(filtered), 12), 4):
                        cols = st.columns(4, gap="medium")

                        for col_idx, rec in enumerate(filtered[row_idx : row_idx + 4]):
                            with cols[col_idx]:
                                unique_idx = row_idx + col_idx
                                cover_url = covers_dict.get(rec["title"], config.FALLBACK_COVER_URL)
                                render_book_card(rec, unique_idx, cover_url, query_text=query)

        with tab2:
            st.markdown('<div class="search-container">', unsafe_allow_html=True)
            st.markdown("### Browse Books by Collection")

            clusters_arr, cluster_names, book_data_with_clusters = load_cluster_data()

            cluster_options = [
                f"{name} ({np.sum(clusters_arr == cluster_id)} books)" for cluster_id, name in cluster_names.items()
            ]

            selected_cluster_option = st.selectbox("Select a Collection", options=cluster_options, index=0)

            selected_cluster_id = int(list(cluster_names.keys())[cluster_options.index(selected_cluster_option)])

            cluster_books_df = book_data_with_clusters[book_data_with_clusters["cluster_id"] == selected_cluster_id]

            st.markdown(
                f'<div class="results-header">Books in {cluster_names[selected_cluster_id]}</div>',
                unsafe_allow_html=True,
            )

            if not cluster_books_df.empty:
                cluster_recs = cluster_books_df.to_dict(orient="records")

                visible_recs = cluster_recs[:12]
                with custom_spinner("Loading book covers..."):
                    covers_dict = load_book_covers_batch(visible_recs)

                for row_idx in range(0, min(len(cluster_recs), 12), 4):
                    cols = st.columns(4, gap="medium")

                    for col_idx, rec in enumerate(cluster_recs[row_idx : row_idx + 4]):
                        with cols[col_idx]:
                            unique_idx = f"cluster_{selected_cluster_id}_{row_idx + col_idx}"
                            cover_url = covers_dict.get(rec["title"], config.FALLBACK_COVER_URL)
                            render_book_card({**rec, "similarity": 1.0}, unique_idx, cover_url)

            else:
                st.info("No books found in this collection.")
            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Something went wrong: {str(e)}")
        logger.error(f"Application error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
