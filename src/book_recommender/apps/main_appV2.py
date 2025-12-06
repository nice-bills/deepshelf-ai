import json
import logging
import os
import uuid
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import sys

# --- PATH SETUP ---
# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# --- MODULE IMPORTS ---
# (Assumed to exist based on your previous code)
import src.book_recommender.core.config as config
from src.book_recommender.core.exceptions import DataNotFoundError
from src.book_recommender.core.logging_config import configure_logging
from src.book_recommender.data.processor import clean_and_prepare_data
from src.book_recommender.ml.clustering import cluster_books, get_cluster_names
from src.book_recommender.ml.embedder import generate_embedding_for_query
from src.book_recommender.ml.explainability import explain_recommendation
from src.book_recommender.ml.feedback import save_feedback
from src.book_recommender.ml.recommender import BookRecommender
from src.book_recommender.utils import get_cover_url_multi_source, load_book_covers_batch

# --- CONFIGURATION ---
configure_logging(log_file="app.log", log_level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="DeepShelf Demo",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- MODERN CSS (V2) ---
st.markdown(
    """
    <style>
    /* VARIABLES & FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary: #4F46E5;
        --primary-hover: #4338CA;
        --bg-color: #F9FAFB;
        --card-bg: #FFFFFF;
        --text-main: #111827;
        --text-sub: #6B7280;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        --radius: 12px;
    }

    /* GLOBAL RESET */
    .stApp {
        background-color: var(--bg-color);
        font-family: 'Inter', sans-serif;
    }
    
    /* HIDE STREAMLIT CHROME */
    #MainMenu, header, footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* STICKY SEARCH BAR (Mobile Friendly) */
    .sticky-header {
        position: sticky;
        top: 0;
        z-index: 999;
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 1rem 0;
        margin-bottom: 2rem;
        border-bottom: 1px solid rgba(0,0,0,0.05);
    }

    /* TYPOGRAPHY */
    h1, h2, h3 { color: var(--text-main); font-weight: 700 !important; letter-spacing: -0.025em; }
    p { color: var(--text-sub); line-height: 1.6; }

    /* CARD SYSTEM */
    .book-card-container {
        background: var(--card-bg);
        border-radius: var(--radius);
        padding: 1rem;
        box-shadow: var(--shadow-sm);
        transition: all 0.2s ease;
        border: 1px solid #E5E7EB;
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    
    .book-card-container:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-lg);
        border-color: var(--primary);
    }

    /* IMAGES */
    .cover-wrapper {
        position: relative;
        width: 100%;
        padding-top: 150%; /* Aspect Ratio 2:3 */
        border-radius: 8px;
        overflow: hidden;
        margin-bottom: 1rem;
        background: #f3f4f6;
    }
    
    .cover-img {
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 100%;
        object-fit: cover;
    }

    /* TEXT CLAMPING (Crucial for alignment) */
    .clamp-title {
        font-weight: 600;
        font-size: 1rem;
        color: var(--text-main);
        margin-bottom: 0.25rem;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        height: 2.5em; /* Enforce height */
    }
    
    .clamp-author {
        font-size: 0.875rem;
        color: var(--text-sub);
        margin-bottom: 0.5rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* BADGES */
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.125rem 0.625rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .badge-match { background: #EEF2FF; color: var(--primary); }
    .badge-genre { background: #F3F4F6; color: #4B5563; margin-right: 4px; margin-bottom: 4px; }

    /* STREAMLIT ELEMENT OVERRIDES */
    .stButton button {
        border-radius: 8px;
        font-weight: 500;
        width: 100%;
        border: 1px solid transparent;
        transition: all 0.2s;
    }
    
    /* Primary Action Buttons */
    div[data-testid="stVerticalBlock"] .stButton button {
        background-color: var(--primary);
        color: white;
    }
    div[data-testid="stVerticalBlock"] .stButton button:hover {
        background-color: var(--primary-hover);
    }
    
    /* Secondary Action Buttons (Like/Dislike) */
    .small-btn button {
        background-color: #F3F4F6 !important;
        color: #4B5563 !important;
        padding: 0.25rem 0.5rem;
    }

    /* SEARCH INPUT */
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #E5E7EB;
        box-shadow: none;
        padding: 1rem;
        font-size: 1rem;
    }
    .stTextArea textarea:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
    }

    /* RESPONSIVE TWEAKS */
    @media (max-width: 640px) {
        .stColumns { gap: 1rem !important; }
        .clamp-title { font-size: 0.95rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- BACKEND FUNCTIONS (Cached) ---

@st.cache_resource(show_spinner=False)
def load_recommender() -> BookRecommender:
    """Load recommender efficiently."""
    files_exist = (
        os.path.exists(config.PROCESSED_DATA_PATH)
        and os.path.exists(config.EMBEDDINGS_PATH)
        and os.path.exists(config.EMBEDDING_METADATA_PATH)
    )

    model_changed = False
    if files_exist:
        with open(config.EMBEDDING_METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        if metadata.get("model_name") != config.EMBEDDING_MODEL:
            model_changed = True

    if files_exist and not model_changed:
        book_data = pd.read_parquet(config.PROCESSED_DATA_PATH)
        embeddings = np.load(config.EMBEDDINGS_PATH)
    else:
        # Fallback to generation if files missing (usually dev env)
        from src.book_recommender.ml.embedder import generate_embeddings
        if not os.path.exists(config.RAW_DATA_PATH):
            raise DataNotFoundError(f"Raw data file not found at: {config.RAW_DATA_PATH}")

        book_data = clean_and_prepare_data(str(config.RAW_DATA_PATH), str(config.PROCESSED_DATA_PATH))
        embeddings = generate_embeddings(book_data, model_name=config.EMBEDDING_MODEL, show_progress_bar=False)
        np.save(config.EMBEDDINGS_PATH, embeddings)

        metadata = {"model_name": config.EMBEDDING_MODEL}
        with open(config.EMBEDDING_METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f)

    return BookRecommender(book_data=book_data, embeddings=embeddings)

@st.cache_resource(show_spinner=False)
def load_cluster_data() -> tuple[np.ndarray, dict, pd.DataFrame]:
    """Load cluster data for the 'Browse' tab."""
    recommender = load_recommender()
    book_data_df = recommender.book_data.copy()
    embeddings_arr = recommender.embeddings
    clusters_arr, _ = cluster_books(embeddings_arr, n_clusters=config.NUM_CLUSTERS)
    book_data_df["cluster_id"] = clusters_arr
    names = get_cluster_names(book_data_df, clusters_arr)
    return clusters_arr, names, book_data_df

# --- UI COMPONENT FUNCTIONS ---

def render_hero():
    """Renders the top Hero section."""
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0 1rem 0;">
            <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem;">📚 DeepShelf Demo</h1>
            <p style="font-size: 1.1rem; color: #6B7280;">Describe what you're craving. We'll handle the rest.</p>
        </div>
    """, unsafe_allow_html=True)

def render_book_card_content(rec, cover_url):
    """
    Renders the HTML content of the card (Image + Text).
    Buttons are handled outside by Streamlit to maintain functionality.
    """
    
    # Process Genres (Limit to 2 for space)
    genres_html = ""
    if rec.get("genres") and isinstance(rec.get("genres"), str):
        genres = [g.strip() for g in rec["genres"].split(",")[:2]]
        for g in genres:
            genres_html += f'<span class="badge badge-genre">{g.title()}</span>'
            
    # Process Similarity
    match_badge = ""
    if "similarity" in rec:
        pct = int(rec["similarity"] * 100)
        match_badge = f'<div style="margin-bottom:0.5rem;"><span class="badge badge-match">{pct}% Match</span></div>'

    # Rating
    rating_display = ""
    if rec.get("rating") and pd.notna(rec["rating"]):
        try:
            r = float(rec["rating"])
            rating_display = f'<span style="color:#F59E0B; font-size:0.875rem;">★ {r:.1f}</span>'
        except: pass

    # HTML Structure
    html = f"""
        <div class="book-card-container">
            <div class="cover-wrapper">
                <img class="cover-img" src="{cover_url}" loading="lazy">
            </div>
            {match_badge}
            <div class="clamp-title" title="{rec['title']}">{rec['title']}</div>
            <div class="clamp-author">by {rec.get('authors', 'Unknown')}</div>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
                <div>{rating_display}</div>
            </div>
            <div style="margin-bottom:auto;">
                {genres_html}
            </div>
        </div>
    """
    st.markdown(html, unsafe_allow_html=True)

@st.dialog("📖 Book Details")
def show_book_details(book, query_text: Optional[str] = None):
    """Modal for book details."""
    
    # Header
    st.subheader(book['title'])
    st.caption(f"by {book.get('authors', 'Unknown Author')}")
    
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        st.image(get_cover_url_multi_source(book["title"], book.get("authors", "")), use_container_width=True)
        
        # Metadata pills
        if book.get("rating"):
            st.metric("Rating", f"{float(book['rating']):.1f}/5")
            
    with col2:
        # AI Explanation
        if query_text and "similarity" in book:
            with st.spinner("Analyzing match..."):
                explanation = explain_recommendation(
                    query_text=query_text, 
                    recommended_book=book, 
                    similarity_score=book["similarity"]
                )
            st.markdown(f"**Why this matches:**")
            st.info(explanation["summary"])
            
        # Description
        st.markdown("**Synopsis:**")
        st.write(book.get("description", "No description available."))
        
        st.divider()
        
        # External Links
        c1, c2 = st.columns(2)
        c1.link_button("Google Search", f"https://www.google.com/search?q={book['title']}+{book.get('authors', '')}", use_container_width=True)
        c2.link_button("Goodreads", f"https://www.goodreads.com/search?q={book['title']}", use_container_width=True)


# --- MAIN APP LOGIC ---

def main():
    # Session State Init
    if "query" not in st.session_state: st.session_state.query = ""
    if "recommendations" not in st.session_state: st.session_state.recommendations = []
    if "session_id" not in st.session_state: st.session_state.session_id = str(uuid.uuid4())

    try:
        recommender = load_recommender()
        
        # --- STICKY HERO & SEARCH ---
        # We put this in a container to act as our sticky header
        with st.container():
            render_hero()
            
            # Search Input Area
            c1, c2, c3 = st.columns([1, 6, 1])
            with c2:
                # Suggestion Chips
                s1, s2, s3 = st.columns(3)
                if s1.button("🧙‍♂️ Fantasy Dragons", use_container_width=True): st.session_state.query = "Epic fantasy with dragons and magic"
                if s2.button("🕵️‍♀️ Murder Mystery", use_container_width=True): st.session_state.query = "Whodunnit mystery thriller"
                if s3.button("🚀 Sci-Fi Space", use_container_width=True): st.session_state.query = "Hard science fiction in space"
                
                query_input = st.text_area(
                    "Search", 
                    value=st.session_state.query,
                    placeholder="Describe the vibe: 'A melancholic book about growing up in the 90s...'", 
                    height=80, 
                    label_visibility="collapsed"
                )
                
                search_clicked = st.button("✨ Find Recommendations", type="primary", use_container_width=True)

        # --- LOGIC HANDLER ---
        if search_clicked and query_input:
            st.session_state.query = query_input # Sync state
            with st.spinner("Reading thousands of books..."):
                query_embedding = generate_embedding_for_query(query_input)
                st.session_state.recommendations = recommender.get_recommendations_from_vector(
                    query_embedding, top_k=12, similarity_threshold=0.20
                )
        
        st.divider()

        # --- RESULTS GRID ---
        if st.session_state.recommendations:
            st.markdown(f"### Found {len(st.session_state.recommendations)} Matches")
            
            # Fetch covers in bulk
            visible_recs = st.session_state.recommendations
            with st.spinner("Fetching covers..."):
                covers_dict = load_book_covers_batch(visible_recs)

            # Responsive Grid Logic
            # We iterate through the results and create rows of 4
            num_cols = 4
            rows = [visible_recs[i:i + num_cols] for i in range(0, len(visible_recs), num_cols)]

            for row in rows:
                cols = st.columns(num_cols)
                for idx, (col, rec) in enumerate(zip(cols, row)):
                    with col:
                        # 1. Render the HTML visual card
                        cover = covers_dict.get(rec["title"], config.FALLBACK_COVER_URL)
                        render_book_card_content(rec, cover)
                        
                        # 2. Render the Streamlit interactive buttons BELOW the HTML card
                        # Because of the fixed height in CSS, these will align nicely
                        b1, b2 = st.columns([1, 3])
                        with b1:
                            # Like button (icon only)
                            st.button("👍", key=f"like_{rec['title']}", help="More like this")
                        with b2:
                            if st.button("View Details", key=f"det_{rec['title']}", use_container_width=True):
                                show_book_details(rec, st.session_state.query)
                        
                        st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)

        elif search_clicked:
            st.warning("No books found matching that description. Try being broader!")

    except Exception as e:
        st.error(f"App Error: {str(e)}")
        logger.error(f"Crash: {e}", exc_info=True)

if __name__ == "__main__":
    main()