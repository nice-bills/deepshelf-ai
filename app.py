# app.py

import streamlit as st
import os
import logging
import json
from src.data_processor import clean_and_prepare_data
from src.exceptions import DataNotFoundError, FileProcessingError, ModelLoadError
from src.recommender import BookRecommender
from src.utils import get_cover_url
from src.embedder import generate_embedding_for_query
import src.config as config
import pandas as pd
import numpy as np

# --- Logging Configuration ---
# Configure logging once at the application's entry point
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler() # Log to console
        # To log to a file, you could add:
        # logging.FileHandler("app.log")
    ]
)

# --- Page Configuration ---
st.set_page_config(
    page_title="Open-Source Book Recommender",
    page_icon="📘",
    layout="wide",
)

# --- Caching Functions ---
@st.cache_resource
def load_recommender() -> BookRecommender:
    """
    Orchestrates the loading of the recommender system, running the
    data pipeline if necessary.

    Returns:
        BookRecommender: An initialized instance of the recommender.
    """
    files_exist = (
        os.path.exists(config.PROCESSED_DATA_PATH) and
        os.path.exists(config.EMBEDDINGS_PATH) and
        os.path.exists(config.EMBEDDING_METADATA_PATH)
    )
    
    model_changed = False
    if files_exist:
        with open(config.EMBEDDING_METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        if metadata.get("model_name") != config.EMBEDDING_MODEL:
            model_changed = True
            st.warning(f"Model changed from '{metadata.get('model_name')}' to '{config.EMBEDDING_MODEL}'. Re-generating embeddings.")

    if files_exist and not model_changed:
        st.info("Loading pre-processed data...")
        with st.spinner("Loading data and embeddings..."):
            book_data = pd.read_parquet(config.PROCESSED_DATA_PATH)
            embeddings = np.load(config.EMBEDDINGS_PATH)
        st.info("Data loaded successfully.")
    else:
        st.info("Processed data not found or model updated. Running first-time setup...")
        
        from src.embedder import generate_embeddings
        
        if not os.path.exists(config.RAW_DATA_PATH):
            raise DataNotFoundError(f"Raw data file not found at: {config.RAW_DATA_PATH}")

        with st.spinner("Step 1/2: Cleaning and processing data..."):
            book_data = clean_and_prepare_data(config.RAW_DATA_PATH, config.PROCESSED_DATA_PATH)
            
        with st.spinner("Step 2/2: Generating book embeddings... (This can be slow on first run)"):
            embeddings = generate_embeddings(book_data, model_name=config.EMBEDDING_MODEL)
            np.save(config.EMBEDDINGS_PATH, embeddings)
            
            # Save metadata
            metadata = {"model_name": config.EMBEDDING_MODEL}
            with open(config.EMBEDDING_METADATA_PATH, 'w') as f:
                json.dump(metadata, f)
        
        st.success("First-time setup complete!")

    # Initialize the recommender with the in-memory data
    recommender = BookRecommender(book_data=book_data, embeddings=embeddings)
    return recommender

def render_header():
    """Renders the main header and title of the Streamlit app."""
    # Custom CSS for modern card-like layout
    st.markdown("""
        <style>
            /* Targets the container of each column */
            div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
                border: 1px solid #333;
                border-radius: 10px;
                padding: 15px;
                margin: 5px;
                box-shadow: 2px 2px 8px rgba(0,0,0,0.2);
                transition: transform 0.2s;
            }
            div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:hover {
                transform: scale(1.02);
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("📘 Open-Source Book Recommender")
    st.markdown(
        "Describe a book you'd like to read, and we'll find similar ones for you."
    )

def render_recommendation_interface(recommender: BookRecommender):
    """
    Renders the main user interface for getting book recommendations.
    
    Args:
        recommender (BookRecommender): An initialized BookRecommender instance.
    """
    st.subheader("Describe a book you'd like to read")
    query = st.text_area(
        "Describe a book you'd like to read", 
        height=100,
        placeholder="e.g., A story about a young wizard attending a magical school, or a thriller about a detective in space.",
        label_visibility="collapsed"
    )

    if st.button("Find Similar Books", type="primary", use_container_width=True):
        if query:
            with st.spinner("Generating embedding for your query and finding recommendations..."):
                query_embedding = generate_embedding_for_query(query)
                recommendations = recommender.get_recommendations_from_vector(
                    query_embedding, 
                    top_k=9  # Fetching 9 recommendations
                )

            st.subheader("Here are some books you might like:")
            st.markdown("---")

            if recommendations:
                # Create 3 columns for the layout
                cols = st.columns(3)
                for i, rec in enumerate(recommendations):
                    with cols[i % 3]:
                        # Fetch cover URL
                        cover_url = get_cover_url(rec['title'], rec.get('authors', ''))
                        st.image(cover_url, caption=f"Similarity: {rec['similarity']:.1%}", use_container_width=True)
                        
                        st.markdown(f"**{rec['title']}**")
                        st.markdown(f"*{rec.get('authors', 'N/A')}*")
                        
                        with st.expander("Details"):
                            if rec.get('genres'):
                                st.markdown(f"**Genres:** {rec['genres'].title()}")
                            if rec.get('rating') and rec.get('rating') != 'N/A':
                                st.markdown(f"**Rating:** {rec['rating']} ⭐")
                            if rec.get('description'):
                                st.caption(f"{rec['description'].capitalize()}")
            else:
                st.info("Couldn't find any books with a high enough similarity score. Try being more descriptive.")
        else:
            st.warning("Please enter a description.")

def render_footer():
    """Renders the footer section of the app."""
    st.markdown("---")
    st.markdown(f"*v{config.APP_VERSION} | Powered by Sentence-Transformers and Streamlit.*")

def main():
    """Main function to run the Streamlit application."""
    render_header()
    try:
        recommender = load_recommender()
        render_recommendation_interface(recommender)
    except DataNotFoundError:
        st.error(f"Error: The raw data file was not found at '{config.RAW_DATA_PATH}'. Please make sure it exists.")
        logging.error("DataNotFoundError caught in Streamlit app.")
    except FileProcessingError as e:
        st.error(f"Error: Failed to process the data file. It may be corrupted. Details: {e}")
        logging.error(f"FileProcessingError caught in Streamlit app: {e}", exc_info=True)
    except ModelLoadError as e:
        st.error(f"Error: Failed to load the machine learning model. Please check your internet connection. Details: {e}")
        logging.error(f"ModelLoadError caught in Streamlit app: {e}", exc_info=True)
    except Exception as e:
        st.error("An unexpected error occurred. Please check the logs for more details.")
        logging.error(f"An unexpected error occurred in Streamlit app: {e}", exc_info=True)
    render_footer()

if __name__ == "__main__":
    main()
