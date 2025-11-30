import logging
import os
import sys
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def prepare_100k_data():
    """
    Convert GoodReads_100k_books.csv format to the format expected by data_processor.
    
    Input Columns: author, bookformat, desc, genre, img, isbn, isbn13, link, pages, rating, reviews, title, totalratings
    Target Columns: title, authors, genres, description, tags, rating, cover_image_url
    """
    input_path = "data/raw/GoodReads_100k_books.csv"
    output_path = "data/raw/books_prepared.csv"

    logger.info(f"Loading new 100k dataset from {input_path}...")

    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return

    try:
        # Load data
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} books.")
        
        # Rename columns
        logger.info("Mapping columns...")
        df = df.rename(columns={
            "author": "authors",
            "desc": "description",
            "genre": "genres",
            "img": "cover_image_url",
            "rating": "rating"
        })
        
        # Create tags column (using genres as base if available, else empty)
        df["tags"] = df["genres"].fillna("")
        
        # Select and Reorder
        target_cols = ["title", "authors", "genres", "description", "tags", "rating", "cover_image_url"]
        
        # Ensure all target columns exist
        for col in target_cols:
            if col not in df.columns:
                df[col] = ""
                logger.warning(f"Column {col} missing in source, filled with empty strings.")

        df = df[target_cols]
        
        # Clean up
        logger.info("Cleaning data...")
        # Remove rows with no title
        df = df.dropna(subset=["title"])
        # Fill NaNs in text columns
        df[["authors", "genres", "description", "cover_image_url"]] = df[["authors", "genres", "description", "cover_image_url"]].fillna("")
        
        logger.info(f"Saving prepared data to {output_path}...")
        df.to_csv(output_path, index=False)
        logger.info(f"Successfully prepared {len(df)} books.")
        
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise

if __name__ == "__main__":
    prepare_100k_data()
