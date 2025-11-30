import pandas as pd
import requests
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/raw")
INPUT_FILE = DATA_DIR / "books_prepared.csv"

def get_openlibrary_cover(title, author):
    try:
        # Simple cleaning
        clean_title = title.replace('&', '').split('(')[0].strip()
        clean_author = author.split(',')[0].strip() if author else ""
        
        query = f"title={clean_title}&author={clean_author}"
        url = f"https://openlibrary.org/search.json?{query}&limit=1"
        
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("docs"):
                doc = data["docs"][0]
                if "cover_i" in doc:
                    return f"https://covers.openlibrary.org/b/id/{doc['cover_i']}-L.jpg"
    except Exception as e:
        logger.warning(f"Error fetching cover for {title}: {e}")
    return None

def enrich_data():
    if not INPUT_FILE.exists():
        logger.error(f"File not found: {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)
    logger.info(f"Loaded {len(df)} books.")

    if "cover_image_url" not in df.columns:
        df["cover_image_url"] = None

    # Filter for rows without covers
    # We check for NaN or empty string
    mask = df["cover_image_url"].isna() | (df["cover_image_url"] == "")
    indices = df[mask].index
    
    logger.info(f"Found {len(indices)} books missing covers.")
    
    # Process a batch (e.g., 50) to demonstrate improvement without timeout
    # The user can run this script repeatedly or increase limit
    BATCH_SIZE = 20
    count = 0
    
    for idx in indices:
        if count >= BATCH_SIZE:
            break
            
        row = df.loc[idx]
        title = row['title']
        author = row['authors']
        
        logger.info(f"[{count+1}/{BATCH_SIZE}] Fetching cover for: {title}")
        cover_url = get_openlibrary_cover(title, author)
        
        if cover_url:
            df.at[idx, 'cover_image_url'] = cover_url
            logger.info(f"  -> Found: {cover_url}")
        else:
            logger.info("  -> No cover found.")
        
        time.sleep(0.2) # Polite delay
        count += 1

    # Save back
    df.to_csv(INPUT_FILE, index=False)
    logger.info(f"Saved enriched data to {INPUT_FILE}")

if __name__ == "__main__":
    enrich_data()
