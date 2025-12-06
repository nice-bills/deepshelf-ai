# Serendipity (Book Discovery Platform)

![Serendipity UI](https://img.shields.io/badge/Frontend-React_18-indigo) ![Backend](https://img.shields.io/badge/Backend-FastAPI-009688) ![Engine](https://img.shields.io/badge/AI_Engine-DeepShelf-blueviolet) ![Status](https://img.shields.io/badge/Status-Active-success)

**Serendipity** is a next-generation book discovery platform that moves beyond keyword matching. It uses **Semantic Search** and **Vector Embeddings** to understand the "vibe", plot, and emotional resonance of a book.

Powered by the **DeepShelf Engine**, it allows users to search for "a cyberpunk detective story set in Tokyo" and get results that match the *meaning*, not just the words.

---

## Key Features

### The Experience (Frontend)
*   **Semantic Search:** Type natural language queries like "sad books that feel like a rainy day."
*   **Persona Picker (New!):** Experience the personalization engine by switching between pre-defined personas (e.g., "The Futurist", "The History Buff") to see how recommendations adapt to different reading histories.
*   **Curated Clusters:** Explore automatically generated collections like "Space Opera", "Victorian Mystery", etc.
*   **Explainability:** The AI explains *why* a book was recommended (e.g., "92% Plot Match", "Similar tone to your history").

### The Engine (Backend)
*   **Microservice Architecture:**
    *   **Books API:** Handles business logic, product data, and frontend communication.
    *   **Personalisation Engine:** A dedicated vector search service (FAISS + Transformers) deployed separately.
*   **Performance:**
    *   **IVF-PQ Indexing:** 100k+ books indexed with 48x compression (150MB -> 3MB) for <50ms query times.
    *   **Hybrid Search:** Combines dense vector retrieval with metadata filtering.

---

## System Architecture

The system consists of two main services and a frontend:

```mermaid
graph LR
    User((User)) --> Frontend[React App]
    Frontend --> BooksAPI[Books API (FastAPI)]
    BooksAPI --> DB[(Book Catalog CSV)]
    BooksAPI --> PersonaliseAPI[Personalisation Engine]
    PersonaliseAPI --> VectorDB[(FAISS Index)]
```

*   **Frontend:** React, Tailwind, Vite (Static Build)
*   **Books API:** Python, FastAPI (Deployed on Hugging Face Spaces)
*   **Personalisation Engine:** Python, Sentence-Transformers, FAISS (Deployed on Hugging Face Spaces)

---

## Getting Started

### Prerequisites
*   **Python 3.10+** (Essential for dependency compatibility)
*   **Node.js 18+**
*   **Git**

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd books
```

### 2. Backend Setup (Books API)

It is highly recommended to use `uv` for fast dependency management, but `pip` works too.

```bash
# 1. Create virtual environment
pip install uv
uv venv .venv

# 2. Activate environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 3. Install dependencies
uv pip install -r requirements.txt
```

**Configuration (`.env`):**
Copy `.env.example` to `.env` and configure the URL of your personalisation service.
```bash
cp .env.example .env
```
In `.env`:
```ini
PORT=8000
# URL of your deployed Personalisation Engine (or http://localhost:8001 if running locally)
PERSONALIZER_URL=https://nice-bill-personalisation-engine.hf.space
```

**Run the API:**
```bash
python src/book_recommender/api/main.py
# API will start at http://localhost:8000
# Swagger Docs: http://localhost:8000/docs
```

### 3. Frontend Setup

```bash
cd frontend

# 1. Install dependencies
npm install

# 2. Start Dev Server
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## Deployment Guide

### Option 1: Docker (Recommended)

The project includes a production-ready `Dockerfile` optimized for Hugging Face Spaces.

```bash
# Build the image
docker build -t serendipity-api -f docker/Dockerfile.backend .

# Run the container
docker run -p 7860:7860 -e PERSONALIZER_URL=https://nice-bill-personalisation-engine.hf.space serendipity-api
```

### Option 2: Hugging Face Spaces

1.  Create a new **Docker** Space on Hugging Face.
2.  Connect it to your GitHub repository (or push manually).
3.  Set the following **Variables** in the Space settings:
    *   `PERSONALIZER_URL`: `https://nice-bill-personalisation-engine.hf.space` (or your engine's URL)
4.  The Space will automatically build using `docker/Dockerfile.backend` and serve on port 7860.

---

## Project Structure

```
books/
├── data/                   # Data storage
│   ├── catalog/            # CSV/Parquet metadata
│   └── embeddings_cache.npy # Vector data
├── docker/                 # Docker configuration
│   └── Dockerfile.backend
├── frontend/               # React Application
│   ├── src/
│   └── public/
├── scripts/                # Data processing scripts
│   ├── download_data.py    # Fetches datasets
│   └── prepare_100k_data.py # Raw data cleaning
├── src/
│   └── book_recommender/
│       ├── api/            # FastAPI endpoints
│       ├── ml/             # Machine Learning logic
│       └── services/       # External service connectors
└── tests/                  # Pytest suite
```

## License
MIT License.