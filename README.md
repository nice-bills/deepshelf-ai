# BookFinder AI 📚✨

![Build Status](https://github.com/your-username/your-repo-name/actions/workflows/ci-cd.yml/badge.svg)

A modern, full-stack **Semantic Book Recommendation Engine**. It uses AI to understand the "vibe" of your request (not just keywords) and finds books that match your description.

> **New:** Now featuring **Generative AI Explanations** powered by Groq (Llama 3), a tactile **React Frontend**, and a production-ready **Dockerized Backend**.

## ✨ Features

*   **🧠 Semantic Search:** Find books by describing plot, mood, or character (e.g., *"A cyberpunk detective story with a noir vibe"*).
*   **🤖 AI Explanations:** Don't just get a list; get a personalized pitch. The app explains *why* a book fits your query using GenAI.
*   **🎨 Dynamic UI:** A beautiful React interface with:
    *   **Interactive Backgrounds:** "Lava Lamp" blobs, solid colors, or custom wallpapers.
    *   **Haptic Feedback:** Tactile buttons and interactions.
    *   **Glassmorphism:** Modern, translucent aesthetics.
*   **🔍 Smart Discovery:**
    *   **"More Like This":** Instantly find similar books from any detail view.
    *   **Dynamic Clusters:** Explore automatically generated collections based on your library's themes.
*   **⚡ High Performance:** Parallelized cover image fetching and optimized FAISS vector search.

## 🛠️ Tech Stack

### **Frontend**
*   **React 19** (Vite)
*   **TypeScript**
*   **Tailwind CSS** (Styling)
*   **Lucide React** (Icons)
*   **Axios** (API Client)

### **Backend**
*   **Python 3.10+**
*   **FastAPI** (Async API)
*   **Sentence-Transformers** (Embeddings)
*   **FAISS** (Vector Search)
*   **Groq API** (Llama 3 for Explanations)
*   **Pandas / NumPy** (Data Processing)

### **Infrastructure**
*   **Docker** (Multi-stage builds)
*   **Uvicorn** (ASGI Server)

---

## 🚀 Quick Start (Local Development)

### 1. Prerequisites
*   Python 3.10+
*   Node.js 18+ & npm
*   [Groq API Key](https://console.groq.com) (Free)

### 2. Backend Setup

1.  **Install Python dependencies:**
    ```bash
    # Using uv (recommended)
    uv pip install -e . -r requirements.txt
    
    # OR using standard pip
    pip install -e . -r requirements.txt
    ```

2.  **Prepare Data:**
    Place your `goodreads_data.csv` in `data/raw/` and run:
    ```bash
    python scripts/prepare_goodreads_data.py
    ```

3.  **Configure Environment:**
    Create a `.env` file in the root directory:
    ```env
    GROQ_API_KEY=gsk_your_actual_key_here
    # Optional: Google Books API for better cover images
    # GOOGLE_BOOKS_API_KEY=...
    ```

4.  **Run API:**
    ```bash
    uv run uvicorn src.book_recommender.api.main:app --reload --host 0.0.0.0 --port 8000
    ```
    API will be live at `http://localhost:8000`.

### 3. Frontend Setup

1.  **Navigate to frontend:**
    ```bash
    cd frontend
    ```

2.  **Install dependencies:**
    ```bash
    npm install
    ```

3.  **Run Dev Server:**
    ```bash
    npm run dev
    ```
    App will be live at `http://localhost:5173`.

---

## 🐳 Docker Deployment

The project includes a production-ready Docker setup for the backend.

**Build the Backend Image:**
```bash
docker build -f docker/Dockerfile.backend -t bookfinder-api .
```

**Run the Container:**
```bash
docker run -p 8000:8000 --env-file .env bookfinder-api
```

This image is optimized (slim) and ready for deployment on platforms like **Render**, **Railway**, or **Fly.io**.

---

## 📂 Project Structure

```
bookfinder-ai/
├── frontend/               # React Application
│   ├── src/
│   │   ├── api.ts          # API Client
│   │   ├── App.tsx         # Main Component
│   │   └── types.ts        # TypeScript Interfaces
├── src/book_recommender/   # Python Package
│   ├── api/                # FastAPI Endpoints
│   ├── ml/                 # AI Logic (FAISS, Explainability)
│   ├── data/               # Data Processing
│   └── utils.py            # Utilities (Cover fetching)
├── data/                   # Data Storage
│   ├── raw/                # Input CSVs
│   └── processed/          # Embeddings & Parquet
├── docker/                 # Docker Configuration
├── scripts/                # Data Prep Scripts
└── tests/                  # Pytest Suite
```

---

## 🔮 Future Roadmap

*   [ ] **Vector Database:** Migrate from local FAISS to Qdrant/Pinecone for scalability.
*   [ ] **User Accounts:** Save favorite books and custom themes to a database.
*   [ ] **Redis Caching:** Cache API responses and cover URLs for instant loading.
*   [ ] **Automated Pipeline:** Auto-ingest new datasets dropped into `data/raw`.

---

**License:** MIT
