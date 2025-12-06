# Deployment Notes & Configuration Guide

*Created on: November 30, 2025*
*Project Status: MVP Complete / Production Ready*

This document serves as a "black box" recovery manual. If you return to this project in the future and need to redeploy it, follow these exact steps.

---

## 1. Environment Variables 
These are the variables you MUST set in your cloud provider (Render/Vercel/Railway) for the app to function.

### **Backend (FastAPI / Docker)**
*Deployment Platform: Render (Web Service) or Railway*

| Variable Name | Value (Example/Format) | Purpose |
| :--- | :--- | :--- |
| `HF_DATASET_ID` | `nice-bill/book-recommender-data` | **CRITICAL.** Tells the app where to download the embeddings/parquet files from Hugging Face. |
| `GROQ_API_KEY` | `gsk_...` | Required for the "Why this match?" AI explanations. Get a key at [console.groq.com](https://console.groq.com). |
| `HF_TOKEN` | `hf_...` | *(Optional)* Only required if you made your Hugging Face dataset PRIVATE. |
| `PORT` | `8000` | (Render sets this automatically, but good to know). |

### **Frontend (React / Vite)**
*Deployment Platform: Vercel or Netlify*

| Variable Name | Value (Example/Format) | Purpose |
| :--- | :--- | :--- |
| `VITE_API_URL` | `https://your-app-name.onrender.com` | Points the frontend to your live backend. **Do not add a trailing slash.** |

---

## 2. Build Settings

### **Backend**
*   **Runtime:** Docker
*   **Dockerfile Path:** `docker/Dockerfile.backend`
*   **Context Directory:** `.` (Root of the repo)
*   **Start Command:** (Handled automatically by Dockerfile: `scripts/download_data.py && uvicorn ...`)

### **Frontend**
*   **Framework Preset:** Vite
*   **Root Directory:** `frontend`
*   **Build Command:** `npm run build`
*   **Output Directory:** `dist`
*   **Node Version:** 18.x or 20.x

---

## 3. "Cold Start" Recovery
If the application is crashing on startup after a long time:

1.  **Check Hugging Face Data:**
    *   Go to: `https://huggingface.co/datasets/nice-bill/book-recommender-data`
    *   Ensure these 4 files still exist: `books_cleaned.parquet`, `book_embeddings.npy`, `cluster_cache.pkl`, `embedding_metadata.json`.

2.  **Check Groq API:**
    *   API keys sometimes expire or quotas change. Generate a new one if the "Why this match?" feature fails.

3.  **Local Run:**
    *   Backend: `uv run src/book_recommender/api/main.py`
    *   Frontend: `cd frontend && npm run dev`

---

## 4. Key Features Summary
*   **Search:** Semantic search using `all-MiniLM-L6-v2` embeddings.
*   **Explanation:** Uses Llama 3 (via Groq) to explain recommendations.
*   **Data Layer:** Self-healing. Downloads data on boot if missing.
*   **UI:** React + Tailwind + Lucide Icons. Features "Glassmorphism" and tactile hover effects.
