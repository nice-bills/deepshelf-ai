# 📘 BookFinder-AI
![Build Status](https://github.com/your-username/your-repo-name/actions/workflows/ci-cd.yml/badge.svg)
[![Codecov](https://codecov.io/gh/your-username/your-repo-name/branch/main/graph/badge.svg?token=your-codecov-token)](https://codecov.io/gh/your-username/your-repo-name)

This project is a Content-Based Book Recommender MVP built using a 100% open-source Python stack. It uses sentence embeddings to find books with similar content and provides a simple, modern web interface to discover books based on a text description.

## ✨ Features

-   **Semantic Search**: Instead of selecting a title, describe the book you want to read, and the recommender will find matching books based on semantic meaning.
-   **Content-Based Recommendations**: Finds similar books using semantic meaning, not just keywords.
-   **Modern Web UI**: A clean, card-based interface built with Streamlit that displays book covers and expandable details.
-   **Open-Source Stack**: Built entirely with free and open-source tools.
-   **Modular & Extendable**: Code is organized into a proper Python package (`src/book_recommender`), making it easy to extend.
-   **Automated Setup**: The data processing and embedding generation can be run with simple commands.

## ⚙️ Tech Stack

- **Python 3.12+**
- **Streamlit**: For the web application UI.
- **FastAPI**: For the RESTful API.
- **Sentence-Transformers**: For generating high-quality sentence embeddings.
- **FAISS**: For efficient, fast similarity search.
- **Pandas**: For data manipulation.
- **Uvicorn**: For running the FastAPI application.
- **Pytest**: For running unit tests.
- **Docker**: For containerization.

---

## 🏗️ Project Structure

The project is organized into a modular `src` layout:

```
bookfinder-ai/
├── src/book_recommender/           # Main package
│   ├── api/                        # REST API layer
│   ├── apps/                       # User interfaces
│   ├── core/                       # Core configuration, exceptions, etc.
│   ├── data/                       # Data processing
│   └── ml/                         # Machine learning
├── tests/                          # Test suite
├── data/                           # Data storage (not versioned by default)
├── scripts/                        # Utility scripts
├── docs/                           # Documentation
├── logs/                           # Application logs
├── .github/workflows/ci-cd.yml     # CI/CD pipeline
├── docker-compose.yml              # Multi-service setup
├── streamlit.Dockerfile            # Dockerfile for Streamlit app
├── api.Dockerfile                  # Dockerfile for FastAPI
├── analytics.Dockerfile            # Dockerfile for Analytics app
├── run_app.bat                     # Script to run the main Streamlit app
├── run_api.bat                     # Script to run the FastAPI app
├── run_analytics.bat               # Script to run the analytics app
...
```

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.12 or higher
- `uv` package manager (`pip install uv`)

### 2. Clone the Repository

```bash
git clone <your-repo-url>
cd book-recommender
```

### 3. Create and Activate Virtual Environment

```bash
uv venv
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\Activate.ps1 # Windows PowerShell
```

### 4. Install Dependencies

Install all required packages in editable mode:

```bash
uv pip install -e . -r requirements.txt -r requirements-dev.txt
```

### 5. Add and Prepare the Data

1.  **Add your dataset**: Place your `goodreads_data.csv` file inside the `data/raw/` directory.
2.  **Prepare the data**: Run the preprocessing script:

```bash
uv run python scripts/prepare_goodreads_data.py
```

### 6. Run the Applications

You can run each service using the provided batch scripts (for Windows).

-   **Run the main Streamlit app:**
    ```bash
    run_app.bat
    ```
    Access at `http://localhost:8501`.

-   **Run the FastAPI:**
    ```bash
    run_api.bat
    ```
    Access the docs at `http://localhost:8000/docs`.

-   **Run the Analytics Dashboard:**
    ```bash
    run_analytics.bat
    ```
    Access at `http://localhost:8502`.

---

## 🧪 Running Tests

Ensure your virtual environment is activated and dependencies are installed.

To run the tests, execute the following command from the project root:

```bash
$env:TESTING_ENV="True"; pytest
```

This sets the `TESTING_ENV` variable to prevent the app from loading real models during tests. The `pyproject.toml` file is configured to automatically handle the `PYTHONPATH`.

---

## 🐳 Docker Compose Setup

For a more production-like setup, you can use Docker Compose to build and run all services.

```bash
# Build and start all services
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop and remove containers
docker-compose down
```

---

## ❓ Troubleshooting

**1. `ModuleNotFoundError: No module named 'book_recommender'`**
   - **Cause**: The `src` directory is not on the `PYTHONPATH`.
   - **Solution**: Make sure you have installed the project in editable mode (`uv pip install -e .`) and that your virtual environment is activated. If running manually, you must set the `PYTHONPATH` as described in the "Getting Started" section.

**2. `DataNotFoundError`**
   - **Cause**: The processed data files (`books_cleaned.parquet`, `book_embeddings.npy`) do not exist.
   - **Solution**: Run the data preparation script: `uv run python scripts/prepare_goodreads_data.py`.

**3. Application is slow on first run**
   - **Cause**: The sentence-transformer model is being downloaded for the first time.
   - **Solution**: This is expected. Subsequent runs will be fast as the model is cached.