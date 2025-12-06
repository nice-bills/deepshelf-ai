# Architecture Overview: DeepShelf

This document provides a detailed overview of the BookFinder application's architecture, outlining its components, data flow, and key technology decisions.

## System Overview

BookFinder is a content-based book recommendation system that leverages natural language processing (NLP) and vector similarity search to help users discover books. It comprises a Streamlit-based web application for user interaction, a FastAPI service for programmatic access, and a suite of Python scripts for data processing and embedding generation.

The core principle is to transform book descriptions into high-dimensional numerical vectors (embeddings) using a pre-trained sentence transformer model. These embeddings are then used to find semantically similar books or to cluster books into thematic collections.

## Component Descriptions

The system is structured into several modular components:

1.  **Data Layer (`data/`)**: Stores raw, prepared, and processed data.
    *   `data/raw/`: Original datasets (e.g., `goodreads_data.csv`).
    *   `data/processed/`: Cleaned book metadata (`books_cleaned.parquet`), pre-computed embeddings (`book_embeddings.npy`), and embedding metadata (`embedding_metadata.json`).

2.  **Scripts Layer (`scripts/`)**: Utility scripts for data handling and model preparation.
    *   `prepare_goodreads_data.py`: Adapts raw data sources to a standardized format.
    *   `download_model.py` (optional): For pre-downloading the sentence transformer model.

3.  **Source Code Layer (`src/book_recommender/`)**: Contains the core logic of the application, organized into sub-packages.
    *   **`src/book_recommender/core/`**: Core utilities and configurations.
        *   `config.py`: Centralized configuration settings and constants.
        *   `exceptions.py`: Custom exception definitions for error handling.
        *   `logging_config.py`: Centralized logging configuration for the entire application.
    *   **`src/book_recommender/data/`**: Data processing components.
        *   `processor.py`: Handles data cleaning, normalization, deduplication, and feature engineering (e.g., creating `combined_text` for embeddings).
    *   **`src/book_recommender/ml/`**: Machine Learning related components.
        *   `embedder.py`: Manages the loading of the Sentence Transformer model and generation of embeddings for books and user queries. It uses `lru_cache` for efficient model loading.
        *   `recommender.py`: Implements the core recommendation logic using a FAISS index for fast Approximate Nearest Neighbor (ANN) search on book embeddings.
        *   `clustering.py`: Contains logic for K-Means clustering of book embeddings and generating descriptive names for these clusters based on common genres.
        *   `explainability.py`: Provides rule-based explanations for recommendations, detailing contributing factors like genre, keywords, and author similarity.
        *   `feedback.py`: Manages saving and retrieving user feedback on recommendations to a JSONL file.
    *   `src/book_recommender/utils.py`: General utility functions, including book cover fetching from various APIs (Google Books, Open Library) and batch processing.

4.  **User Interface Layer (`src/book_recommender/apps/`)**: Streamlit applications.
    *   `main_app.py`: The main interactive web application where users can get book recommendations, browse by genre/query, view explanations, and provide feedback.
    *   `analytics_app.py`: A separate Streamlit dashboard to visualize collected user feedback and system usage statistics.

5.  **API Layer (`src/book_recommender/api/`)**: FastAPI application for programmatic access.
    *   `main.py`: Defines the FastAPI application and its endpoints (recommendations, book listing, search, stats, clusters, explanations, feedback).
    *   `models.py`: Pydantic models for request and response data validation and serialization.
    *   `dependencies.py`: FastAPI dependency injection functions to manage and cache shared resources like the `BookRecommender` and embedding models.

6.  **CI/CD (`.github/workflows/ci-cd.yml`)**: GitHub Actions workflow for automated testing and code quality checks.

7.  **Containerization (`streamlit.Dockerfile`, `api.Dockerfile`, `analytics.Dockerfile`, `docker-compose.yml`)**: For packaging and orchestrating the application services.
    *   `streamlit.Dockerfile`: Defines the build process for the main Streamlit application into a Docker image.
    *   `api.Dockerfile`: Defines the build process for the FastAPI application into a Docker image.
    *   `analytics.Dockerfile`: Defines the build process for the analytics Streamlit application into a Docker image.
    *   `docker-compose.yml`: Orchestrates the `streamlit`, `api`, and `analytics` services for local development and deployment.

## Data Flow

The primary data flow for generating recommendations and user interaction is as follows:

1.  **Raw Data Ingestion**: Raw CSV datasets (e.g., `goodreads_data.csv`) are placed in `data/raw/`.
2.  **Data Preparation**: The `scripts/prepare_goodreads_data.py` script and `src/book_recommender/data/processor.py` clean, standardize, and deduplicate the raw data, saving it as `books_cleaned.parquet` in `data/processed/`.
3.  **Embedding Generation**: `src/book_recommender/ml/embedder.py` loads `books_cleaned.parquet` and generates semantic embeddings for each book's `combined_text`, saving them as `book_embeddings.npy` in `data/processed/`.
4.  **Application Startup**:
    *   The `src/book_recommender/apps/main_app.py` (Streamlit) and `src/book_recommender/api/main.py` (FastAPI) applications load `books_cleaned.parquet` and `book_embeddings.npy` into memory.
    *   `src/book_recommender/ml/recommender.py` initializes a FAISS index with these embeddings for fast similarity search.
    *   `src/book_recommender/ml/clustering.py` generates book clusters and their names from the embeddings.
5.  **User Interaction (Streamlit `src/book_recommender/apps/main_app.py`)**:
    *   User inputs a natural language query or browses clusters.
    *   If a query, `src/book_recommender/ml/embedder.py` generates an embedding for the query.
    *   `src/book_recommender/ml/recommender.py` uses the query embedding (or a book's embedding from a title search) to find similar books.
    *   `src/book_recommender/ml/explainability.py` generates reasons for recommendations.
    *   `src/book_recommender/utils.py` fetches book cover images.
    *   `src/book_recommender/ml/feedback.py` stores user feedback on recommendations.
6.  **API Interaction (FastAPI `src/book_recommender/api/main.py`)**:
    *   External clients send requests to API endpoints (e.g., `/recommend/query`, `/books`, `/feedback`).
    *   `src/book_recommender/api/dependencies.py` ensures efficient loading and caching of `recommender` and `embedder` instances.
    *   Requests are validated using Pydantic models (`src/book_recommender/api/models.py`).
    *   Core logic in `src/book_recommender/` modules is invoked (e.g., `recommender.py`, `explainability.py`, `feedback.py`).
    *   Responses are returned, adhering to defined Pydantic response models.
    
        **Example API Calls:**

        ```bash
        # Health Check
        curl http://localhost:8000/health

        # Recommend by Query
        curl -X POST "http://localhost:8000/recommend/query" \
          -H "Content-Type: application/json" \
          -d '{
            "query": "fantasy adventure with dragons",
            "top_k": 5
          }'

        # List all Clusters
        curl http://localhost:8000/clusters

        # Submit Feedback
        curl -X POST "http://localhost:8000/feedback" \
          -H "Content-Type: application/json" \
          -d '{
            "query": "fantasy adventure with dragons",
            "book_id": "example_book_id",
            "feedback_type": "positive",
            "session_id": "user_session_abc"
          }'
        ```

7.  **Analytics (`src/book_recommender/apps/analytics_app.py`)**:
    *   Loads accumulated feedback data from `data/feedback/user_feedback.jsonl` using `src/book_recommender/ml/feedback.py`.
    *   Processes and visualizes statistics using Streamlit and Plotly.

```mermaid
graph TD
    subgraph Data Flow & Processing
        raw_data[Raw Data (CSV)] --> A(scripts/prepare_goodreads_data.py);
        A --> B[books_prepared.csv];
        B --> C{src/book_recommender/data/processor.py};
        C --> D[books_cleaned.parquet];
        D --contains text--> E{src/book_recommender/ml/embedder.py};
        E --> F[book_embeddings.npy];
    end

    subgraph Application Runtime
        G(src/book_recommender/apps/main_app.py - Streamlit UI) --loads--> D & F;
        G --uses--> H(src/book_recommender/ml/recommender.py);
        G --uses--> I(src/book_recommender/ml/embedder.py);
        G --uses--> J(src/book_recommender/ml/clustering.py);
        G --uses--> K(src/book_recommender/ml/explainability.py);
        G --uses--> L(src/book_recommender/ml/feedback.py);
        
        M(src/book_recommender/api/main.py - FastAPI) --uses--> N(src/book_recommender/api/dependencies.py);
        N --loads--> D & F;
        N --uses--> H & I & J & K & L;

        O(src/book_recommender/apps/analytics_app.py - Streamlit Dashboard) --loads--> P[user_feedback.jsonl];
        P --via--> L;

        User[User] --Interacts with--> G;
        Client[External Client] --Interacts with--> M;
    end

    subgraph Utilities & Configuration
        Q[src/book_recommender/core/config.py];
        R[src/book_recommender/utils.py];
        S[src/book_recommender/core/logging_config.py];
    end

    style raw_data fill:#f9f,stroke:#333,stroke-width:2px
    style D fill:#ccf,stroke:#333,stroke-width:2px
    style F fill:#ccf,stroke:#333,stroke-width:2px
    style P fill:#fcc,stroke:#333,stroke-width:2px
```

## Technology Decisions

*   **Python 3.10+**: Modern, versatile language.
*   **Streamlit**: Chosen for rapid development of interactive web UIs with minimal frontend code. Its caching mechanisms (`@st.cache_resource`, `@st.cache_data`) are crucial for performance with ML models.
*   **FastAPI**: Selected for building a high-performance, asynchronous API with automatic Pydantic-based data validation and Swagger/OpenAPI documentation.
*   **Sentence-Transformers**: A powerful library for generating dense vector embeddings from text, suitable for semantic search.
*   **FAISS**: An efficient library for similarity search and clustering of dense vectors, essential for scaling recommendations.
*   **Pandas / NumPy**: Standard libraries for data manipulation and numerical operations.
*   **Scikit-learn**: Used for traditional machine learning tasks like K-Means clustering.
*   **Plotly Express**: For creating interactive and aesthetically pleasing visualizations in the analytics dashboard.
*   **Pydantic**: Data validation and settings management using Python type hints, integral to FastAPI.
*   **python-dotenv**: For managing environment variables, facilitating flexible configuration across environments.
*   **GitHub Actions**: For Continuous Integration/Continuous Deployment (CI/CD), automating testing, linting, and Docker image builds.
*   **Docker / Docker Compose**: For containerizing the application and orchestrating multi-service deployments, ensuring consistent environments.

## Future Considerations

*   **Data Version Control (DVC)**: Implement DVC for robust tracking of data and model versions, enhancing reproducibility in production.
*   **Scalability**: For extremely large datasets, consider distributed FAISS indexes or cloud-native vector databases.
*   **Advanced Explanations**: Explore more sophisticated XAI techniques beyond rule-based, potentially involving LLMs or specific feature attribution methods.
*   **User Authentication**: For multi-user scenarios, integrate an authentication system (e.g., OAuth2, JWT).
*   **Database Integration**: Replace JSONL feedback storage with a dedicated database (e.g., PostgreSQL) for more robust data management and querying.
*   **Full UI Testing**: Implement UI tests using tools like Playwright or Selenium to ensure frontend consistency and functionality.