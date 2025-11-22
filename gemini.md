# 📘 Open-Source Book Recommender MVP - Project Plan

This document outlines the step-by-step plan to build the Book Recommender MVP. We will track our progress here.

---

## ✅ **Phase 1: Project Setup & Data Foundation**

- [x] **Task 1: Initialize Project Structure**
  - Create the mandatory directory layout (`/data`, `/src`).
  - Create placeholder files (`app.py`, `requirements.txt`, etc.).

- [x] **Task 2: Implement Data Processor**
  - Create `src/data_processor.py`.
  - Load the raw `books.csv`.
  - Clean, normalize, and handle missing metadata.
  - **Added duplicate detection based on title.**
  - Create the `combined_text` field for embeddings.
  - Save the cleaned DataFrame to `/data/processed/books_cleaned.parquet` (formerly `.pkl`).
  - **Added a pre-processing script to adapt new data sources (e.g., `goodreads_data.csv`).**

## ⏳ **Phase 2: Core Recommender Engine**

- [x] **Task 3: Implement Embedder**
  - Create `src/embedder.py`.
  - Load the processed data from `books_cleaned.parquet`.
  - Load the `all-MiniLM-L6-v2` sentence-transformer model.
  - Generate embeddings for the `combined_text` of each book.
  - **Refactored to cache the model and allow on-the-fly embedding of user queries.**
  - Save the embeddings to `/data/processed/book_embeddings.npy`.

- [x] **Task 4: Implement Recommender Logic**
  - Create `src/recommender.py`.
  - Load the book metadata and embeddings.
  - Implement a function to compute cosine similarity between book vectors.
  - **Optimized title-to-index lookup for faster recommendations.**
  - **Refactored to support vector-based search, enabling search-by-description.**
  - Create the `get_recommendations(title: str)` function that returns the top K similar books.

## 🎨 **Phase 3: User Interface & Application**

- [x] **Task 5: Build the Streamlit UI**
  - Develop the main application in `app.py`.
  - Use `@st.cache_resource` to load data and models efficiently.
  - **Implemented search-by-description using a text area, replacing the book selection dropdown.**
  - Add a "Find Similar Books" button to trigger the recommendation logic.
  - **Display results in a modern, 3-column card grid with book covers.**
  - **Fetched and displayed book cover images from the Open Library API.**
  - **Added custom CSS for a more polished, modern design.**
  - Use `st.expander` to show full book details on demand, keeping the main view clean.
  - **UI code refactored into smaller functions for modularity.**
  - **App version displayed in the footer.**

## 🧪 **Phase 4: Testing & Documentation**

- [x] **Task 6: Add Basic Tests**
  - Add a test for `data_processor.py` to check for nulls and correct output.
  - Add a test for `embedder.py` to verify embedding shape and count.
  - Add a test for `recommender.py` to ensure it returns valid recommendations.
  - **Added `tests/test_integration.py` for end-to-end pipeline validation.**
  - **Added `tests/test_app.py` as a placeholder for Streamlit UI tests.**

- [x] **Task 7: Finalize Documentation**
  - Update the `README.md` with comprehensive instructions on how to set up the project, install dependencies, and run the application.

## 🚀 **Phase 5: Deployment**

- [x] **Task 8: Prepare for Deployment (Optional)**
  - Create a `Dockerfile` to containerize the application.
  - **Added `.dockerignore` file for optimized Docker builds.**
  - Ensure `requirements.txt` is complete and accurate.
  - Add instructions for deploying to Streamlit Cloud or Render.

---

## ✨ **Architectural & Code Quality Improvements**

- [x] **Decoupled Components**: Core logic functions are independent of file I/O.
- [x] **Centralized Configuration**: All hardcoded paths and default values are managed in `src/config.py`.
- [x] **Standardized Package Imports**: Resolved import conflicts to ensure reliability when running as a script or as part of the main application.
- [x] **Centralized Logging**: Logging is configured once at the application entry point.
- [x] **Custom Exception Handling**: Introduced custom exceptions for more specific error management.
- [x] **Improved Type Hinting**: Added return type hints to major functions.
- [x] **Refactored `__main__` Blocks**: Standalone scripts now use `argparse` for flexible execution.
- [x] **Safer Data Serialization**: Migrated from `pickle` to `Parquet` and `NumPy (.npy)`.
- [x] **Code Duplication Reduced**: Extracted common logic into `src/utils.py`.

---

## 📝 **Documentation Improvements**

- [x] **Missing Environment Setup Details**: `README.md` now explicitly mentions Python version requirements.
- [x] **Configuration Documentation**: A new section in `README.md` explains `src/config.py`.
- [x] **Missing API Documentation**: All public functions and methods now have comprehensive docstrings.
- [x] **No Troubleshooting Guide**: A basic troubleshooting section has been added to `README.md`.
- [x] **Dataset Requirements Clear**: `README.md` now clearly states the minimum and optional dataset requirements.
- [x] **Model Download Warning**: `README.md` and `src/embedder.py` now warn about the large model download.

---

## 🧪 **Testing Improvements**

- [x] **Tests Verify Recommender Accuracy**: `tests/test_recommender.py` now includes a test case for recommender accuracy.
- [x] **Test Coverage Measured**: `pytest-cov` is included in `requirements-dev.txt`, and instructions are in `README.md`.
- [x] **No Logging Configuration in Tests**: `tests/conftest.py` has been added to manage logging during tests.
- [x] **UI Test Placeholder**: `tests/test_app.py` provides a starting point for UI tests.

---

## 🌟 **Future Enhancements (Documented in `README.md`)**

- [x] **Data & Model Versioning**: Suggestions for DVC or manual versioning.
- [x] **API Layer**: Suggestion for integrating FastAPI.
- [x. **Scalability**: Suggestions for ANN libraries (FAISS, Annoy).
- [x] **Advanced UI/UX**: Multi-page app, autocomplete, mobile responsiveness, dark mode, pagination, user feedback.
- [x] **Performance Optimization**: Batch processing, caching similarity scores.
- [x] **Comprehensive Testing**: Performance tests, more detailed UI tests.
- [x] **Missing Development Dependencies**: `requirements-dev.txt` updated with `black`, `ruff`, `mypy`.
- [x] **Large Model Download**: Warning and guidance added.

---

**Project Status: Complete**
All identified issues and suggested improvements from the comprehensive analysis have been addressed, either by implementation or by documenting them as future enhancements. The project is now in a highly robust, secure, well-documented, and modular state.