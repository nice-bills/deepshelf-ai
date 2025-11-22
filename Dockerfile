# Dockerfile for Streamlit Book Recommender App

# --- Stage 1: Builder ---
# This stage prepares all the necessary assets: dependencies, models, and processed data.
FROM python:3.12-slim-bookworm AS builder

# Install uv and system dependencies
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /bin/
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Set up the application environment
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN uv pip install --no-cache-dir -r requirements.txt

# --- Pre-process data and generate embeddings ---
# Copy all necessary source, scripts, and raw data to run our preparation scripts.
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/raw/ ./data/raw/

# Run the scripts to prepare the data.
# This creates books_cleaned.parquet in /app/data/processed/
RUN python -m src.data_processor
# This downloads the model to cache and creates book_embeddings.npy
RUN python -m src.embedder


# --- Stage 2: Final Image ---
# This stage creates the lightweight final image for production.
FROM python:3.12-slim-bookworm

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN groupadd -r streamlit && useradd --no-create-home -r -g streamlit streamlit

# Set the working directory
WORKDIR /app

# Copy the Python environment with all dependencies from the builder stage
COPY --from=builder /app/.venv ./.venv

# Copy the pre-downloaded model cache from the builder stage
# This ensures the model is available without a runtime download
COPY --from=builder /root/.cache /root/.cache

# Copy the pre-processed data and embeddings from the builder stage
COPY --from=builder /app/data/processed/ ./data/processed/

# Copy the application source code
COPY app.py .
COPY src/ ./src/

# Set the PATH to include our virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Set ownership and switch to the non-root user
RUN chown -R streamlit:streamlit /app
USER streamlit

# Expose the Streamlit port
EXPOSE 8501

# Healthcheck to see if Streamlit is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Define the command to run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.runOnSave=false"]
