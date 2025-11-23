.PHONY: install test lint format clean run-web run-api run-analytics docker-build docker-up

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

install-dev:
	pip install -e ".[dev]"

test:
	pytest --cov=src --cov-report=html --cov-report=term

lint:
	ruff check src/ tests/
	black --check src/ tests/
	mypy src/

format:
	black src/ tests/
	ruff check --fix src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov dist build *.egg-info

run-web:
	PYTHONPATH=. python -m streamlit run src/book_recommender/apps/main_app.py

run-api:
	PYTHONPATH=. python -m uvicorn src.book_recommender.api.main:app --reload

run-analytics:
	PYTHONPATH=. python -m streamlit run src/book_recommender/apps/analytics_app.py --server.port=8502

docker-build:
	docker build -t bookfinder-ai:latest .

docker-up:
	docker-compose up

docker-down:
	docker-compose down

help:
	@echo "Available commands:"
	@echo "  make install        - Install dependencies"
	@echo "  make test          - Run tests with coverage"
	@echo "  make lint          - Run linters"
	@echo "  make format        - Format code"
	@echo "  make run-web       - Start Streamlit app"
	@echo "  make run-api       - Start FastAPI server"
	@echo "  make docker-up     - Start all services with Docker"
