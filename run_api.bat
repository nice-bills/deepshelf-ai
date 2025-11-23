@echo off
call .venv\Scripts\activate.bat
set PYTHONPATH=%CD%
uv run uvicorn src.book_recommender.api.main:app --host 0.0.0.0 --port 8000 --reload