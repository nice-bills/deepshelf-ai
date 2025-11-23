@echo off
call .venv\Scripts\activate.bat
set PYTHONPATH=%CD%
uv run streamlit run src/book_recommender/apps/main_app.py