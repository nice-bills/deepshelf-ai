@echo off
call .venv\Scripts\activate.bat
set PYTHONPATH=%CD%
python -m streamlit run src/book_recommender/apps/analytics_app.py --server.port=8502