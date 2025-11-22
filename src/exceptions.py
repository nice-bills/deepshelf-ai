# src/exceptions.py
"""Custom exception types for the book recommender application."""

class BookRecommenderError(Exception):
    """Base class for exceptions in this application."""
    pass

class DataNotFoundError(BookRecommenderError, FileNotFoundError):
    """Raised when a required data file (e.g., CSV, pickle) is not found."""
    pass

class FileProcessingError(BookRecommenderError):
    """Raised when there's an error during file processing (e.g., CSV parsing)."""
    pass

class ModelLoadError(BookRecommenderError):
    """Raised when the sentence-transformer model cannot be loaded."""
    pass
