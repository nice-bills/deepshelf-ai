# tests/conftest.py
import pytest
import logging

@pytest.fixture(autouse=True)
def caplog_for_tests(caplog):
    """
    Fixture to capture logs during tests and set a default logging level.
    This prevents log messages from cluttering test output unless a test fails.
    """
    caplog.set_level(logging.INFO)
    yield
