import logging
import os
from logging.handlers import RotatingFileHandler


def configure_logging(
    log_file: str = "app.log",
    log_level: str = "INFO",
    max_bytes: int = 5 * 1024 * 1024,  # 5 MB
    backup_count: int = 5,
    log_dir: str = "logs",
):
    """
    Configures application-wide logging.

    Logs to both console and a rotating file.

    Args:
        log_file (str): The name of the log file.
        log_level (str): The minimum logging level to capture (e.g., "INFO", "DEBUG", "WARNING", "ERROR").
        max_bytes (int): Maximum size of the log file before rotation (in bytes).
        backup_count (int): Number of backup log files to keep.
        log_dir (str): Directory where log files will be stored.
    """
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers to prevent duplicate logs in case of re-configuration
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File Handler (rotating)
    file_handler = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logging.info(f"Logging configured. Level: {log_level}, Log File: {log_path}")


if __name__ == "__main__":
    # Example usage if run directly
    configure_logging(log_file="test_app.log", log_level="DEBUG")
    logger = logging.getLogger(__name__)
    logger.debug("This is a debug message from logging_config.py")
    logger.info("This is an info message from logging_config.py")
    logger.warning("This is a warning message from logging_config.py")
    logger.error("This is an error message from logging_config.py")
