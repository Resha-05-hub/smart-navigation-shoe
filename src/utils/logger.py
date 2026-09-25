"""Centralized logging utility for Smart Navigation Shoe."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "smart_shoe",
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """Configures and returns a logger instance.

    Args:
        name: Logger module name.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to output log file.
        log_format: Custom log line format string.

    Returns:
        logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if already configured
    if logger.hasHandlers():
        return logger

    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    fmt = log_format or "%(asctime)s - [%(name)s] - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (Optional)
    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(file_path), encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "smart_shoe") -> logging.Logger:
    """Retrieves an existing logger instance."""
    return logging.getLogger(name)
