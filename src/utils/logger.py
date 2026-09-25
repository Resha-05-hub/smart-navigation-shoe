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


DEFAULT_FORMAT = "%(asctime)s - [%(name)s] - %(levelname)s - %(message)s"


def setup_logging(config: Optional[dict] = None) -> None:
    """Configures application-wide logging once, from the config's logging section.

    - All module loggers (alerts, sensors, camera, ...) write to logging.system_log.
    - The console only shows logging.console_level and above (default WARNING), so the
      redrawn terminal dashboard is not interleaved with INFO lines.
    Calling it again replaces the handlers it added before.
    """
    log_cfg = (config or {}).get("logging", {})
    formatter = logging.Formatter(log_cfg.get("format", DEFAULT_FORMAT))
    root = logging.getLogger()

    for handler in list(root.handlers):
        if getattr(handler, "_smart_shoe", False):
            root.removeHandler(handler)
            handler.close()

    root.setLevel(getattr(logging, str(log_cfg.get("level", "INFO")).upper(), logging.INFO))

    console = logging.StreamHandler()
    console.setLevel(getattr(logging, str(log_cfg.get("console_level", "WARNING")).upper(), logging.WARNING))
    console.setFormatter(formatter)
    console._smart_shoe = True  # type: ignore[attr-defined]
    root.addHandler(console)

    # Third-party libraries that log routine internals at INFO (e.g. the Windows TTS COM bridge)
    for noisy in ("comtypes", "ultralytics", "PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    system_log = log_cfg.get("system_log")
    if log_cfg.get("enabled", True) and system_log:
        path = Path(system_log)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(path), encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler._smart_shoe = True  # type: ignore[attr-defined]
        root.addHandler(file_handler)


def get_logger(name: str = "smart_shoe") -> logging.Logger:
    """Retrieves an existing logger instance."""
    return logging.getLogger(name)
