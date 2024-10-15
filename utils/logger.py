# utils/logger.py

import logging
import os
from pathlib import Path
from logging import Logger

def setup_logger(name: str = __name__, log_dir: str = "logs", log_file: str = "app.log", level: int = logging.INFO) -> Logger:
    """
    Sets up and returns a logger with both console and file handlers.

    Args:
        name (str): Name of the logger. Defaults to the module's name.
        log_dir (str): Directory where the log file will be saved. Defaults to "logs".
        log_file (str): Name of the log file. Defaults to "app.log".
        level (int): Logging level. Defaults to logging.INFO.

    Returns:
        Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Prevents logging from propagating to the root logger multiple times

    if not logger.handlers:
        # Create log directory if it doesn't exist
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_path = Path(log_dir) / log_file

        # Formatter for log messages
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
