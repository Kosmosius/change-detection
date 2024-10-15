# utils/logger.py

import logging
from pathlib import Path
from logging import Logger
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import sys

def setup_logger(
    name: str = __name__,
    log_dir: str = "logs",
    log_file: str = "app.log",
    level: int = logging.INFO,
    max_bytes: int = 10*1024*1024,  # 10 MB
    backup_count: int = 5,
    rotation_when: str = 'midnight',
    rotation_interval: int = 1,
    formatter: logging.Formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
) -> Logger:
    """
    Sets up and returns a logger with both console and file handlers, including log rotation.

    Args:
        name (str): Name of the logger. Defaults to the module's name.
        log_dir (str): Directory where the log file will be saved. Defaults to "logs".
        log_file (str): Name of the log file. Defaults to "app.log".
        level (int): Logging level. Defaults to logging.INFO.
        max_bytes (int): Maximum size of the log file in bytes before rotation. Defaults to 10 MB.
        backup_count (int): Number of backup log files to keep. Defaults to 5.
        rotation_when (str): Type of rotation interval ('S', 'M', 'H', 'D', 'midnight', 'W0'-'W6'). Defaults to 'midnight'.
        rotation_interval (int): How often to rotate based on 'rotation_when'. Defaults to 1.
        formatter (logging.Formatter): Formatter for log messages. Defaults to a standard formatter.

    Returns:
        Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Prevents logging from propagating to the root logger multiple times

    if not logger.handlers:
        # Create log directory if it doesn't exist
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        full_log_path = log_path / log_file

        # Console Handler with colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File Handler with log rotation based on size
        file_handler = RotatingFileHandler(
            full_log_path,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Timed Rotating File Handler (e.g., daily rotation)
        timed_file_handler = TimedRotatingFileHandler(
            full_log_path.with_suffix('.timed.log'),
            when=rotation_when,
            interval=rotation_interval,
            backupCount=backup_count,
            encoding='utf-8',
            delay=False,
            utc=False
        )
        timed_file_handler.setLevel(level)
        timed_file_handler.setFormatter(formatter)
        logger.addHandler(timed_file_handler)

        logger.debug(f"Logger '{name}' initialized with handlers: Console, RotatingFile, TimedRotatingFile.")

    return logger
