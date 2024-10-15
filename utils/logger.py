# utils/logger.py

import logging
from pathlib import Path
from logging import Logger
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import sys
from typing import Optional, Union

def setup_logger(
    name: str = __name__,
    log_dir: Union[str, Path] = "logs",
    log_file: str = "app.log",
    level: int = logging.INFO,
    max_bytes: Optional[int] = None,
    backup_count: int = 5,
    rotation_when: Optional[str] = None,
    rotation_interval: int = 1,
    formatter: Optional[logging.Formatter] = None
) -> Logger:
    """
    Sets up and returns a logger with console and file handlers, including optional log rotation.

    Args:
        name (str): Name of the logger. Defaults to the module's name.
        log_dir (Union[str, Path]): Directory where the log file will be saved. Defaults to "logs".
        log_file (str): Name of the log file. Defaults to "app.log".
        level (int): Logging level. Defaults to logging.INFO.
        max_bytes (Optional[int]): Maximum size of the log file in bytes before rotation. Use None to disable size-based rotation.
        backup_count (int): Number of backup log files to keep. Defaults to 5.
        rotation_when (Optional[str]): Time interval for rotating logs ('S', 'M', 'H', 'D', 'midnight', 'W0'-'W6'). Use None to disable time-based rotation.
        rotation_interval (int): How often to rotate based on 'rotation_when'. Defaults to 1.
        formatter (Optional[logging.Formatter]): Formatter for log messages. Defaults to a standard formatter.

    Returns:
        Logger: Configured logger instance.

    Raises:
        OSError: If the log directory cannot be created.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Prevents logs from propagating to the root logger multiple times

    if not logger.handlers:
        # Create log directory if it doesn't exist
        log_path = Path(log_dir)
        try:
            log_path.mkdir(parents=True, exist_ok=True)
        except (OSError, IOError) as e:
            logger.error("Failed to create log directory '%s': %s", log_path, e)
            raise

        full_log_path = log_path / log_file

        if formatter is None:
            formatter = logging.Formatter(
                fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )

        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File Handler
        if max_bytes:
            # File Handler with log rotation based on size
            file_handler = RotatingFileHandler(
                filename=full_log_path,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
        elif rotation_when:
            # File Handler with log rotation based on time
            file_handler = TimedRotatingFileHandler(
                filename=full_log_path,
                when=rotation_when,
                interval=rotation_interval,
                backupCount=backup_count,
                encoding='utf-8',
                utc=False
            )
        else:
            # Regular File Handler without rotation
            file_handler = logging.FileHandler(filename=full_log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.debug("Logger '%s' initialized with handlers.", name)

    return logger
