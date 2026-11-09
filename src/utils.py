"""
Utility functions for Aircraft Engine Predictive Maintenance (AEPM).

Provides reusable helper functions for logging, directory handling,
file validation, and environment setup.
"""

import logging
import random
from pathlib import Path
from typing import Optional, Union


def setup_logger(
    name: str = "AEPM",
    level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
) -> logging.Logger:
    """Configure and return a standard logger instance.

    Args:
        name: Name of the logger instance.
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
        log_file: Optional file path to write log messages.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if logger is re-initialized
    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Stream handler for console output
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler if log_file is specified
        if log_file:
            log_path = Path(log_file)
            ensure_dir(log_path.parent)
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def ensure_dir(dir_path: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it and parent directories if needed.

    Args:
        dir_path: Target directory path.

    Returns:
        Path: Path object for the target directory.
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_file_exists(file_path: Union[str, Path]) -> Path:
    """Check whether a file exists at the given path.

    Args:
        file_path: Path to the target file.

    Returns:
        Path: Resolved Path object if the file exists.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Required file not found at '{path}'. "
            "Please ensure the file is downloaded and placed in the correct directory."
        )
    return path


def set_random_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across standard libraries.

    Args:
        seed: Integer random seed value.
    """
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
