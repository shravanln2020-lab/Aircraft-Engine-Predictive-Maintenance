"""
Data loader module for NASA C-MAPSS FD001 dataset.

Provides validated, error-handled loading functions for training, test,
and ground-truth RUL text files into pandas DataFrames.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import pandas as pd

from src import config
from src.utils import setup_logger, validate_file_exists

logger = setup_logger(__name__)


def load_train_data(
    file_path: Union[str, Path] = config.TRAIN_FD001_PATH
) -> pd.DataFrame:
    """Load the C-MAPSS FD001 training dataset into a pandas DataFrame.

    Args:
        file_path: Path to train_FD001.txt. Defaults to config.TRAIN_FD001_PATH.

    Returns:
        pd.DataFrame: Training data with assigned C-MAPSS column names.

    Raises:
        FileNotFoundError: If the train data file is missing.
        ValueError: If the file cannot be parsed into a DataFrame.
    """
    valid_path = validate_file_exists(file_path)
    logger.info("Loading training data from: %s", valid_path)

    try:
        df = pd.read_csv(
            valid_path,
            sep=r"\s+",
            header=None,
            names=config.COLUMN_NAMES,
            engine="python",
        )
        logger.info(
            "Successfully loaded training data: %d rows, %d columns.",
            df.shape[0],
            df.shape[1],
        )
        return df
    except Exception as err:
        logger.error("Failed to load training data from %s: %s", valid_path, err)
        raise ValueError(f"Error parsing training dataset at '{valid_path}': {err}") from err


def load_test_data(
    file_path: Union[str, Path] = config.TEST_FD001_PATH
) -> pd.DataFrame:
    """Load the C-MAPSS FD001 test dataset into a pandas DataFrame.

    Args:
        file_path: Path to test_FD001.txt. Defaults to config.TEST_FD001_PATH.

    Returns:
        pd.DataFrame: Test data with assigned C-MAPSS column names.

    Raises:
        FileNotFoundError: If the test data file is missing.
        ValueError: If the file cannot be parsed into a DataFrame.
    """
    valid_path = validate_file_exists(file_path)
    logger.info("Loading test data from: %s", valid_path)

    try:
        df = pd.read_csv(
            valid_path,
            sep=r"\s+",
            header=None,
            names=config.COLUMN_NAMES,
            engine="python",
        )
        logger.info(
            "Successfully loaded test data: %d rows, %d columns.",
            df.shape[0],
            df.shape[1],
        )
        return df
    except Exception as err:
        logger.error("Failed to load test data from %s: %s", valid_path, err)
        raise ValueError(f"Error parsing test dataset at '{valid_path}': {err}") from err


def load_rul_data(
    file_path: Union[str, Path] = config.RUL_FD001_PATH
) -> pd.DataFrame:
    """Load the C-MAPSS FD001 ground-truth RUL values into a pandas DataFrame.

    Args:
        file_path: Path to RUL_FD001.txt. Defaults to config.RUL_FD001_PATH.

    Returns:
        pd.DataFrame: Ground-truth RUL DataFrame with columns ['unit_nr', 'RUL'].

    Raises:
        FileNotFoundError: If the RUL data file is missing.
        ValueError: If the file cannot be parsed into a DataFrame.
    """
    valid_path = validate_file_exists(file_path)
    logger.info("Loading ground-truth RUL data from: %s", valid_path)

    try:
        df = pd.read_csv(
            valid_path,
            sep=r"\s+",
            header=None,
            names=[config.RUL_COL],
            engine="python",
        )
        # Assign 1-indexed unit_nr corresponding to engine IDs in test set
        df.insert(0, "unit_nr", range(1, len(df) + 1))
        logger.info(
            "Successfully loaded RUL data: %d engines, %d columns.",
            df.shape[0],
            df.shape[1],
        )
        return df
    except Exception as err:
        logger.error("Failed to load RUL data from %s: %s", valid_path, err)
        raise ValueError(f"Error parsing RUL dataset at '{valid_path}': {err}") from err


def load_all_fd001_data(
    raw_data_dir: Optional[Union[str, Path]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, test, and ground-truth RUL datasets for FD001.

    Args:
        raw_data_dir: Optional custom raw data directory. Defaults to config.RAW_DATA_DIR.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: DataFrames for train, test, and RUL.
    """
    base_dir = Path(raw_data_dir) if raw_data_dir else config.RAW_DATA_DIR

    train_path = base_dir / config.TRAIN_FD001_FILE
    test_path = base_dir / config.TEST_FD001_FILE
    rul_path = base_dir / config.RUL_FD001_FILE

    train_df = load_train_data(train_path)
    test_df = load_test_data(test_path)
    rul_df = load_rul_data(rul_path)

    return train_df, test_df, rul_df
