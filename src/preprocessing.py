"""
Data Preprocessing module for NASA C-MAPSS FD001 dataset.

Provides production-quality modular functions to calculate Remaining Useful Life (RUL),
identify and drop uninformative/constant features, clean duplicates and missing values,
and export processed datasets to data/processed/.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union
import pandas as pd

from src import config
from src.data_loader import load_all_fd001_data
from src.utils import ensure_dir, setup_logger

logger = setup_logger(__name__)


def calculate_train_rul(train_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Remaining Useful Life (RUL) for every training sample.

    Formula: RUL = Engine Max Cycle - Current Cycle.

    Args:
        train_df: Training DataFrame containing 'unit_nr' and 'time_cycles'.

    Returns:
        pd.DataFrame: Training DataFrame with 'RUL' target column added.
    """
    df = train_df.copy()
    max_cycles = df.groupby("unit_nr")["time_cycles"].transform("max")
    df[config.RUL_COL] = max_cycles - df["time_cycles"]

    logger.info(
        "Calculated training RUL (Range: %d to %d cycles).",
        df[config.RUL_COL].min(),
        df[config.RUL_COL].max(),
    )
    return df


def calculate_test_rul(test_df: pd.DataFrame, rul_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Remaining Useful Life (RUL) for every test set sample.

    Formula: RUL(t) = Ground Truth RUL + (Engine Max Test Cycle - Current Cycle t).

    Args:
        test_df: Test DataFrame containing 'unit_nr' and 'time_cycles'.
        rul_df: Ground truth RUL DataFrame containing 'unit_nr' and 'RUL'.

    Returns:
        pd.DataFrame: Test DataFrame with 'RUL' column added.
    """
    df = test_df.copy()

    # Find the maximum flight cycle recorded for each engine in the test set
    max_test_cycles = (
        df.groupby("unit_nr")["time_cycles"]
        .max()
        .reset_index()
        .rename(columns={"time_cycles": "max_test_cycle"})
    )

    # Merge max test cycle and ground truth RUL
    rul_info = pd.merge(max_test_cycles, rul_df, on="unit_nr", how="inner")

    # Merge RUL info back into the test DataFrame
    df = pd.merge(df, rul_info, on="unit_nr", how="left")

    # Calculate RUL at each time cycle: Ground Truth RUL + (max_test_cycle - current_cycle)
    df[config.RUL_COL] = df[config.RUL_COL] + (df["max_test_cycle"] - df["time_cycles"])

    # Drop temporary calculation column
    df = df.drop(columns=["max_test_cycle"])

    logger.info(
        "Calculated test RUL (Range: %d to %d cycles).",
        df[config.RUL_COL].min(),
        df[config.RUL_COL].max(),
    )
    return df


def identify_constant_columns(
    df: pd.DataFrame,
    threshold: float = 0.0,
    exclude_cols: Optional[List[str]] = None,
) -> List[str]:
    """Identify columns with zero variance or constant values.

    Args:
        df: Input DataFrame.
        threshold: Standard deviation threshold below which columns are considered constant.
        exclude_cols: List of column names to exclude from removal (e.g. index/target cols).

    Returns:
        List[str]: Names of constant or uninformative columns.
    """
    exclude = set(exclude_cols or [])
    numeric_cols = [c for c in df.select_dtypes(include=["number"]).columns if c not in exclude]

    stds = df[numeric_cols].std(numeric_only=True)
    constant_cols = stds[stds <= threshold].index.tolist()

    logger.info(
        "Identified %d constant/uninformative columns (std <= %.4f): %s",
        len(constant_cols),
        threshold,
        constant_cols,
    )
    return constant_cols


def remove_uninformative_columns(
    df: pd.DataFrame, drop_cols: List[str]
) -> pd.DataFrame:
    """Remove specified uninformative columns from a DataFrame.

    Args:
        df: Input DataFrame.
        drop_cols: List of column names to drop.

    Returns:
        pd.DataFrame: Cleaned DataFrame with specified columns removed.
    """
    existing_drop_cols = [col for col in drop_cols if col in df.columns]
    cleaned_df = df.drop(columns=existing_drop_cols)
    logger.info(
        "Removed %d uninformative columns. Remaining columns: %d",
        len(existing_drop_cols),
        cleaned_df.shape[1],
    )
    return cleaned_df


def clean_missing_and_duplicates(
    df: pd.DataFrame, dataset_name: str = "Dataset"
) -> pd.DataFrame:
    """Verify and clean duplicate rows and missing values in dataset.

    Args:
        df: Input DataFrame.
        dataset_name: Dataset name for logging.

    Returns:
        pd.DataFrame: Cleaned DataFrame free of duplicates and verified missing values.
    """
    initial_rows = len(df)

    # Remove duplicate rows if any exist
    df_cleaned = df.drop_duplicates().copy()
    duplicates_removed = initial_rows - len(df_cleaned)

    # Check missing values
    missing_count = df_cleaned.isnull().sum().sum()

    logger.info(
        "[%s] Duplicates removed: %d | Missing values remaining: %d | Total rows: %d",
        dataset_name,
        duplicates_removed,
        missing_count,
        len(df_cleaned),
    )

    if missing_count > 0:
        logger.warning(
            "[%s] Missing values detected. Filling nulls with median values.",
            dataset_name,
        )
        df_cleaned = df_cleaned.fillna(df_cleaned.median(numeric_only=True))

    return df_cleaned


def separate_features_and_target(
    df: pd.DataFrame,
    target_col: str = config.RUL_COL,
    index_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Separate feature matrix, target vector, and feature column names.

    Args:
        df: Input DataFrame containing features and target.
        target_col: Name of target column (defaults to 'RUL').
        index_cols: Column names to exclude from feature matrix (defaults to config.INDEX_COLS).

    Returns:
        Tuple[pd.DataFrame, pd.Series, List[str]]: Feature DataFrame, Target Series, Feature names list.
    """
    excluded = set(index_cols or config.INDEX_COLS)
    excluded.add(target_col)

    feature_cols = [c for c in df.columns if c not in excluded]
    features_df = df[feature_cols].copy()
    target_series = df[target_col].copy() if target_col in df.columns else pd.Series(dtype=float)

    return features_df, target_series, feature_cols


def export_processed_data(
    train_processed: pd.DataFrame,
    test_processed: pd.DataFrame,
    feature_cols: List[str],
    output_dir: Union[str, Path] = config.PROCESSED_DATA_DIR,
) -> Tuple[Path, Path, Path]:
    """Export processed datasets and feature list to CSV files in data/processed/.

    Args:
        train_processed: Preprocessed training DataFrame.
        test_processed: Preprocessed test DataFrame.
        feature_cols: List of remaining feature names.
        output_dir: Directory path for processed outputs.

    Returns:
        Tuple[Path, Path, Path]: Output paths for (train_processed.csv, test_processed.csv, feature_list.csv).
    """
    out_dir = ensure_dir(output_dir)

    train_path = out_dir / "train_processed.csv"
    test_path = out_dir / "test_processed.csv"
    feature_list_path = out_dir / "feature_list.csv"

    train_processed.to_csv(train_path, index=False)
    test_processed.to_csv(test_path, index=False)

    feature_df = pd.DataFrame({"feature_name": feature_cols})
    feature_df.to_csv(feature_list_path, index=False)

    logger.info("Exported train_processed.csv to: %s", train_path)
    logger.info("Exported test_processed.csv to: %s", test_path)
    logger.info("Exported feature_list.csv to: %s", feature_list_path)

    return train_path, test_path, feature_list_path


def run_preprocessing_pipeline(
    raw_data_dir: Optional[Union[str, Path]] = None,
    output_dir: Union[str, Path] = config.PROCESSED_DATA_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Execute the complete data preprocessing pipeline from raw text files to processed CSVs.

    Steps:
    1. Load raw train, test, and ground-truth RUL datasets.
    2. Calculate RUL target column for training set and test set.
    3. Identify zero-variance (constant) features in the training set.
    4. Remove constant features from both train and test sets.
    5. Clean duplicates and verify missing values.
    6. Export train_processed.csv, test_processed.csv, and feature_list.csv.

    Args:
        raw_data_dir: Directory path for raw data text files.
        output_dir: Directory path for saving processed CSV files.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, List[str]]: Processed train_df, test_df, and feature column names.
    """
    logger.info("Starting preprocessing pipeline for NASA C-MAPSS FD001 dataset...")

    # 1. Load raw datasets
    train_raw, test_raw, rul_raw = load_all_fd001_data(raw_data_dir)

    # 2. Calculate RUL for train and test sets
    train_rul = calculate_train_rul(train_raw)
    test_rul = calculate_test_rul(test_raw, rul_raw)

    # 3. Identify constant columns from training set (excluding index and target columns)
    exclude_cols = config.INDEX_COLS + [config.RUL_COL]
    constant_cols = identify_constant_columns(train_rul, threshold=0.0, exclude_cols=exclude_cols)

    # 4. Remove uninformative constant columns
    train_cleaned = remove_uninformative_columns(train_rul, constant_cols)
    test_cleaned = remove_uninformative_columns(test_rul, constant_cols)

    # 5. Clean missing values and duplicates
    train_processed = clean_missing_and_duplicates(train_cleaned, dataset_name="TrainProcessed")
    test_processed = clean_missing_and_duplicates(test_cleaned, dataset_name="TestProcessed")

    # 6. Extract remaining feature column list
    _, _, feature_cols = separate_features_and_target(train_processed)

    # 7. Export processed files to data/processed/
    export_processed_data(train_processed, test_processed, feature_cols, output_dir=output_dir)

    logger.info("Preprocessing pipeline completed successfully.")
    return train_processed, test_processed, feature_cols
