"""
Input Data Validation module for Aircraft Engine Predictive Maintenance (AEPM).

Provides schema, data type, and missing value validation functions to verify raw and
preprocessed engine sensor telemetry prior to model inference.
"""

import logging
from typing import List, Optional, Union
import pandas as pd

from src import config
from src.utils import setup_logger

logger = setup_logger(__name__)

# Minimum required columns for raw C-MAPSS telemetry input
REQUIRED_INPUT_COLS: List[str] = config.COLUMN_NAMES


def validate_input_schema(
    df: pd.DataFrame, required_cols: Optional[List[str]] = None
) -> bool:
    """Verify that input DataFrame contains all required telemetry columns.

    Args:
        df: Input telemetry DataFrame.
        required_cols: List of required column names. Defaults to REQUIRED_INPUT_COLS.

    Returns:
        bool: True if schema validation succeeds.

    Raises:
        ValueError: If required columns are missing.
    """
    req_cols = required_cols or REQUIRED_INPUT_COLS
    missing = [c for c in req_cols if c not in df.columns]

    if missing:
        logger.error("Schema validation failed. Missing required columns (%d): %s", len(missing), missing)
        raise ValueError(
            f"Input DataFrame is missing required columns ({len(missing)}): {missing}"
        )

    logger.info("Schema validation passed (%d required columns present).", len(req_cols))
    return True


def validate_missing_values(
    df: pd.DataFrame, max_missing_pct: float = 0.0
) -> bool:
    """Verify missing/null values in input DataFrame.

    Args:
        df: Input DataFrame.
        max_missing_pct: Maximum allowable percentage of missing values (0.0 to 1.0).

    Returns:
        bool: True if missing values are within allowed threshold.

    Raises:
        ValueError: If missing values exceed the allowed percentage.
    """
    total_cells = df.size
    null_cells = df.isnull().sum().sum()
    missing_pct = null_cells / total_cells if total_cells > 0 else 0.0

    if missing_pct > max_missing_pct:
        logger.error(
            "Missing values validation failed: %.2f%% null cells (threshold: %.2f%%).",
            missing_pct * 100,
            max_missing_pct * 100,
        )
        raise ValueError(
            f"Input dataset contains {null_cells} missing values ({missing_pct * 100:.2f}%), "
            f"exceeding threshold of {max_missing_pct * 100:.2f}%."
        )

    logger.info("Missing values validation passed (%d null cells found).", null_cells)
    return True


def validate_numeric_types(df: pd.DataFrame) -> bool:
    """Verify that operational settings and sensor measurements are numeric.

    Args:
        df: Input DataFrame.

    Returns:
        bool: True if numeric type validation succeeds.

    Raises:
        ValueError: If non-numeric columns are detected among sensor features.
    """
    non_numeric_cols = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric_cols.append(col)

    if non_numeric_cols:
        logger.error("Numeric validation failed. Non-numeric columns found: %s", non_numeric_cols)
        raise ValueError(
            f"Input DataFrame contains non-numeric telemetry columns: {non_numeric_cols}"
        )

    logger.info("Numeric type validation passed for all %d columns.", df.shape[1])
    return True


def validate_input_dataframe(
    df: pd.DataFrame,
    required_cols: Optional[List[str]] = None,
    allow_missing: bool = False,
) -> bool:
    """Run comprehensive validation suite on input telemetry DataFrame.

    Checks:
    1. Empty DataFrame check.
    2. Required schema columns check.
    3. Missing values check.
    4. Numeric data types check.

    Args:
        df: Input DataFrame.
        required_cols: List of required columns. Defaults to REQUIRED_INPUT_COLS.
        allow_missing: Whether to allow missing values. Defaults to False.

    Returns:
        bool: True if all validation checks pass.
    """
    logger.info("Starting input telemetry validation...")

    if df.empty:
        logger.error("Input validation failed: DataFrame is empty.")
        raise ValueError("Input dataset is empty.")

    # 1. Schema check
    validate_input_schema(df, required_cols=required_cols)

    # 2. Missing values check
    if not allow_missing:
        validate_missing_values(df, max_missing_pct=0.0)

    # 3. Data type check
    validate_numeric_types(df)

    logger.info("Input telemetry validation completed successfully.")
    return True
