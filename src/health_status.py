"""
Engine Health Status Classification module for Aircraft Engine Predictive Maintenance (AEPM).

Classifies predicted Remaining Useful Life (RUL) values into 4 operational health categories:
- Healthy: RUL > 100 cycles
- Normal:  50 < RUL <= 100 cycles
- Warning: 20 < RUL <= 50 cycles
- Critical: RUL <= 20 cycles
"""

import logging
from typing import Dict, Optional, Union
import numpy as np
import pandas as pd

from src.utils import setup_logger

logger = setup_logger(__name__)

# Default configurable RUL health thresholds
DEFAULT_RUL_THRESHOLDS: Dict[str, float] = {
    "healthy_min": 100.0,
    "normal_min": 50.0,
    "warning_min": 20.0,
}


def get_health_status(
    rul: float, thresholds: Optional[Dict[str, float]] = None
) -> str:
    """Classify a single numeric RUL value into a health status label.

    Args:
        rul: Predicted Remaining Useful Life in cycles.
        thresholds: Configurable dictionary with 'healthy_min', 'normal_min', 'warning_min'.

    Returns:
        str: Health status label ('Healthy', 'Normal', 'Warning', 'Critical').
    """
    t = thresholds or DEFAULT_RUL_THRESHOLDS
    h_min = t.get("healthy_min", 100.0)
    n_min = t.get("normal_min", 50.0)
    w_min = t.get("warning_min", 20.0)

    if rul > h_min:
        return "Healthy"
    elif rul > n_min:
        return "Normal"
    elif rul > w_min:
        return "Warning"
    else:
        return "Critical"


def add_health_status_column(
    df: pd.DataFrame,
    rul_col: str = "predicted_rul",
    output_col: str = "health_status",
    thresholds: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Add a vector-classified health status column to a DataFrame.

    Args:
        df: Input DataFrame containing predicted RUL column.
        rul_col: Name of RUL prediction column. Defaults to 'predicted_rul'.
        output_col: Name of output health status column. Defaults to 'health_status'.
        thresholds: Custom threshold dictionary.

    Returns:
        pd.DataFrame: DataFrame with health status column added.
    """
    df = df.copy()
    if rul_col not in df.columns:
        raise KeyError(f"Specified RUL column '{rul_col}' not found in DataFrame.")

    t = thresholds or DEFAULT_RUL_THRESHOLDS
    h_min = t.get("healthy_min", 100.0)
    n_min = t.get("normal_min", 50.0)
    w_min = t.get("warning_min", 20.0)

    ruls = df[rul_col].values

    conditions = [
        ruls > h_min,
        (ruls > n_min) & (ruls <= h_min),
        (ruls > w_min) & (ruls <= n_min),
        ruls <= w_min,
    ]
    choices = ["Healthy", "Normal", "Warning", "Critical"]

    df[output_col] = np.select(conditions, choices, default="Critical")

    logger.info(
        "Classified %d predictions into Health Categories (%s).",
        len(df),
        dict(df[output_col].value_counts()),
    )
    return df


def get_health_summary(
    df: pd.DataFrame, status_col: str = "health_status"
) -> pd.DataFrame:
    """Generate summary count and percentage breakdown of engine health categories.

    Args:
        df: Input DataFrame containing health status column.
        status_col: Column name for health status. Defaults to 'health_status'.

    Returns:
        pd.DataFrame: Summary statistics DataFrame with 'count' and 'percentage'.
    """
    if status_col not in df.columns:
        raise KeyError(f"Health status column '{status_col}' not found in DataFrame.")

    total_count = len(df)
    categories = ["Healthy", "Normal", "Warning", "Critical"]

    summary_list = []
    for cat in categories:
        cnt = int((df[status_col] == cat).sum())
        pct = round((cnt / total_count * 100), 2) if total_count > 0 else 0.0
        summary_list.append({"health_status": cat, "count": cnt, "percentage": pct})

    summary_df = pd.DataFrame(summary_list)
    logger.info("Generated health status breakdown summary.")
    return summary_df
