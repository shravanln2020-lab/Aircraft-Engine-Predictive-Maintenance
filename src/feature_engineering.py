"""
Feature Engineering module for NASA C-MAPSS FD001 dataset.

Provides production-quality modular functions to engineer predictive domain features
including Engine Age, Life Percentage, Sensor Deltas, Rolling Means, Rolling Standard
Deviations, and a normalized Engine Health Index. Exports engineered feature datasets
to data/processed/.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src import config
from src.utils import ensure_dir, setup_logger

logger = setup_logger(__name__)

# Key degradation sensors selected based on EDA variance and correlation analysis
DEFAULT_KEY_SENSORS: List[str] = [
    "s_2", "s_3", "s_4", "s_7", "s_8", "s_9", "s_11", "s_12", "s_13", "s_14", "s_15", "s_17", "s_20", "s_21"
]


def add_engine_age_and_life_percentage(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer Engine Age and Life Percentage features.

    - Engine Age: Represented by current flight cycle count ('time_cycles').
    - Life Percentage: Current Cycle / Total Engine Lifespan (Current Cycle + RUL).

    Args:
        df: Input DataFrame containing 'unit_nr', 'time_cycles', and 'RUL'.

    Returns:
        pd.DataFrame: DataFrame with 'engine_age' and 'life_percentage' columns added.
    """
    df = df.copy()
    df["engine_age"] = df["time_cycles"].astype(float)

    if config.RUL_COL in df.columns:
        total_lifespan = df["time_cycles"] + df[config.RUL_COL]
        # Avoid division by zero
        df["life_percentage"] = np.where(
            total_lifespan > 0, df["time_cycles"] / total_lifespan, 0.0
        )
        df["life_percentage"] = df["life_percentage"].clip(0.0, 1.0)
    else:
        # Fallback using engine max cycle in dataset
        max_cycles = df.groupby("unit_nr")["time_cycles"].transform("max")
        df["life_percentage"] = df["time_cycles"] / max_cycles

    logger.info("Engine Age and Life Percentage features generated.")
    return df


def add_sensor_delta_features(
    df: pd.DataFrame, sensor_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """Engineer Sensor Delta features representing deviation from baseline.

    Formula: Sensor_Delta(t) = Sensor_Value(t) - Initial_Sensor_Value(cycle_1)

    Args:
        df: Input DataFrame grouped by 'unit_nr'.
        sensor_cols: List of target sensor columns. Defaults to DEFAULT_KEY_SENSORS.

    Returns:
        pd.DataFrame: DataFrame with sensor delta features added.
    """
    df = df.copy()
    sensors = [c for c in (sensor_cols or DEFAULT_KEY_SENSORS) if c in df.columns]

    for sensor in sensors:
        initial_val = df.groupby("unit_nr")[sensor].transform("first")
        df[f"{sensor}_delta"] = df[sensor] - initial_val

    logger.info("Generated %d Sensor Delta features.", len(sensors))
    return df


def add_rolling_features(
    df: pd.DataFrame, sensor_cols: Optional[List[str]] = None, window: int = 5
) -> pd.DataFrame:
    """Engineer Rolling Mean and Rolling Standard Deviation features per engine unit.

    Computes moving window statistics over flight cycles per engine to capture
    degradation trends and sensor noise variance.

    Args:
        df: Input DataFrame containing 'unit_nr' and sensor columns.
        sensor_cols: List of target sensor columns. Defaults to DEFAULT_KEY_SENSORS.
        window: Rolling window size in cycles. Defaults to 5.

    Returns:
        pd.DataFrame: DataFrame with rolling mean and rolling std features added.
    """
    df = df.copy()
    sensors = [c for c in (sensor_cols or DEFAULT_KEY_SENSORS) if c in df.columns]

    for sensor in sensors:
        # Rolling Mean
        df[f"{sensor}_roll_mean_{window}"] = df.groupby("unit_nr")[sensor].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        # Rolling Standard Deviation
        df[f"{sensor}_roll_std_{window}"] = (
            df.groupby("unit_nr")[sensor]
            .transform(lambda x: x.rolling(window, min_periods=1).std())
            .fillna(0.0)
        )

    logger.info(
        "Generated %d Rolling Mean and %d Rolling Std features (window=%d).",
        len(sensors),
        len(sensors),
        window,
    )
    return df


def add_health_index(
    df: pd.DataFrame, sensor_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """Engineer a normalized Engine Health Index feature.

    Calculation:
    1. Select key degradation sensor signals.
    2. Normalize each sensor signal to [0, 1] using min-max bounds across the dataset.
    3. Compute average normalized degradation score D across selected sensors.
    4. Health Index = 1.0 - D, clipped to range [0.0, 1.0].
       (1.0 indicates a pristine engine at cycle 1; 0.0 indicates severe degradation).

    Args:
        df: Input DataFrame.
        sensor_cols: List of sensors used to compute health index.

    Returns:
        pd.DataFrame: DataFrame with 'health_index' feature added.
    """
    df = df.copy()
    sensors = [c for c in (sensor_cols or DEFAULT_KEY_SENSORS) if c in df.columns]

    if not sensors:
        df["health_index"] = 1.0
        return df

    # Min-Max normalize selected sensors
    sensor_matrix = df[sensors]
    min_vals = sensor_matrix.min()
    max_vals = sensor_matrix.max()
    denom = np.where((max_vals - min_vals) == 0, 1.0, (max_vals - min_vals))

    normalized_sensors = (sensor_matrix - min_vals) / denom
    degradation_score = normalized_sensors.mean(axis=1)

    # Health index starts near 1.0 and degrades toward 0.0
    df["health_index"] = (1.0 - degradation_score).clip(0.0, 1.0)

    logger.info("Generated normalized Engine Health Index feature.")
    return df


def validate_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Validate engineered features matrix for infinite values or missing NaNs.

    Args:
        df: Input DataFrame with engineered features.

    Returns:
        pd.DataFrame: Cleaned DataFrame verified free of NaNs and Infs.
    """
    df_clean = df.replace([np.inf, -np.inf], np.nan).copy()
    null_count = df_clean.isnull().sum().sum()

    if null_count > 0:
        logger.warning("Found %d missing values after feature engineering. Imputing with median.", null_count)
        df_clean = df_clean.fillna(df_clean.median(numeric_only=True))

    logger.info("Feature validation passed. Total columns: %d", df_clean.shape[1])
    return df_clean


def create_engine_feature_summary(
    train_feat_df: pd.DataFrame, test_feat_df: pd.DataFrame
) -> pd.DataFrame:
    """Generate comparative summary report for engineered features across train and test sets.

    Args:
        train_feat_df: Engineered training DataFrame.
        test_feat_df: Engineered testing DataFrame.

    Returns:
        pd.DataFrame: Summary statistics report for all active features.
    """
    exclude = set(config.INDEX_COLS + [config.RUL_COL])
    feature_cols = [c for c in train_feat_df.columns if c not in exclude]

    summary_rows = []
    for col in feature_cols:
        summary_rows.append(
            {
                "feature_name": col,
                "train_mean": round(train_feat_df[col].mean(), 4),
                "train_std": round(train_feat_df[col].std(), 4),
                "test_mean": round(test_feat_df[col].mean(), 4) if col in test_feat_df.columns else np.nan,
                "test_std": round(test_feat_df[col].std(), 4) if col in test_feat_df.columns else np.nan,
                "null_count": int(train_feat_df[col].isnull().sum()),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    logger.info("Engineered feature summary created for %d features.", len(summary_df))
    return summary_df


def export_engineered_datasets(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Union[str, Path] = config.PROCESSED_DATA_DIR,
) -> Tuple[Path, Path, Path]:
    """Save engineered datasets and feature summary CSV files to data/processed/.

    Args:
        train_features: Processed training set with engineered features.
        test_features: Processed test set with engineered features.
        summary_df: Feature summary DataFrame.
        output_dir: Target directory path.

    Returns:
        Tuple[Path, Path, Path]: Output paths for (train_features.csv, test_features.csv, engine_feature_summary.csv).
    """
    out_dir = ensure_dir(output_dir)

    train_path = out_dir / "train_features.csv"
    test_path = out_dir / "test_features.csv"
    summary_path = out_dir / "engine_feature_summary.csv"

    train_features.to_csv(train_path, index=False)
    test_features.to_csv(test_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    logger.info("Exported train_features.csv to: %s", train_path)
    logger.info("Exported test_features.csv to: %s", test_path)
    logger.info("Exported engine_feature_summary.csv to: %s", summary_path)

    return train_path, test_path, summary_path


def run_feature_engineering_pipeline(
    input_dir: Optional[Union[str, Path]] = None,
    output_dir: Union[str, Path] = config.PROCESSED_DATA_DIR,
    window: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute the full feature engineering pipeline end-to-end.

    Steps:
    1. Read preprocessed train_processed.csv and test_processed.csv datasets.
    2. Engineer Engine Age and Life Percentage.
    3. Engineer Sensor Deltas from initial flight cycle baseline.
    4. Engineer Rolling Mean and Rolling Std features (window size = 5 cycles).
    5. Engineer normalized Engine Health Index.
    6. Validate engineered features matrix.
    7. Generate feature summary report.
    8. Export train_features.csv, test_features.csv, and engine_feature_summary.csv.

    Args:
        input_dir: Directory containing preprocessed CSV files. Defaults to config.PROCESSED_DATA_DIR.
        output_dir: Directory to save engineered feature datasets.
        window: Rolling window size for moving averages and std.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Engineered train, test, and feature summary DataFrames.
    """
    logger.info("Starting feature engineering pipeline...")
    base_dir = Path(input_dir) if input_dir else config.PROCESSED_DATA_DIR

    train_path = base_dir / "train_processed.csv"
    test_path = base_dir / "test_processed.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Preprocessed data files not found in {base_dir}. "
            "Please execute Milestone 3 preprocessing before feature engineering."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    logger.info(
        "Loaded preprocessed train (%d rows, %d cols) and test (%d rows, %d cols).",
        train_df.shape[0],
        train_df.shape[1],
        test_df.shape[0],
        test_df.shape[1],
    )

    # 1. Add Engine Age and Life Percentage
    train_feat = add_engine_age_and_life_percentage(train_df)
    test_feat = add_engine_age_and_life_percentage(test_df)

    # 2. Add Sensor Deltas
    train_feat = add_sensor_delta_features(train_feat)
    test_feat = add_sensor_delta_features(test_feat)

    # 3. Add Rolling Mean & Std features
    train_feat = add_rolling_features(train_feat, window=window)
    test_feat = add_rolling_features(test_feat, window=window)

    # 4. Add Engine Health Index
    train_feat = add_health_index(train_feat)
    test_feat = add_health_index(test_feat)

    # 5. Validate engineered matrices
    train_feat = validate_engineered_features(train_feat)
    test_feat = validate_engineered_features(test_feat)

    # 6. Generate feature summary
    summary_df = create_engine_feature_summary(train_feat, test_feat)

    # 7. Export outputs
    export_engineered_datasets(train_feat, test_feat, summary_df, output_dir=output_dir)

    logger.info("Feature engineering pipeline completed successfully.")
    return train_feat, test_feat, summary_df
