"""
End-to-End Inference Pipeline module for Aircraft Engine Predictive Maintenance (AEPM).

Integrates telemetry validation, preprocessing, feature engineering, model prediction,
and engine health status classification into a single unified execution pipeline.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union
import pandas as pd

from src import config
from src.evaluate_model import remove_target_leakage_features
from src.feature_engineering import (
    add_engine_age_and_life_percentage,
    add_health_index,
    add_rolling_features,
    add_sensor_delta_features,
    validate_engineered_features,
)
from src.health_status import add_health_status_column, get_health_summary
from src.predictor import RULPredictor
from src.preprocessing import clean_missing_and_duplicates, remove_uninformative_columns
from src.utils import ensure_dir, setup_logger
from src.validator import validate_input_dataframe

logger = setup_logger(__name__)


def prepare_inference_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw engine telemetry DataFrame into model-ready feature matrix.

    Pipeline Steps:
    1. Validate input schema and data types.
    2. Clean duplicates and missing values.
    3. Remove constant/uninformative columns identified in training.
    4. Engineer Engine Age, Sensor Deltas, 5-cycle Rolling Stats, and Health Index.
    5. Remove target leakage column ('life_percentage').
    6. Validate feature matrix integrity.

    Args:
        raw_df: Raw engine telemetry DataFrame.

    Returns:
        pd.DataFrame: Fully engineered, non-leaking feature matrix.
    """
    # 1. Input Validation
    validate_input_dataframe(raw_df, allow_missing=True)

    # 2. Clean duplicates and nulls
    cleaned_df = clean_missing_and_duplicates(raw_df, dataset_name="InferenceInput")

    # 3. Remove constant columns identified in configuration
    constant_cols = ["setting_3"]  # Constant in FD001 dataset
    cleaned_df = remove_uninformative_columns(cleaned_df, constant_cols)

    # 4. Feature Engineering
    feat_df = add_engine_age_and_life_percentage(cleaned_df)
    feat_df = add_sensor_delta_features(feat_df)
    feat_df = add_rolling_features(feat_df, window=5)
    feat_df = add_health_index(feat_df)

    # 5. Eliminate Target Leakage ('life_percentage')
    feat_df = remove_target_leakage_features(feat_df, leak_cols=["life_percentage"])

    # 6. Validate engineered matrix
    feat_df = validate_engineered_features(feat_df)

    logger.info("Inference feature engineering complete (%d total columns).", feat_df.shape[1])
    return feat_df


def run_inference_pipeline(
    raw_df: pd.DataFrame,
    model_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Execute complete prediction pipeline on telemetry DataFrame.

    Steps:
    1. Prepare feature matrix from raw input.
    2. Load trained RUL Predictor.
    3. Generate RUL predictions ('predicted_rul').
    4. Classify engine health status ('health_status').

    Args:
        raw_df: Raw engine telemetry DataFrame.
        model_path: Optional custom path to model joblib file.

    Returns:
        pd.DataFrame: Output DataFrame containing unit_nr, time_cycles, predicted_rul, and health_status.
    """
    logger.info("Starting inference pipeline execution...")

    # Preserve unit_nr and time_cycles metadata
    metadata_cols = ["unit_nr", "time_cycles"]
    meta_df = raw_df[metadata_cols].copy()

    # 1. Feature Preparation
    feat_df = prepare_inference_features(raw_df)

    # 2. Model Loading & Prediction
    predictor = RULPredictor(model_path=model_path)
    predictions = predictor.predict(feat_df)

    # 3. Combine Metadata with Predictions
    results_df = meta_df.copy()
    results_df["predicted_rul"] = predictions.round(2)

    # 4. Classify Health Status
    results_df = add_health_status_column(results_df, rul_col="predicted_rul", output_col="health_status")

    logger.info("Inference pipeline completed successfully.")
    return results_df


def run_inference_on_file(
    input_file_path: Union[str, Path],
    output_file_path: Optional[Union[str, Path]] = None,
    model_path: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute prediction pipeline on an input data file and export sample predictions CSV.

    Args:
        input_file_path: Path to input dataset file (CSV or space-delimited text).
        output_file_path: Output CSV file path. Defaults to reports/results/sample_predictions.csv.
        model_path: Optional custom path to model file.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Prediction results DataFrame and health summary DataFrame.
    """
    in_path = Path(input_file_path)
    if not in_path.is_file():
        raise FileNotFoundError(f"Input telemetry file not found at '{in_path}'.")

    logger.info("Loading input data from file: %s", in_path)

    # Read raw text or CSV file
    if in_path.suffix == ".csv":
        raw_df = pd.read_csv(in_path)
    else:
        raw_df = pd.read_csv(in_path, sep=r"\s+", header=None, names=config.COLUMN_NAMES, engine="python")

    # Execute inference pipeline
    results_df = run_inference_pipeline(raw_df, model_path=model_path)
    health_summary = get_health_summary(results_df)

    # Export results
    out_path = Path(output_file_path) if output_file_path else config.RESULTS_DIR / "sample_predictions.csv"
    ensure_dir(out_path.parent)
    results_df.to_csv(out_path, index=False)
    logger.info("Exported prediction results to: %s", out_path)

    return results_df, health_summary
