"""
Comprehensive test suite for Aircraft Engine Predictive Maintenance (AEPM).
Covers validation, preprocessing, feature engineering, target leakage removal, RUL prediction,
health classification, malformed input handling, and dashboard utilities.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils import df_to_csv_bytes, load_json_safe, status_badge
from src import config
from src.feature_engineering import (
    add_engine_age_and_life_percentage,
    add_health_index,
    add_rolling_features,
    add_sensor_delta_features,
)
from src.health_status import add_health_status_column, get_health_status, get_health_summary
from src.inference import prepare_inference_features, run_inference_pipeline
from src.predictor import RULPredictor, load_model, predict_rul
from src.validator import (
    validate_input_dataframe,
    validate_input_schema,
    validate_missing_values,
    validate_numeric_types,
)


# ── Sample Data Fixtures ──────────────────────────────────────────

def _create_sample_raw_df(n_rows: int = 20) -> pd.DataFrame:
    """Helper to generate a valid raw C-MAPSS telemetry DataFrame."""
    data = {}
    for col in config.COLUMN_NAMES:
        if col == "unit_nr":
            data[col] = np.repeat([1, 2], n_rows // 2)
        elif col == "time_cycles":
            data[col] = np.tile(np.arange(1, (n_rows // 2) + 1), 2)
        elif col == "setting_3":
            data[col] = 100.0  # Constant setting in FD001
        else:
            data[col] = np.random.uniform(10.0, 500.0, size=n_rows)
    return pd.DataFrame(data)


# ── 1. Schema & Data Type Validation Tests ────────────────────────

def test_input_schema_validation():
    """Test schema validation with complete and missing column DataFrames."""
    df_valid = _create_sample_raw_df()
    assert validate_input_schema(df_valid) is True

    # Drop required column
    df_invalid = df_valid.drop(columns=["s_1"])
    try:
        validate_input_schema(df_invalid)
        assert False, "Should have raised ValueError for missing column 's_1'"
    except ValueError as err:
        assert "missing required columns" in str(err).lower()


def test_missing_values_validation():
    """Test missing value validation threshold logic."""
    df = _create_sample_raw_df()
    assert validate_missing_values(df, max_missing_pct=0.0) is True

    # Inject missing value
    df.loc[0, "s_2"] = np.nan
    try:
        validate_missing_values(df, max_missing_pct=0.0)
        assert False, "Should have raised ValueError for missing values"
    except ValueError as err:
        assert "missing values" in str(err).lower()

    # Allowing missing values in validation suite
    assert validate_input_dataframe(df, allow_missing=True) is True


def test_numeric_types_validation():
    """Test numeric data type validation against non-numeric entries."""
    df = _create_sample_raw_df()
    assert validate_numeric_types(df) is True

    # Inject string entry
    df["s_5"] = df["s_5"].astype(str)
    df.loc[0, "s_5"] = "INVALID_STRING"
    try:
        validate_numeric_types(df)
        assert False, "Should have raised ValueError for non-numeric column"
    except ValueError as err:
        assert "non-numeric" in str(err).lower()


def test_malformed_and_empty_input_handling():
    """Test graceful error handling for empty or malformed input DataFrames."""
    empty_df = pd.DataFrame()
    try:
        validate_input_dataframe(empty_df)
        assert False, "Should have raised ValueError for empty DataFrame"
    except ValueError as err:
        assert "empty" in str(err).lower()

    try:
        prepare_inference_features(empty_df)
        assert False, "Should have raised ValueError on empty feature preparation"
    except ValueError as err:
        assert "empty" in str(err).lower()


# ── 2. Feature Engineering & Target Leakage Removal Tests ─────────

def test_feature_preparation_and_leakage_removal():
    """Test feature engineering steps and verify life_percentage is excluded."""
    raw_df = _create_sample_raw_df(n_rows=20)
    feat_df = prepare_inference_features(raw_df)

    # Check key engineered feature blocks exist
    assert "engine_age" in feat_df.columns
    assert "s_2_delta" in feat_df.columns
    assert "s_2_roll_mean_5" in feat_df.columns
    assert "health_index" in feat_df.columns

    # CRITICAL: Verify target leakage column is NOT present in inference feature matrix
    assert "life_percentage" not in feat_df.columns, "TARGET LEAKAGE DETECTED: 'life_percentage' present in feature matrix!"


# ── 3. RUL Prediction & Model Verification Tests ─────────────────

def test_rul_prediction_validity():
    """Test RUL prediction finiteness, non-negativity, and feature ordering using random_forest_v2.joblib."""
    model_path = config.MODELS_DIR / "random_forest_v2.joblib"
    assert model_path.exists(), f"Production model missing at {model_path}"

    predictor = RULPredictor(model_path=model_path)
    assert hasattr(predictor.model, "feature_names_in_")

    # Verify life_percentage is not in trained model features
    model_features = list(predictor.model.feature_names_in_)
    assert "life_percentage" not in model_features, "TARGET LEAKAGE DETECTED in random_forest_v2.joblib!"
    assert len(model_features) == 67, f"Expected 67 model features, got {len(model_features)}"

    # Generate inference feature matrix
    raw_df = _create_sample_raw_df(n_rows=20)
    feat_df = prepare_inference_features(raw_df)

    predictions = predictor.predict(feat_df)

    # 1. Prediction count matches input
    assert len(predictions) == len(raw_df)

    # 2. All predictions are finite numbers
    assert np.isfinite(predictions).all(), "Predictions contain NaN or Infinite values!"

    # 3. All predictions are non-negative
    assert (predictions >= 0.0).all(), "Predictions contain negative RUL values!"


# ── 4. Health Status Classification Tests ─────────────────────────

def test_health_status_classification():
    """Test health status threshold boundaries."""
    assert get_health_status(120.0) == "Healthy"
    assert get_health_status(80.0) == "Normal"
    assert get_health_status(35.0) == "Warning"
    assert get_health_status(10.0) == "Critical"
    assert get_health_status(0.0) == "Critical"

    # Test vector classification on DataFrame
    df = pd.DataFrame({"predicted_rul": [150.0, 75.0, 30.0, 10.0]})
    res = add_health_status_column(df, rul_col="predicted_rul", output_col="health_status")
    expected = ["Healthy", "Normal", "Warning", "Critical"]
    assert res["health_status"].tolist() == expected

    summary = get_health_summary(res)
    assert len(summary) == 4
    assert summary.loc[summary["health_status"] == "Healthy", "count"].values[0] == 1


# ── 5. End-to-End Prediction Pipeline Test ────────────────────────

def test_end_to_end_inference_pipeline():
    """Test execution of full inference pipeline on raw telemetry input."""
    raw_df = _create_sample_raw_df(n_rows=20)
    model_path = config.MODELS_DIR / "random_forest_v2.joblib"

    results_df = run_inference_pipeline(raw_df, model_path=model_path)

    # Check output schema
    required_out_cols = ["unit_nr", "time_cycles", "predicted_rul", "health_status"]
    for col in required_out_cols:
        assert col in results_df.columns

    # Verify no NaN values in predictions or status
    assert results_df["predicted_rul"].isnull().sum() == 0
    assert results_df["health_status"].isnull().sum() == 0
    assert (results_df["predicted_rul"] >= 0.0).all()


# ── 6. Dashboard Utility Tests ────────────────────────────────────

def test_dashboard_utils():
    """Test dashboard CSS status badges and CSV download serialization."""
    badge = status_badge("Healthy")
    assert "badge-healthy" in badge
    assert "Healthy" in badge

    df = pd.DataFrame({"unit_nr": [1], "predicted_rul": [100.0]})
    csv_bytes = df_to_csv_bytes(df)
    assert isinstance(csv_bytes, bytes)
    assert b"unit_nr,predicted_rul" in csv_bytes
    assert b"1,100.0" in csv_bytes


if __name__ == "__main__":
    test_input_schema_validation()
    test_missing_values_validation()
    test_numeric_types_validation()
    test_malformed_and_empty_input_handling()
    test_feature_preparation_and_leakage_removal()
    test_rul_prediction_validity()
    test_health_status_classification()
    test_end_to_end_inference_pipeline()
    test_dashboard_utils()
    print("[SUCCESS] All pipeline integration tests completed successfully!")
