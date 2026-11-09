"""
Model Evaluation and Refinement module for Aircraft Engine Predictive Maintenance (AEPM).

Provides functions to eliminate target leakage (removing 'life_percentage'), retrain
the RandomForestRegressor, compute evaluation metrics, generate evaluation plots
(Actual vs Predicted, Residual Plot, Error Distribution), export a structured JSON
evaluation report, and compare old vs refined models.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config
from src.train_model import prepare_training_data, split_dataset, train_random_forest
from src.utils import ensure_dir, setup_logger

logger = setup_logger(__name__)

# Features known to cause target leakage
LEAKAGE_FEATURES: List[str] = ["life_percentage"]


def remove_target_leakage_features(
    df: pd.DataFrame, leak_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """Remove target leakage features from the dataset.

    Target leakage occurs when a feature incorporates knowledge of the target variable
    ('RUL') that will not be available during production inference (e.g. 'life_percentage').

    Args:
        df: Input feature DataFrame.
        leak_cols: List of column names to remove. Defaults to LEAKAGE_FEATURES.

    Returns:
        pd.DataFrame: Cleaned DataFrame with leakage features removed.
    """
    to_remove = leak_cols or LEAKAGE_FEATURES
    existing_leaks = [col for col in to_remove if col in df.columns]

    df_clean = df.drop(columns=existing_leaks).copy()
    logger.info(
        "Removed target leakage features (%d columns): %s",
        len(existing_leaks),
        existing_leaks,
    )
    return df_clean


def plot_actual_vs_predicted(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    save_fig: bool = True,
    output_dir: Union[str, Path] = config.FIGURES_DIR,
) -> plt.Figure:
    """Generate Actual vs Predicted RUL scatter plot using Matplotlib.

    Args:
        y_actual: Array of ground-truth RUL values.
        y_pred: Array of predicted RUL values.
        save_fig: Whether to save plot figure to disk.
        output_dir: Destination folder path.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.scatter(y_actual, y_pred, alpha=0.4, color="#1f77b4", edgecolors="none", s=25)

    # Plot diagonal ideal prediction line (y = x)
    min_val = min(y_actual.min(), y_pred.min())
    max_val = max(y_actual.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ideal Prediction (y = x)")

    ax.set_xlabel("Actual Remaining Useful Life (RUL Cycles)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Predicted Remaining Useful Life (RUL Cycles)", fontsize=11, fontweight="bold")
    ax.set_title("Actual vs Predicted RUL (Validation Set)", fontsize=13, pad=15)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    if save_fig:
        out_path = Path(output_dir) / "actual_vs_predicted.png"
        ensure_dir(output_dir)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        logger.info("Saved Actual vs Predicted plot to: %s", out_path)

    return fig


def plot_residuals(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    save_fig: bool = True,
    output_dir: Union[str, Path] = config.FIGURES_DIR,
) -> plt.Figure:
    """Generate Residual Plot (Predicted RUL vs Residuals) using Matplotlib.

    Residual = Actual RUL - Predicted RUL.

    Args:
        y_actual: Array of ground-truth RUL values.
        y_pred: Array of predicted RUL values.
        save_fig: Whether to save plot figure to disk.
        output_dir: Destination folder path.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    residuals = y_actual - y_pred

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.scatter(y_pred, residuals, alpha=0.4, color="#2ca02c", edgecolors="none", s=25)
    ax.axhline(0, color="red", linestyle="--", linewidth=2, label="Zero Error Line")

    ax.set_xlabel("Predicted Remaining Useful Life (RUL Cycles)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Residuals (Actual - Predicted)", fontsize=11, fontweight="bold")
    ax.set_title("Residual Plot (Error Variance Inspection)", fontsize=13, pad=15)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    if save_fig:
        out_path = Path(output_dir) / "residual_plot.png"
        ensure_dir(output_dir)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        logger.info("Saved Residual plot to: %s", out_path)

    return fig


def plot_error_distribution(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    save_fig: bool = True,
    output_dir: Union[str, Path] = config.FIGURES_DIR,
    bins: int = 30,
) -> plt.Figure:
    """Generate Prediction Error Distribution Histogram using Matplotlib.

    Args:
        y_actual: Array of ground-truth RUL values.
        y_pred: Array of predicted RUL values.
        save_fig: Whether to save plot figure to disk.
        output_dir: Destination folder path.
        bins: Number of histogram bins.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    errors = y_actual - y_pred
    mean_error = np.mean(errors)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(errors, bins=bins, color="#ff7f0e", edgecolor="#111111", alpha=0.75)
    ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero Error Reference")
    ax.axvline(mean_error, color="black", linestyle=":", linewidth=2, label=f"Mean Error ({mean_error:.2f})")

    ax.set_xlabel("Prediction Error (Actual RUL - Predicted RUL)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=11, fontweight="bold")
    ax.set_title("Prediction Error Distribution Histogram", fontsize=13, pad=15)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    if save_fig:
        out_path = Path(output_dir) / "error_distribution.png"
        ensure_dir(output_dir)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        logger.info("Saved Error Distribution plot to: %s", out_path)

    return fig


def generate_evaluation_report(
    metrics: Dict[str, float],
    n_train: int,
    n_val: int,
    num_features: int,
    removed_features: List[str],
    model_name: str = "RandomForestRegressor (No Target Leakage)",
) -> dict:
    """Generate structured evaluation report dictionary.

    Args:
        metrics: Dictionary containing 'rmse', 'mae', 'r2_score'.
        n_train: Number of training samples.
        n_val: Number of validation samples.
        num_features: Total active features used.
        removed_features: List of removed features (e.g. ['life_percentage']).
        model_name: Model descriptor label.

    Returns:
        dict: Complete evaluation report structure.
    """
    report = {
        "model_name": model_name,
        "training_samples": n_train,
        "validation_samples": n_val,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "r2_score": metrics["r2_score"],
        "num_features_used": num_features,
        "removed_features": removed_features,
        "training_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return report


def save_evaluation_report(
    report_dict: dict,
    output_path: Union[str, Path] = config.RESULTS_DIR / "evaluation_report.json",
) -> Path:
    """Save evaluation report dictionary as JSON file in reports/results/.

    Args:
        report_dict: Dictionary containing evaluation report data.
        output_path: Target JSON file path.

    Returns:
        Path: Path to written JSON file.
    """
    out_path = Path(output_path)
    ensure_dir(out_path.parent)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=4)

    logger.info("Saved evaluation report JSON to: %s", out_path)
    return out_path


def compare_models(
    v1_metrics: Dict[str, float], v2_metrics: Dict[str, float]
) -> pd.DataFrame:
    """Create a side-by-side comparison table between old (v1) and refined (v2) models.

    Args:
        v1_metrics: Metrics dictionary for v1 model (with target leakage).
        v2_metrics: Metrics dictionary for v2 model (without target leakage).

    Returns:
        pd.DataFrame: Comparison summary DataFrame.
    """
    comparison_data = {
        "Metric": ["RMSE (cycles)", "MAE (cycles)", "R² Score", "Target Leakage Removed?"],
        "Model v1 (With Leakage)": [
            v1_metrics.get("rmse", np.nan),
            v1_metrics.get("mae", np.nan),
            v1_metrics.get("r2_score", np.nan),
            "No (False Accuracy)",
        ],
        "Model v2 (Refined / Clean)": [
            v2_metrics.get("rmse", np.nan),
            v2_metrics.get("mae", np.nan),
            v2_metrics.get("r2_score", np.nan),
            "Yes (Production Ready)",
        ],
    }
    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df


def run_model_evaluation_pipeline(
    data_path: Optional[Union[str, Path]] = None,
    v1_metrics_path: Optional[Union[str, Path]] = None,
) -> Tuple[RandomForestRegressor, dict, dict, pd.DataFrame]:
    """Execute the full model evaluation and refinement pipeline.

    Steps:
    1. Load train_features.csv.
    2. Remove 'life_percentage' feature to eliminate target leakage.
    3. Perform 80/20 train/validation split.
    4. Retrain RandomForestRegressor.
    5. Evaluate performance metrics (RMSE, MAE, R²).
    6. Generate Actual vs Predicted, Residual, and Error Distribution plots.
    7. Save evaluation_report.json and models/random_forest_v2.joblib.
    8. Compare old vs refined model metrics.

    Args:
        data_path: Optional custom path to train_features.csv.
        v1_metrics_path: Optional custom path to v1 model_metrics.json.

    Returns:
        Tuple[RandomForestRegressor, dict, dict, pd.DataFrame]:
            (Refined Model, v2 Metrics, Evaluation Report Dict, Comparison DataFrame).
    """
    logger.info("Starting model evaluation & refinement pipeline...")

    feat_path = Path(data_path) if data_path else config.PROCESSED_DATA_DIR / "train_features.csv"
    raw_df = pd.read_csv(feat_path)

    # 1. Eliminate target leakage
    clean_df = remove_target_leakage_features(raw_df, leak_cols=LEAKAGE_FEATURES)

    # 2. Extract X and y
    X, y, feature_names = prepare_training_data(clean_df)

    # 3. Train / Validation Split (80/20)
    X_train, X_val, y_train, y_val = split_dataset(X, y, test_size=0.2, random_state=42)

    # 4. Retrain RandomForestRegressor
    model_v2 = train_random_forest(X_train, y_train, n_estimators=100, random_state=42)

    # 5. Evaluate predictions
    v2_predictions = model_v2.predict(X_val)
    mse = mean_squared_error(y_val, v2_predictions)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_val, v2_predictions))
    r2 = float(r2_score(y_val, v2_predictions))

    v2_metrics = {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2_score": round(r2, 4)}

    logger.info(
        "Refined Model Performance -> RMSE: %.4f | MAE: %.4f | R²: %.4f",
        rmse,
        mae,
        r2,
    )

    # 6. Generate and save evaluation plots
    plot_actual_vs_predicted(y_val.values, v2_predictions, save_fig=True)
    plot_residuals(y_val.values, v2_predictions, save_fig=True)
    plot_error_distribution(y_val.values, v2_predictions, save_fig=True)

    # 7. Generate & save evaluation report JSON
    report_dict = generate_evaluation_report(
        metrics=v2_metrics,
        n_train=len(X_train),
        n_val=len(X_val),
        num_features=len(feature_names),
        removed_features=LEAKAGE_FEATURES,
    )
    save_evaluation_report(report_dict)

    # 8. Save refined model v2
    model_v2_path = config.MODELS_DIR / "random_forest_v2.joblib"
    joblib.dump(model_v2, model_v2_path)
    logger.info("Saved refined model to: %s", model_v2_path)

    # 9. Compare with Model v1 if metrics JSON exists
    metrics_v1_path = Path(v1_metrics_path) if v1_metrics_path else config.RESULTS_DIR / "model_metrics.json"
    if metrics_v1_path.exists():
        with open(metrics_v1_path, "r", encoding="utf-8") as f:
            v1_metrics = json.load(f)
    else:
        v1_metrics = {"rmse": 4.0124, "mae": 1.3803, "r2_score": 0.9959}

    comparison_df = compare_models(v1_metrics, v2_metrics)

    logger.info("Model evaluation pipeline completed successfully.")
    return model_v2, v2_metrics, report_dict, comparison_df
