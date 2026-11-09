"""
Model Training module for Aircraft Engine Predictive Maintenance (AEPM).

Provides modular production-quality functions to train a RandomForestRegressor on
engineered C-MAPSS feature datasets, evaluate performance metrics (RMSE, MAE, R²),
extract feature importances, and save model artifacts and report results.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src import config
from src.utils import ensure_dir, setup_logger

logger = setup_logger(__name__)


def prepare_training_data(
    df: pd.DataFrame,
    target_col: str = config.RUL_COL,
    index_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Extract feature matrix X, target vector y, and feature names list.

    Args:
        df: Input DataFrame containing engineered features and target.
        target_col: Target column name (defaults to 'RUL').
        index_cols: Identifiers to exclude from features (defaults to config.INDEX_COLS).

    Returns:
        Tuple[pd.DataFrame, pd.Series, List[str]]: Feature matrix X, target vector y, feature names.
    """
    excluded = set(index_cols or config.INDEX_COLS)
    excluded.add(target_col)

    feature_cols = [col for col in df.columns if col not in excluded]
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    logger.info("Extracted %d feature columns and target '%s'.", len(feature_cols), target_col)
    return X, y, feature_cols


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataset into training and validation sets.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        test_size: Fraction of samples for validation. Defaults to 0.2 (80/20 split).
        random_state: Random seed for reproducible splits. Defaults to 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: X_train, X_val, y_train, y_val.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(
        "Split data into Train (%d samples) and Validation (%d samples).",
        X_train.shape[0],
        X_val.shape[0],
    )
    return X_train, X_val, y_train, y_val


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomForestRegressor:
    """Train a RandomForestRegressor model.

    Args:
        X_train: Training feature matrix.
        y_train: Training target labels.
        n_estimators: Number of decision trees. Defaults to 100.
        max_depth: Maximum tree depth. Defaults to None.
        random_state: Seed for reproducibility. Defaults to 42.
        n_jobs: Number of parallel jobs. Defaults to -1 (all CPU cores).

    Returns:
        RandomForestRegressor: Trained scikit-learn model object.
    """
    logger.info(
        "Training RandomForestRegressor (n_estimators=%d, random_state=%d)...",
        n_estimators,
        random_state,
    )
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(X_train, y_train)

    logger.info("RandomForestRegressor training complete.")
    return model


def evaluate_model(
    model: RandomForestRegressor, X_val: pd.DataFrame, y_val: pd.Series
) -> Tuple[dict, np.ndarray]:
    """Evaluate model performance on validation data and compute metrics.

    Metrics calculated:
    - Root Mean Squared Error (RMSE)
    - Mean Absolute Error (MAE)
    - R² Score

    Args:
        model: Trained regression model.
        X_val: Validation feature matrix.
        y_val: Ground truth validation target values.

    Returns:
        Tuple[dict, np.ndarray]: Metrics dictionary and predictions array.
    """
    predictions = model.predict(X_val)

    mse = mean_squared_error(y_val, predictions)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_val, predictions))
    r2 = float(r2_score(y_val, predictions))

    metrics = {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2_score": round(r2, 4),
    }

    logger.info("Validation Performance Metrics -> RMSE: %.4f | MAE: %.4f | R²: %.4f", rmse, mae, r2)
    return metrics, predictions


def extract_feature_importance(
    model: RandomForestRegressor, feature_names: List[str]
) -> pd.DataFrame:
    """Extract and rank feature importances from trained Random Forest model.

    Args:
        model: Trained RandomForestRegressor.
        feature_names: List of feature names corresponding to X columns.

    Returns:
        pd.DataFrame: DataFrame with 'feature_name' and 'importance' columns sorted descending.
    """
    importances = model.feature_importances_
    importance_df = pd.DataFrame(
        {"feature_name": feature_names, "importance": importances}
    ).sort_values(by="importance", ascending=False).reset_index(drop=True)

    logger.info("Extracted feature importances for %d features.", len(importance_df))
    return importance_df


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    save_fig: bool = True,
    output_dir: Union[str, Path] = config.FIGURES_DIR,
) -> plt.Figure:
    """Plot horizontal bar chart of top N important features using Matplotlib.

    Args:
        importance_df: DataFrame containing 'feature_name' and 'importance'.
        top_n: Number of top features to plot. Defaults to 15.
        save_fig: Whether to export figure to disk.
        output_dir: Output folder for figure.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    top_features = importance_df.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top_features["feature_name"], top_features["importance"], color="#2b5c8f", edgecolor="#111111")
    ax.set_xlabel("Relative Feature Importance", fontsize=11, fontweight="bold")
    ax.set_ylabel("Feature Name", fontsize=11, fontweight="bold")
    ax.set_title(f"Random Forest Regressor - Top {top_n} Feature Importances", fontsize=14, pad=15)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    if save_fig:
        out_path = Path(output_dir) / "feature_importance.png"
        ensure_dir(output_dir)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        logger.info("Feature importance plot saved to: %s", out_path)

    return fig


def save_model(
    model: RandomForestRegressor,
    output_path: Union[str, Path] = config.MODELS_DIR / "random_forest_v1.joblib",
) -> Path:
    """Save trained model to disk using joblib.

    Args:
        model: Trained model object.
        output_path: Target file path. Defaults to models/random_forest_v1.joblib.

    Returns:
        Path: Path to saved model file.
    """
    out_path = Path(output_path)
    ensure_dir(out_path.parent)
    joblib.dump(model, out_path)
    logger.info("Trained model saved to: %s", out_path)
    return out_path


def save_training_artifacts(
    metrics: dict,
    importance_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    output_dir: Union[str, Path] = config.RESULTS_DIR,
) -> Tuple[Path, Path, Path]:
    """Save training results including metrics JSON, feature importance CSV, and validation predictions CSV.

    Args:
        metrics: Dictionary of evaluation metrics.
        importance_df: Feature importance DataFrame.
        predictions_df: Validation predictions DataFrame.
        output_dir: Output directory for report results.

    Returns:
        Tuple[Path, Path, Path]: Output paths for (model_metrics.json, feature_importance.csv, training_predictions.csv).
    """
    out_dir = ensure_dir(output_dir)

    metrics_path = out_dir / "model_metrics.json"
    importance_path = out_dir / "feature_importance.csv"
    predictions_path = out_dir / "training_predictions.csv"

    # Save metrics JSON
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    # Save feature importance CSV
    importance_df.to_csv(importance_path, index=False)

    # Save validation predictions CSV
    predictions_df.to_csv(predictions_path, index=False)

    logger.info("Saved model metrics to: %s", metrics_path)
    logger.info("Saved feature importances to: %s", importance_path)
    logger.info("Saved validation predictions to: %s", predictions_path)

    return metrics_path, importance_path, predictions_path


def run_training_pipeline(
    data_path: Optional[Union[str, Path]] = None,
    model_dir: Union[str, Path] = config.MODELS_DIR,
    results_dir: Union[str, Path] = config.RESULTS_DIR,
    figures_dir: Union[str, Path] = config.FIGURES_DIR,
) -> Tuple[RandomForestRegressor, dict, pd.DataFrame]:
    """Execute the complete model training pipeline.

    Steps:
    1. Load train_features.csv.
    2. Separate feature matrix X and target y.
    3. Split into 80% training and 20% validation sets.
    4. Train RandomForestRegressor model.
    5. Evaluate validation metrics (RMSE, MAE, R²).
    6. Extract feature importances and generate bar chart.
    7. Save model to models/random_forest_v1.joblib.
    8. Save report artifacts (model_metrics.json, feature_importance.csv, training_predictions.csv).

    Args:
        data_path: Path to train_features.csv. Defaults to config.PROCESSED_DATA_DIR / train_features.csv.
        model_dir: Directory to save model joblib.
        results_dir: Directory to save CSV/JSON report artifacts.
        figures_dir: Directory to save PNG figures.

    Returns:
        Tuple[RandomForestRegressor, dict, pd.DataFrame]: Trained model, metrics dict, feature importance DataFrame.
    """
    logger.info("Starting model training pipeline...")

    feature_file = Path(data_path) if data_path else config.PROCESSED_DATA_DIR / "train_features.csv"

    if not feature_file.exists():
        raise FileNotFoundError(
            f"Engineered training feature file not found at '{feature_file}'. "
            "Please run Milestone 4 feature engineering before training model."
        )

    # 1. Load dataset
    df = pd.read_csv(feature_file)
    logger.info("Loaded training feature dataset: %d rows, %d columns.", df.shape[0], df.shape[1])

    # 2. Extract X, y, feature names
    X, y, feature_names = prepare_training_data(df)

    # 3. Train / Validation split (80/20)
    X_train, X_val, y_train, y_val = split_dataset(X, y, test_size=0.2, random_state=42)

    # 4. Train RandomForestRegressor
    model = train_random_forest(X_train, y_train, n_estimators=100, random_state=42)

    # 5. Evaluate metrics
    metrics, val_preds = evaluate_model(model, X_val, y_val)

    # Build validation predictions DataFrame with index metadata
    val_indices = X_val.index
    val_predictions_df = pd.DataFrame(
        {
            "unit_nr": df.loc[val_indices, "unit_nr"].values,
            "time_cycles": df.loc[val_indices, "time_cycles"].values,
            "actual_rul": y_val.values,
            "predicted_rul": np.round(val_preds, 2),
            "absolute_error": np.round(np.abs(y_val.values - val_preds), 2),
        }
    )

    # 6. Extract feature importance & plot
    importance_df = extract_feature_importance(model, feature_names)
    plot_feature_importance(importance_df, top_n=15, save_fig=True, output_dir=figures_dir)

    # 7. Save model joblib
    model_path = Path(model_dir) / "random_forest_v1.joblib"
    save_model(model, model_path)

    # 8. Save results artifacts
    save_training_artifacts(metrics, importance_df, val_predictions_df, output_dir=results_dir)

    logger.info("Model training pipeline completed successfully.")
    return model, metrics, importance_df
