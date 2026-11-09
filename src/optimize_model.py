"""
Model Optimization module for Aircraft Engine Predictive Maintenance (AEPM).

Provides modular functions to perform RandomizedSearchCV hyperparameter tuning on a
RandomForestRegressor with 5-fold cross-validation, save the final optimized model,
and export comparative metrics, parameter tables, and visualization figures.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

from src import config
from src.evaluate_model import remove_target_leakage_features
from src.train_model import evaluate_model, extract_feature_importance, plot_feature_importance, prepare_training_data, split_dataset
from src.utils import ensure_dir, setup_logger

logger = setup_logger(__name__)


def get_param_distributions() -> dict:
    """Return the hyperparameter search grid for RandomForestRegressor tuning.

    Returns:
        dict: Hyperparameter distributions for RandomizedSearchCV.
    """
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    return param_dist


def optimize_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_distributions: Optional[dict] = None,
    n_iter: int = 10,
    cv: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Tuple[RandomForestRegressor, dict, pd.DataFrame]:
    """Execute RandomizedSearchCV hyperparameter tuning for RandomForestRegressor.

    Args:
        X_train: Training feature matrix.
        y_train: Training target labels.
        param_distributions: Hyperparameter grid dictionary. Defaults to get_param_distributions().
        n_iter: Number of parameter combinations to sample. Defaults to 10.
        cv: Number of cross-validation folds. Defaults to 5.
        random_state: Random seed for reproducibility. Defaults to 42.
        n_jobs: Parallel CPU jobs. Defaults to -1.

    Returns:
        Tuple[RandomForestRegressor, dict, pd.DataFrame]: Best estimator model, best parameters, CV results.
    """
    param_grid = param_distributions or get_param_distributions()

    base_rf = RandomForestRegressor(random_state=random_state)

    logger.info(
        "Initiating RandomizedSearchCV (n_iter=%d, cv=%d folds)...", n_iter, cv
    )

    search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=1,
    )

    search.fit(X_train, y_train)

    best_model: RandomForestRegressor = search.best_estimator_
    best_params: dict = search.best_params_
    cv_results_df = pd.DataFrame(search.cv_results_)

    logger.info("RandomizedSearchCV complete. Best Score (neg_RMSE): %.4f", search.best_score_)
    logger.info("Best Hyperparameters: %s", best_params)

    return best_model, best_params, cv_results_df


def plot_model_comparison(
    baseline_metrics: Dict[str, float],
    optimized_metrics: Dict[str, float],
    save_fig: bool = True,
    output_dir: Union[str, Path] = config.FIGURES_DIR,
) -> plt.Figure:
    """Plot comparative performance bar chart between Baseline and Optimized models.

    Args:
        baseline_metrics: Metrics dict for baseline model (v2).
        optimized_metrics: Metrics dict for optimized final model.
        save_fig: Whether to save figure to disk.
        output_dir: Destination directory path.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    metrics_keys = ["rmse", "mae", "r2_score"]
    metric_labels = ["RMSE (lower is better)", "MAE (lower is better)", "R² Score (higher is better)"]

    base_vals = [baseline_metrics.get(k, 0.0) for k in metrics_keys]
    opt_vals = [optimized_metrics.get(k, 0.0) for k in metrics_keys]

    x = np.arange(len(metrics_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))

    rects1 = ax.bar(x - width / 2, base_vals, width, label="Baseline (v2)", color="#6c757d", edgecolor="#111111")
    rects2 = ax.bar(x + width / 2, opt_vals, width, label="Optimized (Final)", color="#28a745", edgecolor="#111111")

    ax.set_ylabel("Metric Value", fontsize=11, fontweight="bold")
    ax.set_title("Model Performance Comparison: Baseline vs Optimized Model", fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Annotate bar values
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()

    if save_fig:
        out_path = Path(output_dir) / "model_comparison.png"
        ensure_dir(output_dir)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        logger.info("Saved Model Comparison plot to: %s", out_path)

    return fig


def save_optimization_artifacts(
    best_params: dict,
    optimization_results: dict,
    cv_results_df: pd.DataFrame,
    output_dir: Union[str, Path] = config.RESULTS_DIR,
) -> Tuple[Path, Path, Path]:
    """Save optimization JSON reports and cross-validation results CSV.

    Args:
        best_params: Dictionary of best hyperparameters.
        optimization_results: Summary report dictionary of optimization metrics.
        cv_results_df: Cross-validation results DataFrame.
        output_dir: Destination folder path.

    Returns:
        Tuple[Path, Path, Path]: Output paths for (best_parameters.json, optimization_results.json, cross_validation_results.csv).
    """
    out_dir = ensure_dir(output_dir)

    best_params_path = out_dir / "best_parameters.json"
    results_path = out_dir / "optimization_results.json"
    cv_csv_path = out_dir / "cross_validation_results.csv"

    # Save best_parameters.json
    with open(best_params_path, "w", encoding="utf-8") as f:
        # Convert None to "None" string if present for clean JSON
        clean_params = {k: (v if v is not None else "None") for k, v in best_params.items()}
        json.dump(clean_params, f, indent=4)

    # Save optimization_results.json
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(optimization_results, f, indent=4)

    # Save cross_validation_results.csv
    cv_results_df.to_csv(cv_csv_path, index=False)

    logger.info("Saved best_parameters.json to: %s", best_params_path)
    logger.info("Saved optimization_results.json to: %s", results_path)
    logger.info("Saved cross_validation_results.csv to: %s", cv_csv_path)

    return best_params_path, results_path, cv_csv_path


def run_model_optimization_pipeline(
    data_path: Optional[Union[str, Path]] = None,
    n_iter: int = 10,
    cv: int = 5,
    output_model_path: Union[str, Path] = config.MODELS_DIR / "random_forest_final.joblib",
) -> Tuple[RandomForestRegressor, dict, dict, pd.DataFrame]:
    """Execute the complete model optimization pipeline end-to-end.

    Steps:
    1. Load train_features.csv.
    2. Remove 'life_percentage' feature (target leakage).
    3. Perform 80/20 train/validation split.
    4. Run RandomizedSearchCV hyperparameter search over 5-fold CV.
    5. Evaluate best model on validation set.
    6. Plot final feature importances (reports/figures/feature_importance_final.png).
    7. Plot comparison with baseline model (reports/figures/model_comparison.png).
    8. Save optimized final model to models/random_forest_final.joblib.
    9. Export best_parameters.json, optimization_results.json, cross_validation_results.csv.

    Args:
        data_path: Optional custom path to train_features.csv.
        n_iter: Number of RandomizedSearchCV search iterations.
        cv: Number of cross-validation folds.
        output_model_path: Path to export final joblib model.

    Returns:
        Tuple[RandomForestRegressor, dict, dict, pd.DataFrame]:
            (Final Model, Optimized Metrics, Best Parameters, CV Results DataFrame).
    """
    logger.info("Starting model optimization pipeline...")

    feat_path = Path(data_path) if data_path else config.PROCESSED_DATA_DIR / "train_features.csv"
    raw_df = pd.read_csv(feat_path)

    # 1. Remove target leakage
    clean_df = remove_target_leakage_features(raw_df, leak_cols=["life_percentage"])

    # 2. Extract X and y
    X, y, feature_names = prepare_training_data(clean_df)

    # 3. Train / Validation Split
    X_train, X_val, y_train, y_val = split_dataset(X, y, test_size=0.2, random_state=42)

    # 4. Run RandomizedSearchCV
    best_model, best_params, cv_results_df = optimize_random_forest(
        X_train, y_train, n_iter=n_iter, cv=cv, random_state=42
    )

    # 5. Evaluate optimized model on validation set
    opt_metrics, val_preds = evaluate_model(best_model, X_val, y_val)

    logger.info(
        "Final Optimized Model Validation -> RMSE: %.4f | MAE: %.4f | R²: %.4f",
        opt_metrics["rmse"],
        opt_metrics["mae"],
        opt_metrics["r2_score"],
    )

    # 6. Extract & Plot Final Feature Importance
    importance_df = extract_feature_importance(best_model, feature_names)
    plot_feature_importance(
        importance_df,
        top_n=15,
        save_fig=True,
        output_dir=config.FIGURES_DIR,
    )
    # Also save as feature_importance_final.png specifically
    final_imp_path = config.FIGURES_DIR / "feature_importance_final.png"
    fig_imp = plot_feature_importance(importance_df, top_n=15, save_fig=False)
    fig_imp.savefig(final_imp_path, dpi=300, bbox_inches="tight")
    logger.info("Saved final feature importance plot to: %s", final_imp_path)

    # 7. Compare with Baseline v2 metrics
    v2_report_path = config.RESULTS_DIR / "evaluation_report.json"
    if v2_report_path.exists():
        with open(v2_report_path, "r", encoding="utf-8") as f:
            v2_report = json.load(f)
            baseline_metrics = {"rmse": v2_report["rmse"], "mae": v2_report["mae"], "r2_score": v2_report["r2_score"]}
    else:
        baseline_metrics = {"rmse": 31.0758, "mae": 25.3116, "r2_score": 0.7557}

    plot_model_comparison(baseline_metrics, opt_metrics, save_fig=True)

    # 8. Save final model joblib
    out_model_path = ensure_dir(Path(output_model_path).parent) / Path(output_model_path).name
    joblib.dump(best_model, out_model_path)
    logger.info("Saved final optimized model to: %s", out_model_path)

    # 9. Save optimization artifacts
    optimization_results = {
        "model_name": "RandomForestRegressor (Final Optimized)",
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "cv_folds": cv,
        "search_iterations": n_iter,
        "metrics": opt_metrics,
        "best_parameters": {k: (v if v is not None else "None") for k, v in best_params.items()},
    }

    save_optimization_artifacts(best_params, optimization_results, cv_results_df)

    logger.info("Model optimization pipeline completed successfully.")
    return best_model, opt_metrics, best_params, cv_results_df
