"""
Exploratory Data Analysis (EDA) module for NASA C-MAPSS FD001 dataset.

Provides reusable modular functions to calculate summary statistics, check data
integrity (missing values and duplicate rows), analyze engine flight cycles, and
generate visualizations (sensor histograms, boxplots, correlation heatmaps) using
Matplotlib. Results and figures are automatically saved to reports/.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.utils import ensure_dir, setup_logger

logger = setup_logger(__name__)


def check_missing_and_duplicates(
    df: pd.DataFrame, dataset_name: str = "Dataset"
) -> Tuple[pd.Series, int]:
    """Check missing values count per column and total duplicate rows.

    Args:
        df: Input DataFrame to inspect.
        dataset_name: Name label for dataset logging.

    Returns:
        Tuple[pd.Series, int]: Missing values per column and duplicate rows count.
    """
    missing_series = df.isnull().sum()
    duplicate_count = int(df.duplicated().sum())

    logger.info(
        "[%s] Total missing values: %d | Total duplicate rows: %d",
        dataset_name,
        missing_series.sum(),
        duplicate_count,
    )
    return missing_series, duplicate_count


def get_descriptive_stats(
    df: pd.DataFrame,
    dataset_name: str = "Dataset",
    save_results: bool = True,
    output_dir: Union[str, Path] = config.RESULTS_DIR,
) -> pd.DataFrame:
    """Generate descriptive statistics summary for numerical features and save as CSV.

    Args:
        df: Input DataFrame.
        dataset_name: Name of the dataset for logging and file naming.
        save_results: Whether to export summary table to disk.
        output_dir: Output folder for CSV files.

    Returns:
        pd.DataFrame: Summary statistics including count, mean, std, min, percentiles, max, and skew.
    """
    stats_df = df.describe().T
    stats_df["skew"] = df.skew(numeric_only=True)

    if save_results:
        out_path = Path(output_dir) / f"{dataset_name.lower()}_summary_stats.csv"
        ensure_dir(output_dir)
        stats_df.to_csv(out_path)
        logger.info("[%s] Summary statistics saved to: %s", dataset_name, out_path)

    return stats_df


def get_engine_cycle_stats(
    df: pd.DataFrame,
    dataset_name: str = "Dataset",
    save_results: bool = True,
    output_dir: Union[str, Path] = config.RESULTS_DIR,
) -> pd.DataFrame:
    """Calculate flight cycle statistics per unique engine unit.

    Args:
        df: Input DataFrame containing 'unit_nr' and 'time_cycles'.
        dataset_name: Name label for logging and filenames.
        save_results: Whether to export cycle stats table to CSV.
        output_dir: Destination folder path.

    Returns:
        pd.DataFrame: Aggregated flight cycle metrics per engine unit.
    """
    if "unit_nr" not in df.columns or "time_cycles" not in df.columns:
        raise KeyError("DataFrame must contain 'unit_nr' and 'time_cycles' columns.")

    cycle_stats = (
        df.groupby("unit_nr")["time_cycles"]
        .agg(
            min_cycle="min",
            max_cycle="max",
            mean_cycle="mean",
            total_cycles="count",
        )
        .reset_index()
    )

    num_engines = cycle_stats["unit_nr"].nunique()
    logger.info(
        "[%s] Number of unique engines: %d | Max flight cycles: %d | Avg max cycles: %.2f",
        dataset_name,
        num_engines,
        cycle_stats["max_cycle"].max(),
        cycle_stats["max_cycle"].mean(),
    )

    if save_results:
        out_path = Path(output_dir) / f"{dataset_name.lower()}_engine_cycle_stats.csv"
        ensure_dir(output_dir)
        cycle_stats.to_csv(out_path, index=False)
        logger.info("[%s] Engine cycle statistics saved to: %s", dataset_name, out_path)

    return cycle_stats


def plot_sensor_histograms(
    df: pd.DataFrame,
    sensor_cols: Optional[List[str]] = None,
    dataset_name: str = "Train",
    save_fig: bool = True,
    output_dir: Union[str, Path] = config.FIGURES_DIR,
    bins: int = 30,
) -> plt.Figure:
    """Plot histograms for sensor measurements using Matplotlib.

    Args:
        df: Input DataFrame containing sensor measurements.
        sensor_cols: List of sensor columns. Defaults to config.SENSOR_COLS.
        dataset_name: Dataset name for title and output file.
        save_fig: Whether to save figure to disk.
        output_dir: Directory where the plot figure will be saved.
        bins: Number of histogram bins.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    cols = sensor_cols if sensor_cols is not None else config.SENSOR_COLS
    cols = [col for col in cols if col in df.columns]

    n_cols = 4
    n_rows = (len(cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    fig.suptitle(f"{dataset_name} Set - Sensor Distributions", fontsize=16, y=1.02)
    axes_flat = np.array(axes).flatten()

    for idx, col in enumerate(cols):
        ax = axes_flat[idx]
        ax.hist(df[col].dropna(), bins=bins, color="#1f77b4", edgecolor="#111111", alpha=0.75)
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_xlabel("Value", fontsize=9)
        ax.set_ylabel("Frequency", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)

    # Remove unused axis grids
    for idx in range(len(cols), len(axes_flat)):
        fig.delaxes(axes_flat[idx])

    plt.tight_layout()

    if save_fig:
        out_path = Path(output_dir) / f"{dataset_name.lower()}_sensor_histograms.png"
        ensure_dir(output_dir)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        logger.info("[%s] Sensor histograms figure saved to: %s", dataset_name, out_path)

    return fig


def plot_sensor_boxplots(
    df: pd.DataFrame,
    sensor_cols: Optional[List[str]] = None,
    dataset_name: str = "Train",
    save_fig: bool = True,
    output_dir: Union[str, Path] = config.FIGURES_DIR,
) -> plt.Figure:
    """Plot boxplots for sensor signals to visualize variance and potential outliers.

    Args:
        df: Input DataFrame containing sensor measurements.
        sensor_cols: List of sensor columns. Defaults to config.SENSOR_COLS.
        dataset_name: Dataset name for title and output file.
        save_fig: Whether to save figure to disk.
        output_dir: Output directory for image files.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    cols = sensor_cols if sensor_cols is not None else config.SENSOR_COLS
    cols = [col for col in cols if col in df.columns]

    n_cols = 4
    n_rows = (len(cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    fig.suptitle(f"{dataset_name} Set - Sensor Boxplots (Variance & Outliers)", fontsize=16, y=1.02)
    axes_flat = np.array(axes).flatten()

    for idx, col in enumerate(cols):
        ax = axes_flat[idx]
        ax.boxplot(
            df[col].dropna(),
            patch_artist=True,
            boxprops=dict(facecolor="#2ca02c", color="#111111", alpha=0.7),
            medianprops=dict(color="red", linewidth=1.5),
            flierprops=dict(marker="o", markerfacecolor="red", markersize=3, alpha=0.5),
        )
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_ylabel("Value", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)

    for idx in range(len(cols), len(axes_flat)):
        fig.delaxes(axes_flat[idx])

    plt.tight_layout()

    if save_fig:
        out_path = Path(output_dir) / f"{dataset_name.lower()}_sensor_boxplots.png"
        ensure_dir(output_dir)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        logger.info("[%s] Sensor boxplots figure saved to: %s", dataset_name, out_path)

    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    dataset_name: str = "Train",
    save_fig: bool = True,
    output_dir: Union[str, Path] = config.FIGURES_DIR,
) -> plt.Figure:
    """Plot Pearson correlation matrix heatmap using pure Matplotlib.

    Args:
        df: Input DataFrame.
        feature_cols: List of features to analyze. Defaults to config.SENSOR_COLS.
        dataset_name: Label for plot title and output filename.
        save_fig: Whether to save figure to disk.
        output_dir: Output folder for figure.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    cols = feature_cols if feature_cols is not None else config.SENSOR_COLS
    cols = [col for col in cols if col in df.columns]

    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    cax = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)

    cbar = fig.colorbar(cax, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("Pearson Correlation Coefficient", rotation=-90, va="bottom", fontsize=11)

    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(cols, fontsize=9)

    ax.set_title(f"{dataset_name} Set - Feature Correlation Heatmap", fontsize=15, pad=20)

    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr.iloc[i, j]
            if not np.isnan(val):
                text_color = "white" if abs(val) > 0.6 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=7,
                )

    plt.tight_layout()

    if save_fig:
        out_path = Path(output_dir) / f"{dataset_name.lower()}_correlation_heatmap.png"
        ensure_dir(output_dir)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        logger.info("[%s] Correlation heatmap figure saved to: %s", dataset_name, out_path)

    return fig
