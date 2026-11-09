"""
Configuration settings for Aircraft Engine Predictive Maintenance (AEPM).

This module defines project paths, dataset filenames, column schemas, and global
constants using pathlib for OS-independent path resolution.
"""

from pathlib import Path
from typing import List

# ==========================================
# PATH CONFIGURATION
# ==========================================

# Project root directory resolved relative to config.py location
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

# Model directory
MODELS_DIR: Path = PROJECT_ROOT / "models"

# Report directories
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
RESULTS_DIR: Path = REPORTS_DIR / "results"

# Application, test, documentation, and asset directories
APP_DIR: Path = PROJECT_ROOT / "app"
TESTS_DIR: Path = PROJECT_ROOT / "tests"
DOCS_DIR: Path = PROJECT_ROOT / "docs"
ASSETS_DIR: Path = PROJECT_ROOT / "assets"


# ==========================================
# NASA C-MAPSS FD001 CONSTANTS
# ==========================================

# NASA C-MAPSS FD001 filenames
TRAIN_FD001_FILE: str = "train_FD001.txt"
TEST_FD001_FILE: str = "test_FD001.txt"
RUL_FD001_FILE: str = "RUL_FD001.txt"

# NASA C-MAPSS FD001 default file paths
TRAIN_FD001_PATH: Path = RAW_DATA_DIR / TRAIN_FD001_FILE
TEST_FD001_PATH: Path = RAW_DATA_DIR / TEST_FD001_FILE
RUL_FD001_PATH: Path = RAW_DATA_DIR / RUL_FD001_FILE

# Column name definitions for NASA C-MAPSS FD001 dataset
INDEX_COLS: List[str] = ["unit_nr", "time_cycles"]
SETTING_COLS: List[str] = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS: List[str] = [f"s_{i}" for i in range(1, 22)]

# Complete sequence of column names for train and test files (26 columns total)
COLUMN_NAMES: List[str] = INDEX_COLS + SETTING_COLS + SENSOR_COLS

# Column name for Remaining Useful Life (RUL)
RUL_COL: str = "RUL"
