# System Architecture & Design Specification

## Overview

The **Aircraft Engine Predictive Maintenance (AEPM)** platform implements an end-to-end machine learning system designed to estimate the **Remaining Useful Life (RUL)** of turbofan engines using multivariate sensor telemetry from the **NASA C-MAPSS FD001** dataset.

The system is structured into a modular Python library (`src/`), an automated test suite (`tests/`), and a multi-page interactive Streamlit dashboard (`app/`).

```
AEPM Workspace Architecture
│
├── app/                        # Streamlit Interactive Dashboard
│   ├── dashboard.py            # Entry point & navigation routing
│   ├── utils.py                # CSS injection, Plotly builders & helpers
│   └── pages/                  # Page modules
│       ├── home.py             # Overview, objectives & tech stack
│       ├── predict.py          # Telemetry upload & prediction pipeline
│       ├── analytics.py        # Fleet health distribution & degradation
│       ├── model_info.py       # Metrics, hyperparams & feature importance
│       └── about.py            # Project background, workflow & team
│
├── src/                        # Modular Machine Learning Pipeline
│   ├── config.py               # Central configuration & directory paths
│   ├── validator.py            # Schema, data type & missing value validation
│   ├── data_loader.py          # Data ingestion & C-MAPSS text file parsing
│   ├── preprocessing.py        # Baseline RUL calculation & zero-variance filtering
│   ├── feature_engineering.py  # Age, sensor deltas, rolling stats & health index
│   ├── train_model.py          # RandomForestRegressor training logic
│   ├── evaluate_model.py       # Evaluation metrics & target leakage removal
│   ├── optimize_model.py       # Hyperparameter tuning (RandomizedSearchCV)
│   ├── predictor.py            # RULPredictor wrapper & feature ordering check
│   ├── health_status.py        # Health tier classification (Healthy, Normal, Warning, Critical)
│   └── inference.py            # End-to-end production inference pipeline
│
├── models/                     # Model Serialized Artifacts
│   ├── random_forest_v1.joblib # Initial model (Contains target leakage)
│   └── random_forest_v2.joblib # Production model (Leakage removed, 67 features)
│
├── tests/                      # Test Suite
│   ├── test_pipeline.py        # Pipeline integration tests
│   └── test_dashboard.py       # Dashboard & utility unit tests
│
├── run_tests.py                # Automated test runner script
└── run_dashboard.bat           # One-click dashboard launcher
```

---

## Data Flow Pipeline

```
Raw Telemetry CSV/TXT Input (26 columns: unit_nr, time_cycles, setting_1..3, s_1..s_21)
                            │
                            ▼
           1. Input Telemetry Validation (validator.py)
   (Checks schema presence, missing values threshold, and numeric dtypes)
                            │
                            ▼
           2. Data Preprocessing (preprocessing.py)
   (Deduplication, constant column removal ['setting_3'])
                            │
                            ▼
        3. Automated Feature Engineering (feature_engineering.py)
   (Engine Age, Sensor Deltas [s_i - mean], 5-cycle Rolling Stats [mean, std], Health Index)
                            │
                            ▼
         4. Target Leakage Removal (evaluate_model.py)
   (Excludes 'life_percentage' to prevent target leakage in production)
                            │
                            ▼
           5. Model Inference (predictor.py)
   (Random Forest Regressor 'random_forest_v2.joblib' across 67 features)
                            │
                            ▼
        6. Non-Negative RUL Clipping & Sanity Check
   (Ensures finite float predictions bounded by 0.0)
                            │
                            ▼
         7. Health Tier Classification (health_status.py)
   (Healthy: >100 cycles | Normal: 50-100 | Warning: 20-50 | Critical: <=20)
                            │
                            ▼
      8. Interactive Presentation & Export (app/ & Streamlit)
   (Plotly charts, metrics breakdown, critical alerts, CSV export)
```

---

## Technical Component Responsibilities

1. **`src/config.py`**: Centralizes file paths, NASA C-MAPSS column schemas, index names, default hyperparameters, and output report paths.
2. **`src/validator.py`**: Enforces strict schema validation (26 columns), numeric type checks, and null cell thresholds.
3. **`src/preprocessing.py`**: Standardizes column names, computes run-to-failure cycle limits, removes uninformative operational channels.
4. **`src/feature_engineering.py`**: Computes domain features: Engine Age, Sensor Deltas relative to baseline, 5-flight-cycle rolling window means/stds, and normalized Engine Health Index.
5. **`src/evaluate_model.py`**: Calculates RMSE, MAE, R² metrics, generates residual analysis plots, and eliminates target leakage channels (`life_percentage`).
6. **`src/predictor.py`**: Implements the `RULPredictor` class. Enforces feature name matching using `model.feature_names_in_` and clips raw predictions to non-negative floats.
7. **`src/health_status.py`**: Categorizes engine states into four maintenance tiers based on RUL cycle count.
8. **`src/inference.py`**: Integrates validation, transformation, feature generation, model execution, and health classification into a single call: `run_inference_pipeline()`.
