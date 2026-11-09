# Changelog

All notable changes to the **Aircraft Engine Predictive Maintenance (AEPM)** project are
documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] — 2026-08-10 · Milestone 11: Production Deployment & Final Project Packaging

### Added
- `requirements.txt` — Pinned runtime dependency lockfile (numpy, pandas, scikit-learn,
  scipy, joblib, streamlit, plotly).
- `requirements-dev.txt` — Development/testing dependency file (pytest, pytest-cov, flake8).
- `setup.py` — `setuptools` package declaration supporting `pip install -e .` editable installs.
- `.streamlit/config.toml` — Streamlit server & dark-navy theme configuration applied
  automatically on dashboard launch.
- `run_dashboard.sh` — Unix/macOS bash launcher with venv detection and error checking.
- `run_tests.bat` — Windows batch launcher for the automated test suite with exit-code relay.
- `Dockerfile` — Production container image using `python:3.11-slim`, exposing port 8501.
- `.dockerignore` — Docker build context exclusion list (`.venv/`, `__pycache__/`, `notebooks/`,
  `reports/figures/`, `.git/`).
- `.github/workflows/ci.yml` — GitHub Actions CI workflow: full 14-test suite on Python 3.11
  and 3.12, plus flake8 lint gate.
- `CHANGELOG.md` — This file; complete milestone-by-milestone project history.

### Changed
- `README.md` — Added CI badge, Docker deployment section, `requirements.txt` usage note,
  and final project packaging details.
- `docs/deployment.md` — Added Docker deployment section, `.streamlit/config.toml`
  explanation, and Unix launcher instructions.
- `reports/final_validation_report.json` — Updated to reflect Milestone 11 completion with
  `deployment_artifacts` and `milestone_11_status` fields.

---

## [0.10.0] — 2026-08-10 · Milestone 10: Final Validation, Testing & Deployment Readiness

### Added
- Complete 14-point automated test suite (`tests/test_pipeline.py` + `tests/test_dashboard.py`).
- `run_tests.py` — Custom test runner with timing, pass/fail statistics, and structured output.
- `reports/final_validation_report.json` — Machine-readable validation report confirming
  `READY_FOR_DEPLOYMENT` status.
- Full documentation suite: `docs/architecture.md`, `docs/methodology.md`,
  `docs/model_evaluation.md`, `docs/deployment.md`.

### Fixed
- All 14 tests confirmed passing: 9 pipeline integration tests + 5 dashboard unit tests.
- Verified end-to-end inference pipeline integrity on `test_FD001.txt`.

---

## [0.9.0] — 2026-08-10 · Milestone 9: Professional Streamlit Dashboard

### Added
- `app/dashboard.py` — Multi-page Streamlit entry point with wide layout and dark-navy
  sidebar navigation.
- `app/utils.py` — Dark-mode CSS injection, Plotly chart builders (`plot_health_donut`,
  `plot_rul_histogram`, `plot_engine_rul_bar`, `plot_engine_degradation`),
  `status_badge()`, `df_to_csv_bytes()`, `load_json_safe()`.
- `app/pages/home.py` — Hero metrics, 9-step milestone checklist, technology stack grid.
- `app/pages/predict.py` — CSV/TXT uploader, "⚡ Use Sample FD001 Test Data" button,
  real-time prediction table, health cards, and CSV download.
- `app/pages/analytics.py` — Fleet health donut chart, RUL histogram, engine-wise bar chart,
  multi-engine degradation line chart, critical engine summary table.
- `app/pages/model_info.py` — Production model metrics, model progression comparison table,
  67-feature list, hyperparameter grid, top-20 feature importance Plotly chart.
- `app/pages/about.py` — Background, dataset specifications, algorithm details, health
  threshold legend, and student/guide information.
- `run_dashboard.bat` — Windows one-click Streamlit launcher.

---

## [0.8.0] — Milestone 8: Production Prediction Pipeline

### Added
- `src/inference.py` — `prepare_inference_features()` and `run_inference_pipeline()` for
  end-to-end batch prediction on raw telemetry DataFrames.
- `src/inference.py` — `run_inference_on_file()` for file-based batch inference with CSV export.
- Confirmed target leakage (`life_percentage`) is excluded at inference time.

---

## [0.7.0] — Milestone 7: Hyperparameter Optimization

### Added
- `src/optimize_model.py` — RandomizedSearchCV hyperparameter grid search across
  `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`.
- `reports/results/best_parameters.json` — Optimal hyperparameter set from search.
- `models/random_forest_final.joblib` — Optimized model artifact (baseline reference;
  **not** the production model).

### Notes
- `random_forest_v2.joblib` outperforms optimized model on validation set and remains
  the production model.

---

## [0.6.0] — Milestone 6: Model Evaluation & Target Leakage Removal

### Added
- `src/evaluate_model.py` — `remove_target_leakage_features()` permanently excludes
  `life_percentage` from training and inference pipelines.
- `reports/results/model_metrics.json` — RMSE, MAE, R² for both model versions.
- `reports/results/evaluation_report.json` — Detailed evaluation report with leakage analysis.
- `reports/results/sample_predictions.csv` — Sample predictions CSV from production pipeline.

### Changed
- Retrained model without `life_percentage` → `models/random_forest_v2.joblib`
  (67 features, RMSE 31.08, R² 0.7557).

### Fixed
- **Critical**: Identified `life_percentage` target leakage in Model v1 (R² 0.9959 → inflated).
  Model v1 (`random_forest_v1.joblib`) marked INVALID.

---

## [0.5.0] — Milestone 5: Model Training

### Added
- `src/train_model.py` — Trains `RandomForestRegressor` with full feature matrix.
- `models/random_forest_v1.joblib` — Initial model artifact (68 features, **leakage present**).
- `src/predictor.py` — `RULPredictor` class with `load_model()` and `predict_rul()` functions.

---

## [0.4.0] — Milestone 4: Feature Engineering

### Added
- `src/feature_engineering.py` — Engine Age calculation, sensor baseline delta features
  (`s_*_delta`), 5-cycle rolling mean/std features (`s_*_roll_mean_5`, `s_*_roll_std_5`),
  composite Health Index, and `life_percentage` (later identified as leakage).

---

## [0.3.0] — Milestone 3: Data Preprocessing

### Added
- `src/preprocessing.py` — Duplicate and null cleaning (`clean_missing_and_duplicates`),
  constant column removal (`remove_uninformative_columns`).
- Identified `setting_3` as constant in FD001 dataset and excluded from feature matrix.

---

## [0.2.0] — Milestone 2: Exploratory Data Analysis

### Added
- `src/exploratory_analysis.py` — Sensor distribution plots, correlation heatmaps, RUL
  distribution analysis, engine degradation trajectory visualisations.
- EDA outputs saved to `reports/figures/`.

---

## [0.1.0] — Milestone 1: Project Setup

### Added
- Project directory structure: `src/`, `app/`, `tests/`, `models/`, `data/`, `reports/`,
  `notebooks/`, `docs/`, `assets/`.
- `src/config.py` — Central path configuration and NASA C-MAPSS FD001 column schema.
- `src/validator.py` — `validate_input_schema()`, `validate_missing_values()`,
  `validate_numeric_types()`, `validate_input_dataframe()`.
- `src/health_status.py` — `get_health_status()`, `add_health_status_column()`,
  `get_health_summary()` health classification with 4-tier thresholds.
- `src/utils.py` — `setup_logger()`, `ensure_dir()` shared utilities.
- `src/data_loader.py` — NASA C-MAPSS FD001 raw file loader with space-delimited parsing.
- `README.md` — Comprehensive project documentation with architecture diagram,
  dataset description, and setup instructions.
- NASA C-MAPSS FD001 dataset files: `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`.
- Python virtual environment (`.venv/`) configured with all runtime dependencies.

---

*AEPM v1.0.0 — Production Ready | NASA C-MAPSS FD001 | Random Forest Regressor | 67 Features*
