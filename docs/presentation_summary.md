# AEPM - Presentation Summary

Use this as a 15-slide academic/project evaluation outline. Each slide includes the main points, important numbers, suggested visual/demo, and presenter talking points.

## Slide 1 - Title

**Aircraft Engine Predictive Maintenance using Machine Learning**

- Project name: AEPM
- Version: 1.0.0
- Dataset: NASA C-MAPSS FD001
- Objective: predict Remaining Useful Life of turbofan engines

Important numbers: 100 test engines, 9,468 verified FD001 test rows.

Suggested visual/demo: Open the dashboard Home page.

Presenter talking points: Introduce the project as a complete ML engineering workflow, from raw telemetry to dashboard-based decision support.

## Slide 2 - Problem Statement

- Aircraft engines degrade over operating cycles.
- Unplanned failure creates safety, cost, and scheduling risk.
- Maintenance teams need early warning before failure.
- The project predicts RUL and classifies maintenance health.

Important numbers: Health categories are Healthy, Normal, Warning, and Critical.

Suggested visual/demo: Simple flow: telemetry -> RUL prediction -> health status.

Presenter talking points: Explain why RUL is useful for preventive maintenance planning.

## Slide 3 - Project Objectives

- Validate raw FD001 telemetry.
- Engineer degradation-aware features.
- Train and compare Random Forest models.
- Detect and remove target leakage.
- Build production inference and Streamlit dashboard.
- Package tests, docs, and deployment files.

Important numbers: 14 automated tests; 5 dashboard pages.

Suggested visual/demo: Repository structure or dashboard navigation.

Presenter talking points: Emphasize that the project is not only a model, but a reproducible system.

## Slide 4 - Dataset: NASA C-MAPSS FD001

- NASA turbofan engine degradation benchmark.
- 100 training engines and 100 test engines.
- 26 raw telemetry columns.
- 21 sensors and 3 operating settings.
- Single operating condition and single HPC degradation fault mode.

Important numbers: 9,468 verified test rows.

Suggested visual/demo: Predict RUL page data preview after loading sample FD001 data.

Presenter talking points: Explain that FD001 is a standard predictive maintenance dataset.

## Slide 5 - Complete AEPM Workflow

- Raw telemetry input.
- Validation and preprocessing.
- Feature engineering.
- Leakage-free Random Forest inference.
- Health classification.
- Dashboard analytics and CSV export.

Important numbers: Output schema has 4 columns: `unit_nr`, `time_cycles`, `predicted_rul`, `health_status`.

Suggested visual/demo: Architecture diagram from README or `docs/architecture.md`.

Presenter talking points: Walk through the system from data ingestion to final maintenance insight.

## Slide 6 - Data Preprocessing and Feature Engineering

- Schema validation for all required telemetry columns.
- Missing/null and numeric type checks.
- Removal of uninformative FD001 fields.
- Sensor deltas, rolling statistics, and health index.
- Production inference uses the same preparation path.

Important numbers: Model consumes 67 production features.

Suggested visual/demo: Show `src/inference.py` or feature list in Model Information page.

Presenter talking points: Explain why feature engineering is essential for degradation modeling.

## Slide 7 - Target Leakage Discovery

- Model v1 included `life_percentage`.
- `life_percentage` depends on full engine lifetime.
- Full lifetime is not available for active engines.
- v1 validation looked unrealistically strong.
- Leakage was removed before production.

Important numbers: v1 RMSE 4.0124, R2 0.9959, but invalid.

Suggested visual/demo: Model comparison table.

Presenter talking points: Make this the key engineering judgment moment: the best-looking model was rejected because it was not valid.

## Slide 8 - Model Development and Comparison

- Model v1: 68 features, target leakage present, invalid.
- Model v2: 67 features, leakage removed, production model.
- Optimized baseline: 67 features, not promoted.
- Random Forest selected for tabular nonlinear telemetry.

Important numbers: Model v2 RMSE 31.0758, MAE 25.3116, R2 0.7557.

Suggested visual/demo: `docs/model_evaluation.md` or dashboard Model Information page.

Presenter talking points: Explain that Model v2 has realistic deployable performance.

## Slide 9 - Production Model

- File: `models/random_forest_v2.joblib`.
- Type: `RandomForestRegressor`.
- `feature_names_in_` exists.
- Feature count: 67.
- `life_percentage` is absent.

Important numbers: `n_estimators` 100, `random_state` 42.

Suggested visual/demo: Model Information page.

Presenter talking points: Mention feature-order enforcement and why the artifact is not modified during demo.

## Slide 10 - Validation Results

- Automated suite executed through `run_tests.py`.
- Schema, numeric, null, and malformed input tests passed.
- Feature engineering and leakage exclusion tests passed.
- Dashboard utility and page import tests passed.

Important numbers: 14 tests total, 14 passed, 0 failed.

Suggested visual/demo: Terminal output from final test run.

Presenter talking points: Highlight that tests cover both ML pipeline and dashboard utilities.

## Slide 11 - Streamlit Dashboard

- Five pages: Home, Predict RUL, Dashboard Analytics, Model Information, About Project.
- Predict page supports sample FD001 data.
- Dashboard calls production inference pipeline.
- Plotly charts support interactive exploration.
- CSV download is available.

Important numbers: 5 verified page modules import successfully.

Suggested visual/demo: Navigate through all five pages.

Presenter talking points: Treat the dashboard as the user-facing proof of the pipeline.

## Slide 12 - End-to-End Inference Results

- Dataset: `data/raw/test_FD001.txt`.
- Input rows: 9,468.
- Engines: 100.
- Predictions: 9,468.
- All predictions finite and non-negative.

Important numbers: min RUL 17.03, max RUL 220.63, mean RUL 141.85.

Suggested visual/demo: Prediction table and RUL histogram.

Presenter talking points: Separate row-level prediction count from final-cycle engine summary.

## Slide 13 - Health Classification Results

- Healthy: RUL > 100.
- Normal: 50 < RUL <= 100.
- Warning: 20 < RUL <= 50.
- Critical: RUL <= 20.
- All final-cycle engines classified once.

Important numbers: Healthy 56, Normal 36, Warning 6, Critical 2.

Suggested visual/demo: Health donut and critical/warning callout table.

Presenter talking points: Explain how raw RUL predictions become maintenance-ready categories.

## Slide 14 - Testing and Deployment

- Test runner: `run_tests.py`.
- Runtime dependencies: `requirements.txt`.
- Dev dependencies: `requirements-dev.txt`.
- Dockerfile and `.dockerignore` are present.
- GitHub Actions CI workflow is present.

Important numbers: Docker build and GitHub Actions execution were not run locally.

Suggested visual/demo: Show deployment files and README run commands.

Presenter talking points: Be precise: deployment configuration is present and inspected, but local Docker/CI execution was not performed.

## Slide 15 - Conclusion and Future Work

- AEPM v1.0.0 is ready for academic/project submission.
- Production model is leakage-free.
- End-to-end FD001 inference is verified.
- Dashboard and documentation are complete.
- Future work: FD002-FD004, sequence models, API serving, live telemetry.

Important numbers: final status is READY FOR SUBMISSION.

Suggested visual/demo: Open `reports/release_checklist.md` or `reports/final_demo_report.json`.

Presenter talking points: Close with the full engineering story: detect leakage, fix it, verify the release.
