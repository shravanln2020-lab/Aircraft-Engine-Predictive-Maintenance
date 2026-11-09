# AEPM - Final Project Report

**Project title:** Aircraft Engine Predictive Maintenance using Machine Learning  
**Version:** 1.0.0  
**Milestone:** 12 - Final Demo, Release Verification and Project Presentation  
**Verification date:** 2026-08-10  
**Production model:** `models/random_forest_v2.joblib`

## 1. Project Title

Aircraft Engine Predictive Maintenance using Machine Learning (AEPM) is a final-year style engineering project for predicting aircraft turbofan Remaining Useful Life (RUL) from NASA C-MAPSS FD001 telemetry.

## 2. Executive Summary

AEPM delivers a reproducible predictive maintenance workflow: raw FD001 telemetry validation, preprocessing, feature engineering, Random Forest model evaluation, leakage removal, production inference, health classification, Streamlit dashboard visualization, automated tests, and deployment packaging.

Milestone 12 verified the project without retraining or replacing the production model. The verified production model is `models/random_forest_v2.joblib`, a `RandomForestRegressor` with 67 features and no `life_percentage` target leakage.

## 3. Problem Statement

Aircraft engines degrade over time, and unplanned failures can cause safety risk, delays, and high maintenance cost. The project problem is to predict how many operating cycles remain before an engine reaches failure condition, using multivariate sensor readings and operating settings.

The output is both a numeric RUL prediction and a maintenance-oriented health category.

## 4. Project Objectives

- Validate NASA C-MAPSS FD001 telemetry before modeling or inference.
- Engineer degradation-aware features from cycle-level engine data.
- Train and compare Random Forest model iterations for RUL regression.
- Identify and remove target leakage before production use.
- Provide a production inference function for raw telemetry files.
- Build a Streamlit dashboard for demonstration and interactive analysis.
- Verify the final project with automated tests and release checks.

## 5. Dataset Description

The project uses the NASA C-MAPSS FD001 turbofan degradation dataset.

| Dataset property | Verified value |
|---|---:|
| Training engines | 100 |
| Test engines | 100 |
| Verified FD001 test rows | 9,468 |
| Raw telemetry columns | 26 |
| Sensor channels | 21 |
| Operating settings | 3 |
| Operating condition | Single |
| Fault mode | Single HPC degradation |

Raw files are stored in `data/raw/`: `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt`.

## 6. AEPM Architecture

The repository is organized into clear project layers:

- `src/`: validation, preprocessing, feature engineering, model evaluation, prediction, inference, and health classification.
- `app/`: Streamlit dashboard entry point, shared Plotly/UI helpers, and five page modules.
- `data/`: raw and processed FD001 data.
- `models/`: serialized Random Forest artifacts.
- `tests/`: automated pipeline and dashboard tests.
- `docs/`: architecture, methodology, deployment, final report, presentation summary, and walkthrough.
- `reports/`: validation reports, release checklist, figures, and result files.

## 7. Data Preprocessing

Preprocessing protects the pipeline before feature generation:

- Required 26-column schema validation.
- Missing/null value checks.
- Numeric type validation.
- Duplicate row cleanup.
- Removal of uninformative FD001 columns such as `setting_3`.

Relevant modules: `src/validator.py` and `src/preprocessing.py`.

## 8. Feature Engineering

Feature engineering transforms raw engine cycles into degradation-aware model inputs:

- Engine age and cycle-position features.
- Sensor delta features relative to early-cycle baselines.
- Five-cycle rolling means.
- Five-cycle rolling standard deviations.
- Composite health index.

The verified inference feature preparation creates 69 total columns including metadata (`unit_nr`, `time_cycles`). The model consumes 67 actual production features.

## 9. RUL Formulation

Remaining Useful Life is modeled as a regression target measured in operating cycles. For a given engine at cycle `t`, RUL estimates how many cycles remain before the known failure point in the run-to-failure training trajectory.

The production inference output schema is:

```text
unit_nr, time_cycles, predicted_rul, health_status
```

## 10. Target Leakage Discovery

Model v1 produced unrealistically strong validation metrics because it included `life_percentage`. That feature uses the full engine lifetime, which is not available in real inference for an active engine.

The leakage discovery changed the project decision: the highest-scoring model was rejected, and the leakage-free model became the production model.

## 11. Why `life_percentage` Is Invalid

`life_percentage` depends on total engine lifetime. Since total lifetime equals current cycle plus remaining useful life, it gives the model direct information about the target. In real operation, the final lifetime of an engine is unknown, so the feature cannot be used honestly for deployment.

Because of this, `life_percentage` is excluded from production training and inference.

## 12. Model Iterations

| Model | Feature count | Leakage status | RMSE | MAE | R2 | Decision |
|---|---:|---|---:|---:|---:|---|
| Model v1 | 68 | Present (`life_percentage`) | 4.0124 | 1.3803 | 0.9959 | Invalid |
| Model v2 | 67 | Removed | 31.0758 | 25.3116 | 0.7557 | Production |
| Optimized baseline | 67 | Removed | 34.4250 | 28.8674 | 0.7002 | Not promoted |

## 13. Production Model Justification

`models/random_forest_v2.joblib` is the production model because it:

- Removes the invalid `life_percentage` feature.
- Uses 67 verified production features.
- Stores `feature_names_in_` for feature-order enforcement.
- Achieves realistic validation performance: RMSE 31.0758 cycles, MAE 25.3116 cycles, R2 0.7557.
- Outperforms the recorded optimized baseline in validation metrics.

## 14. Validation Metrics

Production Model v2 validation metrics:

| Metric | Value |
|---|---:|
| RMSE | 31.0758 cycles |
| MAE | 25.3116 cycles |
| R2 | 0.7557 |
| Training samples | 15,952 |
| Validation samples | 3,988 |

## 15. Production Inference Verification

Production inference was verified on `data/raw/test_FD001.txt`.

| Inference check | Observed value |
|---|---:|
| Input rows | 9,468 |
| Engine count | 100 |
| Prediction count | 9,468 |
| Finite predictions | 9,468 (100.0%) |
| Non-negative predictions | 9,468 (100.0%) |
| Minimum RUL | 17.03 cycles |
| Maximum RUL | 220.63 cycles |
| Mean RUL | 141.85 cycles |
| CSV size | 204,451 bytes |

## 16. Health Classification

Project thresholds:

- Healthy: RUL > 100 cycles
- Normal: 50 < RUL <= 100 cycles
- Warning: 20 < RUL <= 50 cycles
- Critical: RUL <= 20 cycles

Final-cycle health distribution across 100 engines:

| Health status | Engines | Percentage |
|---|---:|---:|
| Healthy | 56 | 56.0% |
| Normal | 36 | 36.0% |
| Warning | 6 | 6.0% |
| Critical | 2 | 2.0% |

All engines were classified exactly once.

## 17. Streamlit Dashboard

Dashboard entry point: `app/dashboard.py`.

Verified pages:

- Home
- Predict RUL
- Dashboard Analytics
- Model Information
- About Project

The dashboard supports sample FD001 loading, production inference, prediction table viewing, health metrics, Plotly charts, warning/critical callouts, and CSV download.

## 18. Testing Strategy

The project uses a 14-test automated suite covering:

- Schema validation.
- Missing/null handling.
- Numeric validation.
- Malformed and empty input handling.
- Feature preparation and leakage removal.
- Production model loading and prediction validity.
- Health status classification.
- End-to-end inference output schema.
- CSV serialization.
- Plotly chart builders.
- Dashboard page imports.

Latest verified result: 14 total, 14 passed, 0 failed.

## 19. Deployment/Package Structure

Verified packaging and deployment files:

- `requirements.txt`
- `requirements-dev.txt`
- `setup.py`
- `Dockerfile`
- `.dockerignore`
- `.streamlit/config.toml`
- `run_dashboard.bat`
- `run_dashboard.sh`
- `run_tests.bat`
- `.github/workflows/ci.yml`

Important limitation: Docker build and GitHub Actions execution were not run locally. They were verified by file inspection only.

## 20. Limitations

- The model is trained for FD001 only, with a single operating condition and single fault mode.
- Predictions are benchmark-project outputs, not certified aircraft maintenance decisions.
- Health categories use fixed thresholds and should be calibrated for operational contexts.
- The current system does not include live telemetry ingestion.
- Docker and CI configuration were inspected locally but not executed locally.

## 21. Verification Status

Milestone 12 verified:

- Production model: PASS
- 67 features: PASS
- Target leakage excluded: PASS
- Automated tests: PASS, 14/14
- FD001 inference: PASS
- Health classification: PASS
- Dashboard imports and utilities: PASS
- Deployment files: PASS by inspection
- Docker build: NOT EXECUTED LOCALLY
- GitHub Actions: NOT EXECUTED LOCALLY

## 22. Final Release Status

Final submission status: READY FOR SUBMISSION.

This means the academic/project submission package is complete and verified with local tests, model audit, inference validation, dashboard verification, and documentation. It does not mean Docker build or GitHub Actions were executed locally.
