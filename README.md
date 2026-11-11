<img width="1917" height="912" alt="Screenshot 2026-10-25 190541" src="https://github.com/user-attachments/assets/a0fefa9c-5940-4833-92b3-0589da9fca42" />
<img width="1917" height="908" alt="Screenshot 2026-10-25 190612" src="https://github.com/user-attachments/assets/9cb2f477-2668-4c14-8c4a-78ee604444d2" />
<img width="1916" height="910" alt="Screenshot 2026-10-25 190652" src="https://github.com/user-attachments/assets/268245d7-5eba-4975-a4b0-62bb9bd7642d" />
<img width="1892" height="827" alt="Screenshot 2026-10-25 190719" src="https://github.com/user-attachments/assets/4fc6cc32-b411-4a5b-bfb3-6d3c7d525b33" />
<img width="1917" height="837" alt="Screenshot 2026-10-25 190751" src="https://github.com/user-attachments/assets/fcd8592e-c2bb-45d8-b911-1ed91316fa3f" />
<img width="1906" height="888" alt="Screenshot 2026-10-25 190823" src="https://github.com/user-attachments/assets/2a23abb0-e36f-492b-82b1-95878c1fc39a" />
# ✈️ Aircraft Engine Predictive Maintenance (AEPM) using Machine Learning

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Random--Forest-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive--Charts-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/)
[![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?logo=github-actions&logoColor=white)](.github/workflows/ci.yml)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](Dockerfile)
[![Status](https://img.shields.io/badge/Status-Production_Ready-34D399)](#)

A production-grade, end-to-end Machine Learning platform designed to predict the **Remaining Useful Life (RUL)** of turbofan engines using multivariate sensor telemetry from the **NASA C-MAPSS FD001** benchmark dataset.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Dataset Description](#-dataset-description)
- [System Architecture](#-system-architecture)
- [Preprocessing & Feature Engineering](#-preprocessing--feature-engineering)
- [Target Leakage Discovery & Fix](#-target-leakage-discovery--fix)
- [Model Training & Evaluation](#-model-training--evaluation)
- [Production Inference Pipeline](#-production-inference-pipeline)
- [Interactive Streamlit Dashboard](#-interactive-streamlit-dashboard)
- [Docker Deployment](#-docker-deployment)
- [Limitations & Future Enhancements](#-limitations--future-enhancements)
- [Setup & Run Instructions](#-setup--run-instructions)
- [Automated Testing & Verification](#-automated-testing--verification)
- [Final Submission Documents](#final-submission-documents)
- [Changelog](#-changelog)

---

## 📌 Project Overview

Unplanned aircraft engine failures lead to severe economic disruption, flight cancellations, and critical safety hazards. **Predictive Maintenance (PdM)** utilizes real-time sensor telemetry to estimate the **Remaining Useful Life (RUL)** of an engine — the number of operational flight cycles remaining before component failure.

This project delivers a complete 12-milestone engineering pipeline: from data ingestion and exploratory data analysis (EDA), through domain-specific feature engineering, target leakage elimination, Random Forest Regressor training, 14-point unit test validation, final release verification, and demonstration via an interactive **Streamlit** web dashboard.

---

## 🎯 Problem Statement

Given a stream of 21 sensor measurements and 3 operational control settings recorded during flight cycles of turbofan engines, build a machine learning regression model $f(\mathbf{X}) \to \hat{y}$ that predicts the remaining flight cycles $\text{RUL} \in [0, \infty)$ and classifies the engine into maintenance health tiers:

- 🟢 **Healthy**: $\text{RUL} > 100 \text{ cycles}$ (Routine operation)
- 🔵 **Normal**: $50 < \text{RUL} \le 100 \text{ cycles}$ (Standard maintenance)
- 🟡 **Warning**: $20 < \text{RUL} \le 50 \text{ cycles}$ (Priority inspection)
- 🔴 **Critical**: $\text{RUL} \le 20 \text{ cycles}$ (Immediate overhaul)

---

## 🗂️ Dataset Description

The system uses the **NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) FD001** dataset.

- **Training Engine Units**: 100 turbofan engines (Run-to-failure trajectories)
- **Test Engine Units**: 100 turbofan engines (Truncated before failure)
- **Total Telemetry Channels**: 26 columns (`unit_nr`, `time_cycles`, `setting_1..3`, `s_1..s_21`)
- **Operating Conditions**: Single operating condition (Sea level)
- **Fault Mode**: Single fault mode (HPC degradation)

---

## 🏛️ System Architecture

```
AEPM Workspace
│
├── app/                        # Streamlit Interactive Dashboard
│   ├── dashboard.py            # Entry point & sidebar router
│   ├── utils.py                # Layout CSS & Plotly chart builders
│   └── pages/                  # Page modules (Home, Predict, Analytics, Model Info, About)
│
├── src/                        # Modular ML Engine
│   ├── config.py               # Central paths & column schema
│   ├── validator.py            # Schema, null cell & dtype validation
│   ├── preprocessing.py        # Baseline calculation & constant filtering
│   ├── feature_engineering.py  # Age, sensor deltas, rolling stats & health index
│   ├── evaluate_model.py       # Metrics calculation & leakage removal
│   ├── predictor.py            # Model loading & feature ordering check
│   ├── health_status.py        # Health category tier classification
│   └── inference.py            # End-to-end batch prediction pipeline
│
├── models/                     # Serialized Model Artifacts
│   └── random_forest_v2.joblib # Production model (67 features, Leakage-free)
│
├── docs/                       # Technical Documentation Suite
│   ├── architecture.md         # System architecture specification
│   ├── methodology.md          # Technical methodology & formulas
│   ├── model_evaluation.md     # Model comparison & performance metrics
│   └── deployment.md           # Operations & deployment guide
│
├── reports/                    # Results & Final Validation Report
│   └── final_validation_report.json
│
├── tests/                      # Automated Unit & Integration Tests
│   ├── test_pipeline.py        # Pipeline integration tests
│   └── test_dashboard.py       # Dashboard & utility unit tests
│
├── run_tests.py                # Test runner script
└── run_dashboard.bat           # Windows 1-click launcher
```

---

## ⚙️ Preprocessing & Feature Engineering

1. **Validation & Cleaning**: Checks schema integrity, null cell thresholds, and eliminates uninformative operational channels (`setting_3`).
2. **Engine Age & Baseline Deltas**: Calculates cumulative flight cycles and instantaneous sensor deviations relative to initial flight baseline:
   $$\Delta s_{j,t} = s_{j,t} - \bar{s}_{j,1..5}$$
3. **Rolling Window Statistics**: Computes 5-flight-cycle sliding window means ($\mu_{5}$) and standard deviations ($\sigma_{5}$) for all active sensor channels.
4. **Engine Health Index**: Constructs a normalized composite degradation index using key thermal and pressure degradation signals ($T_{30}, T_{50}, P_{30}$).

---

## 🚨 Target Leakage Discovery & Fix

During initial modeling (Model v1), validation performance yielded unrealistic metrics:
- **RMSE**: 4.0124 cycles | **$R^2$ Score**: 0.9959

### Root Cause
Features included `life_percentage` ($\frac{t}{\text{MaxCycle}} \times 100$). Because $\text{MaxCycle} = t + \text{RUL}$, this feature directly encodes the target variable $\text{RUL}$ into input data, creating **severe target leakage**.

### Production Fix
`life_percentage` was permanently removed from feature engineering pipelines (`src/evaluate_model.py`, `src/inference.py`). The production model (`models/random_forest_v2.joblib`) was retrained on a clean 67-feature matrix, establishing genuine operational performance.

---

## 📊 Model Training & Evaluation

| Model Iteration | Feature Count | Target Leakage Status | Validation RMSE | Validation MAE | Validation $R^2$ | Production Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Model v1** | 68 | ❌ Present (`life_percentage`) | 4.0124 | 1.3803 | 0.9959 | **INVALID** |
| **Model v2 (Production)** | **67** | **✅ Removed** | **31.0758** | **25.3116** | **0.7557** | **PRODUCTION READY** |
| **Optimized Model** | 67 | ✅ Removed | 34.4250 | 28.8674 | 0.7002 | Baseline Reference |

> **Production Model Confirmation**: `models/random_forest_v2.joblib` (RandomForestRegressor, 67 features, RMSE 31.08 cycles, $R^2$ 0.7557).

---

## 🚀 Production Inference Pipeline

The end-to-end pipeline processes raw telemetry data automatically:

```python
import pandas as pd
from src.inference import run_inference_pipeline
from src import config

# Load raw C-MAPSS telemetry file
df = pd.read_csv("data/raw/test_FD001.txt", sep=r"\s+", header=None, names=config.COLUMN_NAMES, engine="python")

# Run production pipeline
results_df = run_inference_pipeline(df, model_path="models/random_forest_v2.joblib")

# Output DataFrame contains: unit_nr, time_cycles, predicted_rul, health_status
print(results_df.head())
```

---

## 🌐 Interactive Streamlit Dashboard

The multi-page dashboard offers wide layout analytics and real-time inference:

1. 🏠 **Home Page**: Hero metrics, project objectives, tech stack, and 9-step project milestone checklist.
2. 🔮 **Predict RUL Page**: CSV/TXT uploader, "⚡ Use Sample FD001 Test Data" button, predictions table, health status cards, Plotly charts, and CSV download.
3. 📊 **Dashboard Analytics Page**: Fleet health distribution donut chart, RUL histogram, engine-wise final RUL bar chart, multi-engine degradation line chart with interactive multi-selector, critical engine table, and CSV download.
4. 🤖 **Model Information Page**: Production model metrics, model progression table, 67-feature breakdown, hyperparameter grid, and top 20 feature importances Plotly chart.
5. ℹ️ **About Project Page**: Background methodology, dataset specifications, algorithm breakdown, health thresholds legend, and student/guide details.

---

## 🐳 Docker Deployment

The AEPM dashboard ships with a production-ready `Dockerfile` using `python:3.11-slim`.

```bash
# 1. Build the image
docker build -t aepm:latest .

# 2. Run the container
docker run -p 8501:8501 aepm:latest

# 3. Open the dashboard
# → http://localhost:8501
```

The `.dockerignore` file excludes `.venv/`, `__pycache__/`, `notebooks/`, and `reports/figures/`
to keep the image lean.

---

## 🛠️ Setup & Run Instructions

### Prerequisites
- Python `3.11+`
- Windows / Linux / macOS

### Installation & Execution

1. **Create & Activate Virtual Environment**:
   ```cmd
   # Windows
   python -m venv .venv
   call .venv\Scripts\activate

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   # For development/testing:
   pip install -r requirements-dev.txt
   ```

3. **Run Automated Test Suite**:
   ```bash
   # Windows
   run_tests.bat

   # Linux / macOS / direct
   .venv/Scripts/python.exe run_tests.py
   ```

4. **Launch Streamlit Dashboard**:
   ```bash
   # Windows (double-click or run)
   run_dashboard.bat

   # Linux / macOS
   chmod +x run_dashboard.sh
   ./run_dashboard.sh

   # Direct command
   python -m streamlit run app/dashboard.py --server.port 8501
   ```

5. Open browser at: `http://localhost:8501`

---

## ⚠️ Limitations & Future Enhancements

- **Operating Conditions**: Current model is trained specifically on C-MAPSS FD001 (single operating condition). Future extensions will incorporate FD002–FD004 (multi-condition, multi-fault datasets).
- **Deep Learning Comparison**: Future iterations can compare Random Forest against LSTM / Temporal Convolutional Networks (TCN) for sequential degradation modeling.
- **Real-Time Streaming**: Production extension could integrate Apache Kafka for real-time telemetry ingestion.
- **REST API Layer**: A FastAPI microservice wrapper around `run_inference_pipeline()` would enable programmatic access.

---

## 📋 Changelog

See [CHANGELOG.md](CHANGELOG.md) for a complete milestone-by-milestone history of all changes.

---

## ✅ Automated Testing & Verification

The project includes a 14-point automated test suite covering all subsystems:

| # | Test | Module | Description |
|---|------|--------|-------------|
| 1 | `test_input_schema_validation` | Pipeline | Schema completeness + missing column error |
| 2 | `test_missing_values_validation` | Pipeline | Null threshold enforcement |
| 3 | `test_numeric_types_validation` | Pipeline | Non-numeric column detection |
| 4 | `test_malformed_and_empty_input_handling` | Pipeline | Empty DataFrame error handling |
| 5 | `test_feature_preparation_and_leakage_removal` | Pipeline | `life_percentage` exclusion |
| 6 | `test_rul_prediction_validity` | Pipeline | Finite, non-negative predictions |
| 7 | `test_health_status_classification` | Pipeline | 4-tier health threshold boundaries |
| 8 | `test_end_to_end_inference_pipeline` | Pipeline | Full pipeline output schema |
| 9 | `test_dashboard_utils` | Pipeline | CSV bytes + badge generation |
| 10 | `test_status_badge` | Dashboard | HTML badge for all 4 statuses |
| 11 | `test_df_to_csv_bytes` | Dashboard | DataFrame → bytes conversion |
| 12 | `test_load_json_safe` | Dashboard | JSON safe-load including missing file |
| 13 | `test_plotly_chart_builders` | Dashboard | All 4 Plotly figure builders |
| 14 | `test_page_modules_import` | Dashboard | All 5 page modules importable |

**Current Status**: 14/14 tests passing

```bash
# Run via Windows batch
run_tests.bat

# Run directly
python run_tests.py
```

---

## Final Submission Documents

Milestone 12 finalization artifacts:

- [Final Project Report](docs/final_project_report.md)
- [Presentation Summary](docs/presentation_summary.md)
- [Live Demo Walkthrough](docs/walkthrough.md)
- [Release Checklist](reports/release_checklist.md)
- [Final Demo Report JSON](reports/final_demo_report.json)

Submission status: **READY FOR SUBMISSION**. Docker build and GitHub Actions execution are documented as not executed locally.
