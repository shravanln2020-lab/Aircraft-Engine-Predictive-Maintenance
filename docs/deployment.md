# Deployment & Production Operations Guide

## 1. System Environment Requirements

- **Python Version**: `Python 3.11+`
- **Operating System**: Windows / Linux / macOS
- **Production Model Artifact**: `models/random_forest_v2.joblib` (126 MB, 67 features)
- **Primary Dependencies**: `streamlit==1.61.0`, `plotly==6.9.0`, `pandas==3.0.5`,
  `numpy==2.5.1`, `scikit-learn==1.9.0`, `joblib==1.5.3`

---

## 2. Environment Setup Instructions

### A. Create & Activate Virtual Environment

```bash
# Windows
python -m venv .venv
call .venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### B. Install Dependencies from Lockfile
```bash
pip install -r requirements.txt

# For development / testing only:
pip install -r requirements-dev.txt
```

### C. Verify Dependencies
```bash
python -c "import streamlit, plotly, pandas, sklearn, joblib; print('Dependencies verified!')"
```

---

## 3. Streamlit Configuration — `.streamlit/config.toml`

The project includes a `.streamlit/config.toml` that is automatically applied by Streamlit
when launching the dashboard. Key settings:

| Setting | Value | Purpose |
|---|---|---|
| `theme.base` | `dark` | Dark navy UI theme |
| `theme.primaryColor` | `#4f8ef7` | Electric blue accent |
| `server.port` | `8501` | Default HTTP port |
| `server.address` | `0.0.0.0` | Bind all interfaces (Docker-compatible) |
| `server.headless` | `true` | Suppress browser auto-open (CI/Docker) |
| `server.maxUploadSize` | `200` | Max CSV upload size (MB) |

No manual configuration is needed — the file is detected automatically.

---

## 4. Automated Test Suite Execution

Run the complete 14-point unit and integration test suite prior to launching production servers:

```bash
# Windows (batch launcher)
run_tests.bat

# Direct execution
python run_tests.py
```

### Expected Test Output
```
==================================================
Test Suite Execution Summary:
Total Tests : 14
Passed      : 14
Failed      : 0
Elapsed     : 1.150 seconds
==================================================
[SUCCESS] All 14 tests passed successfully!
```

---

## 5. Running the Streamlit Dashboard

### Windows
Double-click `run_dashboard.bat` or run from the command line:
```cmd
run_dashboard.bat
```

### Linux / macOS
```bash
chmod +x run_dashboard.sh   # First time only
./run_dashboard.sh
```

### Direct Streamlit Command (all platforms)
```bash
python -m streamlit run app/dashboard.py --server.port 8501
```

Access the dashboard at: **http://localhost:8501**

---

## 6. Docker Deployment

The project ships with a production-ready `Dockerfile` based on `python:3.11-slim`.

### Build the Container Image
```bash
docker build -t aepm:latest .
```

### Run the Container
```bash
docker run -p 8501:8501 aepm:latest
```

### Access Dashboard
Open **http://localhost:8501** in your browser.

### Docker Notes
- The `.dockerignore` file excludes `.venv/`, `__pycache__/`, `notebooks/`, `reports/figures/`,
  and `.git/` to keep the image size minimal.
- The `HEALTHCHECK` instruction polls `/_stcore/health` every 30 seconds.
- Set `PYTHONPATH=/app` is configured as an environment variable in the image.

---

## 7. Production Inference API Usage

To run batch inference on raw telemetry files programmatically:

```python
import pandas as pd
from src.inference import run_inference_pipeline
from src import config

# 1. Load raw C-MAPSS telemetry file
df = pd.read_csv("data/raw/test_FD001.txt", sep=r"\s+", header=None,
                 names=config.COLUMN_NAMES, engine="python")

# 2. Run end-to-end inference pipeline
predictions_df = run_inference_pipeline(df, model_path="models/random_forest_v2.joblib")

# 3. View predicted RUL and maintenance health status
print(predictions_df.head())
# Output columns: unit_nr, time_cycles, predicted_rul, health_status
```

---

## 8. Output CSV Serialization & Export

Results can be exported to UTF-8 CSV string bytes using dashboard utility helpers:

```python
from app.utils import df_to_csv_bytes

csv_bytes = df_to_csv_bytes(predictions_df)
# Safe for web downloads or file system export
```

---

## 9. Deployment Checklist

Before deploying to production, verify each item:

- [ ] `python run_tests.py` → **14/14 tests pass**
- [ ] `models/random_forest_v2.joblib` exists and is accessible
- [ ] `data/raw/test_FD001.txt` exists (required for sample data button)
- [ ] `reports/results/model_metrics.json` exists (required for Model Info page)
- [ ] `.streamlit/config.toml` is present
- [ ] All imports succeed: `python -c "import streamlit, plotly, pandas, sklearn, joblib"`
- [ ] Dashboard launches at `http://localhost:8501`
- [ ] All 5 dashboard pages render without errors
- [ ] Predict RUL page: CSV upload and "Use Sample Data" button both work
- [ ] Predictions CSV download is functional
