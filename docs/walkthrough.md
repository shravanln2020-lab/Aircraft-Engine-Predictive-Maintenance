# AEPM - Live Demo Walkthrough

This is a practical script for presenting AEPM v1.0.0. It uses verified Milestone 12 values from the current workspace. Do not claim Docker build or GitHub Actions execution during the demo; those were inspected but not run locally.

## 1. Start Project

What to click/type: Open the AEPM folder in VS Code.

Evaluator should see: Folders such as `src/`, `app/`, `data/`, `models/`, `tests/`, `docs/`, and `reports/`.

Explain verbally: This is a complete ML project workspace, not only a notebook.

## 2. Launch Streamlit Dashboard

What to click/type:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/dashboard.py --server.port 8501
```

Evaluator should see: Streamlit starts and serves the app at `http://localhost:8501`.

Explain verbally: The dashboard is the live interface for inference and analytics.

## 3. Show Home Page

What to click/type: Open the dashboard URL and select Home in the sidebar.

Evaluator should see: Project overview, dataset/model context, and dashboard navigation.

Explain verbally: AEPM predicts aircraft engine Remaining Useful Life using NASA C-MAPSS FD001 telemetry.

## 4. Explain Predict RUL Page

What to click/type: Click Predict RUL in the sidebar.

Evaluator should see: File uploader and sample FD001 test data button.

Explain verbally: This page accepts telemetry data and runs the production inference pipeline.

## 5. Load Sample FD001 Data

What to click/type: Click `Use Sample FD001 Test Data`.

Evaluator should see: A preview of loaded FD001 telemetry.

Explain verbally: The sample file is the existing repository file `data/raw/test_FD001.txt`, not an invented dataset.

## 6. Run Prediction

What to click/type: Click the Run Prediction Pipeline button.

Evaluator should see: The app processes the sample data and produces predictions.

Explain verbally: The dashboard calls `src.inference.run_inference_pipeline` with `models/random_forest_v2.joblib`.

## 7. Show Prediction Table

What to click/type: Scroll to the prediction results table.

Evaluator should see: Columns `unit_nr`, `time_cycles`, `predicted_rul`, and `health_status`.

Explain verbally: The verified inference run produced 9,468 predictions for 100 engines.

## 8. Show Health Metrics

What to click/type: Point to the health metric cards.

Evaluator should see: Counts for Healthy, Normal, Warning, and Critical predictions.

Explain verbally: RUL is converted into interpretable maintenance categories.

## 9. Show Plotly Charts

What to click/type: Scroll to the visualizations section.

Evaluator should see: Interactive health distribution, RUL histogram, and degradation charts.

Explain verbally: Plotly charts make fleet-level and engine-level behavior easier to inspect.

## 10. Show Critical/Warning Engines

What to click/type: Scroll to the maintenance callout section if warning or critical records are present.

Evaluator should see: Engines with Warning or Critical status highlighted.

Explain verbally: Final-cycle health distribution is Healthy 56, Normal 36, Warning 6, Critical 2 across 100 engines.

## 11. Download Prediction CSV

What to click/type: Click the CSV download button.

Evaluator should see: A browser download prompt for prediction results.

Explain verbally: CSV serialization was verified; the output size was 204,451 bytes.

## 12. Open Dashboard Analytics

What to click/type: Click Dashboard Analytics in the sidebar.

Evaluator should see: Fleet analytics populated from session predictions or generated sample predictions.

Explain verbally: This page summarizes the same production inference outputs.

## 13. Show Fleet Health Distribution

What to click/type: Point to the health distribution donut chart.

Evaluator should see: Health categories displayed visually.

Explain verbally: The final-cycle fleet distribution is 56% Healthy, 36% Normal, 6% Warning, and 2% Critical.

## 14. Show RUL Histogram

What to click/type: Point to the predicted RUL histogram.

Evaluator should see: Distribution of predicted RUL values with threshold markers.

Explain verbally: Verified RUL range is 17.03 to 220.63 cycles, with mean 141.85 cycles.

## 15. Show Engine-Wise RUL

What to click/type: Scroll to the final RUL per engine chart.

Evaluator should see: Engine-level bars sorted by final predicted RUL.

Explain verbally: This helps identify engines that should be prioritized for inspection.

## 16. Show Degradation Trajectory

What to click/type: Select several engine IDs in the multiselect control.

Evaluator should see: Line charts showing predicted RUL over cycles.

Explain verbally: The trajectory view shows how predicted remaining life changes over recorded operating cycles.

## 17. Open Model Information

What to click/type: Click Model Information in the sidebar.

Evaluator should see: Production model metrics and model comparison details.

Explain verbally: This page documents why Model v2 is the production model.

## 18. Explain 67 Features

What to click/type: Point to the feature information section.

Evaluator should see: Production feature count and related model details.

Explain verbally: The production model uses 67 features and `feature_names_in_` is available for ordering.

## 19. Explain Leakage Removal

What to click/type: Point to the model comparison or leakage explanation section.

Evaluator should see: Model v1 marked invalid and Model v2 marked production.

Explain verbally: `life_percentage` was invalid because it depends on total engine life and leaks target information.

## 20. Show Validation Metrics

What to click/type: Point to Model v2 metrics.

Evaluator should see: RMSE 31.0758, MAE 25.3116, R2 0.7557.

Explain verbally: These are the realistic leakage-free validation metrics.

## 21. Open About Project

What to click/type: Click About Project in the sidebar.

Evaluator should see: Project background, dataset notes, and methodology summary.

Explain verbally: The project connects predictive maintenance concepts with an implemented ML workflow.

## 22. Explain Overall Workflow

What to click/type: Use the sidebar pages or documentation as reference.

Evaluator should see: A coherent flow from raw data to final dashboard.

Explain verbally: Data validation, feature engineering, inference, classification, visualization, and export are all connected.

## 23. Show Tests

What to click/type:

```powershell
.\.venv\Scripts\python.exe run_tests.py
```

Evaluator should see: 14 tests pass with 0 failures.

Explain verbally: The tests cover pipeline validation, leakage exclusion, prediction validity, dashboard utilities, and page imports.

## 24. Explain Deployment Readiness

What to click/type: Show `requirements.txt`, `Dockerfile`, `.streamlit/config.toml`, launcher scripts, and `.github/workflows/ci.yml`.

Evaluator should see: Packaging and deployment configuration files.

Explain verbally: The project is ready for submission. Docker build and GitHub Actions were not executed locally, so they are documented as not locally executed rather than claimed as completed.
