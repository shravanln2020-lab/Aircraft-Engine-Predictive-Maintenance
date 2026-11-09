# Model Evaluation & Verification Report

## 1. Summary of Model Iterations

Three distinct model iterations were trained and evaluated during project development on the NASA C-MAPSS FD001 dataset:

| Model Iteration | Feature Dimensions | Target Leakage Status | Validation RMSE | Validation MAE | Validation $R^2$ Score | Production Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Model v1** | 68 | ❌ Present (`life_percentage`) | 4.0124 | 1.3803 | 0.9959 | **INVALID** (Unrealistic) |
| **Model v2 (Production)** | 67 | ✅ Removed | **31.0758** | **25.3116** | **0.7557** | **PRODUCTION READY** |
| **Optimized Model** | 67 | ✅ Removed | 34.4250 | 28.8674 | 0.7002 | Baseline Reference |

---

## 2. Detailed Performance Analysis

### A. Model v1 (Target Leakage Baseline)
Model v1 included `life_percentage`, yielding near-perfect metrics (RMSE: 4.01 cycles, $R^2$: 0.9959). However, because $\text{life\_percentage}$ requires total engine life ($\text{MaxCycle}$), which is unknown during real-time flight operations, Model v1 cannot be deployed in production.

### B. Model v2 (Production Model — `models/random_forest_v2.joblib`)
Model v2 excludes all target-derived metrics. Retrained on 15,952 flight cycles across 67 engineered features (Sensor Deltas, 5-cycle Rolling Means, 5-cycle Rolling Stds, and Engine Health Index), Model v2 achieves an RMSE of **31.08 cycles** and MAE of **25.31 cycles** with an $R^2$ score of **0.7557**. This model demonstrates strong generalization without overfitting or leakage.

### C. Optimized Model (Hyperparameter Tuning Benchmark)
Hyperparameter search via 5-Fold `RandomizedSearchCV` (`n_estimators=100`, `max_depth=30`, `max_features='sqrt'`) achieved an RMSE of **34.43 cycles** and $R^2$ of **0.7002**. Model v2 (`models/random_forest_v2.joblib`) outperformed the tuned candidate on validation metrics and was selected as the final production model artifact.

---

## 3. Feature Importance Analysis

Relative Gini Feature Importance rankings for top sensor channels in Model v2:

1. **`s_11_roll_mean_5`** (Static Pressure at HPC Outlet — 5-cycle Mean) — **0.1842**
2. **`s_4_roll_mean_5`** (LPT Outlet Temperature — 5-cycle Mean) — **0.1415**
3. **`s_12_roll_mean_5`** (Ratio of Fuel Flow to Ps30 — 5-cycle Mean) — **0.1108**
4. **`health_index`** (Composite Engine Health Index) — **0.0924**
5. **`engine_age`** (Cumulative Flight Cycle Count) — **0.0815**
6. **`s_7_roll_mean_5`** (HPC Outlet Pressure — 5-cycle Mean) — **0.0654**

---

## 4. Verification Check Matrix

| Verification Check | Target Standard | Verified Result | Status |
| :--- | :--- | :--- | :---: |
| **Model File Integrity** | `models/random_forest_v2.joblib` exists and loads cleanly | Successfully loaded via `joblib` | ✅ PASS |
| **Feature Ordering Enforcement** | Feature matrix matches `model.feature_names_in_` | 67 features ordered strictly | ✅ PASS |
| **Target Leakage Exclusion** | `life_percentage` excluded from feature matrix & model | `life_percentage` not present | ✅ PASS |
| **Prediction Finiteness** | Predictions contain 0 NaN or Infinite values | `np.isfinite(preds).all() == True` | ✅ PASS |
| **Domain Bound Check** | Predicted RUL is strictly non-negative ($\ge 0.0$) | `(preds >= 0.0).all() == True` | ✅ PASS |
| **Automated Test Suite** | 100% test pass rate across 14 test cases | 14 Passed / 0 Failed | ✅ PASS |
| **Deployment Readiness** | System ready for Streamlit deployment | Status: `READY_FOR_DEPLOYMENT` | ✅ PASS |
