# Project Methodology & Feature Engineering Specification

## 1. NASA C-MAPSS Dataset Overview

The dataset used in this project is the **NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) FD001** benchmark.

### Specifications
- **Operating Conditions**: Single (Sea level)
- **Fault Modes**: Single (HPC Degradation)
- **Training Trajectories**: 100 turbofan engines (Run-to-failure trajectories)
- **Test Trajectories**: 100 turbofan engines (Truncated prior to failure)
- **Sensor Measurements**: 21 time-series sensor channels
- **Operational Settings**: 3 environmental control settings

---

## 2. Mathematical Definition of Remaining Useful Life (RUL)

For an engine unit $i$ at flight cycle $t$:

$$\text{RUL}_{i,t} = \text{MaxCycle}_i - t$$

Where $\text{MaxCycle}_i$ represents the final flight cycle immediately preceding engine breakdown during training.

---

## 3. Automated Feature Engineering

Raw telemetry channels exhibit sensor noise and varying operating scales. To extract physical degradation signals, the system generates four distinct feature representations:

### A. Engine Age & Operational Features
- **Engine Age**: Current flight cycle $t$.
- **Life Percentage** (Training Only): $\frac{t}{\text{MaxCycle}_i} \times 100$. *(Excluded from production feature matrix to eliminate target leakage).*

### B. Sensor Delta Features
Measures instantaneous deviation from the engine's initial operational baseline (first 5 flight cycles):

$$\Delta s_{j,i,t} = s_{j,i,t} - \bar{s}_{j,i,1..5}$$

Where $s_{j,i,t}$ is the reading for sensor $j$ on engine $i$ at cycle $t$.

### C. Rolling Window Statistics
Captures short-term sensor dynamics over a sliding window of $w = 5$ flight cycles:

$$\mu_{j,i,t}^{(w)} = \frac{1}{w} \sum_{k=0}^{w-1} s_{j,i,t-k}$$

$$\sigma_{j,i,t}^{(w)} = \sqrt{\frac{1}{w-1} \sum_{k=0}^{w-1} \left(s_{j,i,t-k} - \mu_{j,i,t}^{(w)}\right)^2}$$

### D. Composite Engine Health Index
Normalizes the primary degrading sensors (High-Pressure Compressor Outlet Temperature $T_{30}$, Low-Pressure Turbine Outlet Temperature $T_{50}$, High-Pressure Compressor Outlet Pressure $P_{30}$) into a unified degradation index normalized between 0.0 (New Engine) and 1.0 (Critical Failure).

---

## 4. Target Leakage Root-Cause Analysis

### Discovery of Target Leakage in Model v1
During Milestone 5, Model v1 achieved unrealistically high validation performance:
- **RMSE**: 4.0124 cycles
- **MAE**: 1.3803 cycles
- **$R^2$ Score**: 0.9959

Investigation in Milestone 6 revealed that `life_percentage` ($\frac{t}{\text{MaxCycle}} \times 100$) was included in the feature matrix. Because $\text{MaxCycle} = t + \text{RUL}$, calculating `life_percentage` directly mathematically encodes the ground truth RUL into the input features:

$$\text{life\_percentage} = \frac{t}{t + \text{RUL}} \times 100 \implies \text{RUL} = t \cdot \left(\frac{100}{\text{life\_percentage}} - 1\right)$$

### Mitigation & Model v2 Refinement
In Milestone 6, `life_percentage` was permanently removed from feature engineering pipelines (`src/evaluate_model.py` and `src/inference.py`). The production model (`random_forest_v2.joblib`) was retrained on a clean 67-feature matrix, yielding honest, realistic generalization metrics:
- **RMSE**: 31.0758 cycles
- **MAE**: 25.3116 cycles
- **$R^2$ Score**: 0.7557

---

## 5. Health Status Classification Thresholds

Predictions are categorized into operational maintenance risk tiers:

| Health Tier | Threshold Condition | Primary Action | Badge Style |
| :--- | :--- | :--- | :--- |
| **Healthy** | $\text{RUL} > 100 \text{ cycles}$ | Routine operational monitoring | Green (`🟢 Healthy`) |
| **Normal** | $50 < \text{RUL} \le 100 \text{ cycles}$ | Standard scheduled maintenance | Blue (`🔵 Normal`) |
| **Warning** | $20 < \text{RUL} \le 50 \text{ cycles}$ | Flag for priority inspection | Yellow (`🟡 Warning`) |
| **Critical** | $\text{RUL} \le 20 \text{ cycles}$ | Immediate grounding & overhaul | Red (`🔴 Critical`) |
