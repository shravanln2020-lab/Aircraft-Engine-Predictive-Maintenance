"""
About Project page for AEPM Dashboard.
Displays project objectives, dataset info, algorithm, team, and guide.
"""

import sys
from pathlib import Path
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.utils import inject_custom_css, page_header, section_header


def render() -> None:
    """Render the About page."""
    inject_custom_css()
    page_header(
        title="About This Project",
        subtitle="Final-Year Engineering Project · Machine Learning · Predictive Maintenance",
        icon="ℹ️",
    )

    # ── Project Card ────────────────────────────────────────────────
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#1a1f35,#1e2538); border:1px solid #2d3147;
                    border-radius:14px; padding:24px 28px; margin-bottom:20px;">
            <h2 style="color:#4f8ef7; margin:0 0 8px;">
                ✈️ Aircraft Engine Predictive Maintenance using Machine Learning
            </h2>
            <p style="color:#c5d0e6; line-height:1.7; margin:0;">
                A production-quality end-to-end machine learning pipeline that predicts the
                <strong>Remaining Useful Life (RUL)</strong> of aircraft turbofan engines using
                multivariate sensor telemetry from the NASA C-MAPSS dataset. The project follows
                industry best practices across nine milestones: from data loading and EDA through
                feature engineering, model training, hyperparameter optimization, and a fully
                interactive Streamlit deployment.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Dataset ─────────────────────────────────────────────────────
    section_header("🗂️ Dataset: NASA C-MAPSS FD001")
    col1, col2 = st.columns(2)
    with col1:
        specs = [
            ("Full Name", "Commercial Modular Aero-Propulsion System Simulation"),
            ("Subset", "FD001 (Single Fault Mode, Single Operating Condition)"),
            ("Training Engines", "100 Units (run-to-failure trajectories)"),
            ("Test Engines", "100 Units (truncated before failure)"),
            ("Sensor Channels", "21 Time-Series Sensor Measurements"),
            ("Operational Settings", "3 Settings (setting_3 = constant)"),
        ]
        for label, val in specs:
            st.markdown(
                f"""
                <div style="display:flex; gap:12px; padding:8px 14px; background:#1a1d2e;
                            border-radius:8px; margin:3px 0; border:1px solid #2d3147;">
                    <span style="color:#8b9dc3; min-width:160px;">{label}</span>
                    <span style="color:#e2e8f0;">{val}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with col2:
        st.markdown(
            """
            <div style="background:#1e2236; border:1px solid #2d3147; border-radius:12px; padding:20px;">
                <p style="color:#8b9dc3; font-size:0.85rem; margin:0 0 8px;">
                    📌 Dataset Source
                </p>
                <p style="color:#e2e8f0;">
                    NASA Prognostics Center of Excellence (PCoE) <br>
                    <a href="https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/"
                       style="color:#4f8ef7;">ti.arc.nasa.gov/pcoe</a>
                </p>
                <hr style="border-color:#2d3147; margin:14px 0;">
                <p style="color:#8b9dc3; font-size:0.85rem; margin:0 0 8px;">
                    📌 RUL Definition
                </p>
                <p style="color:#e2e8f0;">
                    RUL<sub>train</sub> = Max Cycle<sub>engine</sub> − Current Cycle <br>
                    RUL<sub>test</sub> = Ground Truth RUL + (Max Test Cycle − Current Cycle)
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Algorithm ───────────────────────────────────────────────────
    st.markdown("---")
    section_header("🌲 Algorithm: Random Forest Regressor")
    st.markdown(
        """
        **Random Forest** is an ensemble learning algorithm that builds multiple decision trees
        (using bootstrap sampling) and averages their predictions to improve accuracy
        and reduce overfitting. Key properties for this project:

        - **Non-linear**: Captures complex sensor degradation patterns without feature normalization
        - **Robust**: Handles sensor noise and collinearity across 67+ features
        - **Interpretable**: Provides native feature importance rankings for domain analysis
        - **Efficient**: Parallelizable across CPU cores for fast training (`n_jobs=-1`)
        """,
    )

    # ── Workflow ─────────────────────────────────────────────────────
    st.markdown("---")
    section_header("🗺️ Project Workflow")
    workflow_steps = [
        ("1️⃣", "Data Ingestion", "Load NASA C-MAPSS FD001 text files into structured pandas DataFrames with validated schemas."),
        ("2️⃣", "Exploratory Data Analysis", "Analyze sensor distributions, engine flight cycles, missing values, correlations, and generate visualizations."),
        ("3️⃣", "Preprocessing", "Calculate RUL targets, identify zero-variance features, clean duplicates, and export processed CSVs."),
        ("4️⃣", "Feature Engineering", "Create Engine Age, Sensor Deltas, 5-Cycle Rolling Statistics, and normalized Engine Health Index."),
        ("5️⃣", "Model Training", "Train RandomForestRegressor baseline with 80/20 train-validation split and evaluate RMSE, MAE, R²."),
        ("6️⃣", "Model Evaluation", "Identify and remove target leakage (`life_percentage`). Generate evaluation plots and JSON report."),
        ("7️⃣", "Hyperparameter Optimization", "5-Fold RandomizedSearchCV across n_estimators, max_depth, min_samples, max_features."),
        ("8️⃣", "Prediction Pipeline", "Modular inference pipeline: validate → feature engineering → predict → health classify → export."),
        ("9️⃣", "Dashboard Deployment", "Interactive Streamlit dashboard with upload, prediction, analytics, and model info pages."),
    ]
    for icon, title, desc in workflow_steps:
        st.markdown(
            f"""
            <div style="display:flex; gap:16px; padding:12px 18px; background:#1a1d2e;
                        border-radius:10px; margin:5px 0; border:1px solid #2d3147;">
                <span style="font-size:1.3rem; min-width:30px;">{icon}</span>
                <div>
                    <p style="color:#4f8ef7; font-weight:600; margin:0;">{title}</p>
                    <p style="color:#c5d0e6; margin:2px 0 0; font-size:0.87rem;">{desc}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Health Thresholds ────────────────────────────────────────────
    st.markdown("---")
    section_header("🚦 Engine Health Classification Thresholds")
    thresholds = [
        ("🟢", "Healthy", "RUL > 100 cycles", "#34d399", "#064e3b"),
        ("🔵", "Normal", "50 < RUL ≤ 100 cycles", "#38bdf8", "#0c4a6e"),
        ("🟡", "Warning", "20 < RUL ≤ 50 cycles", "#fbbf24", "#713f12"),
        ("🔴", "Critical", "RUL ≤ 20 cycles", "#f87171", "#7f1d1d"),
    ]
    th_cols = st.columns(4)
    for col, (icon, label, rng, fg, bg) in zip(th_cols, thresholds):
        col.markdown(
            f"""
            <div style="background:{bg}; border:1px solid {fg}; border-radius:12px;
                        padding:18px; text-align:center;">
                <div style="font-size:1.8rem;">{icon}</div>
                <p style="color:{fg}; font-weight:700; margin:6px 0;">{label}</p>
                <p style="color:#c5d0e6; font-size:0.82rem; margin:0;">{rng}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Team ─────────────────────────────────────────────────────────
    st.markdown("---")
    section_header("👥 Project Team")
    t1, t2 = st.columns(2)
    team_members = [
        ("👨‍💻", "Student Name 1", "Final Year B.E. Student", "Dept. of Computer Science & Engineering"),
        ("👩‍💻", "Student Name 2", "Final Year B.E. Student", "Dept. of Computer Science & Engineering"),
    ]
    for col, (icon, name, role, dept) in zip([t1, t2], team_members):
        col.markdown(
            f"""
            <div style="background:#1e2236; border:1px solid #2d3147; border-radius:12px;
                        padding:20px; text-align:center;">
                <div style="font-size:2.5rem; margin-bottom:8px;">{icon}</div>
                <p style="color:#e2e8f0; font-weight:700; font-size:1.05rem; margin:0;">{name}</p>
                <p style="color:#4f8ef7; font-size:0.87rem; margin:4px 0;">{role}</p>
                <p style="color:#8b9dc3; font-size:0.82rem; margin:0;">{dept}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    section_header("🎓 Project Guide")
    st.markdown(
        """
        <div style="background:#1e2236; border:1px solid #2d3147; border-radius:12px;
                    padding:20px 28px; display:flex; align-items:center; gap:20px;">
            <div style="font-size:2.5rem;">👨‍🏫</div>
            <div>
                <p style="color:#e2e8f0; font-weight:700; font-size:1.05rem; margin:0;">
                    Guide / Supervisor Name
                </p>
                <p style="color:#4f8ef7; font-size:0.87rem; margin:4px 0;">
                    Assistant Professor, Dept. of Computer Science & Engineering
                </p>
                <p style="color:#8b9dc3; font-size:0.82rem; margin:0;">
                    Institution Name · Academic Year 2025–2026
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
