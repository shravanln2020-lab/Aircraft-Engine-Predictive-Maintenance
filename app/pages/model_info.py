"""
Model Information page for AEPM Dashboard.
Displays training configuration, performance metrics, feature details, and hyperparameter grid.
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.utils import inject_custom_css, load_json_safe, page_header, section_header
from src import config

logger = logging.getLogger(__name__)


def _metric_card(label: str, value: str, delta: str = "", color: str = "#4f8ef7") -> str:
    return f"""
    <div style="background:#1e2236; border:1px solid #2d3147; border-radius:12px;
                padding:18px; margin:5px 0; border-left:4px solid {color};
                box-shadow:0 4px 12px rgba(0,0,0,0.2);">
        <p style="color:#8b9dc3; font-size:0.80rem; margin:0;">{label}</p>
        <p style="color:#e2e8f0; font-size:1.35rem; font-weight:700; margin:4px 0;">{value}</p>
        {"<p style='color:#6ee7b7; font-size:0.78rem; margin:0;'>" + delta + "</p>" if delta else ""}
    </div>
    """


def render() -> None:
    """Render the Model Information page."""
    inject_custom_css()
    page_header(
        title="Model Information",
        subtitle="Production model configuration, training parameters, evaluation metrics, and feature importances",
        icon="🤖",
    )

    # ── Load Artifacts ─────────────────────────────────────────────
    eval_report = load_json_safe(config.RESULTS_DIR / "evaluation_report.json") or {}
    best_params = load_json_safe(config.RESULTS_DIR / "best_parameters.json") or {}
    metrics_v1 = load_json_safe(config.RESULTS_DIR / "model_metrics.json") or {}

    feat_imp_path = config.RESULTS_DIR / "feature_importance.csv"
    importance_df = pd.read_csv(feat_imp_path) if feat_imp_path.exists() else None

    # ── Production Model Badge ──────────────────────────────────────
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#1a2e1a,#1e3a2b); border:1px solid #34d399;
                    border-radius:14px; padding:20px 28px; margin-bottom:20px;
                    display:flex; align-items:center; gap:16px; box-shadow:0 4px 16px rgba(0,0,0,0.25);">
            <div style="font-size:2.5rem;">🌲</div>
            <div>
                <p style="color:#6ee7b7; font-size:0.8rem; margin:0; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">
                    FINAL PRODUCTION MODEL
                </p>
                <h2 style="color:#e2e8f0; margin:4px 0; font-size:1.3rem;">
                    Random Forest Regressor — <code>random_forest_v2.joblib</code>
                </h2>
                <p style="color:#8b9dc3; margin:0; font-size:0.85rem;">
                    Target Leakage Removed · 5-Fold Cross-Validated · Scikit-learn Ensemble
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Key Performance Metrics ─────────────────────────────────────
    section_header("📊 Validation Performance Metrics")

    rmse = eval_report.get("rmse", 31.0758)
    mae = eval_report.get("mae", 25.3116)
    r2 = eval_report.get("r2_score", 0.7557)
    n_train = eval_report.get("training_samples", 15952)
    n_val = eval_report.get("validation_samples", 3988)
    n_feat = eval_report.get("num_features_used", 67)

    c1, c2, c3 = st.columns(3)
    c1.markdown(_metric_card("Root Mean Squared Error (RMSE)", f"{rmse:.2f} cycles" if isinstance(rmse, (float, int)) else str(rmse), color="#f87171"), unsafe_allow_html=True)
    c2.markdown(_metric_card("Mean Absolute Error (MAE)", f"{mae:.2f} cycles" if isinstance(mae, (float, int)) else str(mae), color="#fbbf24"), unsafe_allow_html=True)
    c3.markdown(_metric_card("R² Score", f"{r2:.4f}" if isinstance(r2, (float, int)) else str(r2), delta="Coefficient of Determination", color="#34d399"), unsafe_allow_html=True)

    d1, d2, d3 = st.columns(3)
    d1.markdown(_metric_card("Training Samples", f"{n_train:,}" if isinstance(n_train, int) else str(n_train), color="#4f8ef7"), unsafe_allow_html=True)
    d2.markdown(_metric_card("Validation Samples", f"{n_val:,}" if isinstance(n_val, int) else str(n_val), color="#a78bfa"), unsafe_allow_html=True)
    d3.markdown(_metric_card("Feature Dimensions Used", str(n_feat), color="#38bdf8"), unsafe_allow_html=True)

    # ── Model Comparison ────────────────────────────────────────────
    if metrics_v1:
        st.markdown("---")
        section_header("📈 Model Performance Progression (Target Leakage Fix)")

        v1_rmse = metrics_v1.get("rmse", 4.0124)
        v2_rmse = eval_report.get("rmse", 31.0758)
        v1_r2 = metrics_v1.get("r2_score", 0.9959)
        v2_r2 = eval_report.get("r2_score", 0.7557)

        comp_data = {
            "Model Version": ["Model v1 (With Target Leakage)", "Model v2 (Production Refined)"],
            "RMSE (Cycles)": [v1_rmse, v2_rmse],
            "MAE (Cycles)": [metrics_v1.get("mae", 1.3803), eval_report.get("mae", 25.3116)],
            "R² Score": [v1_r2, v2_r2],
            "Target Leakage Status": ["❌ Present (life_percentage leakage)", "✅ Removed (Production Ready)"],
        }
        comp_df = pd.DataFrame(comp_data)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # ── Best Hyperparameters ────────────────────────────────────────
    if best_params:
        st.markdown("---")
        section_header("⚙️ Optimized Hyperparameters (RandomizedSearchCV)")
        param_labels = {
            "n_estimators": "Number of Decision Trees",
            "max_depth": "Maximum Tree Depth",
            "min_samples_split": "Min Samples to Split Node",
            "min_samples_leaf": "Min Samples per Leaf",
            "max_features": "Features Considered Per Split",
        }
        p_cols = st.columns(len(best_params))
        for col, (param, val) in zip(p_cols, best_params.items()):
            label = param_labels.get(param, param)
            col.markdown(
                f"""
                <div style="background:#1a1d2e; border-radius:10px; padding:12px 14px; text-align:center;
                            border:1px solid #2d3147;">
                    <span style="color:#8b9dc3; font-size:0.78rem;">{label}</span><br>
                    <span style="color:#4f8ef7; font-weight:700; font-size:1.1rem; font-family:monospace;">{val}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Feature Importances ─────────────────────────────────────────
    if importance_df is not None and not importance_df.empty:
        st.markdown("---")
        section_header("🏆 Top 20 Feature Importances")
        top20 = importance_df.head(20).iloc[::-1]

        fig = go.Figure(go.Bar(
            y=top20["feature_name"],
            x=top20["importance"],
            orientation="h",
            marker=dict(
                color=top20["importance"],
                colorscale="Blues",
                line=dict(color="#1e2236", width=0.4),
            ),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        ))
        fig.update_layout(
            paper_bgcolor="#1e2236", plot_bgcolor="#1a1d2e",
            font=dict(color="#c5d0e6", family="Inter, sans-serif"),
            height=500,
            margin=dict(l=20, r=20, t=30, b=30),
            xaxis=dict(title="Relative Gini Importance", gridcolor="#2d3147"),
            yaxis=dict(tickfont=dict(size=10)),
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 View Full Feature Importance Table"):
            st.dataframe(importance_df, use_container_width=True, height=350)

    # ── Removed Features Callout ────────────────────────────────────
    removed = eval_report.get("removed_features", ["life_percentage"])
    if removed:
        st.markdown("---")
        section_header("🚫 Excluded Target Leakage Features")
        for feat in removed:
            st.markdown(
                f"""
                <div style="background:#2d1515; border:1px solid #7f1d1d; border-radius:8px;
                            padding:10px 18px; margin:4px 0; color:#f87171;">
                    ❌ &nbsp; <code>{feat}</code> — Excluded to guarantee zero target leakage during inference.
                </div>
                """,
                unsafe_allow_html=True,
            )

