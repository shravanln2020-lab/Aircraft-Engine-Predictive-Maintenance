"""
Shared utility helpers for the AEPM Streamlit dashboard.

Provides CSS injection, Plotly chart builders, and data formatting helpers
that are reused across all dashboard pages.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# CSS & Layout Helpers
# ──────────────────────────────────────────────

def inject_custom_css() -> None:
    """Apply the AEPM enterprise aerospace visual system."""
    st.markdown(
        """
        <style>
        :root { color-scheme: dark; }
        html, body, [class*="css"] { font-family: Inter, "Segoe UI", Arial, sans-serif; }
        .stApp { background: #0b1220; color: #edf2f8; }
        [data-testid="stHeader"] { background: rgba(11,18,32,.96); }
        .main .block-container { max-width: 1500px; padding: 1.7rem 2.4rem 2.5rem; }
        [data-testid="stSidebar"] { background: #101a2a; border-right: 1px solid #24344a; }
        [data-testid="stSidebar"] > div:first-child { padding-top: 1.1rem; }
        .brand-block { padding: 10px 8px 22px; }
        .brand-mark { display:flex; align-items:center; justify-content:center; width:40px; height:40px; border-radius:10px; background:#17345a; color:#a8cfff; font-size:14px; font-weight:750; letter-spacing:.03em; }
        .brand-name { color:#f2f5fa; font-weight:700; font-size:1.15rem; margin-top:11px; }
        .brand-caption { color:#91a3bb; font-size:.65rem; letter-spacing:.11em; margin-top:3px; }
        .sidebar-foot { border-top:1px solid #27384e; margin-top:22px; padding:16px 8px; display:grid; gap:5px; }
        .sidebar-foot span { color:#91a3bb; font-size:.63rem; letter-spacing:.09em; margin-top:7px; }
        .sidebar-foot b { color:#dce5ef; font-size:.76rem; font-weight:550; }
        [data-testid="stSidebarNav"] a { border-radius:8px; margin:2px 4px; color:#bdc9d8; }
        [data-testid="stSidebarNav"] a[aria-current="page"] { background:#18385d; color:#f3f7fc; }
        h1,h2,h3 { color:#f1f4f8 !important; letter-spacing:-.02em; }
        h1 { font-size:1.8rem !important; line-height:1.22 !important; font-weight:680 !important; margin:0 !important; }
        h2 { font-size:1.32rem !important; font-weight:650 !important; }
        p, li, label { color:#c0cad7; }
        .page-heading { margin:2px 0 22px; }
        .page-subtitle { color:#a3b0c0; font-size:.9rem; margin-top:6px; line-height:1.45; }
        .section-header { margin:22px 0 11px; padding:0; background:none; border:0; box-shadow:none; }
        .section-header h3 { font-size:1.02rem; font-weight:650; margin:0; }
        [data-testid="stMetric"] { background:#111d2d; border:1px solid #2b3d54; border-radius:10px; padding:14px 16px; min-height:92px; }
        [data-testid="stMetricLabel"] { color:#a6b3c2 !important; font-size:.77rem !important; font-weight:500 !important; }
        [data-testid="stMetricValue"] { color:#f2f5f9 !important; font-family:Inter,"Segoe UI",Arial,sans-serif !important; font-size:1.48rem !important; font-weight:680 !important; }
        [data-testid="stFileUploader"] { background:#101b2b; border:1px solid #354a63; border-radius:10px; padding:8px; }
        [data-testid="stSelectbox"] [data-baseweb="select"] > div { min-height:46px; background:#111d2d; border-color:#354a63; border-radius:8px; }
        [data-testid="stRadio"] label { color:#d1d9e3; }
        [data-testid="stDataFrame"] { border:1px solid #2b3d54; border-radius:9px; overflow:hidden; }
        .js-plotly-plot { border:1px solid #2b3d54; border-radius:10px; overflow:hidden; }
        .stButton > button, .stDownloadButton > button { border-radius:8px; min-height:2.7rem; padding:.55rem 1rem; font-family:Inter,"Segoe UI",Arial,sans-serif; font-weight:620; transition:background .15s ease,border-color .15s ease; }
        .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] { background:#245c98; border:1px solid #3679bc; color:#fff; }
        .stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover { background:#2e70b3; border-color:#65a1de; color:#fff; }
        .stButton > button:focus-visible, .stDownloadButton > button:focus-visible, input:focus-visible { outline:2px solid #83baff; outline-offset:2px; }
        .st-key-run_single_engine button, .st-key-run_fleet_analysis button { width:100%; min-height:3.1rem; font-size:1rem; }
        .status-badge { display:inline-block; padding:5px 11px; border-radius:18px; font-size:.76rem; font-weight:650; }
        .badge-healthy { background:#12372f; color:#76dfb8; border:1px solid #216b57; }
        .badge-normal { background:#122f46; color:#7bc9ee; border:1px solid #24638a; }
        .badge-warning { background:#3a2d18; color:#f0cc7c; border:1px solid #735c2a; }
        .badge-critical { background:#3a2028; color:#ffabb3; border:1px solid #79434e; }
        [data-testid="stAlert"] { border-radius:9px; }
        hr { border-color:#27384e !important; }
        @media (max-width: 900px) {
          .main .block-container { padding:1.25rem 1rem 2rem; }
          h1 { font-size:1.55rem !important; }
          [data-testid="stMetric"] { padding:11px 12px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
def clear_workflow_results(previous_workflow: str | None, current_workflow: str) -> None:
    """Clear result-only state when navigation leaves its owning workflow."""
    if previous_workflow == current_workflow:
        return
    if previous_workflow == "predict":
        st.session_state.pop("single_engine_result", None)
        st.session_state.pop("single_engine_result_key", None)
    elif previous_workflow == "analytics":
        st.session_state.pop("fleet_analysis_result", None)
        st.session_state.pop("fleet_engine_ids", None)

def page_header(title: str, subtitle: str = "", icon: str = "") -> None:
    """Render a restrained page title and optional concise supporting text."""
    description = f'<div class="page-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="page-heading"><h1>{title}</h1>{description}</div>',
        unsafe_allow_html=True,
    )
def section_header(title: str) -> None:
    """Render a styled section header with an accent left border.

    Args:
        title: Section title text.
    """
    st.markdown(
        f'<div class="section-header"><h3>{title}</h3></div>',
        unsafe_allow_html=True,
    )


def status_badge(status: str) -> str:
    """Return an HTML status badge for a given health status string.

    Args:
        status: Engine health status ('Healthy', 'Normal', 'Warning', 'Critical').

    Returns:
        str: HTML string representing the styled badge.
    """
    cls_map = {
        "Healthy": "badge-healthy",
        "Normal": "badge-normal",
        "Warning": "badge-warning",
        "Critical": "badge-critical",
    }
    icon_map = {"Healthy": "🟢", "Normal": "🔵", "Warning": "🟡", "Critical": "🔴"}
    cls = cls_map.get(status, "badge-normal")
    icon = icon_map.get(status, "⚪")
    return f'<span class="status-badge {cls}">{icon} {status}</span>'


# ──────────────────────────────────────────────
# Plotly Chart Builders
# ──────────────────────────────────────────────

PLOTLY_THEME = {
    "paper_bgcolor": "#111d2d",
    "plot_bgcolor": "#0f1a29",
    "font_color": "#b5c3d6",
    "gridcolor": "#263850",
}

STATUS_COLORS = {
    "Healthy": "#34d399",
    "Normal": "#38bdf8",
    "Warning": "#fbbf24",
    "Critical": "#f87171",
}


def _base_layout(fig: go.Figure, title: str, height: int = 400) -> go.Figure:
    """Apply common dark-mode layout settings to a Plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(color="#edf4ff", size=15)),
        paper_bgcolor=PLOTLY_THEME["paper_bgcolor"],
        plot_bgcolor=PLOTLY_THEME["plot_bgcolor"],
        font=dict(color=PLOTLY_THEME["font_color"], family="Inter, sans-serif"),
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    fig.update_xaxes(gridcolor=PLOTLY_THEME["gridcolor"], zeroline=False)
    fig.update_yaxes(gridcolor=PLOTLY_THEME["gridcolor"], zeroline=False)
    return fig


def plot_health_donut(df: pd.DataFrame, status_col: str = "health_status") -> go.Figure:
    """Build an interactive donut chart of engine health status distribution.

    Args:
        df: Input DataFrame containing health status column.
        status_col: Column name for health status. Defaults to 'health_status'.

    Returns:
        go.Figure: Interactive Plotly donut chart.
    """
    if df is None or df.empty or status_col not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No health data available", showarrow=False, font=dict(color="#8b9dc3", size=14))
        return _base_layout(fig, "Fleet Health Status Distribution", height=380)

    counts = df[status_col].value_counts().reset_index()
    counts.columns = ["status", "count"]
    ordered_cats = ["Critical", "Warning", "Normal", "Healthy"]
    counts["status"] = pd.Categorical(counts["status"], categories=ordered_cats, ordered=True)
    counts = counts.dropna().sort_values("status")

    colors = [STATUS_COLORS.get(str(s), "#4f8ef7") for s in counts["status"]]

    fig = go.Figure(go.Pie(
        labels=counts["status"],
        values=counts["count"],
        hole=0.58,
        marker=dict(colors=colors, line=dict(color="#1e2236", width=2)),
        textfont=dict(size=13, color="#e2e8f0"),
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>",
    ))

    total = len(df)
    fig.add_annotation(
        text=f"<b>{total:,}</b><br><span style='font-size:10px'>Engines</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="#e2e8f0"),
    )
    return _base_layout(fig, "Fleet Health Status Distribution", height=380)


def plot_rul_histogram(df: pd.DataFrame, rul_col: str = "predicted_rul") -> go.Figure:
    """Build an interactive RUL distribution histogram with health threshold markers.

    Args:
        df: Input DataFrame containing RUL predictions.
        rul_col: RUL column name. Defaults to 'predicted_rul'.

    Returns:
        go.Figure: Interactive Plotly histogram.
    """
    if df is None or df.empty or rul_col not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No RUL data available", showarrow=False, font=dict(color="#8b9dc3", size=14))
        return _base_layout(fig, "Predicted RUL Distribution", height=360)

    fig = go.Figure(go.Histogram(
        x=df[rul_col],
        nbinsx=40,
        marker=dict(color="#4f8ef7", opacity=0.82, line=dict(color="#1e2236", width=0.5)),
        hovertemplate="RUL: %{x:.0f} cycles<br>Count: %{y}<extra></extra>",
    ))

    # Health status threshold lines
    thresholds = [
        (100, "Healthy ≥100", "#34d399"),
        (50, "Normal ≥50", "#38bdf8"),
        (20, "Warning ≥20", "#fbbf24"),
    ]
    for rul_val, label, color in thresholds:
        fig.add_vline(
            x=rul_val,
            line_dash="dash",
            line_color=color,
            line_width=1.5,
            annotation_text=label,
            annotation_position="top",
            annotation=dict(font_color=color, font_size=10),
        )

    fig.update_xaxes(title_text="Predicted RUL (Cycles)")
    fig.update_yaxes(title_text="Frequency")
    return _base_layout(fig, "Predicted RUL Distribution", height=360)


def plot_engine_rul_bar(df: pd.DataFrame) -> go.Figure:
    """Build a horizontal bar chart of the final predicted RUL per engine unit.

    Args:
        df: Input DataFrame containing unit_nr, predicted_rul, health_status.

    Returns:
        go.Figure: Horizontal bar chart showing RUL for each engine unit.
    """
    if df is None or df.empty or "unit_nr" not in df.columns or "predicted_rul" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No engine RUL data available", showarrow=False, font=dict(color="#8b9dc3", size=14))
        return _base_layout(fig, "Final RUL per Engine Unit", height=360)

    sort_col = "time_cycles" if "time_cycles" in df.columns else df.columns[0]
    last = (
        df.sort_values(sort_col)
        .groupby("unit_nr")
        .last()
        .reset_index()
    )
    last = last.sort_values("predicted_rul")

    colors = [
        STATUS_COLORS.get(str(s), "#4f8ef7")
        for s in last.get("health_status", ["Normal"] * len(last))
    ]

    fig = go.Figure(go.Bar(
        y=last["unit_nr"].astype(str),
        x=last["predicted_rul"],
        orientation="h",
        marker=dict(color=colors, line=dict(color="#1e2236", width=0.4)),
        hovertemplate="Engine %{y}<br>RUL: %{x:.1f} cycles<extra></extra>",
    ))

    fig.update_xaxes(title_text="Predicted RUL (Cycles)")
    fig.update_yaxes(title_text="Engine Unit ID", tickfont=dict(size=9))
    chart_height = max(340, min(760, len(last) * 7))
    return _base_layout(fig, "Final RUL per Engine Unit", height=chart_height)


def plot_engine_degradation(df: pd.DataFrame, engine_ids: List[int]) -> go.Figure:
    """Build multi-engine RUL degradation trajectory line chart over flight cycles.

    Args:
        df: Input DataFrame containing unit_nr, time_cycles, predicted_rul.
        engine_ids: List of engine unit IDs to plot.

    Returns:
        go.Figure: Interactive multi-line degradation trajectory chart.
    """
    if df is None or df.empty or not engine_ids:
        fig = go.Figure()
        fig.add_annotation(text="Select engine IDs to view degradation trajectory", showarrow=False, font=dict(color="#8b9dc3", size=14))
        return _base_layout(fig, "Engine RUL Degradation Trajectory", height=420)

    fig = go.Figure()
    palette = px.colors.qualitative.Plotly
    threshold_lines = [
        (100, "#34d399", "Healthy ≥100"),
        (50, "#38bdf8", "Normal ≥50"),
        (20, "#fbbf24", "Warning ≥20"),
    ]

    for i, eid in enumerate(engine_ids):
        eng_df = df[df["unit_nr"] == eid].sort_values("time_cycles")
        if eng_df.empty:
            continue
        fig.add_trace(go.Scatter(
            x=eng_df["time_cycles"],
            y=eng_df["predicted_rul"],
            name=f"Engine {eid}",
            mode="lines",
            line=dict(color=palette[i % len(palette)], width=2),
            hovertemplate=f"Engine {eid}<br>Cycle: %{{x}}<br>RUL: %{{y:.1f}} cycles<extra></extra>",
        ))

    for val, color, label in threshold_lines:
        fig.add_hline(
            y=val,
            line_dash="dot",
            line_color=color,
            line_width=1.4,
            annotation_text=label,
            annotation_position="right",
            annotation=dict(font_color=color, font_size=10),
        )

    fig.update_xaxes(title_text="Flight Cycle Number")
    fig.update_yaxes(title_text="Predicted RUL (Cycles)")
    return _base_layout(fig, "Engine RUL Degradation Trajectory", height=420)


# ──────────────────────────────────────────────
# Data & Model Info Helpers
# ──────────────────────────────────────────────

def load_json_safe(path: Path) -> Optional[dict]:
    """Load a JSON file safely, returning None if unavailable or invalid.

    Args:
        path: Path to the JSON file.

    Returns:
        Optional[dict]: Parsed dictionary content or None.
    """
    try:
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to load JSON file at '%s': %s", path, exc)
        return None


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to UTF-8 encoded CSV bytes for download.

    Args:
        df: Input DataFrame.

    Returns:
        bytes: UTF-8 encoded CSV string bytes.
    """
    if df is None or df.empty:
        return b""
    return df.to_csv(index=False).encode("utf-8")

