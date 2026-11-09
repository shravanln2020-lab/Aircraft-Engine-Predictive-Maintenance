"""Dashboard overview of project configuration and valid prediction results."""
import pandas as pd
import streamlit as st

from app.utils import page_header, plot_health_donut, section_header
from src import config
from src.data_loader import load_test_data


@st.cache_data(show_spinner=False)
def _load_dataset_summary() -> tuple[int, int]:
    data = load_test_data()
    return len(data), int(data["unit_nr"].nunique())


def _available_results() -> tuple[pd.DataFrame | None, str]:
    fleet = st.session_state.get("fleet_analysis_result")
    if isinstance(fleet, dict) and isinstance(fleet.get("final"), pd.DataFrame) and not fleet["final"].empty:
        return fleet["final"], f"Current Fleet Analysis ({fleet.get('mode', 'selected dataset')})"
    single = st.session_state.get("single_engine_result")
    if isinstance(single, pd.DataFrame) and not single.empty:
        return single, "Current Single Engine prediction"
    saved = config.RESULTS_DIR / "sample_predictions.csv"
    if saved.exists():
        try:
            predictions = pd.read_csv(saved)
            required = {"unit_nr", "time_cycles", "predicted_rul", "health_status"}
            if required.issubset(predictions.columns):
                latest = predictions.sort_values("time_cycles").groupby("unit_nr", as_index=False).last()
                return latest, "Saved project prediction output"
        except Exception:
            return None, ""
    return None, ""


def render() -> None:
    page_header("Dashboard")
    try:
        records, engines = _load_dataset_summary()
        dataset_label = f"FD001 / {engines:,} engines"
        record_label = f"{records:,} records"
    except Exception:
        dataset_label = "Unavailable"
        record_label = "Unavailable"

    model = next((p for p in (config.MODELS_DIR / "random_forest_v2.joblib",
                              config.MODELS_DIR / "random_forest_final.joblib") if p.exists()), None)
    predictions, source = _available_results()

    section_header("Project status")
    cards = st.columns(4)
    cards[0].metric("Test dataset", dataset_label)
    cards[1].metric("Telemetry records", record_label)
    cards[2].metric("Model artifact", model.name if model else "Not found")
    cards[3].metric("Prediction results", "Available" if predictions is not None else "Not generated")

    section_header("Latest available engine health")
    if predictions is None or not {"unit_nr", "health_status"}.issubset(predictions.columns):
        st.info("No prediction results are available. Run Predict RUL or Run Fleet Analysis to calculate engine health.")
        return

    left, right = st.columns([1, 1.5], gap="large")
    counts = predictions["health_status"].value_counts()
    with left:
        st.metric("Engines represented", f"{predictions['unit_nr'].nunique():,}")
        for status in ("Critical", "Warning", "Normal", "Healthy"):
            if status in counts.index:
                st.metric(status, f"{int(counts[status]):,}")
    with right:
        st.plotly_chart(plot_health_donut(predictions), width="stretch")
    st.caption(f"Source: {source}. Health counts use one latest result per engine.")