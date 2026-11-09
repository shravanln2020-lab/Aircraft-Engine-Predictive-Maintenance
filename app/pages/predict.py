"""Single-engine inference using the existing FD001 loader and model pipeline."""
import hashlib
import logging
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.utils import df_to_csv_bytes, page_header, plot_engine_degradation, section_header
from src import config
from src.data_loader import load_test_data
from src.inference import run_inference_pipeline

logger = logging.getLogger(__name__)

_PRIORITY = {
    "Healthy": "Routine",
    "Normal": "Routine",
    "Warning": "Elevated",
    "Critical": "Immediate review",
}
_RECOMMENDATION = {
    "Healthy": "Continue routine monitoring.",
    "Normal": "Continue scheduled monitoring and review the next available estimate.",
    "Warning": "Plan a maintenance inspection and review the engine trend.",
    "Critical": "Prioritize an immediate maintenance review.",
}


@st.cache_data(show_spinner=False)
def _load_fd001_test() -> pd.DataFrame:
    """Load FD001 test data through the project's established loader."""
    return load_test_data()


def _read_upload(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith(".csv"):
        raw = pd.read_csv(uploaded_file)
        if raw.empty or raw.columns[0] not in config.COLUMN_NAMES:
            uploaded_file.seek(0)
            raw = pd.read_csv(uploaded_file, header=None, names=config.COLUMN_NAMES)
        return raw
    return pd.read_csv(
        uploaded_file, sep=r"\s+", header=None,
        names=config.COLUMN_NAMES, engine="python",
    )


def _fingerprint(df: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(values).hexdigest()


def render() -> None:
    page_header("Single Engine")
    section_header("Telemetry source")
    source = st.radio(
        "Choose telemetry source",
        ["FD001 test data", "Upload telemetry"],
        horizontal=True,
        key="single_engine_source",
    )
    previous_source = st.session_state.get("_single_engine_result_source")
    if previous_source is not None and previous_source != source:
        st.session_state.pop("single_engine_result", None)
        st.session_state.pop("single_engine_result_key", None)
    st.session_state["_single_engine_result_source"] = source

    raw_df = None
    if source == "FD001 test data":
        try:
            with st.spinner("Loading FD001 test telemetry..."):
                raw_df = _load_fd001_test()
        except Exception as exc:
            st.error(f"FD001 test data could not be loaded: {exc}")
            logger.exception("FD001 test data load failed")
            return
        st.caption("Source: data/raw/test_FD001.txt")
    else:
        uploaded = st.file_uploader(
            "Telemetry file (FD001 text or structured CSV)",
            type=["txt", "csv"],
            key="single_engine_upload",
        )
        if uploaded is None:
            st.info("Upload telemetry to populate the engine selector.")
            return
        try:
            raw_df = _read_upload(uploaded)
        except Exception as exc:
            st.error(f"Telemetry could not be read: {exc}")
            logger.exception("Telemetry upload parse failed")
            return

    if "unit_nr" not in raw_df.columns:
        st.error("The selected telemetry has no unit_nr column, so an engine cannot be selected.")
        return
    engine_ids = sorted(raw_df["unit_nr"].dropna().unique().tolist())
    if not engine_ids:
        st.warning("No engine IDs are available in this dataset.")
        return

    input_fingerprint = _fingerprint(raw_df)
    with st.expander("Input summary", expanded=False):
        left, right = st.columns(2)
        left.metric("Telemetry rows", f"{len(raw_df):,}")
        right.metric("Available engine IDs", f"{len(engine_ids):,}")
        st.dataframe(raw_df.head(8), width="stretch", hide_index=True)

    model_path = config.MODELS_DIR / "random_forest_v2.joblib"
    if not model_path.exists():
        model_path = config.MODELS_DIR / "random_forest_final.joblib"
    if not model_path.exists():
        st.error("No supported model artifact was found in models/.")
        return

    with st.container(border=True):
        selected_id = st.selectbox(
            "Engine ID",
            engine_ids,
            format_func=lambda value: f"Engine {value}",
            key="single_engine_id",
            width="stretch",
        )
        result_key = (source, input_fingerprint, selected_id)
        previous_result_key = st.session_state.get("single_engine_result_key")
        if previous_result_key is not None and previous_result_key != result_key:
            st.session_state.pop("single_engine_result", None)
            st.session_state.pop("single_engine_result_key", None)
        predict_clicked = st.button(
            "Predict RUL",
            type="primary",
            key="run_single_engine",
            width="stretch",
        )

    if predict_clicked:
        engine_input = raw_df.loc[raw_df["unit_nr"] == selected_id].copy()
        if engine_input.empty:
            st.error("No telemetry rows were found for the selected engine.")
            return
        try:
            with st.spinner(f"Running inference for Engine {selected_id}..."):
                result = run_inference_pipeline(engine_input, model_path=model_path)
            required = {"unit_nr", "time_cycles", "predicted_rul", "health_status"}
            if result.empty or not required.issubset(result.columns):
                st.error("The inference pipeline returned no usable RUL result for this engine.")
                return
            result = result.loc[result["unit_nr"] == selected_id].copy()
            if result.empty:
                st.error("The inference output did not include the selected engine.")
                return
            st.session_state["single_engine_result"] = result
            st.session_state["single_engine_result_key"] = result_key
        except Exception as exc:
            st.error(f"RUL inference failed for Engine {selected_id}: {exc}")
            logger.exception("Single-engine inference failed")
            return

    result = st.session_state.get("single_engine_result")
    if st.session_state.get("single_engine_result_key") != result_key or not isinstance(result, pd.DataFrame):
        st.info("Select an engine and run Predict RUL to view its assessment.")
        return
    if result.empty:
        st.warning("No prediction rows were returned for this engine.")
        return

    result = result.sort_values("time_cycles")
    latest = result.iloc[-1]
    status = str(latest["health_status"])
    section_header("Engine assessment")
    metrics = st.columns(4)
    metrics[0].metric("Engine ID", str(selected_id))
    metrics[1].metric("Predicted RUL", f"{float(latest['predicted_rul']):.2f} cycles")
    metrics[2].metric("Health status", status)
    metrics[3].metric("Maintenance priority", _PRIORITY.get(status, "Review"))
    cycle, _ = st.columns([1, 3])
    cycle.metric("Latest cycle", f"{int(latest['time_cycles']):,}")
    with st.container(border=True):
        st.markdown(f"**Recommendation**  \n{_RECOMMENDATION.get(status, 'Review the estimate with the maintenance team.')}")

    section_header("RUL trend and prediction history")
    st.plotly_chart(plot_engine_degradation(result, [selected_id]), width="stretch")
    st.dataframe(result.sort_values("time_cycles", ascending=False), width="stretch", hide_index=True)
    st.download_button(
        "Download selected engine results (CSV)",
        data=df_to_csv_bytes(result),
        file_name=f"engine_{selected_id}_rul_predictions.csv",
        mime="text/csv",
    )