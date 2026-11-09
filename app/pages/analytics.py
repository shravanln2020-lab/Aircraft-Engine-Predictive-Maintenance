"""Fleet inference for the real three-engine FD001 subset and full test fleet."""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.utils import (
    df_to_csv_bytes,
    page_header,
    plot_engine_degradation,
    plot_engine_rul_bar,
    plot_health_donut,
    section_header,
)
from src import config
from src.data_loader import load_test_data
from src.inference import run_inference_pipeline

logger = logging.getLogger(__name__)
_STATUSES = ("Healthy", "Normal", "Warning", "Critical")
_DATASETS = ("3-Engine Sample Fleet", "Full FD001 Test Fleet")


@st.cache_data(show_spinner=False)
def _load_fd001_test() -> pd.DataFrame:
    return load_test_data()


def _run_fleet(mode: str, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list, list]:
    expected_ids = sorted(raw["unit_nr"].dropna().unique().tolist())
    if not expected_ids:
        raise ValueError("The selected dataset contains no engine IDs.")
    if mode == "3-Engine Sample Fleet":
        # Select actual FD001 telemetry for units 1-3; no values are synthesized.
        selected_ids = expected_ids[:3]
        raw = raw.loc[raw["unit_nr"].isin(selected_ids)].copy()
        expected_ids = selected_ids
    if raw.empty:
        raise ValueError("No telemetry rows are available for the selected fleet.")

    model_path = config.MODELS_DIR / "random_forest_v2.joblib"
    if not model_path.exists():
        model_path = config.MODELS_DIR / "random_forest_final.joblib"
    if not model_path.exists():
        raise FileNotFoundError("No supported model artifact was found in models/.")

    predictions = run_inference_pipeline(raw, model_path=model_path)
    required = {"unit_nr", "time_cycles", "predicted_rul", "health_status"}
    if not required.issubset(predictions.columns):
        raise ValueError("Inference output is missing required engine prediction columns.")
    if predictions.empty:
        raise ValueError("Inference returned no prediction rows.")

    predictions = predictions.copy()
    predictions["predicted_rul"] = pd.to_numeric(predictions["predicted_rul"], errors="coerce")
    predictions["time_cycles"] = pd.to_numeric(predictions["time_cycles"], errors="coerce")
    valid_rows = predictions[
        predictions["unit_nr"].notna()
        & predictions["time_cycles"].notna()
        & np.isfinite(predictions["predicted_rul"])
        & predictions["health_status"].isin(_STATUSES)
    ].copy()
    final = (
        valid_rows.sort_values("time_cycles", kind="stable")
        .groupby("unit_nr", as_index=False)
        .tail(1)
        .sort_values("unit_nr")
        .reset_index(drop=True)
    )
    successful = set(final["unit_nr"].tolist())
    failed = [engine_id for engine_id in expected_ids if engine_id not in successful]
    return valid_rows, final, expected_ids, failed


def render() -> None:
    page_header("Fleet Analysis")
    section_header("Fleet dataset")
    with st.container(border=True):
        mode = st.selectbox(
            "Dataset",
            _DATASETS,
            key="fleet_dataset_mode",
            width="stretch",
        )
        current_result = st.session_state.get("fleet_analysis_result")
        if isinstance(current_result, dict) and current_result.get("mode") != mode:
            st.session_state.pop("fleet_analysis_result", None)
            st.session_state.pop("fleet_engine_ids", None)
        if mode == "3-Engine Sample Fleet":
            st.caption("Real FD001 test telemetry for units 1-3.")
        else:
            st.caption("All engine units in data/raw/test_FD001.txt.")
        run_clicked = st.button(
            "Run Fleet Analysis",
            type="primary",
            key="run_fleet_analysis",
            width="stretch",
        )

    try:
        with st.spinner("Loading FD001 test telemetry..."):
            raw = _load_fd001_test()
    except Exception as exc:
        st.error(f"FD001 test data could not be loaded: {exc}")
        logger.exception("Fleet source data load failed")
        return
    if "unit_nr" not in raw.columns or raw.empty:
        st.error("FD001 test telemetry is empty or does not contain engine IDs.")
        return

    actual_ids = sorted(raw["unit_nr"].dropna().unique().tolist())
    expected_for_mode = actual_ids[:3] if mode == "3-Engine Sample Fleet" else actual_ids
    if run_clicked:
        try:
            with st.spinner(f"Running inference for {len(expected_for_mode)} engine(s)..."):
                predictions, final, expected, failed = _run_fleet(mode, raw)
            st.session_state["fleet_analysis_result"] = {
                "mode": mode,
                "predictions": predictions,
                "final": final,
                "expected": expected,
                "failed": failed,
            }
        except Exception as exc:
            st.session_state.pop("fleet_analysis_result", None)
            st.error(f"Fleet inference failed; no fleet results were recorded: {exc}")
            logger.exception("Fleet inference failed")
            return

    result = st.session_state.get("fleet_analysis_result")
    if not isinstance(result, dict) or result.get("mode") != mode:
        st.info("Run Fleet Analysis to calculate results for this dataset.")
        return

    final = result["final"]
    failed = result["failed"]
    if failed:
        st.warning(f"{len(failed)} of {len(result['expected'])} engines did not produce a valid final prediction: {failed}")
    if final.empty:
        st.error("No engines produced a valid final prediction.")
        return

    counts = final["health_status"].value_counts()
    successful_count = len(final)
    section_header("Fleet summary")
    st.caption(f"{successful_count} successfully analyzed of {len(result['expected'])} expected engines.")
    metrics = st.columns(5)
    metrics[0].metric("Engines analyzed", f"{successful_count:,}")
    for column, status in zip(metrics[1:], _STATUSES):
        column.metric(status, f"{int(counts.get(status, 0)):,}")
    if int(sum(counts.get(status, 0) for status in _STATUSES)) != successful_count:
        st.error("Status totals do not match the number of successful engine predictions.")
        return

    section_header("Fleet health status")
    st.plotly_chart(plot_health_donut(final), width="stretch")

    section_header("Final RUL by engine")
    st.plotly_chart(plot_engine_rul_bar(final), width="stretch")

    section_header("Engine RUL trends")
    engine_ids = sorted(final["unit_nr"].tolist())
    chosen = st.multiselect(
        "Engine IDs",
        engine_ids,
        default=engine_ids[:min(8, len(engine_ids))],
        key="fleet_engine_ids",
    )
    if chosen:
        st.plotly_chart(plot_engine_degradation(result["predictions"], chosen), width="stretch")

    section_header("Analyzed engines")
    st.dataframe(final, width="stretch", hide_index=True)
    st.download_button(
        "Download final fleet results (CSV)",
        data=df_to_csv_bytes(final),
        file_name="fleet_final_engine_results.csv",
        mime="text/csv",
    )