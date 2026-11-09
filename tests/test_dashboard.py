"""
Unit test suite for AEPM Streamlit Dashboard utilities and page modules.
"""

import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.pages import about, analytics, home, model_info, predict
from app.utils import (
    df_to_csv_bytes,
    load_json_safe,
    plot_engine_degradation,
    plot_engine_rul_bar,
    plot_health_donut,
    plot_rul_histogram,
    status_badge,
)
from src import config


def test_status_badge():
    """Test HTML status badge generation for all health categories."""
    b_healthy = status_badge("Healthy")
    assert "badge-healthy" in b_healthy
    assert "Healthy" in b_healthy

    b_normal = status_badge("Normal")
    assert "badge-normal" in b_normal

    b_warning = status_badge("Warning")
    assert "badge-warning" in b_warning

    b_critical = status_badge("Critical")
    assert "badge-critical" in b_critical


def test_df_to_csv_bytes():
    """Test DataFrame to CSV byte string conversion."""
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    csv_bytes = df_to_csv_bytes(df)
    assert isinstance(csv_bytes, bytes)
    assert b"a,b" in csv_bytes
    assert b"1,x" in csv_bytes

    # Empty df case
    assert df_to_csv_bytes(pd.DataFrame()) == b""
    assert df_to_csv_bytes(None) == b""


def test_load_json_safe():
    """Test safe loading of JSON file."""
    res_path = config.RESULTS_DIR / "best_parameters.json"
    data = load_json_safe(res_path)
    assert data is None or isinstance(data, dict)

    # Non-existent file
    assert load_json_safe(Path("non_existent_file.json")) is None


def test_plotly_chart_builders():
    """Test Plotly chart builder functions with sample prediction data."""
    sample_df = pd.DataFrame({
        "unit_nr": [1, 1, 2, 2],
        "time_cycles": [1, 2, 1, 2],
        "predicted_rul": [150.0, 140.0, 30.0, 15.0],
        "health_status": ["Healthy", "Healthy", "Warning", "Critical"],
    })

    fig_donut = plot_health_donut(sample_df)
    assert isinstance(fig_donut, go.Figure)

    fig_hist = plot_rul_histogram(sample_df)
    assert isinstance(fig_hist, go.Figure)

    fig_bar = plot_engine_rul_bar(sample_df)
    assert isinstance(fig_bar, go.Figure)

    fig_line = plot_engine_degradation(sample_df, engine_ids=[1, 2])
    assert isinstance(fig_line, go.Figure)


def test_page_modules_import():
    """Test that all dashboard page modules are imported cleanly with callable render()."""
    for page_mod in [home, predict, analytics, model_info, about]:
        assert hasattr(page_mod, "render")
        assert callable(page_mod.render)


if __name__ == "__main__":
    test_status_badge()
    test_df_to_csv_bytes()
    test_load_json_safe()
    test_plotly_chart_builders()
    test_page_modules_import()
    print("[SUCCESS] All dashboard unit tests passed successfully!")


