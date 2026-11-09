"""AEPM application entry point with page-scoped workflow state."""
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="AEPM | Engine Health Monitor",
    page_icon="AE",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.pages.analytics import render as render_fleet
from app.pages.home import render as render_dashboard
from app.pages.predict import render as render_single_engine
from app.utils import clear_workflow_results, inject_custom_css

inject_custom_css()

# st.navigation owns multipage routing and suppresses automatic pages/ discovery.
page = st.navigation(
    [
        st.Page(render_dashboard, title="Dashboard", icon=":material/dashboard:", url_path="home", default=True),
        st.Page(render_single_engine, title="Single Engine", icon=":material/precision_manufacturing:", url_path="predict"),
        st.Page(render_fleet, title="Fleet Analysis", icon=":material/monitoring:", url_path="analytics"),
    ],
    position="sidebar",
)

current_workflow = page.url_path
previous_workflow = st.session_state.get("_aepm_previous_workflow")
clear_workflow_results(previous_workflow, current_workflow)
if previous_workflow != current_workflow:
    st.session_state["_aepm_previous_workflow"] = current_workflow

with st.sidebar:
    st.markdown(
        """<div class="brand-block"><div class="brand-mark">AE</div>
        <div class="brand-name">AEPM</div>
        <div class="brand-caption">ENGINE HEALTH MONITOR</div></div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """<div class="sidebar-foot"><span>DATASET</span><b>NASA C-MAPSS / FD001</b>
        <span>MODEL FAMILY</span><b>Random Forest Regressor</b></div>""",
        unsafe_allow_html=True,
    )

page.run()