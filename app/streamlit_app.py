"""AI-Powered Loan Risk Assessment & Approval System - Streamlit entry point.

Run:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
st.set_page_config(page_title="Loan Risk Assessment", page_icon="🏦", layout="wide")

import common  # noqa: E402

common.setup()

V = Path(__file__).resolve().parent / "views"
pages = {
    "Overview": [
        st.Page(V / "home.py", title="Home Dashboard", icon="🏠", default=True),
    ],
    "Assess": [
        st.Page(V / "upload.py", title="Upload Customer Dataset", icon="📤"),
        st.Page(V / "individual.py", title="Individual Assessment", icon="👤"),
        st.Page(V / "bulk.py", title="Bulk Risk Assessment", icon="📋"),
        st.Page(V / "results.py", title="Applicant Results", icon="🔎"),
    ],
    "Analyse": [
        st.Page(V / "analytics.py", title="Risk Analytics", icon="📊"),
        st.Page(V / "performance.py", title="Model Performance", icon="🎯"),
        st.Page(V / "explainability.py", title="Model Explainability", icon="🧠"),
    ],
    "Manage": [
        st.Page(V / "reports.py", title="Download Reports", icon="⬇️"),
        st.Page(V / "configuration.py", title="Model Configuration", icon="⚙️"),
    ],
}
with st.sidebar:
    st.markdown("### 🏦 Loan Risk AI")
    st.caption("Decision-support prototype · educational use")
nav = st.navigation(pages)
if common.model_ready():
    meta = common.bundle()["meta"]
    st.sidebar.caption(f"Model: **{meta['model_name']}** (calibrated)  \nTrained {meta['trained_at'][:10]}")
    if common.has_results():
        st.sidebar.caption(f"Loaded: **{st.session_state.source}** ({len(st.session_state.results):,} applicants)")
    nav.run()
