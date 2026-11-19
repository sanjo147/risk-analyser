import pandas as pd
import plotly.express as px
import streamlit as st

import common
from loanrisk.reporting import BAND_COLOURS

st.title("🏠 Home Dashboard")
st.caption("Portfolio view of the currently loaded assessment.")

if not common.has_results():
    common.no_results_hint()
    st.stop()

r = st.session_state.results
a = r[r["default_probability"].notna()]
rec = r["recommendation"].value_counts()
c = st.columns(6)
common.kpi(c[0], "Applicants assessed", f"{len(a):,} / {len(r):,}")
common.kpi(c[1], "Recommended approvals", f"{rec.get('APPROVE', 0):,}", "#1b7f3b")
common.kpi(c[2], "Reject recommendations", f"{rec.get('REJECT RECOMMENDATION', 0):,}", "#c0262d")
common.kpi(c[3], "Manual review", f"{rec.get('MANUAL REVIEW', 0):,}", "#b07d00")
common.kpi(c[4], "Avg default probability", f"{a['default_probability'].mean():.1%}" if len(a) else "-")
common.kpi(c[5], "Cannot assess", f"{rec.get('CANNOT ASSESS', 0):,}", "#6b7280")
st.write("")

c1, c2 = st.columns(2)
dist = a["risk_category"].value_counts().reindex(common.BAND_ORDER).fillna(0).reset_index()
dist.columns = ["Risk category", "Applicants"]
c1.plotly_chart(px.bar(dist, x="Risk category", y="Applicants", color="Risk category", text="Applicants",
                       color_discrete_map=BAND_COLOURS, title="Risk distribution").update_layout(showlegend=False),
                width="stretch")
recs = r["recommendation"].value_counts().reset_index()
recs.columns = ["Recommendation", "Applicants"]
c2.plotly_chart(px.pie(recs, names="Recommendation", values="Applicants", hole=0.55, title="Recommendations",
                       color="Recommendation", color_discrete_map=common.DECISION_COLOURS), width="stretch")

c1, c2 = st.columns(2)
c1.plotly_chart(px.histogram(a, x="loan_amount", nbins=30, color="risk_category", title="Loan amount distribution",
                             color_discrete_map=BAND_COLOURS, category_orders={"risk_category": common.BAND_ORDER},
                             labels={"loan_amount": "Requested loan", "risk_category": "Risk"}),
                width="stretch")
c2.plotly_chart(px.histogram(a, x="credit_score", nbins=30, color="risk_category", title="Credit score distribution",
                             color_discrete_map=BAND_COLOURS, category_orders={"risk_category": common.BAND_ORDER},
                             labels={"credit_score": "Credit score", "risk_category": "Risk"}),
                width="stretch")

st.markdown("#### Highest-risk applicants")
common.results_table(a.nlargest(10, "default_probability"), key="home_top", height=300)
st.caption("More charts on **Risk Analytics**; full list with filters on **Bulk Risk Assessment**.")
common.disclaimer()
