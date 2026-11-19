import streamlit as st

import common

st.title("🔎 Applicant Results")
if not common.has_results():
    common.no_results_hint()
    st.stop()

r = st.session_state.results
f = common.filter_bar(r, "res")
st.caption(f"{len(f):,} of {len(r):,} applicants match.")
common.results_table(f, key="res_tbl", height=320)
if len(f):
    ids = f.sort_values("default_probability", ascending=False, na_position="last")["applicant_id"].tolist()
    sel = st.selectbox("Open applicant profile", ids)
    row = r[r["applicant_id"] == sel].iloc[0]
    st.divider()
    common.render_profile(row, st.session_state.details.get(sel))
common.disclaimer()
