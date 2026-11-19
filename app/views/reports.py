import json

import pandas as pd
import streamlit as st

import common
from loanrisk import storage
from loanrisk.config import MODEL_DIR, REPORT_DIR
from loanrisk.reporting import applicant_html, bulk_excel, output_table

st.title("⬇️ Download Reports")

st.subheader("Current assessment")
if common.has_results():
    r = st.session_state.results
    meta = common.bundle()["meta"]
    c1, c2, c3 = st.columns(3)
    c1.download_button("Complete results (Excel, 6 sheets)", bulk_excel(r, st.session_state.issues, meta),
                       "loan_risk_assessment_results.xlsx", type="primary", width="stretch")
    c2.download_button("Results (CSV)", output_table(r).to_csv(index=False).encode("utf-8-sig"),
                       "loan_risk_assessment_results.csv", width="stretch")
    c3.download_button("Validation issues (CSV)", st.session_state.issues.to_csv(index=False).encode("utf-8-sig"),
                       "validation_issues.csv", width="stretch")
    st.caption("Excel sheets: Assessment Results · Portfolio Summary · High Risk · Manual Review · Validation Issues · About.")
    ids = r.sort_values("default_probability", ascending=False, na_position="last")["applicant_id"].tolist()
    sel = st.selectbox("Individual applicant report", ids)
    row = r[r["applicant_id"] == sel].iloc[0]
    st.download_button(f"Report for {sel} (HTML - open and print to PDF)",
                       applicant_html(row, st.session_state.details.get(sel), common.settings(), meta).encode("utf-8"),
                       f"assessment_{sel}.html", mime="text/html")
else:
    common.no_results_hint()

st.subheader("Model documentation")
c1, c2 = st.columns(2)
c1.download_button("Model evaluation report (Markdown)", (REPORT_DIR / "MODEL_EVALUATION.md").read_bytes(),
                   "MODEL_EVALUATION.md", width="stretch")
c2.download_button("Model metrics (JSON)", (MODEL_DIR / "metrics.json").read_bytes(), "metrics.json",
                   width="stretch")

st.subheader("Saved assessment history (local SQLite)")
b = storage.list_batches()
if b.empty:
    st.caption("No saved assessments.")
else:
    st.dataframe(b, hide_index=True, width="stretch")
    c1, c2 = st.columns([2, 1])
    bid = c1.selectbox("Load a previous batch into the app", b["id"].tolist(),
                       format_func=lambda i: f"#{i} - {b.set_index('id').loc[i, 'source']} "
                                             f"({b.set_index('id').loc[i, 'created_at']})")
    if c2.button("Load batch", width="stretch"):
        from loanrisk.schema import FIELDS
        df = storage.load_batch(int(bid))
        raw = df[[f.name for f in FIELDS if f.name in df.columns]]
        common.run_assessment(raw, f"saved batch #{bid} (re-assessed)", persist=False)
        st.success("Loaded and re-assessed with the current model and thresholds. Dashboards now show this batch.")
    with st.expander("Data protection"):
        st.markdown("Applicant data is stored only in the local file `data/assessments.db`; nothing is sent to any "
                    "external service. Delete it when it is no longer needed.")
        if st.checkbox("I understand this permanently deletes all saved assessments"):
            if st.button("Delete all saved applicant data", type="secondary"):
                storage.delete_all()
                st.success("All saved assessments deleted.")
                st.rerun()
common.disclaimer()
