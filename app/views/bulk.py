import streamlit as st

import common
from loanrisk.reporting import bulk_excel, portfolio_summary

st.title("📋 Bulk Risk Assessment")
if not common.has_results():
    common.no_results_hint()
    st.stop()

r = st.session_state.results
st.caption(f"Source: **{st.session_state.source}** · ranked by predicted default probability (highest first).")
summ = portfolio_summary(r)
c = st.columns(4)
vals = dict(zip(summ["Metric"], summ["Value"]))
common.kpi(c[0], "Assessed", f"{vals['Applicants assessed']:,} / {vals['Total applicants in file']:,}")
common.kpi(c[1], "Total requested", common.fmt_money(vals["Total requested amount"]))
common.kpi(c[2], "Expected-loss proxy (Σ PD × amount)", common.fmt_money(vals["Expected loss proxy (sum of PD x amount)"]))
apd = vals["Average predicted default probability"]
common.kpi(c[3], "Average PD", f"{apd:.1%}" if apd is not None else "-")

st.download_button("⬇️ Download complete results (Excel)", bulk_excel(r, st.session_state.issues, common.bundle()["meta"]),
                   file_name="loan_risk_assessment_results.xlsx", type="primary",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

tab1, tab2, tab3, tab4 = st.tabs(["All applicants (ranked)", "🔴 High-risk applicants", "🟡 Manual review",
                                  "⛔ Could not assess"])
with tab1:
    common.results_table(common.filter_bar(r, "bulk"), key="bulk_all")
with tab2:
    hr = r[r["recommendation"] == "REJECT RECOMMENDATION"]
    st.caption(f"{len(hr)} applicant(s) with a reject recommendation (High / Very High risk band or a policy rule).")
    common.results_table(hr, key="bulk_hr")
with tab3:
    mr = r[r["recommendation"] == "MANUAL REVIEW"]
    st.caption(f"{len(mr)} applicant(s) require a human underwriter (Moderate risk, a policy check, or low confidence).")
    common.results_table(mr, key="bulk_mr")
with tab4:
    na = r[r["recommendation"] == "CANNOT ASSESS"]
    if len(na):
        st.dataframe(na[["applicant_id", "policy_notes"]].rename(columns={"policy_notes": "Reason"}),
                     hide_index=True, width="stretch")
    else:
        st.success("Every applicant had the critical information needed.")
st.caption("Open **Applicant Results** to see any applicant's full profile and explanation.")
common.disclaimer()
