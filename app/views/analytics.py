import pandas as pd
import plotly.express as px
import streamlit as st

import common
from loanrisk.reporting import BAND_COLOURS

st.title("📊 Risk Analytics")
if not common.has_results():
    common.no_results_hint()
    st.stop()

a = st.session_state.results
a = a[a["default_probability"].notna()].copy()
a["employment_type"] = a["employment_type"].fillna("Not provided")
a["loan_purpose"] = a["loan_purpose"].fillna("not provided")

with st.expander("Filters", expanded=True):
    c1, c2, c3 = st.columns(3)
    et = c1.multiselect("Employment type", sorted(a["employment_type"].unique()))
    lp = c2.multiselect("Loan purpose", sorted(a["loan_purpose"].unique()))
    rc = c3.multiselect("Risk category", [b for b in common.BAND_ORDER if b in set(a["risk_category"])])
if et:
    a = a[a["employment_type"].isin(et)]
if lp:
    a = a[a["loan_purpose"].isin(lp)]
if rc:
    a = a[a["risk_category"].isin(rc)]
if a.empty:
    st.warning("No applicants match the filters.")
    st.stop()
st.caption(f"{len(a):,} applicants in view.")
a["PD %"] = a["default_probability"] * 100
cmap = dict(color_discrete_map=BAND_COLOURS, category_orders={"risk_category": common.BAND_ORDER})

c1, c2 = st.columns(2)
c1.plotly_chart(px.scatter(a, x="total_monthly_income", y="PD %", color="risk_category", log_x=True,
                           hover_data=["applicant_id", "credit_score", "loan_amount"], title="Income vs default probability",
                           labels={"total_monthly_income": "Monthly income (log scale)", "risk_category": "Risk"}, **cmap),
                width="stretch")
c2.plotly_chart(px.scatter(a, x="credit_score", y="PD %", color="risk_category", title="Credit score vs default probability",
                           hover_data=["applicant_id"], labels={"credit_score": "Credit score", "risk_category": "Risk"},
                           **cmap), width="stretch")

st.subheader("Risk by employment type")
c1, c2 = st.columns(2)
g = a.groupby("employment_type").agg(applicants=("applicant_id", "count"), avg_pd=("default_probability", "mean"),
                                     approve_rate=("recommendation", lambda s: (s == "APPROVE").mean())).reset_index()
c1.plotly_chart(px.bar(g, x="employment_type", y="avg_pd", text=g["avg_pd"].map("{:.1%}".format),
                       title="Average predicted default probability", labels={"avg_pd": "Avg PD", "employment_type": ""})
                .update_yaxes(tickformat=".0%"), width="stretch")
mix = a.groupby(["employment_type", "risk_category"]).size().reset_index(name="n")
c2.plotly_chart(px.bar(mix, x="employment_type", y="n", color="risk_category", barmode="stack", title="Risk mix",
                       labels={"n": "Applicants", "employment_type": "", "risk_category": "Risk"}, **cmap),
                width="stretch")
st.caption("Employment type is not a model input (it is absent from the training data); differences here arise from "
           "correlated inputs such as income, credit score and debt.")

st.subheader("Risk by loan amount")
try:
    a["amount_band"] = pd.qcut(a["loan_amount"], q=min(5, a["loan_amount"].nunique()), duplicates="drop")
    a["amount_band"] = a["amount_band"].map(lambda iv: f"{common.fmt_money(iv.left)} – {common.fmt_money(iv.right)}")
    g = a.groupby("amount_band", sort=False).agg(applicants=("applicant_id", "count"),
                                                avg_pd=("default_probability", "mean")).reset_index()
    order = a.sort_values("loan_amount")["amount_band"].unique().tolist()
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(g, x="amount_band", y="avg_pd", text=g["avg_pd"].map("{:.1%}".format),
                           category_orders={"amount_band": order}, title="Average PD by loan amount quintile",
                           labels={"avg_pd": "Avg PD", "amount_band": "Requested amount"}).update_yaxes(tickformat=".0%"),
                    width="stretch")
    c2.plotly_chart(px.box(a, x="amount_band", y="PD %", category_orders={"amount_band": order},
                           title="PD spread by loan amount quintile", labels={"amount_band": "Requested amount"}),
                    width="stretch")
except ValueError:
    st.info("Not enough variation in loan amounts to band them.")

st.subheader("Portfolio exposure")
a["expected_loss_proxy"] = a["default_probability"] * a["loan_amount"]
e = a.groupby("risk_category").agg(applicants=("applicant_id", "count"), exposure=("loan_amount", "sum"),
                                   expected_loss=("expected_loss_proxy", "sum")).reindex(common.BAND_ORDER).dropna().reset_index()
e["share_of_exposure"] = e["exposure"] / e["exposure"].sum()
c1, c2 = st.columns(2)
c1.plotly_chart(px.bar(e, x="risk_category", y=["exposure", "expected_loss"], barmode="group",
                       title="Requested exposure vs expected-loss proxy by risk band",
                       labels={"value": "Amount", "risk_category": "", "variable": ""}), width="stretch")
c2.plotly_chart(px.treemap(a, path=["recommendation", "risk_category"], values="loan_amount", color="risk_category",
                           color_discrete_map={**BAND_COLOURS, "(?)": "#888"}, title="Exposure by recommendation and band"),
                width="stretch")
e_disp = e.assign(exposure=e["exposure"].map(common.fmt_money), expected_loss=e["expected_loss"].map(common.fmt_money),
                  share_of_exposure=e["share_of_exposure"].map("{:.1%}".format))
st.dataframe(e_disp, hide_index=True, width="stretch")
st.caption("Expected-loss proxy = predicted default probability × requested amount (assumes 100% loss given default "
           "and full exposure at default - a deliberately simple, conservative illustration).")
common.disclaimer()
