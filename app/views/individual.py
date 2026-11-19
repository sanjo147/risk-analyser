import pandas as pd
import streamlit as st

import common
from loanrisk.features import amortized_emi
from loanrisk.predictor import assess
from loanrisk.schema import FIELDS

st.title("👤 Individual Applicant Assessment")
st.caption("Enter one applicant. Fields marked ★ are required; optional fields left blank are filled with typical "
           "values from the training data, which lowers the prediction confidence.")

DEFAULTS = {  # illustrative starting values (editable)
    "applicant_id": "LA9001", "age": 34, "marital_status": "Married", "dependents": 1, "residential_status": "Rented",
    "employment_type": "Salaried", "employer_category": "IT / Software", "employment_years": 6.0,
    "monthly_income": 45000.0, "additional_income": 0.0, "employment_stability": "Stable", "monthly_expenses": 18000.0,
    "existing_emi": 8000.0, "active_loans": 3, "bank_balance": 60000.0, "savings": 150000.0, "credit_score": 760.0,
    "previous_loans": 8, "previous_defaults": 0, "delayed_payments_2y": 0, "recent_inquiries": 1,
    "credit_utilization": 28.0, "outstanding_debt": 40000.0, "credit_history_years": 9.0, "bankruptcies": 0,
    "mortgage_accounts": 0, "loan_amount": 500000.0, "loan_tenure_months": 36, "interest_rate": 12.5,
    "loan_purpose": "home_improvement", "expected_emi": None, "dti_ratio": None,
}
GROUPS = ["Personal", "Employment", "Financial", "Credit", "Loan"]

values = {}
with st.form("applicant"):
    tabs = st.tabs([f"{g} information" for g in GROUPS])
    for tab, g in zip(tabs, GROUPS):
        with tab:
            cols = st.columns(3)
            for i, f in enumerate([f for f in FIELDS if f.group == g]):
                label = ("★ " if f.critical else "") + f.label + (f" ({f.unit})" if f.unit and f.unit != "currency" else "")
                if f.unit == "currency":
                    label += f" ({common.settings()['currency_symbol']})"
                help_ = f.help or ("Used by the model." if f.used_by_model else "Not used by the model.")
                col = cols[i % 3]
                d = DEFAULTS.get(f.name)
                if f.kind == "cat":
                    opts = list(f.choices)
                    values[f.name] = col.selectbox(label, opts, index=opts.index(d) if d in opts else 0, help=help_,
                                                   format_func=lambda x: x.replace("_", " ").capitalize())
                elif f.kind == "text":
                    values[f.name] = col.text_input(label, value=d or "", help=help_)
                else:
                    step = 1.0 if f.kind == "int" else (1000.0 if f.unit == "currency" else 0.5)
                    values[f.name] = col.number_input(label, value=None if d is None else float(d), step=step,
                                                      min_value=float(f.min) if f.min is not None else None,
                                                      max_value=float(f.max) if f.max is not None else None,
                                                      help=help_, placeholder="optional" if not f.critical else "required")
    submitted = st.form_submit_button("Assess applicant", type="primary")

if values.get("loan_amount") and values.get("interest_rate") is not None and values.get("loan_tenure_months"):
    emi = float(amortized_emi(values["loan_amount"], values["interest_rate"], values["loan_tenure_months"]))
    st.caption(f"Computed EMI for the entered loan: **{common.fmt_money(emi)}** / month "
               "(used when *Expected monthly EMI* is blank).")

if submitted:
    raw = pd.DataFrame([values])
    with st.spinner("Scoring with the trained model…"):
        res, issues, details = assess(raw, common.settings())
    st.session_state["individual_last"] = (res, issues, details)
    try:
        from loanrisk import storage
        storage.save_batch(res, f"individual:{values['applicant_id']}", common.bundle()["meta"]["model_name"])
    except Exception:
        pass

if "individual_last" in st.session_state:
    res, issues, details = st.session_state["individual_last"]
    warn = issues[issues.severity != "info"]
    for _, w in warn.iterrows():
        (st.error if w.severity == "error" else st.warning)(w.message)
    info = issues[issues.severity == "info"]
    if len(info):
        st.caption("ℹ️ " + info.iloc[0].message + " Fields: " + info.iloc[0].field)
    row = res.iloc[0]
    st.divider()
    common.render_profile(row, details.get(row["applicant_id"]))
common.disclaimer()
