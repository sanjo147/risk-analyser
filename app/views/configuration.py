import copy

import numpy as np
import pandas as pd
import streamlit as st

import common
from loanrisk.config import DEFAULT_SETTINGS, save_settings, validate_bands
from loanrisk.risk import categorize, wilson_ci

st.title("⚙️ Model Configuration")
st.caption("Risk bands translate the model's probability into a category; the decision policy maps each category to a "
           "recommendation. Both are business choices - they do not change the model itself.")

s = copy.deepcopy(common.settings())
DECISIONS = ["APPROVE", "MANUAL REVIEW", "REJECT RECOMMENDATION"]

st.subheader("1 · Risk bands and decision policy")
bands_df = pd.DataFrame([{"Risk category": b["name"], "Upper bound (%)": round(b["max"] * 100, 2),
                          "Policy decision": b["decision"]} for b in s["risk_bands"]])
edited = st.data_editor(bands_df, hide_index=True, width="stretch", num_rows="fixed", column_config={
    "Risk category": st.column_config.TextColumn(disabled=True),
    "Upper bound (%)": st.column_config.NumberColumn(min_value=0.1, max_value=100.0, step=0.5),
    "Policy decision": st.column_config.SelectboxColumn(options=DECISIONS, required=True),
}, key="bands_editor")
new_bands = [{"name": r["Risk category"], "max": float(r["Upper bound (%)"]) / 100, "decision": r["Policy decision"]}
             for _, r in edited.iterrows()]
try:
    validate_bands(new_bands)
    bands_ok = True
except ValueError as e:
    st.error(str(e))
    bands_ok = False

st.subheader("2 · Calibration of these thresholds against historical outcomes")
st.caption("Held-out test loans (actual outcomes known) are grouped with the bands above. Use this to check that each "
           "band's real default rate matches its label before relying on it.")
tp = common.test_predictions()
if bands_ok:
    cats, decs = categorize(tp["prob"].values, new_bands)
    tp = tp.assign(cat=cats, dec=decs)
    rows = []
    for b in new_bands:
        sub = tp[tp["cat"] == b["name"]]
        k, n = int(sub["y_true"].sum()), len(sub)
        lo, hi = wilson_ci(k, n)
        rows.append({"Risk category": b["name"], "Test loans": n, "Share": f"{n / len(tp):.1%}",
                     "Mean predicted PD": f"{sub['prob'].mean():.1%}" if n else "-",
                     "Observed default rate": f"{k / n:.1%}" if n else "-",
                     "95% CI": f"{lo:.1%} – {hi:.1%}" if n else "-", "Decision": b["decision"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    appr = tp["dec"] == "APPROVE"
    rej = tp["dec"] == "REJECT RECOMMENDATION"
    c = st.columns(4)
    common.kpi(c[0], "Auto-approval rate", f"{appr.mean():.1%}")
    common.kpi(c[1], "Default rate among approved", f"{tp.loc[appr, 'y_true'].mean():.1%}" if appr.any() else "-")
    common.kpi(c[2], "Defaulters not auto-approved", f"{(~appr & (tp.y_true == 1)).sum() / tp.y_true.sum():.1%}")
    common.kpi(c[3], "Reject rate", f"{rej.mean():.1%}")
    st.caption(f"Base default rate in the test data: {tp['y_true'].mean():.1%}. Policy overlays (below) are not "
               "applied here because the training data lacks expenses, savings and age.")

st.subheader("3 · Policy overlays (rule-based, applied after the model)")
st.caption("These rules can only escalate a recommendation (Approve → Review → Reject), never relax it.")
p = s["policy"]
c1, c2, c3 = st.columns(3)
p["enabled"] = c1.toggle("Enable policy overlays", value=p["enabled"])
p["min_credit_score"] = c1.number_input("Minimum credit score (below → reject)", 300, 900, int(p["min_credit_score"]))
p["max_total_emi_to_income"] = c2.slider("Max total EMI / income (above → review)", 0.1, 1.0,
                                         float(p["max_total_emi_to_income"]), 0.05)
p["min_disposable_income"] = c2.number_input("Min disposable income after EMIs (below → review)",
                                             value=float(p["min_disposable_income"]), step=1000.0)
p["min_age"] = c3.number_input("Minimum age (eligibility)", 18, 30, int(p["min_age"]))
p["max_age_at_maturity"] = c3.number_input("Max age at loan maturity", 50, 90, int(p["max_age_at_maturity"]))
p["uncertainty_review_width"] = c1.slider("Send approvals to review if the 95% interval is wider than", 0.02, 0.5,
                                          float(p["uncertainty_review_width"]), 0.01)

st.subheader("4 · Display")
c1, c2 = st.columns(2)
s["currency_symbol"] = c1.text_input("Currency symbol", s["currency_symbol"])
s["currency_locale"] = c2.selectbox("Number grouping", ["IN", "INTL"], index=0 if s["currency_locale"] == "IN" else 1,
                                    format_func=lambda x: "Indian (5,00,000)" if x == "IN" else "International (500,000)")

c1, c2, c3 = st.columns(3)
if c1.button("💾 Save configuration", type="primary", disabled=not bands_ok, width="stretch"):
    s["risk_bands"] = new_bands
    s["policy"] = p
    save_settings(s)
    st.success("Saved to config/settings.json.")
if c2.button("Reset to defaults", width="stretch"):
    save_settings(copy.deepcopy(DEFAULT_SETTINGS))
    st.session_state.pop("bands_editor", None)
    st.rerun()
if c3.button("Re-apply to the loaded assessment", disabled=not common.has_results(), width="stretch"):
    from loanrisk.schema import FIELDS
    r = st.session_state.results
    raw = r[[f.name for f in FIELDS if f.name in r.columns]]
    common.run_assessment(raw, st.session_state.source, persist=False)
    st.success("Loaded assessment re-scored with the saved configuration.")

with st.expander("Model information"):
    meta = common.bundle()["meta"]
    st.json({k: v for k, v in meta.items() if k != "features"})
    st.write("Model features:", ", ".join(meta["features"]))
    st.caption("To retrain: `python -m loanrisk.data_prep` then `python -m loanrisk.train` and restart the app.")
common.disclaimer()
