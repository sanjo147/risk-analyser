"""Shared state, styling and UI components for the Streamlit app."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loanrisk import storage  # noqa: E402
from loanrisk.config import MODEL_DIR, load_settings  # noqa: E402
from loanrisk.predictor import ModelNotTrainedError, assess, load_bundle  # noqa: E402
from loanrisk.reporting import BAND_COLOURS, DECISION_COLOURS, applicant_html, money  # noqa: E402
from loanrisk.schema import FEATURE_LABELS  # noqa: E402

BAND_ORDER = ["Very Low Risk", "Low Risk", "Moderate Risk", "High Risk", "Very High Risk"]
DECISION_ORDER = ["APPROVE", "MANUAL REVIEW", "REJECT RECOMMENDATION", "CANNOT ASSESS"]

CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1400px;}
.kpi {border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: 14px 16px; background: var(--kpi-bg, rgba(31,58,95,.04));}
.kpi .v {font-size: 1.7rem; font-weight: 700; line-height: 1.2;}
.kpi .l {font-size: .8rem; opacity: .75; text-transform: uppercase; letter-spacing: .04em;}
.pill {display:inline-block; color:#fff; padding:3px 12px; border-radius:14px; font-weight:600; font-size:.9rem;}
.disclaimer {font-size:.78rem; opacity:.7; border-top:1px solid rgba(128,128,128,.25); padding-top:8px; margin-top:24px;}
.layer {font-size:.72rem; text-transform:uppercase; letter-spacing:.06em; opacity:.65; margin-bottom:2px;}
</style>
"""


def setup():
    st.markdown(CSS, unsafe_allow_html=True)
    ss = st.session_state
    ss.setdefault("results", None)
    ss.setdefault("issues", None)
    ss.setdefault("details", {})
    ss.setdefault("source", None)
    ss.setdefault("raw_mapped", None)


def settings() -> dict:
    return load_settings()


def model_ready() -> bool:
    try:
        load_bundle()
        return True
    except ModelNotTrainedError:
        st.error("No trained model found. Run `python -m loanrisk.train` (see README) and reload.")
        return False


def bundle() -> dict:
    return load_bundle()


@st.cache_data(show_spinner=False)
def metrics() -> dict:
    p = MODEL_DIR / "metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@st.cache_data(show_spinner=False)
def test_predictions() -> pd.DataFrame:
    return pd.read_csv(MODEL_DIR / "test_predictions.csv.gz")


def run_assessment(raw_mapped: pd.DataFrame, source: str, persist: bool = True):
    with st.spinner(f"Assessing {len(raw_mapped):,} applicant(s) with the trained model..."):
        res, issues, details = assess(raw_mapped, settings())
    ss = st.session_state
    ss.results, ss.issues, ss.details, ss.source, ss.raw_mapped = res, issues, details, source, raw_mapped
    if persist:
        try:
            storage.save_batch(res, source, bundle()["meta"]["model_name"])
        except Exception as e:  # storage must never block an assessment
            st.warning(f"Results could not be saved to the local database: {e}")
    return res, issues, details


def has_results() -> bool:
    r = st.session_state.get("results")
    return r is not None and len(r) > 0


def no_results_hint():
    st.info("No assessment loaded yet. Upload a dataset on **Upload Customer Dataset**, "
            "assess an applicant on **Individual Assessment**, or load the bundled sample below.")
    if st.button("Assess the bundled synthetic sample dataset (250 applicants)", type="primary"):
        from loanrisk.config import SAMPLE_DIR
        from loanrisk.features import apply_mapping, suggest_mapping
        df = pd.read_csv(SAMPLE_DIR / "sample_applicants.csv")
        run_assessment(apply_mapping(df, suggest_mapping(list(df.columns))), "sample_applicants.csv")
        st.rerun()


def fmt_money(v) -> str:
    s = settings()
    return money(v, s.get("currency_symbol", "₹"), s.get("currency_locale", "IN"))


def kpi(col, label, value, colour=None):
    style = f"color:{colour}" if colour else ""
    col.markdown(f"<div class='kpi'><div class='l'>{label}</div><div class='v' style='{style}'>{value}</div></div>",
                 unsafe_allow_html=True)


def pill(text, colour):
    return f"<span class='pill' style='background:{colour}'>{text}</span>"


def disclaimer():
    st.markdown("<div class='disclaimer'>Educational decision-support prototype. Default probabilities are estimates "
                "from a model trained on historical LendingClub (US) loans; recommendations follow an illustrative, "
                "configurable policy and are <b>not</b> lending decisions. Feature contributions show associations, "
                "not causes.</div>", unsafe_allow_html=True)


def gauge(p: float) -> go.Figure:
    bands = settings()["risk_bands"]
    steps, lo = [], 0
    for b in bands:
        steps.append(dict(range=[lo * 100, b["max"] * 100], color=BAND_COLOURS.get(b["name"], "#999")))
        lo = b["max"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100, number=dict(suffix="%", valueformat=".1f"),
        gauge=dict(axis=dict(range=[0, 100], ticksuffix="%"), bar=dict(color="#111827", thickness=0.25),
                   steps=steps, threshold=dict(line=dict(color="#111827", width=4), value=p * 100))))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=30, b=0))
    return fig


def contribution_chart(contrib: dict, top: int = 10) -> go.Figure:
    s = pd.Series(contrib).dropna()
    s = s[s.abs() > 1e-4]
    s = s.reindex(s.abs().sort_values(ascending=False).index)[:top][::-1]
    labels = [FEATURE_LABELS.get(k, k) for k in s.index]
    colours = ["#c0262d" if v > 0 else "#1b7f3b" for v in s.values]
    fig = go.Figure(go.Bar(x=s.values, y=labels, orientation="h", marker_color=colours,
                           hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
    fig.update_layout(height=40 + 32 * len(s), margin=dict(l=10, r=10, t=10, b=30),
                      xaxis_title="← reduces risk   |   increases risk →  (contribution, log-odds)")
    return fig


def render_profile(row: pd.Series, detail: dict | None):
    """Full individual assessment report."""
    s = settings()
    st.markdown(f"### Applicant {row['applicant_id']}")
    if row["recommendation"] == "CANNOT ASSESS" or detail is None:
        st.error(f"**Prediction cannot be generated.** {row.get('policy_notes', '')}")
        return
    p = float(row["default_probability"])
    c1, c2, c3 = st.columns([1.1, 1, 1])
    with c1:
        st.markdown("<div class='layer'>Model prediction</div>", unsafe_allow_html=True)
        st.plotly_chart(gauge(p), width="stretch", key=f"g_{row['applicant_id']}")
    with c2:
        st.markdown("<div class='layer'>Model prediction</div>", unsafe_allow_html=True)
        st.markdown(f"**Predicted default probability:** {p:.1%}")
        st.markdown("**Risk category:** " + pill(row["risk_category"], BAND_COLOURS.get(row["risk_category"], "#666")),
                    unsafe_allow_html=True)
        b = detail["calibration_bin"]
        st.markdown(f"**Confidence:** {row['confidence']}")
        st.caption(f"Historical check: held-out test applicants scored {b['lo']:.1%}–{b['hi']:.1%} defaulted at "
                   f"{b['observed']:.1%} (95% CI {b['ci_low']:.1%}–{b['ci_high']:.1%}). "
                   f"{detail['imputed']} optional input(s) were missing and filled with typical training values."
                   + (f" Outside training range: {row['ood_features']}." if row.get("ood_features") else ""))
    with c3:
        st.markdown("<div class='layer'>Lending policy (illustrative)</div>", unsafe_allow_html=True)
        st.markdown("**Recommendation:**<br>" + pill(row["recommendation"], DECISION_COLOURS[row["recommendation"]]),
                    unsafe_allow_html=True)
        st.caption(f"Band default decision: {row['model_band_decision']}.")
        if row.get("policy_notes"):
            st.warning(row["policy_notes"])
        else:
            st.caption("No policy overlay was triggered.")

    st.markdown("#### Financial summary")
    m = fmt_money
    dti = row.get("dti_ratio")
    summ = pd.DataFrame([
        ("Requested loan", m(row["loan_amount"])), ("Tenure / rate", f"{row['loan_tenure_months']:.0f} months @ {row['interest_rate']:.2f}%"),
        ("Estimated EMI", m(row["expected_emi"])), ("Monthly income (incl. additional)", m(row["total_monthly_income"])),
        ("Existing EMI / liabilities", m(row.get("existing_emi"))), ("Debt-to-income ratio", f"{dti:.1f}%" if pd.notna(dti) else "-"),
        ("Total EMI burden (existing + new)", f"{row['emi_to_income_total']:.0%} of income"),
        ("Disposable income after EMIs", m(row.get("disposable_income"))), ("Credit score", f"{row['credit_score']:.0f}"),
        ("Outstanding debt", m(row.get("outstanding_debt"))),
    ], columns=["Item", "Value"])
    a, b2 = st.columns(2)
    a.dataframe(summ.iloc[:5], hide_index=True, width="stretch")
    b2.dataframe(summ.iloc[5:], hide_index=True, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### ✅ Positive financial indicators")
        for x in detail["positive"] or ["None identified."]:
            st.markdown(f"- {x}")
    with c2:
        st.markdown("#### ⚠️ Negative financial indicators")
        for x in detail["negative"] or ["None identified."]:
            st.markdown(f"- {x}")

    st.markdown("#### Why the model scored it this way")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top factors increasing estimated risk**")
        for d in detail["increasing"] or [{"text": "None."}]:
            st.markdown(f"- {d['text']}")
    with c2:
        st.markdown("**Top factors reducing estimated risk**")
        for d in detail["reducing"] or [{"text": "None."}]:
            st.markdown(f"- {d['text']}")
    st.plotly_chart(contribution_chart(detail["contributions"]), width="stretch",
                    key=f"c_{row['applicant_id']}")
    st.info(f"**Reason for recommendation.** {detail['reason']}")
    st.markdown("#### Suggestions for improving creditworthiness")
    for x in detail["suggestions"]:
        st.markdown(f"- {x}")
    html = applicant_html(row, detail, s, bundle()["meta"])
    st.download_button("⬇️ Download this report (HTML, printable to PDF)", html.encode("utf-8"),
                       file_name=f"assessment_{row['applicant_id']}.html", mime="text/html",
                       key=f"dl_{row['applicant_id']}")


def filter_bar(df: pd.DataFrame, key: str) -> pd.DataFrame:
    c1, c2, c3 = st.columns([1.2, 1.2, 1])
    cats = c1.multiselect("Risk category", [c for c in BAND_ORDER if c in set(df["risk_category"].dropna())],
                          key=f"{key}_cat")
    decs = c2.multiselect("Recommendation", [d for d in DECISION_ORDER if d in set(df["recommendation"])],
                          key=f"{key}_dec")
    q = c3.text_input("Search Applicant ID", key=f"{key}_q")
    out = df
    if cats:
        out = out[out["risk_category"].isin(cats)]
    if decs:
        out = out[out["recommendation"].isin(decs)]
    if q:
        out = out[out["applicant_id"].astype(str).str.contains(q, case=False, regex=False)]
    return out


def results_table(df: pd.DataFrame, key: str, height: int = 420):
    from loanrisk.reporting import output_table
    t = output_table(df)
    t.insert(0, "Rank", np.arange(1, len(t) + 1))
    st.dataframe(t, hide_index=True, width="stretch", height=height, key=key, column_config={
        "Default Probability": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        "Requested Loan": st.column_config.NumberColumn(format="localized"),
        "Monthly Income": st.column_config.NumberColumn(format="localized"),
        "Credit Score": st.column_config.NumberColumn(format="%d"),
    })
