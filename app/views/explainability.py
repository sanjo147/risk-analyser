import pandas as pd
import plotly.express as px
import streamlit as st

import common
from loanrisk.schema import FEATURE_LABELS

st.title("🧠 Model Explainability")
st.markdown("Explanations use **SHAP** (Shapley additive explanations): each input receives a share of the difference "
            "between the applicant's score and the average score. Positive values push the estimated risk up, negative "
            "values pull it down.")
st.info("**Important:** SHAP shows how the model *uses* information, based on patterns in historical loans. It does "
        "not prove that a feature *causes* default, and changing one input in real life may not change risk as the "
        "chart suggests.")

m = common.metrics()
gi = pd.Series(m["global_importance"]).sort_values()
direction = m.get("shap_direction", {})
st.subheader("Overall feature importance")
df = pd.DataFrame({"Feature": [FEATURE_LABELS.get(k, k) for k in gi.index], "Mean |SHAP|": gi.values})
st.plotly_chart(px.bar(df, x="Mean |SHAP|", y="Feature", orientation="h", height=560,
                       title=f"Mean absolute contribution on 2,000 test applicants ({m['best']})"),
                width="stretch")

rows = []
for f in gi.index[::-1]:
    d = direction.get(f)
    if d is None:
        txt = "Categorical - effect depends on the category"
    elif d > 0.3:
        txt = "Higher values → higher estimated risk"
    elif d < -0.3:
        txt = "Higher values → lower estimated risk"
    else:
        txt = "Non-linear / mixed relationship"
    rows.append({"Feature": FEATURE_LABELS.get(f, f), "Typical direction in the model": txt})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

st.subheader("Explain an individual applicant")
if not common.has_results():
    common.no_results_hint()
    st.stop()
r = st.session_state.results
ok = r[r["default_probability"].notna()].sort_values("default_probability", ascending=False)
sel = st.selectbox("Applicant", ok["applicant_id"].tolist(),
                   format_func=lambda i: f"{i} - PD {ok.set_index('applicant_id').loc[i, 'default_probability']:.1%}")
det = st.session_state.details.get(sel)
row = ok[ok["applicant_id"] == sel].iloc[0]
st.markdown(f"Predicted default probability **{row['default_probability']:.1%}** · {row['risk_category']} · "
            f"{row['recommendation']}")
st.plotly_chart(common.contribution_chart(det["contributions"], top=19), width="stretch")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Top 5 factors increasing risk**")
    for d in det["increasing"] or [{"text": "None."}]:
        st.markdown(f"- {d['text']}")
with c2:
    st.markdown("**Top 5 factors reducing risk**")
    for d in det["reducing"] or [{"text": "None."}]:
        st.markdown(f"- {d['text']}")
with st.expander("All feature contributions (table)"):
    feats = det["model_features"]
    t = pd.DataFrame({"Feature": [FEATURE_LABELS.get(k, k) for k in det["contributions"]],
                      "Model input value": [feats.get(k) for k in det["contributions"]],
                      "Contribution (log-odds)": list(det["contributions"].values())})
    st.dataframe(t.sort_values("Contribution (log-odds)", key=abs, ascending=False), hide_index=True,
                 width="stretch")
    st.caption("Ratio inputs (e.g. loan-to-income) are derived from the raw application. Missing optional inputs "
               "appear as blank and were filled with the training median inside the model pipeline.")
common.disclaimer()
