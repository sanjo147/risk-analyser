import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

import common
from loanrisk.config import REPORT_DIR
from loanrisk.risk import approve_threshold

st.title("🎯 Model Performance")
m = common.metrics()
if not m:
    st.error("metrics.json not found - run the training script.")
    st.stop()
best = m["best"]
t = m["test"][best]
st.caption(f"All figures are computed on **{m['meta']['n_test']:,} held-out test loans** that were never used for "
           f"training, calibration or model selection. Selected model: **{best}**. Default rate: "
           f"{m['meta']['default_rate']:.2%}.")

thr_choice = st.radio("Classification threshold for the metrics below",
                      [f"Policy: not auto-approved (PD > {m['policy_threshold']:.0%})", "0.50"], horizontal=True)
x = t["at_policy"] if thr_choice.startswith("Policy") else t["at_0_5"]
c = st.columns(5)
common.kpi(c[0], "ROC-AUC", f"{t['roc_auc']:.3f}")
common.kpi(c[1], "PR-AUC", f"{t['pr_auc']:.3f}")
common.kpi(c[2], "Brier score", f"{t['brier']:.4f}")
common.kpi(c[3], "Accuracy", f"{x['accuracy']:.3f}")
common.kpi(c[4], "F1", f"{x['f1']:.3f}")
c = st.columns(5)
common.kpi(c[0], "Precision", f"{x['precision']:.3f}")
common.kpi(c[1], "Recall (defaulters caught)", f"{x['recall']:.3f}")
common.kpi(c[2], "False negatives", f"{x['fn']:,}", "#c0262d")
common.kpi(c[3], "False-negative rate", f"{x['false_negative_rate']:.1%}", "#c0262d")
common.kpi(c[4], "AUC 95% CI", f"{m['test_auc_95ci'][0]:.3f}–{m['test_auc_95ci'][1]:.3f}")

st.warning(f"**False negatives matter most.** At this threshold {x['fn']:,} of {x['fn'] + x['tp']:,} actual defaulters "
           f"({x['false_negative_rate']:.0%}) would have been scored as safe. A ROC-AUC around {t['roc_auc']:.2f} means "
           "the model ranks risk meaningfully better than chance but is far from perfect - which is why it is positioned "
           f"as decision support with a manual-review band. Accuracy is misleading here: approving everyone would score "
           f"{1 - m['meta']['default_rate']:.1%}.")

tp = common.test_predictions()
y, p = tp["y_true"].values, tp["prob"].values
c1, c2 = st.columns(2)
cm = np.array([[x["tn"], x["fp"]], [x["fn"], x["tp"]]])
fig = px.imshow(cm, text_auto=",d", color_continuous_scale="Blues", x=["Predicted: repaid", "Predicted: default"],
                y=["Actual: repaid", "Actual: default"], title=f"Confusion matrix ({thr_choice})")
fig.update_layout(coloraxis_showscale=False)
c1.plotly_chart(fig, width="stretch")
fx, fy = calibration_curve(y, p, n_bins=15, strategy="quantile")
fig = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Perfect"),
                 go.Scatter(x=fy, y=fx, mode="lines+markers", name=best)])
fig.update_layout(title="Calibration curve (test)", xaxis_title="Predicted probability",
                  yaxis_title="Observed default rate", xaxis_range=[0, max(0.6, fy.max() + .05)],
                  yaxis_range=[0, max(0.6, fx.max() + .05)])
c2.plotly_chart(fig, width="stretch")

c1, c2 = st.columns(2)
fpr, tpr, _ = roc_curve(y, p)
c1.plotly_chart(px.area(x=fpr, y=tpr, title=f"ROC curve (AUC {t['roc_auc']:.3f})",
                        labels={"x": "False positive rate", "y": "True positive rate"}).add_shape(
    type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash")), width="stretch")
pr, rc, _ = precision_recall_curve(y, p)
c2.plotly_chart(px.line(x=rc, y=pr, title=f"Precision-recall curve (AP {t['pr_auc']:.3f})",
                        labels={"x": "Recall", "y": "Precision"}).add_hline(
    y=y.mean(), line_dash="dash", annotation_text="base rate"), width="stretch")

st.subheader("Model comparison")
rows = []
for name, r in m["comparison"].items():
    te = m["test"][name]
    rows.append({"Model": name + (" ✅" if name == best else ""), "CV ROC-AUC": f"{r['cv_roc_auc_mean']:.4f} ± {r['cv_roc_auc_std']:.4f}",
                 "CV PR-AUC": round(r["cv_pr_auc_mean"], 4), "Train AUC": round(r["train_roc_auc"], 4),
                 "Valid AUC": round(r["val_roc_auc"], 4), "Overfit gap": round(r["generalization_gap"], 4),
                 "Valid Brier (calibrated)": round(r["val_brier_calibrated"], 4), "Test ROC-AUC": round(te["roc_auc"], 4),
                 "Test PR-AUC": round(te["pr_auc"], 4), "Test Brier": round(te["brier"], 4),
                 "Test recall @policy": round(te["at_policy"]["recall"], 3), "Fit time (s)": r["fit_seconds"]})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.caption(f"Selection rule: {m['selection_rule']}. Test scores are shown for transparency only; they were not used "
           "to pick the model. 'Overfit gap' = train AUC − validation AUC.")

st.subheader("Fairness and bias evaluation (test set)")
st.caption("Protected attributes (gender, religion, caste, race, age, marital status) are not in the training data and "
           "are never model inputs. Geography is excluded from the model as a potential proxy and appears below only "
           "for auditing. Approval ratio below 0.80 (four-fifths rule) is flagged. FNR = share of actual defaulters "
           "approved; FPR = share of good borrowers not approved.")
for key, g in m["fairness"].items():
    df = pd.DataFrame(g["rows"])
    df["flag"] = np.where(df["approval_ratio_vs_max"] < 0.8, "⚠️ < 0.8", "")
    with st.expander(g["title"], expanded=key == "audit_region"):
        st.dataframe(df.rename(columns={"group": "Group", "n": "n", "actual_default_rate": "Actual default rate",
                                        "mean_predicted": "Mean predicted", "approval_rate": "Approval rate",
                                        "approval_ratio_vs_max": "Approval ratio", "fnr": "FNR", "fpr": "FPR",
                                        "roc_auc": "AUC", "flag": "Flag"}).round(3),
                     hide_index=True, width="stretch")
st.caption("Differences in approval rate largely track differences in actual historical default rates; a gap is a "
           "prompt for investigation, not by itself proof of unfairness or of fairness.")

with st.expander("Full written evaluation report"):
    st.markdown((REPORT_DIR / "MODEL_EVALUATION.md").read_text(encoding="utf-8").split("![ROC")[0])
    st.image(str(REPORT_DIR / "figures" / "score_distribution.png"))
common.disclaimer()
