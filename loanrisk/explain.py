"""SHAP-based explanations translated into business language.

Contributions are computed on the underlying (uncalibrated) model in log-odds space;
calibration is a monotonic transform so the direction and ranking of factors carry over.
SHAP values describe how the *model* uses a feature - they are associations learned from
historical data, not proof that the feature causes default.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from .preprocess import as_frame, transformed_to_original
from .schema import FEATURE_LABELS, MODEL_FEATURES

TREE_TYPES = ("XGBClassifier", "RandomForestClassifier", "DecisionTreeClassifier",
              "GradientBoostingClassifier")


class ModelExplainer:
    def __init__(self, pipeline, background: pd.DataFrame):
        self.pre = pipeline.named_steps["pre"]
        self.model = pipeline.named_steps["model"]
        self.orig = transformed_to_original(self.pre)
        bg = as_frame(self.pre, background)
        kind = type(self.model).__name__
        if kind in TREE_TYPES:
            self.explainer = shap.TreeExplainer(self.model)
        elif kind == "LogisticRegression":
            self.explainer = shap.LinearExplainer(self.model, bg)
        else:  # model-agnostic fallback (e.g. KNN)
            f = lambda x: self.model.predict_proba(x)[:, 1]  # noqa: E731
            self.explainer = shap.Explainer(f, shap.maskers.Independent(bg.values, max_samples=50),
                                            algorithm="permutation")
        self.kind = kind

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-applicant contribution of each original model feature (log-odds units)."""
        Xt = as_frame(self.pre, X)
        if self.kind in TREE_TYPES or self.kind == "LogisticRegression":
            sv = self.explainer.shap_values(Xt)
        else:
            sv = self.explainer(Xt.values).values
        sv = np.asarray(sv)
        if sv.ndim == 3:  # (n, features, classes)
            sv = sv[:, :, 1]
        df = pd.DataFrame(sv, columns=self.orig, index=X.index)
        return df.T.groupby(level=0).sum().T.reindex(columns=MODEL_FEATURES)


# ----------------------------------------------------------------------------- language
def _fmt_value(feature: str, v, raw: dict) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "not provided"
    if feature in ("dti_ratio", "credit_utilization", "interest_rate"):
        return f"{v:.1f}%"
    if feature in ("loan_to_income", "debt_to_annual_income"):
        return f"{v:.2f}x annual income"
    if feature == "emi_to_income":
        return f"{v * 100:.1f}% of monthly income"
    if feature in ("credit_history_years", "employment_years"):
        return f"{v:.1f} years"
    if feature == "loan_tenure_months":
        return f"{v:.0f} months"
    if feature in ("residential_status", "loan_purpose"):
        return str(v).replace("_", " ")
    if isinstance(v, (int, float, np.floating)):
        return f"{v:.0f}"
    return str(v)


def describe_contributions(contrib: pd.Series, features: pd.Series, raw: dict | None = None, top: int = 5):
    """Return (increasing, reducing) lists of dicts with business-language text."""
    raw = raw or {}
    items = []
    for f, c in contrib.items():
        if abs(c) < 1e-4:
            continue
        val = features.get(f)
        label = FEATURE_LABELS.get(f, f)
        vtxt = _fmt_value(f, val, raw)
        if c > 0:
            text = f"{label} ({vtxt}) pushes the estimated default risk up."
        else:
            text = f"{label} ({vtxt}) pulls the estimated default risk down."
        items.append(dict(feature=f, label=label, value=vtxt, contribution=float(c), text=text))
    inc = sorted([i for i in items if i["contribution"] > 0], key=lambda d: -d["contribution"])[:top]
    red = sorted([i for i in items if i["contribution"] < 0], key=lambda d: d["contribution"])[:top]
    return inc, red


SUGGESTIONS = {
    "credit_score": "Improve the credit score: pay every bill on time and avoid new hard enquiries for 6-12 months.",
    "credit_utilization": "Bring credit-card utilisation below 30% by paying down balances or not maxing out limits.",
    "dti_ratio": "Reduce existing EMIs (close or prepay small loans) to lower the debt-to-income ratio.",
    "emi_to_income": "Choose a longer tenure or a smaller amount so the new EMI is a lower share of income.",
    "loan_to_income": "Request a smaller loan amount relative to income, or add a co-applicant's income.",
    "interest_rate": "A higher rate usually reflects a riskier profile; a secured loan or better score can lower it.",
    "delayed_payments_2y": "Maintain a clean repayment record - recent late payments weigh heavily.",
    "recent_inquiries": "Avoid applying with many lenders at once; multiple recent enquiries signal credit hunger.",
    "previous_defaults": "Settle/clear any defaulted or written-off accounts and obtain closure letters.",
    "bankruptcies": "Rebuild credit with a small secured card and consistent on-time payments.",
    "debt_to_annual_income": "Reduce outstanding revolving (card / overdraft) balances.",
    "credit_history_years": "A longer credit history helps - keep old accounts open and in good standing.",
    "employment_years": "Longer continuous employment improves stability; consider applying after more tenure.",
    "loan_tenure_months": "Longer tenures carry more uncertainty; a shorter tenure (if affordable) lowers risk.",
    "active_loans": "Consolidate or close some active credit lines.",
}


def suggestions_for(increasing: list[dict], n: int = 4) -> list[str]:
    out = []
    for item in increasing:
        s = SUGGESTIONS.get(item["feature"])
        if s and s not in out:
            out.append(s)
        if len(out) >= n:
            break
    if not out:
        out.append("Maintain the current repayment discipline and keep debt levels stable.")
    return out


def financial_indicators(row: pd.Series) -> tuple[list[str], list[str]]:
    """Rule-based positive / negative indicators from raw applicant data (independent of the model)."""
    pos, neg = [], []
    cs = row.get("credit_score")
    if pd.notna(cs):
        if cs >= 750:
            pos.append(f"Strong credit score ({cs:.0f}).")
        elif cs >= 700:
            pos.append(f"Good credit score ({cs:.0f}).")
        elif cs < 650:
            neg.append(f"Weak credit score ({cs:.0f}).")
    dti = row.get("dti_ratio")
    if pd.notna(dti):
        if dti <= 20:
            pos.append(f"Manageable existing debt-to-income ratio ({dti:.1f}%).")
        elif dti > 35:
            neg.append(f"High existing debt-to-income ratio ({dti:.1f}%).")
    eti = row.get("emi_to_income_total")
    if pd.notna(eti):
        if eti > 0.5:
            neg.append(f"Total EMIs (existing + new) would take {eti:.0%} of monthly income.")
        elif eti < 0.3:
            pos.append(f"Total EMIs (existing + new) would be a comfortable {eti:.0%} of income.")
    if row.get("delayed_payments_2y", 0) and row.get("delayed_payments_2y") > 0:
        neg.append(f"{row['delayed_payments_2y']:.0f} delayed payment(s) in the last 2 years.")
    elif pd.notna(row.get("delayed_payments_2y")):
        pos.append("No delayed payments in the last 2 years.")
    if pd.notna(row.get("previous_defaults")) and row["previous_defaults"] > 0:
        neg.append(f"{row['previous_defaults']:.0f} previous default / derogatory record(s).")
    cu = row.get("credit_utilization")
    if pd.notna(cu):
        if cu > 75:
            neg.append(f"High credit utilisation ({cu:.0f}%).")
        elif cu < 30:
            pos.append(f"Low credit utilisation ({cu:.0f}%).")
    ey = row.get("employment_years")
    if pd.notna(ey):
        if ey >= 5:
            pos.append(f"Stable employment ({ey:.0f}+ years).")
        elif ey < 1:
            neg.append("Less than 1 year in current employment.")
    ch = row.get("credit_history_years")
    if pd.notna(ch):
        if ch >= 10:
            pos.append(f"Long credit history ({ch:.0f} years).")
        elif ch < 3:
            neg.append(f"Short credit history ({ch:.1f} years).")
    if pd.notna(row.get("existing_emi")) and row["existing_emi"] > 0:
        neg.append("Has existing loan obligations.")
    di = row.get("disposable_income")
    if pd.notna(di):
        if di < 0:
            neg.append("Expenses and EMIs exceed income (negative disposable income).")
        elif row.get("total_monthly_income", 0) and di / row["total_monthly_income"] > 0.3:
            pos.append("Healthy disposable income after expenses and EMIs.")
    sav = row.get("savings")
    if pd.notna(sav) and pd.notna(row.get("expected_emi")) and row["expected_emi"] > 0 and sav >= 6 * row["expected_emi"]:
        pos.append("Savings cover at least 6 months of the new EMI.")
    return pos, neg
