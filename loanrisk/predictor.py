"""Assessment engine: validated applicant data -> probability -> risk band -> policy -> explanation.

Two layers are kept strictly apart:
* MODEL PREDICTION  - calibrated probability of default from the trained model, and the
                      risk band it falls into (configurable thresholds).
* LENDING POLICY    - an illustrative decision policy: each band maps to a default decision,
                      then rule-based overlays (affordability, minimum score, eligibility,
                      uncertainty) may *escalate* that decision.  Overlays never relax it.
"""
from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from .config import MODEL_DIR, load_settings
from .explain import ModelExplainer, describe_contributions, financial_indicators, suggestions_for
from .features import build_model_features, clean_applicants
from .risk import categorize
from .schema import FEATURE_LABELS

DECISION_RANK = {"APPROVE": 0, "MANUAL REVIEW": 1, "REJECT RECOMMENDATION": 2}
# Out-of-range checks: (feature, direction that matters). Values beyond the 0.1-99.9 percentile range of the
# training data mean the model is extrapolating. "high"/"low" = only the risky side is flagged.
OOD_RULES = {
    "credit_score": "low", "interest_rate": "high", "dti_ratio": "high", "loan_to_income": "high",
    "emi_to_income": "high", "debt_to_annual_income": "high", "credit_utilization": "high",
    "delayed_payments_2y": "high", "recent_inquiries": "high", "previous_defaults": "high",
    "bankruptcies": "high", "credit_history_years": "low", "loan_tenure_months": "both",
}
CRITICAL_OOD = {"credit_score", "loan_to_income", "emi_to_income", "dti_ratio", "interest_rate"}


class ModelNotTrainedError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load_bundle(path: str | None = None) -> dict:
    p = MODEL_DIR / "model.joblib" if path is None else path
    try:
        return joblib.load(p)
    except FileNotFoundError as e:
        raise ModelNotTrainedError("No trained model found. Run `python -m loanrisk.train` first.") from e


@lru_cache(maxsize=1)
def get_explainer() -> ModelExplainer:
    b = load_bundle()
    return ModelExplainer(b["pipeline"], b["background"])


def _escalate(current: str, new: str) -> str:
    return new if DECISION_RANK[new] > DECISION_RANK[current] else current


def _calibration_bin(prob: float, bins: list[dict]) -> dict:
    for b in bins:
        if prob <= b["hi"]:
            return b
    return bins[-1]


def assess(raw: pd.DataFrame, settings: dict | None = None, explain: bool = True):
    """Assess applicants given in the canonical schema (after column mapping).

    Returns (results, issues, details) where details[applicant_id] holds the full explanation.
    """
    settings = settings or load_settings()
    bundle = load_bundle()
    stats = bundle["stats"]
    clean, issues = clean_applicants(raw)
    ok = clean["can_predict"].values
    results = clean.copy()
    for c in ["default_probability", "risk_category", "model_band_decision", "recommendation",
              "policy_notes", "confidence", "observed_rate_low", "observed_rate_high", "key_risk_factors",
              "ood_features", "emi_to_income_total", "disposable_income"]:
        results[c] = np.nan
    results = results.astype({c: object for c in ["risk_category", "model_band_decision", "recommendation",
                                                    "policy_notes", "confidence", "key_risk_factors", "ood_features"]})
    details: dict[str, dict] = {}
    if not ok.any():
        results.loc[~ok, "recommendation"] = "CANNOT ASSESS"
        return results, issues, details

    sub = clean.loc[ok]
    X = build_model_features(sub)
    prob = bundle["calibrated"].predict_proba(X)[:, 1]
    cats, band_dec = categorize(prob, settings["risk_bands"])

    income = sub["total_monthly_income"]
    total_emi = sub["existing_emi"].fillna(0) + sub["expected_emi"]
    emi_tot = total_emi / income
    disposable = income - sub["monthly_expenses"] - total_emi  # NaN when expenses unknown

    contrib = get_explainer().contributions(X) if explain else None
    pol = settings["policy"]
    rows_out = []
    for j, (i, row) in enumerate(sub.iterrows()):
        p = float(prob[j])
        decision = band_dec[j]
        notes = []
        # --- uncertainty / out-of-distribution
        ood = []
        lo_s, hi_s = stats.get("lo", stats["p01"]), stats.get("hi", stats["p99"])
        for f, side in OOD_RULES.items():
            v = X.iloc[j][f]
            if pd.isna(v):
                continue
            if (side in ("low", "both") and v < lo_s[f]) or (side in ("high", "both") and v > hi_s[f]):
                ood.append(f)
        b = _calibration_bin(p, bundle["calibration_bins"])
        width = b["ci_high"] - b["ci_low"]
        n_imp = int(row["n_imputed_features"])
        if (set(ood) & CRITICAL_OOD) or n_imp >= 5:
            conf = "Low"
        elif ood or n_imp >= 3 or width > 0.10:
            conf = "Medium"
        else:
            conf = "High"
        # --- policy overlays (escalate only)
        if pol.get("enabled", True):
            cs = row["credit_score"]
            if cs < pol["min_credit_score"]:
                decision = _escalate(decision, "REJECT RECOMMENDATION")
                notes.append(f"Policy: credit score {cs:.0f} is below the minimum {pol['min_credit_score']}.")
            if emi_tot.iloc[j] > pol["max_total_emi_to_income"]:
                decision = _escalate(decision, "MANUAL REVIEW")
                notes.append(f"Policy: total EMIs would be {emi_tot.iloc[j]:.0%} of income "
                             f"(limit {pol['max_total_emi_to_income']:.0%}).")
            if pd.notna(disposable.iloc[j]) and disposable.iloc[j] < pol["min_disposable_income"]:
                decision = _escalate(decision, "MANUAL REVIEW")
                notes.append("Policy: income does not cover expenses plus all EMIs.")
            age = row.get("age")
            if pd.notna(age):
                if age < pol["min_age"]:
                    decision = _escalate(decision, "MANUAL REVIEW")
                    notes.append(f"Eligibility: applicant younger than {pol['min_age']}.")
                elif age + row["loan_tenure_months"] / 12 > pol["max_age_at_maturity"]:
                    decision = _escalate(decision, "MANUAL REVIEW")
                    notes.append(f"Eligibility: age at loan maturity exceeds {pol['max_age_at_maturity']}.")
            if decision == "APPROVE" and conf == "Low":
                decision = "MANUAL REVIEW"
                notes.append("Low prediction confidence (data outside the training range or many missing "
                             "inputs) - human review required.")
            if decision == "APPROVE" and width > pol["uncertainty_review_width"]:
                decision = "MANUAL REVIEW"
                notes.append("Wide uncertainty interval - human review required.")
        # --- explanation
        inc, red = ([], [])
        if contrib is not None:
            inc, red = describe_contributions(contrib.iloc[j], X.iloc[j])
        full = row.copy()
        full["emi_to_income_total"] = emi_tot.iloc[j]
        full["disposable_income"] = disposable.iloc[j]
        pos, neg = financial_indicators(full)
        key = "; ".join(d["label"] for d in inc[:3]) if inc else ""
        rows_out.append((i, p, cats[j], band_dec[j], decision, " ".join(notes), conf, b["ci_low"], b["ci_high"],
                         key, ", ".join(FEATURE_LABELS[f] for f in ood), emi_tot.iloc[j], disposable.iloc[j]))
        details[row["applicant_id"]] = dict(
            increasing=inc, reducing=red, positive=pos, negative=neg, suggestions=suggestions_for(inc),
            contributions=(contrib.iloc[j].to_dict() if contrib is not None else {}),
            model_features=X.iloc[j].to_dict(), calibration_bin=b, imputed=n_imp, ood=ood,
            reason=_reason(cats[j], decision, p, inc, red, notes),
        )
    cols = ["default_probability", "risk_category", "model_band_decision", "recommendation", "policy_notes",
            "confidence", "observed_rate_low", "observed_rate_high", "key_risk_factors", "ood_features",
            "emi_to_income_total", "disposable_income"]
    upd = pd.DataFrame([r[1:] for r in rows_out], index=[r[0] for r in rows_out], columns=cols)
    for c in cols:
        results.loc[upd.index, c] = upd[c]
    results.loc[~ok, "recommendation"] = "CANNOT ASSESS"
    err = issues[issues.severity == "error"].set_index("applicant_id")["message"]
    results.loc[~ok, "policy_notes"] = results.loc[~ok, "applicant_id"].map(err)
    results["default_probability"] = results["default_probability"].astype(float)
    return results, issues, details


def _reason(cat, decision, p, inc, red, notes) -> str:
    drivers_up = ", ".join(d["label"].lower() for d in inc[:2])
    drivers_down = ", ".join(d["label"].lower() for d in red[:2])
    s = f"The model estimates a {p:.1%} probability of default, which falls in the '{cat}' band. "
    if drivers_down:
        s += f"The profile is supported mainly by {drivers_down}. "
    if drivers_up:
        s += f"The main factors raising the estimate are {drivers_up}. "
    if notes:
        s += "Policy checks changed or qualified the outcome: " + " ".join(notes) + " "
    s += f"Recommended action under the illustrative policy: {decision}."
    return s
