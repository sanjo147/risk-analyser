"""Column mapping, validation and feature engineering shared by training and inference."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .schema import (
    CATEGORICAL_FEATURES, CRITICAL_FIELDS, FIELD_MAP, FIELDS, LOAN_PURPOSES, MODEL_FEATURES,
    NUMERIC_FEATURES, PROHIBITED_FEATURES, RESIDENTIAL,
)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


# ----------------------------------------------------------------------------- mapping
def suggest_mapping(columns: list[str]) -> dict[str, str | None]:
    """Suggest uploaded-column -> canonical-field mapping using names, aliases and fuzzy matching."""
    import difflib

    lookup: dict[str, str] = {}
    for f in FIELDS:
        for key in (f.name, f.label, *f.aliases):
            lookup.setdefault(_norm(key), f.name)
            lookup.setdefault(_norm(re.sub(r"\(.*?\)", "", key)), f.name)
    mapping: dict[str, str | None] = {}
    used: set[str] = set()
    for col in columns:
        candidates = [_norm(col), _norm(re.sub(r"\(.*?\)", "", str(col)))]
        target = next((lookup[c] for c in candidates if c in lookup and lookup[c] not in used), None)
        if target is None:
            close = difflib.get_close_matches(candidates[1], list(lookup), n=3, cutoff=0.82)
            target = next((lookup[c] for c in close if lookup[c] not in used), None)
        mapping[col] = target
        if target:
            used.add(target)
    return mapping


def apply_mapping(df: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    cols = {src: dst for src, dst in mapping.items() if dst}
    out = df[list(cols)].rename(columns=cols).copy()
    return out


def prohibited_columns(columns) -> list[str]:
    return [c for c in columns if _norm(c) in {_norm(p) for p in PROHIBITED_FEATURES}]


# ----------------------------------------------------------------------------- cleaning
_PURPOSE_ALIASES = {
    "debt": "debt_consolidation", "debtconsolidation": "debt_consolidation", "consolidation": "debt_consolidation",
    "creditcard": "credit_card", "card": "credit_card",
    "home": "home_improvement", "homeimprovement": "home_improvement", "renovation": "home_improvement",
    "homerenovation": "home_improvement", "homerepair": "home_improvement", "carloan": "car", "personal": "other",
    "majorpurchase": "major_purchase", "purchase": "major_purchase", "consumerdurable": "major_purchase",
    "business": "small_business", "smallbusiness": "small_business",
    "car": "car", "vehicle": "car", "auto": "car", "twowheeler": "car",
    "medical": "medical", "health": "medical",
    "education": "education", "studies": "education",
    "house": "house", "housing": "house", "homepurchase": "house",
    "wedding": "wedding", "marriage": "wedding",
    "vacation": "vacation", "travel": "vacation",
    "moving": "moving", "relocation": "moving",
    "renewableenergy": "renewable_energy", "solar": "renewable_energy",
}
_RES_ALIASES = {
    "own": "Owned", "owned": "Owned", "self": "Owned",
    "mortgage": "Mortgaged", "mortgaged": "Mortgaged", "homeloan": "Mortgaged",
    "rent": "Rented", "rented": "Rented", "renting": "Rented", "tenant": "Rented",
}


def _clean_purpose(v):
    if pd.isna(v) or str(v).strip() == "":
        return np.nan
    k = _norm(v)
    if k in {_norm(p) for p in LOAN_PURPOSES}:
        return next(p for p in LOAN_PURPOSES if _norm(p) == k)
    return _PURPOSE_ALIASES.get(k, "other")


def _clean_residential(v):
    if pd.isna(v) or str(v).strip() == "":
        return np.nan
    k = _norm(v)
    if k in {_norm(r) for r in RESIDENTIAL}:
        return next(r for r in RESIDENTIAL if _norm(r) == k)
    return _RES_ALIASES.get(k, "Other")


def _to_number(s: pd.Series, tenure: bool = False) -> pd.Series:
    if s.dtype.kind in "biuf":
        return s.astype(float)
    txt = s.astype("string").str.strip().str.lower()
    txt = txt.str.replace(r"[,₹$€£%\s]", "", regex=True)
    num = pd.to_numeric(txt.str.extract(r"(-?\d+(?:\.\d+)?)")[0], errors="coerce")
    if tenure:  # "3 years" -> 36 months
        is_years = txt.str.contains(r"^[\d.]+(?:y|yr|yrs|year|years)$", regex=True, na=False)
        num = num.where(~is_years, num * 12)
    return num.astype(float)


def amortized_emi(principal, annual_rate_pct, months):
    principal = np.asarray(principal, dtype=float)
    r = np.asarray(annual_rate_pct, dtype=float) / 1200.0
    n = np.asarray(months, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        emi = np.where(r > 0, principal * r * (1 + r) ** n / ((1 + r) ** n - 1), principal / n)
    return emi


def clean_applicants(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Coerce types, normalise categories, derive missing EMI/DTI and validate.

    Returns (clean_df, issues_df).  issues_df has columns
    [row, applicant_id, severity, field, message]; severity "error" blocks prediction.
    """
    df = df.copy()
    issues: list[dict] = []
    for f in FIELDS:
        if f.name not in df.columns:
            df[f.name] = np.nan
    df = df[[f.name for f in FIELDS] + [c for c in df.columns if c not in FIELD_MAP]]

    df["applicant_id"] = df["applicant_id"].astype("string").str.strip()
    missing_id = df["applicant_id"].isna() | (df["applicant_id"] == "")
    df.loc[missing_id, "applicant_id"] = [f"ROW{i + 1:05d}" for i in np.where(missing_id)[0]]

    for f in FIELDS:
        if f.kind in ("num", "int"):
            df[f.name] = _to_number(df[f.name], tenure=f.name == "loan_tenure_months")
    df["loan_purpose"] = df["loan_purpose"].map(_clean_purpose)
    df["residential_status"] = df["residential_status"].map(_clean_residential)

    # Range checks: out-of-range values are set to missing and reported.
    for f in FIELDS:
        if f.kind not in ("num", "int"):
            continue
        col = df[f.name]
        lo = f.min if f.min is not None else -np.inf
        hi = f.max if f.max is not None else np.inf
        bad = col.notna() & ((col < lo) | (col > hi))
        for i in np.where(bad)[0]:
            issues.append(dict(row=i + 1, applicant_id=df["applicant_id"].iat[i], severity="warning",
                               field=f.label, message=f"Value {col.iat[i]:g} outside allowed range "
                               f"[{f.min}, {f.max}]; treated as missing."))
        df.loc[bad, f.name] = np.nan

    total_income = df["monthly_income"] + df["additional_income"].fillna(0)
    df["total_monthly_income"] = total_income

    # Derive EMI if missing.
    need_emi = df["expected_emi"].isna()
    df.loc[need_emi, "expected_emi"] = amortized_emi(
        df.loc[need_emi, "loan_amount"], df.loc[need_emi, "interest_rate"], df.loc[need_emi, "loan_tenure_months"])
    # Derive DTI if missing (existing obligations only, like the training data).
    need_dti = df["dti_ratio"].isna() & df["existing_emi"].notna() & (total_income > 0)
    df.loc[need_dti, "dti_ratio"] = (df.loc[need_dti, "existing_emi"] / total_income[need_dti] * 100).clip(upper=100)
    df["dti_derived"] = need_dti

    # Duplicate IDs
    dup = df["applicant_id"].duplicated(keep="first")
    for i in np.where(dup)[0]:
        issues.append(dict(row=i + 1, applicant_id=df["applicant_id"].iat[i], severity="warning",
                           field="Applicant ID", message="Duplicate Applicant ID; a suffix was added."))
    df.loc[dup, "applicant_id"] = df.loc[dup, "applicant_id"] + "-" + (np.where(dup)[0] + 1).astype(str)

    # Critical-field check.
    for i in range(len(df)):
        missing = [FIELD_MAP[c].label for c in CRITICAL_FIELDS if c != "applicant_id" and pd.isna(df[c].iat[i])]
        if pd.notna(df["monthly_income"].iat[i]) and df["monthly_income"].iat[i] <= 0:
            missing.append("Monthly income (must be > 0)")
        if missing:
            issues.append(dict(row=i + 1, applicant_id=df["applicant_id"].iat[i], severity="error",
                               field=", ".join(missing),
                               message="Prediction cannot be generated - missing critical information: "
                                       + ", ".join(missing)))
        optional_missing = [FIELD_MAP[c].label for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES
                            if c in FIELD_MAP and c not in CRITICAL_FIELDS and pd.isna(df[c].iat[i])]
        if optional_missing and not missing:
            issues.append(dict(row=i + 1, applicant_id=df["applicant_id"].iat[i], severity="info",
                               field=", ".join(optional_missing),
                               message="Not provided - filled with the training-data typical value "
                                       "(reduces prediction confidence)."))
    issues_df = pd.DataFrame(issues, columns=["row", "applicant_id", "severity", "field", "message"])
    err_ids = set(issues_df.loc[issues_df.severity == "error", "applicant_id"])
    df["can_predict"] = ~df["applicant_id"].isin(err_ids)
    n_missing_optional = df[[c for c in MODEL_FEATURES if c in df.columns and c not in CRITICAL_FIELDS]].isna().sum(axis=1)
    df["n_imputed_features"] = n_missing_optional
    return df.reset_index(drop=True), issues_df


# ----------------------------------------------------------------------------- features
def build_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical applicant fields -> model feature matrix (currency-independent ratios)."""
    income = df["monthly_income"].astype(float) + df["additional_income"].fillna(0).astype(float)
    income = income.where(income > 0)
    emi = df["expected_emi"].astype(float)
    X = pd.DataFrame(index=df.index)
    X["credit_score"] = df["credit_score"]
    X["interest_rate"] = df["interest_rate"]
    X["loan_tenure_months"] = df["loan_tenure_months"]
    X["dti_ratio"] = df["dti_ratio"]
    X["loan_to_income"] = df["loan_amount"] / (income * 12)
    X["emi_to_income"] = emi / income
    X["debt_to_annual_income"] = df["outstanding_debt"] / (income * 12)
    for c in ["credit_utilization", "credit_history_years", "employment_years", "delayed_payments_2y",
              "recent_inquiries", "active_loans", "previous_loans", "previous_defaults", "bankruptcies",
              "mortgage_accounts"]:
        X[c] = df[c]
    X["residential_status"] = df["residential_status"].astype(object)
    X["loan_purpose"] = df["loan_purpose"].astype(object)
    X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].astype(float).replace([np.inf, -np.inf], np.nan)
    return X[MODEL_FEATURES]
