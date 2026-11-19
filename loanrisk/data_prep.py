"""Convert the public LendingClub 'accepted loans 2007-2018Q4' file into the canonical
applicant schema with a binary default label.

Leakage controls
----------------
* Only information known at application time is kept.  All post-origination columns
  (payments, recoveries, last_pymnt_d, out_prncp, hardship/settlement fields, ...) are dropped.
* LendingClub's own risk grade / sub-grade is NOT used (it is the output of their model,
  not an applicant attribute).
* Geography (state / zip) is NOT used as a model feature (proxy for protected
  characteristics); state is kept in a separate column for the fairness audit only.
* Only loans whose outcome is final are kept: Fully Paid (0) vs Charged Off / Default (1).
  To avoid outcome-maturity bias we keep 36-month loans issued 2007-2015 and 60-month
  loans issued 2010-2013, all of which had reached maturity by the 2018Q4 snapshot.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR, RANDOM_STATE, RAW_DIR

RAW_FILE = RAW_DIR / "accepted_2007_to_2018Q4.csv.gz"
USECOLS = [
    "id", "loan_amnt", "term", "int_rate", "installment", "emp_length", "home_ownership",
    "annual_inc", "issue_d", "loan_status", "purpose", "dti", "delinq_2yrs", "earliest_cr_line",
    "fico_range_low", "fico_range_high", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal",
    "revol_util", "total_acc", "mort_acc", "pub_rec_bankruptcies", "application_type", "addr_state",
]
STATUS_MAP = {
    "Fully Paid": 0, "Charged Off": 1, "Default": 1,
    "Does not meet the credit policy. Status:Fully Paid": 0,
    "Does not meet the credit policy. Status:Charged Off": 1,
}
HOME_MAP = {"OWN": "Owned", "MORTGAGE": "Mortgaged", "RENT": "Rented"}
REGION = {  # US Census regions, used ONLY for the fairness audit
    **{s: "Northeast" for s in "CT ME MA NH RI VT NJ NY PA".split()},
    **{s: "Midwest" for s in "IL IN MI OH WI IA KS MN MO NE ND SD".split()},
    **{s: "South" for s in "DE DC FL GA MD NC SC VA WV AL KY MS TN AR LA OK TX".split()},
    **{s: "West" for s in "AZ CO ID MT NV NM UT WY AK CA HI OR WA".split()},
}


def _emp_years(s: pd.Series) -> pd.Series:
    s = s.astype("string")
    out = pd.to_numeric(s.str.extract(r"(\d+)")[0], errors="coerce").astype(float)
    out = out.where(~s.str.contains("<", na=False), 0.5)
    return out


def prepare(sample_size: int | None = 150_000) -> pd.DataFrame:
    raw = pd.read_csv(RAW_FILE, usecols=USECOLS, low_memory=False)
    n_raw = len(raw)
    raw = raw[raw["loan_status"].isin(STATUS_MAP)]
    raw = raw[raw["application_type"].fillna("Individual") == "Individual"]
    issue = pd.to_datetime(raw["issue_d"], format="%b-%Y", errors="coerce")
    term = pd.to_numeric(raw["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    keep = ((term == 36) & issue.dt.year.between(2007, 2015)) | ((term == 60) & issue.dt.year.between(2010, 2013))
    raw, issue, term = raw[keep], issue[keep], term[keep]

    first_cr = pd.to_datetime(raw["earliest_cr_line"], format="%b-%Y", errors="coerce")
    df = pd.DataFrame({
        "applicant_id": "LC" + raw["id"].astype(str),
        "residential_status": raw["home_ownership"].map(HOME_MAP).fillna("Other"),
        "employment_years": _emp_years(raw["emp_length"]),
        "monthly_income": raw["annual_inc"] / 12.0,
        "additional_income": 0.0,
        "dti_ratio": raw["dti"],
        "active_loans": raw["open_acc"],
        "credit_score": (raw["fico_range_low"] + raw["fico_range_high"]) / 2.0,
        "previous_loans": raw["total_acc"],
        "previous_defaults": raw["pub_rec"],
        "delayed_payments_2y": raw["delinq_2yrs"],
        "recent_inquiries": raw["inq_last_6mths"],
        "credit_utilization": pd.to_numeric(raw["revol_util"].astype(str).str.rstrip("%"), errors="coerce"),
        "outstanding_debt": raw["revol_bal"],
        "credit_history_years": ((issue - first_cr).dt.days / 365.25),
        "bankruptcies": raw["pub_rec_bankruptcies"],
        "mortgage_accounts": raw["mort_acc"],
        "loan_amount": raw["loan_amnt"],
        "loan_tenure_months": term,
        "interest_rate": pd.to_numeric(raw["int_rate"].astype(str).str.rstrip("%"), errors="coerce"),
        "loan_purpose": raw["purpose"].replace({"educational": "education"}),
        "expected_emi": raw["installment"],
        "issue_year": issue.dt.year,
        "audit_region": raw["addr_state"].map(REGION).fillna("Other"),
        "default": raw["loan_status"].map(STATUS_MAP).astype(int),
    }).reset_index(drop=True)
    df = df[(df["monthly_income"] > 0) & df["credit_score"].notna()]
    # LendingClub DTI of 999 / negative are data errors
    df.loc[(df["dti_ratio"] < 0) | (df["dti_ratio"] > 100), "dti_ratio"] = np.nan
    n_eligible = len(df)
    if sample_size and len(df) > sample_size:
        # Stratified sample keeps the default rate unchanged.
        from sklearn.model_selection import train_test_split
        df, _ = train_test_split(df, train_size=sample_size, stratify=df["default"],
                                 random_state=RANDOM_STATE)
        df = df.reset_index(drop=True)
    df.attrs.update(n_raw=n_raw, n_eligible=n_eligible)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=150_000)
    args = ap.parse_args()
    df = prepare(args.sample)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "lendingclub_canonical.csv.gz"
    df.to_csv(out, index=False)
    print(f"raw rows={df.attrs['n_raw']:,}  eligible completed loans={df.attrs['n_eligible']:,}  "
          f"sampled={len(df):,}  default rate={df['default'].mean():.3%}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
