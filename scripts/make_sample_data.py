"""Generate SYNTHETIC test applicants for trying the app (seeded, reproducible).

These records are invented: they are not real people, they are not drawn from the training
data, they have no outcome labels and they are never used to train or evaluate the model.
A few rows deliberately contain missing / invalid values to demonstrate validation.

Run:  python scripts/make_sample_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from loanrisk.config import SAMPLE_DIR  # noqa: E402
from loanrisk.features import amortized_emi  # noqa: E402

rng = np.random.default_rng(2026)
N = 250

emp_type = rng.choice(["Salaried", "Self-employed", "Business"], N, p=[0.62, 0.22, 0.16])
emp_cat = np.where(emp_type == "Salaried",
                   rng.choice(["IT / Software", "Government", "Banking & Finance", "Manufacturing", "Healthcare",
                               "Education", "Retail"], N),
                   rng.choice(["Retail trade", "Professional services", "Transport", "Agriculture", "Hospitality",
                               "Construction"], N))
age = rng.integers(22, 62, N)
income = np.round(np.exp(rng.normal(np.log(55000), 0.55, N)), -2)
income = np.clip(income, 12000, 600000)
add_income = np.where(rng.random(N) < 0.3, np.round(income * rng.uniform(0.05, 0.3, N), -2), 0)
quality = rng.normal(0, 1, N)  # latent credit quality driving correlated attributes
credit = np.clip(np.round(720 + 55 * quality + rng.normal(0, 25, N)), 480, 890)
existing_emi = np.round(np.where(rng.random(N) < 0.6, income * np.clip(rng.normal(0.15 - 0.04 * quality, 0.08), 0, 0.6), 0), -2)
expenses = np.round(income * np.clip(rng.normal(0.42, 0.12, N), 0.15, 0.9), -2)
tenure = rng.choice([24, 36, 48, 60], N, p=[0.1, 0.5, 0.15, 0.25])
rate = np.round(np.clip(13.5 - 2.8 * quality + rng.normal(0, 1.5, N), 8.5, 26), 2)
# Mostly 1-5x monthly income; ~8% deliberately large (beyond the training range) to show extrapolation flags.
amount = np.round(income * np.where(rng.random(N) < 0.08, rng.uniform(8, 14, N), rng.uniform(1, 5, N)), -4)
emi = np.round(amortized_emi(amount, rate, tenure))
util = np.round(np.clip(45 - 18 * quality + rng.normal(0, 15, N), 0, 110), 1)
delayed = rng.poisson(np.clip(0.4 - 0.35 * quality, 0.02, 3))
defaults = rng.poisson(np.clip(0.12 - 0.12 * quality, 0.01, 2))
hist = np.round(np.clip(age - 21 - rng.uniform(0, 8, N), 0.5, 40), 1)

df = pd.DataFrame({
    "Applicant ID": [f"LA{1001 + i}" for i in range(N)],
    "Age": age,
    "Marital Status": rng.choice(["Single", "Married", "Divorced", "Widowed"], N, p=[0.35, 0.58, 0.05, 0.02]),
    "Dependents": rng.integers(0, 5, N),
    "Residential Status": rng.choice(["Owned", "Rented", "Mortgaged", "Other"], N, p=[0.35, 0.38, 0.2, 0.07]),
    "Employment Type": emp_type,
    "Employer Category": emp_cat,
    "Years of Employment": np.round(np.clip(rng.gamma(2.2, 3.2, N), 0, age - 20), 1),
    "Monthly Income": income,
    "Additional Income": add_income,
    "Employment Stability": rng.choice(["Stable", "Moderate", "Unstable"], N, p=[0.6, 0.3, 0.1]),
    "Monthly Expenses": expenses,
    "Existing EMI": existing_emi,
    "Number of Active Loans": np.where(existing_emi > 0, rng.integers(1, 6, N), 0) + rng.integers(1, 6, N),
    "Bank Balance": np.round(income * rng.uniform(0.2, 6, N), -2),
    "Savings": np.round(income * np.clip(rng.normal(3 + quality, 2.5, N), 0, 30), -2),
    "Credit Score": credit,
    "Number of Previous Loans": rng.integers(2, 35, N),
    "Previous Defaults": defaults,
    "Delayed Payments (2 yrs)": delayed,
    "Credit Inquiries (6 months)": rng.poisson(np.clip(1.0 - 0.5 * quality, 0.1, 5)),
    "Credit Utilization (%)": util,
    "Outstanding Debt": np.round(income * np.clip(rng.normal(2.5 - 0.8 * quality, 1.5, N), 0, 12), -2),
    "Credit History (years)": hist,
    "Loan Amount": amount,
    "Loan Tenure (months)": tenure,
    "Interest Rate (%)": rate,
    "Loan Purpose": rng.choice(["debt_consolidation", "home_improvement", "car", "education", "medical",
                                "small_business", "major_purchase", "wedding", "credit_card", "other"], N),
    "Expected EMI": emi,
})
# Deliberate data-quality problems for the validation demo
df.loc[5, "Credit Score"] = np.nan                  # missing critical -> cannot assess
df.loc[17, "Monthly Income"] = np.nan               # missing critical -> cannot assess
df.loc[23, "Interest Rate (%)"] = 95                # out of range -> treated as missing -> cannot assess
df.loc[31, "Credit Utilization (%)"] = np.nan       # optional missing -> imputed
df.loc[31, "Outstanding Debt"] = np.nan
df.loc[44, "Expected EMI"] = np.nan                 # derived from amount / rate / tenure
df.loc[60, "Applicant ID"] = "LA1002"               # duplicate ID

SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(SAMPLE_DIR / "sample_applicants.csv", index=False)
df.to_excel(SAMPLE_DIR / "sample_applicants.xlsx", index=False)
df.head(10).to_csv(SAMPLE_DIR / "sample_applicants_small.csv", index=False)
# Same applicants with non-standard column names to demonstrate column mapping
alt = df.head(40).rename(columns={"Applicant ID": "Cust_No", "Monthly Income": "Net Salary",
                                  "Credit Score": "CIBIL", "Loan Amount": "Amt Requested",
                                  "Loan Tenure (months)": "Tenor", "Interest Rate (%)": "ROI"})
alt.to_csv(SAMPLE_DIR / "sample_applicants_custom_columns.csv", index=False)
print(f"wrote {len(df)} synthetic applicants to {SAMPLE_DIR}")
