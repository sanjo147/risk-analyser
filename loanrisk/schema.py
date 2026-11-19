"""Canonical applicant schema, model feature definitions and column aliases.

Every applicant (training record or new applicant) is first expressed in the
*canonical applicant schema* below.  `features.build_model_features` then turns
canonical fields into the exact model feature matrix.  Because training data and
new applicants go through the same function, there is no train/serve skew.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------------------
# Canonical applicant fields (what the UI / upload accepts)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Field:
    name: str
    label: str
    group: str
    kind: str  # "num", "int", "cat", "text"
    unit: str = ""
    choices: tuple = ()
    used_by_model: bool = False
    critical: bool = False  # prediction refused if missing
    min: float | None = None
    max: float | None = None
    help: str = ""
    aliases: tuple = field(default_factory=tuple)


RESIDENTIAL = ("Owned", "Mortgaged", "Rented", "Other")
EMPLOYMENT_TYPES = ("Salaried", "Self-employed", "Business")
MARITAL = ("Single", "Married", "Divorced", "Widowed", "Other")
LOAN_PURPOSES = (
    "debt_consolidation", "credit_card", "home_improvement", "major_purchase",
    "small_business", "car", "medical", "education", "house", "wedding",
    "vacation", "moving", "renewable_energy", "other",
)

FIELDS: list[Field] = [
    # A. Personal
    Field("applicant_id", "Applicant ID", "Personal", "text", critical=True,
          aliases=("id", "applicant", "app_id", "customer_id", "loan_id", "cust_no", "customer_no", "cust_id", "application_no")),
    Field("age", "Age", "Personal", "int", "years", min=18, max=100,
          help="Recorded for eligibility checks only - NOT used by the model (protected attribute).",
          aliases=("applicant_age",)),
    Field("marital_status", "Marital status", "Personal", "cat", choices=MARITAL,
          help="Recorded only - NOT used by the model (protected attribute).",
          aliases=("marital",)),
    Field("dependents", "Number of dependents", "Personal", "int", min=0, max=20,
          help="Used only in the affordability check, not the model.",
          aliases=("num_dependents", "no_of_dependents")),
    Field("residential_status", "Residential status", "Personal", "cat", choices=RESIDENTIAL,
          used_by_model=True, aliases=("home_ownership", "residence", "housing")),
    # B. Employment
    Field("employment_type", "Employment type", "Employment", "cat", choices=EMPLOYMENT_TYPES,
          help="Used for analytics and policy; not present in the training data.",
          aliases=("emp_type", "employment")),
    Field("employer_category", "Employer / business category", "Employment", "text",
          aliases=("employer_type", "business_category", "industry")),
    Field("employment_years", "Years of employment", "Employment", "num", "years", min=0, max=60,
          used_by_model=True, aliases=("emp_length", "years_employed", "employment_length")),
    Field("monthly_income", "Monthly income", "Employment", "num", "currency", min=0,
          used_by_model=True, critical=True,
          aliases=("income", "monthly_salary", "salary", "net_monthly_income", "net_salary", "gross_monthly_income", "take_home")),
    Field("additional_income", "Additional monthly income", "Employment", "num", "currency", min=0,
          used_by_model=True, aliases=("other_income", "extra_income")),
    Field("employment_stability", "Employment stability", "Employment", "cat",
          choices=("Stable", "Moderate", "Unstable"),
          help="Self-declared; used only in the narrative report.",
          aliases=("job_stability",)),
    # C. Financial
    Field("monthly_expenses", "Monthly expenses", "Financial", "num", "currency", min=0,
          help="Used in the affordability check, not the model.",
          aliases=("expenses", "living_expenses")),
    Field("existing_emi", "Existing EMI (monthly)", "Financial", "num", "currency", min=0,
          used_by_model=True, aliases=("current_emi", "emi_existing", "existing_loan_emi", "existing_loan_obligations")),
    Field("active_loans", "Number of active loans / credit lines", "Financial", "int", min=0, max=100,
          used_by_model=True, aliases=("num_active_loans", "open_acc", "active_credit_lines", "number_of_active_loans", "active_accounts")),
    Field("bank_balance", "Bank balance", "Financial", "num", "currency", min=0,
          help="Used in the affordability check, not the model.", aliases=("balance",)),
    Field("savings", "Savings", "Financial", "num", "currency", min=0,
          help="Used in the affordability check, not the model.", aliases=("total_savings",)),
    Field("dti_ratio", "Debt-to-income ratio (%)", "Financial", "num", "%", min=0, max=100,
          used_by_model=True,
          help="If blank it is computed as existing EMI / total monthly income x 100.",
          aliases=("dti", "debt_to_income")),
    # D. Credit
    Field("credit_score", "Credit score", "Credit", "num", "300-900", min=300, max=900,
          used_by_model=True, critical=True, aliases=("cibil", "cibil_score", "fico", "score")),
    Field("previous_loans", "Number of previous loans / credit lines (ever)", "Credit", "int",
          min=0, max=200, used_by_model=True, aliases=("num_previous_loans", "total_acc", "total_credit_lines", "number_of_previous_loans", "previous_loan_count")),
    Field("previous_defaults", "Previous defaults / derogatory records", "Credit", "int",
          min=0, max=50, used_by_model=True, aliases=("defaults", "pub_rec", "num_defaults")),
    Field("delayed_payments_2y", "Delayed payments (30+ days) in last 2 years", "Credit", "int",
          min=0, max=50, used_by_model=True, aliases=("delayed_payments", "delinq_2yrs", "late_payments", "delayed_payment_history", "dpd_count")),
    Field("recent_inquiries", "Credit inquiries in last 6 months", "Credit", "int",
          min=0, max=50, used_by_model=True, aliases=("inquiries", "inq_last_6mths", "credit_inquiries", "enquiries", "credit_enquiries")),
    Field("credit_utilization", "Credit utilization (%)", "Credit", "num", "%", min=0, max=150,
          used_by_model=True, aliases=("revol_util", "utilization", "credit_util")),
    Field("outstanding_debt", "Outstanding revolving debt", "Credit", "num", "currency", min=0,
          used_by_model=True, aliases=("revol_bal", "outstanding_balance", "total_outstanding_debt")),
    Field("credit_history_years", "Credit history length", "Credit", "num", "years", min=0, max=80,
          used_by_model=True, aliases=("credit_history", "credit_age", "credit_history_length")),
    Field("bankruptcies", "Bankruptcies", "Credit", "int", min=0, max=10, used_by_model=True,
          aliases=("pub_rec_bankruptcies",)),
    Field("mortgage_accounts", "Mortgage accounts", "Credit", "int", min=0, max=50, used_by_model=True,
          aliases=("mort_acc",)),
    # E. Loan
    Field("loan_amount", "Loan amount requested", "Loan", "num", "currency", min=1,
          used_by_model=True, critical=True, aliases=("loan_amnt", "amount", "requested_amount", "requested_loan", "amt_requested", "loan_amt", "sanction_amount")),
    Field("loan_tenure_months", "Loan tenure", "Loan", "int", "months", min=3, max=360,
          used_by_model=True, critical=True, aliases=("tenure", "term", "loan_term", "tenure_months", "tenor", "loan_tenure")),
    Field("interest_rate", "Interest rate (% p.a.)", "Loan", "num", "%", min=0, max=60,
          used_by_model=True, critical=True, aliases=("int_rate", "rate", "roi")),
    Field("loan_purpose", "Loan purpose", "Loan", "cat", choices=LOAN_PURPOSES, used_by_model=True,
          aliases=("purpose",)),
    Field("expected_emi", "Expected monthly EMI", "Loan", "num", "currency", min=0, used_by_model=True,
          help="If blank it is computed from amount, rate and tenure.",
          aliases=("emi", "installment", "monthly_emi")),
]

FIELD_MAP = {f.name: f for f in FIELDS}
CRITICAL_FIELDS = [f.name for f in FIELDS if f.critical]

# Protected / sensitive attributes that must never enter the model.
PROHIBITED_FEATURES = {
    "gender", "sex", "religion", "caste", "race", "ethnicity", "nationality",
    "marital_status", "age", "disability", "pregnancy", "zip_code", "pincode", "addr_state",
}

# --------------------------------------------------------------------------------------
# Model feature matrix (output of features.build_model_features)
# --------------------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "credit_score", "interest_rate", "loan_tenure_months", "dti_ratio",
    "loan_to_income", "emi_to_income", "debt_to_annual_income", "credit_utilization",
    "credit_history_years", "employment_years", "delayed_payments_2y", "recent_inquiries",
    "active_loans", "previous_loans", "previous_defaults", "bankruptcies", "mortgage_accounts",
]
CATEGORICAL_FEATURES = ["residential_status", "loan_purpose"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Business-language names used in explanations.
FEATURE_LABELS = {
    "credit_score": "Credit score",
    "interest_rate": "Interest rate on the loan",
    "loan_tenure_months": "Loan tenure",
    "dti_ratio": "Debt-to-income ratio (existing obligations)",
    "loan_to_income": "Loan amount relative to annual income",
    "emi_to_income": "New EMI relative to monthly income",
    "debt_to_annual_income": "Outstanding revolving debt relative to income",
    "credit_utilization": "Credit utilization",
    "credit_history_years": "Length of credit history",
    "employment_years": "Years in employment",
    "delayed_payments_2y": "Delayed payments in last 2 years",
    "recent_inquiries": "Recent credit inquiries",
    "active_loans": "Number of active credit lines",
    "previous_loans": "Total credit lines held historically",
    "previous_defaults": "Previous defaults / derogatory records",
    "bankruptcies": "Bankruptcy records",
    "mortgage_accounts": "Mortgage accounts",
    "residential_status": "Residential status",
    "loan_purpose": "Loan purpose",
}

REQUIRED_OUTPUT_COLUMNS = [
    "Applicant ID", "Requested Loan", "Monthly Income", "Credit Score", "DTI Ratio",
    "Default Probability", "Risk Category", "Recommendation", "Key Risk Factors",
]
