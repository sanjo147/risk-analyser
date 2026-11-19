"""Hand-written DEMO applicants for uploading and testing (fictional people, no outcome labels).

Each row is a persona designed to exercise a specific path through the system: strong / average / weak
profiles, every policy rule, out-of-training-range inputs, missing data and messy formatting.
The 'Scenario' column is ignored by the model (it is left unmapped on upload) - it documents intent.

Run:  python scripts/make_demo_data.py   ->  data/sample/demo_applicants.xlsx and .csv
"""
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from loanrisk.config import SAMPLE_DIR  # noqa: E402

COLS = ["Applicant ID", "Scenario", "Age", "Marital Status", "Dependents", "Residential Status", "Employment Type",
        "Employer Category", "Years of Employment", "Monthly Income", "Additional Income", "Employment Stability",
        "Monthly Expenses", "Existing EMI", "Number of Active Loans", "Bank Balance", "Savings", "Credit Score",
        "Number of Previous Loans", "Previous Defaults", "Delayed Payments (2 yrs)", "Credit Inquiries (6 months)",
        "Credit Utilization (%)", "Outstanding Debt", "Credit History (years)", "Loan Amount", "Loan Tenure (months)",
        "Interest Rate (%)", "Loan Purpose", "Expected EMI"]

R = [
    # --- strong profiles -------------------------------------------------------------------------------
    ("DEMO001", "Prime salaried, small car loan", 38, "Married", 2, "Owned", "Salaried", "Government", 12, 120000, 0, "Stable",
     40000, 0, 2, 300000, 900000, 812, 9, 0, 0, 0, 12, 20000, 16, 400000, 36, 8.9, "car", None),
    ("DEMO002", "High income IT professional, home improvement", 41, "Married", 1, "Mortgaged", "Salaried", "IT / Software", 15, 250000, 20000, "Stable",
     90000, 35000, 4, 800000, 2500000, 795, 14, 0, 0, 1, 18, 60000, 19, 1000000, 36, 9.5, "home_improvement", None),
    ("DEMO003", "Doctor, low utilisation, long history", 45, "Married", 2, "Owned", "Self-employed", "Healthcare", 18, 300000, 0, "Stable",
     100000, 20000, 3, 1200000, 4000000, 828, 20, 0, 0, 0, 8, 40000, 24, 800000, 24, 8.5, "major_purchase", None),
    ("DEMO004", "Young professional, clean record", 27, "Single", 0, "Rented", "Salaried", "Banking & Finance", 4, 85000, 0, "Stable",
     30000, 6000, 3, 150000, 350000, 768, 6, 0, 0, 1, 22, 25000, 6, 300000, 36, 10.5, "debt_consolidation", None),
    ("DEMO005", "Teacher, modest loan", 35, "Married", 2, "Owned", "Salaried", "Education", 9, 60000, 5000, "Stable",
     28000, 4000, 2, 90000, 300000, 752, 7, 0, 0, 0, 25, 15000, 11, 200000, 36, 10.9, "education", None),
    # --- average profiles ------------------------------------------------------------------------------
    ("DEMO006", "Average profile, debt consolidation", 33, "Married", 1, "Rented", "Salaried", "Retail", 5, 55000, 0, "Moderate",
     25000, 8000, 5, 40000, 120000, 712, 12, 0, 0, 2, 55, 90000, 9, 250000, 36, 13.5, "debt_consolidation", None),
    ("DEMO007", "Self-employed shop owner", 44, "Married", 3, "Owned", "Business", "Retail trade", 10, 90000, 10000, "Moderate",
     45000, 12000, 4, 200000, 400000, 726, 10, 0, 0, 1, 40, 80000, 14, 400000, 48, 13.0, "small_business", None),
    ("DEMO008", "Credit card refinance, high utilisation", 31, "Single", 0, "Rented", "Salaried", "Manufacturing", 6, 70000, 0, "Stable",
     30000, 9000, 6, 30000, 80000, 701, 15, 0, 0, 2, 78, 180000, 8, 250000, 36, 14.5, "credit_card", None),
    ("DEMO009", "Wedding loan, 60 months", 29, "Single", 0, "Rented", "Salaried", "IT / Software", 3, 65000, 0, "Stable",
     28000, 5000, 3, 60000, 100000, 735, 5, 0, 0, 1, 35, 30000, 5, 380000, 60, 12.5, "wedding", None),
    ("DEMO010", "Medical emergency, short tenure", 50, "Widowed", 1, "Owned", "Salaried", "Healthcare", 20, 75000, 8000, "Stable",
     35000, 0, 1, 50000, 150000, 744, 8, 0, 0, 0, 20, 10000, 22, 150000, 12, 12.0, "medical", None),
    # --- weaker profiles (model risk rises) ------------------------------------------------------------
    ("DEMO011", "Recent late payments, many enquiries", 34, "Married", 2, "Rented", "Salaried", "Transport", 2, 45000, 0, "Unstable",
     25000, 9000, 7, 8000, 15000, 671, 11, 0, 3, 6, 88, 120000, 7, 200000, 60, 21.5, "debt_consolidation", None),
    ("DEMO012", "Previous default on record", 39, "Divorced", 1, "Rented", "Self-employed", "Hospitality", 4, 50000, 0, "Moderate",
     28000, 10000, 5, 10000, 20000, 668, 9, 2, 1, 3, 70, 90000, 10, 180000, 60, 22.9, "other", None),
    ("DEMO013", "High rate, high DTI, short history", 26, "Single", 0, "Rented", "Salaried", "Retail", 1, 40000, 0, "Unstable",
     22000, 12000, 6, 5000, 8000, 675, 7, 0, 2, 5, 92, 70000, 2.5, 150000, 60, 24.5, "vacation", None),
    ("DEMO014", "Small business, volatile income", 36, "Married", 2, "Rented", "Business", "Construction", 2, 60000, 0, "Unstable",
     30000, 14000, 8, 15000, 30000, 680, 13, 1, 2, 4, 85, 150000, 6, 300000, 60, 20.0, "small_business", None),
    ("DEMO015", "Moving loan, thin file", 23, "Single", 0, "Rented", "Salaried", "Retail", 0.5, 30000, 0, "Unstable",
     18000, 3000, 2, 4000, 5000, 690, 2, 0, 1, 4, 65, 20000, 1.5, 100000, 36, 19.0, "moving", None),
    # --- policy rules ----------------------------------------------------------------------------------
    ("DEMO016", "POLICY: credit score below minimum (reject)", 40, "Married", 2, "Rented", "Salaried", "Manufacturing", 8, 55000, 0, "Stable",
     25000, 6000, 4, 20000, 40000, 520, 10, 1, 2, 2, 60, 60000, 12, 150000, 36, 18.0, "debt_consolidation", None),
    ("DEMO017", "POLICY: total EMI > 60% of income (review)", 37, "Married", 2, "Mortgaged", "Salaried", "Banking & Finance", 10, 80000, 0, "Stable",
     30000, 38000, 6, 50000, 100000, 760, 12, 0, 0, 1, 30, 50000, 14, 380000, 36, 11.0, "home_improvement", None),
    ("DEMO018", "POLICY: expenses + EMIs exceed income (review)", 32, "Married", 3, "Rented", "Salaried", "Retail", 5, 50000, 0, "Moderate",
     38000, 6000, 3, 10000, 20000, 740, 8, 0, 0, 1, 35, 30000, 8, 200000, 36, 12.0, "car", None),
    ("DEMO019", "POLICY: age at maturity above 70 (review)", 66, "Married", 0, "Owned", "Self-employed", "Professional services", 30, 110000, 30000, "Stable",
     40000, 0, 1, 500000, 2000000, 790, 15, 0, 0, 0, 10, 20000, 35, 500000, 60, 10.0, "home_improvement", None),
    ("DEMO020", "POLICY: younger than minimum age (review)", 19, "Single", 0, "Other", "Salaried", "Retail", 1, 25000, 0, "Moderate",
     12000, 0, 1, 10000, 20000, 720, 1, 0, 0, 1, 30, 5000, 1, 50000, 24, 14.0, "education", None),
    # --- out of training range (model extrapolating -> low confidence) ---------------------------------
    ("DEMO021", "OUT OF RANGE: loan ~0.9x annual income", 34, "Married", 1, "Rented", "Salaried", "IT / Software", 6, 45000, 0, "Stable",
     18000, 8000, 3, 60000, 150000, 760, 8, 0, 0, 1, 28, 40000, 9, 500000, 36, 12.5, "home_improvement", None),
    ("DEMO022", "OUT OF RANGE: credit score 640 (below training data)", 42, "Married", 2, "Owned", "Salaried", "Government", 14, 70000, 0, "Stable",
     30000, 5000, 3, 80000, 200000, 640, 10, 0, 1, 1, 50, 50000, 15, 200000, 36, 15.0, "car", None),
    ("DEMO023", "OUT OF RANGE: interest rate 28%", 30, "Single", 0, "Rented", "Salaried", "Retail", 3, 40000, 0, "Moderate",
     20000, 4000, 4, 10000, 20000, 700, 6, 0, 0, 3, 60, 40000, 4, 100000, 36, 28.0, "other", None),
    # --- data-quality cases ----------------------------------------------------------------------------
    ("DEMO024", "MISSING: credit score (cannot assess)", 35, "Married", 1, "Rented", "Salaried", "Retail", 5, 50000, 0, "Stable",
     22000, 5000, 3, 30000, 80000, None, 8, 0, 0, 1, 40, 30000, 8, 150000, 36, 12.0, "car", None),
    ("DEMO025", "MISSING: loan tenure (cannot assess)", 35, "Married", 1, "Rented", "Salaried", "Retail", 5, 50000, 0, "Stable",
     22000, 5000, 3, 30000, 80000, 730, 8, 0, 0, 1, 40, 30000, 8, 150000, None, 12.0, "car", None),
    ("DEMO026", "MISSING: most optional credit fields (imputed)", 30, "Single", 0, "Rented", "Salaried", "IT / Software", None, 70000, None, None,
     None, None, None, None, None, 745, None, None, None, None, None, None, None, 200000, 36, 11.5, None, None),
    ("DEMO027", "TEXT FORMATS: '₹65,000', '3 years', '12.5%'", 33, "Married", 1, "rent", "Salaried", "Banking & Finance", "7 years", "₹65,000", "0", "Stable",
     "₹25,000", "₹5,000", 3, "₹1,00,000", "₹2,50,000", 755, 9, 0, 0, 1, "30%", "₹45,000", 10, "₹2,50,000", "3 years", "12.5%", "Home renovation", None),
    ("DEMO028", "INVALID: interest rate 95% (treated as missing)", 35, "Married", 1, "Rented", "Salaried", "Retail", 5, 50000, 0, "Stable",
     22000, 5000, 3, 30000, 80000, 730, 8, 0, 0, 1, 40, 30000, 8, 150000, 36, 95, "car", None),
    ("DEMO001", "DUPLICATE ID (suffix added)", 38, "Married", 2, "Owned", "Salaried", "Government", 12, 120000, 0, "Stable",
     40000, 0, 2, 300000, 900000, 812, 9, 0, 0, 0, 12, 20000, 16, 400000, 36, 8.9, "car", None),
    ("", "MISSING ID (auto-generated)", 45, "Married", 2, "Owned", "Salaried", "Government", 15, 95000, 0, "Stable",
     35000, 10000, 3, 200000, 600000, 780, 11, 0, 0, 0, 20, 30000, 20, 300000, 36, 10.0, "car", None),
]

df = pd.DataFrame(R, columns=COLS)
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
df.to_excel(SAMPLE_DIR / "demo_applicants.xlsx", index=False)
df.to_csv(SAMPLE_DIR / "demo_applicants.csv", index=False, encoding="utf-8-sig")
print(f"wrote {len(df)} demo applicants -> {SAMPLE_DIR / 'demo_applicants.xlsx'}")
