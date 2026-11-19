"""End-to-end tests.  Run:  python -m pytest -q"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loanrisk.config import DEFAULT_SETTINGS, SAMPLE_DIR  # noqa: E402
from loanrisk.features import apply_mapping, clean_applicants, suggest_mapping  # noqa: E402
from loanrisk.predictor import assess  # noqa: E402
from loanrisk.reporting import bulk_excel, money, output_table  # noqa: E402
from loanrisk.risk import categorize  # noqa: E402
from loanrisk.schema import MODEL_FEATURES, PROHIBITED_FEATURES, REQUIRED_OUTPUT_COLUMNS  # noqa: E402

BASE = dict(applicant_id="T1", monthly_income=60000, credit_score=760, loan_amount=400000, loan_tenure_months=36,
            interest_rate=11.5, existing_emi=5000, credit_utilization=25, credit_history_years=10,
            employment_years=6, residential_status="Owned", loan_purpose="car", delayed_payments_2y=0,
            previous_defaults=0, recent_inquiries=0, active_loans=4, previous_loans=10, outstanding_debt=30000,
            monthly_expenses=20000, age=35)


def test_no_protected_attribute_in_model():
    assert not (set(MODEL_FEATURES) & PROHIBITED_FEATURES)


def test_band_boundaries():
    bands = DEFAULT_SETTINGS["risk_bands"]
    cats, decs = categorize([0.0, 0.10, 0.1001, 0.25, 0.5, 0.75, 0.7501, 1.0], bands)
    assert list(cats) == ["Very Low Risk", "Very Low Risk", "Low Risk", "Low Risk", "Moderate Risk", "High Risk",
                          "Very High Risk", "Very High Risk"]
    assert decs[4] == "MANUAL REVIEW" and decs[-1] == "REJECT RECOMMENDATION"


def test_mapping_custom_columns():
    df = pd.read_csv(SAMPLE_DIR / "sample_applicants_custom_columns.csv")
    m = suggest_mapping(list(df.columns))
    assert m["CIBIL"] == "credit_score" and m["Net Salary"] == "monthly_income" and m["Tenor"] == "loan_tenure_months"


def test_missing_critical_blocks_prediction():
    df = pd.DataFrame([{**BASE, "credit_score": None}])
    res, issues, _ = assess(df)
    assert res.iloc[0]["recommendation"] == "CANNOT ASSESS"
    assert np.isnan(res.iloc[0]["default_probability"])
    assert (issues.severity == "error").any()


def test_deterministic_and_bounded():
    df = pd.DataFrame([BASE])
    p1 = assess(df)[0].iloc[0]["default_probability"]
    p2 = assess(df)[0].iloc[0]["default_probability"]
    assert p1 == p2 and 0 < p1 < 1


def test_sensible_direction():
    good = assess(pd.DataFrame([BASE]))[0].iloc[0]["default_probability"]
    bad = assess(pd.DataFrame([{**BASE, "credit_score": 650, "delayed_payments_2y": 3, "recent_inquiries": 5,
                                "credit_utilization": 95, "interest_rate": 24}]))[0].iloc[0]["default_probability"]
    assert bad > good


def test_policy_overlay_escalates():
    res = assess(pd.DataFrame([{**BASE, "credit_score": 520}]))[0].iloc[0]
    assert res["recommendation"] == "REJECT RECOMMENDATION"
    assert "minimum" in res["policy_notes"]


def test_bulk_sample_and_excel():
    df = pd.read_csv(SAMPLE_DIR / "sample_applicants.csv")
    mapped = apply_mapping(df, suggest_mapping(list(df.columns)))
    res, issues, details = assess(mapped)
    assert len(res) == len(df)
    assert (res["recommendation"] == "CANNOT ASSESS").sum() == 3
    assert res["applicant_id"].is_unique
    t = output_table(res)
    assert list(t.columns[:9]) == REQUIRED_OUTPUT_COLUMNS
    xl = pd.read_excel(io.BytesIO(bulk_excel(res, issues, {"model_name": "x"})), sheet_name=None)
    assert {"Assessment Results", "Portfolio Summary", "High Risk", "Manual Review"} <= set(xl)
    ok = res[res["default_probability"].notna()]
    assert all(len(details[i]["increasing"]) + len(details[i]["reducing"]) > 0 for i in ok["applicant_id"])


def test_money_format():
    assert money(500000) == "₹5,00,000"
    assert money(1234567, "$", "INTL") == "$1,234,567"


def test_cleaning_parses_text_values():
    df = pd.DataFrame([{**BASE, "monthly_income": "₹45,000", "loan_tenure_months": "3 years", "interest_rate": "12.5%"}])
    clean, _ = clean_applicants(df)
    assert clean.iloc[0]["monthly_income"] == 45000
    assert clean.iloc[0]["loan_tenure_months"] == 36
    assert clean.iloc[0]["interest_rate"] == 12.5


def test_api():
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from api.main import app
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    r = c.post("/predict", json=BASE).json()
    assert 0 < r["model_prediction"]["default_probability"] < 1
    with open(SAMPLE_DIR / "sample_applicants_small.csv", "rb") as fh:
        r = c.post("/predict/batch", files={"file": ("s.csv", fh, "text/csv")}).json()
    assert r["n"] == 10
