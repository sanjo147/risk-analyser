"""REST API for the loan risk engine (FastAPI).

Run:   uvicorn api.main:app --port 8000      (docs at http://localhost:8000/docs)
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from loanrisk.config import load_settings  # noqa: E402
from loanrisk.features import apply_mapping, suggest_mapping  # noqa: E402
from loanrisk.predictor import ModelNotTrainedError, assess, load_bundle  # noqa: E402
from loanrisk.reporting import bulk_excel, output_table  # noqa: E402

app = FastAPI(title="Loan Risk Assessment API", version="1.0",
              description="Educational decision-support API. Outputs are model estimates and illustrative "
                          "policy recommendations, not lending decisions.")


class Applicant(BaseModel):
    applicant_id: str
    monthly_income: float = Field(gt=0)
    credit_score: float = Field(ge=300, le=900)
    loan_amount: float = Field(gt=0)
    loan_tenure_months: int = Field(ge=3, le=360)
    interest_rate: float = Field(ge=0, le=60)
    age: Optional[int] = None
    marital_status: Optional[str] = None
    dependents: Optional[int] = None
    residential_status: Optional[Literal["Owned", "Mortgaged", "Rented", "Other"]] = None
    employment_type: Optional[str] = None
    employer_category: Optional[str] = None
    employment_years: Optional[float] = None
    additional_income: Optional[float] = None
    employment_stability: Optional[str] = None
    monthly_expenses: Optional[float] = None
    existing_emi: Optional[float] = None
    active_loans: Optional[int] = None
    bank_balance: Optional[float] = None
    savings: Optional[float] = None
    dti_ratio: Optional[float] = None
    previous_loans: Optional[int] = None
    previous_defaults: Optional[int] = None
    delayed_payments_2y: Optional[int] = None
    recent_inquiries: Optional[int] = None
    credit_utilization: Optional[float] = None
    outstanding_debt: Optional[float] = None
    credit_history_years: Optional[float] = None
    bankruptcies: Optional[int] = None
    mortgage_accounts: Optional[int] = None
    loan_purpose: Optional[str] = None
    expected_emi: Optional[float] = None


def _clean(v):
    if isinstance(v, (np.floating, float)) and np.isnan(v):
        return None
    if isinstance(v, np.generic):
        return v.item()
    return v


@app.get("/health")
def health():
    try:
        meta = load_bundle()["meta"]
        return {"status": "ok", "model": meta["model_name"], "trained_at": meta["trained_at"]}
    except ModelNotTrainedError as e:
        raise HTTPException(503, str(e))


@app.post("/predict")
def predict(applicant: Applicant):
    res, issues, details = assess(pd.DataFrame([applicant.model_dump()]), load_settings())
    row = res.iloc[0]
    out = {
        "applicant_id": row["applicant_id"],
        "model_prediction": {
            "default_probability": _clean(row["default_probability"]),
            "risk_category": _clean(row["risk_category"]),
            "confidence": _clean(row["confidence"]),
            "historical_default_rate_95ci": [_clean(row["observed_rate_low"]), _clean(row["observed_rate_high"])],
        },
        "policy_recommendation": {"decision": row["recommendation"],
                                  "band_default_decision": _clean(row["model_band_decision"]),
                                  "notes": _clean(row["policy_notes"])},
        "validation_issues": issues.to_dict("records"),
        "disclaimer": "Model estimate and illustrative policy recommendation - not a lending decision.",
    }
    d = details.get(row["applicant_id"])
    if d:
        out["explanation"] = {k: d[k] for k in ("increasing", "reducing", "positive", "negative", "suggestions", "reason")}
    return out


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...), format: Literal["json", "xlsx"] = "json"):
    data = await file.read()
    try:
        df = pd.read_excel(io.BytesIO(data)) if file.filename.lower().endswith((".xlsx", ".xls")) \
            else pd.read_csv(io.BytesIO(data))
    except Exception as e:
        raise HTTPException(400, f"Could not read file: {e}")
    mapped = apply_mapping(df, suggest_mapping(list(df.columns)))
    res, issues, _ = assess(mapped, load_settings(), explain=format == "json")
    if format == "xlsx":
        return Response(bulk_excel(res, issues, load_bundle()["meta"]),
                        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        headers={"Content-Disposition": "attachment; filename=assessment_results.xlsx"})
    tbl = output_table(res).replace({np.nan: None})
    return {"n": len(res), "results": tbl.to_dict("records"),
            "validation_issues": issues.replace({np.nan: None}).to_dict("records")}
