"""Project paths and the user-editable settings (risk thresholds, decision policy)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
DB_PATH = DATA_DIR / "assessments.db"
SETTINGS_PATH = ROOT / "config" / "settings.json"

RANDOM_STATE = 42

DEFAULT_SETTINGS = {
    # Upper bound (inclusive) of each band, as a probability 0-1.
    "risk_bands": [
        {"name": "Very Low Risk", "max": 0.10, "decision": "APPROVE"},
        {"name": "Low Risk", "max": 0.25, "decision": "APPROVE"},
        {"name": "Moderate Risk", "max": 0.50, "decision": "MANUAL REVIEW"},
        {"name": "High Risk", "max": 0.75, "decision": "REJECT RECOMMENDATION"},
        {"name": "Very High Risk", "max": 1.00, "decision": "REJECT RECOMMENDATION"},
    ],
    # Policy overlays: rule-based checks that can escalate (never relax) the model-based decision.
    "policy": {
        "enabled": True,
        "min_credit_score": 550,          # below -> REJECT RECOMMENDATION
        "max_total_emi_to_income": 0.60,  # (existing + new EMI) / income above -> MANUAL REVIEW
        "min_disposable_income": 0,       # income - expenses - all EMIs below -> MANUAL REVIEW
        "min_age": 21,
        "max_age_at_maturity": 70,        # age + tenure years above -> MANUAL REVIEW
        "uncertainty_review_width": 0.20, # CI width above -> MANUAL REVIEW for APPROVE cases
    },
    "currency_symbol": "₹",
    "currency_locale": "IN",  # IN -> 5,00,000 grouping; anything else -> 500,000
}


def load_settings() -> dict:
    if SETTINGS_PATH.exists():
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            merged = copy.deepcopy(DEFAULT_SETTINGS)
            merged.update({k: v for k, v in data.items() if k != "policy"})
            merged["policy"].update(data.get("policy", {}))
            return merged
        except (json.JSONDecodeError, OSError):
            pass
    return copy.deepcopy(DEFAULT_SETTINGS)


def save_settings(settings: dict) -> None:
    validate_bands(settings["risk_bands"])
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")


def validate_bands(bands: list[dict]) -> None:
    prev = 0.0
    for b in bands:
        if not (prev < float(b["max"]) <= 1.0):
            raise ValueError("Risk band upper bounds must be strictly increasing and within (0, 1].")
        prev = float(b["max"])
    if abs(prev - 1.0) > 1e-9:
        raise ValueError("The last risk band must end at 100%.")
