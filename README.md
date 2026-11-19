# AI-Powered Loan Risk Assessment & Approval System

An end-to-end, educational **decision-support** system that estimates the probability that a loan applicant will
default, places them in a risk category, gives an illustrative policy recommendation (approve / manual review /
reject) and explains why. It supports a single applicant entered in a form and bulk Excel/CSV uploads.

> **Disclaimer.** The model is trained on historical US peer-to-peer loans (LendingClub). Probabilities are model
> estimates, and recommendations follow an illustrative, configurable policy. They are **not** lending decisions.

## Features
* Six algorithms compared (Logistic Regression, Decision Tree, Random Forest, XGBoost, KNN, Gradient Boosting)
  using stratified 5-fold CV, a validation set for calibration/selection and a held-out test set.
* Calibrated probabilities (Brier score, calibration curve) and per-applicant confidence.
* Configurable risk bands, decision policy and rule-based policy overlays, with each band checked against real
  historical default rates.
* SHAP explanations in plain language: top-5 risk-increasing and risk-reducing factors, global importance.
* Validation of uploads, column mapping, a clear "cannot assess" outcome when critical data is missing.
* Fairness audit, model limitations and a written evaluation report generated from real results.
* Streamlit UI (10 pages), FastAPI REST API, SQLite history, Excel/CSV/HTML reports.

## Project structure
```
loanrisk/            core package
  schema.py          applicant fields, model features, aliases, protected attributes
  features.py        column mapping, validation/cleaning, feature engineering (shared by train + inference)
  data_prep.py       LendingClub raw file -> canonical training table
  preprocess.py      sklearn preprocessing pipeline (winsorise, impute, scale, one-hot)
  train.py           model comparison, calibration, evaluation, fairness audit, figures, report
  predictor.py       assessment engine (prediction -> band -> policy -> explanation)
  explain.py         SHAP + business-language explanations, indicators, suggestions
  risk.py            banding and confidence-interval helpers
  reporting.py       Excel and HTML reports
  storage.py         SQLite persistence
  config.py          paths and editable settings (config/settings.json)
app/                 Streamlit UI (streamlit_app.py + views/)
api/main.py          FastAPI service
scripts/make_sample_data.py   synthetic test applicants
data/sample/         synthetic sample files for testing
data/processed/      prepared training table (150k loans)
models/              trained model, metrics.json, test-set predictions
reports/             MODEL_EVALUATION.md + figures
docs/                DATASET.md, ML_WORKFLOW.md
tests/               end-to-end tests
```

## Installation (Windows PowerShell)
Python 3.10+ is required (developed on 3.12).
```powershell
cd "D:\loan risk"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running
The trained model is already in `models/`. Start the app:
```powershell
streamlit run app/streamlit_app.py
```
Open http://localhost:8501, then e.g. **Home Dashboard → "Assess the bundled synthetic sample dataset"**, or upload
`data/sample/sample_applicants.xlsx` on **Upload Customer Dataset**.

REST API (optional):
```powershell
uvicorn api.main:app --port 8000
```
Interactive docs at http://localhost:8000/docs. Endpoints: `GET /health`, `POST /predict` (JSON applicant),
`POST /predict/batch` (CSV/XLSX upload, `?format=xlsx` returns the Excel report).

Tests:
```powershell
pip install pytest httpx
python -m pytest -q
```

## Re-training from scratch
1. Download `accepted_2007_to_2018Q4.csv.gz` from the Kaggle dataset `wordsforthewise/lending-club` into
   `data/raw/` (≈ 392 MB).
2. Run:
```powershell
python -m loanrisk.data_prep --sample 150000
python -m loanrisk.train
```
Training takes about 5 minutes on a laptop and rewrites `models/`, `reports/MODEL_EVALUATION.md` and figures.

## Pages
| Page | What it does |
|---|---|
| Home Dashboard | KPIs (assessed, approvals, rejections, reviews, avg PD), risk/recommendation/loan/score distributions |
| Upload Customer Dataset | drag-and-drop CSV/XLSX, column mapping, validation report, run assessment |
| Individual Assessment | form for one applicant, gauge, full explained report, HTML download |
| Bulk Risk Assessment | ranked results, high-risk and manual-review lists, filters, Excel download |
| Applicant Results | search/filter and open any applicant's full profile |
| Risk Analytics | income vs PD, risk by employment type and loan amount, exposure analysis |
| Model Performance | test metrics, confusion matrix, ROC/PR, calibration, model comparison, fairness audit |
| Model Explainability | global SHAP importance and per-applicant contributions |
| Download Reports | Excel/CSV/HTML reports, evaluation report, saved history, data deletion |
| Model Configuration | edit risk thresholds and policy, see each band's real historical default rate |

## Input fields
Required (★): Applicant ID, monthly income, credit score, loan amount, loan tenure, interest rate.
Everything else is optional. Fields used by the model, and fields used only by policy rules or reports,
are listed in [docs/DATASET.md](docs/DATASET.md).

## Documentation
* [docs/GUIDE.md](docs/GUIDE.md) - **start here**: how it works, algorithms, accuracy, hosting, optimisation, demo dataset
* [docs/DATASET.md](docs/DATASET.md) - data source, target, selection, leakage controls, splits, limitations
* [docs/ML_WORKFLOW.md](docs/ML_WORKFLOW.md) - full pipeline, imbalance, calibration, uncertainty, explainability, fairness
* [reports/MODEL_EVALUATION.md](reports/MODEL_EVALUATION.md) - generated evaluation results
