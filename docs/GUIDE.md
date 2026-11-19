# Learning Guide - How the Loan Risk System Works

This guide explains the project from the ground up: what happens to an applicant, which algorithms and tools
are used and why, how accuracy is measured (with the real numbers), how to host it with a shareable link, how to
improve it, and how to test it with the demo dataset.

---

## 1. The big picture

The system answers one question: **"Based on loans that happened in the past, how likely is this applicant to
default?"** It learns that from historical loans whose outcome is known, then applies what it learned to new
applicants.

```
 HISTORICAL LOANS (known outcome)                     NEW APPLICANT (unknown outcome)
 150,000 LendingClub loans                            form / Excel / CSV
        │                                                    │
        ▼                                                    ▼
 1. Clean + build features ─────── same code ────────► 1. Validate, clean, build features
        │                                                    │
        ▼                                                    ▼
 2. Train 6 algorithms, compare,                       2. Trained model → probability of default
    calibrate, pick the best ───────── saved model ──►       (e.g. 16.8%)
                                                             │
                                                             ▼
                                                       3. Risk band   (Low Risk: 10–25%)
                                                             │
                                                             ▼
                                                       4. Policy      (Low Risk → APPROVE,
                                                                       then rule checks may escalate)
                                                             │
                                                             ▼
                                                       5. Explanation (SHAP: which inputs pushed risk up/down)
```

### The journey of one applicant, step by step

| Step | Code | What happens |
|---|---|---|
| 1. Upload & map | `features.suggest_mapping` | Column names like `CIBIL` or `Net Salary` are matched to system fields (exact name → alias → fuzzy match). You can override each match. |
| 2. Validate | `features.clean_applicants` | Text like `₹45,000` / `3 years` / `12.5%` is converted to numbers; impossible values (interest 95%) are rejected; duplicate IDs fixed. If a **critical** field (credit score, income, loan amount, tenure, rate) is missing → **CANNOT ASSESS**. Nothing is guessed. |
| 3. Feature engineering | `features.build_model_features` | Raw inputs become 19 model features. Money amounts become **ratios** (loan ÷ annual income, EMI ÷ income, debt ÷ income) so a model trained in US dollars works for rupees. |
| 4. Preprocessing | `preprocess.py` | Extreme values are capped (outliers), missing optional values filled with the training median, numbers scaled, categories one-hot encoded. These settings were learned from training data only. |
| 5. Prediction | `predictor.assess` | The calibrated XGBoost model outputs a probability, e.g. 0.168 = 16.8%. |
| 6. Risk band | `risk.categorize` | 0–10% Very Low · 10–25% Low · 25–50% Moderate · 50–75% High · 75–100% Very High (editable on *Model Configuration*). |
| 7. Policy | `predictor.assess` | Band → default decision (Approve / Review / Reject). Then **policy overlays** can only make it stricter: credit score < 550 → reject; total EMIs > 60% of income → review; income doesn't cover expenses + EMIs → review; age rules → review; low confidence → review. |
| 8. Confidence | `predictor.assess` | High / Medium / Low, based on (a) whether inputs are inside the range the model was trained on, (b) how many inputs were missing, (c) the historical default rate of similar-scored loans with a 95% interval. |
| 9. Explanation | `explain.py` | SHAP splits the prediction into per-input contributions → "Top 5 factors increasing / reducing risk", plus rule-based positive/negative indicators and improvement suggestions. |
| 10. Reports | `reporting.py`, `storage.py` | Excel (6 sheets), CSV, printable HTML report per applicant; every run saved in local SQLite. |

**Model vs policy - the most important design idea.** The *model* only estimates a probability. The *policy*
(bands + rules) is a business decision you can change without retraining. The UI always labels which layer
produced what.

---

## 2. The data

* **LendingClub** (US peer-to-peer lender), 2.26 million loans, of which **676,059** had a final outcome and had
  reached maturity. A stratified random sample of **150,000** was used; **14.87%** of them defaulted.
* Split (stratified, fixed seed 42):

| Set | Loans | Used for |
|---|---|---|
| Training | 90,000 | learning patterns; 5-fold cross-validation |
| Validation | 30,000 | calibrating probabilities and choosing the model |
| Test | 30,000 | final, honest score - looked at once |

**Why three sets?** If you measure a model on data it learned from, it looks better than it is (like marking
your own exam with the answer sheet you studied). The test set simulates "brand-new applicants".

**Leakage prevention:** columns only known *after* a loan is issued (payments made, recoveries...) were removed -
otherwise the model would "cheat" by seeing the answer. LendingClub's own risk grade and the borrower's state were
also excluded. Details: [DATASET.md](DATASET.md).

---

## 3. The algorithms

All six learn a function *inputs → probability of default*. They differ in *how* they draw the boundary between
"likely to repay" and "likely to default".

| Algorithm | How it works (intuition) | Strengths | Weaknesses | Test ROC-AUC |
|---|---|---|---|---|
| **Logistic Regression** | Adds up weighted inputs (score = w₁·credit score + w₂·DTI + …) and squashes the total into 0–1 with the logistic (S-shaped) curve. | Simple, fast, transparent - the classic credit scorecard. | Only straight-line relationships unless you engineer features. | 0.682 |
| **Decision Tree** | A flowchart of yes/no questions ("credit score < 700? → DTI > 25? → …"); each leaf holds a default rate. | Very easy to read. | One tree is unstable and either too simple or overfits. | 0.666 |
| **Random Forest** | 300 trees, each trained on a random sample of rows and features; averages their votes. | Robust, handles interactions, little tuning. | Larger, slower; overfit most here (train AUC 0.75 vs test 0.68). | 0.684 |
| **XGBoost** ✅ | **Boosting**: trees built one after another, each correcting the errors of the ones before (gradient descent on the loss). Regularised, fast "histogram" implementation. | Usually the strongest on tabular data; supports **monotonic constraints**. | More hyper-parameters. | **0.685** |
| **K-Nearest Neighbours** | Finds the 151 most similar past borrowers and uses their default rate. | Intuitive ("people like you"). | Slow on large data, struggles with many features of mixed importance. | 0.671 |
| **Gradient Boosting** (scikit-learn) | Same boosting idea as XGBoost, classic implementation. | Strong, low overfitting here. | Slow (≈150 s vs 17 s), no monotonic constraints. | 0.686 |

### Why XGBoost was chosen although Gradient Boosting scored 0.001 higher
The XGBoost model is trained with **monotonic constraints**: a higher credit score may only *lower* risk, a higher
DTI / loan-to-income / interest rate / late-payment count may only *raise* it. Without that, correlated inputs
produced nonsense explanations (e.g. "a bigger EMI reduces risk"). The rule in `train.py`: pick the top model, but
prefer the constrained XGBoost if it is within **0.005 AUC** of the leader (about one cross-validation standard
deviation - i.e. statistically a tie). Banks do the same; regulators expect explanations that make economic sense.

### Supporting techniques
* **Class imbalance** - only ~15% default, so models are trained with *class weights* (mistakes on defaulters count
  ~5.7× more). This makes raw probabilities too high...
* **Calibration** - ...so a **sigmoid (Platt) calibration** layer is fitted on the validation set to map scores back
  to real-world probabilities. After calibration, "20%" really means ~20 out of 100 such applicants default.
* **Cross-validation** - training data is cut into 5 folds; train on 4, score on 1, rotate. Gives a mean ± spread,
  showing how stable a model is.
* **SHAP** (SHapley Additive exPlanations) - from game theory: treats each input as a "player" and fairly divides
  the prediction among them. Contributions add up exactly to (this applicant's score − average score).

---

## 4. Tools and what they are used for

| Tool | Role in this project |
|---|---|
| **Python 3.12** | Language for everything |
| **pandas** | Reading CSV/Excel, cleaning, table operations |
| **NumPy** | Fast numeric arrays and maths (EMI formula, clipping) |
| **scikit-learn** | Pipelines, imputation, scaling, one-hot encoding, LR/DT/RF/KNN/GB, cross-validation, calibration, all metrics |
| **XGBoost** | The selected gradient-boosting model |
| **SHAP** | Per-applicant and global explanations |
| **Matplotlib / Seaborn** | Static figures in the evaluation report (`reports/figures`) |
| **Plotly** | Interactive charts in the app (gauge, scatter, treemap…) |
| **Streamlit** | The 10-page web interface |
| **FastAPI + Uvicorn** | REST API (`/predict`, `/predict/batch`) for other systems to call |
| **openpyxl** | Writing formatted Excel reports |
| **SQLite** | Local history of assessment runs |
| **joblib** | Saving/loading the trained model (`models/model.joblib`) |
| **pytest** | Automated end-to-end tests (`tests/`) |

---

## 5. How accuracy is calculated - and how accurate it is

### 5.1 The confusion matrix
Each test loan's probability is turned into a yes/no using a **threshold**. Here the natural threshold is the
policy boundary: above **25%** = "not auto-approved" (flagged).

|                       | Predicted: repay | Predicted: default (flagged) |
|---|---|---|
| **Actually repaid**   | TN = 22,529 ✅ | FP = 3,011 (good borrower flagged) |
| **Actually defaulted**| **FN = 3,212** ❌ (bad borrower approved) | TP = 1,248 ✅ |

### 5.2 The formulas (with the real test-set numbers, threshold 25%)

| Metric | Formula | Value | Meaning |
|---|---|---|---|
| **Accuracy** | (TP + TN) / all | (1,248 + 22,529) / 30,000 = **79.3%** | share of correct yes/no calls |
| **Precision** | TP / (TP + FP) | 1,248 / 4,259 = **29.3%** | of flagged applicants, how many really defaulted |
| **Recall** | TP / (TP + FN) | 1,248 / 4,460 = **28.0%** | of real defaulters, how many were caught |
| **F1** | 2·P·R / (P + R) | **0.286** | balance of precision and recall |
| **False-negative rate** | FN / (FN + TP) | **72.0%** | defaulters that slipped through - the costly error |
| **ROC-AUC** | area under the ROC curve | **0.685** (95% CI 0.678 – 0.692) | probability that a random defaulter gets a higher score than a random non-defaulter |
| **PR-AUC** | area under precision–recall curve | **0.263** (random = 0.149) | ranking quality focused on the rare class |
| **Brier score** | mean of (probability − outcome)² | **0.120** (always predicting 14.9% = 0.127) | accuracy of the probabilities themselves; lower is better |

### 5.3 Why "accuracy" alone is misleading here
At the default 0.50 threshold the model scores **85.1% accuracy** - and catches **zero** defaulters
(TP = 0, FN = 4,460). Simply approving everybody also gives 85.1%, because 85% of borrowers repay. That's why the
project reports ROC-AUC, PR-AUC, recall, false negatives and calibration, and why the requirement said "do not
claim reliability from accuracy".

### 5.4 So how good is it, honestly?
* **ROC-AUC 0.685** = clearly better than chance (0.5) and in the range usually seen on LendingClub data
  when the lender's internal grade is not used. It is **not** good enough to auto-decide loans - hence the
  manual-review band.
* **Calibration is good:** within each band, predicted ≈ actual default rate (Very Low: 6.6% vs 6.5%,
  Low: 16.4% vs 16.6%, Moderate: 30.4% vs 29.3%). So the probabilities are trustworthy as *averages*.
* **No model output reaches the High band** (max ≈ 51%) because LendingClub had already rejected the riskiest
  applicants - the model never saw them. Rejections therefore come mostly from policy rules.
* **Fairness:** approval rates are nearly equal across US regions (ratio ≥ 0.99). Renters (0.89) and the lowest
  income quartile (0.86) are approved less often, matching their genuinely higher historical default rates, and
  all stay above the 0.8 four-fifths flag.

All numbers come from `models/metrics.json` and are regenerated by `python -m loanrisk.train`.

---

## 6. Hosting it and getting a shareable link

> ⚠️ **Before sharing publicly:** anyone with the link can upload data, and uploads are processed on the host's
> server. Use only synthetic/demo data on a public link, or make the app private (Option B allows an e-mail
> allow-list). On free cloud hosts the SQLite history is wiped whenever the app restarts.

### Option A - Instant temporary link from your own PC (5 minutes, free)
Your laptop keeps running the app; a tunnel gives it a public HTTPS address.
1. Start the app: `.\.venv\Scripts\streamlit.exe run app/streamlit_app.py`
2. Install Cloudflare's tunnel tool once: `winget install --id Cloudflare.cloudflared`
3. In a second terminal: `cloudflared tunnel --url http://localhost:8501`
4. It prints a link like `https://random-words.trycloudflare.com` - share it. It stops working when you close the
   terminal or your PC sleeps. (ngrok works the same way: `ngrok http 8501`.)

### Option B - Streamlit Community Cloud (free, permanent link, recommended)
1. Create a GitHub account and an **empty repository** (e.g. `loan-risk-ai`).
2. In `D:\loan risk` run:
   ```powershell
   git init
   git add .
   git commit -m "Loan risk assessment system"
   git branch -M main
   git remote add origin https://github.com/<your-username>/loan-risk-ai.git
   git push -u origin main
   ```
   `.gitignore` already excludes the 416 MB raw data, the virtual environment and the local database; what is
   pushed (code + trained model + processed data + samples) is about 8 MB.
3. Go to **share.streamlit.io** → sign in with GitHub → **Create app** → choose the repo, branch `main`,
   main file `app/streamlit_app.py`. Under *Advanced settings* choose **Python 3.12**.
4. Deploy. After ~3–5 minutes you get `https://<your-app-name>.streamlit.app` - that is your shareable link.
   In the app's *Settings → Sharing* you can make it private and invite specific e-mails.

### Option C - Hugging Face Spaces (free, Docker)
1. Create a Space at huggingface.co → SDK **Docker** → blank.
2. Push this folder to the Space's git repo (same commands as above, with the Space URL as remote).
3. Add this header at the very top of `README.md` in the Space so it uses Streamlit's port:
   ```
   ---
   title: Loan Risk AI
   sdk: docker
   app_port: 8501
   ---
   ```
4. The included `Dockerfile` builds and runs the app; link: `https://huggingface.co/spaces/<user>/<space>`.
   The same `Dockerfile` also works on Render, Railway, Google Cloud Run or Azure Container Apps.

### Hosting the REST API
Use the same image with a different start command:
`uvicorn api.main:app --host 0.0.0.0 --port 8000` (e.g. a second Render "web service"). Interactive docs are at
`/docs` on that host.

---

## 7. How to optimise it

### 7.1 Better predictions (biggest impact first)
| Idea | Why it helps | How |
|---|---|---|
| **Train on local data** | The biggest gain: Indian applicants differ from US 2007–2015 borrowers. Even a few thousand local loans with outcomes allow re-calibration; tens of thousands allow re-training. | Map your data to the canonical columns (see `data_prep.py`) and run `train.py`. |
| **Use all 676k loans** instead of 150k | More examples, especially rare high-risk ones. | `python -m loanrisk.data_prep --sample 0` (needs ~8 GB RAM; KNN/GB get slow - consider dropping them). |
| **Hyper-parameter tuning** | Current settings are sensible defaults, not tuned. | `RandomizedSearchCV` or **Optuna** over `max_depth`, `learning_rate`, `n_estimators`, `min_child_weight`, `subsample`, using CV ROC-AUC. Expect small gains (≈ +0.005–0.01 AUC is typical on this data - verify, don't assume). |
| **More features** | LendingClub has ~100 more credit-bureau columns (e.g. `num_tl_90g_dpd_24m`, `percent_bc_gt_75`, `acc_open_past_24mths`). | Add them to `data_prep.py`, `schema.py`, `features.py` - but only if you can collect them from applicants too. |
| **Bank-statement / cash-flow features** | Savings, expenses, salary regularity are strong signals but absent from LendingClub. | Requires a dataset that has them with outcomes. |
| **Cost-based threshold** | A missed defaulter costs far more than a rejected good borrower. | Choose band thresholds that minimise (cost_FN × FN + cost_FP × FP) using the test predictions shown on *Model Configuration*. |
| **Remove interest rate** if your rate is set after the risk check | Avoids circular reasoning (the rate already encodes the lender's view). | Remove it from `NUMERIC_FEATURES` and retrain; AUC will drop - that drop is the honest picture. |
| **Ensembling / stacking** | Averaging XGBoost + LR + RF can add a little. | Keep monotonicity/explanations in mind. |
| **Monitoring & retraining** | Borrower behaviour drifts over time. | Log predictions + later outcomes; recheck calibration quarterly. |

### 7.2 Faster app
* Bulk uploads compute SHAP for every row - for 10,000+ rows call `assess(..., explain=False)` and explain only the
  rows the user opens.
* The model and explainer are already cached (`lru_cache`); Streamlit `st.cache_data` is used for metrics.
* KNN and sklearn Gradient Boosting are the slow trainers; drop them from `candidate_models` if retraining often,
  or switch Gradient Boosting to `HistGradientBoostingClassifier` (much faster, supports monotonic constraints).
* For production scale, move from SQLite to PostgreSQL and serve the API with several Uvicorn workers.

---

## 8. Demo dataset for uploading and testing

**File:** `data/sample/demo_applicants.xlsx` (same data in `demo_applicants.csv`), 30 fictional applicants,
generated by `scripts/make_demo_data.py`. The extra **Scenario** column says what each row is designed to test; it is
automatically left unmapped, so the model never sees it.

**How to use:** *Upload Customer Dataset* → drag the file in → check mapping (all fields auto-match) → review the
validation panel → **Run risk assessment** → explore *Bulk Risk Assessment*, *Applicant Results*, *Risk Analytics*,
*Download Reports*.

**Actual results with the current model** (default thresholds and policy):

| ID | Scenario | PD | Band | Conf. | Recommendation |
|---|---|---|---|---|---|
| DEMO001 | Prime salaried, small car loan | 4.6% | Very Low | High | APPROVE |
| DEMO002 | High-income IT professional | 11.3% | Low | High | APPROVE |
| DEMO003 | Doctor, long credit history | 6.0% | Very Low | Medium | APPROVE |
| DEMO004 | Young professional, clean record | 13.4% | Low | High | APPROVE |
| DEMO005 | Teacher, modest loan | 15.8% | Low | High | APPROVE |
| DEMO006 | Average profile, debt consolidation | 21.9% | Low | High | APPROVE |
| DEMO007 | Self-employed shop owner | 17.7% | Low | High | APPROVE |
| DEMO008 | Card refinance, high utilisation | 23.6% | Low | High | APPROVE |
| DEMO009 | Wedding loan, 60 months | 25.7% | Moderate | High | MANUAL REVIEW |
| DEMO010 | Medical emergency, short tenure | 10.2% | Low | Medium | APPROVE |
| DEMO011 | Recent late payments, many enquiries | 43.0% | Moderate | High | MANUAL REVIEW |
| DEMO012 | Previous default on record | 41.3% | Moderate | High | MANUAL REVIEW |
| DEMO013 | High rate, high DTI, short history | 49.9% | Moderate | Medium | MANUAL REVIEW |
| DEMO014 | Small business, volatile income | 48.6% | Moderate | High | MANUAL REVIEW |
| DEMO015 | Moving loan, thin credit file | 29.8% | Moderate | Medium | MANUAL REVIEW |
| DEMO016 | Policy: credit score 520 | 25.3% | Moderate | Low | **REJECT** (policy) |
| DEMO017 | Policy: total EMIs > 60% of income | 18.8% | Low | Low | REVIEW (policy) |
| DEMO018 | Policy: expenses + EMIs > income | 17.9% | Low | High | REVIEW (policy) |
| DEMO019 | Policy: age 71 at maturity | 6.6% | Very Low | High | REVIEW (policy) |
| DEMO020 | Policy: age 19 | 20.6% | Low | Medium | REVIEW (policy) |
| DEMO021 | Out of range: loan ≈ 0.9× annual income | 25.2% | Moderate | Low | MANUAL REVIEW |
| DEMO022 | Out of range: credit score 640 | 19.2% | Low | Low | REVIEW (low confidence) |
| DEMO023 | Out of range: interest 28% | 32.8% | Moderate | Low | MANUAL REVIEW |
| DEMO024 | Missing credit score | - | - | - | CANNOT ASSESS |
| DEMO025 | Missing loan tenure | - | - | - | CANNOT ASSESS |
| DEMO026 | Most optional fields missing | 14.4% | Low | Low | REVIEW (low confidence) |
| DEMO027 | Text formats `₹65,000`, `3 years`, `12.5%` | 17.9% | Low | High | APPROVE |
| DEMO028 | Interest rate 95% (invalid) | - | - | - | CANNOT ASSESS |
| DEMO001-29 | Duplicate ID → suffix added | 4.6% | Very Low | High | APPROVE |
| ROW00030 | Missing ID → generated | 8.6% | Very Low | High | APPROVE |

Totals: 12 approve · 14 manual review · 1 reject · 3 cannot assess. Things to try next: change the band thresholds or
the minimum credit score on *Model Configuration*, click **Re-apply**, and watch the recommendations change while the
probabilities stay the same - that is the model/policy separation in action.

---

## 9. Glossary
* **PD** - probability of default. **DTI** - debt-to-income ratio. **EMI** - equated monthly instalment.
* **Overfitting** - memorising training data instead of learning general patterns (train score ≫ test score).
* **Calibration** - whether a predicted 20% really happens 20% of the time.
* **Threshold** - the probability cut-off that turns a score into a yes/no decision.
* **Imputation** - filling a missing value with a typical value.
* **Monotonic constraint** - forcing a feature's effect to go in one direction only.
