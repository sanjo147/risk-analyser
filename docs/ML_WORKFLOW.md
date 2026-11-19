# Machine Learning Workflow

```
raw LendingClub CSV ──► data_prep.py ──► canonical applicant table (+ label)
                                              │
uploaded file / form ──► column mapping ──►  │  features.clean_applicants  (types, ranges, IDs, critical-field check)
                                              ▼
                                   features.build_model_features   ◄── same function for training and inference
                                              ▼
              preprocess pipeline: winsorise (0.5/99.5 pct) → median impute → scale │ mode impute → one-hot
                                              ▼
                           classifier (6 candidates) → sigmoid calibration
                                              ▼
           probability ──► risk band (configurable) ──► band decision ──► policy overlays ──► recommendation
                    └──► SHAP contributions ──► business-language explanation
```

## 1. Data preparation (`loanrisk/data_prep.py`)
Selects completed, matured loans, maps LendingClub columns to the canonical applicant schema and drops every
post-origination column. Details and reasoning: [DATASET.md](DATASET.md).

## 2. Validation and cleaning (`loanrisk/features.py`)
* Column auto-mapping by name, alias and fuzzy match; the user can override every mapping in the UI.
* Columns that look like protected attributes are flagged and never used.
* Text such as `₹45,000`, `12.5%`, `3 years` is parsed; out-of-range values (e.g. interest 95 %) are set to
  missing and reported.
* **Critical fields** - credit score, monthly income, loan amount, tenure, interest rate. If any is missing the
  row gets `CANNOT ASSESS` with the reason; no prediction is invented.
* Optional fields that are missing are imputed **inside the fitted pipeline** with training medians, reported
  to the user, and lower the confidence rating.
* EMI is derived from amount/rate/tenure if absent; DTI from existing EMI ÷ income if absent.
* Duplicate IDs are suffixed and reported.

## 3. Features
17 numeric + 2 categorical features (see `schema.py`). Absolute money amounts are converted into ratios
(loan-to-income, EMI-to-income, debt-to-income) so a model trained in USD can score applicants in any currency.

## 4. Preventing data leakage
* Stratified split into train / validation / test **before** anything is fitted.
* All preprocessing statistics (clip limits, medians, scaler, encoder) are learned inside a `Pipeline` fitted on
  the training set only; cross-validation re-fits them in every fold.
* Calibration uses only the validation set; the test set is scored once at the end.
* No post-origination variables, no lender grade.

## 5. Class imbalance
Defaults are ~15 %. Models are trained with class weighting (`class_weight="balanced"`, `scale_pos_weight` for
XGBoost, balanced sample weights for Gradient Boosting; KNN has no weighting option). Weighting inflates raw
probabilities, so every model is **re-calibrated** on the validation set; the calibrated Brier score and the
calibration curve confirm that the published probabilities match observed default rates. Threshold-dependent
metrics are reported at the policy threshold, not only at 0.5.

## 6. Models compared
Logistic Regression, Decision Tree, Random Forest, XGBoost, KNN, Gradient Boosting - each with 5-fold stratified
CV (ROC-AUC, PR-AUC), train AUC (to measure over-fitting), validation AUC, and raw vs calibrated Brier score.
XGBoost is trained with **monotonic constraints** (standard in credit scoring): higher credit score and longer credit history can only lower risk; higher interest rate, tenure, DTI, loan-to-income, EMI-to-income, utilisation, delinquencies, enquiries, defaults and bankruptcies can only raise it. This prevents counter-intuitive effects caused by correlated features and keeps explanations coherent.
The selection rule (in `train.py`) uses only CV and validation results. Test results for all six are reported
for transparency. See [../reports/MODEL_EVALUATION.md](../reports/MODEL_EVALUATION.md).

## 7. From probability to recommendation
1. **Model prediction** - calibrated probability of default, 0-100 %.
2. **Risk category** - configurable bands (default 10 / 25 / 50 / 75 %).
3. **Band decision** - Very Low/Low → APPROVE, Moderate → MANUAL REVIEW, High/Very High → REJECT RECOMMENDATION.
4. **Policy overlays** - rules that can only escalate: minimum credit score, total EMI ≤ 60 % of income,
   non-negative disposable income, age eligibility, low-confidence/out-of-range data → review.

The UI and every report label these layers separately ("Model prediction" vs "Lending policy").

## 8. Confidence and uncertainty
For each applicant the app reports:
* the observed default rate (with 95 % Wilson interval) of held-out test loans with similar predicted
  probability - a direct check of calibration;
* which inputs are outside the 1st-99th percentile range of the training data (extrapolation);
* how many optional inputs were imputed.
These combine into a confidence rating: Low if a key input (credit score, interest rate, DTI, loan- or EMI-to-income) is beyond the training range or 5+ inputs were imputed; Medium if any other input is out of range, 3+ were imputed or the interval is wider than 10 points; otherwise High. Low confidence never auto-approves.

## 9. Explainability
SHAP `TreeExplainer` (or `LinearExplainer` / permutation explainer for other model types) on the uncalibrated
model in log-odds space - calibration is monotonic, so factor rankings carry over. One-hot columns are summed back
to their original feature. The app shows top-5 increasing / reducing factors in plain language, the full
contribution chart, global mean |SHAP| importance, and an explicit warning that importance ≠ causation.
Positive/negative *financial indicators* in reports are separate, transparent rules on the raw data.

## 10. Fairness evaluation
On the test set: approval rate, approval ratio vs the best-treated group (four-fifths rule), false-negative and
false-positive rates and AUC by US region (audit-only variable), residential status, income quartile and
employment length. Results are in the Model Performance page and the evaluation report.

## 11. Security and privacy
Applicant data stays on the local machine (SQLite file `data/assessments.db`); the app has a "delete all saved
data" control; uploads are processed in memory; nothing is sent to external services.

## 12. Retraining
```
python -m loanrisk.data_prep --sample 150000
python -m loanrisk.train
```
Both steps are deterministic (`random_state = 42`).
