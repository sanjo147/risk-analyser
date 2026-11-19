# Dataset Documentation

## 1. Training data source

| Item | Value |
|---|---|
| Dataset | LendingClub accepted loans, 2007 – 2018Q4 |
| Publisher of this copy | Kaggle dataset `wordsforthewise/lending-club`, file `accepted_2007_to_2018Q4.csv.gz` |
| Original source | LendingClub public loan data (US peer-to-peer lending platform) |
| Raw size | 2,260,701 loans × 151 columns (≈ 392 MB compressed) |
| Outcome | `loan_status` - historical repayment outcome |

**Why this dataset?** Of the suggested public datasets it is the only one that, in one table, has
real historical repayment outcomes **and** a credit score (FICO), income, debt-to-income ratio, loan amount,
tenure, interest rate, purpose, employment length, home ownership, delinquencies, enquiries, utilisation and
credit-history length - i.e. almost all of the inputs the project specifies. German Credit (1,000 rows) has
no credit score or income; the UCI Taiwan dataset is credit-card billing data; Home Credit needs multi-table
joins and uses anonymised external scores.

The raw file is not committed (size). `python -m loanrisk.data_prep` rebuilds the processed file from it.

## 2. Target definition

| `loan_status` | Label |
|---|---|
| Fully Paid; "Does not meet the credit policy. Status:Fully Paid" | 0 = repaid |
| Charged Off; Default; "Does not meet the credit policy. Status:Charged Off" | 1 = defaulted |
| Current, Late, In Grace Period | excluded - outcome not yet known |

## 3. Record selection (leakage and bias controls)

1. Individual applications only (joint applications excluded - different income/DTI definitions).
2. **Outcome maturity:** 36-month loans issued 2007-2015 and 60-month loans issued 2010-2013 only. Every one
   of these had reached its scheduled end date before the 2018Q4 snapshot, so "finished" loans are not
   skewed towards early defaulters or early payoffs, which would happen if recent vintages were included.
3. Rows with no income or credit score removed; DTI values < 0 or > 100 set to missing.
4. **Eligible pool: 676,059 completed loans.** A stratified random sample of **150,000** is used to keep
   training (especially KNN and Gradient Boosting) fast. Default rate: **14.87 %**.

## 4. Columns used and how they map to the applicant form

All post-origination columns (payments received, recoveries, last payment date, outstanding principal,
hardship/settlement flags, …) are **dropped** - they would leak the outcome.

| Applicant field (app) | LendingClub source | Notes |
|---|---|---|
| Credit score | mean of `fico_range_low`, `fico_range_high` | FICO 300-850; LendingClub funded only ≈ 630+ |
| Monthly income | `annual_inc / 12` | self-reported |
| Debt-to-income ratio | `dti` | existing monthly debt payments (excl. mortgage and the new loan) ÷ income |
| Loan amount, tenure, interest rate, EMI | `loan_amnt`, `term`, `int_rate`, `installment` | |
| Loan purpose | `purpose` | `educational` → `education` |
| Residential status | `home_ownership` | OWN/MORTGAGE/RENT/other → Owned/Mortgaged/Rented/Other |
| Years of employment | `emp_length` | "< 1 year" → 0.5, "10+ years" → 10 |
| Active loans / credit lines | `open_acc` | |
| Previous loans / credit lines | `total_acc` | |
| Previous defaults / derogatory records | `pub_rec` | |
| Delayed payments (2 yrs) | `delinq_2yrs` | 30+ days past due |
| Credit inquiries (6 months) | `inq_last_6mths` | |
| Credit utilisation | `revol_util` | |
| Outstanding (revolving) debt | `revol_bal` | |
| Credit history length | `issue_d − earliest_cr_line` | in years, computed at application date |
| Bankruptcies / mortgage accounts | `pub_rec_bankruptcies`, `mort_acc` | |

### Deliberately **not** used as model features

| Column | Reason |
|---|---|
| `grade`, `sub_grade` | LendingClub's own risk model output - not an applicant attribute; using it would just copy their model |
| `addr_state`, `zip_code` | geography can proxy for race/ethnicity - kept **only** for the fairness audit (`audit_region`) |
| `emp_title`, `desc`, `title` | free text, can contain sensitive information |
| any payment / recovery columns | post-origination → target leakage |

Gender, religion, caste, race, marital status and age are not present in LendingClub data and the app
never uses them for scoring (age is used only for a transparent eligibility rule in the policy layer).

### Model features are currency-independent

Training data is in US dollars, while the app is configured for ₹. To avoid mixing currencies, the model
uses **ratios** instead of absolute amounts: loan-to-annual-income, new-EMI-to-income, outstanding-debt-to-income,
DTI, utilisation. Absolute amounts appear only in reports and the policy layer.

### Inputs collected but not used by the model

Age, marital status, dependents, employment type, employer category, employment stability, monthly expenses,
bank balance and savings do not exist in the training data. They are shown in reports and drive
**rule-based policy checks** (affordability, eligibility), clearly labelled as policy, not model.

## 5. Data splits

| Set | Rows | Use |
|---|---|---|
| Training | 90,000 (60 %) | model fitting; 5-fold stratified CV on a 50,000-row subsample |
| Validation | 30,000 (20 %) | probability calibration and model selection |
| Test | 30,000 (20 %) | final evaluation, used once |
| New applicants | uploaded files / form | never used for training; scored only |

All splits are stratified on the default label with `random_state = 42`.

## 6. Sample customer files (`data/sample/`)

`sample_applicants.csv/.xlsx` (250 rows), `sample_applicants_small.csv` (10) and
`sample_applicants_custom_columns.csv` (40, renamed headers for the mapping demo) are **synthetic**, created by
`scripts/make_sample_data.py` with a fixed seed. They are not real people, are not drawn from LendingClub, have no
outcome labels and are never used for training or evaluation. Rows 6, 18 and 24 deliberately lack critical data,
row 32 lacks optional data, row 45 has no EMI (derived automatically) and row 61 repeats an ID - to demonstrate
validation.

## 7. Known limitations

* US P2P borrowers 2007-2015 ≠ Indian retail borrowers today. Re-train or re-calibrate on local outcomes before
  any real use.
* Selection bias: only loans LendingClub approved have outcomes ("reject inference" problem). The model has
  never seen applicants with FICO below ≈ 630; such inputs are flagged as out-of-range and escalated by policy.
* CIBIL (300-900) and FICO (300-850) scales are similar but not identical.
* **Interest rate is the model's most influential input.** LendingClub priced each loan from its internal grade,
  so the rate partly encodes the lender's own risk view. It is kept because the project specifies it as an input,
  but in a deployment where the rate is set *after* the risk assessment it should be removed (to avoid a circular
  dependency) and the model retrained.
* Loan sizes relative to income in LendingClub are small (99.9th percentile ≈ 0.5× annual income). Larger
  requests are outside the training range: the app flags them, lowers confidence and routes them to review rather
  than silently extrapolating.
