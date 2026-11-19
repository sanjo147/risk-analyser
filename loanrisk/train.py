"""Train, compare, calibrate and evaluate the default-risk models.

Data split (stratified, random_state=42):
    train 60%  -> model fitting + 5-fold stratified cross-validation
    valid 20%  -> probability calibration (sigmoid / Platt) and model selection
    test  20%  -> touched once, for the final unbiased evaluation
New applicants uploaded in the app are never part of any of these sets.

Run:  python -m loanrisk.train
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV, calibration_curve  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.frozen import FrozenEstimator  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from .config import (  # noqa: E402
    DEFAULT_SETTINGS, FIG_DIR, MODEL_DIR, PROCESSED_DIR, RANDOM_STATE, REPORT_DIR,
)
from .explain import ModelExplainer  # noqa: E402
from .features import build_model_features  # noqa: E402
from .preprocess import make_preprocessor  # noqa: E402
from .risk import approve_threshold, categorize, wilson_ci  # noqa: E402
from .schema import CATEGORICAL_FEATURES, FEATURE_LABELS, MODEL_FEATURES, NUMERIC_FEATURES  # noqa: E402

CV_SUBSAMPLE = 50_000
CONSTRAINED_MODEL = "XGBoost"
SELECTION_TOLERANCE = 0.005


# Monotonic constraints for XGBoost (standard practice in credit scoring): the model may only learn
# relationships in the economically expected direction, which keeps explanations coherent.
# +1 = higher value can only raise risk, -1 = can only lower it, 0 = unconstrained.
MONOTONE = {
    "credit_score": -1, "interest_rate": 1, "loan_tenure_months": 1, "dti_ratio": 1, "loan_to_income": 1,
    "emi_to_income": 1, "credit_utilization": 1, "credit_history_years": -1, "delayed_payments_2y": 1,
    "recent_inquiries": 1, "previous_defaults": 1, "bankruptcies": 1,
}


def monotone_vector() -> tuple:
    # Numeric features come first in the transformed matrix; the trailing one-hot columns (whose count can
    # vary between CV folds) are left unconstrained, which XGBoost allows for a shorter vector.
    return tuple(MONOTONE.get(f, 0) for f in NUMERIC_FEATURES)


def candidate_models(pos_weight: float) -> dict:
    return {
        "Logistic Regression": (LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced"), True, False),
        "Decision Tree": (DecisionTreeClassifier(max_depth=7, min_samples_leaf=200, class_weight="balanced",
                                                 random_state=RANDOM_STATE), False, False),
        "Random Forest": (RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=40,
                                                 max_features="sqrt", class_weight="balanced_subsample",
                                                 n_jobs=-1, random_state=RANDOM_STATE), False, False),
        "XGBoost": (XGBClassifier(n_estimators=500, learning_rate=0.04, max_depth=4, subsample=0.8,
                                  colsample_bytree=0.8, min_child_weight=20, reg_lambda=2.0,
                                  scale_pos_weight=pos_weight, tree_method="hist", n_jobs=-1,
                                  monotone_constraints=monotone_vector(),
                                  eval_metric="logloss", random_state=RANDOM_STATE), False, False),
        "KNN": (KNeighborsClassifier(n_neighbors=151, weights="uniform", n_jobs=-1), True, False),
        "Gradient Boosting": (GradientBoostingClassifier(n_estimators=250, learning_rate=0.06, max_depth=3,
                                                         subsample=0.8, random_state=RANDOM_STATE), False, True),
    }


def build_pipeline(model, scale: bool) -> Pipeline:
    return Pipeline([("pre", make_preprocessor(scale=scale)), ("model", model)])


def threshold_metrics(y, p, t: float) -> dict:
    pred = (p > t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(threshold=t, accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred), f1=f1_score(y, pred), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                false_negative_rate=fn / max(fn + tp, 1), false_positive_rate=fp / max(fp + tn, 1))


def bootstrap_auc(y, p, n=300, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    t0 = time.time()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(PROCESSED_DIR / "lendingclub_canonical.csv.gz")
    X = build_model_features(data)
    y = data["default"].values

    idx = np.arange(len(data))
    idx_train, idx_tmp = train_test_split(idx, test_size=0.4, stratify=y, random_state=RANDOM_STATE)
    idx_val, idx_test = train_test_split(idx_tmp, test_size=0.5, stratify=y[idx_tmp], random_state=RANDOM_STATE)
    assert not (set(idx_train) & set(idx_test)) and not (set(idx_val) & set(idx_test))
    Xtr, Xva, Xte = X.iloc[idx_train], X.iloc[idx_val], X.iloc[idx_test]
    ytr, yva, yte = y[idx_train], y[idx_val], y[idx_test]
    print(f"train={len(ytr):,} valid={len(yva):,} test={len(yte):,}  default rate train={ytr.mean():.3%}")
    pos_weight = float((ytr == 0).sum() / (ytr == 1).sum())

    # CV subsample of the training set only
    cv_idx, _ = train_test_split(np.arange(len(ytr)), train_size=min(CV_SUBSAMPLE, len(ytr) - 1),
                                 stratify=ytr, random_state=RANDOM_STATE)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results, fitted = {}, {}
    for name, (model, scale, needs_sw) in candidate_models(pos_weight).items():
        ts = time.time()
        pipe = build_pipeline(model, scale)
        params = {"model__sample_weight": compute_sample_weight("balanced", ytr[cv_idx])} if needs_sw else None
        cv = cross_validate(pipe, Xtr.iloc[cv_idx], ytr[cv_idx], cv=skf, n_jobs=1, params=params,
                            scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"})
        fit_params = {"model__sample_weight": compute_sample_weight("balanced", ytr)} if needs_sw else {}
        pipe.fit(Xtr, ytr, **fit_params)
        cal = CalibratedClassifierCV(FrozenEstimator(pipe), method="sigmoid").fit(Xva, yva)
        # Train AUC on a subsample (for KNN, train-set predictions are expensive)
        p_tr = pipe.predict_proba(Xtr.iloc[cv_idx[:15000]])[:, 1]
        p_va_raw = pipe.predict_proba(Xva)[:, 1]
        p_va = cal.predict_proba(Xva)[:, 1]
        results[name] = dict(
            cv_roc_auc_mean=float(cv["test_roc_auc"].mean()), cv_roc_auc_std=float(cv["test_roc_auc"].std()),
            cv_pr_auc_mean=float(cv["test_pr_auc"].mean()), cv_pr_auc_std=float(cv["test_pr_auc"].std()),
            train_roc_auc=float(roc_auc_score(ytr[cv_idx[:15000]], p_tr)),
            val_roc_auc=float(roc_auc_score(yva, p_va)), val_pr_auc=float(average_precision_score(yva, p_va)),
            val_brier_uncalibrated=float(brier_score_loss(yva, p_va_raw)),
            val_brier_calibrated=float(brier_score_loss(yva, p_va)),
            fit_seconds=round(time.time() - ts, 1),
        )
        results[name]["generalization_gap"] = results[name]["train_roc_auc"] - results[name]["val_roc_auc"]
        fitted[name] = (pipe, cal)
        print(f"{name:20s} CV AUC {results[name]['cv_roc_auc_mean']:.4f}±{results[name]['cv_roc_auc_std']:.4f} "
              f"val AUC {results[name]['val_roc_auc']:.4f} Brier(cal) {results[name]['val_brier_calibrated']:.4f} "
              f"gap {results[name]['generalization_gap']:+.4f} ({results[name]['fit_seconds']}s)")

    # Model selection on CV + validation data only (the test set is not consulted).
    def score(n):
        r = results[n]
        return (round((r["cv_roc_auc_mean"] + r["val_roc_auc"]) / 2, 3), -r["val_brier_calibrated"])
    best = max(results, key=score)
    # Prefer the monotonic-constrained model when it is statistically indistinguishable from the best
    # (within SELECTION_TOLERANCE AUC, about one CV standard deviation): coherent explanations matter in credit.
    mean_auc = {n: (r["cv_roc_auc_mean"] + r["val_roc_auc"]) / 2 for n, r in results.items()}
    if best != CONSTRAINED_MODEL and mean_auc[best] - mean_auc[CONSTRAINED_MODEL] <= SELECTION_TOLERANCE:
        print(f"{best} leads by {mean_auc[best] - mean_auc[CONSTRAINED_MODEL]:.4f} AUC (<= {SELECTION_TOLERANCE}); "
              f"preferring monotonic-constrained {CONSTRAINED_MODEL}")
        best = CONSTRAINED_MODEL
    print(f"selected: {best}")

    # ------------------------------------------------------------- final test evaluation
    bands = DEFAULT_SETTINGS["risk_bands"]
    t_policy = approve_threshold(bands)
    test_eval = {}
    test_probs = {}
    for name, (pipe, cal) in fitted.items():
        p = cal.predict_proba(Xte)[:, 1]
        test_probs[name] = p
        test_eval[name] = dict(
            roc_auc=float(roc_auc_score(yte, p)), pr_auc=float(average_precision_score(yte, p)),
            brier=float(brier_score_loss(yte, p)),
            at_0_5=threshold_metrics(yte, p, 0.5), at_policy=threshold_metrics(yte, p, t_policy),
        )
    pb = test_probs[best]
    best_pipe, best_cal = fitted[best]
    auc_ci = bootstrap_auc(yte, pb)
    raw_brier = float(brier_score_loss(yte, best_pipe.predict_proba(Xte)[:, 1]))

    # Calibration bins (also used for per-applicant uncertainty in the app)
    edges = np.unique(np.quantile(pb, np.linspace(0, 1, 21)))
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pb >= lo) & (pb <= hi) if hi == edges[-1] else (pb >= lo) & (pb < hi)
        k, n = int(yte[m].sum()), int(m.sum())
        ci = wilson_ci(k, n)
        bins.append(dict(lo=float(lo), hi=float(hi), n=n, defaults=k, observed=k / max(n, 1),
                         predicted=float(pb[m].mean()) if n else np.nan, ci_low=ci[0], ci_high=ci[1]))

    # Test predictions table (enables threshold re-calibration in the UI without retraining)
    test_df = data.iloc[idx_test][["applicant_id", "residential_status", "employment_years", "monthly_income",
                                   "credit_score", "loan_amount", "audit_region"]].copy()
    test_df["y_true"] = yte
    test_df["prob"] = pb
    test_df.to_csv(MODEL_DIR / "test_predictions.csv.gz", index=False)

    # ------------------------------------------------------------- explainability
    bg = Xtr.sample(500, random_state=RANDOM_STATE)
    explainer = ModelExplainer(best_pipe, bg)
    shap_sample = Xte.sample(2000, random_state=RANDOM_STATE)
    contrib = explainer.contributions(shap_sample)
    global_imp = contrib.abs().mean().sort_values(ascending=False)
    direction = {f: float(np.corrcoef(pd.to_numeric(shap_sample[f], errors="coerce").fillna(
        shap_sample[f].median() if f in NUMERIC_FEATURES else 0), contrib[f])[0, 1])
        if f in NUMERIC_FEATURES and shap_sample[f].nunique() > 1 else None for f in MODEL_FEATURES}

    # ------------------------------------------------------------- fairness audit (test set)
    cats, decs = categorize(pb, bands)
    approved = decs == "APPROVE"
    tdf = test_df.assign(approved=approved)
    tdf["income_band"] = pd.qcut(tdf["monthly_income"], 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])
    tdf["employment_band"] = pd.cut(tdf["employment_years"], [-1, 1.9, 4.9, 9.9, 100],
                                    labels=["<2 yrs", "2-4 yrs", "5-9 yrs", "10+ yrs"])
    fairness = {}
    for group_col, title in [("audit_region", "US region (audit only - not a model feature)"),
                             ("residential_status", "Residential status"),
                             ("income_band", "Income quartile"), ("employment_band", "Employment length")]:
        rows = []
        for g, sub in tdf.groupby(group_col, observed=True):
            if len(sub) < 100:
                continue
            d = sub["y_true"] == 1
            rows.append(dict(group=str(g), n=int(len(sub)), actual_default_rate=float(d.mean()),
                             mean_predicted=float(sub["prob"].mean()), approval_rate=float(sub["approved"].mean()),
                             fnr=float((sub["approved"] & d).sum() / max(d.sum(), 1)),
                             fpr=float((~sub["approved"] & ~d).sum() / max((~d).sum(), 1)),
                             roc_auc=float(roc_auc_score(sub["y_true"], sub["prob"])) if d.any() and (~d).any() else None))
        max_appr = max(r["approval_rate"] for r in rows)
        for r in rows:
            r["approval_ratio_vs_max"] = r["approval_rate"] / max_appr if max_appr else None
        fairness[group_col] = dict(title=title, rows=rows)

    # ------------------------------------------------------------- figures
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for name, p in test_probs.items():
        fpr, tpr, _ = roc_curve(yte, p)
        ax[0].plot(fpr, tpr, label=f"{name} (AUC {test_eval[name]['roc_auc']:.3f})", lw=2 if name == best else 1)
        pr, rc, _ = precision_recall_curve(yte, p)
        ax[1].plot(rc, pr, label=f"{name} (AP {test_eval[name]['pr_auc']:.3f})", lw=2 if name == best else 1)
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set(title="ROC curve - test set", xlabel="False positive rate", ylabel="True positive rate")
    ax[1].axhline(yte.mean(), color="k", ls="--", lw=0.8)
    ax[1].set(title="Precision-Recall curve - test set", xlabel="Recall", ylabel="Precision")
    ax[0].legend(fontsize=8)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_pr_curves.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    fx, fy = calibration_curve(yte, best_pipe.predict_proba(Xte)[:, 1], n_bins=15, strategy="quantile")
    cx, cy = calibration_curve(yte, pb, n_bins=15, strategy="quantile")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")
    ax.plot(fy, fx, "o-", label="Before calibration (class-weighted)")
    ax.plot(cy, cx, "s-", label="After sigmoid calibration")
    ax.set(title=f"Calibration curve - {best} (test)", xlabel="Predicted default probability",
           ylabel="Observed default rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "calibration_curve.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for a, key, title in [(axes[0], "at_0_5", "threshold 0.50"), (axes[1], "at_policy", f"policy threshold {t_policy:.2f}")]:
        m = test_eval[best][key]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=a,
                    xticklabels=["Pred: repaid", "Pred: default"], yticklabels=["Actual: repaid", "Actual: default"])
        a.set_title(f"{best} - {title}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    gi = global_imp.rename(index=FEATURE_LABELS)[::-1]
    ax.barh(gi.index, gi.values, color="#1f4e79")
    ax.set(title=f"Global feature importance - mean |SHAP| ({best})", xlabel="Mean absolute contribution (log-odds)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(x=pb, hue=np.where(yte == 1, "Defaulted", "Repaid"), bins=50, stat="density",
                 common_norm=False, element="step", ax=ax)
    ax.set(title="Predicted default probability by actual outcome (test)", xlabel="Predicted probability")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "score_distribution.png", dpi=130)
    plt.close(fig)

    # ------------------------------------------------------------- persist
    stats = {
        "p01": Xtr[NUMERIC_FEATURES].quantile(0.01).to_dict(),
        "p99": Xtr[NUMERIC_FEATURES].quantile(0.99).to_dict(),
        "lo": Xtr[NUMERIC_FEATURES].quantile(0.001).to_dict(),  # range used for out-of-distribution flags
        "hi": Xtr[NUMERIC_FEATURES].quantile(0.999).to_dict(),
        "median": Xtr[NUMERIC_FEATURES].median().to_dict(),
        "categories": {c: sorted(Xtr[c].dropna().unique().tolist()) for c in CATEGORICAL_FEATURES},
    }
    meta = dict(
        model_name=best, trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        dataset="LendingClub accepted loans 2007-2018Q4 (Kaggle: wordsforthewise/lending-club)",
        n_train=int(len(ytr)), n_valid=int(len(yva)), n_test=int(len(yte)),
        n_eligible=None, default_rate=float(y.mean()), features=MODEL_FEATURES, random_state=RANDOM_STATE,
        calibration="sigmoid (Platt) on validation set",
    )
    bundle = dict(pipeline=best_pipe, calibrated=best_cal, meta=meta, stats=stats, calibration_bins=bins,
                  background=bg, global_importance=global_imp.to_dict())
    joblib.dump(bundle, MODEL_DIR / "model.joblib", compress=3)
    metrics = dict(meta=meta, selection_rule="highest mean(CV ROC-AUC, validation ROC-AUC) rounded to 3dp, ties "
                   "broken by lower calibrated validation Brier score; the monotonic-constrained XGBoost is preferred "
                   f"if it is within {SELECTION_TOLERANCE} AUC of the leader (about one CV standard deviation), "
                   "because its explanations are guaranteed to move in the economically expected direction",
                   comparison=results, test=test_eval, best=best, test_auc_95ci=auc_ci,
                   test_brier_uncalibrated=raw_brier, policy_threshold=t_policy, calibration_bins=bins,
                   global_importance=global_imp.to_dict(), shap_direction=direction, fairness=fairness,
                   runtime_seconds=round(time.time() - t0, 1))
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    write_report(metrics)
    print(f"done in {time.time() - t0:.0f}s -> {MODEL_DIR / 'model.joblib'}")


def write_report(m: dict) -> None:
    best = m["best"]
    t = m["test"][best]
    lines = [
        "# Model Evaluation Report", "",
        f"*Generated {m['meta']['trained_at']} by `python -m loanrisk.train`. All numbers below are computed, not typed.*", "",
        "## Data split", "",
        f"- Training: {m['meta']['n_train']:,} loans (model fitting and 5-fold stratified CV on a {CV_SUBSAMPLE:,}-loan subsample)",
        f"- Validation: {m['meta']['n_valid']:,} loans (sigmoid probability calibration and model selection)",
        f"- Test: {m['meta']['n_test']:,} loans (held out; used once for the numbers in this report)",
        f"- Overall default rate: {m['meta']['default_rate']:.2%}", "",
        "## Model comparison", "",
        "| Model | CV ROC-AUC | CV PR-AUC | Train AUC | Valid AUC | Gap | Valid Brier (raw → calibrated) | Test ROC-AUC | Test PR-AUC | Test Brier |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in m["comparison"].items():
        te = m["test"][name]
        mark = (" **(selected)**" if name == best else "") + (" (monotonic)" if name == CONSTRAINED_MODEL else "")
        lines.append(f"| {name}{mark} | {r['cv_roc_auc_mean']:.4f} ± {r['cv_roc_auc_std']:.4f} | "
                     f"{r['cv_pr_auc_mean']:.4f} | {r['train_roc_auc']:.4f} | {r['val_roc_auc']:.4f} | "
                     f"{r['generalization_gap']:+.4f} | {r['val_brier_uncalibrated']:.4f} → {r['val_brier_calibrated']:.4f} | "
                     f"{te['roc_auc']:.4f} | {te['pr_auc']:.4f} | {te['brier']:.4f} |")
    lines += ["", f"Selection rule: {m['selection_rule']}.", "",
              f"## Selected model: {best} (test set)", "",
              f"- ROC-AUC: **{t['roc_auc']:.4f}** (95% bootstrap CI {m['test_auc_95ci'][0]:.4f} – {m['test_auc_95ci'][1]:.4f})",
              f"- PR-AUC: **{t['pr_auc']:.4f}** (baseline = default rate {m['meta']['default_rate']:.4f})",
              f"- Brier score: **{t['brier']:.4f}** (before calibration {m['test_brier_uncalibrated']:.4f})", ""]
    for key, label in [("at_policy", f"Policy threshold (probability > {m['policy_threshold']:.2f} = not auto-approved)"),
                       ("at_0_5", "Threshold 0.50")]:
        x = t[key]
        lines += [f"### {label}", "",
                  "| Accuracy | Precision | Recall | F1 | False-negative rate | False-positive rate |",
                  "|---|---|---|---|---|---|",
                  f"| {x['accuracy']:.4f} | {x['precision']:.4f} | {x['recall']:.4f} | {x['f1']:.4f} | "
                  f"{x['false_negative_rate']:.4f} | {x['false_positive_rate']:.4f} |", "",
                  f"Confusion matrix: TN={x['tn']:,}  FP={x['fp']:,}  **FN={x['fn']:,}**  TP={x['tp']:,}", ""]
    lines += [
        "**False negatives** are actual defaulters the model scored as safe. At the 0.50 threshold most defaulters "
        "are missed, which is why the decision policy routes everything above the approve threshold to review or rejection; "
        "accuracy alone is misleading on an imbalanced dataset (a model that approves everyone would score "
        f"{1 - m['meta']['default_rate']:.1%} accuracy).", "",
        "![ROC and PR](figures/roc_pr_curves.png)", "", "![Calibration](figures/calibration_curve.png)", "",
        "![Confusion](figures/confusion_matrix.png)", "", "![Scores](figures/score_distribution.png)", "",
        "## Calibration by risk decile (test)", "", "| Predicted range | n | Mean predicted | Observed default rate (95% CI) |",
        "|---|---|---|---|",
    ]
    for b in m["calibration_bins"]:
        lines.append(f"| {b['lo']:.3f} – {b['hi']:.3f} | {b['n']:,} | {b['predicted']:.3f} | "
                     f"{b['observed']:.3f} ({b['ci_low']:.3f} – {b['ci_high']:.3f}) |")
    lines += ["", "## Global feature importance (mean |SHAP|)", "",
              "Importance shows how strongly the model *relies* on a feature; it is an association learned from "
              "historical data, not proof that the feature causes default.", "", "![Importance](figures/feature_importance.png)", ""]
    for f, v in m["global_importance"].items():
        lines.append(f"- {FEATURE_LABELS.get(f, f)}: {v:.4f}")
    lines += ["", "## Fairness audit (test set, default policy thresholds)", "",
              "Protected attributes (gender, religion, caste, race, age, marital status) are not in the training data "
              "and are never used as features. Geography is excluded from the model because it can proxy for protected "
              "characteristics; it is used below only to audit outcomes. Approval ratio < 0.8 (the 'four-fifths rule') "
              "flags a potential disparate impact worth investigating.", ""]
    for g in m["fairness"].values():
        lines += [f"### {g['title']}", "", "| Group | n | Actual default | Mean predicted | Approval rate | "
                  "Approval ratio | FNR | FPR | AUC |", "|---|---|---|---|---|---|---|---|---|"]
        for r in g["rows"]:
            auc = f"{r['roc_auc']:.3f}" if r["roc_auc"] is not None else "-"
            lines.append(f"| {r['group']} | {r['n']:,} | {r['actual_default_rate']:.3f} | {r['mean_predicted']:.3f} | "
                         f"{r['approval_rate']:.3f} | {r['approval_ratio_vs_max']:.2f} | {r['fnr']:.3f} | {r['fpr']:.3f} | {auc} |")
        lines.append("")
    lines += ["## Limitations", "",
              "- Trained on US peer-to-peer loans (LendingClub, 2007-2015 originations). Applicants in other markets "
              "(e.g. Indian retail lending) may behave differently; the model should be re-trained or at least "
              "re-calibrated on local outcome data before any real use.",
              "- Credit scores in training are FICO (approx. 630-850 because LendingClub declined lower scores). Scores "
              "below that range are extrapolation; the app flags them and the policy layer escalates them.",
              "- Only approved-and-funded loans have outcomes (reject inference / selection bias).",
              "- Interest rate is the most influential input; LendingClub priced loans from its own grade, so the rate "
              "partly encodes the lender's risk view. Remove it and retrain if rates are set after the risk assessment.",
              "- Inputs such as savings, bank balance, monthly expenses, dependents and employment type are not in the "
              "training data, so they are used only in the rule-based affordability policy, never by the model.",
              "- This is an educational decision-support prototype; outputs are recommendations, not lending decisions."]
    (REPORT_DIR / "MODEL_EVALUATION.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
