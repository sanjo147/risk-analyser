"""Excel and HTML report generation."""
from __future__ import annotations

import html
import io
from datetime import datetime

import numpy as np
import pandas as pd

from .schema import REQUIRED_OUTPUT_COLUMNS

DISCLAIMER = ("Educational decision-support prototype. Probabilities are model estimates learned from historical "
              "LendingClub (US) loans; recommendations follow an illustrative policy and are NOT lending decisions.")

BAND_COLOURS = {
    "Very Low Risk": "#1b7f3b", "Low Risk": "#5fa83a", "Moderate Risk": "#e0a100",
    "High Risk": "#e06400", "Very High Risk": "#c0262d",
}
DECISION_COLOURS = {"APPROVE": "#1b7f3b", "MANUAL REVIEW": "#e0a100", "REJECT RECOMMENDATION": "#c0262d",
                    "CANNOT ASSESS": "#6b7280"}


def money(v, symbol: str = "₹", locale: str = "IN") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    v = float(v)
    neg = v < 0
    n = f"{abs(v):.0f}"
    if locale == "IN" and len(n) > 3:
        head, tail = n[:-3], n[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        n = ",".join(parts) + "," + tail
    elif locale != "IN":
        n = f"{abs(v):,.0f}"
    return f"{'-' if neg else ''}{symbol}{n}"


def output_table(results: pd.DataFrame) -> pd.DataFrame:
    """The required bulk output columns (+ a few useful extras)."""
    out = pd.DataFrame({
        "Applicant ID": results["applicant_id"],
        "Requested Loan": results["loan_amount"],
        "Monthly Income": results["total_monthly_income"],
        "Credit Score": results["credit_score"],
        "DTI Ratio": results["dti_ratio"].round(2),
        "Default Probability": results["default_probability"].round(4),
        "Risk Category": results["risk_category"].fillna("-"),
        "Recommendation": results["recommendation"],
        "Key Risk Factors": results["key_risk_factors"].fillna(""),
        "Confidence": results["confidence"].fillna("-"),
        "Policy / Validation Notes": results["policy_notes"].fillna(""),
    })
    assert list(out.columns[:len(REQUIRED_OUTPUT_COLUMNS)]) == REQUIRED_OUTPUT_COLUMNS
    return out.sort_values("Default Probability", ascending=False, na_position="last")


def portfolio_summary(results: pd.DataFrame) -> pd.DataFrame:
    a = results[results["default_probability"].notna()]
    rec = results["recommendation"].value_counts()
    rows = [
        ("Total applicants in file", len(results)),
        ("Applicants assessed", len(a)),
        ("Could not be assessed (missing critical data)", int((results["recommendation"] == "CANNOT ASSESS").sum())),
        ("Recommended approvals", int(rec.get("APPROVE", 0))),
        ("Manual review", int(rec.get("MANUAL REVIEW", 0))),
        ("Reject recommendations", int(rec.get("REJECT RECOMMENDATION", 0))),
        ("Average predicted default probability", round(float(a["default_probability"].mean()), 4) if len(a) else None),
        ("Total requested amount", float(a["loan_amount"].sum())),
        ("Expected loss proxy (sum of PD x amount)", float((a["default_probability"] * a["loan_amount"]).sum())),
    ]
    for cat, n in a["risk_category"].value_counts().items():
        rows.append((f"Applicants - {cat}", int(n)))
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def bulk_excel(results: pd.DataFrame, issues: pd.DataFrame, meta: dict) -> bytes:
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    buf = io.BytesIO()
    table = output_table(results)
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        table.to_excel(xw, sheet_name="Assessment Results", index=False)
        portfolio_summary(results).to_excel(xw, sheet_name="Portfolio Summary", index=False)
        table[table["Recommendation"] == "REJECT RECOMMENDATION"].to_excel(xw, sheet_name="High Risk", index=False)
        table[table["Recommendation"] == "MANUAL REVIEW"].to_excel(xw, sheet_name="Manual Review", index=False)
        (issues if len(issues) else pd.DataFrame({"message": ["No validation issues."]})).to_excel(
            xw, sheet_name="Validation Issues", index=False)
        pd.DataFrame({"Item": ["Generated", "Model", "Model trained", "Training data", "Disclaimer"],
                      "Value": [datetime.now().isoformat(timespec="seconds"), meta.get("model_name"),
                                meta.get("trained_at"), meta.get("dataset"), DISCLAIMER]}).to_excel(
            xw, sheet_name="About", index=False)
        for ws in xw.book.worksheets:
            for cell in ws[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="1F3A5F")
                cell.alignment = Alignment(wrap_text=True, vertical="center")
            for col in ws.columns:
                width = min(60, max(len(str(c.value or "")) for c in col[:200]) + 2)
                ws.column_dimensions[get_column_letter(col[0].column)].width = max(10, width)
            ws.freeze_panes = "A2"
        for name in ["Assessment Results", "High Risk", "Manual Review"]:
            ws = xw.book[name]
            hdr = [c.value for c in ws[1]]
            ip, ir = hdr.index("Default Probability") + 1, hdr.index("Recommendation") + 1
            for r in range(2, ws.max_row + 1):
                ws.cell(r, ip).number_format = "0.0%"
                for k in ("Requested Loan", "Monthly Income"):
                    ws.cell(r, hdr.index(k) + 1).number_format = "#,##0"
                col = DECISION_COLOURS.get(ws.cell(r, ir).value)
                if col:
                    ws.cell(r, ir).font = Font(bold=True, color=col.lstrip("#"))
    return buf.getvalue()


def gauge_svg(p: float, size: int = 220) -> str:
    """Semicircular risk gauge (inline SVG)."""
    import math
    cx, cy, r = size / 2, size * 0.55, size * 0.4
    segs = [(0, .10, "#1b7f3b"), (.10, .25, "#5fa83a"), (.25, .50, "#e0a100"), (.50, .75, "#e06400"), (.75, 1, "#c0262d")]
    parts = []
    for a, b, c in segs:
        a1, a2 = math.pi * (1 - a), math.pi * (1 - b)
        x1, y1 = cx + r * math.cos(a1), cy - r * math.sin(a1)
        x2, y2 = cx + r * math.cos(a2), cy - r * math.sin(a2)
        parts.append(f'<path d="M{x1:.1f},{y1:.1f} A{r:.1f},{r:.1f} 0 0 1 {x2:.1f},{y2:.1f}" '
                     f'stroke="{c}" stroke-width="{size * 0.09:.0f}" fill="none"/>')
    ang = math.pi * (1 - min(max(p, 0), 1))
    nx, ny = cx + r * 0.85 * math.cos(ang), cy - r * 0.85 * math.sin(ang)
    parts.append(f'<line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}" stroke="#1f2937" stroke-width="4" stroke-linecap="round"/>')
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="7" fill="#1f2937"/>')
    parts.append(f'<text x="{cx}" y="{cy + size * 0.18:.0f}" text-anchor="middle" font-size="{size * 0.13:.0f}" '
                 f'font-weight="700" fill="#1f2937">{p:.1%}</text>')
    return f'<svg viewBox="0 0 {size} {size * 0.78:.0f}" width="{size}" xmlns="http://www.w3.org/2000/svg">{"".join(parts)}</svg>'


def applicant_html(row: pd.Series, detail: dict | None, settings: dict, meta: dict) -> str:
    sym, loc = settings.get("currency_symbol", "₹"), settings.get("currency_locale", "IN")
    e = html.escape
    m = lambda v: money(v, sym, loc)  # noqa: E731
    rec = row.get("recommendation", "")
    if rec == "CANNOT ASSESS" or detail is None:
        body = f"<div class='box bad'><b>Prediction could not be generated.</b><br>{e(str(row.get('policy_notes', '')))}</div>"
        p_html = ""
    else:
        p = float(row["default_probability"])
        li = lambda xs: "".join(f"<li>{e(x)}</li>" for x in xs) or "<li>None identified.</li>"  # noqa: E731
        inc = "".join(f"<li>{e(d['text'])}</li>" for d in detail["increasing"]) or "<li>None.</li>"
        red = "".join(f"<li>{e(d['text'])}</li>" for d in detail["reducing"]) or "<li>None.</li>"
        b = detail["calibration_bin"]
        p_html = gauge_svg(p)
        body = f"""
<div class='grid'>
 <div class='box'><h3>Model prediction</h3>
  <p class='big'>{p:.1%}</p><p>Predicted probability of default</p>
  <p><span class='pill' style='background:{BAND_COLOURS.get(row['risk_category'], '#666')}'>{e(row['risk_category'])}</span></p>
  <p class='small'>Confidence: <b>{e(str(row['confidence']))}</b>. Historically, test-set applicants scored in the same
  range ({b['lo']:.1%}–{b['hi']:.1%}) defaulted at {b['observed']:.1%} (95% CI {b['ci_low']:.1%}–{b['ci_high']:.1%}).
  {detail['imputed']} optional input(s) were missing and filled with typical values.
  {('Outside training range: ' + e(row['ood_features']) + '.') if row.get('ood_features') else ''}</p>
 </div>
 <div class='box'><h3>Lending policy recommendation</h3>
  <p class='big' style='color:{DECISION_COLOURS.get(rec, '#333')}'>{e(rec)}</p>
  <p class='small'>Band default: {e(str(row['model_band_decision']))}. {e(str(row.get('policy_notes') or 'No policy overlay triggered.'))}</p>
 </div>
</div>
<div class='grid'>
 <div class='box'><h3>Positive financial indicators</h3><ul>{li(detail['positive'])}</ul></div>
 <div class='box'><h3>Negative financial indicators</h3><ul>{li(detail['negative'])}</ul></div>
</div>
<div class='grid'>
 <div class='box'><h3>Top factors increasing estimated risk</h3><ul>{inc}</ul></div>
 <div class='box'><h3>Top factors reducing estimated risk</h3><ul>{red}</ul></div>
</div>
<div class='box'><h3>Reason for recommendation</h3><p>{e(detail['reason'])}</p></div>
<div class='box'><h3>Suggestions for improving creditworthiness</h3><ul>{li(detail['suggestions'])}</ul></div>
<p class='small'>Factor contributions show how the model uses each input (SHAP values); they reflect patterns in historical
data and are not proof of cause and effect.</p>"""
    dti = row.get("dti_ratio")
    summary = [
        ("Requested loan", m(row.get("loan_amount"))), ("Tenure", f"{row.get('loan_tenure_months', 0):.0f} months"),
        ("Interest rate", f"{row.get('interest_rate', float('nan')):.2f}%"), ("Estimated EMI", m(row.get("expected_emi"))),
        ("Monthly income (incl. additional)", m(row.get("total_monthly_income"))),
        ("Existing EMI", m(row.get("existing_emi"))), ("Monthly expenses", m(row.get("monthly_expenses"))),
        ("Debt-to-income ratio", f"{dti:.1f}%" if pd.notna(dti) else "-"),
        ("Credit score", f"{row.get('credit_score', float('nan')):.0f}"),
        ("Disposable income after all EMIs", m(row.get("disposable_income"))),
        ("Savings", m(row.get("savings"))), ("Employment", f"{row.get('employment_type') or '-'}"),
    ]
    trs = "".join(f"<tr><td>{e(k)}</td><td><b>{e(str(v))}</b></td></tr>" for k, v in summary)
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>Assessment {e(str(row['applicant_id']))}</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;color:#1f2937;max-width:960px;margin:24px auto;padding:0 16px}}
h1{{color:#1f3a5f;margin-bottom:0}} h3{{margin:0 0 8px;color:#1f3a5f}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:14px 0}}
.box{{border:1px solid #d6dde6;border-radius:10px;padding:14px;margin:14px 0}} .grid .box{{margin:0}}
.big{{font-size:28px;font-weight:700;margin:4px 0}} .small{{font-size:12px;color:#4b5563}}
.pill{{color:#fff;padding:4px 10px;border-radius:12px;font-weight:600}}
table{{border-collapse:collapse;width:100%}} td{{padding:5px 8px;border-bottom:1px solid #eef1f5}}
.bad{{background:#fdecec}} .head{{display:flex;justify-content:space-between;align-items:center}}
@media(max-width:700px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body>
<div class='head'><div><h1>Loan Risk Assessment</h1><p>Applicant ID: <b>{e(str(row['applicant_id']))}</b> ·
Generated {datetime.now():%d %b %Y %H:%M} · Model: {e(str(meta.get('model_name')))}</p></div>{p_html}</div>
<div class='box'><h3>Applicant financial summary</h3><table>{trs}</table></div>
{body}
<p class='small'><b>Disclaimer:</b> {e(DISCLAIMER)}</p></body></html>"""
