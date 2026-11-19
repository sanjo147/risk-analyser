import io

import pandas as pd
import streamlit as st

import common
from loanrisk.config import SAMPLE_DIR
from loanrisk.features import apply_mapping, clean_applicants, prohibited_columns, suggest_mapping
from loanrisk.schema import CRITICAL_FIELDS, FIELD_MAP, FIELDS

st.title("📤 Upload Customer Dataset")
st.caption("Drag and drop an Excel (.xlsx) or CSV file. Columns are matched automatically and can be re-mapped below.")

c1, c2 = st.columns([2, 1])
up = c1.file_uploader("Applicant file", type=["csv", "xlsx", "xls"], accept_multiple_files=False)
with c2:
    st.markdown("**Need a starting point?**")
    tmpl = pd.DataFrame(columns=[f.label for f in FIELDS])
    st.download_button("Blank template (CSV)", tmpl.to_csv(index=False).encode(), "applicant_template.csv")
    st.download_button("Synthetic sample (250 applicants, XLSX)", (SAMPLE_DIR / "sample_applicants.xlsx").read_bytes(),
                       "sample_applicants.xlsx")
    st.download_button("Sample with custom column names (CSV)",
                       (SAMPLE_DIR / "sample_applicants_custom_columns.csv").read_bytes(),
                       "sample_applicants_custom_columns.csv")


@st.cache_data(show_spinner=False)
def read_file(name: str, data: bytes) -> pd.DataFrame:
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data))
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode the CSV file.")


if up is None:
    st.info("Upload a file to begin. Required (critical) fields: "
            + ", ".join(FIELD_MAP[c].label for c in CRITICAL_FIELDS) + ".")
    st.stop()

try:
    df = read_file(up.name, up.getvalue())
except Exception as e:
    st.error(f"The file could not be read: {e}. Please upload a valid .csv or .xlsx file.")
    st.stop()
if df.empty:
    st.error("The file contains no rows.")
    st.stop()
df.columns = [str(c).strip() for c in df.columns]
st.success(f"Read **{len(df):,} rows × {len(df.columns)} columns** from `{up.name}`.")
with st.expander("Preview (first 20 rows)"):
    st.dataframe(df.head(20), width="stretch")

bad = prohibited_columns(df.columns)
if bad:
    st.warning("These columns look like protected / sensitive attributes and will **never** be used by the model: "
               + ", ".join(f"`{b}`" for b in bad) + ". Age is used only for the eligibility rule, never for scoring.")

st.subheader("Column mapping")
st.caption("Map each uploaded column to a system field, or leave it as *(ignore)*. Fields marked ★ are critical.")
auto = suggest_mapping(list(df.columns))
options = ["(ignore)"] + [f.name for f in FIELDS]
fmt = lambda n: n if n == "(ignore)" else ("★ " if FIELD_MAP[n].critical else "") + FIELD_MAP[n].label  # noqa: E731
mapping = {}
cols = st.columns(3)
for i, col in enumerate(df.columns):
    default = auto.get(col) or "(ignore)"
    sel = cols[i % 3].selectbox(f"`{col}`", options, index=options.index(default), format_func=fmt, key=f"map_{up.name}_{col}")
    mapping[col] = None if sel == "(ignore)" else sel

targets = [v for v in mapping.values() if v]
dups = sorted({t for t in targets if targets.count(t) > 1})
missing_crit = [FIELD_MAP[c].label for c in CRITICAL_FIELDS if c not in targets]
if dups:
    st.error("Each system field can be mapped only once. Duplicated: " + ", ".join(FIELD_MAP[d].label for d in dups))
    st.stop()
if missing_crit:
    st.error("Critical fields are not mapped: " + ", ".join(missing_crit) + ". Rows cannot be scored without them.")
    st.stop()
unmapped_model = [f.label for f in FIELDS if f.used_by_model and f.name not in targets and not f.critical]
if unmapped_model:
    st.info("Optional model inputs not present (filled with typical values; lowers confidence): " + ", ".join(unmapped_model))

mapped = apply_mapping(df, mapping)
clean, issues = clean_applicants(mapped)
st.subheader("Validation")
n_err = int((~clean["can_predict"]).sum())
c = st.columns(3)
common.kpi(c[0], "Rows ready to assess", f"{int(clean['can_predict'].sum()):,}", "#1b7f3b")
common.kpi(c[1], "Rows that cannot be assessed", f"{n_err:,}", "#c0262d" if n_err else None)
common.kpi(c[2], "Warnings", f"{int((issues.severity == 'warning').sum()):,}")
if len(issues):
    show = issues[issues.severity != "info"]
    if len(show):
        st.dataframe(show, hide_index=True, width="stretch")
    with st.expander(f"Imputation notes ({int((issues.severity == 'info').sum())})"):
        st.dataframe(issues[issues.severity == "info"], hide_index=True, width="stretch")

if st.button(f"Run risk assessment on {len(df):,} applicants", type="primary"):
    common.run_assessment(mapped, up.name)
    st.success("Assessment complete. Opening Bulk Risk Assessment…")
    st.switch_page("views/bulk.py")
