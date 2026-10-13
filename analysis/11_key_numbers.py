# 11_key_numbers.py
# One table of every number the manuscript quotes, read from the v25 outputs only.
# Each row: section, quantity, value (full precision), value as reported (rounded once:
# HRs and CIs 2 decimals, percentages 1 decimal, medians whole months), unit, source file.
# Count-based ("crude") prevalence columns are never used.
#
# Usage (from D:\early-weaning-analysis), after 02, 04, 05 and 08 have been run on v25:
#   python analysis\11_key_numbers.py
#   python analysis\11_key_numbers.py --cox .\r26_cox_exact_v25\cox_exact_results.xlsx
# Without --cox it uses the PSU-clustered fit from 05 if it exists, else the model-based (--fast)
# fit; then every Cox CI and p is marked provisional = yes (the HRs are identical in both).

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--extract", default=r".\02_extract_v25")
ap.add_argument("--comparative", default=r".\04_comparative_v25")
ap.add_argument("--pooled", default=r".\r27_key_numbers_v25\r27_key_numbers.xlsx")
ap.add_argument("--cox", default=None)
ap.add_argument("--out", default=r".\v25_extras\r31_key_numbers_v25.xlsx")
a = ap.parse_args()

EX = Path(a.extract)
CMP = Path(a.comparative)
F_COMBINED = EX / "combined_breastfeeding_data.csv"
F_YEARS = EX / "survey_year_conversion.csv"
F_STATES = EX / "india_state_files_excluded.csv"
F_DR = EX / "dr_special_surveys_excluded.csv"
F_RANK = CMP / "rankings" / "country_rankings_complete.xlsx"
F_REGION = CMP / "regional" / "regional_comparison.xlsx"
F_INEQ = CMP / "inequalities" / "within_country_inequalities.xlsx"
F_POOLED = Path(a.pooled)
F_COX = Path(a.cox) if a.cox else next(
    p for p in (Path(r".\r26_cox_exact_v25\cox_exact_results.xlsx"),
                Path(r".\r26_cox_exact_v25_fast\cox_exact_results.xlsx")) if p.exists())
CAP = 36.0  # 04 reports a median the curve never reaches as 36 months

rows = []


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def add(section, quantity, value, kind, source, unit="", provisional=False):
    """kind: count | pct | median | hr | p | stat | c | year | text"""
    if kind == "count":
        rep = f"{int(value):,}"
    elif kind == "pct":
        rep = f"{value:.1f}"
    elif kind == "median":
        rep = f"{value:.0f}"
    elif kind == "hr":
        rep = f"{value:.2f}"
    elif kind == "p":
        rep = fmt_p(value)
    elif kind == "stat":
        rep = f"{value:.2f}"
    elif kind == "c":
        rep = f"{value:.3f}"
    elif kind == "year":
        rep = f"{int(value)}"
    else:
        rep = str(value)
    if kind in ("count", "year") and not isinstance(value, str):
        value = int(value)
    rows.append({"section": section, "quantity": quantity, "value": value,
                 "value_reported": rep, "unit": unit, "source_file": str(source),
                 "provisional": "yes" if provisional else "no"})


# -------------------- inputs --------------------
df = pd.read_csv(F_COMBINED, low_memory=False)
years = pd.read_csv(F_YEARS)
states = pd.read_csv(F_STATES, usecols=["source_zip"])
dr = pd.read_csv(F_DR, usecols=["source_zip"]) if F_DR.exists() else pd.DataFrame(columns=["source_zip"])
rank = pd.read_excel(F_RANK)
region = pd.read_excel(F_REGION)
ineq = pd.read_excel(F_INEQ)
pooled = pd.read_excel(F_POOLED)
cox = pd.read_excel(F_COX, sheet_name="cox_summary", index_col=0)
cox_sample = pd.read_excel(F_COX, sheet_name="sample").set_index("metric")["value"]
cox_surveys = pd.read_excel(F_COX, sheet_name="excluded_surveys")
ph = pd.read_excel(F_COX, sheet_name="ph_test", index_col=0)


def pv(group, quantity):
    v = pooled.loc[(pooled.group == group) & (pooled.quantity == quantity), "value"].iloc[0]
    try:
        return float(v)
    except ValueError:
        return v


CLUSTERED = bool(cox_sample.get("clustered_SE", False))
if not CLUSTERED:
    print(f"WARNING: {F_COX} has model-based SEs (--fast); the CIs are not the reportable ones")

# -------------------- sample --------------------
S = "sample"
add(S, "countries", df["country"].nunique(), "count", F_COMBINED)
add(S, "children", len(df), "count", F_COMBINED)
add(S, "country-survey datasets (survey files)", df["source_zip"].nunique(), "count", F_COMBINED)
add(S, "country x survey-year groups", df.groupby(["country", "survey_year"]).ngroups, "count", F_COMBINED)
for ct, label in [("exact", "exact durations"), ("right", "right-censored (m4 = 95)"),
                  ("left", "left-censored (m4 = 93)")]:
    add(S, f"children with {label}", int((df["censor_type"] == ct).sum()), "count", F_COMBINED)
add(S, "zero-weight children (excluded from weighted estimates)", pv("sample", "zero_weight_children"),
    "count", F_POOLED)
add(S, "rural children (weighted sample)", pv("rural", "n"), "count", F_POOLED)
add(S, "urban children (weighted sample)", pv("urban", "n"), "count", F_POOLED)
inc_years = years[years["source_zip"].isin(df["source_zip"].unique())]
add(S, "first survey year", inc_years["survey_year"].min(), "year", F_YEARS)
add(S, "last survey year", inc_years["survey_year"].max(), "year", F_YEARS)

S = "sample: Cox model"
add(S, "children", cox_sample["n_children"], "count", F_COX)
add(S, "events", cox_sample["n_events"], "count", F_COX)
add(S, "right-censored", cox_sample["n_censored"], "count", F_COX)
add(S, "countries", cox_sample["countries"], "count", F_COX)
add(S, "country x survey-year strata", cox_sample["strata"], "count", F_COX)
add(S, "primary sampling units", cox_sample["psus"], "count", F_COX)
add(S, "surveys with retrospective durations", cox_sample["surveys_retrospective"], "count", F_COX)
add(S, "current-status-only surveys", cox_sample["surveys_current_status"], "count", F_COX)

# Exclusions by reason, applied in the order 05 uses them
S = "exclusions"
add(S, "NFHS-2 Indian state files: children", len(states), "count", F_STATES)
add(S, "NFHS-2 Indian state files: files", states["source_zip"].nunique(), "count", F_STATES)
if len(dr):
    add(S, "Dominican Republic batey special surveys: children", len(dr), "count", F_DR)
    add(S, "Dominican Republic batey special surveys: files", dr["source_zip"].nunique(), "count", F_DR)

# Sequential chain from the analytic sample to the Cox sample; each step counts the
# children it removes from what is left, so the steps add up exactly.
d = df.copy()
d["weight"] = pd.to_numeric(d["v005"], errors="coerce") / 1e6
retro = set(map(tuple, cox_surveys.loc[cox_surveys["retrospective"].astype(bool),
                                       ["country", "survey_year"]].values))
src = f"{F_COMBINED} + {F_COX.name} (sequential)"
add(S, "Cox chain: analytic sample (start)", len(d), "count", src)
steps = [
    ("survey has no retrospective durations (current-status only)",
     lambda x: np.array([(c, y) in retro for c, y in zip(x["country"], x["survey_year"])])),
    ("left-censored (m4 = 93) within retrospective surveys", lambda x: x["censor_type"].ne("left").values),
    ("no residence", lambda x: x["urban"].notna().values),
    ("no wealth index", lambda x: x["wealth_quintile"].notna().values),
    ("missing maternal education", lambda x: x["mother_educ"].notna().values),
    ("no PSU (v021)", lambda x: x["v021"].notna().values),
    ("zero survey weight", lambda x: (x["weight"] > 0).values),
]
for label, keep in steps:
    k = keep(d)
    add(S, f"Cox chain: minus {label}", int((~k).sum()), "count", src)
    d = d[k]
if len(d) != int(cox_sample["n_children"]):
    print(f"WARNING: exclusion chain leaves {len(d):,}, 05 sample is {int(cox_sample['n_children']):,}")
add(S, "Cox chain: remaining = Cox sample", len(d), "count", src)

# -------------------- pooled --------------------
S = "pooled"
add(S, "estimator", pv("pooled", "estimator"), "text", F_POOLED)
add(S, "median duration of any breastfeeding", pv("pooled", "median"), "median", F_POOLED, "months")
for t in (6, 12, 24):
    s = pv("pooled", f"S{t}")
    add(S, f"S({t}): still breastfeeding at {t} months", 100 * s, "pct", F_POOLED, "%")
    add(S, f"1 - S({t}): ceased by {t} months", 100 * (1 - s), "pct", F_POOLED, "%")
add(S, "countries with median >= 24 months", int((rank["median_duration_months"] >= 24).sum()), "count", F_RANK)
add(S, "countries with S(24) >= 50%", int((rank["pct_still_bf_at_24mo"] >= 50).sum()), "count", F_RANK)
capped = rank[rank["median_duration_months"] >= CAP].sort_values("country_name")
add(S, "countries at the 36-month median ceiling", len(capped), "count", F_RANK)
add(S, "countries at the 36-month median ceiling: names",
    "; ".join(capped["country_name"]) if len(capped) else "none", "text", F_RANK)

# -------------------- by country --------------------
S = "by country"
for _, r in rank.sort_values("country_name").iterrows():
    nm = f"{r['country_name']} ({r['country_code']})"
    add(S, f"{nm}: n", r["n_children"], "count", F_RANK)
    add(S, f"{nm}: n left-censored", r["n_left_censored"], "count", F_RANK)
    add(S, f"{nm}: median" + (" (at 36-month ceiling)" if r["median_duration_months"] >= CAP else ""),
        r["median_duration_months"], "median", F_RANK, "months")
    add(S, f"{nm}: S(6)", 100 - r["pct_weaned_before_6mo"], "pct", F_RANK, "%")
    add(S, f"{nm}: S(12)", r["pct_still_bf_at_12mo"], "pct", F_RANK, "%")
    add(S, f"{nm}: S(24)", r["pct_still_bf_at_24mo"], "pct", F_RANK, "%")
    add(S, f"{nm}: 1 - S(6)", r["pct_weaned_before_6mo"], "pct", F_RANK, "%")
    add(S, f"{nm}: never breastfed (m4 = 94)", r["pct_never_breastfed"], "pct", F_RANK, "%")

# -------------------- by region --------------------
S = "by region"
for _, r in region.sort_values("region").iterrows():
    nm = r["region"]
    add(S, f"{nm}: countries", r["n_countries"], "count", F_REGION)
    add(S, f"{nm}: n", r["n_children"], "count", F_REGION)
    add(S, f"{nm}: median", r["median_duration"], "median", F_REGION, "months")
    add(S, f"{nm}: 1 - S(6)", r["pct_early_weaning"], "pct", F_REGION, "%")
    add(S, f"{nm}: S(24)", r["pct_extended_bf"], "pct", F_REGION, "%")
    add(S, f"{nm}: never breastfed (m4 = 94)", r["pct_never_bf"], "pct", F_REGION, "%")

# -------------------- urban / rural --------------------
S = "urban/rural"
for g in ("rural", "urban"):
    add(S, f"{g}: n", pv(g, "n"), "count", F_POOLED)
    add(S, f"{g}: median", pv(g, "median"), "median", F_POOLED, "months")
    for t in (6, 12, 24):
        add(S, f"{g}: S({t})", 100 * pv(g, f"S{t}"), "pct", F_POOLED, "%")

# -------------------- 12 largest countries --------------------
S = "12 largest countries"
for i, (_, r) in enumerate(rank.nlargest(12, "n_children").iterrows(), 1):
    nm = f"{i}. {r['country_name']} ({r['country_code']})"
    add(S, f"{nm}: n", r["n_children"], "count", F_RANK)
    add(S, f"{nm}: median", r["median_duration_months"], "median", F_RANK, "months")
    add(S, f"{nm}: S(12)", r["pct_still_bf_at_12mo"], "pct", F_RANK, "%")
    add(S, f"{nm}: S(24)", r["pct_still_bf_at_24mo"], "pct", F_RANK, "%")

# -------------------- Cox --------------------
S = "Cox (PSU-clustered)" if bool(cox_sample.get("clustered_SE", False)) else "Cox (MODEL-BASED SEs)"
for cov, r in cox.iterrows():
    add(S, f"{r['label']}: HR", r["exp(coef)"], "hr", F_COX)
    add(S, f"{r['label']}: 95% CI lower", r["exp(coef) lower 95%"], "hr", F_COX, provisional=not CLUSTERED)
    add(S, f"{r['label']}: 95% CI upper", r["exp(coef) upper 95%"], "hr", F_COX, provisional=not CLUSTERED)
    add(S, f"{r['label']}: p", r["p"], "p", F_COX, provisional=not CLUSTERED)
add(S, "concordance (C)", cox_sample["C_index"], "c", F_COX)
S = "Schoenfeld test (Table S3)"
labels = cox["label"].to_dict()
for cov, r in ph.iterrows():
    add(S, f"{labels.get(cov, cov)}: test statistic", r["test_statistic"], "stat", F_COX)
    add(S, f"{labels.get(cov, cov)}: p", r["p"], "p", F_COX)

# -------------------- within-country gaps --------------------
S = "within-country gaps"
gap_defs = [("urban_rural_gap", "rural_median", "urban_median", "pro-rural (rural median > urban)"),
            ("wealth_gap", "poorest_median", "richest_median", "pro-poor (poorest Q1 median > richest Q5)"),
            ("education_gap", "no_education_median", "secondary_plus_median",
             "pro-no-education (no education median > secondary+)")]
for gap, hi, lo, label in gap_defs:
    g = ineq.dropna(subset=[gap])
    add(S, f"countries with a gap estimate: {gap}", len(g), "count", F_INEQ)
    add(S, f"countries with a {label} gap", int((g[gap] > 0).sum()), "count", F_INEQ)
    for which, r in [("largest", g.loc[g[gap].idxmax()]), ("most negative", g.loc[g[gap].idxmin()])]:
        flag = [c for c in (hi, lo) if r[c] >= CAP]
        note = f" [CAPPED at 36: {', '.join(flag)}]" if flag else ""
        add(S, f"{which} {gap}: {r['country_name']} ({r['country_code']}){note}", r[gap], "median", F_INEQ, "months")
        add(S, f"{which} {gap}: {r['country_name']} {hi}", r[hi], "median", F_INEQ, "months")
        add(S, f"{which} {gap}: {r['country_name']} {lo}", r[lo], "median", F_INEQ, "months")
    n_cap = int(((g[hi] >= CAP) | (g[lo] >= CAP)).sum())
    add(S, f"{gap}: countries where a subgroup median is capped at 36", n_cap, "count", F_INEQ)

out = pd.DataFrame(rows)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with pd.ExcelWriter(a.out, engine="openpyxl") as xw:
    out.to_excel(xw, sheet_name="key_numbers", index=False)
    ws = xw.sheets["key_numbers"]
    for col, w in zip("ABCDEFG", (26, 70, 18, 16, 8, 60, 11)):
        ws.column_dimensions[col].width = w
    ws.freeze_panes = "A2"
print(f"{len(out)} numbers written to {Path(a.out).resolve()}")
