#!/usr/bin/env python3
"""
r22_audit_extraction.py
Re-reads every ready KR dataset straight from the zips in .\\data and counts, for
each file: rows, alive children, children < 60 months, the m4 breakdown, and how
many children survive the exact filters 02_extract_breastfeeding_data.py applies.
It then puts those counts next to (a) the readiness scan and (b) the combined
extraction output, so any file where the three disagree is visible.

Why: the readiness scan reports 2,521,858 eligible children with valid m4 across
the 307 ready datasets, the extraction produced 2,046,191, and for 98 datasets
the two disagree by more than five children (e.g. NGKR7BFL: 30,713 vs 10,440).
One of the two counts is wrong, and Figure 1 cannot be finalised until we know
which. This script decides it by reading the files themselves.

Usage (from D:\\DiscoverPublicHealth\\early-weaning-analysis):
    python r22_audit_extraction.py
    python r22_audit_extraction.py --zip-dir ".\\data" --scan ".\\Bfeed_scan_full.xlsx" ^
        --combined ".\\02_extract_FULL\\combined_breastfeeding_data.csv" --out ".\\r22_audit"

Takes roughly as long as one extraction pass. Add --limit 20 for a quick look first.
Outputs: r22_audit\\extraction_audit.xlsx, r22_audit\\r22_audit.txt
"""
import argparse
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

WANT = ["m4", "b5", "b19", "v008", "b3", "v005", "v007", "v021", "v024", "v025"]


def read_kr(zip_path: Path):
    with zipfile.ZipFile(zip_path) as zf:
        members = [n for n in zf.namelist() if n.upper().endswith(".DTA")]
        if not members:
            raise RuntimeError("no .DTA member")
        member = members[0]
        with tempfile.TemporaryDirectory() as td:
            zf.extract(member, path=td)
            p = Path(td) / member
            try:
                import pyreadstat
                df, _ = pyreadstat.read_dta(str(p))
                reader = "pyreadstat"
            except Exception:
                df = pd.read_stata(p, convert_categoricals=False)
                reader = "pandas"
    df.columns = [c.lower() for c in df.columns]
    return df, Path(member).name, reader


def audit_one(zip_path: Path):
    df, member, reader = read_kr(zip_path)
    n_rows = len(df)
    num = lambda c: pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(np.nan, index=df.index)
    b5, m4 = num("b5"), num("m4")
    age = num("b19") if "b19" in df.columns else num("v008") - num("b3")

    alive = (b5 == 1) if "b5" in df.columns else pd.Series(True, index=df.index)
    elig = alive & age.notna() & (age >= 0) & (age < 60)
    real = elig & (m4 >= 0) & (m4 <= 93)
    never = elig & (m4 == 94)
    cens = elig & (m4 == 95)
    invalid = elig & m4.isin([96, 97, 98, 99])
    missing = elig & m4.isna()
    dur = np.where(real, m4, np.where(never, 0, age))
    keep = (real | never | cens) & (pd.Series(dur, index=df.index) >= 0) & (pd.Series(dur, index=df.index) <= 60)

    return dict(dta=member.upper(), reader=reader, file_rows=n_rows,
                alive=int(alive.sum()), eligible_u60=int(elig.sum()),
                m4_real=int(real.sum()), m4_never94=int(never.sum()), m4_cens95=int(cens.sum()),
                m4_invalid=int(invalid.sum()), m4_missing=int(missing.sum()),
                would_extract=int(keep.sum()),
                survey_year=(int(num("v007").mode().iloc[0]) if "v007" in df.columns and num("v007").notna().any() else None),
                size_mb=round(zip_path.stat().st_size / 1e6, 2))


def main(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lines = []
    def log(s=""):
        print(s, flush=True); lines.append(str(s))

    scan = pd.read_excel(a.scan, sheet_name="KR_Readiness")
    scan["dta"] = scan["dta"].str.upper()
    ready = scan[scan.status == "ready"].copy()
    log(f"ready datasets in scan: {len(ready)}")

    zdir = Path(a.zip_dir)
    rows, problems = [], []
    todo = list(ready.itertuples())[: a.limit] if a.limit else list(ready.itertuples())
    for i, r in enumerate(todo, 1):
        zp = zdir / r.zip
        if not zp.exists():
            problems.append(dict(zip=r.zip, dta=r.dta, issue="zip not found in --zip-dir"))
            continue
        try:
            rec = audit_one(zp)
        except Exception as e:
            problems.append(dict(zip=r.zip, dta=r.dta, issue=f"read failed: {e}"))
            continue
        rec["zip"] = r.zip
        rec["scan_rows"] = r.n_rows
        rec["scan_eligible"] = r.n_eligible_u60_alive
        rec["scan_valid_m4"] = r.n_event + r.n_censored
        rows.append(rec)
        if i % 25 == 0:
            log(f"  ...{i}/{len(todo)}")

    d = pd.DataFrame(rows)
    if d.empty:
        log("nothing read — check --zip-dir"); return
    d["rows_vs_scan"] = d.file_rows - d.scan_rows
    d["valid_vs_scan"] = d.would_extract - d.scan_valid_m4

    if a.combined:
        c = pd.read_csv(a.combined, low_memory=False)
        got = c.groupby(["country", "survey_year"]).size().rename("in_combined").reset_index()
        d["country"] = d.dta.str[:2]
        d = d.merge(got, how="left", left_on=["country", "survey_year"], right_on=["country", "survey_year"])

    log()
    log(f"files audited:                         {len(d)}")
    log(f"sum of file rows:                      {int(d.file_rows.sum()):,}")
    log(f"sum eligible (<60 months, alive):      {int(d.eligible_u60.sum()):,}")
    log(f"sum that pass 02's filters:            {int(d.would_extract.sum()):,}")
    log(f"sum the scan called valid m4:          {int(d.scan_valid_m4.sum()):,}")
    log()
    log(f"files whose row count differs from the scan: {int((d.rows_vs_scan != 0).sum())}")
    if (d.rows_vs_scan != 0).any():
        log("  largest differences (negative = file on disk is smaller than when scanned):")
        log(d.reindex(d.rows_vs_scan.abs().sort_values(ascending=False).index)
              [["dta", "size_mb", "file_rows", "scan_rows", "rows_vs_scan"]].head(15).to_string(index=False))
    log()
    log(f"files where 02's filters and the scan's valid-m4 count differ by >5: {int((d.valid_vs_scan.abs() > 5).sum())}")
    if (d.valid_vs_scan.abs() > 5).any():
        log(d.reindex(d.valid_vs_scan.abs().sort_values(ascending=False).index)
              [["dta", "eligible_u60", "m4_real", "m4_never94", "m4_cens95", "m4_missing",
                "would_extract", "scan_valid_m4", "valid_vs_scan"]].head(15).to_string(index=False))
    log()
    log("m4 totals across all audited files:")
    for c in ["m4_real", "m4_never94", "m4_cens95", "m4_invalid", "m4_missing"]:
        log(f"  {c:12s} {int(d[c].sum()):,}")

    pr = pd.DataFrame(problems)
    if len(pr):
        log(); log(f"files that could not be audited: {len(pr)}"); log(pr.to_string(index=False))

    with pd.ExcelWriter(out / "extraction_audit.xlsx") as xw:
        d.to_excel(xw, sheet_name="per_dataset", index=False)
        if len(pr):
            pr.to_excel(xw, sheet_name="problems", index=False)
    (out / "r22_audit.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDone. Written to {out.resolve()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--zip-dir", default=r".\data")
    p.add_argument("--scan", default=r".\Bfeed_scan_full.xlsx")
    p.add_argument("--combined", default=r".\02_extract_FULL\combined_breastfeeding_data.csv")
    p.add_argument("--out", default=r".\r22_audit")
    p.add_argument("--limit", type=int, default=0, help="audit only the first N datasets")
    main(p.parse_args())
