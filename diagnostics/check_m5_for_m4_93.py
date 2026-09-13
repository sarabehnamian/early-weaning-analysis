#!/usr/bin/env python3
"""
r23_check_m5.py
In many DHS files m4 = 93 is the CODE "ever breastfed, not currently breastfeeding",
not a duration of 93 months. 02_extract_breastfeeding_data.py treats m4 <= 93 as a
reported duration, so those children get duration 93, fail the "duration <= 60"
check and disappear: in 92 of 300 country-surveys not one cessation event with a
positive duration survives.

The duration for those children is normally in m5 ("months of breastfeeding").
This script checks, per dataset: how many children have m4 = 93, whether m5 exists,
and what m5 holds for exactly those children — i.e. how many events are recoverable.

Usage (from D:\\DiscoverPublicHealth\\early-weaning-analysis):
    python r23_check_m5.py --limit 20        # quick look
    python r23_check_m5.py                   # all 307 ready datasets

Outputs: r23_m5\\m4_m5_audit.xlsx, r23_m5\\r23_m5.txt
"""
import argparse
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def read_kr(zip_path: Path):
    with zipfile.ZipFile(zip_path) as zf:
        member = [n for n in zf.namelist() if n.upper().endswith(".DTA")][0]
        with tempfile.TemporaryDirectory() as td:
            zf.extract(member, path=td)
            p = Path(td) / member
            try:
                import pyreadstat
                df, _ = pyreadstat.read_dta(str(p))
            except Exception:
                df = pd.read_stata(p, convert_categoricals=False)
    df.columns = [c.lower() for c in df.columns]
    return df, Path(member).name


def audit(zip_path: Path):
    df, member = read_kr(zip_path)
    num = lambda c: pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(np.nan, index=df.index)
    b5, m4, m5 = num("b5"), num("m4"), num("m5")
    age = num("b19") if "b19" in df.columns else num("v008") - num("b3")
    elig = ((b5 == 1) if "b5" in df.columns else pd.Series(True, index=df.index)) & age.notna() & (age >= 0) & (age < 60)

    e93 = elig & (m4 == 93)
    has_m5 = "m5" in df.columns
    m5_93 = m5[e93] if has_m5 else pd.Series(dtype=float)
    rec = dict(
        dta=member.upper(),
        eligible=int(elig.sum()),
        m4_lt93=int((elig & (m4 >= 0) & (m4 < 93)).sum()),
        m4_eq93=int(e93.sum()),
        m4_94=int((elig & (m4 == 94)).sum()),
        m4_95=int((elig & (m4 == 95)).sum()),
        has_m5=has_m5,
        m5_usable_for_m4_93=int(((m5_93 >= 0) & (m5_93 <= 60)).sum()) if has_m5 else 0,
        m5_missing_for_m4_93=int(m5_93.isna().sum()) if has_m5 else int(e93.sum()),
        m5_special_for_m4_93=int(m5_93.isin([94, 95, 96, 97, 98, 99]).sum()) if has_m5 else 0,
        m5_median_for_m4_93=(float(m5_93[(m5_93 >= 0) & (m5_93 <= 60)].median())
                             if has_m5 and ((m5_93 >= 0) & (m5_93 <= 60)).any() else None),
    )
    if has_m5:
        u = m5[elig & (m5 >= 0) & (m5 <= 60)]
        rec["m5_usable_all_eligible"] = int(len(u))
    else:
        rec["m5_usable_all_eligible"] = 0
    return rec


def main(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lines = []
    def log(s=""):
        print(s, flush=True); lines.append(str(s))

    scan = pd.read_excel(a.scan, sheet_name="KR_Readiness")
    ready = scan[scan.status == "ready"]
    todo = list(ready.itertuples())[: a.limit] if a.limit else list(ready.itertuples())
    rows, bad = [], []
    for i, r in enumerate(todo, 1):
        zp = Path(a.zip_dir) / r.zip
        if not zp.exists():
            bad.append((r.zip, "not found")); continue
        try:
            rows.append(audit(zp))
        except Exception as e:
            bad.append((r.zip, str(e)[:80]))
        if i % 25 == 0:
            log(f"  ...{i}/{len(todo)}")

    d = pd.DataFrame(rows)
    if d.empty:
        log("nothing read"); return
    aff = d[d.m4_eq93 > 0]
    log()
    log(f"datasets audited:                              {len(d)}")
    log(f"datasets using the m4 = 93 code:               {len(aff)}")
    log(f"children coded m4 = 93 (currently dropped):    {int(d.m4_eq93.sum()):,}")
    log(f"  of these, m5 gives a usable duration 0-60:   {int(d.m5_usable_for_m4_93.sum()):,}")
    log(f"  m5 missing:                                  {int(d.m5_missing_for_m4_93.sum()):,}")
    log(f"  m5 holds a special code (94-99):             {int(d.m5_special_for_m4_93.sum()):,}")
    log(f"datasets without an m5 variable:               {int((~d.has_m5).sum())}")
    log()
    if len(aff):
        log("datasets with the most children coded m4 = 93:")
        log(aff.sort_values("m4_eq93", ascending=False)
              [["dta", "eligible", "m4_lt93", "m4_eq93", "m4_94", "m4_95",
                "has_m5", "m5_usable_for_m4_93", "m5_median_for_m4_93"]].head(20).to_string(index=False))
    if bad:
        log(); log(f"could not audit {len(bad)}:"); [log(f"  {z}: {m}") for z, m in bad]

    d.to_excel(out / "m4_m5_audit.xlsx", index=False)
    (out / "r23_m5.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDone. Written to {out.resolve()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--zip-dir", default=r".\data")
    p.add_argument("--scan", default=r".\Bfeed_scan_full.xlsx")
    p.add_argument("--out", default=r".\r23_m5")
    p.add_argument("--limit", type=int, default=0)
    main(p.parse_args())
