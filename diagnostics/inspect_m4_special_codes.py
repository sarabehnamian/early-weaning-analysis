#!/usr/bin/env python3
"""
r24_m4_m5_values.py
Prints the m4 x m5 cross-tabulation for one KR file, plus the value counts of
every other variable that could hold a breastfeeding duration, so we can see
where (if anywhere) the duration for children coded m4 = 93 is recorded.

Usage (from D:\\DiscoverPublicHealth\\early-weaning-analysis):
    python r24_m4_m5_values.py --zip ".\\data\\AOKR71DT.zip"
"""
import argparse
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)


def read_kr(zip_path: Path):
    with zipfile.ZipFile(zip_path) as zf:
        member = [n for n in zf.namelist() if n.upper().endswith(".DTA")][0]
        with tempfile.TemporaryDirectory() as td:
            zf.extract(member, path=td)
            p = Path(td) / member
            try:
                import pyreadstat
                df, meta = pyreadstat.read_dta(str(p))
                labels = getattr(meta, "column_names_to_labels", {}) or {}
            except Exception:
                df = pd.read_stata(p, convert_categoricals=False)
                labels = {}
    df.columns = [c.lower() for c in df.columns]
    return df, Path(member).name, {k.lower(): v for k, v in labels.items()}


def main(a):
    df, member, labels = read_kr(Path(a.zip))
    num = lambda c: pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(np.nan, index=df.index)
    b5 = num("b5")
    age = num("b19") if "b19" in df.columns else num("v008") - num("b3")
    elig = (b5 == 1) & age.notna() & (age >= 0) & (age < 60)
    d = df[elig]
    print(f"File: {member}   eligible children: {len(d):,}\n")

    m4, m5 = pd.to_numeric(d.get("m4"), errors="coerce"), pd.to_numeric(d.get("m5"), errors="coerce")
    print("m4 value counts:"); print(m4.value_counts(dropna=False).to_string()); print()
    if "m5" in d.columns:
        print("m5 value counts:"); print(m5.value_counts(dropna=False).to_string()); print()
        print("cross-tabulation m4 (rows) x m5 (columns):")
        print(pd.crosstab(m4, m5, dropna=False).to_string()); print()
        print("for children coded m4 = 93, m5 is:")
        print(m5[m4 == 93].value_counts(dropna=False).to_string()); print()
    else:
        print("no m5 variable in this file\n")

    # any other variable that might carry a duration in months
    cands = [c for c in d.columns
             if c in ("m4", "m5", "m6", "m7", "v404", "v405", "v406", "v409", "b19", "hw1")
             or ("breast" in str(labels.get(c, "")).lower())
             or ("duration" in str(labels.get(c, "")).lower())]
    print("candidate duration-related variables present in this file:")
    for c in sorted(set(cands)):
        s = pd.to_numeric(d[c], errors="coerce")
        inrange = int(((s >= 0) & (s <= 60)).sum())
        print(f"  {c:6s} {str(labels.get(c, ''))[:58]:60s} values 0-60: {inrange:6,}  "
              f"missing: {int(s.isna().sum()):6,}  distinct: {s.nunique()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--zip", required=True)
    main(p.parse_args())
