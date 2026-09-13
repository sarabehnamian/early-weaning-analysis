#!/usr/bin/env python3
"""
r21_check_ke1999.py
Why does the current extraction take only 687 children from the 1999 Kenya
survey (KEKR42FL) when the readiness scan lists 5,447 eligible?

Reads the KR file directly and counts, step by step, how many children survive
each condition 02_extract_breastfeeding_data.py applies. Prints one line per
step so the losing step is obvious. Nothing is written or changed.

Usage (from D:\\DiscoverPublicHealth\\early-weaning-analysis):
    python r21_check_ke1999.py --zip "<path to>\\KEKR42DT.zip"
    python r21_check_ke1999.py --dta "<path to>\\KEKR42FL.DTA"
    python r21_check_ke1999.py --dta ... --compare-csv ".\\02_extract_FULL\\combined_breastfeeding_data.csv"
"""
import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def load(zip_path, dta_path):
    if dta_path:
        return pd.read_stata(dta_path, convert_categoricals=False), Path(dta_path).name
    with zipfile.ZipFile(zip_path) as z:
        name = [n for n in z.namelist() if n.upper().endswith(".DTA")][0]
        with z.open(name) as fh:
            return pd.read_stata(io.BytesIO(fh.read()), convert_categoricals=False), name


def main(a):
    df, name = load(a.zip, a.dta)
    print(f"File: {name}")
    print(f"rows in KR file:                          {len(df):,}")
    cols = {c.lower(): c for c in df.columns}
    def col(x):
        return cols.get(x)

    for v in ["m4", "b5", "b19", "v008", "b3", "v005", "v021", "v025", "v106", "v190", "v007"]:
        print(f"  column {v:5s} present: {v in cols}")

    d = df.copy()
    if col("b5") is not None:
        alive = pd.to_numeric(d[col("b5")], errors="coerce") == 1
        print(f"alive (b5 == 1):                          {int(alive.sum()):,}")
        d = d[alive]

    if col("b19") is not None:
        age = pd.to_numeric(d[col("b19")], errors="coerce")
        src = "b19"
    else:
        age = pd.to_numeric(d[col("v008")], errors="coerce") - pd.to_numeric(d[col("b3")], errors="coerce")
        src = "v008 - b3"
    print(f"age in months from {src}: missing {int(age.isna().sum()):,}, "
          f"min {np.nanmin(age.values) if age.notna().any() else 'n/a'}, "
          f"max {np.nanmax(age.values) if age.notna().any() else 'n/a'}")
    u60 = age < 60
    print(f"aged < 60 months:                         {int(u60.sum()):,}")
    d = d[u60.fillna(False)]

    m4 = pd.to_numeric(d[col("m4")], errors="coerce")
    print(f"m4 missing (NaN):                         {int(m4.isna().sum()):,}")
    print("m4 value counts (top 15):")
    print(m4.value_counts(dropna=False).head(15).to_string())
    print(f"  m4 = 94 (never breastfed):              {int((m4 == 94).sum()):,}")
    print(f"  m4 = 95 (still breastfeeding):          {int((m4 == 95).sum()):,}")
    print(f"  m4 in 96/97/98/99 (other/DK/missing):   {int(m4.isin([96, 97, 98, 99]).sum()):,}")
    print(f"  m4 <= 60 (reported duration):           {int((m4 <= 60).sum()):,}")
    usable = m4.isin([94, 95]) | (m4 <= 60)
    print(f"usable m4 (94, 95 or <= 60 months):       {int(usable.sum()):,}")

    if col("v005") is not None:
        w = pd.to_numeric(d[col("v005")], errors="coerce") / 1e6
        print(f"weight v005 missing:                      {int(w.isna().sum()):,}")
        print(f"weight <= 0:                              {int((w <= 0).sum()):,}")

    if a.compare_csv:
        c = pd.read_csv(a.compare_csv, low_memory=False)
        ke = c[(c["country"] == "KE")]
        print()
        print("In the combined file, Kenya by survey_year:")
        print(ke.groupby("survey_year").size().to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--zip")
    p.add_argument("--dta")
    p.add_argument("--compare-csv")
    a = p.parse_args()
    if not (a.zip or a.dta):
        p.error("give --zip or --dta")
    main(a)
