#!/usr/bin/env python3
"""
r27_key_numbers.py
The few descriptive numbers the manuscript quotes that neither 03 nor 04 writes
to a file: the pooled median duration of any breastfeeding, and the median and
continuation probabilities for rural and urban children.

Uses exactly the rule applied in 03 and 04 — the Turnbull NPMLE where
left-censored children (m4 = 93) are present, the ordinary Kaplan-Meier
estimator otherwise — so the values agree with the figures and tables.

It also re-prints the pooled cessation percentages and the censoring mix, so the
Methods, Results and Figure 1 can be checked against one output.

Usage (from D:\\DiscoverPublicHealth\\early-weaning-analysis):
    python r27_key_numbers.py --data ".\\02_extract_v23\\combined_breastfeeding_data.csv"

Runtime: a few minutes. Outputs r27_key_numbers\\r27_key_numbers.txt
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter


def fit(d: pd.DataFrame):
    kmf = KaplanMeierFitter()
    w = d["weight"].to_numpy(dtype=float)
    n_left = int((d["censor_type"] == "left").sum()) if "censor_type" in d.columns else 0
    if n_left > 0 and {"t_lower", "t_upper"}.issubset(d.columns):
        lo = pd.to_numeric(d["t_lower"], errors="coerce").to_numpy(dtype=float)
        up = pd.to_numeric(d["t_upper"], errors="coerce").to_numpy(dtype=float)
        up = np.where(np.isnan(up), np.inf, up)
        kmf.fit_interval_censoring(lo, up, weights=w)
        est = "Turnbull"
    else:
        kmf.fit(d["duration_months"], d["event"], weights=w)
        est = "Kaplan-Meier"
    med = kmf.median_survival_time_
    if isinstance(med, (pd.DataFrame, pd.Series)):
        med = float(np.asarray(med, dtype=float).ravel().mean())
    sf = kmf.survival_function_
    def at(t):
        i = sf.index.get_indexer([t], method="ffill")[0]
        return float(sf.iloc[i, 0]) if i >= 0 else 1.0
    return est, float(med), at(6), at(12), at(24)


def main(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lines = []
    def log(s=""):
        print(s, flush=True); lines.append(str(s))

    df = pd.read_csv(a.data, low_memory=False)
    log(f"rows loaded:                 {len(df):,}")
    df = df[df["country"].notna()].copy()
    log(f"with a country code:         {len(df):,}   (this is the analytic sample)")
    vc = df["censor_type"].value_counts()
    log(f"  exact {int(vc.get('exact',0)):,} | right {int(vc.get('right',0)):,} | left {int(vc.get('left',0)):,}")
    log(f"  country-survey datasets:   {df.groupby(['country','survey_year']).ngroups}")
    df["weight"] = pd.to_numeric(df["v005"], errors="coerce") / 1e6
    n_zero = int((df["weight"] <= 0).sum())
    df = df[df["weight"] > 0]
    log(f"  zero weights excluded:     {n_zero:,}; weighted sample {len(df):,}")

    log()
    est, med, s6, s12, s24 = fit(df)
    log(f"POOLED ({est})")
    log(f"  median duration of any breastfeeding: {med:.0f} months")
    log(f"  still breastfeeding at  6 months:     {100*s6:.1f}%   (ceased {100*(1-s6):.1f}%)")
    log(f"  still breastfeeding at 12 months:     {100*s12:.1f}%   (ceased {100*(1-s12):.1f}%)")
    log(f"  still breastfeeding at 24 months:     {100*s24:.1f}%   (ceased {100*(1-s24):.1f}%)")

    if "urban" in df.columns:
        log()
        log("BY RESIDENCE")
        for val, name in [(0, "Rural"), (1, "Urban")]:
            d = df[df["urban"] == val]
            est, med, s6, s12, s24 = fit(d)
            log(f"  {name}: n = {len(d):,} ({est})")
            log(f"    median {med:.0f} months; still breastfeeding at 6/12/24 months: "
                f"{100*s6:.1f}% / {100*s12:.1f}% / {100*s24:.1f}%")
        log(f"  children with missing residence: {int(df['urban'].isna().sum()):,}")

    (out / "r27_key_numbers.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDone. Written to {out.resolve()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=r".\02_extract_v23\combined_breastfeeding_data.csv")
    p.add_argument("--out", default=r".\r27_key_numbers")
    main(p.parse_args())
