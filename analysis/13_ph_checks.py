# 13_ph_checks.py
# Proportional-hazards checks for the primary Cox model on v25 (model-based SEs).
#  a) Schoenfeld test (scaled residuals, rank-transformed time) on the FULL Cox sample;
#     if that runs out of memory, on 10 random 200,000-child subsamples (seeds 1-10),
#     reporting the median statistic and p and the number of significant runs.
#  b) Time-split model: separate HRs for urban and the richest quintile (Q5) in the
#     intervals (0,6], (6,12], (12,24] and >24 months by episode splitting; other
#     covariates have one HR. Same strata (country x survey year) and weights.
#
# Usage (from D:\early-weaning-analysis):  python analysis\13_ph_checks.py
# Outputs: v25_extras\r33_ph_checks_v25.xlsx and .txt

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

from cox_common import BASE, COVARS, LABELS, cox_sample, describe, fit, load

OUT = Path(r".\v25_extras")
OUT.mkdir(exist_ok=True)
lines = []


def log(s=""):
    print(s, flush=True)
    lines.append(str(s))


import sys
d = cox_sample(load())
if "--test" in sys.argv:  # quick check of the code on a 50,000-child subsample
    d = d.sample(n=50_000, random_state=0)
    OUT = OUT / "test"
    OUT.mkdir(exist_ok=True)
info = describe(d)
log(f"Cox sample: {info['n']:,} children, {info['events']:,} events, "
    f"{info['countries']} countries, {info['strata']} strata")

# -------------------- a) Schoenfeld test --------------------
t0 = time.time()
try:
    cph = fit(d)
    ph = proportional_hazard_test(cph, d[BASE + COVARS], time_transform="rank").summary
    ph = ph.reset_index().rename(columns={"index": "covariate"})
    ph.insert(1, "label", ph["covariate"].map(LABELS))
    ph["significant_0.05"] = ph["p"] < 0.05
    ph_mode = f"full Cox sample (n = {len(d):,})"
    ph_runs = None
    del cph
except MemoryError:
    log("Full-sample Schoenfeld test ran out of memory; using 10 subsamples of 200,000")
    runs = []
    for seed in range(1, 11):
        sub = d.sample(n=200_000, random_state=seed)
        c2 = fit(sub)
        r = proportional_hazard_test(c2, sub[BASE + COVARS], time_transform="rank").summary
        r = r.reset_index().rename(columns={"index": "covariate"})
        r["seed"] = seed
        runs.append(r)
    ph_runs = pd.concat(runs)
    ph = ph_runs.groupby("covariate").agg(
        median_test_statistic=("test_statistic", "median"), median_p=("p", "median"),
        runs_significant_0_05=("p", lambda p: int((p < 0.05).sum()))).reset_index()
    ph.insert(1, "label", ph["covariate"].map(LABELS))
    ph_mode = "10 subsamples of 200,000 (seeds 1-10)"
gc.collect()
log(f"\na) Schoenfeld test, rank-transformed time, {ph_mode} ({time.time() - t0:.0f}s)")
log(ph.round(4).to_string(index=False))

# -------------------- b) time-split model --------------------
t0 = time.time()
CUTS = [(0, 6, "0-6"), (6, 12, "6-12"), (12, 24, "12-24"), (24, np.inf, ">24")]
OTHER = ["wealth_q1", "wealth_q2", "wealth_q4", "educ_none", "educ_secondary_plus"]
T = d["duration_months"].to_numpy(dtype=float)
E = d["event"].to_numpy(dtype=int)
eps = []
for lo, hi, lab in CUTS:
    # interval (lo, hi]; a child with T = 0 belongs to the first interval only
    at_risk = (T > lo) | ((lo == 0) & (T >= 0))
    idx = np.flatnonzero(at_risk)
    # durations of 0 (mostly never breastfed) become 0.01 so that stop > entry; they still
    # tie with each other and precede month 1, so every risk set is unchanged
    stop = np.minimum(np.where(T[idx] == 0, 0.01, T[idx]), hi)
    ev = np.where(T[idx] <= hi, E[idx], 0)
    part = d.iloc[idx][["weight", "country", "survey_year", "urban", "wealth_q5"] + OTHER].copy()
    part["start"] = float(lo)
    part["stop"] = stop
    part["event"] = ev
    for k, (_, _, lab2) in enumerate(CUTS):
        on = int(lab2 == lab)
        part[f"urban_{lab2}"] = part["urban"].to_numpy() * on
        part[f"q5_{lab2}"] = part["wealth_q5"].to_numpy() * on
    eps.append(part.drop(columns=["urban", "wealth_q5"]))
ep = pd.concat(eps, ignore_index=True)
del eps
gc.collect()
tv_cols = [f"urban_{c[2]}" for c in CUTS] + [f"q5_{c[2]}" for c in CUTS]
log(f"\nb) Time-split model: {len(ep):,} episodes from {len(d):,} children; "
    f"events {int(ep['event'].sum()):,} (should equal {int(d['event'].sum()):,})")
cph_ts = CoxPHFitter()
cph_ts.fit(ep[["start", "stop", "event", "weight", "country", "survey_year"] + tv_cols + OTHER],
           duration_col="stop", event_col="event", entry_col="start", weights_col="weight",
           strata=["country", "survey_year"], robust=False)
s = cph_ts.summary
ts = pd.DataFrame({"term": s.index, "HR": s["exp(coef)"].values,
                   "lower95": s["exp(coef) lower 95%"].values,
                   "upper95": s["exp(coef) upper 95%"].values, "p": s["p"].values})
ts["covariate"] = ts["term"].str.replace(r"_(0-6|6-12|12-24|>24)$", "", regex=True)
ts["interval_months"] = ts["term"].str.extract(r"_(0-6|6-12|12-24|>24)$")[0].fillna("all")
ts["covariate"] = ts["covariate"].replace({"q5": "wealth_q5"})
ts["label"] = ts["covariate"].map(LABELS)
ts = ts[["covariate", "label", "interval_months", "HR", "lower95", "upper95", "p", "term"]]
log(f"  fitted in {time.time() - t0:.0f}s")
log(ts.round(4).to_string(index=False))

with pd.ExcelWriter(OUT / "r33_ph_checks_v25.xlsx") as xw:
    pd.DataFrame([{**info, "schoenfeld_on": ph_mode, "time_split_episodes": len(ep),
                   "SEs": "model-based"}]).to_excel(xw, sheet_name="sample", index=False)
    ph.to_excel(xw, sheet_name="schoenfeld", index=False)
    if ph_runs is not None:
        ph_runs.to_excel(xw, sheet_name="schoenfeld_runs", index=False)
    ts.to_excel(xw, sheet_name="time_split", index=False)
(OUT / "r33_ph_checks_v25.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"\nDone. Written to {OUT.resolve()}")
