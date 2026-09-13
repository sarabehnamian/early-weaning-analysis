#!/usr/bin/env python3
"""
r26_cox_exact.py
Fits the primary Cox model on the v2.3 extraction.

Proportional-hazards regression cannot use left-censored observations, so the
model is restricted to country-surveys that recorded retrospective durations
(those with at least one cessation event after time zero). Within them, children
must also have residence, wealth quintile, maternal education, a PSU identifier
and a positive survey weight.

Reports the analytic sample, the hazard ratios with standard errors clustered on
the survey-specific PSU, and the proportional-hazards test.

Usage (from D:\\DiscoverPublicHealth\\early-weaning-analysis):
    python r26_cox_exact.py --fast     # model-based SEs, minutes: use to check the sample first
    python r26_cox_exact.py            # PSU-clustered SEs, slow (hours): the reportable fit
    python r26_cox_exact.py --no-wealth --fast   # residence + education only, all surveys with
                                                 # retrospective durations (no wealth index needed)

Outputs (folder r26_cox_exact):
    cox_exact_results.xlsx   sheets: cox_summary, sample, excluded_surveys, ph_test
    r26_cox_exact.txt
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

warnings.filterwarnings("ignore")

COVARS_FULL = ["urban", "wealth_q1", "wealth_q2", "wealth_q4", "wealth_q5",
               "educ_none", "educ_secondary_plus"]
COVARS_NOWEALTH = ["urban", "educ_none", "educ_secondary_plus"]
BASE = ["duration_months", "event", "weight", "country", "survey_year"]
LABELS = {"urban": "Urban residence (ref. rural)", "wealth_q1": "Poorest quintile (Q1)",
          "wealth_q2": "Second quintile (Q2)", "wealth_q4": "Fourth quintile (Q4)",
          "wealth_q5": "Richest quintile (Q5)", "educ_none": "No formal education",
          "educ_secondary_plus": "Secondary or higher"}


def main(a):
    global COVARS
    COVARS = COVARS_NOWEALTH if a.no_wealth else COVARS_FULL
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lines = []
    def log(s=""):
        print(s, flush=True); lines.append(str(s))

    df = pd.read_csv(a.data, low_memory=False)
    log(f"Rows loaded:                              {len(df):,}")
    df = df[df["country"].notna()].copy()
    df["weight"] = pd.to_numeric(df["v005"], errors="coerce") / 1e6
    n_zero = int((df["weight"] <= 0).sum())
    df = df[df["weight"] > 0]
    log(f"After dropping zero weights ({n_zero:,}):        {len(df):,}")

    # surveys that recorded retrospective durations
    sv = df.groupby(["country", "survey_year"]).apply(
        lambda x: pd.Series({"n": len(x),
                             "events_gt0": int(((x.event == 1) & (x.duration_months > 0)).sum()),
                             "n_left": int((x.censor_type == "left").sum())}),
        include_groups=False).reset_index()
    sv["retrospective"] = sv.events_gt0 > 0
    keep = set(map(tuple, sv.loc[sv.retrospective, ["country", "survey_year"]].values))
    log(f"country-surveys:                          {len(sv)}")
    log(f"  with retrospective durations:           {int(sv.retrospective.sum())}")
    log(f"  current-status only (excluded here):    {int((~sv.retrospective).sum())} "
        f"({int(sv.loc[~sv.retrospective,'n'].sum()):,} children)")

    d = df[[(c, y) in keep for c, y in zip(df.country, df.survey_year)]].copy()
    log(f"Children in retrospective-duration surveys:{len(d):,}")
    n_left_in = int((d.censor_type == "left").sum())
    if n_left_in:
        log(f"  left-censored children within them, dropped: {n_left_in:,}")
        d = d[d.censor_type != "left"]

    need = ["urban", "mother_educ", "v021"] if a.no_wealth else ["urban", "wealth_quintile", "mother_educ", "v021"]
    if a.no_wealth and a.match_primary:
        need = need + ["wealth_quintile"]
        log("  --match-primary: same children as the primary model; only the covariate set differs")
    elif a.no_wealth:
        log("  --no-wealth: wealth quintile not required, so surveys without the DHS wealth "
            "index (v190) are retained")
    for c in need:
        log(f"  missing {c:16s} {int(d[c].isna().sum()):,}" if c in d.columns else f"  {c} ABSENT")
    d = d.dropna(subset=need)
    if not a.no_wealth:
        for q in [1, 2, 4, 5]:
            d[f"wealth_q{q}"] = (d["wealth_quintile"] == q).astype(int)
    d["educ_none"] = (d["mother_educ"] == 0).astype(int)
    d["educ_secondary_plus"] = (d["mother_educ"] >= 2).astype(int)
    d["psu_id"] = (d["country"].astype(str) + "_" + d["survey_year"].astype(str)
                   + "_" + d["v021"].astype(int).astype(str))
    d = d[BASE + COVARS + ["psu_id"]].dropna()

    n_strata = d.groupby(["country", "survey_year"]).ngroups
    log()
    log("COX ANALYTIC SAMPLE:")
    log(f"  children:                               {len(d):,}")
    log(f"  events:                                 {int(d.event.sum()):,}")
    log(f"  right-censored:                         {int((d.event == 0).sum()):,}")
    log(f"  countries:                              {d.country.nunique()}")
    log(f"  country x survey_year strata:           {n_strata}")
    log(f"  primary sampling units:                 {d.psu_id.nunique():,}")

    log()
    log(f"Fitting stratified Cox ({'model-based SEs' if a.fast else 'PSU-clustered SEs'}) ...")
    cph = CoxPHFitter()
    if a.fast:
        cph.fit(d[BASE + COVARS], duration_col="duration_months", event_col="event",
                weights_col="weight", strata=["country", "survey_year"], robust=False)
    else:
        cph.fit(d[BASE + COVARS + ["psu_id"]], duration_col="duration_months", event_col="event",
                weights_col="weight", strata=["country", "survey_year"],
                cluster_col="psu_id", robust=True)
    s = cph.summary.copy()
    s.insert(0, "label", [LABELS.get(i, i) for i in s.index])
    log(f"  C = {cph.concordance_index_:.3f}")
    log(s[["label", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]
        .round(4).to_string())

    log()
    log(f"Proportional-hazards test on a random subsample of {a.ph_n:,} ...")
    sub = d.sample(n=min(a.ph_n, len(d)), random_state=a.seed)
    c2 = CoxPHFitter().fit(sub[BASE + COVARS], duration_col="duration_months", event_col="event",
                           weights_col="weight", strata=["country", "survey_year"], robust=False)
    ph = proportional_hazard_test(c2, sub[BASE + COVARS], time_transform="rank").summary
    log(ph.round(4).to_string())

    sample = pd.DataFrame({
        "metric": ["n_children", "n_events", "n_censored", "countries", "strata", "psus",
                   "C_index", "clustered_SE", "surveys_retrospective", "surveys_current_status"],
        "value": [len(d), int(d.event.sum()), int((d.event == 0).sum()), d.country.nunique(),
                  n_strata, d.psu_id.nunique(), cph.concordance_index_, not a.fast,
                  int(sv.retrospective.sum()), int((~sv.retrospective).sum())]})
    with pd.ExcelWriter(out / "cox_exact_results.xlsx") as xw:
        s.to_excel(xw, sheet_name="cox_summary")
        sample.to_excel(xw, sheet_name="sample", index=False)
        sv.sort_values(["country", "survey_year"]).to_excel(xw, sheet_name="excluded_surveys", index=False)
        ph.to_excel(xw, sheet_name="ph_test")
    (out / "r26_cox_exact.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDone. Written to {out.resolve()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=r".\02_extract_v23\combined_breastfeeding_data.csv")
    p.add_argument("--out", default=None)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--no-wealth", action="store_true",
                   help="fit residence + education only, keeping surveys with no wealth index")
    p.add_argument("--match-primary", action="store_true",
                   help="with --no-wealth: keep only children who also have a wealth quintile, so "
                        "the sample matches the primary model and only the covariate set differs")
    p.add_argument("--ph-n", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.out is None:
        args.out = (r".\r26_cox_nowealth_matched" if (args.no_wealth and args.match_primary)
                    else r".\r26_cox_nowealth" if args.no_wealth else r".\r26_cox_exact")
    main(args)
