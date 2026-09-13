r"""
r28_sensitivities_v23.py
Re-runs the three sensitivity analyses quoted in the manuscript and in the replies
to R1.1, R1.2/R2.9 and R1.3, on the v2.3 extraction and on the same sample as the
primary Cox model (r26_cox_exact.py): surveys that recorded retrospective
durations, children with residence, wealth quintile, maternal education and a PSU
identifier, and a positive survey weight. Left-censored children (m4 = 93) cannot
enter a proportional-hazards model and are excluded here; they are retained in the
descriptive analyses through the Turnbull estimator.

  (a) never-breastfed: primary model vs model excluding never-breastfed children,
      plus a weighted logistic model of ever-breastfeeding
  (b) proportional hazards: scaled Schoenfeld test on a random subsample
  (c) temporal: refit on surveys from 2010 and from 2015 onward

Usage (from D:\DiscoverPublicHealth\early-weaning-analysis):
    python r28_sensitivities_v23.py --data ".\02_extract_v23\combined_breastfeeding_data.csv"

Model-based SEs by default (fast; hazard ratios identical); --robust for sandwich SEs.
Outputs (folder r28_sensitivities): sensitivities_v23.xlsx, r28_sensitivities.txt
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test

warnings.filterwarnings("ignore")

COVARS = ["urban", "wealth_q1", "wealth_q2", "wealth_q4", "wealth_q5",
          "educ_none", "educ_secondary_plus"]
BASE = ["duration_months", "event", "weight", "country", "survey_year"]


def normalise_year(y):
    """v007 comes as 4-digit, 2-digit, or Nepali Bikram Sambat years."""
    y = pd.to_numeric(y, errors="coerce")
    out = y.copy()
    two = y < 100
    out[two & (y >= 50)] = y[two & (y >= 50)] + 1900
    out[two & (y < 50)] = y[two & (y < 50)] + 2000
    out[out > 2030] = np.nan          # Bikram Sambat (e.g. 2057, 2078): not comparable
    return out


def prepare(data_file, log):
    df = pd.read_csv(data_file, low_memory=False)
    log(f"Rows loaded:                        {len(df):,}")
    df = df[df["country"].notna()].copy()
    df["weight"] = pd.to_numeric(df["v005"], errors="coerce") / 1e6
    df = df[df["weight"] > 0]
    df["year_gregorian"] = normalise_year(df["survey_year"])

    # never-breastfed flag
    if "never_breastfed" in df.columns:
        df["never_bf"] = df["never_breastfed"].fillna(0).astype(int)
        src = "never_breastfed column"
    elif "m4" in df.columns:
        df["never_bf"] = (pd.to_numeric(df["m4"], errors="coerce") == 94).astype(int)
        src = "m4 == 94"
    else:
        df["never_bf"] = ((df["duration_months"] == 0) & (df["event"] == 1)).astype(int)
        src = "duration 0 & event (no m4 in file — includes cessation within month 0)"
    log(f"Never-breastfed flag from:          {src}")

    # same analytic sample as the primary PSU-clustered model (r19): covariates AND v021 observed
    # same analytic sample as the primary Cox model: surveys with retrospective
    # durations, then complete covariates and a PSU identifier
    sv = df.groupby(["country", "survey_year"]).apply(
        lambda x: int(((x["event"] == 1) & (x["duration_months"] > 0)).sum()),
        include_groups=False).rename("events_gt0").reset_index()
    keep = set(map(tuple, sv.loc[sv.events_gt0 > 0, ["country", "survey_year"]].values))
    n_before = len(df)
    df = df[[(c, y) in keep for c, y in zip(df.country, df.survey_year)]]
    log(f"surveys with retrospective durations: {len(keep)} of {len(sv)}; "
        f"children {len(df):,} of {n_before:,}")
    if "censor_type" in df.columns:
        n_left = int((df["censor_type"] == "left").sum())
        if n_left:
            log(f"left-censored children within them, excluded from these models: {n_left:,}")
            df = df[df["censor_type"] != "left"]
    cc = df.dropna(subset=["urban", "wealth_quintile", "mother_educ", "v021"]).copy()
    for q in [1, 2, 4, 5]:
        cc[f"wealth_q{q}"] = (cc["wealth_quintile"] == q).astype(int)
    cc["educ_none"] = (cc["mother_educ"] == 0).astype(int)
    cc["educ_secondary_plus"] = (cc["mother_educ"] >= 2).astype(int)
    keep = BASE + COVARS + ["never_bf", "year_gregorian"] + (["v021"] if "v021" in cc.columns else [])
    cc = cc[keep].dropna(subset=BASE + COVARS)
    log(f"Complete-case sample:               {len(cc):,} children, {cc['country'].nunique()} countries, "
        f"{cc.groupby(['country','survey_year']).ngroups} strata")
    return cc


def fit(d, label, robust):
    cph = CoxPHFitter()
    cph.fit(d[BASE + COVARS], duration_col="duration_months", event_col="event",
            weights_col="weight", strata=["country", "survey_year"], robust=robust)
    s = cph.summary
    tbl = pd.DataFrame({
        "model": label, "n": len(d), "events": int(d["event"].sum()),
        "countries": d["country"].nunique(),
        "HR": s["exp(coef)"], "HR_low": s["exp(coef) lower 95%"],
        "HR_high": s["exp(coef) upper 95%"], "p": s["p"],
    })
    return cph, tbl


def weighted_km_median(d):
    kmf = KaplanMeierFitter()
    kmf.fit(d["duration_months"], event_observed=d["event"], weights=d["weight"])
    return kmf.median_survival_time_


def main(data_file, out_dir, robust, ph_n, seed):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(s=""):
        print(s, flush=True); lines.append(s)

    cc = prepare(data_file, log)
    n_never = int(cc["never_bf"].sum())
    pct_w = 100 * cc.loc[cc.never_bf == 1, "weight"].sum() / cc["weight"].sum()
    log(f"Never-breastfed in this sample:     {n_never:,} ({100*n_never/len(cc):.2f}% unweighted, {pct_w:.2f}% weighted)")

    # ---- (a) never-breastfed sensitivity ----
    log(); log(f"(a) Fitting main model (robust={robust}) ...")
    cph_main, t_main = fit(cc, "Main (never-BF = time-0 events)", robust)
    log(f"    C = {cph_main.concordance_index_:.3f}, n = {len(cc):,}")
    excl = cc[cc["never_bf"] == 0]
    log("    Fitting model excluding never-breastfed ...")
    cph_ex, t_ex = fit(excl, "Excluding never-breastfed", robust)
    log(f"    C = {cph_ex.concordance_index_:.3f}, n = {len(excl):,}")
    never_tbl = pd.concat([t_main, t_ex])
    log(never_tbl[["model", "HR", "HR_low", "HR_high"]].round(3).to_string())
    log(f"    Weighted KM median: incl never = {weighted_km_median(cc)}   excl never = {weighted_km_median(excl)}")

    init_tbl = None
    try:
        import statsmodels.api as sm
        d = cc.copy(); d["ever_bf"] = 1 - d["never_bf"]
        X = sm.add_constant(d[COVARS].astype(float))
        m = sm.GLM(d["ever_bf"], X, family=sm.families.Binomial(), freq_weights=d["weight"])
        if "v021" in d.columns and d["v021"].notna().all():
            r = m.fit(cov_type="cluster", cov_kwds={"groups": d["v021"].astype(int)})
            note = "cluster-robust by PSU (v021)"
        else:
            r = m.fit(); note = "model-based SEs"
        ci = r.conf_int()
        init_tbl = pd.DataFrame({"OR": np.exp(r.params), "OR_low": np.exp(ci[0]),
                                 "OR_high": np.exp(ci[1]), "p": r.pvalues})
        log(f"    Initiation model (odds of EVER breastfeeding), {note}:")
        log(init_tbl.round(3).to_string())
    except Exception as e:
        log(f"    (initiation model skipped: {e})")

    # ---- (b) proportional-hazards test on a subsample ----
    log(); log(f"(b) PH test on random subsample of {ph_n:,} children ...")
    sub = cc.sample(n=min(ph_n, len(cc)), random_state=seed)
    cph_sub = CoxPHFitter()
    cph_sub.fit(sub[BASE + COVARS], duration_col="duration_months", event_col="event",
                weights_col="weight", strata=["country", "survey_year"], robust=False)
    ph = proportional_hazard_test(cph_sub, sub[BASE + COVARS], time_transform="rank")
    ph_tbl = ph.summary
    log(f"    subsample strata: {sub.groupby(['country','survey_year']).ngroups}, C = {cph_sub.concordance_index_:.3f}")
    log(ph_tbl.round(4).to_string())

    # ---- (c) temporal sensitivity ----
    log(); log("(c) Temporal sensitivity (Gregorian survey year; Bikram Sambat years excluded)")
    temp_rows = [t_main.assign(model="All surveys")]
    for cut in (2010, 2015):
        d = cc[cc["year_gregorian"] >= cut]
        log(f"    >= {cut}: n = {len(d):,}, countries = {d['country'].nunique()}, "
            f"strata = {d.groupby(['country','survey_year']).ngroups}")
        cph_t, t_t = fit(d, f">= {cut}", robust)
        log(f"    C = {cph_t.concordance_index_:.3f}")
        temp_rows.append(t_t)
    temporal = pd.concat(temp_rows)
    log(temporal.pivot_table(index=temporal.index, columns="model", values="HR").round(3).to_string())

    sample = pd.DataFrame({
        "metric": ["n_complete_case", "countries", "strata", "n_never", "pct_never_weighted",
                   "C_main", "C_excl_never", "robust_SE", "ph_subsample_n"],
        "value": [len(cc), cc["country"].nunique(), cc.groupby(["country", "survey_year"]).ngroups,
                  n_never, pct_w, cph_main.concordance_index_, cph_ex.concordance_index_, robust, len(sub)],
    })
    with pd.ExcelWriter(out / "sensitivities_v23.xlsx") as xw:
        never_tbl.to_excel(xw, sheet_name="never_bf")
        if init_tbl is not None:
            init_tbl.to_excel(xw, sheet_name="initiation")
        ph_tbl.to_excel(xw, sheet_name="ph_test")
        temporal.to_excel(xw, sheet_name="temporal")
        sample.to_excel(xw, sheet_name="sample", index=False)
    (out / "r28_sensitivities.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nDone. Written to {out.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r".\02_extract_v23\combined_breastfeeding_data.csv")
    ap.add_argument("--out", default=r".\r28_sensitivities")
    ap.add_argument("--robust", action="store_true", help="sandwich SEs (slow); default model-based")
    ap.add_argument("--ph-n", type=int, default=200_000, help="subsample size for the PH test")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.data, a.out, a.robust, a.ph_n, a.seed)
