#!/usr/bin/env python3
"""
r25_turnbull_summaries.py
Descriptive survival results for the v2.3 extraction, which contains three kinds
of observation:

    exact  duration reported, or never breastfed (event at time 0)
    right  still breastfeeding at interview (censored at current age)
    left   stopped before interview, duration never recorded (m4 = 93)

An ordinary Kaplan-Meier estimator cannot use the left-censored children; the
Turnbull NPMLE can, so this script estimates the survival curve from the interval
bounds [t_lower, t_upper] for the whole sample, and reports the weighted median
duration of any breastfeeding and the probability of still breastfeeding at 6, 12
and 24 months, by country, by region and by urban/rural residence.

For comparison it also reports, for each group, the ordinary Kaplan-Meier estimate
computed after dropping the left-censored children, which is what the submitted
analysis effectively did. Surveys with no retrospective durations at all are
listed separately: for those, only the Turnbull column is meaningful.

Usage (from D:\\DiscoverPublicHealth\\early-weaning-analysis):
    python r25_turnbull_summaries.py --data ".\\02_extract_v23\\combined_breastfeeding_data.csv"

Outputs (folder r25_turnbull):
    turnbull_summaries.xlsx   sheets: overall, by_region, by_residence, by_country,
                                      survey_censoring
    r25_turnbull.txt
"""
import argparse
import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

warnings.filterwarnings("ignore")
TIMES = [6, 12, 24]


def load_assign_region(path_03: str):
    spec = importlib.util.spec_from_file_location("mod03", path_03)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.assign_region


def _at(sf: pd.DataFrame, t: float) -> float:
    col = sf.columns[0]
    idx = sf.index.get_indexer([t], method="ffill")[0]
    return float(sf.iloc[idx][col]) if idx >= 0 else 1.0


def turnbull(g: pd.DataFrame) -> dict:
    """Weighted Turnbull NPMLE from the interval bounds."""
    lo = g["t_lower"].to_numpy(dtype=float)
    up = g["t_upper"].to_numpy(dtype=float)
    up = np.where(np.isnan(up), np.inf, up)
    k = KaplanMeierFitter()
    k.fit_interval_censoring(lo, up, weights=g["weight"].to_numpy(dtype=float))
    sf = k.survival_function_[[k.survival_function_.columns[0]]]
    med = k.median_survival_time_
    # with interval censoring lifelines returns a DataFrame of bounds for the median
    if isinstance(med, (pd.DataFrame, pd.Series)):
        med = float(np.asarray(med, dtype=float).ravel().mean())
    med = float(med)
    return {"tb_median": (np.nan if not np.isfinite(med) else med),
            "tb_S6": _at(sf, 6), "tb_S12": _at(sf, 12), "tb_S24": _at(sf, 24)}


def plain_km(g: pd.DataFrame) -> dict:
    """Ordinary KM after dropping left-censored children (the old approach)."""
    d = g[g["censor_type"] != "left"]
    if d.empty or d["event"].sum() == 0:
        return {"km_median": np.nan, "km_S6": np.nan, "km_S12": np.nan, "km_S24": np.nan,
                "km_n": len(d)}
    k = KaplanMeierFitter()
    k.fit(d["duration_months"], event_observed=d["event"], weights=d["weight"])
    sf = k.survival_function_
    med = k.median_survival_time_
    return {"km_median": (np.nan if not np.isfinite(med) else float(med)),
            "km_S6": _at(sf, 6), "km_S12": _at(sf, 12), "km_S24": _at(sf, 24),
            "km_n": len(d)}


def summarise(g: pd.DataFrame) -> dict:
    out = {"n": len(g),
           "n_exact": int((g.censor_type == "exact").sum()),
           "n_right": int((g.censor_type == "right").sum()),
           "n_left": int((g.censor_type == "left").sum())}
    out["pct_left"] = 100 * out["n_left"] / out["n"]
    out.update(turnbull(g))
    out.update(plain_km(g))
    return out


def main(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lines = []
    def log(s=""):
        print(s, flush=True); lines.append(str(s))

    df = pd.read_csv(a.data, low_memory=False)
    log(f"Rows loaded:                 {len(df):,}")
    df = df[df["country"].notna()].copy()
    df["weight"] = pd.to_numeric(df["v005"], errors="coerce") / 1e6
    df = df[df["weight"] > 0]
    if "t_upper" in df.columns:
        df["t_upper"] = pd.to_numeric(df["t_upper"], errors="coerce")
    log(f"With country and weight > 0: {len(df):,}")
    log(f"  exact {int((df.censor_type=='exact').sum()):,}, "
        f"right {int((df.censor_type=='right').sum()):,}, "
        f"left {int((df.censor_type=='left').sum()):,}")

    # which surveys carry retrospective durations at all
    sv = df.groupby(["country", "survey_year"]).apply(
        lambda x: pd.Series({
            "n": len(x),
            "n_left": int((x.censor_type == "left").sum()),
            "n_events_gt0": int(((x.event == 1) & (x.duration_months > 0)).sum()),
        }), include_groups=False).reset_index()
    sv["retrospective"] = sv.n_events_gt0 > 0
    log(f"country-surveys: {len(sv)}, with retrospective durations: {int(sv.retrospective.sum())}, "
        f"current-status only: {int((~sv.retrospective).sum())}")
    log(f"children in current-status-only surveys: {int(sv.loc[~sv.retrospective,'n'].sum()):,}")

    df["region"] = df["country"].apply(load_assign_region(a.script03))

    log(); log("OVERALL")
    overall = pd.DataFrame([summarise(df)], index=["All children"])
    log(overall.round(3).to_string())

    log(); log("BY REGION (tb_ = Turnbull, all children; km_ = ordinary KM, left-censored dropped)")
    reg = pd.DataFrame({r: summarise(g) for r, g in df.groupby("region")}).T
    reg = reg.sort_values("tb_median", ascending=False)
    log(reg.round(3).to_string())

    res = pd.DataFrame()
    if "urban" in df.columns:
        d = df[df["urban"].notna()]
        res = pd.DataFrame({name: summarise(d[d.urban == k])
                            for k, name in [(0, "Rural"), (1, "Urban")]}).T
        log(); log("BY RESIDENCE"); log(res.round(3).to_string())

    log(); log("BY COUNTRY (n >= 100)")
    cty = pd.DataFrame({c: summarise(g) for c, g in df.groupby("country") if len(g) >= 100}).T
    cty["region"] = [load_assign_region(a.script03)(c) for c in cty.index]
    cty = cty.sort_values("tb_median", ascending=False)
    log(cty.round(3).head(20).to_string())
    log(); log(f"Turnbull median range across {len(cty)} countries: "
               f"{cty.tb_median.min():.0f}-{cty.tb_median.max():.0f} months "
               f"({int(cty.tb_median.isna().sum())} without an estimable median)")
    log(f"countries with Turnbull median >= 24 months: {int((cty.tb_median >= 24).sum())}")

    with pd.ExcelWriter(out / "turnbull_summaries.xlsx") as xw:
        overall.to_excel(xw, sheet_name="overall")
        reg.to_excel(xw, sheet_name="by_region")
        if len(res):
            res.to_excel(xw, sheet_name="by_residence")
        cty.to_excel(xw, sheet_name="by_country")
        sv.sort_values(["country", "survey_year"]).to_excel(xw, sheet_name="survey_censoring", index=False)
    (out / "r25_turnbull.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDone. Written to {out.resolve()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=r".\02_extract_v23\combined_breastfeeding_data.csv")
    p.add_argument("--script03", default=r".\03_survival_analysis.py")
    p.add_argument("--out", default=r".\r25_turnbull")
    main(p.parse_args())
