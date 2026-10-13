# 15_bootstrap_ci.py
# Cluster-bootstrap 95% CIs for the median duration of any breastfeeding, S(6) and S(24),
# for the 10 regions and the 12 largest countries (v25).
#
# Estimator as in 04: Turnbull NPMLE where the group has left-censored children (m4 = 93),
# Kaplan-Meier otherwise; weights v005/1e6; a median the curve never reaches is 36 months.
# Bootstrap: within each survey (source zip), PSUs (v021) are resampled with replacement,
# as many as the survey has; 200 replicates; percentile CIs. Children without a PSU code
# are treated as their own PSU. Each group's data are collapsed to unique (lower, upper)
# intervals with summed replicate weights before fitting, which gives the same estimate.
#
# Usage (from D:\early-weaning-analysis):  python analysis\15_bootstrap_ci.py [--reps 200] [--workers 12]
# Outputs: v25_extras\r35_bootstrap_ci_v25.xlsx and .txt

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(r".\02_extract_v25\combined_breastfeeding_data.csv")
RANK = Path(r".\04_comparative_v25\rankings\country_rankings_complete.xlsx")
REGION = Path(r".\04_comparative_v25\regional\regional_comparison.xlsx")
OUT = Path(r".\v25_extras")
CAP = 36.0
LEFT_FLAG = 0.30


def estimate(args):
    """(lo, up, event, w, turnbull) of one collapsed group -> (median, S6, S24)."""
    import warnings
    warnings.filterwarnings("ignore")
    from lifelines import KaplanMeierFitter
    lo, up, ev, w, turnbull = args
    keep = w > 0
    lo, up, ev, w = lo[keep], up[keep], ev[keep], w[keep]
    k = KaplanMeierFitter()
    if turnbull:
        k.fit_interval_censoring(lo, up, weights=w)
    else:
        k.fit(lo, ev, weights=w)
    m = k.median_survival_time_
    if isinstance(m, (pd.DataFrame, pd.Series)):
        m = float(np.asarray(m, dtype=float).ravel().mean())
    m = CAP if np.isinf(m) else float(m)
    sf = k.survival_function_

    def at(t):
        i = sf.index.get_indexer([t], method="ffill")[0]
        return float(sf.iloc[i, 0]) if i >= 0 else 1.0
    return m, at(6), at(24)


def main(a):
    OUT.mkdir(exist_ok=True)
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(str(s))

    t0 = time.time()
    df = pd.read_csv(DATA, usecols=["country", "source_zip", "v021", "t_lower", "t_upper",
                                    "duration_months", "event", "censor_type", "v005"],
                     low_memory=False)
    df["w"] = df["v005"] / 1e6
    df = df[df["w"] > 0].reset_index(drop=True)
    rank = pd.read_excel(RANK)
    df["region"] = df["country"].map(rank.set_index("country_code")["region"])

    # PSU index; children without v021 are their own PSU
    psu_key = df["source_zip"] + "_" + df["v021"].astype("Int64").astype(str)
    no_psu = df["v021"].isna()
    psu_key[no_psu] = "child_" + df.index[no_psu].astype(str)
    df["psu"], psu_labels = pd.factorize(psu_key)
    psu_survey = pd.Series(df["source_zip"].values, index=df["psu"]).groupby(level=0).first()
    psu_survey = pd.factorize(psu_survey.sort_index().values)[0]
    n_psu = len(psu_labels)
    surv_psus = [np.flatnonzero(psu_survey == s) for s in range(psu_survey.max() + 1)]
    log(f"{len(df):,} children with positive weight, {n_psu:,} PSUs in {len(surv_psus)} surveys "
        f"({int(no_psu.sum()):,} children without a PSU code resampled individually)")

    # groups
    largest = rank.nlargest(12, "n_children")
    groups = [("region", r, df.index[df["region"] == r].to_numpy())
              for r in sorted(df["region"].dropna().unique())]
    groups += [("country", f"{c.country_name} ({c.country_code})",
                df.index[df["country"] == c.country_code].to_numpy()) for c in largest.itertuples()]

    # collapse each group to cells
    lo_all = df["t_lower"].to_numpy(dtype=float)
    up_all = np.where(df["t_upper"].isna(), np.inf, df["t_upper"].to_numpy(dtype=float))
    dur_all = df["duration_months"].to_numpy(dtype=float)
    ev_all = df["event"].to_numpy(dtype=int)
    left_all = (df["censor_type"] == "left").to_numpy()
    w_all = df["w"].to_numpy()
    psu_all = df["psu"].to_numpy()
    G = []
    for kind, name, idx in groups:
        n_left = int(left_all[idx].sum())
        turnbull = n_left > 0
        if turnbull:
            key = pd.MultiIndex.from_arrays([lo_all[idx], up_all[idx]])
        else:
            key = pd.MultiIndex.from_arrays([dur_all[idx], ev_all[idx].astype(float)])
        cell, uniq = pd.factorize(key)
        a0 = uniq.get_level_values(0).to_numpy(dtype=float)
        a1 = uniq.get_level_values(1).to_numpy(dtype=float)
        if turnbull:
            lo_c, up_c, ev_c = a0, a1, np.zeros(len(a0), dtype=int)
        else:
            lo_c, up_c, ev_c = a0, a0, a1.astype(int)
        G.append(dict(kind=kind, name=name, idx=idx, cell=cell, ncell=len(a0), lo=lo_c, up=up_c,
                      ev=ev_c, turnbull=turnbull, n=len(idx), n_left=n_left,
                      surveys=df["source_zip"].iloc[idx].nunique(),
                      psus=len(np.unique(psu_all[idx]))))

    # replicate cell weights
    rng = np.random.default_rng(a.seed)
    rep_w = [np.empty((a.reps + 1, g["ncell"])) for g in G]
    for b in range(a.reps + 1):
        if b == 0:
            mult = np.ones(n_psu)  # the point estimate
        else:
            draws = np.concatenate([p[rng.integers(0, len(p), len(p))] for p in surv_psus])
            mult = np.bincount(draws, minlength=n_psu).astype(float)
        cw = w_all * mult[psu_all]
        for g, m in zip(G, rep_w):
            m[b] = np.bincount(g["cell"], weights=cw[g["idx"]], minlength=g["ncell"])
    log(f"replicate weights built ({time.time() - t0:.0f}s); fitting "
        f"{len(G) * (a.reps + 1):,} curves on {a.workers} workers")

    tasks = [(g["lo"], g["up"], g["ev"], m[b], g["turnbull"])
             for g, m in zip(G, rep_w) for b in range(a.reps + 1)]
    with Pool(a.workers) as pool:
        res = pool.map(estimate, tasks, chunksize=4)
    res = np.array(res).reshape(len(G), a.reps + 1, 3)
    log(f"fitted ({time.time() - t0:.0f}s)")

    # 04 values for comparison
    reg04 = pd.read_excel(REGION).set_index("region")
    rk04 = rank.set_index("country_code")
    rows = []
    for g, r in zip(G, res):
        pt, boot = r[0], r[1:]
        lo95 = np.percentile(boot, 2.5, axis=0)
        hi95 = np.percentile(boot, 97.5, axis=0)
        if g["kind"] == "region":
            m04 = reg04.loc[g["name"], "median_duration"]
            s6_04 = 1 - reg04.loc[g["name"], "pct_early_weaning"] / 100
            s24_04 = reg04.loc[g["name"], "pct_extended_bf"] / 100
        else:
            code = g["name"][-3:-1]
            m04 = rk04.loc[code, "median_duration_months"]
            s6_04 = 1 - rk04.loc[code, "pct_weaned_before_6mo"] / 100
            s24_04 = rk04.loc[code, "pct_still_bf_at_24mo"] / 100
        share_left = g["n_left"] / g["n"]
        rows.append({
            "group": g["kind"], "name": g["name"], "n": g["n"], "surveys": g["surveys"],
            "psus": g["psus"], "n_left_censored": g["n_left"], "pct_left_censored": 100 * share_left,
            "FLAG_left_censored_gt_30pct": share_left > LEFT_FLAG,
            "estimator": "Turnbull" if g["turnbull"] else "Kaplan-Meier",
            "median": pt[0], "median_lo95": lo95[0], "median_hi95": hi95[0],
            "median_at_36_cap_in_pct_of_reps": 100 * np.mean(boot[:, 0] >= CAP),
            "S6_pct": 100 * pt[1], "S6_lo95": 100 * lo95[1], "S6_hi95": 100 * hi95[1],
            "S24_pct": 100 * pt[2], "S24_lo95": 100 * lo95[2], "S24_hi95": 100 * hi95[2],
            "check_median_04": m04, "check_S6_04": 100 * s6_04, "check_S24_04": 100 * s24_04,
        })
    tab = pd.DataFrame(rows)
    tab["matches_04"] = ((tab["median"] - tab["check_median_04"]).abs() < 0.01) & \
                        ((tab["S6_pct"] - tab["check_S6_04"]).abs() < 0.01) & \
                        ((tab["S24_pct"] - tab["check_S24_04"]).abs() < 0.01)
    show = tab[["group", "name", "n", "pct_left_censored", "FLAG_left_censored_gt_30pct", "estimator",
                "median", "median_lo95", "median_hi95", "S6_pct", "S6_lo95", "S6_hi95",
                "S24_pct", "S24_lo95", "S24_hi95", "matches_04"]]
    pd.set_option("display.width", 250)
    log("\n" + show.round(1).to_string(index=False))
    with pd.ExcelWriter(OUT / "r35_bootstrap_ci_v25.xlsx") as xw:
        tab.to_excel(xw, sheet_name="bootstrap_ci", index=False)
        pd.DataFrame([{"replicates": a.reps, "seed": a.seed, "resampling": "PSUs within survey",
                       "CI": "percentile 2.5-97.5"}]).to_excel(xw, sheet_name="settings", index=False)
    (OUT / "r35_bootstrap_ci_v25.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDone. Written to {OUT.resolve()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=200)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--seed", type=int, default=2025)
    main(p.parse_args())
