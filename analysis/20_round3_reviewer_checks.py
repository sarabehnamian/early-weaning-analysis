# 20_round3_reviewer_checks.py
# Analyses for the third-round comments of Reviewer 2 (Discover Public Health, 31 Oct 2026).
# Reads the locked v25 extraction; nothing in the main results is refitted or overwritten.
#
#  A) R2.1  pooling: pooled and regional median, S(6), S(12), S(24) under different survey weights
#           dhs            v005/1e6 as in the paper (each survey counts by its sample size)
#           survey_equal   every survey has the same total weight
#           country_equal  every country has the same total weight, its surveys equal within it
#           latest_only    the most recent survey of each country only, countries equal
#           population     every country weighted by its under-5 population (only with --pop)
#  B) R2.3  the 93 current-status datasets (m4 = 93, no recorded durations) excluded:
#           Kaplan-Meier on the 185 retrospective datasets vs the Turnbull estimates on all 278
#  C) R2.2  recent surveys only (2010 onwards, 2015 onwards): descriptive estimates
#  D) R2.4  selection into the Cox sample:
#           D1 children in the 85 Cox strata vs the 193 excluded country-surveys
#           D2 the 55 Cox countries vs the 19 countries with no survey in the Cox model
#           D3 proportional-hazards model for current-status data (complementary log-log
#              regression of "stopped by the interview" on the same covariates, a spline in
#              child age as the baseline and a fixed effect per country-survey). exp(coef) is
#              a hazard ratio under proportional hazards, so it can be compared with the Cox HRs.
#              a) the current-status surveys the Cox model cannot use (mostly 2010-2024)
#              b) the Cox surveys themselves, reduced to current status (checks the method
#                 against the Cox estimates)
#  E) R2.7  medians: for every country and every within-country subgroup, whether the curve
#           reaches 50% at all, the uncapped median, and S(36)
#
# Usage (from D:\early-weaning-analysis):
#   python analysis\20_round3_reviewer_checks.py
#   python analysis\20_round3_reviewer_checks.py --pop un_wpp_under5.csv   # adds population weights
#     (CSV with columns: country = DHS code, pop = under-5 population, any reference year)
#   python analysis\20_round3_reviewer_checks.py --parts FBDEG            # only these parts
#  F) estimator check: the submitted pipeline used Turnbull where left-censored children are
#     present and Kaplan-Meier elsewhere; lifelines reports the first as P(duration >= t) and the
#     second as P(duration > t). F compares that with one estimator for every group.
# Outputs: v25_extras\r40_round3_checks.xlsx and .txt
import argparse
import gc
import importlib.util
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from cox_common import COVARS, LABELS, cox_sample, load as load_cox

warnings.filterwarnings("ignore")
DATA = Path("02_extract_v25") / "combined_breastfeeding_data.csv"
OUT = Path("v25_extras")
CAP = 36.0
TIMES = [6, 12, 24, 36]
COLS = ["country", "survey_year", "source_zip", "v021", "t_lower", "t_upper", "duration_months",
        "event", "censor_type", "v005", "urban", "wealth_quintile", "mother_educ",
        "never_breastfed", "age_months"]
lines = []


def log(s=""):
    print(s, flush=True)
    lines.append(str(s))


# ---------------------------------------------------------------- survival estimates
def estimate(g, w=None, mixed=False):
    """Weighted survival estimates on collapsed cells.

    Default: the interval-censored (Turnbull) estimator for EVERY group. With exact and
    right-censored data only it reduces to Kaplan-Meier, but lifelines reports its S(t) as
    P(duration >= t) whereas KaplanMeierFitter.fit reports P(duration > t). With durations in
    whole months and heaping at 6, 12 and 24, the two differ by a month in the median and by up
    to ~15 points in S(24), so one estimator must be used for every group.
    mixed=True reproduces the submitted pipeline (Turnbull only where left-censored children
    are present, Kaplan-Meier otherwise), for comparison."""
    from lifelines import KaplanMeierFitter
    w = g["w"].to_numpy() if w is None else np.asarray(w, dtype=float)
    keep = w > 0
    g, w = g[keep], w[keep]
    n_left = int((g["censor_type"] == "left").sum())
    use_tb = (n_left > 0) or not mixed
    out = {"n": len(g), "surveys": g["source_zip"].nunique(), "countries": g["country"].nunique(),
           "pct_left": 100 * n_left / max(len(g), 1),
           "estimator": "Turnbull" if use_tb else "Kaplan-Meier"}
    if len(g) < 30:
        return out
    k = KaplanMeierFitter()
    if use_tb:
        lo = g["t_lower"].to_numpy(dtype=float)
        up = np.where(g["t_upper"].isna(), np.inf, g["t_upper"].to_numpy(dtype=float))
        cell = pd.DataFrame({"lo": lo, "up": up, "w": w}).groupby(["lo", "up"], sort=False)["w"].sum()
        k.fit_interval_censoring(cell.index.get_level_values(0).to_numpy(),
                                 cell.index.get_level_values(1).to_numpy(), weights=cell.to_numpy())
        med = k.median_survival_time_
        med = float(np.asarray(med, dtype=float).ravel().mean()) if hasattr(med, "shape") else float(med)
    else:
        cell = pd.DataFrame({"t": g["duration_months"].to_numpy(dtype=float),
                             "e": g["event"].to_numpy(dtype=int), "w": w}
                            ).groupby(["t", "e"], sort=False)["w"].sum()
        k.fit(cell.index.get_level_values(0).to_numpy(), cell.index.get_level_values(1).to_numpy(),
              weights=cell.to_numpy())
        med = float(k.median_survival_time_)
    sf = k.survival_function_
    def at(t):
        i = sf.index.get_indexer([t], method="ffill")[0]
        return 100 * float(sf.iloc[i, 0]) if i >= 0 else 100.0
    out.update({"median_reached": bool(np.isfinite(med)),
                "median_uncapped": med if np.isfinite(med) else np.nan,
                "median_as_reported": min(med, CAP) if np.isfinite(med) else CAP,
                "last_time": float(sf.index.max())})
    out.update({f"S{t}": at(t) for t in TIMES})
    return out


def table(groups, label):
    rows = []
    for name, g, w in groups:
        rows.append({"analysis": label, "group": name, **estimate(g, w)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- weighting schemes
def scheme_weights(d, scheme, pop=None):
    w = d["w"]
    s_tot = w.groupby(d["source_zip"]).transform("sum")
    if scheme == "dhs":
        return w
    if scheme == "survey_equal":
        return w / s_tot
    n_s = d.groupby("country")["source_zip"].transform("nunique")
    if scheme in ("country_equal", "latest_only"):
        return w / s_tot / n_s
    if scheme == "population":
        return w / s_tot / n_s * d["country"].map(pop).astype(float)
    raise ValueError(scheme)


def part_a(df, pop):
    log("\nA) R2.1 POOLING: pooled and regional estimates under different weights")
    latest = df.groupby("country")["survey_year"].transform("max") == df["survey_year"]
    schemes = ["dhs", "survey_equal", "country_equal", "latest_only"] + (["population"] if pop else [])
    res = []
    for sc in schemes:
        d = df[latest] if sc == "latest_only" else df
        if sc == "population":
            missing = sorted(set(d["country"]) - set(pop))
            if missing:
                log(f"  population weights: no population for {missing}; those countries dropped")
                d = d[d["country"].isin(pop)]
        w = scheme_weights(d, sc, pop)
        groups = [("Pooled", d, w)] + [(r, d[d.region == r], w[d.region == r])
                                       for r in sorted(d["region"].unique())]
        t = table(groups, sc)
        res.append(t)
        log(f"  {sc:14s} pooled: median {t.iloc[0]['median_as_reported']:.0f}, "
            f"S6 {t.iloc[0]['S6']:.1f}, S24 {t.iloc[0]['S24']:.1f}")
    res = pd.concat(res, ignore_index=True)
    wide = res.pivot(index="group", columns="analysis", values="median_as_reported")
    log("\n  Median (months) by weighting scheme:\n" + wide.round(0).to_string())
    log("\n  S(24) % by weighting scheme:\n"
        + res.pivot(index="group", columns="analysis", values="S24").round(1).to_string())
    return res


def part_b(df):
    log("\nB) R2.3 CURRENT-STATUS DATASETS EXCLUDED (same estimator on retrospective surveys only)")
    r = df[df.retrospective]
    gone = sorted(set(df.country) - set(r.country))
    log(f"  retrospective datasets: {r.source_zip.nunique()} ({len(r):,} children, "
        f"{r.country.nunique()} countries); countries lost entirely: {', '.join(gone)}")
    log(f"  left-censored children left in retrospective datasets: "
        f"{int((r.censor_type == 'left').sum()):,}")
    out = []
    ALL, RETRO = "all 278 datasets", "185 retrospective datasets only"
    for lab, d in [(ALL, df), (RETRO, r)]:
        groups = [("Pooled", d, None)]
        groups += [(f"Residence: {k}", d[d.urban == v], None) for v, k in [(0, "rural"), (1, "urban")]]
        groups += [(f"Region: {x}", d[d.region == x], None) for x in sorted(df.region.unique())]
        groups += [(f"Country: {c}", d[d.country == c], None) for c in sorted(df.country.unique())]
        out.append(table(groups, lab))
    out = pd.concat(out, ignore_index=True)
    cmp = pd.DataFrame(index=out["group"].unique())
    for v in ["median_as_reported", "S6", "S24"]:
        p = out.pivot(index="group", columns="analysis", values=v)
        short = "median" if v == "median_as_reported" else v
        cmp[f"{short}_all"] = p[ALL]
        cmp[f"{short}_retro"] = p[RETRO]
    cmp["d_median"] = cmp.median_retro - cmp.median_all
    cmp["d_S24"] = cmp.S24_retro - cmp.S24_all
    keep = [i for i in cmp.index if not i.startswith("Country")]
    log(cmp.loc[keep].round(1).to_string())
    c = cmp[[i.startswith("Country") for i in cmp.index]].dropna()
    log(f"  countries: median changed by >2 months in {int((c.d_median.abs() > 2).sum())} of "
        f"{len(c)}; S24 by >10 points in {int((c.d_S24.abs() > 10).sum())}")
    return out, cmp.reset_index()


def part_c(df):
    log("\nC) R2.2 RECENT SURVEYS (descriptive, DHS weights)")
    out = []
    for lab, d in [("all surveys", df), ("surveys from 2010", df[df.survey_year >= 2010]),
                   ("surveys from 2015", df[df.survey_year >= 2015])]:
        groups = [("Pooled", d, None)]
        groups += [(f"Residence: {k}", d[d.urban == v], None) for v, k in [(0, "rural"), (1, "urban")]]
        groups += [(f"Region: {x}", d[d.region == x], None) for x in sorted(d.region.unique())]
        t = table(groups, lab)
        out.append(t)
        p = t.iloc[0]
        log(f"  {lab:18s} n {p.n:,}, {p.countries} countries, {p.surveys} surveys: median "
            f"{p.median_as_reported:.0f}, S6 {p.S6:.1f}, S24 {p.S24:.1f}; rural/urban medians "
            f"{t.iloc[1].median_as_reported:.0f}/{t.iloc[2].median_as_reported:.0f}")
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- selection
def characteristics(d, name):
    w = d["w"]
    def wmean(x):
        m = x.notna() & (w > 0)
        return 100 * np.average(x[m], weights=w[m]) if m.any() else np.nan
    est = estimate(d)
    return {"group": name, "children": len(d), "countries": d.country.nunique(),
            "surveys": d.source_zip.nunique(),
            "years": f"{int(d.survey_year.min())}-{int(d.survey_year.max())}",
            "median_survey_year": float(np.median(d.groupby('source_zip').survey_year.first())),
            "pct_urban": wmean(d.urban.astype(float)),
            "pct_no_educ": wmean((d.mother_educ == 0).where(d.mother_educ.notna())),
            "pct_secondary_plus": wmean((d.mother_educ >= 2).where(d.mother_educ.notna())),
            "pct_never_bf": wmean(d.never_breastfed.astype(float)),
            "pct_wealth_missing": 100 * d.wealth_quintile.isna().mean(),
            "median": est.get("median_as_reported"), "S6": est.get("S6"), "S24": est.get("S24"),
            "estimator": est.get("estimator")}


def part_d_descriptive(df, cox_keys, cox_countries):
    log("\nD1/D2) R2.4 WHO IS IN THE COX SAMPLE")
    in_cox = pd.Series([(c, y) in cox_keys for c, y in zip(df.country, df.survey_year)], index=df.index)
    rows = [characteristics(df, "All children (278 country-surveys)"),
            characteristics(df[in_cox], "Cox strata (85 country-surveys), all their children"),
            characteristics(df[~in_cox], "Excluded country-surveys (193)"),
            characteristics(df[~in_cox & ~df.retrospective], "  excluded: current-status surveys"),
            characteristics(df[~in_cox & df.retrospective], "  excluded: retrospective but no wealth/PSU"),
            characteristics(df[df.country.isin(cox_countries)], "55 countries with a Cox survey"),
            characteristics(df[~df.country.isin(cox_countries)], "19 countries with no Cox survey")]
    t = pd.DataFrame(rows)
    log(t.drop(columns=["estimator"]).round(1).to_string(index=False))
    return t


def cloglog_fit(d, label, chunk=100_000):
    """Complementary log-log regression of 'stopped by the interview', fitted by Fisher scoring
    with the design built 100,000 rows at a time, so memory stays small (the full design matrix
    is never formed). Columns: covariates, a natural cubic spline in child age (the baseline),
    one fixed effect per country-survey (first dropped). Sandwich variance clustered on
    survey-specific PSUs."""
    import patsy
    t0 = time.time()
    d = d.dropna(subset=["urban", "wealth_quintile", "mother_educ", "v021", "age_months"])
    d = d[(d.w > 0) & (d.age_months >= 0)]
    y = ((d.censor_type == "left") | (d.event == 1)).to_numpy(dtype=float)
    # country-surveys with no variation in the outcome add nothing and break the fit
    zips = d["source_zip"].to_numpy()
    zs = pd.Series(y).groupby(zips)
    ok = pd.Series(zips).map((zs.min() != zs.max())).to_numpy()
    d, y = d[ok], y[ok]
    C = np.column_stack([d["urban"].to_numpy(float)]
                        + [(d.wealth_quintile == q).to_numpy(float) for q in [1, 2, 4, 5]]
                        + [(d.mother_educ == 0).to_numpy(float), (d.mother_educ >= 2).to_numpy(float)])
    SPL = np.asarray(patsy.dmatrix("cr(age_months, df=6) - 1", {"age_months": d["age_months"].to_numpy(float)}))
    fe, fe_lab = pd.factorize(d["source_zip"].to_numpy(), sort=True)
    w = d["w"].to_numpy(float)
    psu = pd.factorize(d["source_zip"].astype(str) + "_" + d["v021"].astype(int).astype(str))[0]
    n, k, ns, nf = len(y), C.shape[1], SPL.shape[1], len(fe_lab) - 1
    p = k + ns + nf

    def X(sl):
        A = np.zeros((sl.stop - sl.start, p))
        A[:, :k] = C[sl]
        A[:, k:k + ns] = SPL[sl]
        f = fe[sl]
        rows = np.flatnonzero(f > 0)
        A[rows, k + ns + f[rows] - 1] = 1.0
        return A

    def pieces(beta, sl):
        A = X(sl)
        eta = np.clip(A @ beta, -30, 5)
        mu = np.clip(-np.expm1(-np.exp(eta)), 1e-12, 1 - 1e-12)
        dmu = np.exp(eta - np.exp(eta))
        return A, eta, mu, dmu

    slices = [slice(i, min(i + chunk, n)) for i in range(0, n, chunk)]
    beta = np.zeros(p)
    beta[k:k + ns] = np.log(-np.log(1 - np.clip(y.mean(), 0.01, 0.99)))
    for it in range(60):
        XtWX = np.zeros((p, p)); XtWz = np.zeros(p)
        for sl in slices:
            A, eta, mu, dmu = pieces(beta, sl)
            W = w[sl] * dmu ** 2 / (mu * (1 - mu))
            z = eta + (y[sl] - mu) / dmu
            XtWX += A.T @ (A * W[:, None]); XtWz += A.T @ (W * z)
        new = np.linalg.solve(XtWX, XtWz)
        step = np.max(np.abs(new - beta))
        beta = new
        if step < 1e-8:
            break
    # sandwich: bread = inverse Fisher information, meat = PSU sums of weighted scores
    G = np.zeros((psu.max() + 1, p))
    XtWX = np.zeros((p, p))
    for sl in slices:
        A, eta, mu, dmu = pieces(beta, sl)
        XtWX += A.T @ (A * (w[sl] * dmu ** 2 / (mu * (1 - mu)))[:, None])
        u = (w[sl] * (y[sl] - mu) * dmu / (mu * (1 - mu)))[:, None] * A
        np.add.at(G, psu[sl], u)
    n_g = G.shape[0]
    bread = np.linalg.inv(XtWX)
    V = bread @ (G.T @ G) @ bread * n_g / (n_g - 1)
    b, se = beta[:k], np.sqrt(np.diag(V))[:k]
    t = pd.DataFrame({"model": label, "covariate": COVARS, "label": [LABELS[c] for c in COVARS],
                      "HR": np.exp(b), "lower95": np.exp(b - 1.96 * se), "upper95": np.exp(b + 1.96 * se),
                      "p": 2 * stats.norm.sf(np.abs(b / se)), "children": n, "countries": d.country.nunique(),
                      "country_surveys": nf + 1, "psus": n_g, "iterations": it + 1,
                      "years": f"{int(d.survey_year.min())}-{int(d.survey_year.max())}"})
    log(f"  {label}: {n:,} children, {d.country.nunique()} countries, {nf + 1} surveys, "
        f"{n_g:,} PSUs, {it + 1} iterations ({time.time() - t0:.0f}s)")
    log(t.set_index("label")[["HR", "lower95", "upper95"]].round(3).to_string())
    return t


def part_d_cloglog(df, cox_zips):
    log("\nD3) R2.4 PROPORTIONAL-HAZARDS MODEL FOR CURRENT-STATUS DATA (complementary log-log)")
    cs = df[~df.retrospective]
    a = cloglog_fit(cs, "a) current-status surveys excluded from the Cox model")
    b = cloglog_fit(df[df.source_zip.isin(cox_zips) & (df.censor_type != "left")],
                    "b) Cox surveys reduced to current status (method check)")
    return pd.concat([a, b], ignore_index=True)


# ---------------------------------------------------------------- one estimator for all groups
def part_f(df):
    log("\nF) ESTIMATOR CHECK: submitted pipeline (Kaplan-Meier where no child is left-censored)"
        " vs one estimator for every group")
    rows = []
    groups = [("Pooled", df)] + [(f"Region: {r}", g) for r, g in df.groupby("region")]
    groups += [(f"Residence: {k}", df[df.urban == v]) for v, k in [(0, "rural"), (1, "urban")]]
    groups += [(f"Country: {c}", g) for c, g in df.groupby("country")]
    for name, g in groups:
        a, b = estimate(g, mixed=True), estimate(g)
        rows.append({"group": name, "n": a["n"], "pct_left": a["pct_left"],
                     "estimator_submitted": a["estimator"],
                     "median_submitted": a.get("median_as_reported"), "median_unified": b.get("median_as_reported"),
                     "ceased6_submitted": 100 - a.get("S6", np.nan), "ceased6_unified": 100 - b.get("S6", np.nan),
                     "S12_submitted": a.get("S12"), "S12_unified": b.get("S12"),
                     "S24_submitted": a.get("S24"), "S24_unified": b.get("S24")})
    t = pd.DataFrame(rows)
    ch = t[(t.median_submitted != t.median_unified) | ((t.S24_submitted - t.S24_unified).abs() > 0.05)
           | ((t.ceased6_submitted - t.ceased6_unified).abs() > 0.05)]
    log(f"  groups that change: {len(ch)} of {len(t)}")
    log(ch.drop(columns=["n", "pct_left"]).round(1).to_string(index=False))
    u = t[t.group.str.startswith("Country")]
    log(f"  countries with median >= 24: submitted {int((u.median_submitted >= 24).sum())}, "
        f"unified {int((u.median_unified >= 24).sum())}; S24 >= 50%: submitted "
        f"{int((u.S24_submitted >= 50).sum())}, unified {int((u.S24_unified >= 50).sum())}; "
        f"ceased before 6 months > 20%: submitted {int((u.ceased6_submitted > 20).sum())}, "
        f"unified {int((u.ceased6_unified > 20).sum())}")
    return t


# ---------------------------------------------------------------- medians not reached
def part_e(df):
    log("\nE) R2.7 MEDIANS THAT ARE NOT REACHED")
    rows = []
    subsets = [("All children", None),
               ("Rural", lambda d: d.urban == 0), ("Urban", lambda d: d.urban == 1),
               ("Poorest quintile", lambda d: d.wealth_quintile == 1),
               ("Richest quintile", lambda d: d.wealth_quintile == 5),
               ("No education", lambda d: d.mother_educ == 0),
               ("Secondary or higher", lambda d: d.mother_educ >= 2)]
    for c, d in df.groupby("country"):
        for name, f in subsets:
            g = d if f is None else d[f(d)]
            if len(g) < 30:
                continue
            rows.append({"country": c, "subgroup": name, **estimate(g)})
    t = pd.DataFrame(rows)
    nr = t[~t.median_reached.astype(bool)]
    log(f"  groups whose curve never falls to 50%: {len(nr)}")
    log(nr[["country", "subgroup", "n", "estimator", "last_time", "S24", "S36"]].round(1)
        .to_string(index=False))
    return t


def main(a):
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    df = pd.read_csv(DATA, usecols=lambda c: c in COLS, low_memory=False)
    df["w"] = pd.to_numeric(df["v005"], errors="coerce") / 1e6
    df = df[df["w"] > 0].copy()
    spec = importlib.util.spec_from_file_location("s03", "03_survival_analysis.py")
    s03 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s03)
    df["region"] = df["country"].apply(s03.assign_region)
    ev_gt0 = ((df.event == 1) & (df.duration_months > 0)).groupby([df.country, df.survey_year]).transform("sum")
    df["retrospective"] = ev_gt0 > 0
    log(f"{len(df):,} children with positive weight, {df.country.nunique()} countries, "
        f"{df.source_zip.nunique()} datasets ({int(df.groupby('source_zip').retrospective.first().sum())} "
        f"retrospective); loaded in {time.time() - t0:.0f}s")

    pop = None
    if a.pop:
        p = pd.read_csv(a.pop)
        pop = dict(zip(p["country"].str.strip(), p["pop"]))

    cox_d = cox_sample(load_cox())
    cox_keys = set(zip(cox_d.country, cox_d.survey_year))
    cox_countries = set(cox_d.country)
    cox_zips = set(cox_d.source_zip)
    log(f"Cox sample: {len(cox_d):,} children, {len(cox_countries)} countries, {len(cox_keys)} strata")
    del cox_d
    gc.collect()

    parts = a.parts.upper()
    res = {}
    if "F" in parts:
        res["F_estimator_check"] = part_f(df)
    if "A" in parts:
        res["A_pooling"] = part_a(df, pop)
    if "B" in parts:
        res["B_no_current_status"], res["B_comparison"] = part_b(df)
    if "C" in parts:
        res["C_recent_surveys"] = part_c(df)
    if "D" in parts:
        res["D_selection"] = part_d_descriptive(df, cox_keys, cox_countries)
    if "G" in parts:  # the cloglog models
        res["D_cloglog_HR"] = part_d_cloglog(df, cox_zips)
    if "E" in parts:
        res["E_medians"] = part_e(df)

    tag = "" if parts == "ABCDEFG" else f"_{parts}"
    xl = OUT / f"r40_round3_checks{tag}.xlsx"
    with pd.ExcelWriter(xl) as xw:
        for k, v in res.items():
            v.to_excel(xw, sheet_name=k[:31], index=False)
    (OUT / f"r40_round3_checks{tag}.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDone in {time.time() - t0:.0f}s. Written to {xl.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", default=None, help="CSV: country (DHS code), pop (under-5 population)")
    ap.add_argument("--parts", default="ABCDEFG",
                    help="which parts to run, e.g. FBDEG (G = the cloglog models of part D)")
    main(ap.parse_args())
