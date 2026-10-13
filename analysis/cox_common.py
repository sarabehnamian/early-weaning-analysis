# cox_common.py
# Shared by 13, 14, 16, 17 and 19: builds the Cox analytic sample of 05 from the v25 extraction
# and fits the stratified Cox model with model-based SEs.
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

DATA = Path(r".\02_extract_v25\combined_breastfeeding_data.csv")
COVARS = ["urban", "wealth_q1", "wealth_q2", "wealth_q4", "wealth_q5",
          "educ_none", "educ_secondary_plus"]
COVARS_NOWEALTH = ["urban", "educ_none", "educ_secondary_plus"]
BASE = ["duration_months", "event", "weight", "country", "survey_year"]
LABELS = {"urban": "Urban residence (ref. rural)", "wealth_q1": "Poorest quintile (Q1)",
          "wealth_q2": "Second quintile (Q2)", "wealth_q4": "Fourth quintile (Q4)",
          "wealth_q5": "Richest quintile (Q5)", "educ_none": "No formal education",
          "educ_secondary_plus": "Secondary or higher"}
USECOLS = ["country", "survey_year", "source_zip", "duration_months", "event", "censor_type",
           "never_breastfed", "v005", "v021", "v022", "v023", "urban", "wealth_quintile",
           "mother_educ"]


def load(path=DATA):
    df = pd.read_csv(path, usecols=lambda c: c in USECOLS, low_memory=False)
    df["weight"] = pd.to_numeric(df["v005"], errors="coerce") / 1e6
    return df


def retrospective_surveys(df):
    """Country-surveys with at least one cessation after time zero (as in 05)."""
    d = df[df["weight"] > 0]
    ev = ((d["event"] == 1) & (d["duration_months"] > 0)).groupby(
        [d["country"], d["survey_year"]]).sum()
    return set(ev[ev > 0].index)


def cox_sample(df, wealth=True):
    """Cox sample of 05: positive weight, retrospective survey, not left-censored, complete covariates + PSU."""
    keep = retrospective_surveys(df)
    d = df[df["weight"] > 0]
    d = d[[(c, y) in keep for c, y in zip(d["country"], d["survey_year"])]]
    d = d[d["censor_type"] != "left"]
    need = ["urban", "mother_educ", "v021"] + (["wealth_quintile"] if wealth else [])
    d = d.dropna(subset=need).copy()
    if wealth:
        for q in [1, 2, 4, 5]:
            d[f"wealth_q{q}"] = (d["wealth_quintile"] == q).astype(int)
    d["educ_none"] = (d["mother_educ"] == 0).astype(int)
    d["educ_secondary_plus"] = (d["mother_educ"] >= 2).astype(int)
    return d


def fit(d, covars=COVARS, **kw):
    cph = CoxPHFitter()
    cph.fit(d[BASE + covars], duration_col="duration_months", event_col="event",
            weights_col="weight", strata=["country", "survey_year"], robust=False, **kw)
    return cph


def hr_table(cph, model):
    s = cph.summary
    return pd.DataFrame({"model": model, "covariate": s.index,
                         "label": [LABELS.get(i, i) for i in s.index],
                         "HR": s["exp(coef)"].values, "lower95": s["exp(coef) lower 95%"].values,
                         "upper95": s["exp(coef) upper 95%"].values, "p": s["p"].values})


def describe(d):
    return {"n": len(d), "events": int(d["event"].sum()), "countries": d["country"].nunique(),
            "strata": d.groupby(["country", "survey_year"]).ngroups}
