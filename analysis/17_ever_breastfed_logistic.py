# 17_ever_breastfed_logistic.py
# Initiation model on v25: weighted logistic regression of EVER breastfeeding
# (1 - never_breastfed, i.e. m4 != 94) on the primary-model covariates, in the
# primary Cox sample (05: retrospective-duration surveys, complete residence, wealth,
# education and PSU, positive weight). Same specification as the initiation model in
# analysis/06_cox_sensitivity_checks.py (v24: OR urban 0.84, no education 0.74,
# secondary+ 0.54, Q1 0.87, Q5 1.08): statsmodels binomial GLM, freq_weights = v005/1e6,
# cluster-robust SEs.
#
# Clustering is reported two ways:
#   v021 alone       as in 06/v24; PSU numbers repeat across surveys, so this pools
#                    PSU k of every survey into one cluster
#   survey + v021    survey-specific PSUs, as in the 05 Cox model (recommended)
#
# Revised model (R2): adds country-survey (country x survey_year, the Cox strata) as a
# categorical fixed effect and uses var_weights instead of freq_weights (freq_weights
# treat v005/1e6 as case counts), with SEs clustered on survey-specific PSUs. Country-
# surveys with no never-breastfed child are dropped from this fit: their fixed effect is
# infinite and they carry no information on the covariates.
#
# Usage (from D:\early-weaning-analysis):  python analysis\17_ever_breastfed_logistic.py
# Outputs: v25_extras\r37_ever_bf_logistic_v25.xlsx and .txt

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from cox_common import COVARS, LABELS, cox_sample, load

OUT = Path(r".\v25_extras")
OUT.mkdir(exist_ok=True)
lines = []


def log(s=""):
    print(s, flush=True)
    lines.append(str(s))


d = cox_sample(load())
d["ever_bf"] = 1 - d["never_breastfed"].astype(int)
d["psu_survey"] = pd.factorize(d["source_zip"] + "_" + d["v021"].astype(int).astype(str))[0]
log(f"n = {len(d):,} children ({d['country'].nunique()} countries, "
    f"{d.groupby(['country', 'survey_year']).ngroups} country-surveys); "
    f"never breastfed {int((d.ever_bf == 0).sum()):,} "
    f"({100 * d.loc[d.ever_bf == 0, 'weight'].sum() / d['weight'].sum():.2f}% weighted)")

X = sm.add_constant(d[COVARS].astype(float))
model = sm.GLM(d["ever_bf"], X, family=sm.families.Binomial(), freq_weights=d["weight"])
tabs = []
for name, groups in [("PSU = v021 (as in v24)", d["v021"].astype(int)),
                     ("PSU = survey + v021", d["psu_survey"])]:
    r = model.fit(cov_type="cluster", cov_kwds={"groups": np.asarray(groups)})
    ci = r.conf_int()
    t = pd.DataFrame({"clustering": name, "covariate": r.params.index,
                      "OR": np.exp(r.params.values), "lower95": np.exp(ci[0].values),
                      "upper95": np.exp(ci[1].values), "p": r.pvalues.values,
                      "n": len(d), "clusters": pd.Series(groups).nunique()})
    t["label"] = t["covariate"].map(LABELS).fillna(t["covariate"])
    tabs.append(t)
    log(f"\nOdds of EVER breastfeeding, cluster-robust SEs, {name} "
        f"({pd.Series(groups).nunique():,} clusters):")
    log(t.set_index("label")[["OR", "lower95", "upper95", "p"]].round(3).to_string())

# Revised: country-survey fixed effects + var_weights, survey-specific PSU clusters
d["stratum"] = d["country"] + "_" + d["survey_year"].astype(int).astype(str)
has_nbf = d.groupby("stratum")["ever_bf"].transform("min") == 0
fe = d.loc[has_nbf, COVARS + ["ever_bf", "weight", "stratum", "psu_survey"]]
log(f"\nRevised model: {d['stratum'].nunique() - fe['stratum'].nunique()} country-surveys "
    f"with no never-breastfed child dropped ({int((~has_nbf).sum()):,} children); "
    f"n = {len(fe):,}, {fe['stratum'].nunique()} country-surveys")
del d, model
gc.collect()
# Design as one float64 array (n x 92); Newton-Raphson needs one weighted copy per step,
# where the default IRLS/pinv fit runs out of memory on 16 GB
codes, strata = pd.factorize(fe["stratum"], sort=True)
Xfe = np.zeros((len(fe), 1 + len(COVARS) + len(strata) - 1))
Xfe[:, 0] = 1.0
Xfe[:, 1:1 + len(COVARS)] = fe[COVARS].to_numpy(dtype=float)
rows = np.flatnonzero(codes > 0)
Xfe[rows, len(COVARS) + codes[rows]] = 1.0
groups = fe["psu_survey"].to_numpy()
r = sm.GLM(fe["ever_bf"].to_numpy(dtype=float), Xfe, family=sm.families.Binomial(),
           var_weights=fe["weight"].to_numpy()).fit(
    method="newton", maxiter=50, cov_type="cluster", cov_kwds={"groups": groups})
b, ci = r.params[1:1 + len(COVARS)], r.conf_int()[1:1 + len(COVARS)]
name = "Country-survey FE + var_weights, PSU = survey + v021"
t = pd.DataFrame({"clustering": name, "covariate": COVARS,
                  "OR": np.exp(b), "lower95": np.exp(ci[:, 0]), "upper95": np.exp(ci[:, 1]),
                  "p": r.pvalues[1:1 + len(COVARS)], "n": len(fe),
                  "clusters": pd.Series(groups).nunique()})
t["label"] = t["covariate"].map(LABELS)
tabs.append(t)
log(f"\nOdds of EVER breastfeeding, {name} ({pd.Series(groups).nunique():,} clusters, "
    f"converged={r.mle_retvals['converged']}, {r.mle_retvals.get('iterations')} iterations):")
log(t.set_index("label")[["OR", "lower95", "upper95", "p"]].round(3).to_string())

fmt = lambda x: f"{x.OR:.2f} ({x.lower95:.2f}-{x.upper95:.2f})"
side = pd.DataFrame({
    "Current (no FE, freq_weights)": tabs[1].set_index("covariate").apply(fmt, axis=1),
    "Revised (country-survey FE, var_weights)": t.set_index("covariate").apply(fmt, axis=1),
}).loc[COVARS].rename(index=LABELS)
log("\nOR (95% CI), both clustered on survey-specific PSU:")
log(side.to_string())

res = pd.concat(tabs, ignore_index=True)
v24 = {"urban": 0.837, "wealth_q1": 0.873, "wealth_q2": 0.946, "wealth_q4": 1.022,
       "wealth_q5": 1.080, "educ_none": 0.735, "educ_secondary_plus": 0.539}
res["OR_v24"] = res["covariate"].map(v24)
with pd.ExcelWriter(OUT / "r37_ever_bf_logistic_v25.xlsx") as xw:
    res.to_excel(xw, sheet_name="odds_ratios", index=False)
    side.to_excel(xw, sheet_name="current_vs_revised")
(OUT / "r37_ever_bf_logistic_v25.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"\nDone. Written to {OUT.resolve()}")
