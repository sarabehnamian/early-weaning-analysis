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
# Usage (from D:\early-weaning-analysis):  python analysis\17_ever_breastfed_logistic.py
# Outputs: v25_extras\r37_ever_bf_logistic_v25.xlsx and .txt

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

res = pd.concat(tabs, ignore_index=True)
v24 = {"urban": 0.837, "wealth_q1": 0.873, "wealth_q2": 0.946, "wealth_q4": 1.022,
       "wealth_q5": 1.080, "educ_none": 0.735, "educ_secondary_plus": 0.539}
res["OR_v24"] = res["covariate"].map(v24)
with pd.ExcelWriter(OUT / "r37_ever_bf_logistic_v25.xlsx") as xw:
    res.to_excel(xw, sheet_name="odds_ratios", index=False)
(OUT / "r37_ever_bf_logistic_v25.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"\nDone. Written to {OUT.resolve()}")
