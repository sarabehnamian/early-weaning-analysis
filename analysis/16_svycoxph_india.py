# 16_svycoxph_india.py
# Main Cox model for India NFHS-5 (IAKR7EDT.zip, the most recent Indian survey), fitted on the
# same children (the Cox sample of 05 restricted to that survey) with:
#   lifelines, model-based SEs
#   R survival::coxph + cluster(v021): the PSU-clustered sandwich that lifelines computes with
#     robust=True (as in 05). lifelines' own version is roughly quadratic in stratum size and
#     takes hours on one 179k-child survey, so it is run here only on a 5,000-child validation
#     subsample, where it is compared with the R version.
#   R survey::svycoxph, design ids = v021, strata = v022 / v023, weights = v005/1e6
#
# Usage (from D:\early-weaning-analysis):  python analysis\16_svycoxph_india.py
# Outputs: v25_extras\r36_svycoxph_india_v25.xlsx (+ input CSVs and the R output CSV)

import math
import subprocess
from pathlib import Path

import pandas as pd
from lifelines import CoxPHFitter

from cox_common import COVARS, LABELS, cox_sample, load

OUT = Path(r".\v25_extras")
OUT.mkdir(exist_ok=True)
RSCRIPT = r"C:\Program Files\R\R-4.5.1\bin\Rscript.exe"
ZIP = "IAKR7EDT.zip"

d = cox_sample(load())
d = d[d["source_zip"] == ZIP].copy()
cols = ["duration_months", "event", "weight", "v021", "v022", "v023"] + COVARS
csv = OUT / "r36_india_IAKR7E_cox_sample.csv"
val_csv = OUT / "r36_india_IAKR7E_validation_5k.csv"
d[cols].to_csv(csv, index=False)
v = d.sample(n=5000, random_state=1)
v[cols].to_csv(val_csv, index=False)
print(f"{ZIP}: {len(d):,} children, {int(d.event.sum()):,} events, {d.v021.nunique():,} PSUs, "
      f"{d.v022.nunique()} v022 strata, {d.v023.nunique()} v023 strata", flush=True)


def lifelines_tab(data, label, **kw):
    use = ["duration_months", "event", "weight"] + COVARS + (["v021"] if "cluster_col" in kw else [])
    c = CoxPHFitter().fit(data[use], duration_col="duration_months", event_col="event",
                          weights_col="weight", **kw)
    s = c.summary
    return pd.DataFrame({"model": label, "covariate": s.index, "HR": s["exp(coef)"].values,
                         "lower95": s["exp(coef) lower 95%"].values,
                         "upper95": s["exp(coef) upper 95%"].values,
                         "se": s["se(coef)"].values, "p": s["p"].values, "n": len(data)})


rows = [lifelines_tab(d, "lifelines, model-based SEs", robust=False),
        lifelines_tab(v, "VALIDATION 5k: lifelines, model-based SEs", robust=False),
        lifelines_tab(v, "VALIDATION 5k: lifelines, PSU-clustered sandwich (v021)",
                      robust=True, cluster_col="v021")]
print("lifelines fits done", flush=True)

r_out = OUT / "r36_svycoxph_india_R_output.csv"
r = subprocess.run([RSCRIPT, str(Path(__file__).with_name("16_svycoxph_india.R")), str(csv), str(val_csv), str(r_out)],
                   capture_output=True, text=True)
print(r.stdout[-2000:], r.stderr[-2000:], flush=True)
rows.append(pd.read_csv(r_out))

long = pd.concat(rows, ignore_index=True)
long["label"] = long["covariate"].map(LABELS)
long["CI_width_log"] = [math.log(u / l) for u, l in zip(long["upper95"], long["lower95"])]
ref = long[long.model == "R coxph, PSU-clustered sandwich (v021)"].set_index("covariate")
long["HR_ratio_vs_clustered"] = long["HR"] / long["covariate"].map(ref["HR"])
long["CI_width_ratio_vs_clustered"] = long["CI_width_log"] / long["covariate"].map(ref["CI_width_log"])
long["HR_CI"] = [f"{h:.2f} ({l:.2f}-{u:.2f})" for h, l, u in zip(long.HR, long.lower95, long.upper95)]

order = [m for m in long["model"].unique()]
wide = long.pivot(index="covariate", columns="model", values="HR_CI").reindex(COVARS)[order]
wide.insert(0, "label", [LABELS[c] for c in wide.index])
se = long.pivot(index="covariate", columns="model", values="se").reindex(COVARS)[order]
pd.set_option("display.width", 300)
print(wide.to_string())
with pd.ExcelWriter(OUT / "r36_svycoxph_india_v25.xlsx") as xw:
    wide.to_excel(xw, sheet_name="comparison")
    se.to_excel(xw, sheet_name="standard_errors")
    long.to_excel(xw, sheet_name="long", index=False)
print(f"\nDone. Written to {OUT.resolve()}")
