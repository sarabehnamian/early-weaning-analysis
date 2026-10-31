# 21_svycoxph_full.py
# Reviewer 2, comment 5: is the DHS sampling design handled adequately? Refits the main Cox
# model on the WHOLE Cox sample (1,001,512 children, 85 country-surveys) in R, with
#   a) survival::coxph + cluster(PSU)      the estimator of the main model
#   b) survey::svycoxph, ids = PSU, strata = DHS sampling stratum, nest = TRUE
# Both keep strata(country-survey) for the baseline hazards, as in the main model.
# The DHS sampling stratum is v022 where present, else v023, else the survey as one stratum;
# strata and PSUs are made survey-specific.
#
# Usage (from D:\early-weaning-analysis):  python analysis\21_svycoxph_full.py
# Needs R with the survival and survey packages (path in RSCRIPT, as in 16).
# Outputs: v25_extras\r41_svycoxph_full.xlsx (+ the input CSV and R output CSV)
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from cox_common import COVARS, LABELS, cox_sample, load

OUT = Path("v25_extras")
OUT.mkdir(exist_ok=True)
RSCRIPT = r"C:\Program Files\R\R-4.5.1\bin\Rscript.exe"

d = cox_sample(load())
d["cox_stratum"] = d["country"].astype(str) + "_" + d["survey_year"].astype(int).astype(str)
sz = d["source_zip"].astype(str)
d["psu"] = pd.factorize(sz + "_" + d["v021"].astype(int).astype(str))[0] + 1
st = d["v022"].where(d["v022"].notna(), d["v023"])
d["design_stratum"] = pd.factorize(sz + "_" + st.fillna(-1).astype(int).astype(str))[0] + 1
n_nostrat = int(st.isna().sum())
cols = ["duration_months", "event", "weight", "cox_stratum", "psu", "design_stratum"] + COVARS
csv = OUT / "r41_cox_sample_for_R.csv"
d[cols].to_csv(csv, index=False)
print(f"{len(d):,} children, {d.cox_stratum.nunique()} country-surveys, {d.psu.nunique():,} PSUs, "
      f"{d.design_stratum.nunique():,} DHS sampling strata ({n_nostrat:,} children without v022/v023, "
      f"their survey used as one stratum)", flush=True)

r_out = OUT / "r41_svycoxph_full_R_output.csv"
r = subprocess.run([RSCRIPT, str(Path(__file__).with_name("21_svycoxph_full.R")), str(csv), str(r_out)],
                   capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
res = pd.read_csv(r_out)
res["label"] = res["covariate"].map(LABELS)
res["HR_CI"] = [f"{h:.2f} ({l:.2f}-{u:.2f})" for h, l, u in zip(res.HR, res.lower95, res.upper95)]
wide = res.pivot(index="label", columns="model", values="HR_CI")
se = res.pivot(index="label", columns="model", values="se")
ratio = (se["svycoxph, PSU + DHS sampling strata"] /
         se["coxph, PSU-clustered sandwich (as main model)"]).rename("SE ratio svycoxph / clustered")
pd.set_option("display.width", 250)
print(wide.join(ratio.round(3)).to_string())
with pd.ExcelWriter(OUT / "r41_svycoxph_full.xlsx") as xw:
    wide.join(ratio).to_excel(xw, sheet_name="comparison")
    res.to_excel(xw, sheet_name="long", index=False)
print(f"\nDone. Written to {(OUT / 'r41_svycoxph_full.xlsx').resolve()}")
