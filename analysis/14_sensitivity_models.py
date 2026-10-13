# 14_sensitivity_models.py
# Sensitivity analyses for the primary Cox model on v25 (model-based SEs).
#  main    the primary model (05 sample), for comparison
#  a)      never breastfed excluded: drops ONLY m4 = 94 (never_breastfed flag), not other
#          duration-0 children
#  b)      no-wealth model: urban + education, all retrospective-duration surveys with a PSU
#          (wealth index not required)
#  c)      recent surveys: survey year >= 2010 and >= 2015 (4-digit Gregorian years from 02 v2.5)
#
# Usage (from D:\early-weaning-analysis):  python analysis\14_sensitivity_models.py
# Outputs: v25_extras\r34_sensitivity_v25.xlsx and .txt

import time
from pathlib import Path

import pandas as pd

from cox_common import COVARS, COVARS_NOWEALTH, LABELS, cox_sample, describe, fit, hr_table, load

OUT = Path(r".\v25_extras")
OUT.mkdir(exist_ok=True)
lines = []


def log(s=""):
    print(s, flush=True)
    lines.append(str(s))


df = load()
main = cox_sample(df)
models = {
    "main": (main, COVARS),
    "a) never breastfed (m4 = 94) excluded": (main[main["never_breastfed"] != 1], COVARS),
    "b) no wealth (urban + education)": (cox_sample(df, wealth=False), COVARS_NOWEALTH),
    "c) surveys from 2010": (main[main["survey_year"] >= 2010], COVARS),
    "c) surveys from 2015": (main[main["survey_year"] >= 2015], COVARS),
}

samples, hrs = [], []
for name, (d, cov) in models.items():
    t0 = time.time()
    info = describe(d)
    info["excluded_vs_main"] = len(main) - len(d) if d is not main and len(d) <= len(main) else None
    cph = fit(d, cov)
    samples.append({"model": name, **info, "C_index": cph.concordance_index_,
                    "years": f"{int(d['survey_year'].min())}-{int(d['survey_year'].max())}"})
    hrs.append(hr_table(cph, name))
    log(f"{name}: n = {info['n']:,}, countries {info['countries']}, strata {info['strata']} "
        f"({time.time() - t0:.0f}s)")

samples = pd.DataFrame(samples)
hrs = pd.concat(hrs, ignore_index=True)
main_hr = hrs[hrs["model"] == "main"].set_index("covariate")["HR"]
hrs["HR_main"] = hrs["covariate"].map(main_hr)
hrs["pct_change_vs_main"] = 100 * (hrs["HR"] / hrs["HR_main"] - 1)

wide = hrs.pivot(index="covariate", columns="model", values="HR").reindex(
    [c for c in COVARS])[list(models)]
wide.insert(0, "label", [LABELS[c] for c in wide.index])
log("\nSamples:\n" + samples.to_string(index=False))
log("\nHazard ratios (model-based):\n" + wide.round(3).to_string())
log(f"\nNever-breastfed children excluded in a): "
    f"{int(samples.loc[samples.model.str.startswith('a)'), 'excluded_vs_main'].iloc[0]):,}")

with pd.ExcelWriter(OUT / "r34_sensitivity_v25.xlsx") as xw:
    samples.to_excel(xw, sheet_name="samples", index=False)
    wide.to_excel(xw, sheet_name="HR_comparison")
    hrs.to_excel(xw, sheet_name="HR_long", index=False)
(OUT / "r34_sensitivity_v25.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"\nDone. Written to {OUT.resolve()}")
