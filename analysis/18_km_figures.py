# 18_km_figures.py
# Redraws the survival curves by country, region and residence (Figures 4-6) with the
# 24-month reference line only; the 6-month line is dropped (see 03 kaplan_meier_by_group).
# Uses the plotting function from 03 unchanged, so curves match a full 03 run.
#
# Usage (from D:\early-weaning-analysis):  python analysis\18_km_figures.py
# Outputs: 03_survival_v25\survival_curves\km_by_{country,region,residence}.png, copied to
#          figures_v25\Figure{4,5,6}_curves_by_{country,region,residence}.png

import argparse
import importlib.util
import shutil
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=r".\02_extract_v25\combined_breastfeeding_data.csv")
ap.add_argument("--curves", default=r".\03_survival_v25\survival_curves")
ap.add_argument("--out", default=r".\figures_v25")
a = ap.parse_args()

spec = importlib.util.spec_from_file_location("surv03", "03_survival_analysis.py")
surv03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(surv03)

curves, out = Path(a.curves), Path(a.out)
curves.mkdir(parents=True, exist_ok=True)
out.mkdir(parents=True, exist_ok=True)

# Same preparation as 03 main()
data = pd.read_csv(a.data, low_memory=False)
data = data[data["country"].notna()]
data["weight"] = pd.to_numeric(data["v005"], errors="coerce") / 1e6
data["region"] = data["country"].apply(surv03.assign_region)
data["residence"] = data["urban"].map({0: "Rural", 1: "Urban"})

for var, title, fig in [("country", "Country", "Figure4_curves_by_country.png"),
                        ("region", "Region", "Figure5_curves_by_region.png"),
                        ("residence", "Urban/Rural Residence", "Figure6_curves_by_residence.png")]:
    print(f"Drawing km_by_{var}...", flush=True)
    surv03.kaplan_meier_by_group(data, var, curves, title)
    shutil.copy2(curves / f"km_by_{var}.png", out / fig)
    print(f"  {out / fig}", flush=True)
