# 09_check_country_codes.py
# Lists every KR zip with the DTA inside it and the country code the DHS data itself
# carries (v000), next to the 2-letter prefix the pipeline currently uses as "country".
# Usage (from D:\early-weaning-analysis):
#   python analysis\09_check_country_codes.py ".\data" --scan ".\Bfeed_scan.xlsx" --out ".\r29_country_code_check.xlsx"

import argparse, zipfile, tempfile
from pathlib import Path
import pandas as pd
import pyreadstat

ap = argparse.ArgumentParser()
ap.add_argument("zip_dir")
ap.add_argument("--scan", default="Bfeed_scan.xlsx")
ap.add_argument("--out", default="r29_country_code_check.xlsx")
a = ap.parse_args()

ready = pd.read_excel(a.scan, sheet_name="KR_Readiness")
ready = ready[ready["status"] == "ready"]

rows = []
for _, r in ready.iterrows():
    zp = Path(a.zip_dir) / r["zip"]
    rec = {"zip": r["zip"], "dta": r["dta"], "prefix_used_as_country": r["zip"][:2].upper(),
           "zip_size_bytes": zp.stat().st_size if zp.exists() else None}
    try:
        with zipfile.ZipFile(zp) as zf, tempfile.TemporaryDirectory() as td:
            zf.extract(r["dta"], td)
            fp = str(Path(td) / r["dta"])
            try:
                df, meta = pyreadstat.read_dta(fp, usecols=["v000", "v007", "v024"], row_limit=50)
            except Exception:
                # some old DHS files (e.g. GAKR41, TRKR41) are not UTF-8
                df, meta = pyreadstat.read_dta(fp, usecols=["v000", "v007", "v024"], row_limit=50,
                                               encoding="latin1")
        rec["v000"] = str(df["v000"].iloc[0]).strip()
        rec["v007"] = int(df["v007"].mode().iloc[0])
        lab = meta.variable_value_labels.get("v024", {})
        rec["v024_first_label"] = lab.get(df["v024"].iloc[0], "")
        rec["file_label"] = meta.file_label
    except Exception as e:
        rec["error"] = str(e)
    rows.append(rec)

out = pd.DataFrame(rows)
out["v000_country"] = out["v000"].str[:2]
out["MISMATCH"] = out["v000_country"] != out["prefix_used_as_country"]
out.sort_values(["MISMATCH", "zip"], ascending=[False, True]).to_excel(a.out, index=False)
print(out[out["MISMATCH"]].to_string())
print(f"\n{int(out['MISMATCH'].sum())} of {len(out)} ready files have a prefix that differs from v000")
print("Duplicate DTA names:", out[out.duplicated("dta", keep=False)]["zip"].tolist())
