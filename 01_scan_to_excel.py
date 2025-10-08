# 01_scan_to_excel.py
#python 01_scan_to_excel.py ".\ZIPS" --out "Bfeed_scan.xlsx"

# Scans DHS zip files and writes an Excel workbook with:
#  - Inventory of all .dta files and guessed recode
#  - KR readiness (breastfeeding survival: counts + reasons)
#  - ColumnsPreview for useful variables by recode
#  - BadZips (very small zips likely to be login pages or errors)
#  - README sheet with recode definitions and notes
#
# Why KR for breastfeeding survival?
#   The event/time variables are in KR at the child level:
#   - m4: breastfeeding status/duration (event/censor)
#   - age: b19 or (v008 - b3)
#   - design: v005 (weights), v021 (PSU), v022/v023 (strata), v024, v025
#
# “Test first” options:
#   --peek N             process only the first N matching zips (fast sanity check)
#   --include PATTERN    restrict to certain zips, e.g. --include "AF*DT.zip" --include "BD*DT.zip"
#
# Example quick check (first 10 zips):
#   python 01_scan_to_excel.py ".\ZIPS" --out ".\Bfeed_scan_peek.xlsx" --peek 10
#
# Full run:
#   python 01_scan_to_excel.py ".\ZIPS" --out ".\Bfeed_scan.xlsx"
#
# Requirements: pandas; optional: pyreadstat, xlsxwriter (for formatting).
# If xlsxwriter isn’t installed, formatting is skipped gracefully.

import re, zipfile, tempfile, argparse, fnmatch
from pathlib import Path
import pandas as pd

# ------------ Config ------------
REQ_ANY_STRATA = ["v022", "v023"]  # accept either one for strata
INVALID_M4 = {96, 97, 98, 99}

WANTED = {
    "KR": ["m4","b19","v008","b3","v005","v021","v022","v023","v024","v025","b5"],
    "IR": ["v012","v025","v021","v022","v023","v024","v005"],
    "PR": ["hv103","hv025","hv021","hv023","hv024","hv005"],
    "HR": ["hv024","hv025","hv005"],
    "BR": ["b3","b5","v021","v022","v023","v024","v025","v005"],
    "MR": ["mv012","mv025","mv021","mv022","mv023","mv024","mv005"],
    "CR": ["v025","v021","v022","v023","v024","v005","cm4"],
    "UNK": []
}

# ------------ Helpers ------------

def guess_recode(name: str, zip_name: str = "") -> str:
    """Guess DHS recode type from member filename and/or the zip name."""
    txt = f"{name} {zip_name}".upper()
    m = re.search(r'(KR|IR|PR|HR|BR|MR|CR)(?=\d{2})', txt)
    if m: return m.group(1)
    m = re.search(r'_(KR|IR|PR|HR|BR|MR|CR)(?:\D|$)', txt)
    if m: return m.group(1)
    m = re.search(r'(KR|IR|PR|HR|BR|MR|CR)', txt)
    return m.group(1) if m else "UNK"

def find_dta_members(zf: zipfile.ZipFile):
    return [m for m in zf.namelist() if m.lower().endswith(".dta")]

def _read_stata_any(path: Path, columns=None, nrows=None) -> pd.DataFrame:
    """
    Robust Stata reader that works across pandas versions:
      1) Try pandas.read_stata with columns/nrows (newer pandas)
      2) Else try pyreadstat.read_dta(usecols=..., row_limit=...)
      3) Else fallback to full pandas read then subset/head in memory.
    """
    # 1) Newer pandas fast path
    try:
        return pd.read_stata(path, convert_categoricals=False, columns=columns, nrows=nrows)
    except TypeError:
        pass

    # 2) pyreadstat fast path (if available)
    try:
        import pyreadstat
        df, _meta = pyreadstat.read_dta(
            str(path),
            usecols=columns if columns is not None else None,
            row_limit=nrows if nrows is not None else None
        )
        return df
    except Exception:
        pass

    # 3) Old pandas fallback: full read, then subset/head in memory
    df = pd.read_stata(path, convert_categoricals=False)
    if columns is not None:
        keep = [c for c in columns if c in df.columns]
        df = df[keep]
    if nrows is not None:
        df = df.head(nrows)
    return df

def read_dta_from_zip(zip_path: Path, member: str, columns=None, nrows=None) -> pd.DataFrame:
    """Extract a member to a temp dir and read with the robust reader above."""
    with zipfile.ZipFile(zip_path) as zf, tempfile.TemporaryDirectory() as td:
        out = Path(td) / Path(member).name
        zf.extract(member, path=td)
        return _read_stata_any(out, columns=columns, nrows=nrows)

def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def check_kr_readiness(df: pd.DataFrame) -> dict:
    present = set(df.columns)
    missing = []
    needed = ["m4","v005","v021","v024","v025","b5"]
    for k in needed:
        if k not in present:
            missing.append(k)
    if not any(x in present for x in REQ_ANY_STRATA):
        missing.append("v022|v023")

    notes = []
    ok = True if not missing else False

    # Age
    if "b19" in present:
        age = to_num(df["b19"])
    elif all(k in present for k in ["v008","b3"]):
        age = to_num(df["v008"]) - to_num(df["b3"])
    else:
        age = pd.Series([pd.NA]*len(df))
        ok = False
        notes.append("no age variable (b19 or v008-b3)")

    b5 = to_num(df["b5"]) if "b5" in present else pd.Series([pd.NA]*len(df))
    elig = (b5 == 1) & age.notna() & (age >= 0) & (age < 60)

    m4 = to_num(df["m4"]) if "m4" in present else pd.Series([pd.NA]*len(df))
    valid_mask = elig & m4.notna()

    n_rows = int(len(df))
    n_elig = int(elig.sum()) if hasattr(elig, "sum") else 0
    n_event = int(((m4 < 95) & valid_mask).sum())
    n_cens  = int(((m4 == 95) & valid_mask).sum())
    n_invalid = int(((m4.isin(INVALID_M4)) & elig).sum())
    n_missing_m4 = int(((m4.isna()) & elig).sum())

    if (n_event + n_cens) == 0:
        ok = False
        notes.append("no valid m4 among eligible")

    status = "ready" if ok and (n_event + n_cens) > 0 else "limited"
    reason = "; ".join(missing + notes)

    return dict(
        n_rows=n_rows,
        n_eligible_u60_alive=n_elig,
        n_event=n_event,
        n_censored=n_cens,
        n_invalid_m4=n_invalid,
        n_missing_m4=n_missing_m4,
        status=status,
        limiter_reason=reason
    )

def pick_preview_cols(recode: str, cols: list) -> list:
    return [c for c in WANTED.get(recode, []) if c in cols]

# ------------ Excel helpers ------------

def safe_format_sheet(writer: pd.ExcelWriter, sheet: str, df: pd.DataFrame, status_col: str | None = None):
    """Format if using xlsxwriter; otherwise no-op (openpyxl fallback)."""
    try:
        if writer.engine != "xlsxwriter":
            return
        wb = writer.book
        ws = writer.sheets[sheet]
        if df is not None and not df.empty:
            ws.freeze_panes(1, 0)
            ws.autofilter(0, 0, df.shape[0], df.shape[1]-1)
        if df is not None and not df.empty:
            for idx, col in enumerate(df.columns):
                series = df[col].astype(str)
                maxlen = max([len(str(col))] + [len(s) for s in series.tolist()])
                ws.set_column(idx, idx, min(maxlen + 2, 60))
        if status_col and (df is not None) and (status_col in df.columns) and not df.empty:
            c = list(df.columns).index(status_col)
            fmt_ready = wb.add_format({"font_color": "green", "bold": True})
            fmt_lim = wb.add_format({"font_color": "red", "bold": True})
            last_row = max(df.shape[0], 1)
            ws.conditional_format(1, c, last_row, c,
                {"type":"text","criteria":"containing","value":"ready","format":fmt_ready})
            ws.conditional_format(1, c, last_row, c,
                {"type":"text","criteria":"containing","value":"limited","format":fmt_lim})
    except Exception:
        pass

# ------------ Main scan ------------

def main(zip_dir: str, out_xlsx: str, peek: int | None, include_patterns: list[str]):
    zip_dir = Path(zip_dir)
    all_zips = sorted(zip_dir.glob("*.zip"), key=lambda p: p.name.lower())
    if include_patterns:
        zips = [zp for zp in all_zips if any(fnmatch.fnmatch(zp.name, pat) for pat in include_patterns)]
    else:
        zips = all_zips

    if peek is not None and peek > 0:
        zips = zips[:peek]

    if not zips:
        print(f"No matching .zip files in {zip_dir}")
        return

    inv_rows, kr_rows, colprev_rows, badzip_rows = [], [], [], []

    for zp in zips:
        size = zp.stat().st_size
        if size < 100_000:
            badzip_rows.append({"zip": zp.name, "size_bytes": size})
        try:
            with zipfile.ZipFile(zp) as zf:
                members = find_dta_members(zf)
        except Exception as e:
            inv_rows.append({"zip": zp.name, "dta": None, "recode":"UNK",
                             "zip_error": str(e), "size_bytes": size})
            continue

        if not members:
            inv_rows.append({"zip": zp.name, "dta": None, "recode":"UNK",
                             "zip_error": "no .dta members", "size_bytes": size})
            continue

        for mem in members:
            recode = guess_recode(mem, zp.name)
            inv_rows.append({"zip": zp.name, "dta": Path(mem).name, "recode": recode, "size_bytes": size})

            # Try to read a small head to list columns quickly.
            cols = []
            head_error = None
            try:
                df_head = read_dta_from_zip(zp, mem, columns=None, nrows=500)
                cols = list(df_head.columns)
            except Exception as e:
                head_error = f"read error (head): {e}"

            present_cols = pick_preview_cols(recode, cols) if cols else []
            colprev_rows.append({
                "zip": zp.name, "dta": Path(mem).name, "recode": recode,
                "present_cols": ", ".join(present_cols),
                "has_design_weight": int(("v005" in cols) or ("hv005" in cols) or ("mv005" in cols)),
                "error": head_error
            })

            # KR readiness (only for KR) — do it EVEN IF head peek failed
            if recode == "KR":
                need_cols = sorted(set(WANTED["KR"] + ["v022","v023"]))
                try:
                    df_kr = read_dta_from_zip(zp, mem, columns=need_cols, nrows=None)  # full read if needed
                    res = check_kr_readiness(df_kr)
                    have = set(df_kr.columns)
                except Exception as e:
                    res = dict(status="limited", limiter_reason=f"read error: {e}",
                               n_rows=pd.NA, n_eligible_u60_alive=pd.NA,
                               n_event=pd.NA, n_censored=pd.NA,
                               n_invalid_m4=pd.NA, n_missing_m4=pd.NA)
                    have = set()

                kr_rows.append({
                    "zip": zp.name, "dta": Path(mem).name,
                    "has_b19": int("b19" in have),
                    "has_v008_b3": int(("v008" in have) and ("b3" in have)),
                    "has_v021": int("v021" in have),
                    "has_v022_or_v023": int(("v022" in have) or ("v023" in have)),
                    "has_v005": int("v005" in have),
                    "has_v024": int("v024" in have),
                    "has_v025": int("v025" in have),
                    **res
                })

    # ------------ Write Excel ------------
    engine = "xlsxwriter" if _xlsxwriter_available() else None
    with pd.ExcelWriter(out_xlsx, engine=engine) as wr:
        # Inventory
        df_inv = pd.DataFrame(inv_rows)
        if not df_inv.empty:
            df_inv.sort_values(["zip","recode","dta"], inplace=True, na_position="last")
        df_inv.to_excel(wr, index=False, sheet_name="Inventory")
        safe_format_sheet(wr, "Inventory", df_inv)

        # KR_Readiness
        df_kr = pd.DataFrame(kr_rows)
        if not df_kr.empty:
            df_kr.sort_values(["status","zip","dta"], inplace=True, na_position="last")
        df_kr.to_excel(wr, index=False, sheet_name="KR_Readiness")
        safe_format_sheet(wr, "KR_Readiness", df_kr, status_col="status")

        # ColumnsPreview
        df_cols = pd.DataFrame(colprev_rows)
        if not df_cols.empty:
            df_cols.sort_values(["recode","zip","dta"], inplace=True, na_position="last")
        df_cols.to_excel(wr, index=False, sheet_name="ColumnsPreview")
        safe_format_sheet(wr, "ColumnsPreview", df_cols)

        # BadZips
        df_bad = pd.DataFrame(badzip_rows)
        if not df_bad.empty:
            df_bad.sort_values(["size_bytes","zip"], inplace=True)
        df_bad.to_excel(wr, index=False, sheet_name="BadZips")
        safe_format_sheet(wr, "BadZips", df_bad)

        # README
        readme = pd.DataFrame([
            ["KR","Children’s recode — child-level (≈last 5y). Breastfeeding m4; age b19 or v008-b3; design v005, v021, v022/23, v024, v025."],
            ["IR","Women’s recode — woman-level v*."],
            ["PR","Household members recode — person-level hv* in household."],
            ["HR","Household recode — household-level hv*."],
            ["BR","Births recode — complete birth histories for interviewed women."],
            ["MR","Men’s recode — man-level mv*."],
            ["CR","Couples recode — matched woman–partner records."]
        ], columns=["Recode","What it is"])
        readme.to_excel(wr, index=False, sheet_name="README")
        try:
            if wr.engine == "xlsxwriter":
                ws = wr.sheets["README"]
                ws.write(readme.shape[0]+2, 0, "Notes:")
                ws.write(readme.shape[0]+3, 0, "- Breastfeeding survival needs KR: m4, age (b19 or v008-b3), design vars (v021, v022/v023, v005).")
                ws.write(readme.shape[0]+4, 0, "- 'KR_Readiness' flags what's missing (status + limiter_reason).")
                ws.write(readme.shape[0]+5, 0, "- Use --peek/--include to test a few zips fast; then run full.")
        except Exception:
            pass

    print(f"Wrote {out_xlsx}")

def _xlsxwriter_available() -> bool:
    try:
        import xlsxwriter  # noqa: F401
        return True
    except Exception:
        return False

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_dir", help="Path to folder with DHS *.zip files (e.g., .\\ZIPS)")
    ap.add_argument("--out", default="Bfeed_scan.xlsx", help="Output Excel path")
    ap.add_argument("--peek", type=int, default=None, help="Process only the first N matching zips (quick test)")
    ap.add_argument("--include", action="append", default=[], help='Glob pattern(s) of zip names to include, e.g. "AF*DT.zip" (can repeat)')
    args = ap.parse_args()
    main(args.zip_dir, args.out, args.peek, args.include)
