# 00_check_members.py
# Quickly list .dta members in DHS zips and show detected recodes.
# Usage:
#    python .\00_check_members.py ".\ZIPS"
#   python 00_check_members.py ".\ZIPS" --peek 20

import re, zipfile, argparse
from pathlib import Path
from collections import Counter, defaultdict

def guess_recode(name: str) -> str:
    base = Path(name).name.upper()
    m = re.search(r'_(KR|IR|PR|HR|BR|MR|CR)(?:\D|$)', base)
    if not m:
        # look for KR/IR/... followed by a digit or non-letter or end (works for AFKR71FL.DTA)
        m = re.search(r'(KR|IR|PR|HR|BR|MR|CR)(?=\d|[^A-Z]|$)', base)
    if not m:
        m = re.search(r'(KR|IR|PR|HR|BR|MR|CR)', base)
    return m.group(1) if m else "UNK"

def main(zip_dir: str, peek: int | None):
    zips = sorted(Path(zip_dir).glob("*.zip"))
    if peek:
        zips = zips[:peek]
    if not zips:
        print("No zip files found."); return

    by_recode = Counter()
    examples = defaultdict(list)

    for zp in zips:
        try:
            with zipfile.ZipFile(zp) as zf:
                for mem in zf.namelist():
                    if mem.lower().endswith(".dta"):
                        rc = guess_recode(mem)
                        by_recode[rc] += 1
                        if len(examples[rc]) < 3:
                            examples[rc].append(f"{zp.name} -> {Path(mem).name}")
        except Exception as e:
            print(f"[BAD ZIP] {zp.name}: {e}")

    print("Detected .dta counts by recode:")
    for rc, n in sorted(by_recode.items()):
        print(f"  {rc:>3}: {n}")
    print("\nExamples:")
    for rc in sorted(examples):
        for ex in examples[rc]:
            print(f"  {rc}: {ex}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_dir")
    ap.add_argument("--peek", type=int)
    args = ap.parse_args()
    main(args.zip_dir, args.peek)
