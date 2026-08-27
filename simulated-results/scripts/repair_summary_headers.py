"""Normalise summary.csv files whose header predates a later column.

`run_sweep.py --append` reuses whatever header a file already has, so appending with a
newer FIELDS list produces rows wider than the header and pandas refuses to read the file.
Every column added so far has gone on the END of FIELDS, so the repair is to widen the
header and pad the older rows -- no values move, and nothing is lost.

Idempotent: a file already at the current width is left untouched.  Refuses to guess if
the old header is not a prefix of FIELDS, since that would mean a column was inserted in
the middle and padding at the end would silently misalign every row.

    python repair_summary_headers.py            # report only
    python repair_summary_headers.py --apply
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import paths
from run_sweep import FIELDS, migrate_csv

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write; otherwise report only")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="directory names to leave alone, e.g. ones being written now")
    args = ap.parse_args()

    n_fixed = n_ok = 0
    for path in sorted(paths.OUTPUTS.glob("*/summary.csv")):
        if any(s in str(path.parent.name) for s in args.skip):
            print(f"  skip     {path.parent.name}  (in flight)")
            continue
        try:
            moved = migrate_csv(path, FIELDS, apply=args.apply)
        except ValueError as exc:
            print(f"  REFUSED  {path.parent.name}: {exc}")
            continue
        if moved is None:
            n_ok += 1
            continue
        hdr, n_pad = moved
        print(f"  repair   {path.parent.name}: header {len(hdr)} -> {len(FIELDS)}, "
              f"padded {n_pad} rows ({', '.join(FIELDS[len(hdr):])})")
        n_fixed += args.apply

    print(f"\n{n_ok} already current, {n_fixed} repaired"
          + ("" if args.apply else "  (dry run -- pass --apply to write)"))


if __name__ == "__main__":
    main()
