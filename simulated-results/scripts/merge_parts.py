"""Merge the per-job parts written by revision_jobs.py into ordinary sweep directories.

    python merge_parts.py                  # every sweep under outputs/rev/
    python merge_parts.py ss_carbon28_complex_1e4 ...

Each sweep's parts/<angle>_<draw>/ holds the CSVs and runs/ of one job.  The merged sweep
directory gets the concatenated summary, drift and boundaries CSVs, all runs/*.npz (their
names already carry angle and seed, so they never collide), and one _reference.npz -- the
same layout as the published sweeps, so figpanels and summary_table read it unchanged.
Parts without a DONE marker are skipped and reported, so a partial merge is never silent.
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

import paths

REV = paths.OUTPUTS / "rev"
CSVS = ("summary.csv", "drift.csv", "boundaries.csv", "search.csv")


def merge(sweep_dir):
    parts = sorted((sweep_dir / "parts").glob("*/"))
    done = [p for p in parts if (p / "DONE").exists()]
    missing = [p.name for p in parts if p not in done]
    if not done:
        return 0, missing
    for name in CSVS:
        frames = [pd.read_csv(p / name) for p in done if (p / name).exists()
                  and (p / name).stat().st_size > 0]
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(sweep_dir / name, index=False)
    runs = sweep_dir / "runs"
    runs.mkdir(exist_ok=True)
    for p in done:
        for f in (p / "runs").glob("*.npz"):
            if f.name == "_reference.npz":
                if not (runs / f.name).exists():
                    shutil.copy2(f, runs / f.name)
            else:
                shutil.copy2(f, runs / f.name)
    return len(done), missing


def main():
    names = sys.argv[1:]
    dirs = ([REV / n for n in names] if names
            else sorted(d for d in REV.glob("*/") if (d / "parts").exists()))
    for d in dirs:
        n, missing = merge(d)
        flag = f"  INCOMPLETE: {', '.join(missing)}" if missing else ""
        print(f"{d.name:40s} {n:3d} parts merged{flag}")


if __name__ == "__main__":
    main()
