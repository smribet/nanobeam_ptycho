"""Rebuild summary.csv from stored reconstructions, without re-running ePIE.

    python reanalyse.py

Use after changing anything in analysis.py -- region definitions, metrics, the
Wigner-Seitz reduction.  The reconstructions themselves are untouched.
"""
import csv, json
from pathlib import Path

import numpy as np

import sim
import analysis
from run_sweep import FIELDS, global_corr
import paths

OUT = paths.OUTPUTS / "sweep"
RUNS = OUT / "runs"
SUMMARY = OUT / "summary.csv"
BOUNDS = OUT / "boundaries.csv"


def main():
    ref = np.load(RUNS / "_reference.npz", allow_pickle=True)
    truth = ref["truth"].astype(np.float64)
    sampling = float(ref["sampling"])

    regions = analysis.build_regions(truth, sampling, sim.PARTICLES)
    print(f"{len(regions)} regions: " + ", ".join(r["name"] for r in regions))

    # refresh the reference file so stored masks and bases match the new definitions
    keep = {k: ref[k] for k in ref.files
            if k not in ("region_names", "region_G", "region_masks")}
    np.savez_compressed(
        RUNS / "_reference.npz", **keep,
        region_names=json.dumps([r["name"] for r in regions]),
        region_G=json.dumps({r["name"]: (None if r["G"] is None else r["G"].tolist())
                             for r in regions}),
        region_masks=np.stack([r["mask"] for r in regions]))

    BFIELDS = ["angle_mrad", "seed", "coupling", "boundary", "registry_err_A",
               "d_shared_A", "frac_of_d", "sign_flip", "ambiguous"]
    runs = sorted(RUNS.glob("a*.npz"))
    bfh = open(BOUNDS, "w", newline="")
    bwriter = csv.DictWriter(bfh, fieldnames=BFIELDS)
    bwriter.writeheader()

    with open(SUMMARY, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for f in runs:
            d = np.load(f, allow_pickle=True)
            phase = d["phase"].astype(np.float64)
            angle = float(d["angle_mrad"])
            history = d["error_history"]
            rows = analysis.summarise(phase, truth, sampling, regions)
            gc = global_corr(phase, truth)
            for r in rows:
                writer.writerow({**r,
                                 "angle_mrad": angle,
                                 "defocus_A": float(d["defocus_A"]),
                                 "seed": int(d["seed"]),
                                 "coupling": sim.COUPLING[angle],
                                 "global_corr": gc,
                                 "final_error": float(history[-1]),
                                 "n_iter": int(d["n_iter"]),
                                 "runtime_s": ""})
            for b in analysis.twin_boundaries(rows, regions):
                bwriter.writerow({**b, "angle_mrad": angle, "seed": int(d["seed"]),
                                  "coupling": sim.COUPLING[angle]})
            amb = sum(r["peak_ratio"] > 0.8 for r in rows)
            print(f"  {f.name}: global {gc:.3f}" + (f"  [{amb} ambiguous]" if amb else ""))

    bfh.close()
    print(f"\nrewrote {SUMMARY} and {BOUNDS} from {len(runs)} stored reconstructions")


if __name__ == "__main__":
    main()
