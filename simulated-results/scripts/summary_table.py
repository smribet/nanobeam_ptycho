"""Every condition x angle x dose, as a table.  Writes markdown and CSV to figures/.

    python summary_table.py

Columns:
  groups       number of (dataset, region) groups the rate is computed over: draws x
               regions.  Noiseless data have no Poisson realisation, so there is one
               dataset and the denominator is just the 7 regions -- the rate is then
               quantised in units of 1/7.  The within-dataset test is 10 initial guesses
               in every row, noiseless included; only the number of datasets differs
  non-unique   fraction of (dataset, region) groups where different initial guesses
               disagree -- the decision-relevant number, and the one you cannot check
               experimentally without repeating the reconstruction
  reversed     fraction of regions returned with inverted contrast
  med q        median offset as a fraction of the Wigner-Seitz cell area enclosed;
               0 is perfect registration, 0.5 is indistinguishable from random
  |corr|       mean absolute correlation to ground truth -- how good the lattice is,
               regardless of where it sits or which sign it has
  degraded     fraction of regions with |corr| < 0.5, i.e. no lattice recovered at all

All rates are gated at |corr| >= 0.5: a region whose lattice was never recovered has a
flat correlation surface and will report an arbitrary sign and shift, which would
otherwise be counted as a spectacular artifact.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import paths
import analysis
import figpanels

REF = paths.REFERENCE
DOSES = ["inf", "1e6", "1e5", "1e4"]
DLAB = {"inf": "noiseless", "1e6": "1e6", "1e5": "1e5", "1e4": "1e4"}

CONDITIONS = [
    ("matched, on carbon", {d: f"ss_carbon_complex_{d}" for d in DOSES}),
    ("matched, free-standing", {d: f"ss_bare_complex_{d}" for d in DOSES}),
    ("multislice, complex", {d: f"ms_carbon_a3_complex_{d}" for d in DOSES}),
    ("multislice, pure-phase", {d: f"ms_carbon_a3_phase_{d}" for d in DOSES}),
    ("multislice + phonons", {d: f"msfp_carbon_a3_complex_{d}" for d in DOSES}),
]
MULTI_ANGLE = {"matched, on carbon", "matched, free-standing"}


def metrics(d, angle, QF, min_corr=0.5):
    f = paths.sweep(d) / "summary.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f).dropna(subset=["ws_x"])
    df = df[df.angle_mrad == angle]
    if not len(df):
        return None
    df = df.assign(ns=df.seed // 100,
                   q=[QF[p](np.hypot(x, y))
                      for p, x, y in zip(df.particle, df.ws_x, df.ws_y)])
    good = df.signed_corr.abs() >= min_corr
    g = df[good]
    # shared with the figures, so the table and the panel labels cannot drift apart; it
    # counts a polarity flip or an offset spread above a quarter of the cell, rather than
    # any bitwise difference in the offsets
    nonuniq, ngroups = figpanels.nonunanimity(d, angle=angle, min_corr=min_corr)
    return dict(n_runs=df.seed.nunique(), n_groups=ngroups,
                nonuniq=nonuniq,
                reversed=g["reversed"].mean() if len(g) else np.nan,
                medq=g.q.median() if len(g) else np.nan,
                acorr=df.signed_corr.abs().mean(),
                degraded=(~good).mean())


def main():
    ref = np.load(paths.sweep(REF) / "runs" / "_reference.npz", allow_pickle=True)
    G = {k: (None if v is None else analysis.reduce_basis(np.array(v)))
         for k, v in json.loads(str(ref["region_G"])).items()}
    QF = {k: analysis.cell_quantile_fn(v) for k, v in G.items() if v is not None}

    rows = []
    for name, dirs in CONDITIONS:
        angles = (12.0, 6.0, 3.0) if name in MULTI_ANGLE else (3.0,)
        for a in angles:
            for d in DOSES:
                m = metrics(dirs[d], a, QF)
                rows.append(dict(condition=name, angle_mrad=a, dose=DLAB[d],
                                 **(m or dict(n_runs=0, n_groups=0, nonuniq=np.nan,
                                              reversed=np.nan, medq=np.nan,
                                              acorr=np.nan, degraded=np.nan))))
    T = pd.DataFrame(rows)
    out = paths.figures()
    T.to_csv(out / "summary_table.csv", index=False)

    def fmt(v, p=2):
        return "--" if not np.isfinite(v) else f"{v:.{p}f}"

    lines = ["| condition | angle | dose | runs | groups | non-unique | reversed | med q | \\|corr\\| | degraded |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in T.iterrows():
        lines.append(f"| {r.condition} | {r.angle_mrad:g} mrad | {r.dose} | "
                     f"{int(r.n_runs)} | {int(r.n_groups)} | {fmt(r.nonuniq)} | {fmt(r['reversed'])} | "
                     f"{fmt(r.medq)} | {fmt(r.acorr, 3)} | {fmt(r.degraded)} |")
    md = "\n".join(lines)
    (out / "summary_table.md").write_text(md + "\n")
    print(md)
    print(f"\nwrote {out/'summary_table.md'} and .csv")


if __name__ == "__main__":
    main()
