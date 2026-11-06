"""One table for every experimental re-reconstruction: do different seeds agree?

    python e1_summary.py        # writes recons/e1/summary.md, and agreement-map figures
    python e1_summary.py "PtyRAD, cosine"   # only the sets whose label starts with this,
                                            # written to summary_<label>.md

For every engine and variant directory and every scan, each seed is compared with the
lowest seed of the same set by the local agreement of register_lattice.py, over the whole
lattice band and over {111}+{200} and {220} separately, the reflections that do and do not
overlap the bright-field disk at 6 mrad.  A run against itself 50 iterations before its end
is the convergence check.  Fractions are of the particle area.
"""
import sys
from pathlib import Path

import numpy as np

import register_lattice as R
import analysis

E1 = R.E1
SETS = [("quantEM", E1, False), ("quantEM, probe frozen", E1, True),
        ("quantEM, 1 probe mode", E1 / "p1", False),
        ("ePIE", E1 / "epie", False), ("ePIE, model probe fixed", E1 / "epie", True),
        ("ePIE, unbinned", E1 / "epie_bin1", False),
        ("PtyRAD", E1 / "ptyrad", False),
        ("PtyRAD, cosine schedule", E1 / "ptyrad_cosine", False)]
BANDS = {"lattice": R.LATTICE, "{111}+{200}": (0.38, 0.53), "{220}": (0.62, 0.76)}
ALPHA = {23: 24, 24: 12, 25: 6, 26: 3, 27: 1.5}


def main():
    lines = ["| set | scan | alpha | seeds | " + " | ".join(f"agree {b}" for b in BANDS)
             + " | reversed (lattice) | self-agreement, 50 it before end |",
             "|---|---|---|---|" + "---|" * (len(BANDS) + 2)]
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for label, d, fixed in SETS:
        if only and not label.startswith(only):
            continue
        R.E1 = d
        for scan in sorted(ALPHA):
            files = [f for f in R.runs(scan)
                     if ("fixedprobe" in f.name) == fixed and "crop" not in f.name]
            if not files:
                continue
            phs = [R.phases(f) for f in files]
            s = phs[0][1]
            ref = phs[0][0][max(phs[0][0])]
            env = R.envelope(ref, s)
            on = R.particle_area(ref, s)
            agree = {b: [] for b in BANDS}
            rev = []
            for ph, _ in phs[1:]:
                al, _ = R.align(ph[max(ph)], env, s)
                for b, (lo, hi) in BANDS.items():
                    m = R.local_agreement(analysis.band_limit(al, s, lo, hi),
                                          analysis.band_limit(ref, s, lo, hi), s)
                    agree[b].append((m[on] > 0.5).mean())
                    if b == "lattice":
                        rev.append((m[on] < -0.5).mean())
            its = sorted(phs[0][0])
            early = [i for i in its if i <= its[-1] - 50]
            selfa = ""
            if early:
                m = R.local_agreement(R.lattice(phs[0][0][early[-1]], s), R.lattice(ref, s), s)
                selfa = f"{(m[on] > 0.5).mean():.2f}"
            fmt = lambda v: ("/".join(f"{x:.2f}" for x in v)) if v else "--"
            lines.append(f"| {label} | {scan} | {ALPHA[scan]} | {len(files)} | "
                         + " | ".join(fmt(agree[b]) for b in BANDS)
                         + f" | {fmt(rev)} | {selfa} |")
            print(lines[-1], flush=True)
    out = E1 / ("summary.md" if not only else
                f"summary_{only.replace(' ', '').replace(',', '_')}.md")
    out.write_text("Seed agreement of the experimental re-reconstructions (fraction of "
                   "particle area with local correlation > 0.5 against the lowest seed of "
                   "the set; one value per other seed)\n\n" + "\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
