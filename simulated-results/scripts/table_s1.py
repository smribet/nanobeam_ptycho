"""The revised Table S1: every simulated condition, counted by dataset, on the physical film.

    python table_s1.py          # writes figures/table_s1.{tex,csv}, table_s2.tex, table_s3.tex

Per condition, angle, and dose:
  datasets     non-unanimous datasets / all datasets (a dataset is one Poisson draw, or the
               one noiseless pattern set), the independent unit (draw_level.py)
  non-unique   fraction of (dataset, region) groups whose ten initial guesses disagree, as
               in the submitted table, with a 95% interval from a bootstrap over datasets
  reversed, shifted, |corr|, degraded   as in the submitted table: reversed and shifted
               (q > 0.25) are rates over regions with |corr| >= 0.5, degraded is the rest

On-carbon rows at 3 mrad pool the four realizations of the 2.8 nm film (draw_level.py);
at 6 and 12 mrad they use the R0 film only.  Free-standing rows are the submitted sweeps.

Tables S2 (detector sampling) and S3 (scan geometry) count non-unique region groups at 3 mrad,
free-standing (noiseless and 1e5) and on the R0 film at 1e4.
"""
import numpy as np
import pandas as pd

import draw_level as L
import figpanels
import paths

DOSES = ("inf", "1e6", "1e5", "1e4")
DLAB = {"inf": "noiseless", "1e6": "$10^{6}$", "2e5": "$2\\times10^{5}$",
        "1e5": "$10^{5}$", "1e4": "$10^{4}$"}


def r0(fw, obj, a, d):
    return [f"rev/{fw}_carbon28_a{a:g}_{obj}_{d}"]


BLOCKS = [
    ("matched, on carbon",
     [(a, d, [f"rev/ss_carbon28_complex_{d}"]) for a in (12.0, 6.0) for d in DOSES]
     + [(3.0, d, L.matched(d)) for d in DOSES]),
    ("matched, free-standing",
     [(a, d, [f"ss_bare_complex_{d}"]) for a in (12.0, 6.0, 3.0) for d in DOSES]),
    ("multislice, complex",
     [(a, d, r0("ms", "complex", a, d)) for a in (12.0, 6.0) for d in ("1e5", "1e4")]
     + [(3.0, d, L.pooled("ms", "complex", d)) for d in DOSES]),
    ("multislice, pure-phase",
     [(a, d, r0("ms", "phase", a, d)) for a in (12.0, 6.0) for d in ("1e5", "1e4")]
     + [(3.0, d, L.pooled("ms", "phase", d)) for d in DOSES]),
    ("multislice + phonons",
     [(a, d, r0("msfp", "complex", a, d)) for a in (12.0, 6.0) for d in ("1e5", "1e4")]
     + [(3.0, d, L.pooled("msfp", "complex", d)) for d in DOSES]),
    ("experimental dose",
     [(3.0, "2e5", ["rev/ss_carbon28_complex_2e5"], "matched"),
      (3.0, "2e5", ["rev/mm_ms_known_2e5"], "multislice"),
      (3.0, "2e5", ["rev/mm_msfp_known_2e5"], "+ phonons"),
      (3.0, "2e5", ["rev/mm_msall_known_2e5"], "+ phonons, source, focal"),
      (3.0, "2e5", ["rev/mm_msall_blind_pe_2e5"], "\\quad and refined probe")]),
]


# Tables S2 and S3: non-unique (dataset, region) groups at 3 mrad, as counts.  Columns are
# free-standing noiseless, free-standing 1e5, and on the 2.8 nm film (R0) at 1e4.
MATCHED = ("ss_bare_complex_inf", "ss_bare_complex_1e5", "rev/ss_carbon28_complex_1e4")
TABLE_S2 = [("0.29~mrad/px", "21", MATCHED),
            ("binned $2\\times$", "10", ("rev/samp_bin2_bare_inf", "rev/samp_bin2_bare_1e5",
                                         "rev/samp_bin2_carbon28_1e4")),
            ("binned $4\\times$", "5", ("rev/samp_bin4_bare_inf", "rev/samp_bin4_bare_1e5",
                                        "rev/samp_bin4_carbon28_1e4")),
            ("192-pixel detector", "21", ("rev/samp_crop192_bare_inf",
                                          "rev/samp_crop192_bare_1e5", None)),
            ("$2\\times$ finer sampling", "41", ("rev/samp_field136_bare_inf",
                                                  "rev/samp_field136_bare_1e5",
                                                  "rev/samp_field136_carbon28_1e4")),
            ("$2\\times$ finer, 192-pixel detector", "41",
             ("rev/samp_field136crop192_bare_inf", "rev/samp_field136crop192_bare_1e5", None))]
# refined probes start 1500 A defocus and 150 A astigmatism off (the _pe sweeps), except the
# noiseless regular grid, which was run only from the first runs' 20 A defocus error
TABLE_S3 = [("regular, known", MATCHED),
            ("perturbed, known", ("rev/jit_known_bare_inf", "rev/jit_known_bare_1e5",
                                  "rev/jit_known_carbon28_1e4")),
            ("regular, refined", ("rev/grid_blind_bare_inf", "rev/grid_blind_pe_bare_1e5",
                                  "rev/grid_blind_pe_carbon28_1e4")),
            ("perturbed, refined", ("rev/jit_blind_pe_bare_inf", "rev/jit_blind_pe_bare_1e5",
                                    "rev/jit_blind_pe_carbon28_1e4"))]


def counts(sweep, angle=3.0):
    if sweep is None:
        return "--"
    g = figpanels.nonunanimity_groups(sweep, angle=angle)
    return "missing" if g is None else f"{int(g.disagree.sum())}/{len(g)}"


def small_tables(out):
    s2 = [f"{lab} & {px} & " + " & ".join(counts(s) for s in sw) + " \\\\"
          for lab, px, sw in TABLE_S2]
    s3 = [f"{lab} & " + " & ".join(counts(s) for s in sw) + " \\\\" for lab, sw in TABLE_S3]
    (out / "table_s2.tex").write_text("\n".join(s2) + "\n")
    (out / "table_s3.tex").write_text("\n".join(s3) + "\n")
    print("\nTable S2\n" + "\n".join(s2) + "\n\nTable S3\n" + "\n".join(s3))


def regions(sweeps, angle, QF, min_corr=0.5, q_tol=0.25):
    frames = []
    for s in sweeps:
        f = paths.sweep(s) / "summary.csv"
        if f.exists():
            frames.append(pd.read_csv(f))
    df = pd.concat(frames).dropna(subset=["ws_x"])
    df = df[df.angle_mrad == angle]
    q = np.array([QF[p](np.hypot(x, y)) for p, x, y in zip(df.particle, df.ws_x, df.ws_y)])
    good = (df.signed_corr.abs() >= min_corr).values
    rev = df["reversed"].values.astype(bool)
    return dict(runs=len(df) // df.particle.nunique(),
                reversed=rev[good].mean() if good.any() else np.nan,
                shifted=(q[good] > q_tol).mean() if good.any() else np.nan,
                acorr=df.signed_corr.abs().mean(), degraded=(~good).mean())


def main():
    QF = figpanels._default_QF()
    rows = []
    for block, entries in BLOCKS:
        for e in entries:
            a, d, sweeps = e[:3]
            name = e[3] if len(e) > 3 else block
            D = L.datasets(sweeps, angle=a)
            if not len(D):
                print(f"  missing: {block} {a:g} {d}")
                continue
            k = int((D[:, 0] > 0).sum())
            g = D[:, 0].sum() / D[:, 1].sum()
            lo, hi = L.bootstrap(D) if len(D) > 1 else (np.nan, np.nan)
            rows.append(dict(block=block, condition=name, angle=a, dose=d,
                             films=sum((paths.sweep(s) / "summary.csv").exists()
                                       for s in sweeps),
                             datasets=len(D), nonunanimous=k, nonuniq=g, lo=lo, hi=hi,
                             **regions(sweeps, a, QF)))
    T = pd.DataFrame(rows)
    out = paths.figures()
    T.to_csv(out / "table_s1.csv", index=False)

    def f(v, p=2):
        return "--" if not np.isfinite(v) else f"{v:.{p}f}"

    tex, last = [], None
    for r in T.itertuples():
        if last is not None and r.block != last:
            tex.append("\\hline")
        last = r.block
        ci = (f"{f(r.nonuniq)} [{f(r.lo)}, {f(r.hi)}]" if np.isfinite(r.lo)
              else f"{f(r.nonuniq)}")
        tex.append(f"{r.condition} & {r.angle:g}~mrad & {DLAB[r.dose]} & "
                   f"{r.nonunanimous}/{r.datasets} & {ci} & {f(r.reversed)} & "
                   f"{f(r.shifted)} & {f(r.acorr, 3)} & {f(r.degraded)} \\\\")
    (out / "table_s1.tex").write_text("\n".join(tex) + "\n")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(T.drop(columns=["block"]).round(3).to_string(index=False))
    small_tables(out)
    print(f"\nwrote {out / 'table_s1.tex'} and .csv, table_s2.tex, table_s3.tex")


if __name__ == "__main__":
    main()
