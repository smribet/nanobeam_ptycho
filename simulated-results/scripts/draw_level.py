"""Non-uniqueness counted by dataset, the independent unit, for the revised Table S1.

    python draw_level.py        # writes figures/draw_level.{md,csv}

The regions of one dataset fail together (a non-unanimous dataset typically has 4-7 of
its 7 regions disagree, a unanimous one none), so the (dataset, region) groups behind the
published rates are not independent, and intervals computed over groups are too narrow.
Each condition is therefore reported as

  datasets   non-unanimous datasets out of all (a dataset = one Poisson draw, or the one
             noiseless pattern set), with a Wilson 95% interval over datasets
  groups     the published rate, non-unanimous (dataset, region) groups over all groups,
             with a 95% interval from a bootstrap that resamples whole datasets
  regions    mean number of disagreeing regions in a non-unanimous dataset

On-carbon rows pool every realization of the 2.8 nm film that was run for that condition
(the R0 one, seed 10, and seeds 11-13), each realization's draws as separate datasets.
"""
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

import figpanels
import paths
from revision_checks import wilson

FILMS = (11, 12, 13)
DOSES = ("inf", "1e6", "2e5", "1e5", "1e4")


def r0(fw, obj, d):
    return f"rev/{fw}_carbon28_a3_{obj}_{d}"


def pooled(fw, obj, d):
    return [r0(fw, obj, d)] + [f"rev/pool_{fw}_carbon28_s{c}_a3_{obj}_{d}" for c in FILMS]


def matched(d):
    return [f"rev/ss_carbon28_complex_{d}"] + [f"rev/film_new_s{c}_{d}" for c in FILMS]


ROWS = (
    [("free-standing, matched", d, [f"ss_bare_complex_{d}"]) for d in ("inf", "1e6", "1e5", "1e4")]
    + [("2.8 nm a-C, matched", d, matched(d)) for d in DOSES]
    + [("2.8 nm a-C, multislice", d, pooled("ms", "complex", d)) for d in ("inf", "1e6", "1e5", "1e4")]
    + [("2.8 nm a-C, multislice, pure phase", d, pooled("ms", "phase", d)) for d in ("inf", "1e6", "1e5", "1e4")]
    + [("2.8 nm a-C, multislice + phonons", d, pooled("msfp", "complex", d)) for d in ("inf", "1e6", "1e5", "1e4")]
    # N9: the unavoidable mismatches, on the R0 film
    + [("N9: multislice", "2e5", ["rev/mm_ms_known_2e5"]),
       ("N9: multislice, refined probe", "1e5", ["rev/mm_ms_blind_pe_1e5"]),
       ("N9: multislice, refined probe", "2e5", ["rev/mm_ms_blind_pe_2e5"]),
       ("N9: + source size", "1e5", ["rev/mm_mssrc_known_1e5"]),
       ("N9: + source size", "2e5", ["rev/mm_mssrc_known_2e5"]),
       ("N9: + focal spread", "1e5", ["rev/mm_msfoc_known_1e5"]),
       ("N9: + focal spread", "2e5", ["rev/mm_msfoc_known_2e5"]),
       ("N9: + phonons", "2e5", ["rev/mm_msfp_known_2e5"]),
       ("N9: all three", "1e5", ["rev/mm_msall_known_1e5"]),
       ("N9: all three", "2e5", ["rev/mm_msall_known_2e5"]),
       ("N9: all three, refined probe", "1e5", ["rev/mm_msall_blind_pe_1e5"]),
       ("N9: all three, refined probe", "2e5", ["rev/mm_msall_blind_pe_2e5"])]
    # the support: physical film thickness (wrap-free scoring), and the published film
    + [(f"a-C {t / 10:g} nm, matched", d, [f"rev/dw_thick_ss_c{t}_{d}"])
       for t in (10, 28, 50, 100, 200) for d in ("1e5", "1e4")]
    + [(f"a-C {t / 10:g} nm, multislice", "1e5", [f"rev/dw_thick_ms_c{t}_1e5"])
       for t in (10, 28, 50, 100, 200)]
    + [("published film, matched", d, [f"ss_carbon_complex_{d}"]
        + [f"rev/film_legacy_s{c}_{d}" for c in (11, 12)]) for d in ("1e5", "1e4")]
)


def datasets(sweeps, angle=3.0):
    """Per dataset: (number of disagreeing regions, number of regions), over all sweeps."""
    out = []
    for sw in sweeps:
        if not (paths.sweep(sw) / "summary.csv").exists():
            continue
        fl = figpanels.nonunanimity_groups(sw, angle=angle)
        if fl is None or not len(fl):
            continue
        for _, g in fl.groupby("ns"):
            out.append((int(g.disagree.sum()), len(g)))
    return np.array(out, dtype=int).reshape(-1, 2)


def quality(sweeps, angle=3.0):
    frames = [pd.read_csv(paths.sweep(s) / "summary.csv") for s in sweeps
              if (paths.sweep(s) / "summary.csv").exists()]
    df = pd.concat(frames)
    df = df[df.angle_mrad == angle]
    good = df.signed_corr.abs() >= 0.5
    return df.signed_corr.abs().mean(), (~good).mean(), df[good].reversed.mean()


def bootstrap(D, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(D), size=(n, len(D)))
    r = D[i, 0].sum(1) / D[i, 1].sum(1)
    return np.quantile(r, [0.025, 0.975])


def main():
    rows = []
    for name, dose, sweeps in ROWS:
        D = datasets(sweeps)
        if not len(D):
            continue
        k = int((D[:, 0] > 0).sum())
        lo, hi = wilson(k, len(D))
        g = D[:, 0].sum() / D[:, 1].sum()
        blo, bhi = bootstrap(D) if len(D) > 1 else (np.nan, np.nan)
        acorr, degraded, rev = quality(sweeps)
        rows.append(dict(condition=name, dose=dose, films=len(sweeps), datasets=len(D),
                         nonunanimous=k, ds_rate=k / len(D), ds_lo=lo, ds_hi=hi,
                         groups=int(D[:, 1].sum()), group_rate=g, g_lo=blo, g_hi=bhi,
                         regions_per_failed=(D[D[:, 0] > 0, 0].mean() if k else 0.0),
                         acorr=acorr, degraded=degraded, reversed=rev))
    T = pd.DataFrame(rows)
    out = paths.figures()
    T.to_csv(out / "draw_level.csv", index=False)

    def f(v, p=2):
        return "--" if not np.isfinite(v) else f"{v:.{p}f}"

    md = ["| condition | dose | datasets non-unanimous | 95% CI | groups non-unique | 95% CI (bootstrap) "
          "| regions per failed dataset | \\|corr\\| | degraded |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in T.itertuples():
        md.append(f"| {r.condition} | {r.dose} | {r.nonunanimous}/{r.datasets} | "
                  f"{f(r.ds_lo)}-{f(r.ds_hi)} | {f(r.group_rate)} | {f(r.g_lo)}-{f(r.g_hi)} | "
                  f"{f(r.regions_per_failed, 1)} | {f(r.acorr, 3)} | {f(r.degraded)} |")

    # the comparisons the letter quotes, by dataset (Fisher exact, two-sided)
    def row(c, d):
        return T[(T.condition == c) & (T.dose == d)].iloc[0]

    md += ["", "| comparison | dose | datasets | p (Fisher, by dataset) |", "|---|---|---|---|"]
    for a, b, d in [("2.8 nm a-C, multislice + phonons", "2.8 nm a-C, multislice", x)
                    for x in ("1e6", "1e5", "1e4")] + \
                   [("2.8 nm a-C, multislice, pure phase", "2.8 nm a-C, multislice", x)
                    for x in ("inf", "1e6", "1e5", "1e4")] + \
                   [("N9: all three, refined probe", "2.8 nm a-C, matched", "2e5"),
                    ("N9: all three", "2.8 nm a-C, matched", "2e5"),
                    ("2.8 nm a-C, matched", "published film, matched", "1e4"),
                    ("2.8 nm a-C, matched", "published film, matched", "1e5")]:
        ra, rb = row(a, d), row(b, d)
        p = fisher_exact([[ra.nonunanimous, ra.datasets - ra.nonunanimous],
                          [rb.nonunanimous, rb.datasets - rb.nonunanimous]])[1]
        md.append(f"| {a} vs {b} | {d} | {ra.nonunanimous}/{ra.datasets} vs "
                  f"{rb.nonunanimous}/{rb.datasets} | {p:.2g} |")
    (out / "draw_level.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
