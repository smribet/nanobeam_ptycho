"""Checks for the reviewer response that need no new reconstructions.

    python revision_checks.py

Everything here reads the committed summary.csv files (plus one carbon build for Z5), so
it runs in seconds and needs no GPU.  Writes figures/revision_checks.md.

  Z1  loss equivalence   do reversed or shifted reconstructions fit the data worse than
                         correct ones?  If not, the data cannot prefer the truth, and the
                         spread between initial guesses is degeneracy, not local minima.
  Z2  global shift       could the per-region offsets of one run be a single rigid shift
                         of the whole object, i.e. the trivial probe/object coordinate
                         freedom?  A run with mixed polarity, or with offsets that no one
                         translation reproduces in every region's lattice, cannot be.
  Z3  uncertainty        Wilson intervals on every non-uniqueness rate in Table S1, and
                         exact tests for the comparisons the text leans on.
  Z4  detector sampling  pixels across the bright-field disk, simulated and experimental,
                         and the finest sampling a 192 px detector allows if it must still
                         record the first Bragg ring.
  Z5  carbon support     projected areal density of the published film, and the a-C
                         thickness it corresponds to.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import paths
import analysis
import figpanels
import summary_table

LAM_300 = 0.019687          # Angstrom, 300 keV
EXP_MRAD_PER_PX = 0.93246184  # binned 2x in software, as reconstructed
ARINA_PX = 192
AU_A = 4.08


def _qf_and_G(ref_dir=paths.REFERENCE):
    ref = np.load(paths.sweep(ref_dir) / "runs" / "_reference.npz", allow_pickle=True)
    G = {k: (None if v is None else analysis.reduce_basis(np.array(v)))
         for k, v in json.loads(str(ref["region_G"])).items()}
    return {k: analysis.cell_quantile_fn(v) for k, v in G.items() if v is not None}, G


def _load(d, angle):
    df = pd.read_csv(paths.sweep(d) / "summary.csv")
    df = df[df.angle_mrad == angle].copy()
    df["ns"] = df.seed // 100
    return df


# ---------------------------------------------------------------- Z1

def loss_equivalence(sweeps, angle=3.0):
    """Final loss of runs that came back correct versus runs that came back reversed.

    `final_error` is the normalised amplitude misfit at the last iteration.  Within one
    dataset every run inverts identical data, so a lower loss is a strictly better fit.
    """
    rows = []
    for label, d in sweeps:
        df = _load(d, angle)
        per = (df.groupby(["ns", "seed"])
                 .agg(err=("final_error", "first"),
                      n_rev=("reversed", "sum"))
                 .reset_index())
        for ns, g in per.groupby("ns"):
            ok, bad = g[g.n_rev == 0], g[g.n_rev >= 3]
            rho = spearmanr(g.n_rev, g.err).statistic if g.n_rev.nunique() > 1 else np.nan
            rows.append(dict(
                condition=label, draw=int(ns), runs=len(g),
                err_min=g.err.min(), err_max=g.err.max(),
                spread_pct=100 * (g.err.max() - g.err.min()) / g.err.mean(),
                n_all_correct=len(ok), n_ge3_reversed=len(bad),
                err_all_correct=ok.err.mean() if len(ok) else np.nan,
                err_ge3_reversed=bad.err.mean() if len(bad) else np.nan,
                best_run_n_rev=int(g.loc[g.err.idxmin(), "n_rev"]),
                spearman_nrev_err=rho))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Z2

def global_shift_test(d, angle, QF, G, q_tol=0.25, min_corr=0.5):
    """Fraction of runs whose per-region results no single rigid translation explains.

    A shared translation delta appears in region r as the Wigner-Seitz image of delta in
    r's own lattice.  Every region's measured raw shift is tried as delta, and the run is
    consistent if, for some candidate, every region's residual lies within `q_tol` of its
    cell.  A translation also cannot reverse contrast, so mixed polarity fails outright.
    """
    df = _load(d, angle)
    df = df[df.signed_corr.abs() >= min_corr]
    out = []
    for seed, g in df.groupby("seed"):
        if len(g) < 2:
            continue
        mixed = g["reversed"].nunique() > 1
        consistent = False
        # with two non-collinear reflections no translation reverses contrast, so any
        # reversed region already rules out a coordinate shift
        if not g["reversed"].any():
            S = g[["shift_x", "shift_y"]].to_numpy()
            for delta in S:
                ok = True
                for (_, r), s in zip(g.iterrows(), S):
                    if G.get(r.particle) is None:
                        continue
                    res, _ = analysis.wigner_seitz(s - delta, G[r.particle])
                    if QF[r.particle](np.hypot(*res)) > q_tol:
                        ok = False
                        break
                if ok:
                    consistent = True
                    break
        out.append(dict(seed=int(seed), mixed_polarity=mixed,
                        any_wrong=bool(g["reversed"].any()
                                       or any(QF[p](np.hypot(x, y)) > q_tol
                                              for p, x, y in zip(g.particle, g.ws_x, g.ws_y))),
                        global_shift_consistent=consistent))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- Z3

def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def table_with_ci():
    rows = []
    for name, dirs in summary_table.CONDITIONS:
        angles = (12.0, 6.0, 3.0) if name in summary_table.MULTI_ANGLE else (3.0,)
        for a in angles:
            for dose in summary_table.DOSES:
                fl = figpanels.nonunanimity_groups(dirs[dose], angle=a)
                if fl is None or not len(fl):
                    continue
                k, n = int(fl.disagree.sum()), len(fl)
                lo, hi = wilson(k, n)
                rows.append(dict(condition=name, angle_mrad=a,
                                 dose=summary_table.DLAB[dose], k=k, n=n,
                                 nonuniq=k / n, ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows)


def compare(T, a, b):
    """Two-sided Fisher exact test between two rows of `table_with_ci`."""
    ra = T[(T.condition == a[0]) & (T.angle_mrad == a[1]) & (T.dose == a[2])].iloc[0]
    rb = T[(T.condition == b[0]) & (T.angle_mrad == b[1]) & (T.dose == b[2])].iloc[0]
    _, p = fisher_exact([[ra.k, ra.n - ra.k], [rb.k, rb.n - rb.k]])
    return ra, rb, p


# ---------------------------------------------------------------- Z4

def sampling_table():
    lam = LAM_300
    g111 = np.sqrt(3) / AU_A
    g220 = np.sqrt(8) / AU_A
    rows = []
    for a in (24.0, 12.0, 6.0, 3.0, 1.5):
        dia_mrad = 2 * a
        rows.append(dict(
            alpha_mrad=a,
            sim_px=(dia_mrad * 1e-3 / lam) * 68.0 if a in (3.0, 6.0, 12.0) else np.nan,
            exp_binned_px=dia_mrad / EXP_MRAD_PER_PX,
            exp_unbinned_px=dia_mrad / (EXP_MRAD_PER_PX / 2),
            # finest sampling that still records the whole {111} (or {220}) disk on a
            # 192 px detector, i.e. half-width = theta_g + alpha
            best_px_keep_111=dia_mrad / ((lam * g111 * 1e3 + a) / (ARINA_PX / 2)),
            best_px_keep_220=dia_mrad / ((lam * g220 * 1e3 + a) / (ARINA_PX / 2))))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Z5

def carbon_density():
    import sim
    c = sim.amorphous_carbon()
    n = len(c)
    areal = n / sim.L ** 2
    per_A3 = lambda rho: rho / 12.011 * 0.60221          # atoms per A^3 at rho g/cm^3
    zext = np.ptp(c.positions[:, 2])
    return dict(atoms=n, z_extent_A=zext, areal_per_A2=areal,
                local_density_gcc=areal / zext / 0.60221 * 12.011,
                equiv_aC_nm={rho: areal / per_A3(rho) / 10 for rho in (1.8, 2.0, 2.2)})


# ---------------------------------------------------------------- driver

def main():
    QF, G = _qf_and_G()
    md = []

    md.append("## Z1 loss equivalence (3 mrad)\n")
    Z1 = loss_equivalence([
        ("free-standing noiseless", "ss_bare_complex_inf"),
        ("free-standing 1e6", "ss_bare_complex_1e6"),
        ("free-standing 1e5", "ss_bare_complex_1e5"),
        ("free-standing 1e4", "ss_bare_complex_1e4"),
        ("on carbon 1e4", "ss_carbon_complex_1e4"),
        ("ms pure-phase 1e4", "ms_carbon_a3_phase_1e4"),
    ])
    md.append(Z1.to_markdown(index=False, floatfmt=".4g"))

    md.append("\n\n## Z2 could the offsets be one global shift? (3 mrad)\n")
    for label, d in [("free-standing noiseless", "ss_bare_complex_inf"),
                     ("free-standing 1e5", "ss_bare_complex_1e5"),
                     ("free-standing 1e4", "ss_bare_complex_1e4"),
                     ("ms pure-phase 1e4", "ms_carbon_a3_phase_1e4")]:
        Z2 = global_shift_test(d, 3.0, QF, G)
        wrong = Z2[Z2.any_wrong]
        md.append(f"- {label}: {len(Z2)} runs, {len(wrong)} with any reversed or shifted "
                  f"region; of those, {int(wrong.mixed_polarity.sum())} have mixed polarity "
                  f"across regions, and only {int(wrong.global_shift_consistent.sum())} "
                  f"(shifted, none reversed) could be a single rigid shift of the whole "
                  f"object")

    md.append("\n\n## Z3 non-uniqueness with 95% Wilson intervals\n")
    T = table_with_ci()
    md.append(T.to_markdown(index=False, floatfmt=".2f"))
    md.append("\nGroups within one Poisson draw share data, so these intervals treat "
              "correlated groups as independent and are, if anything, too narrow.\n")
    for a, b in [(("multislice, complex", 3.0, "1e4"), ("multislice + phonons", 3.0, "1e4")),
                 (("matched, on carbon", 3.0, "1e4"), ("multislice + phonons", 3.0, "1e4")),
                 (("multislice, complex", 3.0, "1e5"), ("multislice + phonons", 3.0, "1e5")),
                 (("multislice, complex", 3.0, "1e4"), ("multislice, pure-phase", 3.0, "1e4")),
                 (("multislice, complex", 3.0, "1e5"), ("multislice, pure-phase", 3.0, "1e5"))]:
        ra, rb, p = compare(T, a, b)
        md.append(f"- {a[0]} {a[2]} ({ra.k}/{ra.n}) vs {b[0]} {b[2]} ({rb.k}/{rb.n}): "
                  f"Fisher p = {p:.3g}")

    md.append("\n\n## Z4 pixels across the bright-field disk\n")
    md.append(sampling_table().to_markdown(index=False, floatfmt=".1f"))

    md.append("\n\n## Z5 published carbon film\n")
    C = carbon_density()
    md.append(f"- {C['atoms']} C atoms over {68.0:g}^2 A^2 -> {C['areal_per_A2']:.3f} "
              f"atoms/A^2, folded into a {C['z_extent_A']:.1f} A slab "
              f"({C['local_density_gcc']:.1f} g/cm^3 locally)")
    md.append("- same projected density as a-C of thickness "
              + ", ".join(f"{v:.2f} nm at {r} g/cm^3" for r, v in C["equiv_aC_nm"].items()))

    text = "\n".join(md) + "\n"
    out = paths.figures() / "revision_checks.md"
    out.write_text(text)
    print(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
