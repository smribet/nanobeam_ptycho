"""Regenerate the layout figures into figures/, so they live with the code.

    python make_figures.py                  # the three the paper uses
    python make_figures.py fig_si_phonons   # just one

Paper figures, named as they are in the manuscript:
  fig_sweep         dose, then object model -- one variable per row
  fig_si_substrate  free-standing vs on amorphous carbon, everything else matched
  fig_si_phonons    with and without 8 frozen-phonon configurations

The other two are separate scripts, each doing work this module does not:
`make_fig_si_sparklines.py` lays out its own small-multiple grid, and
`make_fig_stitching.py` re-runs the forward model.

The exploratory groups behind them (fixednoise, ms, phonons, phonons_layout, pair, quad)
still run by name; they are the intermediate figures the final three were chosen from.
"""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import paths
import figpanels as fp

OUT = paths.figures()
DPI = 400
ANGLES = (3.0, 6.0, 12.0)
DOSES = ("inf", "1e6", "1e5", "1e4")
DOSEV = {"inf": None, "1e6": 1e6, "1e5": 1e5, "1e4": 1e4}


def save(fig, name):
    """Write both a raster and a vector copy; the PDF is what goes to the journal."""
    p = OUT / name
    for ext in (".png", ".pdf"):
        fig.savefig(p.with_suffix(ext), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p.with_suffix('.png').name} + .pdf  "
          f"({p.with_suffix('.png').stat().st_size/1e6:.1f} / "
          f"{p.with_suffix('.pdf').stat().st_size/1e6:.1f} MB)")


def fixednoise():
    for tag in ("1e6", "1e5", "1e4"):
        ctx = fp.load(f"ss_carbon_complex_{tag}")
        _, seeds, score = fp.pick_seeds(ctx)
        fig = fp.layout_A(
            ctx, dose=DOSEV[tag], seeds=seeds,
            title=f"Single-slice forward, fixed noise, {fp.DOSE_LAB[tag]}"
                  "  -  3 datasets x 10 initial guesses, 200 iterations")
        save(fig, f"fixednoise_{tag}.png")


def ms():
    for model in ("complex", "phase"):
        for dose in DOSES:
            dirs = {a: f"ms_carbon_a{a:g}_{model}_{dose}" for a in ANGLES}
            if not all((paths.sweep(d) / "summary.csv").exists() for d in dirs.values()):
                print(f"  skipping {model} {dose}: incomplete")
                continue
            ctx = fp.load_multi(dirs)
            _, seeds, score = fp.pick_seeds(ctx)
            fig = fp.layout_A(
                ctx, dose=DOSEV[dose], seeds=seeds,
                title=f"Multislice forward, single-slice {model} inverse, "
                      f"{fp.DOSE_LAB[dose]}  -  200 iterations "
                      f"(artifact score {score:.1f})")
            save(fig, f"ms_{model}_{dose}.png")


def phonons(angle=3.0):
    have = {d: f"msfp_carbon_a{angle:g}_complex_{d}" for d in DOSES
            if (paths.sweep(f"msfp_carbon_a{angle:g}_complex_{d}")
                / "summary.csv").exists()}
    if not have:
        print("  no frozen-phonon runs yet")
        return
    ctxs = {d: fp.load(p) for d, p in have.items()}
    save(fp.layout_artifacts(
        ctxs, angle=angle,
        title=f"Multislice + 8 frozen phonons, single-slice complex inverse - "
              f"{angle:g} mrad, 300 keV"), f"phonons_artifacts_a{angle:g}.png")

    both = {("ms", d): fp.load(f"ms_carbon_a{angle:g}_complex_{d}") for d in have}
    both.update({("msfp", d): ctxs[d] for d in have})
    save(fp.layout_ms(
        both, angle=angle, doses=tuple(have), models=("ms", "msfp"),
        title=f"{angle:g} mrad, complex object: multislice (left) vs "
              f"multislice + 8 frozen phonons (right)"), f"phonons_compare_a{angle:g}.png")


def phonons_layout(sub=None):
    """Layout A for the frozen-phonon runs, standing in `sub` for angles not yet finished."""
    for dose in DOSES:
        dirs = {}
        for a in ANGLES:
            d = f"msfp_carbon_a{a:g}_complex_{dose}"
            if (paths.sweep(d) / "summary.csv").exists():
                dirs[a] = d
            elif sub is not None and (paths.sweep(f"msfp_carbon_a{sub:g}_complex_{dose}")
                                      / "summary.csv").exists():
                dirs[a] = (f"msfp_carbon_a{sub:g}_complex_{dose}", sub)
        if len(dirs) < len(ANGLES):
            print(f"  skipping {dose}: only {sorted(dirs)}")
            continue
        ctx = fp.load_multi(dirs)
        _, seeds, score = fp.pick_seeds(ctx)
        prov = (f"  [PROVISIONAL: {', '.join(f'{a:g}' for a in sorted(ctx['placeholder']))}"
                f" mrad shows {sub:g} mrad data]" if ctx["placeholder"] else "")
        fig = fp.layout_A(
            ctx, dose=DOSEV[dose], seeds=seeds,
            title=f"Multislice + 8 frozen phonons, complex inverse, {fp.DOSE_LAB[dose]}"
                  f"  -  artifact score {score:.1f}{prov}")
        save(fig, f"phonons_layoutA_{dose}.png")


CONDITIONS = {
    "msfp":    lambda d: {a: f"msfp_carbon_a{a:g}_complex_{d}" for a in ANGLES},
    "ms":      lambda d: {a: f"ms_carbon_a{a:g}_complex_{d}" for a in ANGLES},
    "msphase": lambda d: {a: f"ms_carbon_a{a:g}_phase_{d}" for a in ANGLES},
    "matched": lambda d: f"ss_carbon_complex_{d}",
    "bare":    lambda d: f"ss_bare_complex_{d}",
}
LABELS = {"msfp": "multislice + frozen phonons, complex object",
          "ms": "multislice, complex object",
          "msphase": "multislice, pure-phase object",
          "matched": "matched single-slice model, on amorphous carbon",
          "bare": "matched single-slice model, free-standing"}


def pair(cond="msfp", lo="1e5", hi="1e4"):
    """Main-body candidate: one condition at two doses, works vs fails."""
    dirs = {d: CONDITIONS[cond](d) for d in (lo, hi)}
    fig = fp.layout_pair(dirs, doses=(lo, hi),
                         label=f"{LABELS[cond]}  -  300 keV, Au on amorphous carbon, "
                               f"200 iterations")
    save(fig, f"main_pair_{cond}_{lo}_{hi}.png")


def fig_sweep(which="model_then_dose"):
    """Two three-row variants, each changing one variable per step.

    "model_then_dose"  complex -> pure-phase at 1e5, then pure-phase 1e5 -> 1e4
    "dose_then_model"  complex 1e5 -> 1e4, then complex -> pure-phase at 1e4

    Both start from the same essentially-clean control, so the reader can attribute each
    step.  The second is the stricter reading of "change only one thing from the control",
    since dose and model are varied along separate arms from the same origin.
    """
    R = {"c5": ("complex-valued object, $10^5$ e/Å$^2$", CONDITIONS["ms"]("1e5"), 1e5),
         "c4": ("complex-valued object, $10^4$ e/Å$^2$", CONDITIONS["ms"]("1e4"), 1e4),
         "p5": ("pure-phase object, $10^5$ e/Å$^2$", CONDITIONS["msphase"]("1e5"), 1e5),
         "p4": ("pure-phase object, $10^4$ e/Å$^2$", CONDITIONS["msphase"]("1e4"), 1e4)}
    SETS = {"model_then_dose": (["c5", "p5", "p4"],
                                "Object model, then dose: one variable per row"),
            "dose_then_model": (["c5", "c4", "p4"],
                                "Dose, then object model: one variable per row")}
    keys, lab = SETS[which]
    fig = fp.layout_rows([R[k] for k in keys])
    save(fig, "fig_sweep.png" if which == "dose_then_model"
              else f"fig_sweep_{which}.png")


def quad():
    """Main-figure prototype with a free-standing row on top.

    NOTE the first step changes two things at once: the substrate is removed AND the
    forward model reverts to single-slice, because no multislice free-standing sweep was
    run.  Use quad(matched_middle=True) for a strictly one-variable chain, at the cost of
    the multislice rows.
    """
    bare = fp.limit_draws(fp.load("ss_bare_complex_1e5"), 2)   # match the 2 draws elsewhere
    rows = [("free-standing, $10^5$ e/Å$^2$", bare, 1e5),
            ("on carbon, $10^5$ e/Å$^2$", CONDITIONS["ms"]("1e5"), 1e5),
            ("on carbon, $10^4$ e/Å$^2$", CONDITIONS["ms"]("1e4"), 1e4),
            ("pure-phase object, $10^4$ e/Å$^2$", CONDITIONS["msphase"]("1e4"), 1e4)]
    fig = fp.layout_rows(rows)
    save(fig, "main_quad.png")


def fig_si_substrate(dose="1e5"):
    """SI: the substrate control, in the same layout as the main figure.

    Two rows differing in exactly one respect -- whether the particles sit on amorphous
    carbon.  Same matched single-slice model, same dose, same 3 x 10 fixed-noise design,
    so the sparkline blocks are directly comparable row to row.  Each row draws panel (a)
    from its own sample: the free-standing patterns have empty gaps between the disks,
    which is the mechanism the rest of the row is about.
    """
    bare = f"ss_bare_complex_{dose}"
    rows = [(f"free-standing, {DOSE_L[dose]}", bare, DOSEV[dose],
             {"dp_dir": "dp_cache/bare.npz"}),
            (f"on amorphous carbon, {DOSE_L[dose]}", CONDITIONS["matched"](dose),
             DOSEV[dose])]
    fig = fp.layout_rows(rows)
    save(fig, "fig_si_substrate.png")


DOSE_L = {"inf": "noiseless", "1e6": "$10^6$ e/Å$^2$", "1e5": "$10^5$ e/Å$^2$",
          "1e4": "$10^4$ e/Å$^2$"}


def fig_si_phonons(dose="1e4"):
    """SI: what thermal motion costs, isolated to one variable.

    The same multislice data at the same dose, differing only in whether the forward model
    averaged over 8 frozen-phonon configurations.  Phonons are the one model mismatch an
    experiment cannot choose to avoid, and they fail differently from the pure-phase case
    in the main figure: as well as roughly doubling the non-uniqueness they *degrade* the
    reconstructions, which is what makes the ambiguity hard to spot on real data.
    """
    rows = [(f"no thermal motion, {DOSE_L[dose]}", CONDITIONS["ms"](dose), DOSEV[dose]),
            (f"8 frozen phonons, {DOSE_L[dose]}", CONDITIONS["msfp"](dose), DOSEV[dose])]
    fig = fp.layout_rows(rows)
    save(fig, "fig_si_phonons.png")


# The three figures the paper uses.  The other two are their own scripts, because each
# does work this module does not: make_fig_si_sparklines.py lays out its own small-multiple
# grid, and make_fig_stitching.py re-runs the forward model.
PAPER = {"fig_sweep":        lambda: fig_sweep("dose_then_model"),
         "fig_si_substrate": lambda: fig_si_substrate("1e5"),
         "fig_si_phonons":   lambda: fig_si_phonons("1e4")}

EXPLORATORY = ("fixednoise", "ms", "phonons", "phonons_layout", "pair", "quad")


if __name__ == "__main__":
    plt.rcParams["figure.dpi"] = 110
    names = sys.argv[1:] or list(PAPER)
    for g in names:
        print(f"=== {g}")
        if g in PAPER:
            PAPER[g]()
        elif g in EXPLORATORY:                 # kept for re-deriving the intermediate
            globals()[g](*([6.0] if g == "phonons_layout" else []))
        else:
            sys.exit(f"unknown figure {g!r}; choose from "
                     f"{', '.join(list(PAPER) + list(EXPLORATORY))}")
    print(f"\nfigures in {OUT}")
