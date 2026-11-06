"""Summary plots for the response to the reviewers, from the sweep tables alone.

    python make_fig_response.py              # every figure, into figures/response/
    python make_fig_response.py carbon size  # just these

Each needs only the sweeps' summary.csv files (and, for `carbon`, the sample builder), so
the figures can be made away from the GPU hosts.  Rates are those of the revised Table S1
(draw_level.py): the fraction of (dataset, region) groups whose ten initial guesses disagree,
with a 95% interval from a bootstrap over datasets, or, where marked, the fraction of
datasets in which any region disagrees, with a Wilson interval.

  mismatch    comment 1.1: multislice, pure phase, and thermal motion against dose; the
              unavoidable mismatches at the experimental 2e5 e/A^2
  carbon      comment 2.7: support thickness, film realizations, and each film's own diffuse
              diffraction intensity
  size        comment 2.1: the [111] particle at 6 mrad against its size and its neighbors
  loss        comment 2.4: the misfit reached from random, true, and gauge-transformed starts
  dynamical   comment 2.6: position dependence of thick-crystal patterns against angle
"""
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import draw_level as L
import figpanels as fp
import paths
from revision_checks import wilson

OUT = paths.figures() / "response"
# the colorblind-safe categorical set of seaborn ('colorblind'), in a fixed order
BLUE, ORANGE, GREEN, RED, PURPLE, BROWN, GRAY = ("#0173B2", "#DE8F05", "#029E73", "#D55E00",
                                                 "#CC78BC", "#CA9161", "#949494")
DOSE_LAB = {"inf": "∞", "1e6": "$10^6$", "2e5": "$2{\\times}10^5$", "1e5": "$10^5$",
            "1e4": "$10^4$"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
                     "legend.frameon": False, "axes.linewidth": 0.8})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf")


def rate(sweeps, angle=3.0, n_boot=4000, seed=0):
    """(group rate, lo, hi, datasets non-unanimous, datasets) pooled over `sweeps`."""
    D = L.datasets(sweeps, angle)
    if not len(D):
        return (np.nan,) * 5
    g = D[:, 0].sum() / D[:, 1].sum()
    if len(D) > 1:
        rng = np.random.default_rng(seed)
        i = rng.integers(0, len(D), (n_boot, len(D)))
        lo, hi = np.quantile(D[i, 0].sum(1) / D[i, 1].sum(1), [0.025, 0.975])
    else:
        lo = hi = np.nan
    return g, lo, hi, int((D[:, 0] > 0).sum()), len(D)


def errdot(ax, x, r, color, label=None, marker="o", ms=5):
    g, lo, hi = r[:3]
    yerr = None if not np.isfinite(lo) else [[g - lo], [hi - g]]
    ax.errorbar([x], [g], yerr=yerr, fmt=marker, color=color, ms=ms, capsize=2.5, lw=1.1,
                label=label, mec="white", mew=0.6)


def rate_axis(ax, label="non-unique fraction\nof region groups"):
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel(label)
    ax.grid(axis="y", color="0.9", lw=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------- figures

def mismatch():
    """(a, b) each mismatch against the same static multislice data, by dose; (c) every
    unavoidable mismatch at the experimental dose, by dataset."""
    # (a) and (b) share their dose and rate axes, so they sit close together; (c) needs
    # room on its left for its row labels
    fig = plt.figure(figsize=(7.4, 2.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.0, 1.25], wspace=0.95,
                          left=0.09, right=0.98, bottom=0.18, top=0.88)
    gl = gs[0].subgridspec(1, 2, wspace=0.08)
    a = fig.add_subplot(gl[0])
    b = fig.add_subplot(gl[1], sharey=a)
    c = fig.add_subplot(gs[1])
    doses = ("inf", "1e6", "1e5", "1e4")
    static = lambda d: L.pooled("ms", "complex", d)
    for ax, (lab, fn, col), title in (
            (a, ("+ phonons", lambda d: L.pooled("msfp", "complex", d), RED), "(a) thermal motion"),
            (b, ("pure-phase object", lambda d: L.pooled("ms", "phase", d), ORANGE),
             "(b) pure-phase constraint")):
        for i, d in enumerate(doses):
            errdot(ax, i - 0.1, rate(static(d)), GREEN, label="multislice" if i == 0 else None)
            errdot(ax, i + 0.1, rate(fn(d)), col, label=lab if i == 0 else None, marker="s")
        ax.set_xticks(range(len(doses)), [DOSE_LAB[d] for d in doses])
        ax.set_xlim(-0.5, 3.5)
        ax.set_xlabel("dose (e/Å$^2$)")
        rate_axis(ax)
        ax.set_title(title)
        # the upper left of both panels is empty
        ax.legend(loc="upper left", fontsize=7, handletextpad=0.3, borderaxespad=0.4)
    b.set_ylabel("")
    b.tick_params(labelleft=False)

    rows = [("no mismatch (matched)", ["rev/ss_carbon28_complex_2e5"], BLUE),
            ("multislice", ["rev/mm_ms_known_2e5"], GREEN),
            ("+ phonons", ["rev/mm_msfp_known_2e5"], RED),
            ("+ phonons, source, focal", ["rev/mm_msall_known_2e5"], BROWN),
            ("  and refined probe", ["rev/mm_msall_blind_pe_2e5"], PURPLE)]
    for k, (lab, sw, col) in enumerate(rows):
        _, _, _, kk, n = rate(sw)
        lo, hi = wilson(kk, n)
        y = len(rows) - 1 - k
        c.errorbar([kk / n], [y], xerr=[[kk / n - lo], [hi - kk / n]], fmt="s", color=col,
                   ms=5, capsize=2.5, lw=1.1, mec="white", mew=0.6)
        c.text(0.98, y, f"{kk}/{n}", va="center", ha="right", fontsize=7, color="0.3",
               transform=c.get_yaxis_transform())
    c.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], fontsize=7)
    c.set_xlim(-0.04, 1.0)
    c.set_xlabel("datasets with any disagreement")
    c.grid(axis="x", color="0.9", lw=0.6)
    c.set_axisbelow(True)
    c.set_title("(c) at the experimental 2×10$^5$ e/Å$^2$")
    save(fig, "fig_resp_mismatch")


def _diffuse(film_seed=10, legacy=False, n_bins=60):
    """Azimuthally averaged diffraction intensity of the carbon film alone: the
    transmission of particles + film divided by that of the particles, in projection."""
    import sim
    on = sim.build_sample(support=True, carbon_thickness=None if legacy else 28.0,
                          carbon_seed=film_seed)
    off = sim.build_sample(support=False)
    t_on, _, s_ = sim.projected_object(on)
    t_off, _, _ = sim.projected_object(off)
    t = t_on * np.conj(t_off)
    F = np.abs(np.fft.fft2(t - t.mean())) ** 2 / t.size ** 2
    f = np.fft.fftfreq(t.shape[0], s_)
    q = np.hypot(*np.meshgrid(f, f, indexing="ij"))
    edges = np.linspace(0.05, 0.9, n_bins + 1)
    I = np.array([F[(q >= a) & (q < b)].mean() for a, b in zip(edges[:-1], edges[1:])])
    return 0.5 * (edges[1:] + edges[:-1]), I


def carbon():
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.4, 2.5),
                                  gridspec_kw=dict(width_ratios=[1.2, 0.85, 1.1]))
    th = (10, 28, 50, 100, 200)
    for lab, fn, col, dx, mk in (
            ("matched, $10^5$", lambda t: [f"rev/dw_thick_ss_c{t}_1e5"], BLUE, -0.12, "o"),
            ("matched, $10^4$", lambda t: [f"rev/dw_thick_ss_c{t}_1e4"], ORANGE, 0.0, "s"),
            ("multislice, $10^5$", lambda t: [f"rev/dw_thick_ms_c{t}_1e5"], GREEN, 0.12, "^")):
        for i, t in enumerate(th):
            errdot(a, i + 1 + dx, rate(fn(t)), col, label=lab if i == 0 else None, marker=mk)
    errdot(a, -0.12, rate(["ss_bare_complex_1e5"]), BLUE)
    errdot(a, 0.0, rate(["ss_bare_complex_1e4"]), ORANGE, marker="s")
    a.set_xticks(range(6), ["0", "1", "2.8", "5", "10", "20"])
    a.set_xlim(-0.5, 5.5)
    a.set_xlabel("physical a-C thickness (nm)")
    rate_axis(a)
    a.set_title("(a) support thickness")
    a.legend(loc="lower left", fontsize=7)

    for j, d in enumerate(("1e5", "1e4")):
        phys = [f"rev/ss_carbon28_complex_{d}"] + [f"rev/film_new_s{c}_{d}" for c in L.FILMS]
        pub = [f"ss_carbon_complex_{d}"] + [f"rev/film_legacy_s{c}_{d}" for c in (11, 12)]
        b.plot(j - 0.15 + 0.06 * np.arange(len(pub)) - 0.06, [rate([sw])[0] for sw in pub],
               "o", color=GRAY, ms=4.5, mec="white", mew=0.5,
               label="folded 8 Å layer" if j == 0 else None)
        b.plot(j + 0.15 + 0.06 * np.arange(len(phys)) - 0.09, [rate([sw])[0] for sw in phys],
               "D", color=BLUE, ms=4.5, mec="white", mew=0.5,
               label="physical 2.8 nm" if j == 0 else None)
    b.set_xticks([0, 1], ["$10^5$", "$10^4$"])
    b.set_xlim(-0.55, 1.55)
    b.set_xlabel("dose (e/Å$^2$)")
    rate_axis(b, "")
    b.set_title("(b) one point per film")
    b.legend(loc="center left", fontsize=7)

    for lab, kw, col in (("folded 8 Å layer", dict(legacy=True), GRAY),
                         ("physical 2.8 nm", dict(), BLUE)):
        q, I = _diffuse(**kw)
        c.semilogy(q, I, color=col, lw=1.4, label=lab)
    lam = 0.0196875
    bf, g111, g200 = 3e-3 / lam, 0.4245, 0.4902
    c.axvspan(bf, g111 - bf, color=ORANGE, alpha=0.15, lw=0)
    c.axvline(bf, color="0.5", lw=0.8, ls=":")
    c.text(bf + 0.005, 0.97, " BF edge", transform=c.get_xaxis_transform(), fontsize=6.5,
           va="top", color="0.35")
    c.text((bf + g111 - bf) / 2, 0.06, "gap to\n{111}", transform=c.get_xaxis_transform(),
           fontsize=6.5, ha="center", color="0.35")
    c.set(xlim=(0.05, 0.9), xlabel="q (Å$^{-1}$)", ylabel="diffuse intensity (arb.)")
    c.set_title("(c) the films' own scattering at 3 mrad")
    c.legend(loc="upper right", fontsize=7)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_resp_carbon")


def _wrong_111(sweep, angle):
    """Fraction of runs whose [111] lattice is reversed or shifted > 1/4 cell, Wilson CI."""
    d = pd.read_csv(paths.sweep(sweep) / "summary.csv")
    d = d[(d.angle_mrad == angle) & (d.particle == "au_111") & (d.signed_corr.abs() >= 0.5)]
    q = fp._default_QF()["au_111"](np.hypot(d.ws_x, d.ws_y))
    k, n = int((d.reversed | (q > 0.25)).sum()), len(d)
    lo, hi = wilson(k, n)
    return k / n, lo, hi


def size():
    """One metric throughout: the fraction of runs whose [111] lattice comes back reversed
    or shifted by more than a quarter cell, over 4 draws x 10 initial guesses at 1e5."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw=dict(width_ratios=[1, 1.6]))
    D0 = 2.0
    for angle, col, mk, dx in ((6.0, ORANGE, "o", -0.04), (3.0, RED, "s", 0.04)):
        r = [_wrong_111(f"rev/single111_scale{s:g}_bare_1e5", angle) for s in (1, 1.6, 2)]
        x = D0 * np.array([1, 1.6, 2]) + dx
        a.plot(x, [v[0] for v in r], "-", color=col, lw=0.8, alpha=0.5)
        for xi, v in zip(x, r):
            errdot(a, xi, v, col, marker=mk, label=None)
        a.plot([], [], mk, color=col, label=f"{angle:g} mrad")
    a.set(xlabel="diameter of the isolated [111] particle (nm)", xlim=(1.6, 4.4))
    rate_axis(a, "[111] runs reversed or\nshifted > 1/4 cell")
    a.set_title("(a) particle size, [111] alone")
    a.legend(loc="center right")

    series1 = [("alone", "rev/single111_pubpos_bare_1e5"),
               ("20 Å", "rev/pair_gap20_bare_1e5"), ("10 Å", "rev/pair_gap10_bare_1e5"),
               ("5 Å", "rev/pair_gap5_bare_1e5")]
    series2 = [("alone", "rev/single111_pubpos_bare_1e5"),
               ("+[100]", "rev/pair_pub100_bare_1e5"), ("+twin", "rev/pair_pubdeca_bare_1e5"),
               ("+[100]\n+twin", "ss_bare_complex_1e5")]
    for x0, ser, col in ((0, series1, BLUE), (5, series2, GREEN)):
        r = [_wrong_111(sw, 6.0) for _, sw in ser]
        x = x0 + np.arange(len(ser))
        b.plot(x, [v[0] for v in r], "-", color=col, lw=0.8, alpha=0.5)
        for xi, v in zip(x, r):
            errdot(b, xi, v, col)
    b.set_xticks(list(range(4)) + list(range(5, 9)),
                 [l for l, _ in series1] + [l for l, _ in series2], fontsize=7)
    b.axvline(4, color="0.8", lw=0.8)
    b.text(1.5, 0.97, "one [100] neighbor, edge gap", ha="center", va="top", fontsize=7,
           color=BLUE, transform=b.get_xaxis_transform())
    b.text(6.5, 0.97, "neighbors in the sample", ha="center", va="top", fontsize=7,
           color=GREEN, transform=b.get_xaxis_transform())
    b.set_xlim(-0.5, 8.5)
    rate_axis(b, "")
    b.set_ylim(-0.05, 0.75)
    b.set_title("(b) the [111] at 6 mrad, at its position in the sample")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_resp_size")


# a count of reversed regions, 0-7: one discrete color each
NREV_CMAP = matplotlib.colors.ListedColormap(plt.get_cmap("viridis")(np.linspace(0, 1, 8)))
NREV_NORM = matplotlib.colors.BoundaryNorm(np.arange(-0.5, 8.5), 8)


def loss():
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for ax, (title, rand, pref) in zip(axs, (
            ("(a) free-standing", "ss_bare_complex_1e5", "bare"),
            ("(b) on 2.8 nm a-C", "rev/ss_carbon28_complex_1e5", "carbon28"))):
        for k in (0, 1):
            x0 = 3 * k
            d = pd.read_csv(paths.sweep(rand) / "summary.csv")
            d = d[(d.angle_mrad == 3.0) & (d.seed // 100 == k)]
            runs = d.groupby("seed").agg(err=("final_error", "first"), nrev=("reversed", "sum"))
            e0 = None
            for j, (init, col, mk) in enumerate((("truth", BLUE, "o"), ("reversed", RED, "s"),
                                                 ("shifted", ORANGE, "D"))):
                e = pd.read_csv(paths.sweep(f"rev/init_{init}_{pref}_1e5") / "summary.csv")
                e = e[(e.angle_mrad == 3.0) & (e.seed // 100 == k)]
                v = e.groupby("seed").final_error.first()
                e0 = v.mean() if init == "truth" else e0
                ax.scatter(x0 + 0.75 + 0.3 * j + np.zeros(len(v)), 100 * (v / e0 - 1),
                           color=col, marker=mk, s=18, edgecolor="white", linewidth=0.4,
                           zorder=3, label=(f"from the {init} truth" if init != "truth"
                                            else "from the truth") if k == 0 else None)
            sc = ax.scatter(x0 + np.random.default_rng(k).uniform(-0.3, 0.3, len(runs)),
                            100 * (runs.err / e0 - 1), c=runs.nrev, cmap=NREV_CMAP, norm=NREV_NORM, s=16, edgecolor="white", linewidth=0.4, zorder=3)
            te = e.groupby("seed").truth_error.first().iloc[0]
            tes = getattr(ax, "_tes", []) + [100 * (te / e0 - 1)]
            ax._tes = tes
        ax.axhline(0, color="0.8", lw=0.6)
        ax.set_xticks([0, 0.75 + 0.3, 3, 3.75 + 0.3], ["random\nstarts", "seeded\nstarts"] * 2,
                      fontsize=7)
        ax.text(0.45, -0.24, "dataset 1", transform=ax.get_xaxis_transform(), ha="center",
                fontsize=7)
        ax.text(3.45, -0.24, "dataset 2", transform=ax.get_xaxis_transform(), ha="center",
                fontsize=7)
        ax.set(ylim=(-0.3, 1.7), xlim=(-0.6, 4.6))
        ax.set_title(f"{title}\n(ground truth: +{min(ax._tes):.0f}%, off scale)")
    axs[0].set_ylabel("misfit above the truth-started run (%)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=3, fontsize=7)
    cb = fig.colorbar(sc, ax=axs, shrink=0.85, pad=0.01, ticks=range(8))
    cb.set_label("number of the 7 regions reversed\n(random starts)", fontsize=7)
    save(fig, "fig_resp_loss")


def dynamical():
    import json
    d = json.loads((paths.figures() / "dynamical_check.json").read_text())
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    for t, col, mk in ((20.0, BLUE, "o"), (50.0, ORANGE, "s"), (100.0, GREEN, "^")):
        r = sorted((x["alpha_mrad"], x["max_rel_dev"]) for x in d if x["thickness_A"] == t)
        ax.semilogy(*zip(*r), marker=mk, color=col, lw=1.1, ms=4.5, label=f"{t / 10:g} nm")
    for a, lab in ((3.41, "FOLZ"), (4.83, "{200}")):
        ax.axvline(a, color="0.6", lw=0.8, ls=":")
        ax.text(a, 1.6, lab, ha="center", va="bottom", fontsize=6.5, color="0.35")
    ax.set(xlabel="convergence semi-angle (mrad)", xlim=(2.5, 12.5), ylim=(2e-7, 1.5),
           ylabel="max$_R$ ‖I$_R$ − ⟨I⟩‖ / ‖⟨I⟩‖")
    ax.legend(loc="lower right", fontsize=7, title="Au [001] slab", title_fontsize=7)
    ax.set_title("Thick crystals: dependence of the patterns\non probe position", pad=12)
    fig.tight_layout()
    save(fig, "fig_resp_dynamical")


FIGURES = dict(mismatch=mismatch, carbon=carbon, size=size, loss=loss, dynamical=dynamical)

if __name__ == "__main__":
    for name in sys.argv[1:] or FIGURES:
        print(f"=== {name}")
        FIGURES[name]()
