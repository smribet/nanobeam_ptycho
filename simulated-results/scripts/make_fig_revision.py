"""Figures for the revision, from the stored "show" runs and the stage-2 tests.

    python make_fig_revision.py                  # every figure whose data exist
    python make_fig_revision.py seeded pair      # just these

Each figure is written to figures/revision/ as a PDF for the SI and a small JPEG preview;
the preview is all that needs to leave the GPU host.  The panels follow the paper's: the
band-limited phase in gray, regions returned reversed outlined in red and regions shifted
by more than a quarter cell (or with a competing correlation peak) in amber.

  seeded     N2: ground-truth starts, plain and gauge-transformed, free-standing and on
             carbon, through 200 iterations, with each region's polarity history
  pair       N5b: the [111] particle at 6 mrad alone and beside the [100] one at three
             gaps, and how often it comes back wrong at each gap
  search     gauge search: the reconstruction before and after, and the misfit of every
             candidate registry relative to the correct one
  thickness  N4: one reconstruction per carbon thickness, scored as published and with
             the film's slowly varying phase divided out before the phase is taken
  probe      N3: probes refined in blind runs, regular and jittered scans, against the
             true and the starting probe
  mixed      the four modes of the mixed-state object, with and without thermal motion
  mismatch   N9: initial guesses on the all-mismatch data at 2e5, known and refined probe
"""
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import analysis
import figpanels as fp
import paths
import sim
from revision_checks import wilson

OUT = paths.figures() / "revision"
DPI, PREVIEW_DPI = 300, 70
COL = {"reversed": "#ff2d2d", "shifted": "#ffb000"}
C6 = "#ff7f0e"                 # the paper's 6 mrad color
C3 = "#d62728"                 # and 3 mrad
PX = 1.30                      # inches per image panel


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", dpi=DPI, bbox_inches="tight")
    # a JPEG preview, small enough to look at over a slow link
    fig.savefig(OUT / f"{name}_preview.jpg", dpi=PREVIEW_DPI, bbox_inches="tight",
                pil_kwargs=dict(quality=70))
    plt.close(fig)
    print(f"  wrote {name}.pdf ({(OUT / f'{name}.pdf').stat().st_size / 1e6:.1f} MB) "
          f"+ preview ({(OUT / f'{name}_preview.jpg').stat().st_size / 1e3:.0f} kB)")


class Sweep:
    """One sweep's ground truth, region masks, and per-run outputs."""

    def __init__(self, name, ref=None):
        self.name, self.dir = name, self.where(name)
        r = np.load(self.where(ref or name) / "runs" / "_reference.npz", allow_pickle=True)
        self.truth = r["truth"].astype(float)
        self.sampling, self.L = float(r["sampling"]), float(r["L"])
        self.particles = json.loads(str(r["particles"]))
        self.masks = dict(zip(json.loads(str(r["region_names"])), r["region_masks"]))
        f = self.dir / "summary.csv"
        self.df = pd.read_csv(f) if f.exists() else None
        f = self.dir / "drift.csv"
        self.drift = pd.read_csv(f).assign(noise_seed=0) if f.exists() else None

    @staticmethod
    def where(name):
        """A revision sweep (outputs/rev/), or else a published one (outputs/)."""
        d = paths.sweep(f"rev/{name}")
        return d if d.exists() else paths.sweep(name)

    @staticmethod
    def exists(name):
        return (Sweep.where(name) / "runs").exists()

    def seeds(self, angle):
        return sorted(self.df[self.df.angle_mrad == angle].seed.unique())

    def stored(self, angle, n):
        """The first `n` seeds, of the first dataset, whose final phase is stored."""
        ok = [s for s in self.seeds(angle) if (self.dir / "runs" / f"a{angle:g}_s{s:03d}.npz").exists()]
        first = [s for s in ok if s // 100 == ok[0] // 100]
        return first[:n]

    def run(self, angle, seed):
        return np.load(self.dir / "runs" / f"a{angle:g}_s{seed:03d}.npz")

    def marks(self, angle, seed, iteration=None):
        """region -> 'reversed' / 'shifted' / None, at the final or a given iteration."""
        d = self.df if iteration is None else self.drift[self.drift.iteration == iteration]
        return {r.particle: status(r) for r in
                d[(d.angle_mrad == angle) & (d.seed == seed)].itertuples()}


def status(r):
    """A region's flag: reversed if its correlation with the truth is negative, as in the
    paper's panels, else shifted by more than a quarter of the Wigner-Seitz cell.  A
    shift is read off the correlation peak, so it is flagged only where the lattice is
    recovered (|corr| >= 0.5, the tables' threshold).  The paper's panels also flagged
    a competing correlation peak (peak_ratio > 0.7) as ambiguous; that fires on
    correctly placed lattices with few reflections, so it is not used here."""
    if not np.isfinite(r.ws_x):
        return None
    if r.signed_corr < 0:
        return "reversed"
    if r.signed_corr < 0.5:
        return None
    q = fp._default_QF()[r.particle](np.hypot(r.ws_x, r.ws_y))
    return "shifted" if q > 0.25 else None


def phase_panel(ax, sw, ph, marks=None, vlim=None, scalebar=True, count=True, window=None):
    """Band-limited phase with every flagged region outlined, as in the paper's panels.
    `window` (A) restricts the view, e.g. to the scanned area of an enlarged field."""
    x0, x1 = window or (0, sw.L)
    inside = slice(int(x0 / sw.sampling), int(x1 / sw.sampling))
    lo, hi = vlim or np.quantile(ph[inside, inside], (0.01, 0.99))
    ax.imshow(ph.T, cmap="gray", origin="lower", vmin=lo, vmax=hi,
              extent=[0, sw.L, 0, sw.L])
    n = {"reversed": 0, "shifted": 0}
    for name, s in (marks or {}).items():
        if s and name in sw.masks:
            ax.contour(sw.masks[name].T.astype(float), levels=[0.5], colors=[COL[s]],
                       linewidths=0.75, linestyles="--", extent=[0, sw.L, 0, sw.L],
                       origin="lower")
            n[s] += 1
    ax.set(xlim=(x0, x1), ylim=(x0, x1))
    fp._bare(ax)
    if count and marks is not None:
        bits = [(f"{v} {k}", COL[k]) for k, v in n.items() if v] or \
               [(f"0/{len(marks)}", "0.12")]
        for i, (t, c) in enumerate(bits):
            ax.text(0.03, 0.03 + 0.09 * i, t, transform=ax.transAxes, fontsize=6.5,
                    color=c, ha="left", va="bottom")
    if scalebar:
        fp._sbar(ax, 20.0, "20 Å", fs=6.5)


def grid(nrows, ncols, extra_w=0.0, px=PX, gap=0.06, left=0.95, top=0.55):
    """A figure of equal square panels, with `extra_w` inches spare on the right."""
    W = left + ncols * px + (ncols - 1) * gap + extra_w + 0.1
    H = top + nrows * px + (nrows - 1) * gap + 0.12
    fig = plt.figure(figsize=(W, H))
    axs = [[fig.add_axes([(left + j * (px + gap)) / W,
                          1 - (top + (i + 1) * px + i * gap) / H, px / W, px / H])
            for j in range(ncols)] for i in range(nrows)]
    return fig, axs, (W, H, left, top, px, gap)


def row_label(ax, text):
    # multialignment centers each line of a multi-line label, not just the block
    ax.text(-0.06, 0.5, text, transform=ax.transAxes, rotation=90, ha="right",
            va="center", ma="center", fontsize=7.5)


# ---------------------------------------------------------------- figures

def seeded():
    rows = [("init_truth_bare_1e5", "truth\nfree-standing"),
            ("init_reversed_bare_1e5", "reversed truth\nfree-standing"),
            ("init_reversed_carbon28_1e5", "reversed truth\non carbon"),
            ("init_shifted_bare_1e5", "shifted truth\nfree-standing"),
            ("init_shifted_carbon28_1e5", "shifted truth\non carbon")]
    its = (10, 30, 100, 200)
    fig, axs, (W, H, left, top, px, gap) = grid(len(rows), len(its) + 1)
    for i, (name, lab) in enumerate(rows):
        sw = Sweep(f"show_{name}")
        seed = sw.seeds(3.0)[0]
        z = sw.run(3.0, seed)
        snap = dict(zip(z["snapshot_iters"], z["snapshot_phases"].astype(float)))
        vlim = tuple(np.quantile(snap[200], (0.01, 0.99)))
        phase_panel(axs[i][0], sw, sw.truth, scalebar=(i == 0), count=False)
        for j, it in enumerate(its, 1):
            phase_panel(axs[i][j], sw, snap[it], sw.marks(3.0, seed, it), vlim=vlim,
                        scalebar=False)
            if i == 0:
                axs[i][j].set_title(f"iteration {it}", fontsize=8)
        row_label(axs[i][0], lab)
    axs[0][0].set_title("ground truth", fontsize=8)
    save(fig, "rev_seeded")


def recon_rows(rows, angle, title, name, n_seeds=3, window=None, own_scale=()):
    """Rows of conditions, each the ground truth and `n_seeds` reconstructions of one
    dataset, with flagged regions outlined.  rows: (sweep, label[, window]).  Rows whose
    index is in `own_scale` scale each panel on its own rather than the row together."""
    fig, axs, (W, H, left, top, px, gap) = grid(len(rows), n_seeds + 1)
    for i, row in enumerate(rows):
        sw_name, lab = row[:2]
        win = row[2] if len(row) > 2 else window
        sw = Sweep(sw_name)
        seeds = sw.stored(angle, n_seeds)
        phs = [sw.run(angle, sd)["phase"].astype(float) for sd in seeds]
        x0, x1 = win or (0, sw.L)
        cut = slice(int(x0 / sw.sampling), int(x1 / sw.sampling))
        vlim = tuple(np.quantile(np.concatenate([p[cut, cut].ravel() for p in phs]),
                                 (0.01, 0.99)))
        phase_panel(axs[i][0], sw, sw.truth, scalebar=(i == 0), count=False, window=win)
        for j, (sd, ph) in enumerate(zip(seeds, phs), 1):
            phase_panel(axs[i][j], sw, ph, sw.marks(angle, sd),
                        vlim=None if i in own_scale else vlim, scalebar=False, window=win)
        row_label(axs[i][0], lab)
    axs[0][0].set_title("ground truth", fontsize=8)
    for j in range(1, n_seeds + 1):
        axs[0][j].set_title(f"initial guess {j}", fontsize=8)
    save(fig, name)


def sampling():
    scan = (34.0, 102.0)          # the scanned 68 A of the enlarged 136 A fields
    recon_rows([("ss_bare_complex_1e5", "0.29 mrad/px\n(21 px)"),
                ("samp_bin2_bare_1e5", "binned 2×\n(10 px)"),
                ("samp_bin4_bare_1e5", "binned 4×\n(5 px)"),
                ("samp_crop192_bare_1e5", "192-px detector\n(21 px)"),
                ("samp_field136_bare_1e5", "2× finer\n(41 px)", scan),
                ("samp_field136crop192_bare_1e5", "2× finer, 192 px\n(41 px)", scan)],
               3.0, "Detector sampling, free-standing, 3 mrad, 10$^5$ e/Å$^2$ "
                    "(pixels across the bright-field disk)", "rev_sampling")


def scan():
    recon_rows([("ss_bare_complex_1e5", "regular,\nknown probe"),
                ("jit_known_bare_1e5", "jittered,\nknown probe"),
                ("grid_blind_pe_bare_1e5", "regular,\nrefined probe"),
                ("jit_blind_pe_bare_1e5", "jittered,\nrefined probe")],
               3.0, "Scan geometry, free-standing, 3 mrad, 10$^5$ e/Å$^2$ (jitter ±1 Å, "
                    "known; refined probes start 1500 Å defocus, 150 Å astigmatism off)",
               "rev_scan")


def pair():
    """The [111] particle at 6 mrad, in two ordinal series: one [100] neighbor brought
    closer, then the published neighbors added one at a time.  The [111] sits at its
    published position throughout."""
    cols = [("single111_pubpos_bare_1e5", "alone"),
            *[(f"pair_gap{g}_bare_1e5", f"+[100], {g} Å gap") for g in (20, 10, 5)],
            ("pair_pub100_bare_1e5", "+[100]"),
            ("pair_pubdeca_bare_1e5", "+twin"),
            ("ss_bare_complex_1e5", "+[100] +twin")]
    n = 3
    fig, axs, (W, H, left, top, px, gap) = grid(n, len(cols), top=0.75)
    for j, (name, lab) in enumerate(cols):
        sw = Sweep(name)
        # the runs whose [111] lattice is reversed or shifted first, so a column shows its
        # failures where it has any; the count is over all draws and initial guesses
        ok = [sd for sd in sw.seeds(6.0)
              if (sw.dir / "runs" / f"a6_s{sd:03d}.npz").exists()]
        bad = [sd for sd in ok if sw.marks(6.0, sd).get("au_111")]
        allbad = sum(bool(sw.marks(6.0, sd).get("au_111")) for sd in sw.seeds(6.0))
        for i, sd in enumerate((bad + [x for x in ok if x not in bad])[:n]):
            ph = sw.run(6.0, sd)["phase"].astype(float)
            phase_panel(axs[i][j], sw, ph, sw.marks(6.0, sd), scalebar=(i == j == 0))
            axs[i][j].text(0.97, 0.97, f"draw {sd // 100 + 1}, init {sd % 100}",
                           transform=axs[i][j].transAxes, ha="right", va="top",
                           fontsize=6, color="w")
        axs[0][j].set_title(f"{lab}\n[111] wrong in {allbad}/{len(sw.seeds(6.0))}", fontsize=7)
    # the two series, marked above the panels
    for j0, j1, lab in ((0, 3, "one neighbor, closer"),
                        (4, 6, "neighbors at their original positions")):
        xa = (left + j0 * (px + gap)) / W
        xb = (left + j1 * (px + gap) + px) / W
        y = 1 - 0.12 / H
        fig.add_artist(plt.Line2D([xa, xb], [y, y], color="0.4", lw=0.8))
        fig.text((xa + xb) / 2, y + 0.01, lab, ha="center", va="bottom", fontsize=7.5,
                 color="0.3")
    save(fig, "rev_pair")


GS_SHOW = (("gsfull_show_ss_bare_1e5", 2, "au_111"),
           ("gsfull_show_ss_bare_1e5", 2, "decahedron_w0"))
SHIFT_LAB = ("no shift", "shift ½ a$_1$", "shift ½ a$_2$", "shift ½ (a$_1$+a$_2$)")
BLUE = "#1f77b4"
CYAN = "#00c8ff"


def search():
    """The full gauge search (gauge_search_full.py --store), free-standing, 3 mrad, 1e5,
    initial guess 2: one region of the converged reconstruction is edited into each of
    eight states (polarity kept or reversed, times four half-cell shifts) and the full
    reconstruction re-converged from each.  One block per edited region: ground truth and
    the reconstruction before editing (edited region in cyan), then the eight re-converged
    candidates with reversed/shifted regions outlined, each labelled with its final misfit
    relative to the unedited candidate; the lowest misfit is framed in blue."""
    sw = Sweep("grid_blind_pe_bare_1e5")           # the free-standing sample's masks
    P_, gap, gx, left, top, bgap = 1.05, 0.05, 0.30, 0.42, 0.22, 0.30
    W = left + P_ + gx + 4 * P_ + 3 * gap + 0.08
    H = top + len(GS_SHOW) * (2 * P_ + gap) + (len(GS_SHOW) - 1) * bgap + 0.05
    fig = plt.figure(figsize=(W, H))

    def at(col, y):
        x = left + (0 if col == 0 else P_ + gx + (col - 1) * (P_ + gap))
        return fig.add_axes([x / W, 1 - (y + P_) / H, P_ / W, P_ / H])

    def tag(ax, text):
        ax.text(0.97, 0.97, text, transform=ax.transAxes, ha="right", va="top",
                fontsize=6.5, color="w")

    for b, (name, seed, region) in enumerate(GS_SHOW):
        z = np.load(Sweep.where(name) / f"s{seed:03d}_{region}" /
                    f"phases_s{seed:03d}_{region}.npz")
        ph, mis = z["phases"].astype(float), z["misfits"]
        ver = json.loads(str(z["verdicts"]))
        vlim = tuple(np.quantile(ph, (0.01, 0.99)))
        y0 = top + b * (2 * P_ + gap + bgap)
        a = at(0, y0)
        phase_panel(a, sw, z["truth"].astype(float), count=False)
        tag(a, "ground truth")
        a = at(0, y0 + P_ + gap)
        phase_panel(a, sw, ph[0], ver[0], vlim=vlim, scalebar=False)
        a.contour(sw.masks[region].T.astype(float), levels=[0.5], colors=[CYAN],
                  linewidths=1.1, extent=[0, sw.L, 0, sw.L], origin="lower")
        tag(a, "before editing")
        best = 1 + int(np.argmin(mis[1:]))
        for k in range(1, 9):
            p, s = divmod(k - 1, 4)
            a = at(1 + s, y0 + p * (P_ + gap))
            phase_panel(a, sw, ph[k], ver[k], vlim=vlim, scalebar=False)
            tag(a, "unedited" if k == 1 else f"{100 * (mis[k] / mis[1] - 1):+.3f}%")
            if k == best:
                for sp in a.spines.values():
                    sp.set(edgecolor=BLUE, linewidth=2.0, visible=True)
            if p == 0:
                a.set_title(SHIFT_LAB[s], fontsize=7, pad=2.5)
        for p, t in enumerate(("polarity kept", "polarity reversed")):
            fig.text((left + P_ + gx * 0.45) / W, 1 - (y0 + p * (P_ + gap) + P_ / 2) / H,
                     t, rotation=90, ha="center", va="center", fontsize=6.5)
        fig.text(left * 0.4 / W, 1 - (y0 + P_ + gap / 2) / H,
                 f"{fp.REGION_SHORT[region]} edited", rotation=90, ha="center",
                 va="center", fontsize=8)
    save(fig, "rev_search")


def _dewrapped(O, sampling):
    """Scored as the stage-3 thickness sweeps are (run_sweep.py --phase-smooth 4)."""
    return analysis.object_phase(O, sampling, smooth=4.0)


def thickness():
    cols = [(10, "show_thick_ss_c10_1e5"), (28, "show_ss_carbon28_complex_1e5"),
            (50, "show_thick_ss_c50_1e5"), (100, "show_thick_ss_c100_1e5"),
            (200, "show_thick_ss_c200_1e5")]
    fig, axs, _ = grid(2, len(cols))
    for j, (t, name) in enumerate(cols):
        sw = Sweep(name)
        sd = sw.seeds(3.0)[0]
        z = sw.run(3.0, sd)
        obj, _, sampling = sim.projected_object(
            sim.build_sample(support=True, carbon_thickness=t))
        truth2 = _dewrapped(obj, sampling)
        regions = analysis.build_regions(truth2, sampling, sim.PARTICLES)
        ph2 = _dewrapped(z["object"], sampling)
        rows = pd.DataFrame(analysis.summarise(ph2, truth2, sampling, regions))
        rows = rows.assign(angle_mrad=3.0, seed=sd)
        mk2 = {r.particle: status(r) for r in rows.itertuples()}
        c1 = sw.df[(sw.df.seed == sd)].signed_corr.abs().mean()
        c2 = rows.signed_corr.abs().mean()
        print(f"  {t / 10:.1f} nm: mean |corr| as published {c1:.3f}, de-wrapped {c2:.3f}")
        phase_panel(axs[0][j], sw, z["phase"].astype(float), sw.marks(3.0, sd),
                    scalebar=(j == 0))
        phase_panel(axs[1][j], sw, ph2, mk2, scalebar=False)
        axs[0][j].set_title(f"{t / 10:g} nm a-C", fontsize=8)
        for i, c in enumerate((c1, c2)):
            axs[i][j].text(0.97, 0.97, f"|corr| {c:.2f}", transform=axs[i][j].transAxes,
                           ha="right", va="top", fontsize=6.5, color="w")
    row_label(axs[0][0], "phase as\nreconstructed")
    row_label(axs[1][0], "smooth phase\ndivided out")
    save(fig, "rev_thickness")


def _centred(P, half_px):
    P = np.fft.fftshift(P)
    i, j = np.unravel_index(np.argmax(np.abs(P)), P.shape)
    return np.roll(P, (P.shape[0] // 2 - i, P.shape[1] // 2 - j), axis=(0, 1))[
        P.shape[0] // 2 - half_px:P.shape[0] // 2 + half_px,
        P.shape[1] // 2 - half_px:P.shape[1] // 2 + half_px]


def probe():
    """Probes refined in blind runs started from a wrong probe (1500 A defocus and 150 A
    astigmatism error), against the true and the starting probe.  Every probe is moved onto
    the true one by its best sub-pixel translation and scaled by the least-squares complex
    factor first: probe and object may translate together (Eq. 5), and ePIE trades scale
    and global phase between them, so none of these is an error."""
    from pe_results import register
    angle, half, khalf = 3.0, 40, 14      # 40 px = +-8 A at 0.2 A; 14 px = 0.21 1/A
    true = sim.make_probe(angle, sim.ANGLES[angle])
    start = sim.make_probe(angle, sim.ANGLES[angle] + 1500.0, C12=150.0, phi12=np.pi / 6)
    blocks = [("grid_blind_pe_bare_1e5", "regular scan"),
              ("jit_blind_pe_bare_1e5", "jittered scan")]
    runs = []
    for name, lab in blocks:
        sw = Sweep(name)
        runs += [(lab, sd, sw.run(angle, sd)["probe"]) for sd in sw.stored(angle, 2)]
    probes = [("true", true), ("start: 1500 Å defocus,\n150 Å astigmatism", start)] + \
             [(f"{lab}\ninitial guess {sd % 100 + 1}", p) for lab, sd, p in runs]
    n = true.shape[0]

    def crop(a, h):
        a = np.fft.fftshift(a)
        return a[n // 2 - h:n // 2 + h, n // 2 - h:n // 2 + h]

    fig, axs, (W, H, left, top, px, gap) = grid(3, len(probes), px=1.05, extra_w=0.45,
                                                top=0.75)
    amax = np.abs(true).max()
    for j, (lab, P) in enumerate(probes):
        r = register(P, true)
        err = np.abs(r["P"] - true)
        rel = np.linalg.norm(r["P"] - true) / np.linalg.norm(true)
        axs[0][j].imshow(crop(np.abs(r["P"]), half).T, cmap="gray", origin="lower",
                         vmin=0, vmax=amax)
        im_ph = axs[1][j].imshow(crop(r["phase"], khalf).T, cmap="RdBu_r", origin="lower",
                                 vmin=-1.5, vmax=1.5, interpolation="nearest")
        axs[1][j].set_facecolor("0.85")
        axs[1][j].text(0.04, 0.04, f"{r['rms']:.2f} rad rms", transform=axs[1][j].transAxes,
                       fontsize=6.2, ha="left", va="bottom")
        axs[2][j].imshow(crop(err, half).T, cmap="gray", origin="lower", vmin=0,
                         vmax=0.1 * amax)
        axs[2][j].text(0.04, 0.04, f"{100 * rel:.1f}%", transform=axs[2][j].transAxes,
                       fontsize=6.5, color="w", ha="left", va="bottom")
        if j >= 2:
            s = np.hypot(*r["shift_A"])
            axs[0][j].text(0.96, 0.04, f"moved {s:.2f} Å", transform=axs[0][j].transAxes,
                           fontsize=6.2, color="w", ha="right", va="bottom")
        for ax in (axs[0][j], axs[1][j], axs[2][j]):
            fp._bare(ax)
        axs[0][j].set_title(lab, fontsize=6.8)
    row_label(axs[0][0], "amplitude")
    row_label(axs[1][0], "aperture phase\nminus true")
    row_label(axs[2][0], "|probe − true|\n(0–10% of peak)")
    # 0.2 A pixels, so 25 px is 5 A
    for i in (0, 2):
        axs[i][0].add_patch(plt.Rectangle((2 * half - 32, 6), 25, 2.0, color="w"))
        axs[i][0].text(2 * half - 19.5, 9, "5 Å", color="w", ha="center", va="bottom",
                       fontsize=6.5)
    # reciprocal pixel 1/L; a 0.1 1/A bar
    kbar = 0.1 * sim.L
    axs[1][0].add_patch(plt.Rectangle((2 * khalf - 2 - kbar, 1.2), kbar, 0.6, color="k"))
    axs[1][0].text(2 * khalf - 2 - kbar / 2, 2.0, "0.1 Å$^{-1}$", ha="center", va="bottom",
                   fontsize=6.2)
    x0 = (left + len(probes) * (px + gap)) / W
    y = 1 - (top + 2 * px + gap) / H
    cb = fig.colorbar(im_ph, cax=fig.add_axes([x0, y + 0.1 * px / H, 0.08 / W, 0.8 * px / H]))
    cb.ax.tick_params(labelsize=6)
    cb.set_label("rad", fontsize=6.5)
    save(fig, "rev_probe")


def mixed():
    """Single and four-mode mixed-state objects on the same datasets (2.8 nm a-C, R0 film,
    first draw), multislice data with and without thermal motion.  The mixed object's
    first mode is shown (each mode's mean phase set to zero, mixfix runs); its other three
    modes together hold under 2% of the occupancy."""
    recon_rows([("msfp_carbon28_a3_complex_1e5", "single object\nwith phonons"),
                ("mixfix_msfp_1e5", "mixed object\nwith phonons"),
                ("ms_carbon28_a3_complex_1e5", "single object\nstatic"),
                ("mixfix_ms_1e5", "mixed object\nstatic")],
               3.0, "Mixed-state object (four modes, first shown) against a single object, "
                    "multislice data, 3 mrad, 10$^5$ e/Å$^2$", "rev_mixed")


def mismatch():
    """The one non-unanimous dataset (draw 2) of the 12 with a known probe, and the same
    dataset and initial guesses with the probe refined from the wrong start."""
    seeds = (203, 205, 207)
    rows = [("show_mm_msall_known_2e5", "known probe"),
            ("mm_msall_blind_pe_2e5", "refined probe\n(wrong start)")]
    fig, axs, _ = grid(len(rows), 4)
    for i, (name, lab) in enumerate(rows):
        sw = Sweep(name)
        phs = [sw.run(3.0, sd)["phase"].astype(float) for sd in seeds]
        vlim = tuple(np.quantile(np.concatenate([p.ravel() for p in phs]), (0.01, 0.99)))
        phase_panel(axs[i][0], sw, sw.truth, scalebar=(i == 0), count=False)
        for j, (sd, ph) in enumerate(zip(seeds, phs), 1):
            phase_panel(axs[i][j], sw, ph, sw.marks(3.0, sd), vlim=vlim, scalebar=False)
        row_label(axs[i][0], lab)
    axs[0][0].set_title("ground truth", fontsize=8)
    for j in range(1, 4):
        axs[0][j].set_title(f"initial guess {j}", fontsize=8)
    save(fig, "rev_mismatch")


FIGURES = dict(seeded=seeded, pair=pair, sampling=sampling, scan=scan, search=search,
               thickness=thickness, probe=probe,
               mixed=mixed, mismatch=mismatch)

if __name__ == "__main__":
    for name in sys.argv[1:] or FIGURES:
        print(f"=== {name}")
        try:
            FIGURES[name]()
        except FileNotFoundError as e:       # its runs have not finished yet
            print(f"  skipped: {e}")
