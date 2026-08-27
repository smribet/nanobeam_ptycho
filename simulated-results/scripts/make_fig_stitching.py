"""Nanodiffraction stitching as a phase-free negative control.

    python make_fig_stitching.py            # writes figures/fig_stitching.png

Below the disk-overlap threshold a diffraction pattern is a list of reflection positions
and intensities, and nothing in it depends on the relative phase of two reflections.  This
builds a real-space map from exactly that -- never a retrieved phase -- and shows it
failing in the same two ways the ePIE sweeps do: a rigid sub-cell translation, and a
contrast reversal.  That is what makes those failures *predictable* rather than incidental.

Conventions follow the other figures: free-standing particles, 3 mrad, 10^5 e/A^2, and the
ePIE panel is not re-run here but read straight out of the `ss_bare_complex_1e5` sweep, so it
is literally one of the reconstructions counted in the substrate control figure.
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Wedge

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import paths
import abtem
import py4DSTEM
import sim
import analysis
import figpanels as fp

abtem.config.set({"dask.lazy": False})

ANGLE, DEFOCUS = 3.0, 0.0
DOSE, NOISE_SEED = 1e5, 0
SWEEP, EPIE_SEED = paths.REFERENCE_BARE, 0
TOL = 0.04                       # reflection-family clustering, 1/A
OUT = paths.figures()

TINT = ["#ffd23f", "#4fc3f7", "#ff6ec7", "#8ddc6b"]
RED = "#ff2d2d"
LABELS = [r"$[111]$ particle", r"$[100]$ particle", "twin boundary", "vacuum"]


def main():
    lam = abtem.core.energy.energy2wavelength(sim.ENERGY)
    atoms = sim.build_sample(support=False)

    # the sweep's own reference, so `truth` and the stored ePIE phase are the same objects
    # the substrate figure counts -- not a re-derivation that might differ in band limit
    ref = np.load(paths.sweep(SWEEP) / "runs" / "_reference.npz", allow_pickle=True)
    truth, s = ref["truth"].astype(float), float(ref["sampling"])
    regions = analysis.build_regions(truth, s, sim.PARTICLES)

    pot = abtem.Potential(atoms, gpts=(sim.GPTS, sim.GPTS),
                          slice_thickness=float(atoms.cell[2, 2])).build()
    probe = abtem.Probe(energy=sim.ENERGY, semiangle_cutoff=ANGLE).match_grid(pot)
    pint = np.squeeze(probe.build(abtem.CustomScan([[sim.L / 2, sim.L / 2]]))
                      .intensity().array)
    pint /= pint.max()
    c0 = sim.GPTS // 2
    xs = np.arange(sim.GPTS) * s
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    print(f"{len(atoms)} atoms (free-standing), {sim.L:.0f} A field, {sim.GPTS}^2 grid")

    # ---- matched filter: a tapered probe disk MINUS a surrounding annulus ----
    # A template that is merely zero-mean over the whole pattern scores the weak outer
    # disks NEGATIVE -- its moat lands on the much brighter inner reflections -- and the
    # intensity threshold then discards them, losing {220} entirely.
    vac = probe.build(abtem.CustomScan([[0., 0.]])).diffraction_patterns(max_angle="valid")[0]
    tm = vac.array / vac.array.max()
    ny, nx = tm.shape
    qy, qx = np.meshgrid(np.fft.fftfreq(ny) * ny, np.fft.fftfreq(nx) * nx, indexing="ij")
    qr = np.hypot(qy, qx)
    RAD = np.sqrt((tm > 0.5).sum() / np.pi)
    t = np.clip(1 - (qr - 0.85 * RAD) / (0.3 * RAD), 0, 1)
    core = t ** 2 * (3 - 2 * t)
    ann = ((qr > 1.15 * RAD) & (qr < 1.45 * RAD)).astype(float)
    tFT = np.conj(np.fft.fft2(np.fft.fftshift(core / core.sum() - ann / ann.sum())))
    ct = (np.array(vac.shape) + 0.5) // 2
    dk = np.array(vac.sampling)
    MAXA = vac.shape[0] / 2 * vac.angular_sampling[0]
    rng = np.random.default_rng(NOISE_SEED)
    n_e = DOSE * sim.STEP ** 2
    print(f"DP {vac.shape}, +/-{MAXA:.0f} mrad, disk radius {RAD:.1f} px, "
          f"{n_e:.0f} e/position")

    def measure(pos):
        """One diffraction pattern at this dose, and the reflections detected in it."""
        dp = probe.multislice(pot, abtem.CustomScan([list(pos)]))[0] \
            .diffraction_patterns(max_angle="valid").array
        tot = dp.sum()
        dp = rng.poisson(n_e * dp / tot).astype(np.float64) / n_e * tot
        cc = np.fft.fftshift(np.real(np.fft.ifft2(np.fft.fft2(dp) * tFT)))
        pk = py4DSTEM.process.utils.get_maxima_2D(
            cc, subpixel="multicorr", upsample_factor=16, minRelativeIntensity=5e-3,
            minSpacing=8, edgeBoundary=2, maxNumPeaks=60)
        g = np.stack([(pk["x"] - ct[0]) * dk[0], (pk["y"] - ct[1]) * dk[1]], 1)
        k = np.hypot(g[:, 0], g[:, 1]) > 0.1
        return dp, g[k], pk["intensity"][k]

    def n_families(g, tol=0.06):
        """Distinct reflections here, treating +g and -g as one.

        Counting spots // 2 silently loses a reflection whose Friedel mate is missing, and
        at a twin boundary one is: the probe illuminates two crystals at once, so the
        scattering volume is not centrosymmetric and need not obey Friedel's law.
        """
        fs = []
        for v in g:
            if not any(min(np.hypot(*(v - f)), np.hypot(*(v + f))) < tol for f in fs):
                fs.append(v)
        return len(fs)

    # Subpixel detection returns a slightly different g every position, so raw g values
    # never key the same twice.  Clustering into families is what makes the synthesis
    # stable; the running centroid is the family's reciprocal vector.
    fams = []

    def assign(g):
        best, bd, bg = None, TOL, None
        for f in fams:
            for sg in (1., -1.):                 # +g and -g are the same family
                d = np.hypot(*(f["g"] - sg * g))
                if d < bd:
                    best, bd, bg = f, d, sg * g
        if best is None:
            best = {"g": g.copy(), "n": 0}
            fams.append(best)
            bg = g
        best["g"] = (best["g"] * best["n"] + bg) / (best["n"] + 1)
        best["n"] += 1
        return best

    grid = np.arange(sim.STEP / 2, sim.L, sim.STEP)
    positions = [(px, py) for px in grid for py in grid]
    records, informative = [], 0
    t0 = time.time()
    for i, (px, py) in enumerate(positions):
        _, g, I = measure((px, py))
        informative += len(g) > 0
        grp = {}
        for gv, Ig in zip(g, I):
            f = assign(gv)
            grp.setdefault(id(f), {"f": f, "I": []})["I"].append(Ig)
        records.append(((px, py), [(e["f"], float(np.sqrt(np.mean(e["I"]))))
                                   for e in grp.values()]))
        if i and i % 250 == 0:
            print(f"   {i}/{len(positions)} positions ({time.time()-t0:.0f}s)")
    print(f"scan: {len(positions)} positions ({informative} informative) in "
          f"{time.time()-t0:.0f}s, {len([f for f in fams if f['n'] >= 5])} families")

    def synth(delta=(0., 0.), flip=False):
        """Real-space map from reflection positions and intensities only.

        Every phase is a free parameter the measurement never determined; `delta` and
        `flip` are the two operations that leave every recorded intensity untouched.
        """
        C = np.zeros((sim.GPTS, sim.GPTS))
        W = np.zeros((sim.GPTS, sim.GPTS))
        for (px, py), contribs in records:
            env = np.roll(pint, (int(round(px / s)) - c0, int(round(py / s)) - c0),
                          axis=(0, 1))
            i0, i1 = max(0, int((px - 14) / s)), min(sim.GPTS, int((px + 14) / s))
            j0, j1 = max(0, int((py - 14) / s)), min(sim.GPTS, int((py + 14) / s))
            sl = (slice(i0, i1), slice(j0, j1))
            Xw, Yw, ew = X[sl], Y[sl], env[sl]
            for fam, amp in contribs:
                gx, gy = fam["g"]
                ph = 2 * np.pi * (gx * delta[0] + gy * delta[1]) + (np.pi if flip else 0.)
                C[sl] += ew * amp * np.cos(2 * np.pi * (gx * Xw + gy * Yw) + ph)
            W[sl] += ew
        return C / np.maximum(W, 1e-9)

    # ---- the four probe positions ----
    CEN = sim.PARTICLES
    theta0 = analysis.find_wedge_offset(truth, s, CEN["decahedron"], 11.0, 5)
    bnd = np.radians(theta0 + 72.0)              # a twin plane, found by lattice sharpness
    PROBE_POS = [tuple(CEN["au_111"]), tuple(CEN["au_100"]),
                 (CEN["decahedron"][0] + 7 * np.cos(bnd),
                  CEN["decahedron"][1] + 7 * np.sin(bnd)),
                 (10.0, 12.0)]
    dps, gs = [], []
    for p in PROBE_POS:
        dp, g, _ = measure(p)
        dps.append(dp)
        gs.append(g)
        print(f"  probe at ({p[0]:5.1f}, {p[1]:5.1f}): {len(g):2d} spots, "
              f"{n_families(g)} distinct reflections")

    # ---- the ePIE panel: read from the sweep, not re-run ----
    run = np.load(paths.sweep(SWEEP) / "runs" / f"a{ANGLE:g}_s{EPIE_SEED:03d}.npz",
                  allow_pickle=True)
    epie_phase = run["phase"].astype(float)
    n_iter = int(run["n_iter"])

    G111 = [r["G"] for r in regions if r["name"] == "au_111"][0]
    DELTA = tuple(np.linalg.inv(G111).T[0] / 2)  # half a lattice repeat of [111]
    stitch, stitch_shift, stitch_flip = synth(), synth(delta=DELTA), synth(flip=True)

    def corr(a, b):
        A, B = a - a.mean(), b - b.mean()
        xc = np.real(np.fft.ifft2(np.fft.fft2(A) * np.conj(np.fft.fft2(B))))
        return float(xc.max() / np.sqrt(np.sum(A ** 2) * np.sum(B ** 2)))

    stitch_rows = analysis.summarise(stitch, truth, s, regions)
    epie_rows = analysis.summarise(epie_phase, truth, s, regions)
    print(f"\nglobal corr vs truth: ePIE {corr(epie_phase, truth):.3f}, "
          f"stitch {corr(stitch, truth):.3f}")
    print(f"   {'region':>14s}  {'stitch':>7s} {'ratio':>6s}   {'ePIE':>7s} {'ratio':>6s}")
    for a, b in zip(stitch_rows, epie_rows):
        print(f"   {fp.REGION_SHORT[a['particle']]:>14s}  {a['signed_corr']:+7.3f} "
              f"{a['peak_ratio']:6.3f}   {b['signed_corr']:+7.3f} {b['peak_ratio']:6.3f}")

    # ------------------------------------------------------------------ figure
    RPROBE = 1.22 * lam / (ANGLE * 1e-3) / 2     # Airy first-zero radius, ~4 A
    FULL = [0, sim.L, 0, sim.L]

    def show(ax, a, extent, gamma=None, clip=(0.01, 0.99), vlim=None):
        """`clip` puts the display range on the structure rather than on edge ringing."""
        if gamma is not None:
            ax.imshow(a.T ** gamma, cmap="gray", origin="lower", extent=extent)
        else:
            if vlim is not None:
                lo, hi = vlim
            else:
                lo, hi = (np.quantile(a, clip) if clip
                          else (-np.abs(a).max(), np.abs(a).max()))
            ax.imshow(a.T, cmap="gray", origin="lower", vmin=lo, vmax=hi, extent=extent)
        ax.set_xticks([]), ax.set_yticks([])

    # Geometry in inches rather than tight_layout, so the panels come out the same size
    # and the whitespace matches the other figures instead of being padded out to clear
    # the two-line titles.
    PS, GI_, TG_ = 1.36, 0.05, 0.30          # panel, gap, title strip
    LEFT, RIGHT, TOP_, BOT_, RG = 0.14, 0.06, 0.02, 0.06, 0.08
    fig_w = LEFT + 5 * PS + 4 * GI_ + RIGHT
    fig_h = TOP_ + 2 * (TG_ + PS) + RG + BOT_
    fig = plt.figure(figsize=(fig_w, fig_h))
    axs = [[None] * 5 for _ in range(2)]
    for r in range(2):
        for c in range(5):
            x = LEFT + c * (PS + GI_)
            y = TOP_ + TG_ + r * (PS + RG + TG_)
            axs[r][c] = fig.add_axes([x / fig_w, 1 - (y + PS) / fig_h,
                                      PS / fig_w, PS / fig_h])

    ax = axs[0][0]
    show(ax, truth, FULL, clip=None)
    for k, p in enumerate(PROBE_POS):
        ax.add_patch(Circle(p, RPROBE, fill=False, color=TINT[k], lw=1.0))
        ax.text(p[0], p[1] + RPROBE + 1.2, "1234"[k], color=TINT[k], ha="center",
                va="bottom", fontsize=7, fontweight="bold")
    fp._sbar(ax, 10.0, "10 Å", fs=5.0)
    ax.set_title("ground truth, band limited\nfree-standing Au particles",
                 fontsize=6.2)

    for k in range(4):
        ax = axs[0][k + 1]
        show(ax, dps[k], [-MAXA, MAXA, -MAXA, MAXA], gamma=0.22)
        for gv in gs[k]:
            ax.add_patch(Circle((gv[0] * lam * 1e3, gv[1] * lam * 1e3), ANGLE,
                                fill=False, color=TINT[k], lw=0.8))
        for sp in ax.spines.values():
            sp.set_color(TINT[k])
            sp.set_linewidth(1.1)
        if k == 0:                       # the four patterns share one angular scale
            fp._sbar(ax, 10.0, "10 mrad", fs=5.0)
        ax.set_title(f"position {k+1}: {LABELS[k]}\n"
                     f"{n_families(gs[k])} reflections detected", fontsize=6.2)

    for k, (a, title) in enumerate([
            (epie_phase, f"iterative ptychography\n{n_iter} ePIE iterations, "
                         f"init {EPIE_SEED}"),
            (stitch, "phase-free stitch\n$\\phi_g = 0$")]):
        ax = axs[1][k]
        show(ax, a, FULL)               # same field of view as a, which carries the bar
        ax.set_title(title, fontsize=6.2)

    # Mark what each method got wrong.  Both panels are annotated, because the point of
    # the figure is that they fail the same WAY, not in the same PLACE: which regions come
    # out inverted is arbitrary, and differs between the two.  A reversal is also not
    # obvious by eye against a neighbour of a different zone axis, and one wedge inverted
    # against its neighbours is the degeneracy acting *inside* a single particle.
    th0 = [r["wedge_theta0"] for r in regions if r["name"].startswith("decahedron")][0]
    dc = sim.PARTICLES["decahedron"]

    def mark_reversed(ax, rows):
        rev = {r["particle"] for r in rows if r["signed_corr"] < 0}
        for name, cen in (("au_111", PROBE_POS[0]), ("au_100", PROBE_POS[1])):
            if name in rev:
                ax.add_patch(Circle(cen, 13.5, fill=False, color=RED, lw=0.8, ls="--"))
        wedges = sorted(n for n in rev if n.startswith("decahedron"))
        for name in wedges:
            k = int(name.rsplit("w", 1)[1])
            ax.add_patch(Wedge(dc, 11.0, th0 + 72 * k, th0 + 72 * (k + 1), fill=False,
                               color=RED, lw=1.4, ls="--"))
        note = [f"{len(rev) - len(wedges)}/2 crystals" if rev - set(wedges) else "",
                f"{len(wedges)}/5 wedges" if wedges else ""]
        # lower left: the only corner that is plain background in both panels, so the
        # label reads without an outline behind it (the scalebar holds the lower right)
        # lower left, stacked narrow: the only corner that is plain background in both
        # panels and clear of the twin, so the label reads without an outline behind it
        # (the scalebar holds the lower right)
        ax.text(0.035, 0.04, "\n".join(["reversed:"] + [x for x in note if x]),
                transform=ax.transAxes, color=RED, fontsize=5.0, ha="left",
                va="bottom", linespacing=1.35)

    mark_reversed(axs[1][0], epie_rows)
    mark_reversed(axs[1][1], stitch_rows)

    ZC, ZH = PROBE_POS[0], 7.5
    cols = np.unique(np.round(atoms.positions[:, :2] / 0.35).astype(int), axis=0) * 0.35
    cols = cols[(np.abs(cols[:, 0] - ZC[0]) < ZH + 1)
                & (np.abs(cols[:, 1] - ZC[1]) < ZH + 1)]
    ext = [ZC[0] - ZH, ZC[0] + ZH, ZC[1] - ZH, ZC[1] + ZH]
    sl = (slice(int((ZC[0] - ZH) / s), int((ZC[0] + ZH) / s)),
          slice(int((ZC[1] - ZH) / s), int((ZC[1] + ZH) / s)))
    axs[1][1].add_patch(Rectangle((ext[0], ext[2]), 2 * ZH, 2 * ZH, fill=False,
                                  color=TINT[0], lw=0.9))
    axs[1][1].text(ext[0] + ZH, ext[3] + 1.0, "h-j", color=TINT[0], fontsize=5.5,
                   ha="center", va="bottom", fontweight="bold")

    zvlim = tuple(np.quantile(np.concatenate([z[sl].ravel() for z in
                                              (stitch, stitch_shift, stitch_flip)]),
                              (0.01, 0.99)))
    for k, (a, title) in enumerate([
            (stitch, "zoomed in region of g\nreference for i and j"),
            (stitch_shift, "$\\phi_g \\rightarrow \\phi_g + 2\\pi\\,g\\cdot\\delta$\n"
                           f"slides by ({DELTA[0]:.2f}, {DELTA[1]:.2f}) Å"),
            (stitch_flip, "$\\phi_g \\rightarrow \\phi_g + \\pi$\ncontrast reverses")]):
        ax = axs[1][k + 2]
        show(ax, a[sl], ext, vlim=zvlim)
        ax.scatter(cols[:, 0], cols[:, 1], s=7, facecolors="none", edgecolors=RED,
                   linewidths=0.5)
        ax.set(xlim=ext[:2], ylim=ext[2:])
        for sp in ax.spines.values():                # match the box drawn in g
            sp.set_color(TINT[0]), sp.set_linewidth(1.1)
        if k == 0:                       # the three zooms share one scale
            fp._sbar(ax, 5.0, "5 Å", fs=5.0)
        ax.set_title(title, fontsize=6.2)

    for r in range(2):
        for c in range(5):
            dark = (r == 0 and c > 0)          # the diffraction patterns
            axs[r][c].text(0.035, 0.965, "abcdefghij"[r * 5 + c],
                           transform=axs[r][c].transAxes, ha="left", va="top",
                           fontsize=8, weight="bold",
                           color="w" if dark else "#1a1a19", zorder=6)

    OUT.mkdir(exist_ok=True)
    p = OUT / "fig_stitching.png"
    for ext in (".png", ".pdf"):
        fig.savefig(p.with_suffix(ext), dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {p} ({p.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
