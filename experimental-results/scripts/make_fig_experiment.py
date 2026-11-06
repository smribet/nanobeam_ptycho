"""Response figures from the experimental re-reconstructions (e1_export.py output).

    python make_fig_experiment.py RENDER_DIR      # fig3_hi.png and haadf_hi.png in RENDER_DIR

  fov        comment 1.4: the HAADF images of Fig. S1 with the 4D-STEM field of view and the
             two particles of Fig. 3 outlined.  The published 24 mrad panel of Fig. 3 is
             registered onto our 24 mrad reconstruction, and that onto each HAADF image,
             both by normalised cross-correlation of the particle envelopes over scale,
             the eight 90-degree rotations and reflections, and small rotations.
  overlay    comment 1.3: Fig. 3 regenerated from the re-reconstructions, with the two
             seeds shown in magenta and green so that agreement adds to white
  engines    comments 2.4/2.5: agreement between seeds for each engine and convergence
             angle, and between engines at 24 and 3 mrad

RENDER_DIR holds raster renders of figures/fig_non_uniform_3.pdf and
SI_figures/fig_si_HAADF.pdf (4000 px wide), the only form in which the published panels and
the HAADF images exist.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from PIL import Image
from scipy import ndimage
from scipy.signal import fftconvolve

HERE = Path(__file__).resolve().parent
EXPORT = HERE.parent / "outputs"
OUT = HERE.parent / "figures" / "response"
GRID = 1.0                      # A per pixel at which images are matched
ANGLES = {23: 24, 24: 12, 25: 6, 26: 3, 27: 1.5}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8})

# the published Fig. 3 (4000 px render): the four top panels, and the red and blue boxes
# of its first panel as fractions of that panel (x right, y down)
FIG3_PANELS = [(225, 1074), (1175, 2024), (2150, 2999), (3100, 3949)]
FIG3_ROWS = (145, 994)
FIG3_SAMPLING = 100.0 / 354     # A per render pixel, from the 10 nm scale bar
BOXES = {"red": (0.523, 0.380, 0.727, 0.585), "blue": (0.740, 0.484, 0.943, 0.686)}
HAADF_PANELS = [(0, 1972), (2027, 3999)]
HAADF_ROWS = (55, 2027)
HAADF_SAMPLING = 50.0 / 211     # A per render pixel, from the 5 nm scale bar (approximate)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    print(f"  wrote {name}.pdf")


def magma_to_scalar(rgb):
    lut = matplotlib.colormaps["magma"](np.linspace(0, 1, 256))[:, :3]
    flat = rgb.reshape(-1, 3)
    out = np.empty(len(flat))
    for i in range(0, len(flat), 200000):
        d = ((flat[i:i + 200000, None, :] - lut[None]) ** 2).sum(-1)
        out[i:i + 200000] = d.argmin(1) / 255
    return out.reshape(rgb.shape[:2])


def envelope(img, sampling, sigma_A=4.0):
    """Image resampled to GRID A/px and smoothed to the particle shapes."""
    z = ndimage.zoom(img, sampling / GRID, order=1)
    return ndimage.gaussian_filter(z, sigma_A / GRID)


def dihedral(a, k):
    return np.rot90(a.T if k >= 4 else a, k % 4)


def ncc_map(image, tmpl):
    """Normalised cross-correlation of `tmpl` at every position fully inside `image`."""
    t = tmpl - tmpl.mean()
    n = t.size
    ones = np.ones_like(t)
    num = fftconvolve(image, t[::-1, ::-1], mode="valid")
    s1 = fftconvolve(image, ones, mode="valid")
    s2 = fftconvolve(image ** 2, ones, mode="valid")
    den = np.sqrt(np.maximum(s2 - s1 ** 2 / n, 1e-12)) * np.sqrt((t ** 2).sum())
    return num / den


def locate(tmpl, image, scales, rotations=(0.0,), kinds=range(8)):
    """Best (ncc, kind, scale, rotation, (row, col) of the template's corner) of the
    template, transformed, inside `image`, both already on GRID."""
    best = None
    for k in kinds:
        tk = dihedral(tmpl, k)
        for sc in scales:
            ts = ndimage.zoom(tk, sc, order=1)
            for rot in rotations:
                tr = ndimage.rotate(ts, rot, reshape=False, order=1, mode="nearest") if rot else ts
                if tr.shape[0] >= image.shape[0] or tr.shape[1] >= image.shape[1]:
                    continue
                m = ncc_map(image, tr)
                i = np.unravel_index(np.argmax(m), m.shape)
                if best is None or m[i] > best[0]:
                    best = (float(m[i]), k, sc, rot, i, tr.shape)
    return best


def map_points(pts, shape, k, sc, rot, corner):
    """Points (row, col) of the template, on GRID, to the image, after `locate`'s transform."""
    pts = np.asarray(pts, float)
    h, w = shape
    r, c = pts[:, 0], pts[:, 1]
    # the dihedral transform of np.rot90 / .T, on coordinates
    if k >= 4:
        r, c = c, r
        h, w = w, h
    for _ in range(k % 4):              # rot90 counter-clockwise: (r, c) -> (w - 1 - c, r)
        r, c = w - 1 - c, r
        h, w = w, h
    r, c = r * sc, c * sc
    H, W = h * sc, w * sc
    th = np.deg2rad(rot)
    cr, cc = (H - 1) / 2, (W - 1) / 2
    rr = cr + np.cos(th) * (r - cr) + np.sin(th) * (c - cc)
    cc2 = cc - np.sin(th) * (r - cr) + np.cos(th) * (c - cc)
    return np.stack([rr + corner[0], cc2 + corner[1]], 1)


def load_renders(rdir):
    fig3 = np.asarray(Image.open(rdir / "fig3_hi.png").convert("L")).astype(float) / 255
    haadf_rgb = np.asarray(Image.open(rdir / "haadf_hi.png").convert("RGB")).astype(float) / 255
    p3 = fig3[FIG3_ROWS[0]:FIG3_ROWS[1], FIG3_PANELS[0][0]:FIG3_PANELS[0][1]]
    hd = [magma_to_scalar(haadf_rgb[HAADF_ROWS[0]:HAADF_ROWS[1], a:b]) for a, b in HAADF_PANELS]
    return p3, hd


def fov_and_boxes(phases, rdir):
    """Our 24 mrad reconstruction located in Fig. 3 and in both HAADF images."""
    s = float(phases["sampling"])
    ours = phases["scan23_s0"]
    p3, hd = load_renders(rdir)
    e_ours = envelope(ours, s)
    e_p3 = envelope(p3, FIG3_SAMPLING)
    # (1) the published panel within our reconstruction: same field, so match the central
    #     80% of the panel inside ours
    m = int(0.1 * e_p3.shape[0])
    core = e_p3[m:-m, m:-m]
    b3 = locate(core, e_ours, scales=np.linspace(0.94, 1.06, 13))
    print(f"  Fig. 3 panel in our reconstruction: ncc {b3[0]:.2f}, orientation {b3[1]}, "
          f"scale {b3[2]:.3f}")
    # the boxes, as points of the panel on GRID, then in our reconstruction (GRID)
    n3 = e_p3.shape[0]
    boxes = {}
    for name, (x0, y0, x1, y1) in BOXES.items():
        corners = np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]]) * n3 - m
        boxes[name] = map_points(corners, core.shape, *b3[1:4], b3[4])
    # (2) our reconstruction within each HAADF image
    out = []
    for j, h in enumerate(hd):
        e_h = envelope(h, HAADF_SAMPLING)
        bh = locate(e_ours, e_h, scales=np.linspace(0.8, 1.2, 21),
                    rotations=np.arange(-4, 4.1, 1.0))
        n = e_ours.shape[0]
        fov = map_points([[0, 0], [0, n - 1], [n - 1, n - 1], [n - 1, 0]], e_ours.shape,
                         *bh[1:4], bh[4])
        bx = {k: map_points(v, e_ours.shape, *bh[1:4], bh[4]) for k, v in boxes.items()}
        print(f"  {'before' if j == 0 else 'after'} HAADF: ncc {bh[0]:.2f}, orientation "
              f"{bh[1]}, scale {bh[2]:.3f} (HAADF calibration x{1 / bh[2]:.3f}), "
              f"rotation {bh[3]:+.0f} deg")
        out.append((h, e_h.shape, fov, bx, bh))
    return out, boxes, b3


def fig_fov(phases, rdir):
    res, _, _ = fov_and_boxes(phases, rdir)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.7))
    for ax, (h, shape, fov, bx, bh), lab in zip(axs, res, ("before", "after")):
        z = GRID / HAADF_SAMPLING        # GRID pixels -> render pixels
        ax.imshow(h, cmap="magma", vmin=np.quantile(h, 0.01), vmax=np.quantile(h, 0.995))
        ax.add_patch(Polygon(fov[:, ::-1] * z, closed=True, fill=False, ec="w", lw=1.3, ls="--"))
        for name, col in (("red", "#ff3b30"), ("blue", "#4fb3ff")):
            ax.add_patch(Polygon(bx[name][:, ::-1] * z, closed=True, fill=False, ec=col, lw=1.3))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{lab} 4D-STEM acquisition")
        print(f"  {lab}: envelope match {bh[0]:.2f}")
    fig.tight_layout()
    save(fig, "fig_resp_haadf_fov")


def scalebar(ax, sampling, length_A, label, color="w"):
    """A bar of `length_A` in the lower right of an image axis whose pixels are `sampling` A."""
    x0, x1 = sorted(ax.get_xlim()); y0, y1 = sorted(ax.get_ylim())
    w, h = x1 - x0, y1 - y0
    n = length_A / sampling
    xs, ys = x1 - 0.06 * w - n, y1 - 0.07 * h       # images are drawn with y down
    ax.add_patch(plt.Rectangle((xs, ys), n, 0.025 * h, color=color, zorder=5))
    ax.text(xs + n / 2, ys - 0.015 * h, label, color=color, ha="center", va="bottom",
            fontsize=6.5, zorder=5)


def lattice(ph, s, lo=0.05, hi=0.8):
    fr, fc = np.fft.fftfreq(ph.shape[0], s), np.fft.fftfreq(ph.shape[1], s)
    q = np.hypot(*np.meshgrid(fr, fc, indexing="ij"))
    return np.real(np.fft.ifft2(np.fft.fft2(ph) * ((q > lo) & (q < hi))))


def composite(a, b, s, lo_q=0.05):
    """Magenta (a) + green (b) composite of two phases' lattice bands: where they agree the
    colors add to white; magenta or green alone marks a feature of one only."""
    ch = []
    for x in (a, b):
        x = lattice(x, s, lo=lo_q)
        lo, hi = np.quantile(x, (0.02, 0.98))
        ch.append(np.clip((x - lo) / (hi - lo), 0, 1))
    return np.stack([ch[0], ch[1], ch[0]], -1)


# the two Fig. 3 particles inside the crop window of e1_crops.py (rows, cols)
CROP_BOX = {"red": (slice(20, 246), slice(20, 245)), "blue": (slice(135, 357), slice(259, 483))}


def fig_overlay(phases, rdir):
    """Fig. 3 from the re-reconstructions: seeds 0 and 1 at 24 and 1.5 mrad, each scan in
    its own frame with the boxes moved by its drift, and the two boxed particles as
    magenta (seed 0) + green (seed 1) composites."""
    from register_lattice import align, envelope as env_fn
    s = float(phases["sampling"])
    _, boxes, _ = fov_and_boxes(phases, rdir)
    crops = np.load(EXPORT / "e1_crops.npz")
    ref_env = env_fn(phases["scan23_s0"], s)
    fig = plt.figure(figsize=(7.2, 3.6))
    gs = fig.add_gridspec(2, 4, hspace=0.2, wspace=0.06)
    letters = iter("abcdefgh")

    def letter(ax):
        ax.text(0.0, 1.02, next(letters), transform=ax.transAxes, ha="left", va="bottom",
                fontsize=10, weight="bold")
    col = 0
    for scan in (23, 27):
        n = phases[f"scan{scan}_s0"].shape[0]
        k = np.array([(v + n // 2) % n - n // 2
                      for v in align(phases[f"scan{scan}_s0"], ref_env, s)[1]])
        for sd in (0, 1):
            ax = fig.add_subplot(gs[0, col])
            ph = ndimage.gaussian_filter(phases[f"scan{scan}_s{sd}"], 0.6)
            ax.imshow(ph, cmap="gray", vmin=np.quantile(ph, 0.01), vmax=np.quantile(ph, 0.99))
            for name, c in (("red", "#ff3b30"), ("blue", "#4fb3ff")):
                ax.add_patch(Polygon(boxes[name][:, ::-1] * GRID / s - k[::-1], closed=True,
                                     fill=False, ec=c, lw=1.0))
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                scalebar(ax, s, 100.0, "10 nm")
            ax.set_title(f"{ANGLES[scan]:g} mrad, seed {sd}")
            letter(ax)
            col += 1
    col = 0
    for scan in (23, 27):
        for name, c in (("red", "#ff3b30"), ("blue", "#4fb3ff")):
            ax = fig.add_subplot(gs[1, col])
            r, cc = CROP_BOX[name]
            ax.imshow(composite(crops[f"quantEM_{scan}_s0"], crops[f"quantEM_{scan}_s1"], s)[r, cc])
            for sp in ax.spines.values():
                sp.set_color(c); sp.set_linewidth(2)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                scalebar(ax, s, 20.0, "2 nm")
            letter(ax)
            col += 1
    save(fig, "fig_resp_fig3_overlay")


def fig_engines():
    """The two Fig. 3 particles from every engine and angle, as seed 0 (magenta) + seed 1
    (green) composites, and quantEM's seed 0 against PtyRAD's, all in quantEM's frame."""
    z = np.load(EXPORT / "e1_crops.npz")
    s = float(z["sampling"])
    cols = [("quantEM", "quantEM_{}_s0", "quantEM_{}_s1"), ("ePIE", "ePIE_{}_s0", "ePIE_{}_s1"),
            ("PtyRAD", "PtyRAD_{}_s0", "PtyRAD_{}_s1"),
            ("quantEM vs PtyRAD", "quantEM_{}_s0", "PtyRAD_{}_s0")]
    scans = (23, 24, 25, 26, 27)
    r, c = CROP_BOX["blue"]
    pw = 1.45
    fig, axs = plt.subplots(len(scans), len(cols),
                            figsize=(len(cols) * pw + 0.5, len(scans) * pw + 0.5))
    for i, scan in enumerate(scans):
        for j, (lab, a, b) in enumerate(cols):
            ax = axs[i, j]
            # above 0.15 1/A: the lattice and the particle edge, without ePIE's slow stripes
            ax.imshow(composite(z[a.format(scan)][r, c], z[b.format(scan)][r, c], s, lo_q=0.15))
            ax.set_xticks([]); ax.set_yticks([])
            if i == j == 0:
                scalebar(ax, s, 20.0, "2 nm")
            if i == 0:
                ax.set_title(lab if j == 3 else f"{lab}, seeds 0 + 1", fontsize=7.5)
            if j == 0:
                ax.set_ylabel(f"{ANGLES[scan]:g} mrad", fontsize=8)
    fig.text(0.5, 0.01, "magenta: first reconstruction; green: second; white where they agree",
             ha="center", fontsize=7)
    fig.tight_layout(h_pad=0.3, w_pad=0.3, rect=(0, 0.02, 1, 1))
    save(fig, "fig_resp_engines")


def fig_unbinned():
    """quantEM on the central 256 x 256 positions, 200 iterations, binned 2x as published
    and as recorded: seeds 0 (magenta) + 1 (green) of the red particle of Fig. 3, in the
    frame of quantEM's full-field reconstruction of the same scan."""
    z = np.load(EXPORT / "e1_qcrop.npz")
    s = float(z["sampling"])
    r, c = CROP_BOX["red"]
    fig, axs = plt.subplots(2, 2, figsize=(3.6, 3.75))
    for i, scan in enumerate((26, 25)):
        for j, (b, lab) in enumerate(((2, "binned 2× (96 px)"), (1, "unbinned (192 px)"))):
            ax = axs[i, j]
            a0 = z[f"qcrop_bin{b}_{scan}_s0"].astype(np.float32)
            a1 = z[f"qcrop_bin{b}_{scan}_s1"].astype(np.float32)
            ax.imshow(composite(a0, a1, s)[r, c])
            ax.set_xticks([]); ax.set_yticks([])
            if i == j == 0:
                scalebar(ax, s, 20.0, "2 nm")
            if i == 0:
                ax.set_title(lab, fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{ANGLES[scan]:g} mrad", fontsize=8)
    fig.tight_layout(h_pad=0.4, w_pad=0.4)
    save(fig, "fig_resp_unbinned")


def fig_fixed():
    """quantEM with the probe refined (seeds 0 + 1) and with a converged probe frozen
    (two new seeds), the blue particle of Fig. 3, in quantEM's frame."""
    z = np.load(EXPORT / "e1_crops.npz")
    zf = np.load(EXPORT / "e1_fixed_crops.npz")
    s = float(z["sampling"])
    r, c = CROP_BOX["blue"]
    fig, axs = plt.subplots(2, 3, figsize=(4.6, 3.3))
    for j, scan in enumerate((25, 26, 27)):
        fx = sorted(k for k in zf.files if k.startswith(f"fixed_{scan}_"))
        for i, (a_, b_, lab) in enumerate(((z[f"quantEM_{scan}_s0"], z[f"quantEM_{scan}_s1"],
                                            "probe refined"),
                                           (zf[fx[0]], zf[fx[1]], "probe frozen"))):
            ax = axs[i, j]
            ax.imshow(composite(a_[r, c], b_[r, c], s, lo_q=0.15))
            ax.set_xticks([]); ax.set_yticks([])
            if i == j == 0:
                scalebar(ax, s, 20.0, "2 nm")
            if i == 0:
                ax.set_title(f"{ANGLES[scan]:g} mrad", fontsize=8)
            if j == 0:
                ax.set_ylabel(lab, fontsize=8)
    fig.tight_layout(h_pad=0.4, w_pad=0.4)
    save(fig, "fig_resp_fixed_probe")


if __name__ == "__main__":
    rdir = Path(sys.argv[1])
    phases = np.load(EXPORT / "e1_phases.npz")
    which = sys.argv[2:] or ["engines", "fov", "overlay", "unbinned"]
    if "unbinned" in which:
        fig_unbinned()
    if "fixed" in which:
        fig_fixed()
    if "engines" in which:
        fig_engines()
    if "fov" in which:
        fig_fov(phases, rdir)
    if "overlay" in which:
        fig_overlay(phases, rdir)
