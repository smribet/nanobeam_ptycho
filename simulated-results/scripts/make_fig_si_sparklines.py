"""Signed-correlation sparklines as a standalone small-multiples figure.

    python make_fig_si_sparklines.py

Every run, every checkpoint, every region, for three conditions: 7 regions across, 40 runs
down, grouped into four blocks of ten by Poisson draw.  Rows inside a block inverted the
SAME data set and differ only in their initial guess, so disagreement within a block is
non-uniqueness; a block that differs from its neighbour is only a different noise draw.

Two encodings, deliberately redundant, because either alone was hard to read at panel size:

  height    |signed correlation| -- how good the lattice is
  direction up = correct polarity, down = reversed

Colour repeats the direction rather than carrying magnitude.  A continuous diverging map
was tried and rejected: bar height is already the magnitude channel, so mapping it to
colour as well washes low-|corr| bars toward the midpoint colour and they vanish against
the page.  cmasher's dark-centred maps (redshift, guppy) invert the problem rather than
solving it -- at our data range |corr| ~ 0.8 lands on their palest steps while "no lattice"
renders black, so the strongest signal looks weakest.  Two saturated hues keep short bars
visible.  The pair is validated (worst adjacent dE 21.1 protan on a white print surface).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import paths
import figpanels as fp

OUT = paths.figures()
ANGLE = 3.0
POS, NEG = "#2166ac", "#b2182b"
INK, MUTED, RULE = "#1a1a19", "#6b6b68", "#d9d9d5"

CONDITIONS = [
    ("complex object, $10^5$ e/Å$^2$", "rev/ms_carbon28_a3_complex_1e5"),
    ("complex object, $10^4$ e/Å$^2$", "rev/ms_carbon28_a3_complex_1e4"),
    ("pure-phase object, $10^4$ e/Å$^2$", "rev/ms_carbon28_a3_phase_1e4"),
]
LABELS = ["[111]", "[100]"] + [f"wedge {i}" for i in range(1, 6)]

NB, RGAP, RPITCH, BGAP = 10, 2.6, 11.0, 1.9      # data units


def draw_block(ax, ctx, angle=ANGLE, brackets=False):
    """One condition: regions across, runs down, blocked by Poisson draw."""
    d = ctx["drift"]
    d = d[d.angle_mrad == angle]
    xs, hs, bs, cs = [], [], [], []
    row = 0
    for bi, ns in enumerate(sorted(d.noise_seed.unique())):
        seeds = sorted(d[d.noise_seed == ns].seed.unique())
        for sd in seeds:
            y0 = -(row * RGAP + bi * BGAP)
            for ci, reg in enumerate(fp.COLS):
                g = d[(d.seed == sd) & (d.particle == reg)].sort_values("iteration")
                if not len(g):
                    continue
                v = g.signed_corr.to_numpy()
                x0 = ci * RPITCH
                xs.extend(x0 + np.arange(len(v)))
                hs.extend(v), bs.extend([y0] * len(v))
                cs.extend([POS if t >= 0 else NEG for t in v])
                ax.plot([x0 - 0.5, x0 + NB - 0.5], [y0, y0], color=RULE, lw=0.3, zorder=0)
            row += 1
        if brackets:
            top = -(row - len(seeds)) * RGAP - bi * BGAP + 1.05
            bot = -(row - 1) * RGAP - bi * BGAP - 1.05
            ax.plot([-3.4, -3.4], [bot, top], color="#9a9a97", lw=1.0, clip_on=False)
            ax.text(-7.2, 0.5 * (top + bot), f"Poisson draw {bi + 1}", rotation=90,
                    ha="center", va="center", fontsize=7, color=MUTED, clip_on=False)
    for b in ax.bar(xs, hs, bottom=bs, width=0.88, color=cs, linewidth=0):
        b.set_snap(False)                        # uniform widths under rasterisation
    ax.set(xlim=(-1.0, len(fp.COLS) * RPITCH - (RPITCH - NB) + 0.4),
           ylim=(-(row - 1) * RGAP - 3 * BGAP - 1.4, 1.4))
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return row


def anatomy(ax):
    """A magnified single cell, annotated -- a bar means nothing without this."""
    demo = np.array([0.82, 0.80, -0.78, -0.81, 0.79, -0.77, 0.80, 0.83, 0.81, 0.82])
    ax.axhline(0, color=RULE, lw=0.6, zorder=0)
    ax.bar(np.arange(NB), demo, width=0.88, linewidth=0,
           color=[POS if v >= 0 else NEG for v in demo])
    ax.set(xlim=(-4.4, NB + 7.0), ylim=(-2.5, 2.3))
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.annotate("", xy=(NB - 0.4, -1.35), xytext=(-0.6, -1.35),
                arrowprops=dict(arrowstyle="->", lw=0.7, color=MUTED))
    ax.text((NB - 1) / 2, -1.62, "ePIE iteration  20 → 200", ha="center", va="top",
            fontsize=7, color=MUTED)
    ax.text(-1.2, 0.41, "|corr|", ha="right", va="center", fontsize=7, color=MUTED)
    ax.text(NB + 0.4, 0.85, "up = correct polarity", ha="left", va="center",
            fontsize=7, color=POS)
    ax.text(NB + 0.4, -0.85, "down = contrast reversed", ha="left", va="center",
            fontsize=7, color=NEG)
    ax.text((NB - 1) / 2, 2.25, "one row = one initial guess, one region",
            ha="center", va="top", fontsize=7, color=INK)


def main():
    ctxs = [(lab, fp.load(d)) for lab, d in CONDITIONS]

    BW, BGX = 1.94, 0.28                          # block width, gap between conditions
    LEFT, RIGHT = 0.74, 0.10
    LEG, CONDLAB, COLLAB, BOT = 0.78, 0.30, 0.62, 0.08
    GRID = 5.24
    fig_w = LEFT + 3 * BW + 2 * BGX + RIGHT
    fig_h = LEG + CONDLAB + GRID + COLLAB + BOT
    fig = plt.figure(figsize=(fig_w, fig_h))

    aw = BW * 1.30
    axl = fig.add_axes([(fig_w - aw) / 2 / fig_w, 1 - (LEG - 0.06) / fig_h,
                        aw / fig_w, (LEG - 0.22) / fig_h])
    anatomy(axl)


    for k, (lab, ctx) in enumerate(ctxs):
        x = LEFT + k * (BW + BGX)
        ax = fig.add_axes([x / fig_w, (BOT + COLLAB) / fig_h,
                           BW / fig_w, GRID / fig_h])
        draw_block(ax, ctx, brackets=(k == 0))
        ybot = ax.get_ylim()[0]
        for ci, t in enumerate(LABELS):
            ax.text(ci * RPITCH + (NB - 1) / 2, ybot - 0.6, t, rotation=45,
                    rotation_mode="anchor", ha="right", va="top", fontsize=7,
                    color=MUTED, clip_on=False)
        fig.text((x + BW / 2) / fig_w, 1 - (LEG + 0.02) / fig_h, lab, ha="center",
                 va="top", fontsize=8, color=INK)
        fig.text((x - 0.30) / fig_w, 1 - (LEG + 0.02) / fig_h, "abc"[k],
                 ha="left", va="top", fontsize=10, weight="bold", color=INK)


    OUT.mkdir(exist_ok=True)
    p = OUT / "fig_si_sparklines.png"
    for ext in (".png", ".pdf"):
        fig.savefig(p.with_suffix(ext), dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {p} ({p.stat().st_size/1e6:.1f} MB), {fig_w:.2f} x {fig_h:.2f} in")


if __name__ == "__main__":
    main()
