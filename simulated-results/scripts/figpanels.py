"""Reusable panels for the composite sweep figures.

Layout A from sweep_figure.ipynb, with the beeswarm column replaced by signed-correlation
sparklines -- one mini bar chart per run, so the polarity history of every seed is visible
rather than only its final state.  That distinction is the whole point for the fixed-noise
sweeps: a run can hold a stable sign for 200 iterations or oscillate the entire way while
its correlation *magnitude* barely moves, and only the trajectory tells them apart.

Panel geometry is in inches so every image square comes out the same size and rows line up
across blocks; see build_figure_nb.py for the reasoning.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Wedge

import sim
import analysis
import paths

GROUPS = {"au_111": ["au_111"], "au_100": ["au_100"],
          "decahedron": [f"decahedron_w{k}" for k in range(5)]}
SHORT = {"au_111": "[111]", "au_100": "[100]", "decahedron": "[110] twin"}
# wedges carry their zone axis like the single crystals do, and are numbered from 1 to
# match the Poisson draw labels
REGION_SHORT = {"au_111": "[111]", "au_100": "[100]",
                **{f"decahedron_w{k}": f"[110] wedge {k + 1}" for k in range(5)}}
TRIO = ["au_111", "au_100", "decahedron"]
COLS = ["au_111", "au_100"] + [f"decahedron_w{k}" for k in range(5)]

P, GI = 2.15, 0.045
LG_L, LG_M = 0.28, 0.55
TG, HG = 0.24, 0.34
GH, GV = 0.32, 0.34
MB_TOP, MB_BOT = 0.06, 0.44
SPARK_W, SPARK_GAP = 7.20, 0.50
BW = 3 * P + 2 * GI
HEAD = dict(fontsize=11, weight="bold", ha="center")


def load(outdir, ref_dir=paths.REFERENCE, dp_dir="dp_cache/carbon.npz"):
    """Everything the panels need, from one sweep output directory.

    `dp_dir` is either a cache file or a sweep directory holding a `dp_cache.npz`; the
    caches ship separately from the sweeps because they depend only on the sample, not on
    the reconstruction, and one cache serves many sweeps.
    """
    out = paths.sweep(outdir)
    ref = np.load(paths.sweep(ref_dir) / "runs" / "_reference.npz", allow_pickle=True)
    region_G = {k: (None if v is None else analysis.reduce_basis(np.array(v)))
                for k, v in json.loads(str(ref["region_G"])).items()}
    df = pd.read_csv(out / "summary.csv")
    drift = pd.read_csv(out / "drift.csv")
    for d in (df, drift):
        d["group"] = np.where(d.particle.str.startswith("decahedron"), "decahedron",
                              d.particle)
        # drift.csv predates the noise_seed column; the seed offset encodes it
        if "noise_seed" not in d or d["noise_seed"].isna().all():
            d["noise_seed"] = d.seed // 100

    FRAME, CELL = {}, {}
    for g, members in GROUPS.items():
        Gref = region_G[members[0]]
        U = _upright(Gref)
        for m in members:
            th, _ = analysis.align_rotation(region_G[m], Gref)
            c, s_ = np.cos(th), np.sin(th)
            FRAME[m] = U @ np.array([[c, -s_], [s_, c]])
        CELL[g] = (U @ analysis.ws_polygon(Gref).T).T

    dpp = paths.sweep(dp_dir)
    dpc = np.load(dpp if dpp.suffix == ".npz" else dpp / "dp_cache.npz", allow_pickle=True)
    return dict(out=out, df=df, drift=drift, region_G=region_G, FRAME=FRAME, CELL=CELL,
                angles=sorted(df.angle_mrad.unique()),
                AORDER=sorted(df.angle_mrad.unique(), reverse=True),
                sampling=float(ref["sampling"]), L=float(ref["L"]),
                PARTICLES=json.loads(str(ref["particles"])),
                THETA0=float(dpc["theta0"]), dpc=dpc,
                dp_meta=json.loads(str(dpc["meta"])),
                dp_refl={k: np.array(v) for k, v in json.loads(str(dpc["refl"])).items()},
                CM=dict(zip(sorted(df.angle_mrad.unique()),
                            ["#d62728", "#ff7f0e", "#2ca02c"])),
                QF={k: analysis.cell_quantile_fn(v)
                    for k, v in region_G.items() if v is not None},
                NREG=df.particle.nunique())


def pick_seeds(ctx, n=3, verbose=False, min_corr=0.5):
    """The n initial guesses, from one dataset, that show the most visible failure.

    A reversal is obvious, but so is a large rigid shift, so both are scored: 2 points per
    reversed region plus 3 points per unit of cell-quantile offset.  Quantile rather than
    Angstrom, because half a cell is equally wrong whatever the lattice spacing.

    `min_corr` gates on reconstruction quality, and it matters.  A region whose lattice was
    never recovered has a flat correlation surface, and `signed_offset` will still return a
    sign and a shift from it -- so an unreconstructed region scores as a spectacular
    artifact.  Requiring |corr| >= min_corr keeps the score to what it is meant to measure:
    a *good* lattice in the wrong place or with the wrong sign, which is the reported
    experimental failure.  Regions below the gate are counted separately as degraded.

    Returns (dataset, seeds, score).  A score of zero means there is nothing to show.
    """
    df = ctx["df"].dropna(subset=["ws_x"]).copy()
    df["q"] = [ctx["QF"][p](np.hypot(x, y))
               for p, x, y in zip(df.particle, df.ws_x, df.ws_y)]
    good = df.signed_corr.abs() >= min_corr
    amb = np.clip((df.peak_ratio - 0.7) / 0.3, 0.0, 1.0)
    df["score"] = np.where(good, 2.0 * df["reversed"] + 3.0 * df["q"] + 0.8 * amb, 0.0)
    per_run = df.groupby(["noise_seed", "seed"])["score"].sum()
    best_ds = per_run.groupby("noise_seed").sum().idxmax()
    ranked = per_run.loc[best_ds].sort_values(ascending=False)
    seeds = sorted(ranked.index[:n])
    if verbose:
        deg = (~good).mean()
        print(f"  dataset {best_ds}: score by init "
              + ", ".join(f"{int(s)}:{v:.1f}" for s, v in ranked.items())
              + f"   [{100*deg:.0f}% of regions below |corr| {min_corr} -> degraded, "
                f"not scored]")
    return int(best_ds), [int(s) for s in seeds], float(ranked.iloc[:n].sum())


def _upright(G):
    A = np.linalg.inv(G).T
    th = np.pi / 2 - np.arctan2(A[0, 1], A[0, 0])
    c, s_ = np.cos(th), np.sin(th)
    return np.array([[c, -s_], [s_, c]])


def _bare(ax):
    ax.set_xticks([]); ax.set_yticks([])


def _sbar(ax, ln, lab, col="w", loc="right", fs=7.5):
    """Scalebar in place of ticks.  `loc` frees the corner when a marker occupies it."""
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    w, h = x1 - x0, y1 - y0
    xs = x0 + 0.08 * w if loc == "left" else x1 - 0.13 * w - ln
    ax.add_patch(Rectangle((xs, y0 + 0.075 * h), ln, 0.028 * h, color=col, zorder=5))
    ax.text(xs + ln / 2, y0 + 0.115 * h, lab, color=col, ha="center",
            va="bottom", fontsize=fs, zorder=5)


def _framed(ctx, sub):
    raw = np.column_stack([sub.ws_x.values, sub.ws_y.values])
    return np.stack([ctx["FRAME"][p] @ v for p, v in zip(sub.particle, raw)])


def draw_dps(fig, gs, ctx, dose=None, seed=0, fs=1.0, fs_in=None):
    """(a) region x angle, at this figure's dose so the panels match the reconstructions."""
    fs_in = fs if fs_in is None else fs_in
    rng = np.random.default_rng(seed)
    for i, part in enumerate(TRIO):
        for k, a in enumerate(ctx["AORDER"]):
            ax = fig.add_subplot(gs[i, k])
            key = f"{part}_{a:g}"
            mx = ctx["dp_meta"][key]["maxangle"]
            dp = ctx["dpc"][key].astype(float)
            if dose:
                n_e = dose * sim.STEP ** 2
                tot = dp.sum()
                dp = rng.poisson(n_e * dp / tot) / n_e * tot
            ax.imshow(dp.T ** 0.22, cmap="gray", origin="lower", aspect="auto",
                      extent=[-mx, mx, -mx, mx])
            g = ctx["dp_refl"][part]
            g = g[np.hypot(g[:, 0], g[:, 1]) < mx * 0.95]
            d = np.hypot(*(g[:, None, :] - g[None, :, :]).transpose(2, 0, 1))
            np.fill_diagonal(d, np.inf)
            for gv, c in zip(g, (d < 2 * a).any(1)):
                ax.add_patch(plt.Circle(gv, a, fill=False, lw=0.6,
                                        color="#ffd23f" if c else "#7fd4ff",
                                        alpha=0.95 if c else 0.6,
                                        ls="-" if c else (0, (3, 2))))
            _bare(ax); ax.set(xlim=(-mx, mx), ylim=(-mx, mx))
            if i == 0 and k == 0:       # one bar per block; all nine share a scale
                _sbar(ax, 10.0, "10 mrad", fs=7.5 * fs_in)
            if i == 0:
                ax.set_title(angle_label(ctx, a), fontsize=10 * fs)
            if k == 0:
                ax.set_ylabel(SHORT[part], fontsize=10 * fs)


def draw_recons(fig, gs, ctx, seeds=(0, 1, 2), fs=1.0, fs_in=None,
                clip=(0.01, 0.99)):
    """(b) initial guess x angle, with every contrast-reversed region outlined."""
    fs_in = fs if fs_in is None else fs_in
    # a common range per column, so the three initial guesses are comparable to each other
    vlims = {a: (tuple(np.quantile(np.concatenate(
                    [_load_phase(ctx, sd, a).ravel() for sd in seeds]), clip))
                 if clip else None)
             for a in ctx["AORDER"]}
    for i, sd in enumerate(seeds):
        for k, a in enumerate(ctx["AORDER"]):
            ax = fig.add_subplot(gs[i, k])
            recon_axes(ax, ctx, sd, a, scalebar=(i == 0 and k == 0),
                       fs=fs, fs_in=fs_in, vlim=vlims[a])
            if i == 0:
                ax.set_title(angle_label(ctx, a), fontsize=10 * fs)
            if k == 0:
                ax.set_ylabel(f"init {sd}", fontsize=10 * fs)


def draw_trails(fig, gs, ctx, fs=1.0):
    """(c) region x angle: each run's offset wandering inside its Wigner-Seitz cell."""
    for i, part in enumerate(TRIO):
        poly = ctx["CELL"][part]; lim = 1.18 * np.abs(poly).max()
        for k, a in enumerate(ctx["AORDER"]):
            ax = fig.add_subplot(gs[i, k])
            sub = ctx["drift"][(ctx["drift"].angle_mrad == a)
                               & ctx["drift"].particle.isin(GROUPS[part])]
            for _, s_ in sub.groupby(["particle", "seed"]):
                pts = _framed(ctx, s_.sort_values("iteration"))
                ax.plot(pts[:, 0], pts[:, 1], "-", lw=0.5, alpha=0.22, color=ctx["CM"][a])
                ax.plot(pts[-1, 0], pts[-1, 1], "o", ms=1.35, alpha=0.65,
                        color=ctx["CM"][a])
            ax.add_patch(plt.Polygon(poly, closed=True, fill=False, edgecolor="k", lw=0.66))
            ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="auto"); _bare(ax)
            if i == 0:
                ax.set_title(angle_label(ctx, a), fontsize=10 * fs)
            if k == 0:
                ax.set_ylabel(SHORT[part], fontsize=10 * fs)


def draw_ridges(fig, gs, ctx):
    """(d) region x angle: population of signed correlations at each checkpoint."""
    its = sorted(ctx["drift"].iteration.unique())
    ytick = list(range(0, len(its), 3)) + [len(its) - 1]
    for i, part in enumerate(TRIO):
        for k, a in enumerate(ctx["AORDER"]):
            ax = fig.add_subplot(gs[i, k])
            sub = ctx["drift"][(ctx["drift"].angle_mrad == a)
                               & ctx["drift"].particle.isin(GROUPS[part])]
            for j, it in enumerate(its):
                v = sub[sub.iteration == it].signed_corr.values
                h, e = np.histogram(v, bins=np.linspace(-1, 1, 61), density=True)
                c = 0.5 * (e[1:] + e[:-1])
                ax.fill_between(c, j, j + h / max(h.max(), 1e-9) * 1.9,
                                color=ctx["CM"][a], alpha=0.5, lw=0.3, edgecolor="k")
            ax.axvline(0, color="k", lw=0.8, ls=":")
            ax.set(xlim=(-1.05, 1.05), ylim=(-0.5, len(its) + 1.6),
                   yticks=ytick, xticks=[-1, 0, 1])
            ax.set_yticklabels([str(its[t]) for t in ytick] if k == 0 else [], fontsize=7)
            ax.tick_params(labelsize=7)
            if i != len(TRIO) - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("signed correlation", fontsize=8)
            if i == 0:
                ax.set_title(angle_label(ctx, a), fontsize=10 * fs)
            if k == 0:
                ax.set_ylabel(f"{SHORT[part]}\niteration", fontsize=9, weight="bold")


def spark_block(ax, sub, color, its=None, nbar_gap=1.7, row_gap=0.58, block_gap=1.6,
                label_cols=True, row_label="data", cols=None, context=False,
                group_name="Poisson draw"):
    """One grid of signed-correlation sparklines: regions across, runs down.

    Light bars are correct polarity, dark bars reversed.  Rows are grouped by the dataset
    they inverted, so disagreement *within* a group is non-uniqueness while a group that
    differs from its neighbour is just a different noise draw.
    """
    its = its or sorted(sub.iteration.unique())
    cols = cols or COLS
    nb = len(its)
    dark = tuple(0.45 * v for v in plt.matplotlib.colors.to_rgb(color))
    xs, hs, bs, cs = [], [], [], []
    yticks, ylabels, row = [], [], 0
    for bi, ns in enumerate(sorted(sub.noise_seed.unique())):
        seeds = sorted(sub[sub.noise_seed == ns].seed.unique())
        yticks.append(-(row + len(seeds) / 2) * (2 + row_gap) - bi * block_gap)
        ylabels.append("")
        if group_name and sub.noise_seed.nunique() > 1:
            top = -row * (2 + row_gap) - bi * block_gap + 1.0
            bot = -(row + len(seeds) - 1) * (2 + row_gap) - bi * block_gap - 1.0
            ax.plot([-2.2, -2.2], [bot, top], color="0.45", lw=1.1, clip_on=False)
            ax.text(-3.2, 0.5 * (top + bot), f"{group_name} {bi + 1:02d}", rotation=90,
                    ha="center", va="center", fontsize=8, color="0.3")
        for sd in seeds:
            y0 = -row * (2 + row_gap) - bi * block_gap
            for ci, reg in enumerate(cols):
                g = sub[(sub.seed == sd) & (sub.particle == reg)].sort_values("iteration")
                if not len(g):
                    continue
                x0 = ci * (nb + nbar_gap)
                v = g.signed_corr.values
                xs.extend(x0 + np.arange(len(v)))
                hs.extend(v); bs.extend([y0] * len(v))
                cs.extend([color if t >= 0 else dark for t in v])
                ax.plot([x0 - 0.5, x0 + nb - 0.5], [y0, y0], color="0.85", lw=0.3, zorder=0)
            row += 1
    for b in ax.bar(xs, hs, bottom=bs, width=0.86, color=cs, linewidth=0):
        b.set_snap(False)          # keep bar widths uniform under rasterisation
    if label_cols:
        for ci, reg in enumerate(cols):
            ax.text(ci * (nb + nbar_gap) + nb / 2 - 0.5, 1.55, REGION_SHORT[reg],
                    ha="center", va="bottom", fontsize=7.6, weight="bold")
    ax.set(xlim=(-4.6 if group_name else -1.2,
                 len(cols) * (nb + nbar_gap) - nbar_gap + 0.2),
           ylim=(min(bs) - 1.4, 2.95 if label_cols else 2.0), yticks=yticks)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_visible(False)
    if context:
        # a bar means nothing without this: height is |correlation|, colour is its sign,
        # and the horizontal axis is ePIE iteration.  The colour key lives in the panel
        # header instead of a legend, which would otherwise collide with the title.
        y0 = min(bs) - 1.0
        for ci in range(len(cols)):
            x0 = ci * (nb + nbar_gap)
            ax.annotate("", xy=(x0 + nb - 0.5, y0), xytext=(x0 - 0.5, y0),
                        arrowprops=dict(arrowstyle="->", lw=0.7, color="0.45"))
        ax.text(-0.5, y0 - 0.7,
                f"iteration {its[0]}\u2192{its[-1]}   \u00b7   bar height = |signed "
                f"correlation|   \u00b7   one row per initial guess",
                fontsize=7.2, color="0.35", va="top")
        ax.set_ylim(y0 - 1.65, ax.get_ylim()[1])


def draw_sparklines(fig, slots, ctx, **kw):
    """(e) one sparkline grid per convergence angle."""
    its = sorted(ctx["drift"].iteration.unique())
    for slot, a in zip(slots, ctx["AORDER"]):
        ax = fig.add_axes(slot) if isinstance(slot, list) else fig.add_subplot(slot)
        spark_block(ax, ctx["drift"][ctx["drift"].angle_mrad == a], ctx["CM"][a], its,
                    context=(a == ctx["AORDER"][-1]), **kw)
        ax.set_title(angle_label(ctx, a).replace("\n", "  "), fontsize=10.5,
                     weight="bold", pad=13)


def _load_phase(ctx, seed, angle):
    """Band-limited reconstructed phase for one run."""
    # a multi-angle context keeps one directory per angle; a single-sweep one has just "out"
    out = ctx.get("out_by_angle", {}).get(angle, ctx["out"])
    fa = ctx.get("src_angle", {}).get(angle, angle)
    return analysis.band_limit(
        np.load(out / "runs" / f"a{fa:g}_s{seed:03d}.npz")["phase"].astype(float),
        ctx["sampling"])


def recon_axes(ax, ctx, seed, angle, scalebar=True, fs=1.0, fs_in=None,
               clip=(0.01, 0.99), vlim=None):
    """One reconstruction with every contrast-reversed region outlined."""
    fs_in = fs if fs_in is None else fs_in
    L = ctx["L"]
    ph = _load_phase(ctx, seed, angle)
    # Scaling to +/-max(|phase|) lets a few edge-ringing pixels set the range and flattens
    # the lattice, so the tails are clipped instead.  `vlim` lets the caller impose one
    # range across several runs -- without it each panel self-scales and runs that differ
    # only in noise look like they differ in contrast.
    if vlim is not None:
        lo, hi = vlim
    else:
        lo, hi = np.quantile(ph, clip) if clip else (-np.abs(ph).max(), np.abs(ph).max())
    ax.imshow(ph.T, cmap="gray", origin="lower", vmin=lo, vmax=hi, aspect="auto",
              extent=[0, L, 0, L])
    nrev = namb = 0
    for _, r in ctx["df"][(ctx["df"].angle_mrad == angle)
                          & (ctx["df"].seed == seed)].iterrows():
        if not np.isfinite(r.ws_x):
            continue
        q = ctx["QF"][r.particle](np.hypot(r.ws_x, r.ws_y))
        rev = r.signed_corr < 0
        # peak_ratio near 1 means the correlation surface has a competing extremum: the
        # lattice may be correctly placed yet carry a half-cell interstitial component,
        # which looks like real structure and is missed by offset and polarity alike
        amb = (not rev) and (r.peak_ratio > 0.7 or q > 0.25)
        if not (rev or amb):
            continue
        col = "#ff2d2d" if rev else "#ffb000"
        nrev += rev; namb += amb
        if r.particle.startswith("decahedron"):
            w = int(r.particle.rsplit("w", 1)[1])
            ax.add_patch(Wedge(ctx["PARTICLES"]["decahedron"], 11.0,
                               ctx["THETA0"] + 72 * w, ctx["THETA0"] + 72 * (w + 1),
                               fill=False, color=col, lw=0.72, ls="--"))
        else:
            ax.add_patch(plt.Circle(ctx["PARTICLES"][r.particle], 11.5, fill=False,
                                    color=col, lw=0.72, ls="--"))
    bits = []
    if nrev:
        bits.append(("%d reversed" % nrev, "#ff2d2d"))
    if namb:
        bits.append(("%d ambiguous" % namb, "#ffb000"))
    if not bits:
        bits = [("0/%d" % ctx["NREG"], "0.12")]
    for n, (t, c) in enumerate(bits):
        ax.text(0.03, 0.03 + 0.085 * n, t, transform=ax.transAxes, ha="left",
                va="bottom", fontsize=7.5 * fs_in, color=c)
    _bare(ax)
    if scalebar:
        _sbar(ax, 20.0, "20 Å", fs=7.5 * fs_in)
    return nrev


def layout_A(ctx, dose=None, seeds=None, title=""):
    """Four 3x3 image quadrants plus a sparkline column, all squares equal and aligned."""
    TP = 0.40 if title else 0.0          # room for the figure title above the headers
    fig_w = LG_L + BW + GH + LG_M + BW + SPARK_GAP + SPARK_W + 0.12
    fig_h = MB_TOP + TP + HG + TG + BW + GV + HG + TG + BW + MB_BOT
    fig = plt.figure(figsize=(fig_w, fig_h))
    xL, xM = LG_L, LG_L + BW + GH + LG_M
    yT = MB_TOP + TP + HG + TG
    yB = yT + BW + GV + HG + TG

    def blk(x, y):
        return dict(left=x / fig_w, right=(x + BW) / fig_w,
                    top=1 - y / fig_h, bottom=1 - (y + BW) / fig_h,
                    wspace=GI / P, hspace=GI / P)

    draw_dps(fig, fig.add_gridspec(3, 3, **blk(xL, yT)), ctx, dose=dose)
    draw_trails(fig, fig.add_gridspec(3, 3, **blk(xM, yT)), ctx)
    if seeds is None:
        _, seeds, _ = pick_seeds(ctx)
    draw_recons(fig, fig.add_gridspec(3, 3, **blk(xL, yB)), ctx, seeds=seeds)
    draw_ridges(fig, fig.add_gridspec(3, 3, **blk(xM, yB)), ctx)

    bx = xM + BW + SPARK_GAP
    span = (yB + BW) - yT
    ph = (span - 2 * 0.55) / 3.0
    slots = [[(bx + 0.55) / fig_w, 1 - (yT + k * (ph + 0.55) + ph) / fig_h,
              (SPARK_W - 0.58) / fig_w, ph / fig_h] for k in range(3)]
    draw_sparklines(fig, slots, ctx)

    cxL, cxM = (xL + BW / 2) / fig_w, (xM + BW / 2) / fig_w
    cxS = (bx + SPARK_W / 2) / fig_w
    hyT = 1 - (MB_TOP + TP + HG * 0.35) / fig_h
    hyB = 1 - (yT + BW + GV + HG * 0.35) / fig_h
    fig.text(cxL, hyT, "(a)  diffraction patterns: which disks overlap", **HEAD)
    fig.text(cxM, hyT, "(c)  offset trails in the Wigner-Seitz cell", **HEAD)
    fig.text(cxL, hyB, "(b)  reconstructions, three initial guesses, one dataset", **HEAD)
    fig.text(cxM, hyB, "(d)  polarity population vs iteration", **HEAD)
    fig.text(cxS, hyT, "(e)  signed correlation per run, every checkpoint", **HEAD)
    if title:
        fig.text(0.5, 1 - MB_TOP / fig_h, title, ha="center", va="top", fontsize=13.5,
                 weight="bold")
    return fig


# ---------------------------------------------------------------- multislice comparison

DOSE_CM = {"inf": "#1f4e79", "1e6": "#2ca02c", "1e5": "#ff7f0e", "1e4": "#d62728"}
DOSE_LAB = {"inf": "noiseless", "1e6": r"$10^6$ e/Å$^2$", "1e5": r"$10^5$ e/Å$^2$",
            "1e4": r"$10^4$ e/Å$^2$"}


def layout_ms(ctxs, angle=3.0, doses=("inf", "1e6", "1e5", "1e4"),
              models=("complex", "phase"), title="", col_labels=None):
    """Multislice forward, single-slice inverse: object model x dose, at one angle.

    `ctxs` maps (model, dose) -> context.  Two sparkline grids per dose sit side by side so
    the free complex object and the |O| = 1 phase object can be compared row by row on the
    same data, with the worst-failing reconstruction of each shown alongside.
    """
    SP, RC, GAPC, GAPR = 5.15, 2.35, 0.34, 0.26
    LEFT, RIGHT, TOPM, BOTM = 0.62, 0.16, 0.72, 0.10
    HDR = 0.30
    row_h = 2.62
    fig_w = LEFT + 2 * SP + GAPC + 0.55 + 2 * RC + GAPC + RIGHT
    fig_h = TOPM + len(doses) * row_h + (len(doses) - 1) * GAPR + BOTM
    fig = plt.figure(figsize=(fig_w, fig_h))

    xs_spark = [LEFT, LEFT + SP + GAPC]
    xs_rec = [LEFT + 2 * SP + GAPC + 0.55, LEFT + 2 * SP + GAPC + 0.55 + RC + GAPC]

    for r, dose in enumerate(doses):
        y = TOPM + r * (row_h + GAPR)
        for c, model in enumerate(models):
            ctx = ctxs[(model, dose)]
            sub = ctx["drift"][ctx["drift"].angle_mrad == angle]
            ax = fig.add_axes([xs_spark[c] / fig_w, 1 - (y + row_h) / fig_h,
                               SP / fig_w, row_h / fig_h])
            spark_block(ax, sub, DOSE_CM[dose], label_cols=(r == 0), row_label="data")
            if r == 0:
                ax.set_title((col_labels or {}).get(model, f"{model} object"),
                             fontsize=11.5, weight="bold", pad=16)

            _, seeds, score = pick_seeds(ctx, n=1)
            axr = fig.add_axes([xs_rec[c] / fig_w, 1 - (y + RC) / fig_h,
                                RC / fig_w, RC / fig_h])
            recon_axes(axr, ctx, seeds[0], angle)
            axr.set_xlabel(f"init {seeds[0]}", fontsize=8, labelpad=1)
            if r == 0:
                axr.set_title("worst run,\n"
                              + (col_labels or {}).get(model, model).split(",")[0],
                              fontsize=9.5, weight="bold", pad=4)
        fig.text(LEFT * 0.42 / fig_w, 1 - (y + row_h / 2) / fig_h, DOSE_LAB[dose],
                 rotation=90, va="center", ha="center", fontsize=12, weight="bold")

    if title:
        fig.text(0.5, 1 - 0.08 / fig_h, title, ha="center", va="top", fontsize=13.5,
                 weight="bold")
    fig.text((LEFT + SP + GAPC / 2) / fig_w, 1 - (TOPM - HDR) / fig_h,
             "signed correlation per run, every checkpoint", ha="center", va="bottom",
             fontsize=11, weight="bold")
    return fig


def load_multi(dirs_by_angle, **kw):
    """One context spanning several per-angle output directories.

    The multislice sweeps write one directory per (angle, model, dose), but Layout A needs
    all three angles together, so the tables are concatenated and the per-angle directory
    is remembered for loading the stored reconstructions.

    A value may be `dir` or `(dir, source_angle)`.  The second form stands another angle's
    results in a column whose own run has not finished, which is useful for a provisional
    figure -- the substituted column is recorded in ctx["placeholder"] so it can be labelled
    as such, because nothing else in the figure would reveal the swap.
    """
    src = {a: (v if isinstance(v, tuple) else (v, a)) for a, v in dirs_by_angle.items()}
    ctxs = {a: load(d, **kw) for a, (d, _) in src.items()}
    first = ctxs[sorted(ctxs)[0]]
    ctx = dict(first)
    ctx["df"] = pd.concat([c["df"].assign(angle_mrad=a) for a, c in ctxs.items()],
                          ignore_index=True)
    ctx["drift"] = pd.concat([c["drift"].assign(angle_mrad=a) for a, c in ctxs.items()],
                             ignore_index=True)
    ctx["angles"] = sorted(dirs_by_angle)
    ctx["AORDER"] = sorted(dirs_by_angle, reverse=True)
    ctx["CM"] = dict(zip(ctx["angles"], ["#d62728", "#ff7f0e", "#2ca02c"]))
    ctx["out_by_angle"] = {a: c["out"] for a, c in ctxs.items()}
    ctx["src_angle"] = {a: sa for a, (_, sa) in src.items()}
    ctx["placeholder"] = {a for a, sa in ctx["src_angle"].items() if sa != a}
    return ctx


def angle_label(ctx, a):
    """Column heading, flagged when the column is standing in for a missing run."""
    if a in ctx.get("placeholder", ()):
        return f"{a:g} mrad\n[{ctx['src_angle'][a]:g} mrad data]"
    return f"{a:g} mrad"


def _au_columns(pitch=0.35):
    au = sim.build_sample(support=False)
    return np.unique(np.round(au.positions[:, :2] / pitch).astype(int), axis=0) * pitch


def layout_artifacts(ctxs, angle=3.0, doses=("inf", "1e6", "1e5", "1e4"), n_show=3,
                     zoom_half=8.0, title=""):
    """The failing reconstructions themselves: worst runs per dose, plus a zoom.

    The right-hand column enlarges whichever region failed worst in that run and overlays
    the true Au column positions, because a rigid sub-cell shift is invisible at full field
    but obvious once the lattice is set against the atoms it is supposed to represent.
    """
    cols = _au_columns()
    PC, ZC = 2.30, 3.05
    LEFT, RIGHT, TOPM, BOTM, GX, GY = 0.72, 0.16, 0.74, 0.20, 0.10, 0.30
    fig_w = LEFT + n_show * (PC + GX) + 0.45 + ZC + RIGHT
    fig_h = TOPM + len(doses) * (ZC + GY) + BOTM
    fig = plt.figure(figsize=(fig_w, fig_h))

    for r, dose in enumerate(doses):
        ctx = ctxs[dose]
        y = TOPM + r * (ZC + GY)
        df = ctx["df"][ctx["df"].angle_mrad == angle].dropna(subset=["ws_x"]).copy()
        df["q"] = [ctx["QF"][p](np.hypot(x, y_))
                   for p, x, y_ in zip(df.particle, df.ws_x, df.ws_y)]
        _amb = np.clip((df.peak_ratio - 0.7) / 0.3, 0.0, 1.0)
        df["score"] = np.where(df.signed_corr.abs() >= 0.5,
                               2.0 * df["reversed"] + 3.0 * df["q"] + 0.8 * _amb, 0.0)
        ranked = df.groupby("seed")["score"].sum().sort_values(ascending=False)
        seeds = list(ranked.index[:n_show])

        for c, sd in enumerate(seeds):
            ax = fig.add_axes([(LEFT + c * (PC + GX)) / fig_w,
                               1 - (y + PC + (ZC - PC) / 2) / fig_h, PC / fig_w, PC / fig_h])
            recon_axes(ax, ctx, int(sd), angle)
            ax.set_xlabel(f"init {int(sd)}   score {ranked.loc[sd]:.1f}", fontsize=8,
                          labelpad=1)
            if r == 0:
                ax.set_title(f"worst run {c + 1}", fontsize=9.5, weight="bold", pad=4)

        worst = df[df.seed == seeds[0]].sort_values("score", ascending=False).iloc[0]
        base = "decahedron" if worst.particle.startswith("decahedron") else worst.particle
        cen = ctx["PARTICLES"][base]
        samp, L = ctx["sampling"], ctx["L"]
        zout = ctx.get("out_by_angle", {}).get(angle, ctx["out"])
        ph = analysis.band_limit(
            np.load(zout / "runs" /
                    f"a{angle:g}_s{int(seeds[0]):03d}.npz")["phase"].astype(float), samp)
        sl = (slice(int((cen[0] - zoom_half) / samp), int((cen[0] + zoom_half) / samp)),
              slice(int((cen[1] - zoom_half) / samp), int((cen[1] + zoom_half) / samp)))
        ext = [cen[0] - zoom_half, cen[0] + zoom_half,
               cen[1] - zoom_half, cen[1] + zoom_half]
        axz = fig.add_axes([(LEFT + n_show * (PC + GX) + 0.45) / fig_w,
                            1 - (y + ZC) / fig_h, ZC / fig_w, ZC / fig_h])
        p = ph[sl]; v = np.abs(p).max()
        axz.imshow(p.T, cmap="gray", origin="lower", vmin=-v, vmax=v, extent=ext,
                   aspect="auto")
        zc = cols[(np.abs(cols[:, 0] - cen[0]) < zoom_half + 1)
                  & (np.abs(cols[:, 1] - cen[1]) < zoom_half + 1)]
        axz.scatter(zc[:, 0], zc[:, 1], s=26, facecolors="none", edgecolors="#ff4d4d",
                    lw=1.0)
        axz.set(xlim=ext[:2], ylim=ext[2:]); _bare(axz); _sbar(axz, 5.0, "5 Å")
        axz.set_xlabel(f"{REGION_SHORT[worst.particle]}  corr {worst.signed_corr:+.2f}, "
                       f"offset {np.hypot(worst.ws_x, worst.ws_y):.2f} Å "
                       f"(q {worst.q:.2f})", fontsize=8.5, labelpad=2)
        if r == 0:
            axz.set_title("worst region, true Au columns in red", fontsize=9.5,
                          weight="bold", pad=4)
        fig.text(LEFT * 0.36 / fig_w, 1 - (y + ZC / 2) / fig_h, DOSE_LAB[dose],
                 rotation=90, va="center", ha="center", fontsize=12, weight="bold")

    if title:
        fig.text(0.5, 1 - 0.09 / fig_h, title, ha="center", va="top", fontsize=13,
                 weight="bold")
    return fig


# ---------------------------------------------------------------- main-body figure

@lru_cache(maxsize=1)
def _default_QF(ref_dir=paths.REFERENCE):
    """Cell-quantile functions keyed by region, shared by every sweep.

    The sample geometry is identical across sweeps, so these depend only on the reference
    lattice and can be built once for callers that pass a bare directory.
    """
    ref = np.load(paths.sweep(ref_dir) / "runs" / "_reference.npz", allow_pickle=True)
    G = {k: (None if v is None else analysis.reduce_basis(np.array(v)))
         for k, v in json.loads(str(ref["region_G"])).items()}
    return {k: analysis.cell_quantile_fn(v) for k, v in G.items() if v is not None}


def nonunanimity(d, angle=None, min_corr=0.5, q_tol=0.25):
    """Fraction of (dataset, region) groups whose initial guesses disagree.

    The decision-relevant number: given one measurement, do different starting points give
    different answers?  Regions whose lattice was never recovered (|corr| < min_corr) are
    excluded, so this counts genuine non-uniqueness and not reconstruction failure.

    A group disagrees if its runs differ in POLARITY, or if their offsets span more than
    `q_tol` of the Wigner-Seitz cell.  Comparing offsets exactly would instead count a
    single pixel of numerical jitter as non-uniqueness -- on the matched 1e5 control no
    two offsets are bitwise equal, yet they agree to 1.6% of the cell, which would report
    0.11 for a condition where nothing whatever is ambiguous.  The 0.25 threshold is the
    one the reconstruction panels already use to flag a region as visibly shifted.

    Returns (rate, n_groups); rate is nan when the run does not exist yet.
    """
    if isinstance(d, dict) and "df" in d:          # an already-loaded (possibly
        df = d["df"].dropna(subset=["ws_x"])        # draw-limited) context
        QF = d.get("QF") or _default_QF()
    else:
        f = paths.sweep(d) / "summary.csv"
        if not f.exists():
            return float("nan"), 0
        df = pd.read_csv(f).dropna(subset=["ws_x"])
        QF = _default_QF()
    if angle is not None:
        df = df[df.angle_mrad == angle]
    df = df[df.signed_corr.abs() >= min_corr]
    if not len(df):
        return float("nan"), 0
    df = df.assign(ns=df.seed // 100)
    tot = dis = 0
    for (_, part), g in df.groupby(["ns", "particle"]):
        if len(g) < 2:
            continue
        tot += 1
        if g["reversed"].nunique() > 1:
            dis += 1
            continue
        P = g[["ws_x", "ws_y"]].to_numpy()
        sep = np.hypot(*(P[:, None, :] - P[None, :, :]).transpose(2, 0, 1)).max()
        qf = QF.get(part)
        dis += int(qf is not None and float(qf(sep)) > q_tol)
    return (dis / tot if tot else float("nan")), tot


def _heat(ax, M, rowlab, collab, title, cmap="magma_r", vmax=1.0):
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(collab)), collab, fontsize=8)
    ax.set_yticks(range(len(rowlab)), rowlab, fontsize=8.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, "n/a" if not np.isfinite(v) else f"{v:.2f}", ha="center",
                    va="center", fontsize=8,
                    color="w" if (np.isfinite(v) and v > 0.55) else "0.15")
    ax.set_title(title, fontsize=9.5, weight="bold", pad=4)
    return im


def layout_main(dp_dir="dp_cache/carbon.npz", ref_dir=paths.REFERENCE):
    """Single main-body figure: what the artifact looks like, and when it appears."""
    ctx4 = load("ss_carbon_complex_1e4", ref_dir=ref_dir, dp_dir=dp_dir)
    cols = _au_columns()
    DOSES_ = ["inf", "1e6", "1e5", "1e4"]
    DLAB = ["noiseless", "$10^6$", "$10^5$", "$10^4$"]
    MATCHED = {"inf": ref_dir, "1e6": "ss_carbon_complex_1e6",
               "1e5": "ss_carbon_complex_1e5", "1e4": "ss_carbon_complex_1e4"}

    fig = plt.figure(figsize=(15.4, 9.4))
    gs = fig.add_gridspec(2, 3, wspace=0.30, hspace=0.30,
                          height_ratios=[1.06, 1.0],
                          left=0.055, right=0.975, top=0.885, bottom=0.065)

    # ---- (a) the controlling variable: does any pair of disks overlap? -------------
    sub = gs[0, 0].subgridspec(1, 3, wspace=0.06)
    for k, a in enumerate(ctx4["AORDER"]):
        ax = fig.add_subplot(sub[0, k])
        key = f"au_111_{a:g}"
        mx = ctx4["dp_meta"][key]["maxangle"]
        ax.imshow(ctx4["dpc"][key].T ** 0.22, cmap="gray", origin="lower", aspect="auto",
                  extent=[-mx, mx, -mx, mx])
        g = ctx4["dp_refl"]["au_111"]
        g = g[np.hypot(g[:, 0], g[:, 1]) < mx * 0.95]
        d = np.hypot(*(g[:, None, :] - g[None, :, :]).transpose(2, 0, 1))
        np.fill_diagonal(d, np.inf)
        for gv, c in zip(g, (d < 2 * a).any(1)):
            ax.add_patch(plt.Circle(gv, a, fill=False, lw=0.9,
                                    color="#ffd23f" if c else "#7fd4ff",
                                    ls="-" if c else (0, (3, 2)), alpha=0.9))
        _bare(ax); ax.set(xlim=(-mx, mx), ylim=(-mx, mx)); ax.set_box_aspect(1)
        ax.set_title(f"{a:g} mrad", fontsize=9)
        if k == 0:
            ax.set_ylabel("$[111]$", fontsize=9, weight="bold")
    fig.text(0.055 + (0.975 - 0.055) / 6, 0.935,
             "(a)  overlap sets everything", ha="center", fontsize=10.5, weight="bold")

    # ---- (b) what the failure looks like ------------------------------------------
    d4 = ctx4["df"][ctx4["df"].angle_mrad == 3.0].dropna(subset=["ws_x"]).copy()
    d4["q"] = [ctx4["QF"][p](np.hypot(x, y))
               for p, x, y in zip(d4.particle, d4.ws_x, d4.ws_y)]
    ok = d4[d4.signed_corr.abs() >= 0.5]
    shifted = ok.sort_values("q", ascending=False).iloc[0]
    flipped = ok[ok["reversed"]].sort_values("signed_corr").iloc[0]
    ctx12 = load("ss_carbon_complex_1e4", ref_dir=ref_dir, dp_dir=dp_dir)
    good12 = ctx12["df"][(ctx12["df"].angle_mrad == 12.0)].iloc[0]
    sub = gs[0, 1].subgridspec(1, 3, wspace=0.06)
    panels = [(12.0, int(good12.seed), good12.particle, "correct\n(12 mrad, overlap)"),
              (3.0, int(shifted.seed), shifted.particle,
               f"offset {np.hypot(shifted.ws_x, shifted.ws_y):.1f} Å\n(q {shifted.q:.2f})"),
              (3.0, int(flipped.seed), flipped.particle,
               f"polarity reversed\ncorr {flipped.signed_corr:+.2f}")]
    for k, (a, sd, reg, lab) in enumerate(panels):
        ax = fig.add_subplot(sub[0, k])
        base = "decahedron" if reg.startswith("decahedron") else reg
        cen = ctx4["PARTICLES"][base]
        samp = ctx4["sampling"]; zh = 7.0
        ph = analysis.band_limit(np.load(
            ctx4["out"] / "runs" / f"a{a:g}_s{sd:03d}.npz")["phase"].astype(float), samp)
        sl = (slice(int((cen[0]-zh)/samp), int((cen[0]+zh)/samp)),
              slice(int((cen[1]-zh)/samp), int((cen[1]+zh)/samp)))
        ext = [cen[0]-zh, cen[0]+zh, cen[1]-zh, cen[1]+zh]
        p = ph[sl]; v = np.abs(p).max()
        ax.imshow(p.T, cmap="gray", origin="lower", vmin=-v, vmax=v, extent=ext,
                  aspect="auto")
        zc = cols[(np.abs(cols[:, 0]-cen[0]) < zh+1) & (np.abs(cols[:, 1]-cen[1]) < zh+1)]
        ax.scatter(zc[:, 0], zc[:, 1], s=16, facecolors="none", edgecolors="#ff4d4d", lw=0.8)
        ax.set(xlim=ext[:2], ylim=ext[2:]); _bare(ax); _sbar(ax, 5.0, "5 Å")
        ax.set_box_aspect(1)
        ax.set_title(lab, fontsize=8, pad=3)
    fig.text(0.055 + 3 * (0.975 - 0.055) / 6, 0.935,
             "(b)  the artifact is predictable", ha="center", fontsize=10.5, weight="bold")

    # ---- (c) undiagnosable within a single run ------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    dr = ctx4["drift"]
    dr = dr[(dr.angle_mrad == 3.0) & (dr.noise_seed == 0)]
    keep = ["au_111", "au_100", "decahedron_w0"]
    spark_block(ax, dr[dr.particle.isin(keep)], "#d62728", label_cols=True,
                row_label="", cols=keep)
    ax.set_title("(c)  sign wanders during one run,\nwhile the lattice stays correct",
                 fontsize=10.5, weight="bold", pad=16)

    # ---- (d) only without overlap --------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    M = np.array([[nonunanimity(MATCHED[d], angle=a)[0] for d in DOSES_]
                  for a in (12.0, 6.0, 3.0)])
    _heat(ax, M, ["12 mrad", "6 mrad", "3 mrad"], DLAB,
          "(d)  non-uniqueness rate: only where disks miss")

    # ---- (e) only with a substrate, only at high dose -------------------------------
    ax = fig.add_subplot(gs[1, 1])
    BARE = {d: f"ss_bare_complex_{d}" for d in ("inf", "1e6", "1e5", "1e4")}
    M = np.array([[nonunanimity(MATCHED[d], angle=3.0)[0] for d in DOSES_],
                  [nonunanimity(BARE[d], angle=3.0)[0] for d in DOSES_]])
    _heat(ax, M, ["on carbon", "free-standing"], DLAB,
          "(e)  substrate and dose, at 3 mrad")
    if not np.isfinite(M[1]).any():
        ax.text(1.5, 1, "running", ha="center", va="center", fontsize=11,
                weight="bold", color="#1f77b4",
                bbox=dict(fc="w", ec="#1f77b4", boxstyle="round,pad=0.3"))

    # ---- (f) only with a matched model ---------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    MODELS = [("matched", MATCHED),
              ("multislice", {d: f"ms_carbon_a3_complex_{d}" for d in DOSES_}),
              ("+ phonons", {d: f"msfp_carbon_a3_complex_{d}" for d in DOSES_}),
              ("pure-phase obj.", {d: f"ms_carbon_a3_phase_{d}" for d in DOSES_})]
    M = np.array([[nonunanimity(m[d], angle=3.0)[0] for d in DOSES_] for _, m in MODELS])
    im = _heat(ax, M, [n for n, _ in MODELS], DLAB,
               "(f)  model mismatch, at 3 mrad")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("fraction of regions where\ninitial guesses disagree", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)

    fig.suptitle("Nanobeam ptychography below the disk-overlap threshold: when it fails, "
                 "and how the failure looks", fontsize=13, weight="bold", y=0.975)
    return fig


# ---------------------------------------------------------------- paired-dose figure

def _load_any(spec, **kw):
    """A sweep directory, {angle: directory}, or an already-loaded context."""
    if isinstance(spec, dict) and "df" in spec:
        return spec
    return load_multi(spec, **kw) if isinstance(spec, dict) else load(spec, **kw)


def limit_draws(ctx, n):
    """Keep only the first n Poisson realisations.

    Sweeps were not all run with the same number of noise draws, and the sparkline blocks
    are only comparable row to row if each shows the same number of rows.
    """
    out = dict(ctx)
    for k in ("df", "drift"):
        out[k] = ctx[k][ctx[k].noise_seed < n].copy()
    return out


def layout_rows(rows, spark_angle=3.0, n_inits=3, fig_w=7.20, FS=0.80,
                FS_IN=0.66):
    """One band per (condition, dose): diffraction data, reconstructions, offset trails.

    `rows` is a list of (row_label, spec, dose_value[, load_kwargs]), where spec is a sweep
    directory or {angle: directory}.  The optional fourth element is passed to the loader --
    `dp_dir` in particular, so a row whose sample differs shows its own diffraction data
    rather than the default supported cache.  Keeping the rows arbitrary rather than tying
    them to one condition lets a figure change one variable at a time -- object model
    between two rows, dose between the next two -- which is what makes a multi-row
    comparison readable.

    Sized to the final page rather than to a large canvas that is scaled down at
    typesetting: a 10 pt label on a 16-inch figure reduced to 7.2 in prints at 4.5 pt.
    Everything here is laid out at `fig_w` inches, so the point sizes are what the reader
    gets.  Carries no title and no panel headings -- those belong in the caption.
    """
    GAP, BAND = 0.30, 0.26
    LEFT, RIGHT, TOP, BOT = 0.50, 0.05, 0.26, 0.08
    BW_ = (fig_w - LEFT - RIGHT - 2 * GAP) / 3
    GI_ = 0.022
    P_ = (BW_ - 2 * GI_) / 3
    fig_h = TOP + len(rows) * (BW_ + BAND) + BOT
    fig = plt.figure(figsize=(fig_w, fig_h))

    def blk(x, y):
        return dict(left=x / fig_w, right=(x + BW_) / fig_w,
                    top=1 - y / fig_h, bottom=1 - (y + BW_) / fig_h,
                    wspace=GI_ / P_, hspace=GI_ / P_)

    xs = [LEFT + k * (BW_ + GAP) for k in range(3)]
    for r, row in enumerate(rows):
        rlab, spec, dosev = row[:3]
        ctx = _load_any(spec, **(row[3] if len(row) > 3 else {}))
        y = TOP + r * (BW_ + BAND)
        _, seeds, _ = pick_seeds(ctx, n=n_inits)

        draw_dps(fig, fig.add_gridspec(3, 3, **blk(xs[0], y)), ctx, dose=dosev, fs=FS, fs_in=FS_IN)
        draw_recons(fig, fig.add_gridspec(3, 3, **blk(xs[1], y)), ctx, seeds=seeds, fs=FS, fs_in=FS_IN)
        draw_trails(fig, fig.add_gridspec(3, 3, **blk(xs[2], y)), ctx, fs=FS)

        fig.text(0.085 / fig_w, 1 - (y + BW_ / 2) / fig_h, rlab, rotation=90,
                 va="center", ha="center", fontsize=8.5)
        if r == 0:                      # one letter per column, not per row, so that
            for k, letter in enumerate("abc"):   # "panel b" is unambiguous
                fig.text((xs[k] - 0.185) / fig_w, 1 - (y - 0.03) / fig_h, letter,
                         ha="left", va="bottom", fontsize=10.5, weight="bold")
    return fig


def layout_pair(dirs_by_dose, label="", doses=("1e5", "1e4"), **kw):
    """Two doses of one condition -- a thin wrapper over layout_rows."""
    return layout_rows([(DOSE_LAB.get(d, d), dirs_by_dose[d], DOSEV_.get(d))
                        for d in doses], label=label, **kw)


DOSEV_ = {"inf": None, "1e6": 1e6, "1e5": 1e5, "1e4": 1e4}


def _first_dir(spec):
    if isinstance(spec, dict) and "df" in spec:
        return spec                                # nonunanimity reads it directly
    return spec if not isinstance(spec, dict) else spec[sorted(spec)[0]]
