"""Particle-by-particle lattice registry of the experimental reconstructions.

    python register_lattice.py --scan 26                 # every run of scan 26 in e1/
    python register_lattice.py --scan 26 --ref-scan 23   # and against the 24 mrad run

Each reconstruction's phase is band-passed to the lattice (0.3-0.8 1/A, above every
particle's envelope and below the information limit) and compared particle by particle
with a reference run of the same scan, seed 0 by default.  The signed peak of their
masked cross-correlation gives the relative polarity, and its position the relative
lattice offset, reduced to the particle's Wigner-Seitz cell: the same scoring as the
simulated regions (simulated-results/scripts/analysis.py).  This is done at every stored
snapshot, so the 50-iteration point is the published setting.

Particles are found once per scan, from the envelope of the reference run (its phase
low-passed below 0.08 1/A), as connected regions above an Otsu threshold, eroded by 3 A
so that edges do not dominate.  Every run is first aligned to the reference by the
cross-correlation of envelopes, so a rigid shift of the whole object, which a refined
probe can absorb, is not counted as a lattice offset.

Writes registry_scan<N>.csv (one row per run, snapshot, and particle) next to the runs
and prints, per particle, how often the runs disagree with the reference.
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "simulated-results/scripts"))
import analysis  # noqa: E402

E1 = Path.home() / "data/nanobeam_ptycho/recons/e1"
LATTICE, ENVELOPE = (0.3, 0.8), 0.08          # 1/Angstrom
MIN_AREA_A2, ERODE_A = 150.0, 3.0


def runs(scan):
    return sorted(E1.glob(f"scan{scan}_bin*_s*_it*.h5"))


def phases(f):
    """{iteration: phase} for every snapshot of one run, plus its sampling."""
    with h5py.File(f) as h:
        s = float(np.mean(h["sampling"][()]))
        its = list(h["snapshot_iters"][()])
        return dict(zip(its, h["snapshot_phases"][()].astype(float))), s


def envelope(ph, s):
    return analysis.band_limit(ph, s, lo=0.0, hi=ENVELOPE)


def lattice(ph, s):
    return analysis.band_limit(ph, s, *LATTICE)


def align(ph, ref_env, s):
    """Roll `ph` so its envelope best matches the reference's (integer pixels)."""
    e = envelope(ph, s)
    xc = np.real(np.fft.ifft2(np.fft.fft2(ref_env - ref_env.mean())
                              * np.conj(np.fft.fft2(e - e.mean()))))
    k = np.unravel_index(np.argmax(xc), xc.shape)
    return np.roll(ph, k, axis=(0, 1)), k


def particle_area(ph, s, quantile=0.55, background=25.0, edge=10.0):
    """Where agreement is scored: the particles, inside the reconstructed field of view.

    The field of view is where the lattice band carries power, eroded by `edge` A so the
    reconstruction's border artifacts are left out; unscanned corners carry none.  The
    particles are the upper part of the envelope there, after a smooth `background` (A)
    is removed, so that a slow phase ramp across the field is not mistaken for them.
    """
    env = envelope(ph, s)
    env = env - ndimage.gaussian_filter(env, background / s, mode="nearest")
    power = ndimage.gaussian_filter(lattice(ph, s) ** 2, 8.0 / s)
    fov = power > 0.05 * np.median(power)
    fov = ndimage.binary_erosion(fov, iterations=int(round(edge / s)))
    return fov & (env > np.quantile(env[fov], quantile))


def local_agreement(a, b, s, sigma=8.0):
    """Signed correlation of two lattice-band phases in a Gaussian window of `sigma` A.

    +1 where the lattices coincide, -1 where one is the contrast reversal of the other,
    near 0 where one is shifted.  Needs no segmentation, so a mask that happens to span
    two crystals cannot average their verdicts away.
    """
    g = lambda x: ndimage.gaussian_filter(x, sigma / s, mode="wrap")
    return g(a * b) / np.sqrt(g(a * a) * g(b * b) + 1e-12)


def particles(env, s):
    """Masks of the particles, from the reference envelope."""
    v = env.ravel()
    hist, edges = np.histogram(v, 256)
    w = np.cumsum(hist); m = np.cumsum(hist * 0.5 * (edges[1:] + edges[:-1]))
    between = (m[-1] * w / w[-1] - m) ** 2 / np.maximum(w * (w[-1] - w), 1)   # Otsu
    thr = edges[np.argmax(between)]
    lab, n = ndimage.label(env > thr)
    masks = []
    for k in range(1, n + 1):
        m = ndimage.binary_erosion(lab == k, iterations=int(round(ERODE_A / s)))
        edge = m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any()
        if m.sum() * s ** 2 >= MIN_AREA_A2 and not edge:
            masks.append(m)
    return masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=int, required=True)
    ap.add_argument("--ref", default=None, help="reference run (default: seed 0)")
    ap.add_argument("--ref-scan", type=int, default=None,
                    help="also register every particle against this scan's seed-0 run")
    ap.add_argument("--dir", default=None, help="runs directory (default: e1/, quantEM)")
    args = ap.parse_args()
    global E1
    E1 = Path(args.dir) if args.dir else E1
    files = runs(args.scan)
    ref_f = Path(args.ref) if args.ref else next(f for f in files if "_s000_" in f.name)
    ref, s = phases(ref_f)
    last = max(ref)
    ref_env = envelope(ref[last], s)
    masks = particles(ref_env, s)
    G = [analysis.lattice_vectors(lattice(ref[last], s), m, s) for m in masks]
    keep = [i for i, g in enumerate(G) if g is not None]
    QF = {i: analysis.cell_quantile_fn(G[i]) for i in keep}
    print(f"scan {args.scan}: {len(masks)} particles, {len(keep)} with a lattice; "
          f"reference {ref_f.name}")

    refs = {"seed 0": (ref, ref_f.name)}
    if args.ref_scan is not None:
        other = next(f for f in runs(args.ref_scan) if "_s000_" in f.name)
        o, so = phases(other)
        if abs(so - s) > 1e-3 * s:
            raise SystemExit(f"sampling differs: {so} vs {s}")
        o_last = max(o)
        aligned, k = align(o[o_last], ref_env, s)
        print(f"  scan {args.ref_scan} aligned to scan {args.scan} by {k} px")
        refs[f"scan {args.ref_scan}"] = ({it: aligned for it in ref}, other.name)

    rows = []
    for f in files:
        ph_runs, _ = phases(f)
        for it, ph in ph_runs.items():
            ph, _ = align(ph, ref_env, s)
            lat = lattice(ph, s)
            for against, (R, rname) in refs.items():
                if rname == f.name or it not in R:
                    continue
                rl = lattice(R[it], s)
                for i in keep:
                    corr, shift, ratio = analysis.signed_offset(lat, rl, masks[i], s)
                    ws, _ = analysis.wigner_seitz(shift, G[i])
                    q = float(QF[i](np.hypot(*ws)))
                    rows.append(dict(scan=args.scan, run=f.name, iteration=it,
                                     against=against, particle=i,
                                     area_nm2=masks[i].sum() * s ** 2 / 100,
                                     signed_corr=corr, reversed=corr < 0, peak_ratio=ratio,
                                     ws_x=ws[0], ws_y=ws[1], q=q))
    df = pd.DataFrame(rows)
    out = E1 / f"registry_scan{args.scan}.csv"
    df.to_csv(out, index=False)
    for (against, it), g in df.groupby(["against", "iteration"]):
        wrong = g.reversed | (g.q > 0.25)
        per = wrong.groupby(g.particle).mean()
        print(f"  vs {against:8s} it {it:3d}: {len(g.run.unique())} runs, "
              f"{per.gt(0).sum()}/{len(per)} particles differ in some run; "
              f"reversed {g.reversed.mean():.2f}, shifted > 1/4 cell {(g.q > 0.25).mean():.2f}, "
              f"mean |corr| {g.signed_corr.abs().mean():.2f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
