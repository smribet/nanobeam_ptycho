"""Cache one diffraction pattern per (region, convergence angle) for the summary figure.

    python make_dp_cache.py

Nine multislice patterns, no reconstruction -- this exists so the figure notebook stays
pure visualisation.  The probe defocus at each angle is the sweep's own, so the patterns
are the data the reconstructions actually saw.

Disk outlines are drawn from each region's *lattice*, taken from the reference file, rather
than from peak detection: at 12 mrad the disks overlap and detection is no longer the right
tool for saying where a reflection is.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import abtem

import sim
import paths

DEFAULT_OUT = paths.OUTPUTS / paths.REFERENCE

# a point inside one wedge, not the five-fold centre where all five lattices superpose
WEDGE_R, WEDGE_DEG = 6.0, 36.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="sweep directory to read runs/_reference.npz from")
    ap.add_argument("--dest", default=None,
                    help="where to write the cache (default: <out>/dp_cache.npz).  The "
                         "shipped caches live in dp_cache/, since they depend only on the "
                         "sample and are shared across sweeps.")
    ap.add_argument("--support", action="store_true")
    ap.add_argument("--energy", type=float, default=sim.ENERGY)
    args = ap.parse_args()
    if args.energy != sim.ENERGY:
        sim.set_energy(args.energy)
    OUT = Path(args.out)
    DEST = Path(args.dest) if args.dest else OUT / "dp_cache.npz"
    DEST.parent.mkdir(parents=True, exist_ok=True)

    abtem.config.set({"dask.lazy": False})
    ref = np.load(OUT / "runs" / "_reference.npz", allow_pickle=True)
    region_G = {k: (None if v is None else np.array(v))
                for k, v in json.loads(str(ref["region_G"])).items()}
    theta0 = 6.0                      # wedge orientation found by lattice sharpness
    try:
        import analysis
        truth = ref["truth"].astype(float)
        theta0 = analysis.find_wedge_offset(truth, float(ref["sampling"]),
                                            sim.PARTICLES["decahedron"], 11.0, 5)
    except Exception as exc:                                   # pragma: no cover
        print(f"  (falling back to theta0 = {theta0}: {exc})")

    th = np.radians(theta0 + WEDGE_DEG)
    dc = sim.PARTICLES["decahedron"]
    POS = {
        "au_111": sim.PARTICLES["au_111"],
        "au_100": sim.PARTICLES["au_100"],
        "decahedron": (dc[0] + WEDGE_R * np.cos(th), dc[1] + WEDGE_R * np.sin(th)),
    }
    LATTICE = {"au_111": "au_111", "au_100": "au_100", "decahedron": "decahedron_w0"}

    atoms = sim.build_sample(support=args.support)
    pot = abtem.Potential(atoms, gpts=(sim.GPTS, sim.GPTS),
                          slice_thickness=float(atoms.cell[2, 2])).build()
    lam = abtem.core.energy.energy2wavelength(sim.ENERGY)
    print(f"{len(atoms)} atoms, lambda = {lam:.4f} A")

    dps, meta = {}, {}
    for angle in sorted(sim.ANGLES):
        defocus = sim.ANGLES[angle]
        probe = abtem.Probe(energy=sim.ENERGY, semiangle_cutoff=angle, defocus=defocus,
                            extent=[sim.L, sim.L], gpts=(sim.GPTS, sim.GPTS))
        for name, p in POS.items():
            dp = probe.multislice(pot, abtem.CustomScan([list(p)]))[0] \
                .diffraction_patterns(max_angle="valid")
            dps[f"{name}_{angle:g}"] = dp.array.astype(np.float32)
            meta[f"{name}_{angle:g}"] = dict(
                maxangle=float(dp.shape[0] / 2 * dp.angular_sampling[0]),
                angle=float(angle), defocus=float(defocus), position=list(p))
        print(f"  {angle:g} mrad (defocus {defocus:.0f} A): {dp.shape} patterns")

    # reflections of each lattice, in mrad, plus which pairs overlap at each angle
    refl, overlap = {}, {}
    for name, key in LATTICE.items():
        G = region_G[key]
        gs = np.array([m * G[0] + n * G[1]
                       for m in range(-4, 5) for n in range(-4, 5)])
        q = np.hypot(gs[:, 0], gs[:, 1])
        gs = gs[q < 0.85]
        refl[name] = gs * lam * 1e3                                # mrad
        for angle in sorted(sim.ANGLES):
            # two disks touch when |dg| < 2 alpha / lambda; count unordered pairs
            d = np.hypot(*(gs[:, None, :] - gs[None, :, :]).transpose(2, 0, 1))
            n_ov = int((np.triu(d, 1) > 0).sum() and
                       ((np.triu(d, 1) > 1e-9) & (np.triu(d, 1) < 2 * angle * 1e-3 / lam)).sum())
            overlap[f"{name}_{angle:g}"] = n_ov

    np.savez_compressed(DEST, meta=json.dumps(meta), overlap=json.dumps(overlap),
                        refl=json.dumps({k: v.tolist() for k, v in refl.items()}),
                        theta0=theta0, lam=lam, **dps)
    print(f"\nwrote {DEST} ({DEST.stat().st_size/1e6:.1f} MB)")
    print("overlapping reflection pairs within the collected band:")
    for name in POS:
        row = "  ".join(f"{a:g} mrad: {overlap[f'{name}_{a:g}']:3d}"
                        for a in sorted(sim.ANGLES))
        print(f"   {name:>12s}   {row}")


if __name__ == "__main__":
    main()
