"""Pre-compute diffraction data with the multislice forward model.

    python make_multislice_data.py --out ../outputs/forward_data/ms_carbon

Writes one array per angle, shape (n_positions, GPTS, GPTS) float32, on the same k-grid the
single-slice reconstruction uses.  Generating once and reusing keeps the reconstruction
sweeps cheap and guarantees every seed inverts identical data.

Both models are written:

  data_ms_a{angle}.npy   multislice, transmission/propagation over sliced potential
  data_ss_a{angle}.npy   the SAME potential collapsed to one slice

The single-slice version goes through the identical oversample / antialias / crop chain, so
comparing the two isolates dynamical scattering from the band-limiting.  Comparing against
the existing 340-grid sweeps would confound the two, because those were never band-limited.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import abtem

import sim
import multislice as ms
import paths

CHUNK = 32          # probe positions held on the oversampled grid at once


def build(angle, defocus, atoms, gpts_o, chunk=CHUNK, phonons=0, seed=0):
    sigma_e = abtem.core.energy.energy2sigma(sim.ENERGY)
    prop, aper = ms.propagator(gpts_o, sim.L / gpts_o, ms.SLICE_THICKNESS, sim.ENERGY)
    pixels = sim.scan_pixels(sim.L / sim.GPTS)
    pos = [(i * sim.L / sim.GPTS, j * sim.L / sim.GPTS) for i, j in pixels]

    configs = ([atoms] if phonons == 0
               else [ms.displaced(atoms, np.random.default_rng(seed + c))
                     for c in range(phonons)])
    n_cfg = len(configs)
    out_ms = np.zeros((len(pos), sim.GPTS, sim.GPTS), dtype=np.float32)
    out_ss = np.zeros_like(out_ms)

    for c, cfg in enumerate(configs):
        V, _ = ms.slice_potential(cfg, gpts_o)
        Vproj = V.sum(0)[None]
        for lo in range(0, len(pos), chunk):
            P = ms.probe_batch(pos[lo:lo + chunk], angle, defocus, gpts_o, sim.ENERGY)
            for tgt, pot in ((out_ms, V), (out_ss, Vproj)):
                dp = ms.multislice_batch(P, pot, prop, sigma_e, aper)
                tgt[lo:lo + chunk] += ms.crop_to_recon_grid(
                    dp, sim.GPTS).astype(np.float32)
        print(f"    config {c + 1}/{n_cfg} done", flush=True)
    return out_ms / n_cfg, out_ss / n_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(paths.FORWARD_DATA / "ms_carbon"))
    ap.add_argument("--angles", type=float, nargs="*", default=None)
    ap.add_argument("--phonons", type=int, default=0,
                    help="frozen-phonon configurations; 0 = static lattice")
    ap.add_argument("--support", action="store_true", default=True)
    args = ap.parse_args()

    abtem.config.set({"dask.lazy": False})
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    gpts_o = sim.GPTS * ms.OVERSAMPLE
    atoms = sim.build_sample(support=args.support)
    angles = args.angles or sorted(sim.ANGLES)
    print(f"{len(atoms)} atoms | simulate {gpts_o}^2 ({sim.L/gpts_o:.3f} A), "
          f"antialias at k_max/2, crop to {sim.GPTS}^2 | {args.phonons or 'no'} phonons")

    meta = dict(energy=sim.ENERGY, gpts=sim.GPTS, gpts_sim=gpts_o, L=sim.L,
                slice_thickness=ms.SLICE_THICKNESS, phonons=args.phonons,
                support=bool(args.support), angles={})
    for a in angles:
        d = sim.ANGLES[a]
        t0 = time.time()
        print(f"\nalpha = {a:g} mrad, defocus {d:.0f} A")
        dp_ms, dp_ss = build(a, d, atoms, gpts_o, phonons=args.phonons)
        np.save(out / f"data_ms_a{a:g}.npy", dp_ms)
        np.save(out / f"data_ss_a{a:g}.npy", dp_ss)
        meta["angles"][f"{a:g}"] = dict(defocus=d, seconds=round(time.time() - t0, 1))
        print(f"  wrote {dp_ms.shape} in {time.time()-t0:.0f}s "
              f"({dp_ms.nbytes/1e6:.0f} MB each)")

    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"\nwrote {out}/meta.json")


if __name__ == "__main__":
    main()
