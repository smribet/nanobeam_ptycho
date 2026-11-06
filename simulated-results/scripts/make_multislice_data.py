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

Incoherent averages, each optional and all multiplicative in cost:

  --phonons N         frozen-phonon configurations (thermal motion)
  --source-sigma s    rms size of the demagnified source, Angstrom (spatial coherence)
  --focal-sigma f     rms focal spread, Angstrom (temporal coherence)

The torch backend runs the same chain on the GPU; `multislice.numpy_vs_torch` checks the
two against each other.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import abtem

import sim
import multislice as ms
import paths

CHUNK = 32          # probe positions held on the oversampled grid at once (numpy)
CHUNK_TORCH = 128   # same, on the GPU


def build(angle, defocus, atoms, gpts_o, chunk=CHUNK, phonons=0, seed=0, jitter=0.0,
          jitter_seed=0, source_sigma=0.0, focal_sigma=0.0, backend="numpy", ss=True,
          n_source=5, n_focal=5):
    sigma_e = abtem.core.energy.energy2sigma(sim.ENERGY)
    prop, aper = ms.propagator(gpts_o, sim.L / gpts_o, ms.SLICE_THICKNESS, sim.ENERGY)
    pixels = sim.scan_pixels(sim.L / sim.GPTS, jitter, jitter_seed)
    pos = [(i * sim.L / sim.GPTS, j * sim.L / sim.GPTS) for i, j in pixels]
    samples = ms.coherence_samples(source_sigma, focal_sigma, n_source, n_focal)

    configs = ([atoms] if phonons == 0
               else [ms.displaced(atoms, np.random.default_rng(seed + c))
                     for c in range(phonons)])
    n_cfg = len(configs)
    out_ms = np.zeros((len(pos), sim.GPTS, sim.GPTS), dtype=np.float32)
    out_ss = np.zeros_like(out_ms)

    if backend == "torch":
        import torch
        dev = ms.device()
        prop_t, aper_t = torch.from_numpy(prop).to(dev), torch.from_numpy(aper).to(dev)
        probes = {}
        for _, _, df, _ in samples:
            if df not in probes:
                probes[df] = torch.from_numpy(
                    ms.centred_probe(angle, defocus + df, gpts_o, sim.ENERGY)).to(dev)

    for c, cfg in enumerate(configs):
        V, _ = ms.slice_potential(cfg, gpts_o)
        Vproj = V.sum(0)[None]
        if backend == "torch":
            pots = ((out_ms, ms.transmission_torch(V, sigma_e, dev)),
                    (out_ss, ms.transmission_torch(Vproj, sigma_e, dev)))[:2 if ss else 1]
            for lo in range(0, len(pos), chunk):
                acc = [torch.zeros((min(chunk, len(pos) - lo), sim.GPTS, sim.GPTS),
                                   device=dev) for _ in pots]
                for dx, dy, df, w in samples:
                    P = ms.probe_batch_torch(
                        probes[df], [(x + dx, y + dy) for x, y in pos[lo:lo + chunk]],
                        sim.L / gpts_o)
                    for a, (_, T) in zip(acc, pots):
                        a += w * ms.crop_to_recon_grid_torch(
                            ms.multislice_batch_torch(P, T, prop_t, aper_t), sim.GPTS)
                for a, (tgt, _) in zip(acc, pots):
                    tgt[lo:lo + chunk] += a.cpu().numpy()
        else:
            for lo in range(0, len(pos), chunk):
                for dx, dy, df, w in samples:
                    P = ms.probe_batch([(x + dx, y + dy) for x, y in pos[lo:lo + chunk]],
                                       angle, defocus + df, gpts_o, sim.ENERGY)
                    for tgt, pot in ((out_ms, V), (out_ss, Vproj))[:2 if ss else 1]:
                        dp = ms.multislice_batch(P, pot, prop, sigma_e, aper)
                        tgt[lo:lo + chunk] += w * ms.crop_to_recon_grid(
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
    ap.add_argument("--no-support", dest="support", action="store_false")
    ap.add_argument("--carbon-thickness", type=float, default=None,
                    help="Angstrom; a physical a-C film (sim.amorphous_carbon_film). "
                         "Omit for the published folded film")
    ap.add_argument("--scan-jitter", type=float, default=0.0,
                    help="known random offset of each probe position, +-Angstrom")
    ap.add_argument("--jitter-seed", type=int, default=0)
    ap.add_argument("--source-sigma", type=float, default=0.0,
                    help="rms source size, Angstrom (partial spatial coherence)")
    ap.add_argument("--focal-sigma", type=float, default=0.0,
                    help="rms focal spread, Angstrom (partial temporal coherence)")
    ap.add_argument("--n-source", type=int, default=5,
                    help="source quadrature points per axis")
    ap.add_argument("--n-focal", type=int, default=5, help="focal-spread quadrature points")
    ap.add_argument("--backend", default="numpy", choices=["numpy", "torch"])
    ap.add_argument("--no-ss", dest="ss", action="store_false",
                    help="skip the single-slice twin, halving time and disk")
    ap.add_argument("--carbon-seed", type=int, default=10,
                    help="realisation of the carbon film (10 = the published and R0 one)")
    args = ap.parse_args()

    abtem.config.set({"dask.lazy": False})
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    gpts_o = sim.GPTS * ms.OVERSAMPLE
    atoms = sim.build_sample(support=args.support, carbon_thickness=args.carbon_thickness,
                             carbon_seed=args.carbon_seed)
    angles = args.angles or sorted(sim.ANGLES)
    chunk = CHUNK_TORCH if args.backend == "torch" else CHUNK
    print(f"{len(atoms)} atoms | simulate {gpts_o}^2 ({sim.L/gpts_o:.3f} A), "
          f"antialias at k_max/2, crop to {sim.GPTS}^2 | {args.phonons or 'no'} phonons | "
          f"source {args.source_sigma} A, focal {args.focal_sigma} A | {args.backend}",
          flush=True)

    meta = dict(energy=sim.ENERGY, gpts=sim.GPTS, gpts_sim=gpts_o, L=sim.L,
                slice_thickness=ms.SLICE_THICKNESS, phonons=args.phonons,
                support=bool(args.support), carbon_thickness=args.carbon_thickness,
                carbon_seed=args.carbon_seed, scan_jitter=args.scan_jitter, jitter_seed=args.jitter_seed,
                source_sigma=args.source_sigma, focal_sigma=args.focal_sigma,
                n_source=args.n_source, n_focal=args.n_focal,
                backend=args.backend, angles={})
    for a in angles:
        d = sim.ANGLES[a]
        t0 = time.time()
        print(f"\nalpha = {a:g} mrad, defocus {d:.0f} A", flush=True)
        dp_ms, dp_ss = build(a, d, atoms, gpts_o, chunk=chunk, phonons=args.phonons,
                             jitter=args.scan_jitter, jitter_seed=args.jitter_seed,
                             source_sigma=args.source_sigma, focal_sigma=args.focal_sigma,
                             backend=args.backend, ss=args.ss, n_source=args.n_source,
                             n_focal=args.n_focal)
        np.save(out / f"data_ms_a{a:g}.npy", dp_ms)
        if args.ss:
            np.save(out / f"data_ss_a{a:g}.npy", dp_ss)
        meta["angles"][f"{a:g}"] = dict(defocus=d, seconds=round(time.time() - t0, 1))
        print(f"  wrote {dp_ms.shape} in {time.time()-t0:.0f}s "
              f"({dp_ms.nbytes/1e6:.0f} MB each)", flush=True)
        # written after every angle, so a partial run still records what it produced
        (out / "meta.json").write_text(json.dumps(meta, indent=1))

    print(f"\nwrote {out}/meta.json")


if __name__ == "__main__":
    main()
