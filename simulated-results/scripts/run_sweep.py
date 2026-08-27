"""Run N ePIE reconstructions per convergence angle and record the outcome.

    python run_sweep.py --seeds 3                             # test
    python run_sweep.py --seeds 50                            # production
    python run_sweep.py --seeds 50 --seed-start 50 --append   # add 50 more

40 iterations by default: at 6 mrad the defocused probe leaves ePIE still descending at
~3%/iteration after 20, which would otherwise confound "partial coupling" with "not yet
converged".  See convergence_probe.py.

Two kinds of variation are recorded, because below the coupling threshold both are real:

  summary.csv    -- one row per (angle, seed, region) at the final iteration
  drift.csv      -- the same metrics at every snapshot, separating the wander *within*
                    a single run from the scatter *between* seeds
  boundaries.csv -- registry error across each twin plane

Full reconstruction arrays are kept at every snapshot for the first `--array-seeds`
seeds, and at the final iteration only for the rest, which holds the output directory
to a few hundred MB at N = 50.
"""
import argparse, csv, json, shutil, time
from pathlib import Path

import numpy as np

import sim
import analysis
import paths

DEFAULT_OUT = paths.OUTPUTS / "sweep"

FIELDS = ["angle_mrad", "defocus_A", "seed", "coupling", "particle", "signed_corr",
          "reversed", "peak_ratio", "shift_x", "shift_y", "ws_x", "ws_y", "frac_u",
          "frac_v", "has_lattice", "global_corr", "final_error", "n_iter", "runtime_s",
          "amp_dev", "dose", "noise_seed", "occupancy"]
DFIELDS = ["angle_mrad", "seed", "coupling", "iteration", "error", "particle",
           "signed_corr", "reversed", "peak_ratio", "ws_x", "ws_y", "frac_u", "frac_v"]
BFIELDS = ["angle_mrad", "seed", "coupling", "boundary", "registry_err_A",
           "d_shared_A", "frac_of_d", "sign_flip", "ambiguous"]


def global_corr(a, b):
    A, B = a - a.mean(), b - b.mean()
    xc = np.real(np.fft.ifft2(np.fft.fft2(A) * np.conj(np.fft.fft2(B))))
    return float(xc.max() / np.sqrt(np.sum(A ** 2) * np.sum(B ** 2)))


def migrate_csv(path, fields, apply=True):
    """Widen an existing CSV whose header predates a later column.

    Appending with a newer `fields` onto a file written with an older one silently
    produces rows wider than the header, which no CSV reader will parse.  Every column
    added so far has gone on the END of `fields`, so widening the header and padding the
    older rows moves no values.

    Returns (old_header, n_padded) if a migration was needed, else None.  Raises if the
    old header is not a prefix of `fields`, since that means a column was inserted
    mid-list and padding at the end would misalign every row.
    """
    rows = list(csv.reader(open(path)))
    if not rows:
        return None
    hdr, body = rows[0], rows[1:]
    if hdr == list(fields):
        return None
    if hdr != list(fields[:len(hdr)]):
        raise ValueError(f"{path}: header is not a prefix of the current fields -- a "
                         f"column was inserted mid-list, refusing to pad.\n"
                         f"  on disk:  {hdr}\n  expected: {list(fields)}")
    n = sum(1 for r in body if len(r) < len(fields))
    if apply:
        shutil.copy2(path, path.with_suffix(".csv.bak"))
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(fields)
            for r in body:
                w.writerow(r + [""] * (len(fields) - len(r)))
    return hdr, n


def _open(path, fields, append):
    existing = append and path.exists()
    if existing:
        # the file may have been written before a column was added to `fields`
        moved = migrate_csv(path, fields)
        if moved:
            print(f"   migrated {path.name}: header {len(moved[0])} -> {len(fields)} "
                  f"columns, padded {moved[1]} rows (backup alongside)")
    fh = open(path, "a" if append else "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=fields)
    if not existing:
        writer.writeheader()
    return fh, writer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3, help="number of seeds to run")
    ap.add_argument("--seed-start", type=int, default=0,
                    help="first seed; use with --append to extend an existing sweep")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--snapshot-every", type=int, default=10)
    ap.add_argument("--array-seeds", type=int, default=1,
                    help="seeds for which every snapshot's array is stored")
    ap.add_argument("--angles", type=float, nargs="*", default=None)
    ap.add_argument("--backend", default="auto", choices=["auto", "torch", "numpy"])
    ap.add_argument("--append", action="store_true", help="add to existing CSVs")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="output directory; use a fresh one to keep an earlier sweep")
    ap.add_argument("--support", action="store_true",
                    help="add the amorphous carbon film under the particles")
    ap.add_argument("--object", default="complex",
                    choices=["complex", "phase", "mixed"],
                    help="'phase' constrains |O| = 1; 'mixed' uses several mutually "
                         "incoherent object modes, which is the matched model for "
                         "thermal motion")
    ap.add_argument("--modes", type=int, default=4,
                    help="object modes when --object mixed")
    ap.add_argument("--data-dir", default=None,
                    help="load precomputed diffraction data instead of simulating it, e.g. "
                         "the multislice datasets from make_multislice_data.py.  Use with "
                         "--data-kind to pick the forward model.")
    ap.add_argument("--data-kind", default="ms", choices=["ms", "ss"],
                    help="with --data-dir: 'ms' multislice, 'ss' the matched single-slice "
                         "control that went through the same band-limiting")
    ap.add_argument("--dose", type=float, default=None,
                    help="electrons per square Angstrom; omit for noiseless. A fresh "
                         "noise realisation is drawn per seed, so each seed is an "
                         "independent experiment rather than a re-initialised fit to one "
                         "noisy dataset.")
    ap.add_argument("--noise-seed", type=int, default=None,
                    help="fix the noise realisation, so every reconstruction seed inverts "
                         "the SAME noisy dataset and only the initial guess and update "
                         "order differ.  This is the experiment you can actually run on "
                         "real data: it asks whether the reconstruction is unique given "
                         "one measurement, rather than whether the information survives "
                         "the dose.")
    ap.add_argument("--energy", type=float, default=sim.ENERGY,
                    help="accelerating voltage in eV; angles and defoci are rescaled so "
                         "the overlap geometry and the probe are unchanged")
    args = ap.parse_args()

    if args.energy != sim.ENERGY:
        sim.set_energy(args.energy)
        print(f"energy set to {args.energy/1e3:.0f} keV -> "
              + ", ".join(f"{a:g} mrad (defocus {d:.0f} A)"
                          for a, d in sorted(sim.ANGLES.items())))
    if args.angles is None:
        args.angles = sorted(sim.ANGLES)

    OUT = Path(args.out)
    RUNS = OUT / "runs"
    SUMMARY, DRIFT, BOUNDS = OUT / "summary.csv", OUT / "drift.csv", OUT / "boundaries.csv"
    RUNS.mkdir(parents=True, exist_ok=True)
    print(f"backend {args.backend} (device: {sim.best_device()}) | "
          f"{args.iters} iterations, snapshots every {args.snapshot_every}")

    atoms = sim.build_sample(support=args.support)
    obj, proj, sampling = sim.projected_object(atoms)
    pixels = sim.scan_pixels(sampling)
    truth = analysis.band_limit(np.angle(obj), sampling)
    regions = analysis.build_regions(truth, sampling, sim.PARTICLES)
    print(f"{len(atoms)} atoms ({'with' if args.support else 'no'} carbon support) | "
          f"{sim.GPTS}^2 grid | {len(pixels)} positions | {len(regions)} regions")

    np.savez_compressed(
        RUNS / "_reference.npz",
        truth=truth.astype(np.float32),
        projected_potential=proj.astype(np.float32),
        sampling=sampling, L=sim.L, particles=json.dumps(sim.PARTICLES),
        region_names=json.dumps([r["name"] for r in regions]),
        region_G=json.dumps({r["name"]: (None if r["G"] is None else r["G"].tolist())
                             for r in regions}),
        region_masks=np.stack([r["mask"] for r in regions]))

    sfh, swriter = _open(SUMMARY, FIELDS, args.append)
    dfh, dwriter = _open(DRIFT, DFIELDS, args.append)
    bfh, bwriter = _open(BOUNDS, BFIELDS, args.append)

    for angle in args.angles:
        defocus = sim.ANGLES[angle]
        probe = sim.make_probe(angle, defocus)
        if args.data_dir:
            f = Path(args.data_dir) / f"data_{args.data_kind}_a{angle:g}.npy"
            data = np.load(f)
            print(f"   loaded {args.data_kind} data {data.shape} from {f.name}")
        else:
            data = sim.simulate_data(probe, obj, pixels)
        print(f"\nalpha = {angle} mrad, defocus {defocus:.0f} A, "
              f"couples {sim.COUPLING[angle]}")

        for seed in range(args.seed_start, args.seed_start + args.seeds):
            snaps, last_sign, n_flip = [], {}, [0]

            def cb(it, O, err, _s=snaps, _l=last_sign, _n=n_flip):
                if it % args.snapshot_every:
                    return
                ph = analysis.band_limit(np.angle(O), sampling)
                for r in analysis.summarise(ph, truth, sampling, regions):
                    dwriter.writerow({
                        **{k: r[k] for k in ("particle", "signed_corr", "reversed",
                                             "peak_ratio", "ws_x", "ws_y",
                                             "frac_u", "frac_v")},
                        "angle_mrad": angle, "seed": seed,
                        "coupling": sim.COUPLING[angle],
                        "iteration": it, "error": err})
                    prev = _l.get(r["particle"])
                    if prev is not None and prev != r["reversed"]:
                        _n[0] += 1
                    _l[r["particle"]] = r["reversed"]
                if seed < args.array_seeds:
                    _s.append((it, ph.astype(np.float32)))

            _info = {}
            t0 = time.time()
            nseed = args.noise_seed if args.noise_seed is not None else seed
            noisy = sim.apply_dose(data, args.dose, nseed) if args.dose else data
            O, history = sim.epie(probe, noisy, pixels, args.iters, seed,
                                  callback=cb, backend=args.backend,
                                  obj_model=args.object, n_modes=args.modes,
                                  info=_info)
            del noisy
            dt = time.time() - t0

            phase = analysis.band_limit(np.angle(O), sampling)
            gc = global_corr(phase, truth)
            # The object is reconstructed as a general complex function while the truth is
            # pure phase, so ePIE can trade phase against amplitude.  At low kV sigma*V is
            # large and that freedom matters, hence record how far |O| strays from 1.
            amp_dev = float(np.mean(np.abs(np.abs(O) - 1.0)))
            rows = analysis.summarise(phase, truth, sampling, regions)

            payload = dict(phase=phase.astype(np.float32),
                           error_history=np.array(history, dtype=np.float32),
                           angle_mrad=angle, defocus_A=defocus, seed=seed,
                           n_iter=args.iters, sampling=sampling,
                           stats=json.dumps(rows))
            if snaps:
                payload["snapshot_iters"] = np.array([i for i, _ in snaps])
                payload["snapshot_phases"] = np.stack([p for _, p in snaps])
            np.savez_compressed(RUNS / f"a{angle:g}_s{seed:03d}.npz", **payload)

            for r in rows:
                swriter.writerow({**r, "angle_mrad": angle, "defocus_A": defocus,
                                  "seed": seed, "coupling": sim.COUPLING[angle],
                                  "global_corr": gc, "final_error": history[-1],
                                  "n_iter": args.iters, "runtime_s": round(dt, 1),
                                  "amp_dev": round(amp_dev, 5),
                                  "dose": args.dose if args.dose else "",
                                  "noise_seed": nseed if args.dose else "",
                                  "occupancy": ("|".join(f"{o:.4f}" for o
                                                         in _info["occupancy"][:4])
                                                if "occupancy" in _info else "")})
            for b in analysis.twin_boundaries(rows, regions):
                bwriter.writerow({**b, "angle_mrad": angle, "seed": seed,
                                  "coupling": sim.COUPLING[angle]})
            for fh in (sfh, dfh, bfh):
                fh.flush()

            amb = sum(r["peak_ratio"] > 0.8 for r in rows)
            print(f"   seed {seed:3d}: {dt:5.0f}s  err {history[-1]:.4f}  "
                  f"global {gc:.3f}  polarity flips {n_flip[0]}"
                  + (f"  [{amb} ambiguous]" if amb else ""))

    for fh in (sfh, dfh, bfh):
        fh.close()
    size = sum(p.stat().st_size for p in RUNS.glob("*.npz")) / 1e6
    print(f"\n{len(list(RUNS.glob('a*.npz')))} runs stored, {size:.1f} MB")


if __name__ == "__main__":
    main()
