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
import argparse, csv, json, shutil, sys, time
from pathlib import Path

import numpy as np

import sim
import analysis
import paths

DEFAULT_OUT = paths.OUTPUTS / "sweep"

FIELDS = ["angle_mrad", "defocus_A", "seed", "coupling", "particle", "signed_corr",
          "reversed", "peak_ratio", "shift_x", "shift_y", "ws_x", "ws_y", "frac_u",
          "frac_v", "has_lattice", "global_corr", "final_error", "n_iter", "runtime_s",
          "amp_dev", "dose", "noise_seed", "occupancy",
          # revision: how the run was set up, and the ground truth's own misfit
          "init", "truth_error", "probe_update", "probe_shift_x", "probe_shift_y",
          "scan_jitter", "det_bin", "det_crop", "carbon_thickness", "scale", "field",
          "carbon_seed"]
DFIELDS = ["angle_mrad", "seed", "coupling", "iteration", "error", "particle",
           "signed_corr", "reversed", "peak_ratio", "ws_x", "ws_y", "frac_u", "frac_v"]
BFIELDS = ["angle_mrad", "seed", "coupling", "boundary", "registry_err_A",
           "d_shared_A", "frac_of_d", "sign_flip", "ambiguous"]


def global_corr(a, b):
    A, B = a - a.mean(), b - b.mean()
    xc = np.real(np.fft.ifft2(np.fft.fft2(A) * np.conj(np.fft.fft2(B))))
    return float(xc.max() / np.sqrt(np.sum(A ** 2) * np.sum(B ** 2)))


def probe_shift(P_rec, P_true):
    """Integer-pixel translation of a refined probe relative to the true one.

    Shifting probe and object together leaves every pattern unchanged, so in blind
    ptychography the object may come back translated by exactly this amount.  Undoing it
    before scoring separates that trivial coordinate freedom (reviewer 2, comment 5) from
    a lattice that moved relative to its own particle.
    """
    a, b = np.abs(P_rec) ** 2, np.abs(P_true) ** 2
    xc = np.real(np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))))
    n = a.shape[0]
    k = np.unravel_index(np.argmax(xc), xc.shape)
    return tuple(int(v if v < n // 2 else v - n) for v in k)


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
    # --- revision options
    ap.add_argument("--carbon-thickness", type=float, default=None,
                    help="with --support: a physical a-C film this many Angstrom thick; "
                         "omit for the published folded film")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="particle linear size relative to the published sample")
    ap.add_argument("--particles", nargs="*", default=None,
                    help="subset of particles to include; a single one is centred in "
                         "the field, which leaves room to enlarge it with --scale")
    ap.add_argument("--field", type=float, default=None,
                    help="enlarge the periodic field to this many Angstrom (finer "
                         "reciprocal sampling, same scan and specimen)")
    ap.add_argument("--scan-jitter", type=float, default=0.0,
                    help="known random offset of every probe position, +-Angstrom")
    ap.add_argument("--jitter-seed", type=int, default=0)
    ap.add_argument("--init", default="random",
                    choices=["random", "truth", "reversed", "shifted"],
                    help="initial object: random, the ground truth, or the ground truth "
                         "with every region's lattice reversed or shifted by half a cell")
    ap.add_argument("--probe-update", action="store_true",
                    help="refine the probe as well (blind ptychography)")
    ap.add_argument("--probe-defocus-error", type=float, default=0.0,
                    help="with --probe-update: defocus error of the starting probe, A")
    ap.add_argument("--probe-astig-error", type=float, default=0.0,
                    help="with --probe-update: two-fold astigmatism C12 of the starting "
                         "probe, A, at 30 deg to the scan axes")
    ap.add_argument("--bin", type=int, default=1, help="detector binning")
    ap.add_argument("--crop-px", type=int, default=0,
                    help="square detector this many (binned) pixels wide; 0 = whole grid")
    ap.add_argument("--carbon-seed", type=int, default=10,
                    help="realisation of the carbon film (10 = the published and R0 one)")
    ap.add_argument("--place", nargs=3, action="append", default=[],
                    metavar=("PARTICLE", "X", "Y"),
                    help="move a particle's centre to (X, Y) Angstrom; repeatable")
    ap.add_argument("--phase-smooth", type=float, default=None, metavar="A",
                    help="score the phase relative to a copy of the object smoothed over "
                         "this many Angstrom, so a thick film cannot wrap it")
    ap.add_argument("--store-object", action="store_true",
                    help="also store the complex object (every mode, for --object mixed) "
                         "and, with --probe-update, the refined probe")
    args = ap.parse_args()
    print("argv: " + " ".join(sys.argv), flush=True)

    if args.field:
        sim.set_field(args.field)
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

    if args.particles:
        sim.PARTICLES = {p: sim.PARTICLES[p] for p in args.particles}
        if len(args.particles) == 1:
            sim.PARTICLES = {args.particles[0]: (sim.L / 2, sim.L / 2)}
    for name, x, y in args.place:
        sim.PARTICLES[name] = (float(x), float(y))
    atoms = sim.build_sample(support=args.support, carbon_thickness=args.carbon_thickness,
                             scale=args.scale, particles=args.particles,
                             carbon_seed=args.carbon_seed)
    obj, proj, sampling = sim.projected_object(atoms)
    pixels = sim.scan_pixels(sampling, args.scan_jitter, args.jitter_seed)
    truth = analysis.object_phase(obj, sampling, args.phase_smooth)
    regions = analysis.build_regions(truth, sampling, sim.PARTICLES,
                                     radius=11.0 * args.scale,
                                     twin_radius=11.0 * args.scale)
    print(f"{len(atoms)} atoms ({'with' if args.support else 'no'} carbon support) | "
          f"{sim.GPTS}^2 grid | {len(pixels)} positions | {len(regions)} regions",
          flush=True)
    O0 = {"random": None, "truth": obj}.get(args.init)
    if args.init in ("reversed", "shifted"):
        O0 = sim.gauge_transformed(obj, sampling, regions, args.init)
    nb = sim.GPTS // args.bin
    det_mask = sim.detector_mask(nb, args.crop_px)
    if args.data_dir:
        # the stored patterns must have been made for this exact scan and specimen
        meta = json.loads((Path(args.data_dir) / "meta.json").read_text())
        for k, v, default in (("scan_jitter", args.scan_jitter, 0.0),
                              ("carbon_thickness", args.carbon_thickness, None),
                              ("carbon_seed", args.carbon_seed, 10)):
            if meta.get(k, default) != v:
                raise SystemExit(f"{args.data_dir}: {k}={meta.get(k)} but --{k} {v}")

    np.savez_compressed(
        RUNS / "_reference.npz",
        truth=truth.astype(np.float32),
        projected_potential=proj.astype(np.float32),
        sampling=sampling, L=sim.L, particles=json.dumps(sim.PARTICLES),
        region_names=json.dumps([r["name"] for r in regions]),
        region_G=json.dumps({r["name"]: (None if r["G"] is None else r["G"].tolist())
                             for r in regions}),
        region_masks=np.stack([r["mask"] for r in regions]))

    truth_err = {}      # (angle, noise draw) -> misfit of the ground truth itself
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
                ph = analysis.object_phase(O, sampling, args.phase_smooth)
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
            if args.bin > 1 or det_mask is not None:
                noisy = [sim.bin_pattern(p, args.bin) for p in noisy]
                if det_mask is not None:
                    noisy = [p * det_mask for p in noisy]
            key = (angle, nseed if args.dose else None)
            if key not in truth_err:
                truth_err[key] = sim.misfit(probe, obj, noisy, pixels, args.bin, det_mask)
            p_start = (sim.make_probe(angle, defocus + args.probe_defocus_error,
                                      C12=args.probe_astig_error, phi12=np.pi / 6)
                       if args.probe_update else probe)
            O, history = sim.epie(p_start, noisy, pixels, args.iters, seed,
                                  callback=cb, backend=args.backend,
                                  obj_model=args.object, n_modes=args.modes,
                                  info=_info, O0=O0, probe_update=args.probe_update,
                                  det_bin=args.bin, det_mask=det_mask)
            del noisy
            dt = time.time() - t0
            shift = (0, 0)
            if args.probe_update:
                shift = probe_shift(_info["probe"], probe)
                O = np.roll(O, (-shift[0], -shift[1]), axis=(0, 1))

            phase = analysis.object_phase(O, sampling, args.phase_smooth)
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
            if args.store_object:
                payload["object"] = np.asarray(_info.get("modes", O), dtype=np.complex64)
                if args.probe_update:
                    payload["probe"] = _info["probe"].astype(np.complex64)
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
                                                if "occupancy" in _info else ""),
                                  "init": args.init, "truth_error": truth_err[key],
                                  "probe_update": args.probe_update,
                                  "probe_shift_x": shift[0] * sampling,
                                  "probe_shift_y": shift[1] * sampling,
                                  "scan_jitter": args.scan_jitter, "det_bin": args.bin,
                                  "det_crop": args.crop_px,
                                  "carbon_thickness": ("" if args.carbon_thickness is None
                                                       else args.carbon_thickness),
                                  "scale": args.scale, "field": sim.L,
                                  "carbon_seed": args.carbon_seed})
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
