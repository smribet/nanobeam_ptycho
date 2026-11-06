"""Iterative reconstructions of the experimental scans with quantEM, as the notebooks ran
them, and the variations that test whether their ambiguity belongs to the reconstruction.

    python reconstruct_quantem.py --scan 26 --seeds 0 1 2 --iters 50     # as published
    python reconstruct_quantem.py --scan 26 --seeds 0 1 2 --iters 400    # converged?
    python reconstruct_quantem.py --scan 26 --seeds 10 11 --iters 400 \
        --probe-from OUT/scan26_bin2_s000_it400.h5                       # probe frozen

Settings follow scan_<NN>_reconstruct.ipynb: detector binned 2x, CoM rotation -169 deg,
PtychoLite with one slice, a complex object, four mixed-state probe modes, learning rates
5e-3 (object) and 5e-2 (probe), batch 1536, 32 px padding, and the notebooks' fitted C10,
C12, phi12 for each scan.

Unlike the notebooks, the seed is passed to quantEM (`rng=`).  The notebooks seeded numpy
and torch globally, but quantEM draws the initial object from a generator of its own that
is unseeded unless given a seed, so those runs cannot be repeated exactly.

With --probe-from, the probe modes of an earlier run are loaded and held fixed while the
object starts again from a new seed: if lattices still land in different places, the
ambiguity is the object's and not a trade between probe and object.

Writes one h5 per run: object, probe modes, the loss after every iteration, the object's
phase every --snapshot-every iterations, and the settings as attributes.
"""
import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

import quantem as em
from quantem.core import config
from quantem.diffractive_imaging.dataset_models import PtychographyDatasetRaster
from quantem.diffractive_imaging.ptychography_lite import PtychoLite

RAW = Path.home() / "data/nanobeam_ptycho/raw"
ENERGY = 300e3
MRAD_PER_PX = 0.93246184 / 2          # unbinned; the notebooks' value is for bin 2
# convergence semi-angle and the aberrations each notebook fitted (polar_parameters)
SCANS = {
    23: dict(alpha=24.0, C10=-176.7006242519572),
    24: dict(alpha=12.0, C10=-371.1552644221782, C12=10.556224594891535,
             phi12=0.9362357192051852),
    25: dict(alpha=6.0, C10=-1054.3866938255023, C12=36.18675354644571,
             phi12=-1.122986900495411),
    26: dict(alpha=3.0, C10=-1898.4092159168968, C12=44.99354350435432,
             phi12=0.8991894514761393),
    27: dict(alpha=1.5, C10=-4972.5526374207175, C12=35.3278559098221,
             phi12=0.041924051633952575),
}
CONSTRAINTS = {"object": {"identical_slices": True},
               "probe": {"orthogonalize_probe": False}}


def load(scan, binning, crop=None):
    """The scan as quantEM's ptychography dataset, binned `binning` x `binning` on the
    detector.  With `crop` (r0, r1, c0, c1, scan positions) only those positions are read
    (with epie_independent's reader, which bins identically), so an unbinned
    reconstruction of part of the field fits in GPU memory."""
    if crop is None:
        dc = em.io.read_4dstem(str(RAW / f"nanobeam_ptycho_{scan}_master.h5"),
                               file_type="arina")
        if binning > 1:
            dc.bin(binning, axes=(2, 3), modify_in_place=True)
    else:
        import epie_independent as E
        from quantem.core.datastructures import Dataset4dstem
        E.set_binning(binning)
        dc = Dataset4dstem.from_array(E.load_raw(scan, crop), origin=(0, 0, 0, 0),
                                      sampling=(E.STEP, E.STEP, 1.0, 1.0),
                                      units=["A", "A", "pixels", "pixels"])
    dc.sampling[2] = dc.sampling[3] = MRAD_PER_PX * binning
    dc.units[2:] = ["mrad", "mrad"]
    pdset = PtychographyDatasetRaster.from_dataset4dstem(dc)
    pdset.preprocess(com_fit_function="constant", plot_rotation=False, plot_com=False,
                     probe_energy=ENERGY, force_com_rotation=-169,
                     force_com_transpose=False)
    return pdset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=int, required=True, choices=sorted(SCANS))
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--snapshot-every", type=int, default=50)
    ap.add_argument("--bin", type=int, default=2, choices=[1, 2])
    ap.add_argument("--probes", type=int, default=4, help="mixed-state probe modes")
    ap.add_argument("--lr-obj", type=float, default=5e-3)
    ap.add_argument("--lr-probe", type=float, default=5e-2)
    ap.add_argument("--batch", type=int, default=1536)
    ap.add_argument("--probe-from", default=None,
                    help="h5 of an earlier run whose probe modes are loaded and frozen")
    ap.add_argument("--crop", type=int, nargs=4, default=None, metavar=("R0", "R1", "C0", "C1"),
                    help="reconstruct only these scan rows and columns")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config.set_device(0)                  # the worker's CUDA_VISIBLE_DEVICES picks the GPU
    s = SCANS[args.scan]
    polar = {k: v for k, v in s.items() if k != "alpha"}

    t0 = time.time()
    pdset = load(args.scan, args.bin, args.crop)
    print(f"scan {args.scan}: alpha {s['alpha']} mrad, bin {args.bin}, loaded and "
          f"preprocessed in {time.time() - t0:.0f} s", flush=True)
    frozen = None
    if args.probe_from:
        with h5py.File(args.probe_from) as f:
            frozen = torch.from_numpy(f["probe"][()])

    for seed in args.seeds:
        tag = (f"scan{args.scan}_bin{args.bin}_s{seed:03d}_it{args.iters}"
               + ("_fixedprobe" if frozen is not None else "")
               + ("" if args.crop is None else "_crop{}-{}_{}-{}".format(*args.crop)))
        if (out / f"{tag}.h5").exists():
            print(f"  {tag}: exists, skipped")
            continue
        t0 = time.time()
        p = PtychoLite.from_dataset(
            dset=pdset, num_slices=1, obj_type="complex", num_probes=args.probes,
            energy=ENERGY, semiangle_cutoff=s["alpha"], obj_padding_px=(32, 32),
            device="cpu", polar_parameters=polar, rng=seed, verbose=False)
        if frozen is not None:
            # reconstruct(reset=True) restores the initial probe, so the frozen modes go in
            # as that as well
            p.probe_model.initial_probe = frozen
            p.probe_model.probe = frozen
        snaps, done = [], 0
        while done < args.iters:
            n = min(args.snapshot_every, args.iters - done)
            p.reconstruct(num_iters=n, reset=(done == 0), lr_obj=args.lr_obj,
                          learn_probe=frozen is None, lr_probe=args.lr_probe,
                          batch_size=args.batch, constraints=CONSTRAINTS, device="gpu",
                          verbose=False)
            done += n
            # the phase is what gets registered; the complex object is kept at the end only
            snaps.append((done, np.angle(np.asarray(p.obj_cropped)[0]).astype(np.float32)))
            print(f"  {tag}: iteration {done}, loss {p.iter_losses[-1]:.6g}, "
                  f"{time.time() - t0:.0f} s", flush=True)
        if frozen is not None:
            moved = float(np.abs(np.asarray(p.probe) - frozen.numpy()).max())
            print(f"  {tag}: probe held fixed, max |change| {moved:.2e}", flush=True)
        with h5py.File(out / f"{tag}.h5", "w") as f:
            f.create_dataset("obj", data=np.asarray(p.obj_cropped).astype(np.complex64),
                             compression="gzip")
            f.create_dataset("probe", data=np.asarray(p.probe).astype(np.complex64))
            f.create_dataset("sampling", data=np.asarray(p.sampling))
            f.create_dataset("iter_losses", data=np.asarray(p.iter_losses))
            f.create_dataset("snapshot_iters", data=[i for i, _ in snaps])
            f.create_dataset("snapshot_phases", data=np.stack([o for _, o in snaps]),
                             compression="gzip")
            f.attrs.update(scan=args.scan, seed=seed, iters=args.iters, bin=args.bin,
                           probes=args.probes, lr_obj=args.lr_obj, lr_probe=args.lr_probe,
                           batch=args.batch, probe_from=args.probe_from or "",
                           polar=json.dumps(polar), alpha=s["alpha"],
                           runtime_s=round(time.time() - t0, 1))
        print(f"  wrote {tag}.h5", flush=True)


if __name__ == "__main__":
    main()
