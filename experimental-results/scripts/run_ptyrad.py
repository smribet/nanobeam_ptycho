"""The experimental scans reconstructed with PtyRAD (Lee et al. 2025), a third engine.

    python run_ptyrad.py --scan 26 --orient            # find PtyRAD's scan orientation
    python run_ptyrad.py --scan 26 --seeds 0 1 2 3 --iters 100 --flipT 0 0 0 --rot 169.14

The data are the raw Arina counts binned 2x2 (epie_independent.load_raw), written once per
scan to local scratch (outside the /home quota).  Settings match the other two engines: the
notebooks' convergence angle and C10, C12, phi12 (phi12 in degrees for PtyRAD), four
mixed-state probe modes, one slice, positions fixed on the exact raster (no random initial
displacement, no position refinement), and none of PtyRAD's object constraints; only its
probe-mode orthogonalisation and probe-intensity fixing, both gauge choices, are kept.  The
seed sets PtyRAD's own random_seed, which draws the initial object (flat to 1e-8 rad) and
the batch order.

--orient tries the diffraction flips and scan rotations PtyRAD's conventions could need on
the central 64 x 64 positions and prints the misfit of each: the scan rotation itself is
already known (169.14 deg, from the misfit scan of epie_independent.py), only its sign and
handedness in PtyRAD's frame are not.

Each seed's object phase at every save is converted to the h5 layout of the other engines
(snapshot_phases, iter_losses, ...) in recons/e1/ptyrad/, so register_lattice.py,
make_fig_e1.py and e1_summary.py read it unchanged.
"""
import argparse
import itertools
import time
from pathlib import Path

import h5py
import numpy as np
import yaml

import epie_independent as E

SCRATCH = Path("/tmp/gvarnavides_ptyrad")
OUT = Path.home() / "data/nanobeam_ptycho/recons/e1/ptyrad"


def data_file(scan, crop=None):
    """Counts as PtyRAD takes them, (N, E.N, E.N) float32 at the binning set by
    E.set_binning, written once; `crop` = (r0, r1, c0, c1) in scan positions."""
    tag = "" if crop is None else "_crop{}-{}_{}-{}".format(*crop)
    f = SCRATCH / "data" / f"scan{scan}_bin{E.BIN}{tag}.h5"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        d = E.load_raw(scan, crop)
        with h5py.File(f.with_suffix(".tmp"), "w") as h:
            h.create_dataset("dp", data=d.reshape(-1, *d.shape[2:]))
        f.with_suffix(".tmp").rename(f)
    return f


def params(scan, seed, iters, flipT, rot, crop=None, save_every=25, batch=32, out=None,
           cosine=False):
    s = E.SCANS[scan]
    ab = {"C10": s["C10"]}
    if "C12" in s:
        ab.update(C12=s["C12"], phi12=float(np.rad2deg(s["phi12"])))
    n = 512 if crop is None else crop[1] - crop[0]
    off = {"start_iter": None, "step": 1, "end_iter": None}
    return {
        "init_params": {
            "random_seed": seed,
            "probe_kv": 300, "probe_conv_angle": s["alpha"], "probe_aberrations": ab,
            # the detector's own calibration, rather than PtyRAD's default of inferring it
            # from the nominal angle and the fitted disk radius (1.7% off at 24 mrad)
            "meas_calibration": {"mode": "da", "value": E.MRAD_PER_PX},
            "meas_Npix": E.N, "pos_N_scan_slow": n, "pos_N_scan_fast": n,
            "pos_scan_step_size": E.STEP, "probe_pmode_max": 4,
            "obj_Nlayer": 1, "obj_slice_thickness": 20,
            "meas_flipT": list(flipT),
            "pos_scan_affine": [1, 0, rot, 0], "pos_scan_rand_std": None,
            "meas_params": {"path": str(data_file(scan, crop)), "key": "dp"},
        },
        "model_params": {**({"scheduler_params": {"name": "CosineAnnealingLR",
                                                  "configs": {"T_max": iters}}}
                            if cosine else {}),
                         "update_params": {
            "obja": {"start_iter": 1, "lr": 5.0e-4, "end_iter": None},
            "objp": {"start_iter": 1, "lr": 5.0e-4, "end_iter": None},
            "probe": {"start_iter": 1, "lr": 1.0e-4, "end_iter": None},
            "probe_pos_shifts": {"start_iter": None, "lr": 5.0e-4, "end_iter": None}}},
        "loss_params": {"loss_single": {"state": True, "weight": 1.0, "dp_pow": 0.5}},
        "constraint_params": {
            "ortho_pmode": {"start_iter": 1, "step": 1, "end_iter": None},
            "fix_probe_int": {"start_iter": 1, "step": 1, "end_iter": None},
            "obj_zblur": off, "obja_thresh": off, "obj_rblur": off, "objp_postiv": off},
        "recon_params": {
            "NITER": iters, "BATCH_SIZE": {"size": batch, "grad_accumulation": 1},
            "GROUP_MODE": "random", "SAVE_ITERS": save_every,
            "output_dir": str(out or SCRATCH / "output" / f"scan{scan}"),
            "prefix_time": False, "prefix": f"s{seed:03d}", "save_result": ["model"],
            "selected_figs": [], "copy_params": True},
    }


def run(p, gpu=0):
    from ptyrad.params import load_params
    from ptyrad.runtime.device import set_gpu_device
    from ptyrad.solver import PtyRADSolver
    f = Path(p["recon_params"]["output_dir"]) / f"{p['recon_params']['prefix']}.yaml"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(yaml.safe_dump(p, sort_keys=False))
    solver = PtyRADSolver(load_params(str(f), validate=True), device=set_gpu_device(gpuid=gpu),
                          seed=p["init_params"]["random_seed"])
    solver.run()
    return solver.reconstruct_results


def collect(out, seed):
    """(iterations, phases, dx) of every save of one seed, read with PtyRAD's own reader;
    the phase summed over slices, of the first object mode, cropped to the scanned area."""
    from ptyrad.analysis import Analyzer
    saved = sorted(out.glob(f"s{seed:03d}*/model_iter*.hdf5"),
                   key=lambda f: int(f.stem.split("iter")[-1]))
    its, phs, dx = [], [], None
    for f in saved:
        a = Analyzer(str(f))
        p = np.asarray(a.get_object_phase(fov="crop"))       # (omode, Nz, Ny, Nx)
        phs.append(p[0].sum(0).astype(np.float32))
        dx = float(a.dx)
        its.append(int(f.stem.split("iter")[-1]))
    n = min(min(p.shape) for p in phs)                       # square, as the scoring assumes
    return its, [p[:n, :n] for p in phs], dx


def losses(model):
    for name in ("loss_iters", "iter_losses", "losses"):
        v = getattr(model, name, None)
        if v is not None:
            return np.array([x[1] if isinstance(x, (tuple, list)) else x for x in v], float)
    return np.array([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=int, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--orient", action="store_true")
    ap.add_argument("--flipT", type=int, nargs=3, default=[0, 0, 0])
    ap.add_argument("--rot", type=float, default=169.14)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--cosine", action="store_true",
                    help="decay the learning rates to zero over the run (cosine annealing)")
    ap.add_argument("--bin", type=int, default=2, choices=[1, 2],
                    help="detector binning: 2 as published (96 x 96), 1 as recorded")
    ap.add_argument("--crop", type=int, nargs=4, default=None, metavar=("R0", "R1", "C0", "C1"),
                    help="reconstruct only these scan rows and columns")
    args = ap.parse_args()
    E.set_binning(args.bin)

    if args.orient:
        crop = (224, 288, 224, 288)
        res = {}
        for flipT, rot in itertools.product(([0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]),
                                            (169.14, -169.14, 10.86, -10.86)):
            t0 = time.time()
            m = run(params(args.scan, 0, 10, flipT, rot, crop=crop, save_every=None,
                           out=SCRATCH / "orient" / f"scan{args.scan}"))
            L = losses(m)
            res[(tuple(flipT), rot)] = L[-1] if len(L) else np.nan
            print(f"  flipT {flipT} rot {rot:7.2f}: misfit {res[(tuple(flipT), rot)]:.6g} "
                  f"({time.time() - t0:.0f} s)", flush=True)
        best = min(res, key=lambda k: res[k])
        print(f"best: flipT {list(best[0])} rot {best[1]}")
        return

    # cropped runs go apart, so seed comparisons of the full field never mix the two
    dest = (OUT.with_name("ptyrad_crop") if args.crop is not None
            else OUT.with_name("ptyrad_cosine") if args.cosine else OUT)
    dest.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        tag = (f"scan{args.scan}_bin{args.bin}_s{seed:03d}_it{args.iters}"
               + ("" if args.crop is None else "_crop{}-{}_{}-{}".format(*args.crop))
               + ("_cosine" if args.cosine else ""))
        if (dest / f"{tag}.h5").exists():
            print(f"  {tag}: exists, skipped")
            continue
        t0 = time.time()
        out = SCRATCH / "output" / tag
        m = run(params(args.scan, seed, args.iters, args.flipT, args.rot, batch=args.batch,
                       crop=args.crop, out=out, cosine=args.cosine))
        its, phs, dx = collect(out, seed)
        with h5py.File(dest / f"{tag}.h5", "w") as f:
            f.create_dataset("snapshot_phases", data=np.stack(phs), compression="gzip")
            f.create_dataset("snapshot_iters", data=its)
            f.create_dataset("iter_losses", data=losses(m))
            f.create_dataset("sampling", data=[dx or 0.2199, dx or 0.2199])
            f.attrs.update(scan=args.scan, seed=seed, iters=args.iters, engine="ptyrad",
                           flipT=args.flipT, rot=args.rot, batch=args.batch, bin=args.bin,
                           crop=str(args.crop),
                           runtime_s=round(time.time() - t0, 1))
        print(f"  wrote {tag}.h5 ({len(its)} snapshots, {time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
