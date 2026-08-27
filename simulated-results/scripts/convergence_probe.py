"""Track reconstruction quality against ePIE iteration, for each convergence angle.

Answers whether a given angle is genuinely worse or merely under-converged, and whether
the per-region polarity settles or keeps moving.

    python convergence_probe.py --iters 60
    python convergence_probe.py --iters 150 --seeds 2 --support --energy 30e3 \
        --out ../outputs/ss_carbon_complex_inf_30kv

At lower voltage the interaction constant is ~2.4x larger, so the object is a stronger
phase object and the fixed ePIE step may need more iterations to reach the same place.
Comparing two energies at a fixed iteration count is only fair if both have converged, so
run this before trusting such a comparison.
"""
import argparse, csv, json
from pathlib import Path

import numpy as np

import sim
import analysis
import paths

DEFAULT_OUT = paths.OUTPUTS / "sweep"


def global_corr(a, b):
    A, B = a - a.mean(), b - b.mean()
    xc = np.real(np.fft.ifft2(np.fft.fft2(A) * np.conj(np.fft.fft2(B))))
    return float(xc.max() / np.sqrt(np.sum(A ** 2) * np.sum(B ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--support", action="store_true")
    ap.add_argument("--energy", type=float, default=sim.ENERGY)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if args.energy != sim.ENERGY:
        sim.set_energy(args.energy)
    OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.energy/1e3:.0f}kv"

    atoms = sim.build_sample(support=args.support)
    obj, proj, sampling = sim.projected_object(atoms)
    pixels = sim.scan_pixels(sampling)
    truth = analysis.band_limit(np.angle(obj), sampling)
    regions = analysis.build_regions(truth, sampling, sim.PARTICLES)
    print(f"{len(atoms)} atoms | {args.energy/1e3:.0f} keV | "
          f"angles {sorted(sim.ANGLES)} | {len(regions)} regions")

    rows, track = [], {}
    for angle in sorted(sim.ANGLES):
        probe = sim.make_probe(angle, sim.ANGLES[angle])
        data = sim.simulate_data(probe, obj, pixels)
        print(f"\nalpha = {angle:g} mrad, defocus {sim.ANGLES[angle]:.0f} A")
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            hist = []

            def cb(it, O, err, _h=hist, _a=angle, _s=seed):
                if it % args.every and it != 1:
                    return
                ph = analysis.band_limit(np.angle(O), sampling)
                st = analysis.summarise(ph, truth, sampling, regions)
                gc = global_corr(ph, truth)
                amb = sum(r["peak_ratio"] > 0.8 for r in st)
                nrev = sum(r["reversed"] for r in st)
                _h.append(dict(iteration=it, error=err, global_corr=gc,
                               ambiguous=amb, n_reversed=nrev))
                rows.append(dict(energy_kv=args.energy/1e3, angle_mrad=_a, seed=_s,
                                 iteration=it, error=err, global_corr=gc,
                                 ambiguous=amb, n_reversed=nrev,
                                 mean_abs_corr=float(np.mean(
                                     [abs(r["signed_corr"]) for r in st]))))
                if it % (args.every * 5) == 0 or it == 1:
                    print(f"   seed {_s} it {it:4d}  err {err:.4f}  global {gc:.3f}  "
                          f"|corr| {rows[-1]['mean_abs_corr']:.3f}  amb {amb}  rev {nrev}")

            sim.epie(probe, data, pixels, args.iters, seed, callback=cb)
            track[f"{angle:g}_s{seed}"] = hist

    csv_path = OUT / f"convergence_{tag}.csv"
    with open(csv_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
    (OUT / f"convergence_{tag}.json").write_text(json.dumps(track, indent=1))
    print(f"\nwrote {csv_path}")

    # has anything still been moving over the last quarter of the run?
    print("\nchange over the final quarter of the run:")
    print(f"  {'angle':>8} {'seed':>5} {'d_err':>9} {'d_global':>9} {'d|corr|':>9}")
    for (a, s_), grp in _by(rows):
        tail = [r for r in grp if r["iteration"] > 0.75 * args.iters]
        if len(tail) < 2:
            continue
        print(f"  {a:8g} {s_:5d} {tail[-1]['error']-tail[0]['error']:+9.4f} "
              f"{tail[-1]['global_corr']-tail[0]['global_corr']:+9.4f} "
              f"{tail[-1]['mean_abs_corr']-tail[0]['mean_abs_corr']:+9.4f}")


def _by(rows):
    keys = sorted({(r["angle_mrad"], r["seed"]) for r in rows})
    return [(k, [r for r in rows if (r["angle_mrad"], r["seed"]) == k]) for k in keys]


if __name__ == "__main__":
    main()
