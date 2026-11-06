"""Per-particle gauge search as W. de Kleijne proposed it: re-converge from every candidate.

    python gauge_search_full.py --out DIR --angles 3 --seed 4 --region au_111
    python gauge_search_full.py --out DIR --angles 3 --seed 2 --region au_100 \
        --dose 1e5 --noise-seed 0

The converged reconstruction of one initial guess (run exactly as run_sweep.py runs it) is
edited in one region at a time: the region's lattice is reversed (Eq. 6) and/or translated
by half a lattice vector along a1, a2, or both (Eq. 5), its envelope unchanged, and a full
reconstruction -- every probe position, the same number of iterations -- is started from
the edited object and allowed to converge.  The unedited object is re-converged the same
way, so all eight candidates get identical treatment.  If the correct registry is the
global minimum, the candidate whose re-converged misfit is lowest should be the correct one.

Unlike gauge_search.py, which relaxed each candidate briefly on the positions near the
region only, nothing here is local or truncated.  Writes candidates.csv: one row per
candidate, with its final misfit and the edited region's signed correlation and offset
against the ground truth after re-convergence, and the same for every other region.
"""
import argparse, csv, json, sys, time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

import analysis
import sim
from gauge_search import apply_candidate

FIELDS = ["angle_mrad", "seed", "dose", "noise_seed", "region", "polarity", "fu", "fv",
          "misfit", "region_corr", "region_reversed", "region_q", "others_reversed",
          "others_shifted", "start_corr", "start_reversed", "start_q", "start_misfit",
          "runtime_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--angles", type=float, nargs=1, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--region", required=True)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--support", action="store_true")
    ap.add_argument("--carbon-thickness", type=float, default=None)
    ap.add_argument("--dose", type=float, default=None)
    ap.add_argument("--noise-seed", type=int, default=None)
    ap.add_argument("--store", action="store_true",
                    help="also save every candidate's band-limited phase (float16) and "
                         "per-region verdicts to phases_s<seed>_<region>.npz, for figures")
    args = ap.parse_args()
    print("argv: " + " ".join(sys.argv), flush=True)
    angle = args.angles[0]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    atoms = sim.build_sample(support=args.support, carbon_thickness=args.carbon_thickness)
    obj, _, sampling = sim.projected_object(atoms)
    pixels = sim.scan_pixels(sampling)
    truth = analysis.band_limit(np.angle(obj), sampling)
    regions = analysis.build_regions(truth, sampling, sim.PARTICLES)
    QF = {r["name"]: analysis.cell_quantile_fn(r["G"]) for r in regions if r["G"] is not None}
    reg = next(r for r in regions if r["name"] == args.region)
    probe = sim.make_probe(angle, sim.ANGLES[angle])
    data = sim.simulate_data(probe, obj, pixels)
    noisy = sim.apply_dose(data, args.dose, args.noise_seed) if args.dose else data

    kept = dict(phases=[], labels=[], misfits=[], verdicts=[])

    def score(O):
        ph = analysis.band_limit(np.angle(O), sampling)
        rows = {r["particle"]: r for r in analysis.summarise(ph, truth, sampling, regions)}
        q = {k: float(QF[k](np.hypot(v["ws_x"], v["ws_y"]))) for k, v in rows.items()}
        score.last = ph
        return rows, q

    def keep(label, misfit, rows, q):
        """One candidate for the figure: its phase and each region's verdict, as counted
        in candidates.csv (reversed, else shifted if q > 0.25, else correct)."""
        if not args.store:
            return
        kept["phases"].append(score.last.astype(np.float16))
        kept["labels"].append(label)
        kept["misfits"].append(misfit)
        kept["verdicts"].append({k: ("reversed" if v["reversed"] else "shifted"
                                     if q[k] > 0.25 else None)
                                 for k, v in rows.items() if np.isfinite(v["ws_x"])})

    t0 = time.time()
    O0, h0 = sim.epie(probe, noisy, pixels, args.iters, args.seed, backend="torch")
    r0, q0 = score(O0)
    keep("start", h0[-1], r0, q0)
    print(f"start: misfit {h0[-1]:.6f}, {args.region} corr {r0[args.region]['signed_corr']:+.2f} "
          f"q {q0[args.region]:.2f} ({time.time() - t0:.0f} s)", flush=True)

    n = O0.shape[0]
    f = np.fft.fftfreq(n, sampling)
    KX, KY = np.meshgrid(f, f, indexing="ij")
    low = np.hypot(KX, KY) < analysis.BAND[0]
    m_soft = gaussian_filter(reg["mask"].astype(float), 1.0 / sampling, mode="wrap")
    fh = open(out / f"candidates_s{args.seed:03d}_{args.region}.csv", "w", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    w.writeheader()
    for pol in (1, -1):
        for frac in ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)):
            t1 = time.time()
            start = apply_candidate(O0, KX, KY, low, m_soft, reg["G"], pol, frac)
            O, h = sim.epie(probe, noisy, pixels, args.iters, args.seed, backend="torch",
                            O0=start)
            rows, q = score(O)
            keep(f"{pol:+d} {frac[0]:g} {frac[1]:g}", h[-1], rows, q)
            others = [k for k in rows if k != args.region and np.isfinite(rows[k]["ws_x"])]
            w.writerow(dict(
                angle_mrad=angle, seed=args.seed, dose=args.dose or "",
                noise_seed=args.noise_seed if args.dose else "", region=args.region,
                polarity=pol, fu=frac[0], fv=frac[1], misfit=h[-1],
                region_corr=rows[args.region]["signed_corr"],
                region_reversed=rows[args.region]["reversed"], region_q=q[args.region],
                others_reversed=sum(rows[k]["reversed"] for k in others),
                others_shifted=sum(q[k] > 0.25 for k in others),
                start_corr=r0[args.region]["signed_corr"],
                start_reversed=r0[args.region]["reversed"], start_q=q0[args.region],
                start_misfit=h0[-1], runtime_s=round(time.time() - t1, 1)))
            fh.flush()
            print(f"  polarity {pol:+d} shift {frac}: misfit {h[-1]:.6f}, corr "
                  f"{rows[args.region]['signed_corr']:+.2f} q {q[args.region]:.2f} "
                  f"({time.time() - t1:.0f} s)", flush=True)
    fh.close()
    if args.store:
        np.savez_compressed(out / f"phases_s{args.seed:03d}_{args.region}.npz",
                            phases=np.stack(kept["phases"]), misfits=np.array(kept["misfits"]),
                            labels=np.array(kept["labels"]),
                            verdicts=json.dumps(kept["verdicts"]),
                            truth=truth.astype(np.float16), sampling=sampling)


if __name__ == "__main__":
    main()
