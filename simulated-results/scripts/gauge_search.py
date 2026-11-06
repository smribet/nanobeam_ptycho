"""Per-particle gauge search: can the data misfit alone pick each lattice's registry?

    python gauge_search.py --out DIR --angles 3 --seed-start 100 --seeds 10 \
        --dose 1e5 --noise-seed 1

The idea (W. de Kleijne): for a finite particle the transformations of Eqs. (5)-(6) are
not exact symmetries, so the correct registry may sit at a slightly lower misfit than the
wrong ones.  An optimizer started at random settles wherever it lands; an exhaustive search
over the transformations compares them directly.

Each reconstruction is first re-run from its seed exactly as run_sweep.py does (the same
seed gives the same run on any backend).  Then, one region at a time, the region's lattice
is offered every candidate -- both polarities times an n x n grid of translations spanning
its unit cell, the identity included -- and each candidate is relaxed for a few ePIE
iterations on the probe positions that illuminate that region.  The candidate with the
lowest misfit is kept before moving on.  Every region is scored against ground truth before
and after, and the misfit of every candidate is saved, so the margin between the correct
registry and its best competitor can be read off directly.

Writes search.csv (one row per seed and region) and runs/gs_a<angle>_s<seed>.npz (the
misfit of every candidate, and the band-limited phase before and after the search).
"""
import argparse, csv, json, sys, time
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

import sim
import analysis

FIELDS = ["angle_mrad", "seed", "dose", "noise_seed", "particle", "n_positions",
          "before_corr", "before_reversed", "before_q", "after_corr", "after_reversed",
          "after_q", "chosen_polarity", "chosen_fu", "chosen_fv", "loss_identity",
          "loss_chosen", "loss_correct", "loss_best_wrong", "global_before", "global_after",
          "runtime_s"]


def global_corr(a, b):
    A, B = a - a.mean(), b - b.mean()
    xc = np.real(np.fft.ifft2(np.fft.fft2(A) * np.conj(np.fft.fft2(B))))
    return float(xc.max() / np.sqrt(np.sum(A ** 2) * np.sum(B ** 2)))


class Relaxer:
    """ePIE with a fixed probe on one region's probe positions, data held on the GPU.

    The same update as sim._epie_torch, with the patterns uploaded once per region rather
    than once per candidate, and the same update order for every candidate.
    """
    def __init__(self, probe, data, pixels, seed):
        import torch
        self.torch, self.dev = torch, torch.device(sim.best_device())
        self.P = torch.from_numpy(probe).to(self.dev)
        self.Pmax = (self.P.abs() ** 2).max()
        self.amps = [torch.from_numpy(np.sqrt(d)).to(self.dev) for d in data]
        self.total = float(sum(float(d.sum()) for d in data))
        self.pixels, self.seed = pixels, seed

    def __call__(self, O0, n_iter, beta=0.9):
        torch = self.torch
        O = torch.from_numpy(O0).to(self.dev)
        rng = np.random.default_rng(self.seed)
        err = None
        for _ in range(n_iter):
            err = torch.zeros((), dtype=torch.float32, device=self.dev)
            for k in rng.permutation(len(self.pixels)):
                i, j = self.pixels[k]
                Pr = torch.roll(self.P, (int(i), int(j)), dims=(0, 1))
                psi = Pr * O
                Psi = torch.fft.fft2(psi)
                amp = self.amps[k]
                err = err + ((Psi.abs() - amp) ** 2).sum()
                corrected = torch.fft.ifft2(amp * torch.exp(1j * Psi.angle()))
                O = O + beta * torch.conj(Pr) / self.Pmax * (corrected - psi)
        return O.cpu().numpy(), float(torch.sqrt(err / self.total))


def apply_candidate(O, KX, KY, low, m_soft, G, polarity, frac):
    """Eq. (6) and/or Eq. (5) applied to the lattice part of O inside one soft mask.

    `low` selects the envelope (kept); the rest is multiplied by `polarity` and translated
    by frac[0] a1 + frac[1] a2.
    """
    F = np.fft.fft2(O)
    hi = O - np.fft.ifft2(F * low)
    A = np.linalg.inv(G).T
    d = frac[0] * A[0] + frac[1] * A[1]
    new = polarity * np.fft.ifft2(np.fft.fft2(hi)
                                  * np.exp(-2j * np.pi * (KX * d[0] + KY * d[1])))
    return (O + m_soft * (new - hi)).astype(np.complex64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--angles", type=float, nargs=1, required=True)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--iters", type=int, default=200, help="initial reconstruction")
    ap.add_argument("--relax-iters", type=int, default=50,
                    help="per candidate; enough for a transformed candidate to settle into "
                         "its own minimum, or the converged identity wins by default")
    ap.add_argument("--n-shift", type=int, default=4, help="translations per cell axis")
    ap.add_argument("--reach", type=float, default=8.0,
                    help="Angstrom; positions this close to a region relax its candidates")
    ap.add_argument("--support", action="store_true")
    ap.add_argument("--carbon-thickness", type=float, default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--data-kind", default="ms", choices=["ms", "ss"])
    ap.add_argument("--dose", type=float, default=None)
    ap.add_argument("--noise-seed", type=int, default=None)
    args = ap.parse_args()
    print("argv: " + " ".join(sys.argv), flush=True)

    angle = args.angles[0]
    out = Path(args.out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    atoms = sim.build_sample(support=args.support, carbon_thickness=args.carbon_thickness)
    obj, _, sampling = sim.projected_object(atoms)
    pixels = sim.scan_pixels(sampling)
    truth = analysis.band_limit(np.angle(obj), sampling)
    regions = analysis.build_regions(truth, sampling, sim.PARTICLES)
    QF = {r["name"]: analysis.cell_quantile_fn(r["G"]) for r in regions if r["G"] is not None}
    probe = sim.make_probe(angle, sim.ANGLES[angle])
    if args.data_dir:
        data = np.load(Path(args.data_dir) / f"data_{args.data_kind}_a{angle:g}.npy")
    else:
        data = sim.simulate_data(probe, obj, pixels)

    n = obj.shape[0]
    f = np.fft.fftfreq(n, sampling)
    KX, KY = np.meshgrid(f, f, indexing="ij")
    low = np.hypot(KX, KY) < analysis.BAND[0]
    cands = [(p, (i / args.n_shift, j / args.n_shift))
             for p in (1, -1) for i in range(args.n_shift) for j in range(args.n_shift)]
    local = {}
    for r in regions:
        near = distance_transform_edt(~r["mask"]) * sampling <= args.reach
        local[r["name"]] = [k for k, (i, j) in enumerate(pixels) if near[i % n, j % n]]

    fh = open(out / "search.csv", "w", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    w.writeheader()
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        t0 = time.time()
        nseed = args.noise_seed if args.noise_seed is not None else seed
        noisy = sim.apply_dose(data, args.dose, nseed) if args.dose else data
        O, _ = sim.epie(probe, noisy, pixels, args.iters, seed, backend="torch")
        phase0 = analysis.band_limit(np.angle(O), sampling)
        before = {r["particle"]: r for r in analysis.summarise(phase0, truth, sampling, regions)}
        g0 = global_corr(phase0, truth)

        losses, chosen = {}, {}
        for reg in regions:
            name = reg["name"]
            if reg["G"] is None:
                continue
            sub = local[name]
            relax = Relaxer(probe, [noisy[k] for k in sub], [pixels[k] for k in sub], seed)
            m_soft = gaussian_filter(reg["mask"].astype(float), 1.0 / sampling, mode="wrap")
            L = np.full((2, args.n_shift, args.n_shift), np.nan)
            best = None
            for pol, frac in cands:
                # every candidate, the identity included, gets the same relaxation, so the
                # converged incumbent has no head start over the transformed ones
                Or, e = relax(apply_candidate(O, KX, KY, low, m_soft, reg["G"], pol, frac),
                              args.relax_iters)
                i, j = (int(round(v * args.n_shift)) for v in frac)
                L[0 if pol == 1 else 1, i, j] = e
                if best is None or e < best[0]:
                    best = (e, pol, frac, Or)
            O = best[3]
            losses[name], chosen[name] = L, best[:3]
            del relax

        phase1 = analysis.band_limit(np.angle(O), sampling)
        after = {r["particle"]: r for r in analysis.summarise(phase1, truth, sampling, regions)}
        g1 = global_corr(phase1, truth)
        dt = time.time() - t0

        for name, L in losses.items():
            b, a = before[name], after[name]
            # the candidate that undoes the measured offset (and reversal) of the run before
            # the search, rounded to the grid: the registry the search should find
            ci, cj = (int(round((-v % 1.0) * args.n_shift)) % args.n_shift
                      for v in (b["frac_u"], b["frac_v"]))
            cp = 1 if b["reversed"] else 0
            wrong = L.copy()
            wrong[cp, ci, cj] = np.nan
            w.writerow(dict(
                angle_mrad=angle, seed=seed, dose=args.dose or "",
                noise_seed=nseed if args.dose else "", particle=name,
                n_positions=len(local[name]),
                before_corr=b["signed_corr"], before_reversed=b["reversed"],
                before_q=float(QF[name](np.hypot(b["ws_x"], b["ws_y"]))),
                after_corr=a["signed_corr"], after_reversed=a["reversed"],
                after_q=float(QF[name](np.hypot(a["ws_x"], a["ws_y"]))),
                chosen_polarity=chosen[name][1], chosen_fu=chosen[name][2][0],
                chosen_fv=chosen[name][2][1], loss_identity=L[0, 0, 0],
                loss_chosen=chosen[name][0], loss_correct=L[cp, ci, cj],
                loss_best_wrong=float(np.nanmin(wrong)), global_before=g0,
                global_after=g1, runtime_s=round(dt, 1)))
        fh.flush()
        np.savez_compressed(out / "runs" / f"gs_a{angle:g}_s{seed:03d}.npz",
                            names=json.dumps(list(losses)),
                            losses=np.stack(list(losses.values())),
                            phase_before=phase0.astype(np.float32),
                            phase_after=phase1.astype(np.float32))
        n_fixed = sum(before[k]["reversed"] != after[k]["reversed"] for k in losses)
        print(f"   seed {seed:3d}: {dt:5.0f}s  global {g0:.3f} -> {g1:.3f}  "
              f"reversed {sum(before[k]['reversed'] for k in losses)} -> "
              f"{sum(after[k]['reversed'] for k in losses)}  ({n_fixed} changed)", flush=True)
    fh.close()


if __name__ == "__main__":
    main()
