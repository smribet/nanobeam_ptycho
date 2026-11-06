"""Mixed-state ePIE on the raw experimental counts, sharing nothing with quantEM but the file.

    python epie_independent.py --scan 23 --seeds 0 1 --iters 100       # positive control
    python epie_independent.py --scan 26 --seeds 0 1 2 3 --iters 200
    python epie_independent.py --scan 26 --seeds 0 --iters 3 --crop 192 320 192 320  # test

A reviewer can fairly ask whether the ambiguity of the quantEM reconstructions comes from
quantEM: from its preprocessing, which re-centres every pattern by a sub-pixel Fourier
shift and clips negatives, or from its object model, which clips |O| to [0, 1] and
removes the mean phase at every step.  This pipeline does neither.

  data      the Arina counts read with h5py and binned 2x2 by summation, nothing else:
            no re-centring, no normalisation, no clipping
  geometry  the scan step from the acquisition (0.4747 A) and the scan rotation, found
            from the data: the curl-free centre-of-mass field gives it to a degree or so
            (and leaves 180 deg open), short trial runs settle the 180 deg and the sign
            convention of the aberration phase, and the misfit over +-10 deg refines it.
            On scan 23 this gives 169.14 deg, the notebooks' parallax-fit -169 deg in the
            mirrored convention, and that value is then used for every scan
  probe     aperture and aberrations (the notebooks' C10, C12, phi12) built directly on
            the detector grid, centred on the measured disk centre, so the data are never
            resampled; four mixed-state modes, refined
  engine    mixed-state ePIE (Thibault & Menzel 2013) in mini-batches, with each batch's
            updates averaged where its positions overlap; sub-pixel positions by Fourier
            shifts of the probe; no constraint of any kind on object or probe.  The
            probe-object scale and global phase (P O is unchanged by O -> O/c, P -> P c)
            are fixed exactly after every iteration rather than by clipping |O|

Writes the same h5 layout as reconstruct_quantem.py, into e1/epie/, so register_lattice.py
and make_fig_e1.py read either.
"""
import argparse
import json
import os
import time
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  (registers the bitshuffle filter the Arina files use)
import numpy as np
import torch

RAW = Path.home() / "data/nanobeam_ptycho/raw"
OUT = Path.home() / "data/nanobeam_ptycho/recons/e1/epie"
STEP = 0.47470101                     # A, from the acquisition
MRAD_PER_PX_UNBINNED = 0.93246184 / 2  # the notebooks' calibration, for 192 x 192
BIN, N, MRAD_PER_PX = 2, 96, 0.93246184  # detector binning and the resulting pattern
WAVELENGTH = 0.0196875                # A at 300 kV


def set_binning(b):
    """Detector binning 1 (192 x 192, as recorded) or 2 (96 x 96, as published).  The
    real-space pixel is set by the total collection angle and so does not change; finer
    reciprocal sampling instead widens the probe window."""
    global BIN, N, MRAD_PER_PX
    BIN, N, MRAD_PER_PX = b, 192 // b, MRAD_PER_PX_UNBINNED * b
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


def load_raw(scan, crop=None, chunk=8192):
    """Counts binned BIN x BIN, shape (rows, cols, N, N), float32; `crop` = (r0, r1, c0, c1)."""
    os.chdir(RAW)                         # the master file links its data files by name
    with h5py.File(f"nanobeam_ptycho_{scan}_master.h5") as f:
        parts = [f["entry/data"][k] for k in sorted(f["entry/data"])]
        n = sum(p.shape[0] for p in parts)
        side = int(round(np.sqrt(n)))
        r0, r1, c0, c1 = crop or (0, side, 0, side)
        out = np.zeros((r1 - r0, c1 - c0, N, N), np.float32)
        start = 0
        for p in parts:
            for a in range(0, p.shape[0], chunk):
                b = min(a + chunk, p.shape[0])
                idx = np.arange(start + a, start + b)
                rr, cc = idx // side, idx % side
                keep = (rr >= r0) & (rr < r1) & (cc >= c0) & (cc < c1)
                if not keep.any():
                    continue
                x = p[a:b][keep].astype(np.float32)
                if BIN > 1:
                    x = x.reshape(-1, N, BIN, N, BIN).sum((2, 4))
                out[rr[keep] - r0, cc[keep] - c0] = x
            start += p.shape[0]
    return out


def disk(mean_dp):
    """Centre (row, col) and radius of the bright-field disk, in binned pixels."""
    m = mean_dp > 0.5 * mean_dp.max()
    r, c = np.indices(m.shape)
    w = mean_dp * m
    return (float((r * w).sum() / w.sum()), float((c * w).sum() / w.sum()),
            float(np.sqrt(m.sum() / np.pi)))


def curl_rotation(data, centre):
    """Scan rotation (deg) and transpose that make the centre-of-mass field curl-free."""
    q = np.indices(data.shape[2:]).astype(np.float32)
    tot = data.sum((2, 3))
    com = np.stack([(data * q[i]).sum((2, 3)) / tot - centre[i] for i in (0, 1)])
    com -= com.mean((1, 2), keepdims=True)
    best = None
    for transpose in (False, True):
        cr, cc = (com[1], com[0]) if transpose else (com[0], com[1])
        for deg in np.arange(-180.0, 180.0, 0.5):
            t = np.deg2rad(deg)
            # the deflection, expressed in scan coordinates
            gr = np.cos(t) * cr + np.sin(t) * cc
            gc = -np.sin(t) * cr + np.cos(t) * cc
            curl = np.mean((np.gradient(gc, axis=0) - np.gradient(gr, axis=1)) ** 2)
            if best is None or curl < best[0]:
                best = (curl, deg, transpose)
    return best[1], best[2]


def make_probe(alpha, polar, centre, n=None, sign=-1.0):
    """Probe on the detector grid: aperture of `alpha` mrad centred on the measured disk,
    phase from C10, C12, phi12 (A, rad), real space centred in the n x n window."""
    n = n or N
    r, c = np.indices((n, n)).astype(np.float64)
    tr = (r - centre[0]) * MRAD_PER_PX * 1e-3          # rad
    tc = (c - centre[1]) * MRAD_PER_PX * 1e-3
    th, ph = np.hypot(tr, tc), np.arctan2(tr, tc)
    edge = (alpha * 1e-3 - th) / (MRAD_PER_PX * 1e-3)   # in pixels
    A = np.clip(0.5 + edge, 0.0, 1.0)                   # one-pixel soft edge
    chi = (2 * np.pi / WAVELENGTH) * 0.5 * th ** 2 * (
        polar.get("C10", 0.0) + polar.get("C12", 0.0) * np.cos(2 * (ph - polar.get("phi12", 0.0))))
    Pk = A * np.exp(1j * sign * chi)
    # detector frame is fftshifted (disk near the centre); real space centred in the window
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Pk), norm="ortho"))


def initial_modes(P, n_modes, weights=(1.0, 0.25, 0.25, 0.1)):
    """Mode 0 the probe; further modes the probe times x, y, xy, orthonormalised."""
    n = P.shape[0]
    y, x = (np.indices((n, n)) - n / 2) / (n / 4)
    modes = [P]
    for f in (x, y, x * y)[: n_modes - 1]:
        m = P * f
        for o in modes:
            m = m - np.vdot(o, m) / np.vdot(o, o) * o
        modes.append(m / np.linalg.norm(m) * np.linalg.norm(P))
    w = np.sqrt(np.array(weights[:n_modes]) / sum(weights[:n_modes]))
    return np.stack([m * wi for m, wi in zip(modes, w)])


class EPIE:
    def __init__(self, amps, pos_px, probes, obj_shape, dev, seed, batch=1024):
        self.dev, self.batch = dev, batch
        # measured amplitudes held in half precision: 4.8 GB for a full scan instead of 9.7,
        # at a relative precision of ~5e-4, far below the Poisson noise
        self.total = float((amps.astype(np.float64) ** 2).sum())
        self.amps = torch.from_numpy(amps.astype(np.float16)).to(dev)   # (N, 96, 96)
        self.pos = torch.from_numpy(pos_px).to(dev)                    # (N, 2), float
        self.ipos = torch.floor(self.pos).long()
        self.frac = (self.pos - self.ipos).float()
        self.P = torch.from_numpy(probes.astype(np.complex64)).to(dev)  # (M, n, n)
        n = self.P.shape[-1]
        self.n = n
        g = torch.Generator(device="cpu").manual_seed(seed)
        ph = (torch.randn(obj_shape, generator=g) - 0.5) * 1e-6        # as quantEM: ~flat
        self.O = torch.exp(1j * ph).to(torch.complex64).to(dev)
        self.gen = g
        k = torch.fft.fftfreq(n, device=dev)
        self.KR, self.KC = torch.meshgrid(k, k, indexing="ij")
        ar = torch.arange(n, device=dev)
        self.wr, self.wc = torch.meshgrid(ar, ar, indexing="ij")
        # the illuminated area, over which the scale and phase gauge is fixed
        cover = torch.zeros(obj_shape, device=dev)
        I = (self.P.abs() ** 2).sum(0)
        for b in torch.arange(self.pos.shape[0], device=dev).split(4096):
            cover.index_put_((self.ipos[b, 0, None, None] + self.wr,
                              self.ipos[b, 1, None, None] + self.wc),
                             I.expand(len(b), n, n).contiguous(), accumulate=True)
        self.fov = cover > 0.1 * cover.max()

    def fix_gauge(self):
        """Exact gauge fixing: O -> O / c and P -> P c with c the mean of O over the
        illuminated area, so the object has unit mean modulus and zero mean phase there.
        P O, and so every predicted pattern, is unchanged; this only stops the scale and
        global phase drifting along their flat directions, without constraining |O|."""
        o = self.O[self.fov]
        c = o.abs().mean() * torch.exp(1j * torch.angle(o.mean()))
        self.O /= c
        self.P *= c

    def shifted_probes(self, frac):
        """(B, M, n, n): the probe modes shifted by each position's sub-pixel remainder."""
        ramp = torch.exp(-2j * np.pi * (frac[:, 0, None, None] * self.KR
                                        + frac[:, 1, None, None] * self.KC))
        Pk = torch.fft.fft2(self.P)
        return torch.fft.ifft2(Pk[None] * ramp[:, None]), ramp

    def iterate(self, learn_probe=True, beta=1.0, alpha=1.0, eps=0.1):
        """One pass over every position.  The denominators are regularised as in rPIE
        (Maiden et al. 2017), `eps` of their maximum, so weakly illuminated pixels are not
        divided by nearly nothing."""
        err = 0.0
        perm = torch.randperm(self.pos.shape[0], generator=self.gen).to(self.dev)
        for b in perm.split(self.batch):
            rr = self.ipos[b, 0, None, None] + self.wr
            cc = self.ipos[b, 1, None, None] + self.wc
            Op = self.O[rr, cc]                                          # (B, n, n)
            Ps, ramp = self.shifted_probes(self.frac[b])                 # (B, M, n, n)
            psi = Ps * Op[:, None]
            Psi = torch.fft.fft2(psi, norm="ortho")
            mod = torch.sqrt((Psi.abs() ** 2).sum(1))                    # (B, n, n), unshifted
            meas = torch.fft.ifftshift(self.amps[b].float(), dim=(-2, -1))
            err += float(((mod - meas) ** 2).sum())
            Psi2 = Psi * (meas / (mod + 1e-6))[:, None]
            d = torch.fft.ifft2(Psi2, norm="ortho") - psi                # (B, M, n, n)
            # object: every position's update, averaged where positions overlap (real and
            # imaginary parts accumulated separately, as CUDA cannot add complex atomically)
            u = (Ps.conj() * d).sum(1)
            nr = torch.zeros(self.O.shape, device=self.dev)
            ni = torch.zeros(self.O.shape, device=self.dev)
            den = torch.zeros(self.O.shape, device=self.dev)
            nr.index_put_((rr, cc), u.real.contiguous(), accumulate=True)
            ni.index_put_((rr, cc), u.imag.contiguous(), accumulate=True)
            den.index_put_((rr, cc), (Ps.abs() ** 2).sum(1), accumulate=True)
            self.O += beta * torch.complex(nr, ni) / (den + eps * den.max())
            if learn_probe:
                # the gradient on the unshifted modes: shift each contribution back
                g = torch.fft.ifft2(torch.fft.fft2(Op.conj()[:, None] * d) * ramp.conj()[:, None])
                w = (Op.abs() ** 2).sum(0)
                self.P += alpha * g.sum(0) / (w + eps * w.max())[None]
        self.fix_gauge()
        return float(np.sqrt(err / self.total))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=int, required=True, choices=sorted(SCANS))
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--snapshot-every", type=int, default=50)
    ap.add_argument("--probes", type=int, default=4)
    ap.add_argument("--probe-delay", type=int, default=5,
                    help="iterations with the probe held before it is refined")
    ap.add_argument("--fixed-probe", action="store_true")
    ap.add_argument("--rotation", type=float, default=None,
                    help="scan rotation, deg (default: the curl-free one found here)")
    ap.add_argument("--transpose", action="store_true")
    ap.add_argument("--geometry-only", action="store_true",
                    help="find rotation and phase sign, print them, and stop")
    ap.add_argument("--bin", type=int, default=2, choices=[1, 2],
                    help="detector binning: 2 as published (96 x 96), 1 as recorded")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--phase-sign", type=float, default=-1.0, choices=[-1.0, 1.0])
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--crop", type=int, nargs=4, default=None, metavar=("R0", "R1", "C0", "C1"))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    set_binning(args.bin)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    s = SCANS[args.scan]
    polar = {k: v for k, v in s.items() if k != "alpha"}

    t0 = time.time()
    data = load_raw(args.scan, args.crop)
    mean_dp = data.mean((0, 1))
    cr, cc, rad = disk(mean_dp)
    print(f"scan {args.scan}: {data.shape} read and binned in {time.time() - t0:.0f} s; "
          f"{data.sum((2, 3)).mean():.0f} counts per pattern; disk centre ({cr:.2f}, {cc:.2f}), "
          f"radius {rad:.2f} px = {rad * MRAD_PER_PX:.2f} mrad (nominal {s['alpha']})", flush=True)
    rot, tr = curl_rotation(data[::4, ::4], (cr, cc))
    print(f"  curl-free scan rotation {rot:.1f} deg (or {rot + 180 - 360 * (rot > 0):.1f}), "
          f"transpose {tr}  (notebooks: -169, False)", flush=True)
    dev = torch.device(args.device)
    dx = WAVELENGTH / (N * MRAD_PER_PX * 1e-3)

    def setup(d, rot, tr, sign):
        R, C = np.indices(d.shape[:2]).reshape(2, -1) * STEP
        if tr:
            R, C = C, R
        t = np.deg2rad(rot)
        pos = np.stack([np.cos(t) * R - np.sin(t) * C, np.sin(t) * R + np.cos(t) * C], 1) / dx
        pad = 16
        pos -= pos.min(0) - pad
        m = int(np.ceil(pos.max())) + N + pad           # square, as the scoring assumes
        amps = np.sqrt(np.maximum(d.reshape(-1, N, N), 0))
        P = make_probe(s["alpha"], polar, (cr, cc), sign=sign)
        P *= np.sqrt((amps ** 2).sum((1, 2)).mean() / (np.abs(P) ** 2).sum())
        return amps, pos, initial_modes(P, args.probes), (m, m)

    # the curl leaves rotation by 180 deg open, and the sign of the aberration phase is a
    # convention: decide both by the misfit after a few iterations on the central 48 x 48
    # positions, from the data alone
    if args.rotation is not None:
        rot, tr, sign = args.rotation, args.transpose, args.phase_sign
    else:
        h, w = data.shape[0] // 2, min(64, data.shape[0] // 2)
        sub = data[h - w:h + w, h - w:h + w]

        def misfit(rr, sg, n=3):
            e = EPIE(*setup(sub, rr, tr, sg), dev, seed=0, batch=args.batch)
            return [e.iterate(learn_probe=False) for _ in range(n)][-1]

        trial = {(rr, sg): misfit(rr, sg) for rr in (rot, rot + 180.0) for sg in (-1.0, 1.0)}
        rot, sign = min(trial, key=trial.get)
        print("  geometry trial (rotation, phase sign -> misfit): "
              + ", ".join(f"({r:.1f}, {g:+.0f}) {v:.4f}" for (r, g), v in trial.items())
              + " (a 180 deg turn with the opposite sign is the twin solution, equally good)",
              flush=True)
        # the curl from a subsampled scan is good to a few degrees; refine by the misfit
        grid = rot + np.arange(-10.0, 10.5, 1.0)
        L = np.array([misfit(r, sign) for r in grid])
        k = int(np.argmin(L))
        if 0 < k < len(grid) - 1:
            a, b, _ = np.polyfit(grid[k - 1:k + 2], L[k - 1:k + 2], 2)
            rot = float(-b / (2 * a))
        else:
            rot = float(grid[k])
        print("  rotation scan: " + " ".join(f"{r:.0f}:{v:.4f}" for r, v in zip(grid, L))
              + f"\n  -> scan rotation {rot:.2f} deg, phase sign {sign:+.0f}", flush=True)
    if args.geometry_only:
        return
    amps, pos, probes, obj_shape = setup(data, rot, tr, sign)
    del data
    print(f"  object {obj_shape} at {dx:.4f} A/px, {len(pos)} positions, {args.probes} modes",
          flush=True)

    for seed in args.seeds:
        tag = (f"scan{args.scan}_bin{BIN}_s{seed:03d}_it{args.iters}"
               + ("_fixedprobe" if args.fixed_probe else "") + ("_crop" if args.crop else ""))
        if (out / f"{tag}.h5").exists():
            print(f"  {tag}: exists, skipped")
            continue
        t0 = time.time()
        e = EPIE(amps, pos, probes, obj_shape, dev, seed, args.batch)
        losses, snaps = [], []
        for it in range(1, args.iters + 1):
            learn = (not args.fixed_probe) and it > args.probe_delay
            losses.append(e.iterate(learn_probe=learn))
            if it % args.snapshot_every == 0 or it == args.iters:
                snaps.append((it, torch.angle(e.O).cpu().numpy().astype(np.float32)))
                print(f"  {tag}: iteration {it}, loss {losses[-1]:.6g}, {time.time() - t0:.0f} s",
                      flush=True)
        with h5py.File(out / f"{tag}.h5", "w") as f:
            f.create_dataset("obj", data=e.O.cpu().numpy()[None], compression="gzip")
            f.create_dataset("probe", data=e.P.cpu().numpy())
            f.create_dataset("sampling", data=[dx, dx])
            f.create_dataset("iter_losses", data=np.array(losses))
            f.create_dataset("snapshot_iters", data=[i for i, _ in snaps])
            f.create_dataset("snapshot_phases", data=np.stack([o for _, o in snaps]),
                             compression="gzip")
            f.attrs.update(scan=args.scan, seed=seed, iters=args.iters, engine="epie",
                           probes=args.probes, fixed_probe=args.fixed_probe,
                           rotation=rot, transpose=tr, phase_sign=sign,
                           polar=json.dumps(polar), alpha=s["alpha"],
                           disk=json.dumps([cr, cc, rad]), crop=json.dumps(args.crop),
                           runtime_s=round(time.time() - t0, 1))
        print(f"  wrote {tag}.h5", flush=True)
        del e
        if dev.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
