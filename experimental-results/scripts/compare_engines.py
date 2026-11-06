"""Do the three engines reconstruct the same object?

    python compare_engines.py --scans 23 26           # seed 0 of each engine, pairwise

quantEM, the independent ePIE and PtyRAD write their objects in different frames: quantEM
rotates the scan by -169 deg in its convention, the other two by +169.14 deg in theirs,
the fields are cropped differently, and ptychography leaves the twin (the object rotated
by 180 deg and conjugated) as good a fit as the object.  So each pair is aligned first:
the second engine's phase is tried in all eight 90-degree rotations and transposes, and
the one whose envelope cross-correlates best with the first engine's, at the best
translation, is kept.  A negative peak marks the twin, whose phase is then negated.  The
aligned pair is scored like the seeds (register_lattice.local_agreement over the
particles), over the whole lattice band and over {111}+{200} and {220}.
"""
import argparse

import numpy as np

import register_lattice as R      # also puts simulated-results/scripts on the path
import analysis

E1 = R.E1
ENGINES = {"quantEM": E1, "ePIE": E1 / "epie", "PtyRAD": E1 / "ptyrad"}
BANDS = {"lattice": R.LATTICE, "{111}+{200}": (0.38, 0.53), "{220}": (0.62, 0.76)}


def final_phase(d, scan):
    R.E1 = d
    f = sorted(f for f in R.runs(scan) if "fixedprobe" not in f.name and "crop" not in f.name)[0]
    ph, s = R.phases(f)
    return ph[max(ph)], s, f.name


def padded(a, n):
    """`a` centred in an n x n field filled with its own mean, so the border is no edge."""
    out = np.full((n, n), a.mean())
    o0, o1 = (n - a.shape[0]) // 2, (n - a.shape[1]) // 2
    out[o0:o0 + a.shape[0], o1:o1 + a.shape[1]] = a
    return out


SHAPES = (0.015, 0.08)       # 1/A: the particles' shapes, above any phase ramp across the field
# quantEM's object lies in the scan frame and the other two in the detector frame, which
# the scan rotation (169.14 deg) separates; with the 90-degree turns that leaves +-10.86
EXTRA = (0.0, 10.86, -10.86)


def transformed(a, k, t, theta):
    from scipy.ndimage import rotate
    b = np.rot90(a.T if t else a, k)
    return rotate(b, theta, reshape=False, order=1, mode="nearest") if theta else b


def best_frame(mov, ref, s):
    """The turn, transpose, extra rotation, translation and sign that best map `mov` onto
    `ref`, judged on the particle shapes."""
    er = analysis.band_limit(ref, s, *SHAPES)
    Fr = np.fft.fft2(er)
    best = None
    for t in (False, True):
        for k in range(4):
            for theta in EXTRA:
                em = analysis.band_limit(transformed(mov, k, t, theta), s, *SHAPES)
                xc = np.real(np.fft.ifft2(Fr * np.conj(np.fft.fft2(em))))
                xc /= np.sqrt((er ** 2).sum() * (em ** 2).sum())
                i = np.unravel_index(np.argmax(np.abs(xc)), xc.shape)
                if best is None or abs(xc[i]) > abs(best[0]):
                    best = (float(xc[i]), k, t, theta, i)
    c, k, t, theta, i = best
    m = np.roll(transformed(mov, k, t, theta), i, axis=(0, 1))
    return np.sign(c) * m, dict(rot90=k, transpose=t, rotation=theta, shift=i,
                                envelope_corr=c, twin=c < 0)


def warp(a, k, t, theta, shift):
    """The exact turn/transpose, then a rotation by `theta` deg about the centre and a
    translation by `shift` px, by cubic interpolation."""
    from scipy.ndimage import affine_transform
    b = np.rot90(a.T if t else a, k)
    th = np.deg2rad(theta)
    M = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])   # output -> input
    c = (np.array(b.shape) - 1) / 2
    return affine_transform(b, M, offset=c - M @ (c + np.asarray(shift)), order=3,
                            mode="nearest")


def register(mov, ref, s):
    """The frame of `mov` in `ref`'s, to lattice precision: the coarse frame from the particle
    shapes, then rotation and sub-pixel translation refined by the lattice correlation over
    the particles.  Only meaningful where the lattice is unique (above threshold)."""
    from scipy.optimize import minimize
    _, f = best_frame(mov, ref, s)
    n = ref.shape[0]
    sh = np.array([(v + n // 2) % n - n // 2 for v in f["shift"]], float)
    on = R.particle_area(ref, s)
    lr = R.lattice(ref, s)

    def cost(p):
        m = R.lattice(warp(mov, f["rot90"], f["transpose"], p[0], p[1:]), s)
        return -np.sign(f["envelope_corr"]) * (m * lr)[on].sum() / np.sqrt(
            (m[on] ** 2).sum() * (lr[on] ** 2).sum())

    p0 = np.array([f["rotation"], *sh])
    r = minimize(cost, p0, method="Nelder-Mead",
                 options=dict(initial_simplex=[p0, p0 + [0.1, 0, 0], p0 + [0, 0.7, 0],
                                               p0 + [0, 0, 0.7]], xatol=0.005, fatol=1e-4))
    return dict(f, rotation=float(r.x[0]), shift=r.x[1:], lattice_corr=-r.fun)


def search(mov, ref, s, f, reach_px=25, dtheta=0.4, n_theta=9):
    """Best global alignment of `mov`'s lattice onto `ref`'s, near frame `f`.

    The turn, transpose and twin come from `f`; the rotation is searched over
    f['rotation'] +- dtheta, and for each the translation is found by FFT cross-correlation
    of the lattice bands over `ref`'s particles, within `reach_px` of f['shift'], then
    refined below a pixel.  Below threshold no global alignment can make independently
    offset particles agree, so what remains disagreeing afterwards is disagreement between
    the reconstructions, not between their frames.
    """
    from scipy.optimize import minimize
    n = ref.shape[0]
    on = R.particle_area(ref, s)
    lr = R.lattice(ref, s)
    Fr = np.fft.fft2(lr * on)
    sgn = np.sign(f["envelope_corr"])
    s0 = np.array([(v + n // 2) % n - n // 2 for v in np.round(f["shift"])], int)
    best = None
    for th in f["rotation"] + np.linspace(-dtheta, dtheta, n_theta):
        lm = sgn * R.lattice(warp(mov, f["rot90"], f["transpose"], th, (0, 0)), s)
        xc = np.real(np.fft.ifft2(Fr * np.conj(np.fft.fft2(lm))))
        xc = np.roll(xc, (n // 2, n // 2), axis=(0, 1))     # zero shift at the centre
        c = n // 2 + s0
        win = xc[c[0] - reach_px:c[0] + reach_px + 1, c[1] - reach_px:c[1] + reach_px + 1]
        i = np.unravel_index(np.argmax(win), win.shape)
        if best is None or win[i] > best[0]:
            best = (win[i], th, s0 + np.array(i) - reach_px)
    _, th, sh = best

    def cost(p):
        m = sgn * R.lattice(warp(mov, f["rot90"], f["transpose"], p[0], p[1:]), s)
        return -(m * lr)[on].sum() / np.sqrt((m[on] ** 2).sum() * (lr[on] ** 2).sum())

    # a cross-correlation peak at k means lr(x) ~ lm(x - k): warp's shift, same sign
    p0 = np.array([th, *sh.astype(float)])
    r = minimize(cost, p0, method="Nelder-Mead",
                 options=dict(initial_simplex=[p0, p0 + [0.05, 0, 0], p0 + [0, 0.5, 0],
                                               p0 + [0, 0, 0.5]], xatol=0.005, fatol=1e-4))
    if -r.fun < -cost(p0):
        r.x, r.fun = p0, cost(p0)
    return dict(f, rotation=float(r.x[0]), shift=r.x[1:], lattice_corr=-r.fun)


def per_scan(args, names):
    """Each scan aligned on its own lattice, starting from the frame registered on
    --register-on (the turn, transpose and twin are scan-independent)."""
    ph = {n: final_phase(d, args.register_on) for n, d in ENGINES.items()}
    s = ph["quantEM"][1]
    n = max(v[0].shape[0] for v in ph.values())
    ref = padded(ph["quantEM"][0], n)
    base = {b: register(padded(ph[b][0], n), ref, s) for b in names[1:]}
    for scan in args.scans:
        ph = {nm: final_phase(d, scan) for nm, d in ENGINES.items()}
        aligned = {"quantEM": padded(ph["quantEM"][0], n)}
        for b in names[1:]:
            mov = padded(ph[b][0], n)
            f = search(mov, aligned["quantEM"], s, base[b])
            aligned[b] = np.sign(f["envelope_corr"]) * warp(mov, f["rot90"], f["transpose"],
                                                            f["rotation"], f["shift"])
            d = np.asarray(f["shift"]) - np.asarray(base[b]["shift"])
            print(f"  scan {scan}  {b} frame: rotation {f['rotation']:+.3f} deg "
                  f"({f['rotation'] - base[b]['rotation']:+.3f} vs scan {args.register_on}), "
                  f"shift ({d[0]:+.2f}, {d[1]:+.2f}) px vs scan {args.register_on}, "
                  f"lattice corr {f['lattice_corr']:.2f}", flush=True)
        on = R.particle_area(aligned["quantEM"], s)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                out = []
                for band, (lo, hi) in BANDS.items():
                    m = R.local_agreement(analysis.band_limit(aligned[b], s, lo, hi),
                                          analysis.band_limit(aligned[a], s, lo, hi), s)
                    out.append(f"{band} {(m[on] > 0.5).mean():.2f}/{(m[on] < -0.5).mean():.2f}")
                print(f"  scan {scan}  {a:8s} vs {b:7s}: agree/reversed  " + "  ".join(out),
                      flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans", type=int, nargs="+", default=[23, 26])
    ap.add_argument("--per-scan", action="store_true",
                    help="with --register-on: align every scan on its own lattice")
    ap.add_argument("--register-on", type=int, default=None,
                    help="register the engines on this (above-threshold) scan, to lattice "
                         "precision, and apply that frame to every scan")
    args = ap.parse_args()
    names = list(ENGINES)
    if args.register_on is not None and args.per_scan:
        return per_scan(args, names)
    if args.register_on is not None:
        ph = {n: final_phase(d, args.register_on) for n, d in ENGINES.items()}
        s = ph["quantEM"][1]
        n = max(v[0].shape[0] for v in ph.values())
        ref = padded(ph["quantEM"][0], n)
        frames = {}
        for b in names[1:]:
            frames[b] = register(padded(ph[b][0], n), ref, s)
            f = frames[b]
            print(f"{b} -> quantEM, registered on scan {args.register_on}: rot90 x{f['rot90']}"
                  f", transpose {f['transpose']}, rotation {f['rotation']:+.3f} deg, shift "
                  f"({f['shift'][0]:+.2f}, {f['shift'][1]:+.2f}) px, twin {f['twin']}, "
                  f"lattice corr {f['lattice_corr']:.2f}", flush=True)
        for scan in args.scans:
            ph = {nm: final_phase(d, scan) for nm, d in ENGINES.items()}
            aligned = {"quantEM": padded(ph["quantEM"][0], n)}
            for b in names[1:]:
                f = frames[b]
                aligned[b] = np.sign(f["envelope_corr"]) * warp(
                    padded(ph[b][0], n), f["rot90"], f["transpose"], f["rotation"], f["shift"])
            on = R.particle_area(aligned["quantEM"], s)
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    out = []
                    for band, (lo, hi) in BANDS.items():
                        m = R.local_agreement(analysis.band_limit(aligned[b], s, lo, hi),
                                              analysis.band_limit(aligned[a], s, lo, hi), s)
                        out.append(f"{band} {(m[on] > 0.5).mean():.2f}/{(m[on] < -0.5).mean():.2f}")
                    print(f"  scan {scan}  {a:8s} vs {b:7s}: agree/reversed  " + "  ".join(out),
                          flush=True)
        return
    for scan in args.scans:
        ph = {n: final_phase(d, scan) for n, d in ENGINES.items()}
        print(f"scan {scan}: " + ", ".join(f"{n} {v[2]}" for n, v in ph.items()))
        # every engine is mapped into quantEM's frame, whose field is the scan itself and so
        # has no tilted edge for the alignment to lock onto; pairs are compared there
        s = ph["quantEM"][1]
        n = max(v[0].shape[0] for v in ph.values())
        ref = padded(ph["quantEM"][0], n)
        aligned = {"quantEM": ref}
        for b in names[1:]:
            aligned[b], frame = best_frame(padded(ph[b][0], n), ref, s)
            print(f"  {b} -> quantEM frame: rot90 x{frame['rot90']}, transpose "
                  f"{frame['transpose']}, extra {frame['rotation']:+.2f} deg, twin "
                  f"{frame['twin']}, shape corr {abs(frame['envelope_corr']):.2f}")
        on = R.particle_area(ref, s)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                out = []
                for band, (lo, hi) in BANDS.items():
                    m = R.local_agreement(analysis.band_limit(aligned[b], s, lo, hi),
                                          analysis.band_limit(aligned[a], s, lo, hi), s)
                    out.append(f"{band} agree {(m[on] > 0.5).mean():.2f} / reversed "
                               f"{(m[on] < -0.5).mean():.2f}")
                print(f"  {a:8s} vs {b:7s}: " + " | ".join(out), flush=True)


if __name__ == "__main__":
    main()
