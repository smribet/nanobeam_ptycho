"""Crops of the two Fig. 3 particles from every reconstruction, in one display frame.

    python e1_crops.py        # writes recons/e1/export/e1_crops.npz

The window holds both particles of Fig. 3, located in quantEM's 24 mrad reconstruction.
For each scan it is moved by that scan's drift (the envelope shift between quantEM's
reconstructions of the two scans), so every crop shows the same particles.  ePIE and PtyRAD
objects are mapped into quantEM's frame of the same scan (compare_engines: the frame found
at 24 mrad, refined on each scan's own lattice), and a second seed of any engine is first
aligned to that engine's first seed by its envelope.  The cropped PtyRAD runs (binned and
unbinned, central 256 x 256 positions) are mapped the same way, from the particle shapes.
Nothing is wrapped: crops are cut from the mapped arrays.
"""
import numpy as np

import register_lattice as R
import compare_engines as C

E1 = R.E1
OUT = E1 / "export"
WINDOW = (397, 774, 555, 1058)           # rows, cols of quantEM scan 23 seed 0 (px)
SCANS = (23, 24, 25, 26, 27)


def runs(d, scan, keep=lambda f: True):
    R.E1 = d
    fs = [f for f in R.runs(scan) if "fixedprobe" not in f.name and keep(f)]
    out = []
    for f in fs:
        ph, s = R.phases(f)
        out.append((int(f.name.split("_s")[1][:3]), ph[max(ph)], s))
    return sorted(out, key=lambda t: t[0])


def roll_shift(ph, ref_env, s):
    """The integer shift k with ph(x - k) ~ ref(x), centred."""
    _, k = R.align(ph, ref_env, s)
    n = ph.shape[0]
    return np.array([(v + n // 2) % n - n // 2 for v in k])


def cut(a, r0, c0, h, w):
    return a[r0:r0 + h, c0:c0 + w].astype(np.float32)


def main():
    OUT.mkdir(exist_ok=True)
    r0, r1, c0, c1 = WINDOW
    h, w = r1 - r0, c1 - c0
    q23 = runs(E1, 23, lambda f: "crop" not in f.name)
    s = q23[0][2]
    env23 = R.envelope(q23[0][1], s)
    # the engines' frames, from the 24 mrad data
    ph23 = {n: C.final_phase(d, 23) for n, d in C.ENGINES.items()}
    n = max(v[0].shape[0] for v in ph23.values())
    off = (n - q23[0][1].shape[0]) // 2               # quantEM inside the padded field
    ref23 = C.padded(ph23["quantEM"][0], n)
    base = {b: C.register(C.padded(ph23[b][0], n), ref23, s) for b in ("ePIE", "PtyRAD")}
    out = {"sampling": s, "window": np.array(WINDOW)}
    for scan in SCANS:
        q = runs(E1, scan, lambda f: "crop" not in f.name)
        k = roll_shift(q[0][1], env23, s)              # this scan's drift from scan 23
        wr, wc = r0 - k[0], c0 - k[1]                  # the window in this scan's frame
        qref = C.padded(q[0][1], n)
        qenv = R.envelope(q[0][1], s)
        for sd, ph, _ in q[:2]:
            kk = roll_shift(ph, qenv, s) if sd != q[0][0] else (0, 0)
            out[f"quantEM_{scan}_s{sd}"] = cut(ph, wr - kk[0], wc - kk[1], h, w)
        for b, d in (("ePIE", E1 / "epie"), ("PtyRAD", E1 / "ptyrad")):
            e = runs(d, scan, lambda f: "crop" not in f.name)
            f = C.search(C.padded(e[0][1], n), qref, s, base[b])
            eenv = R.envelope(e[0][1], s)
            for sd, ph, _ in e[:2]:
                if sd != e[0][0]:
                    ph = R.align(ph, eenv, s)[0]
                m = np.sign(f["envelope_corr"]) * C.warp(C.padded(ph, n), f["rot90"],
                                                         f["transpose"], f["rotation"],
                                                         f["shift"])
                out[f"{b}_{scan}_s{sd}"] = cut(m, wr + off, wc + off, h, w)
            print(f"scan {scan}: drift {k} px, {b} lattice corr {f['lattice_corr']:.2f}",
                  flush=True)
        if scan in (25, 26):
            for b in (2, 1):
                cr = runs(E1 / "ptyrad_crop", scan, lambda f, b=b: f"_bin{b}_" in f.name)
                mov = C.padded(cr[0][1], n)
                f0 = C.best_frame(mov, qref, s)[1]
                f = C.search(mov, qref, s, dict(f0, shift=np.array(f0["shift"], float)))
                cenv = R.envelope(cr[0][1], s)
                for sd, ph, _ in cr[:2]:
                    if sd != cr[0][0]:
                        ph = R.align(ph, cenv, s)[0]
                    m = np.sign(f["envelope_corr"]) * C.warp(C.padded(ph, n), f["rot90"],
                                                             f["transpose"], f["rotation"],
                                                             f["shift"])
                    out[f"crop_bin{b}_{scan}_s{sd}"] = cut(m, wr + off, wc + off, h, w)
                print(f"scan {scan}: cropped PtyRAD bin {b}, frame rot90 x{f['rot90']}, "
                      f"rotation {f['rotation']:+.2f}, lattice corr {f['lattice_corr']:.2f}",
                      flush=True)
    np.savez_compressed(OUT / "e1_crops.npz", **out)
    print("wrote", OUT / "e1_crops.npz")


def main_qcrop():
    """Only the cropped quantEM runs (central 256 x 256 positions, binned and unbinned,
    200 iterations), mapped into quantEM's full-field frame of the same scan the way the
    cropped PtyRAD runs are; written to their own small file, e1_qcrop.npz."""
    r0, r1, c0, c1 = WINDOW
    h, w = r1 - r0, c1 - c0
    q23 = runs(E1, 23, lambda f: "crop" not in f.name)
    s = q23[0][2]
    env23 = R.envelope(q23[0][1], s)
    out = {"sampling": s, "window": np.array(WINDOW)}
    for scan in (25, 26):
        q = runs(E1, scan, lambda f: "crop" not in f.name)
        k = roll_shift(q[0][1], env23, s)
        wr, wc = r0 - k[0], c0 - k[1]
        qref = q[0][1]
        n = qref.shape[0]
        for b in (2, 1):
            cr = runs(E1 / "quantem_crop", scan, lambda f, b=b: f"_bin{b}_" in f.name)
            mov = C.padded(cr[0][1], n)
            f0 = C.best_frame(mov, qref, s)[1]
            f = C.search(mov, qref, s, dict(f0, shift=np.array(f0["shift"], float)))
            cenv = R.envelope(cr[0][1], s)
            for sd, ph, _ in cr[:2]:
                if sd != cr[0][0]:
                    ph = R.align(ph, cenv, s)[0]
                m = np.sign(f["envelope_corr"]) * C.warp(C.padded(ph, n), f["rot90"],
                                                         f["transpose"], f["rotation"],
                                                         f["shift"])
                out[f"qcrop_bin{b}_{scan}_s{sd}"] = cut(m, wr, wc, h, w).astype(np.float16)
            print(f"scan {scan}: cropped quantEM bin {b}, frame rot90 x{f['rot90']}, "
                  f"transpose {f['transpose']}, rotation {f['rotation']:+.2f}, "
                  f"lattice corr {f['lattice_corr']:.2f}", flush=True)
    np.savez_compressed(OUT / "e1_qcrop.npz", **out)
    print("wrote", OUT / "e1_qcrop.npz")


if __name__ == "__main__":
    import sys
    main_qcrop() if "--qcrop" in sys.argv else main()
