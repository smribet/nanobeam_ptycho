"""Crops of the Fig. 3 window from quantEM's frozen-probe runs (seeds 10-12), for R13."""
import numpy as np
import register_lattice as R
import e1_crops as X

E1 = R.E1
r0, r1, c0, c1 = X.WINDOW
h, w = r1 - r0, c1 - c0
q23 = X.runs(E1, 23, lambda f: "crop" not in f.name)
s = q23[0][2]
env23 = R.envelope(q23[0][1], s)
out = {"sampling": s}
for scan in (25, 26, 27):
    q = X.runs(E1, scan, lambda f: "crop" not in f.name)
    k = X.roll_shift(q[0][1], env23, s)
    qenv = R.envelope(q[0][1], s)
    R.E1 = E1
    fx = []
    for f in sorted(f for f in R.runs(scan) if "fixedprobe" in f.name):
        ph, _ = R.phases(f)
        fx.append((int(f.name.split("_s")[1][:3]), ph[max(ph)]))
    for sd, ph in fx[:2]:
        kk = X.roll_shift(ph, qenv, s)
        out[f"fixed_{scan}_s{sd}"] = X.cut(ph, r0 - k[0] - kk[0], c0 - k[1] - kk[1], h, w)
    print(scan, [sd for sd, _ in fx[:2]], flush=True)
np.savez_compressed(X.OUT / "e1_fixed_crops.npz", **out)
print("wrote", X.OUT / "e1_fixed_crops.npz")
