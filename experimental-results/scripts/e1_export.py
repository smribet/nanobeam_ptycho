"""Everything the response figures need from the experimental re-reconstructions, small.

    python e1_export.py        # writes recons/e1/export/e1_maps.npz and e1_phases.npz

e1_maps.npz      for every engine and scan: the local agreement of its second seed with its
                 first (register_lattice.local_agreement) and the particle area, both
                 block-averaged 4 x 4; the agreement fractions of every seed; and, at 24 and
                 3 mrad, the agreement between engines after each scan is aligned on its own
                 lattice (compare_engines.search), in quantEM's frame
e1_phases.npz    the final phases of quantEM seeds 0 and 1 at 24 and 1.5 mrad, full size, for
                 the Fig. 3 overlay and the location of the field of view in Fig. S1
"""
import numpy as np

import register_lattice as R
import compare_engines as C

E1 = R.E1
OUT = E1 / "export"
ENGINES = {"quantEM": E1, "ePIE": E1 / "epie", "PtyRAD": E1 / "ptyrad"}
SCANS = (23, 24, 25, 26, 27)


def block(a, k=4):
    n = (a.shape[0] // k) * k
    return a[:n, :n].reshape(n // k, k, n // k, k).mean((1, 3))


def seeds(d, scan):
    R.E1 = d
    fs = [f for f in R.runs(scan) if "fixedprobe" not in f.name and "crop" not in f.name]
    out = []
    for f in fs:
        ph, s = R.phases(f)
        out.append((int(f.name.split("_s")[1][:3]), ph[max(ph)], s))
    return sorted(out, key=lambda t: t[0])


def main():
    OUT.mkdir(exist_ok=True)
    maps = {}
    for eng, d in ENGINES.items():
        for scan in SCANS:
            runs = seeds(d, scan)
            if len(runs) < 2:
                continue
            ref, s = runs[0][1], runs[0][2]
            env = R.envelope(ref, s)
            on = R.particle_area(ref, s)
            fr = []
            for i, (sd, ph, _) in enumerate(runs[1:]):
                m = R.local_agreement(R.lattice(R.align(ph, env, s)[0], s), R.lattice(ref, s), s)
                fr.append(((m[on] > 0.5).mean(), (m[on] < -0.5).mean()))
                if i == 0:
                    maps[f"{eng}_{scan}_map"] = block(m).astype(np.float16)
            maps[f"{eng}_{scan}_on"] = block(on.astype(float)) > 0.5
            maps[f"{eng}_{scan}_fractions"] = np.array(fr)
            maps[f"{eng}_{scan}_sampling"] = s * 4
            print(f"{eng:8s} scan {scan}: {len(runs)} seeds, agree "
                  + "/".join(f"{a:.2f}" for a, _ in fr), flush=True)

    # between engines, each scan aligned on its own lattice from the frame found at 24 mrad
    ph = {n: C.final_phase(d, 23) for n, d in C.ENGINES.items()}
    s = ph["quantEM"][1]
    n = max(v[0].shape[0] for v in ph.values())
    ref = C.padded(ph["quantEM"][0], n)
    base = {b: C.register(C.padded(ph[b][0], n), ref, s) for b in ("ePIE", "PtyRAD")}
    for scan in (23, 26):
        ph = {nm: C.final_phase(d, scan) for nm, d in C.ENGINES.items()}
        q = C.padded(ph["quantEM"][0], n)
        on = R.particle_area(q, s)
        for b in ("ePIE", "PtyRAD"):
            mov = C.padded(ph[b][0], n)
            f = C.search(mov, q, s, base[b])
            al = np.sign(f["envelope_corr"]) * C.warp(mov, f["rot90"], f["transpose"],
                                                      f["rotation"], f["shift"])
            m = R.local_agreement(R.lattice(al, s), R.lattice(q, s), s)
            maps[f"cross_{scan}_quantEM_{b}_map"] = block(m).astype(np.float16)
            maps[f"cross_{scan}_quantEM_{b}_fractions"] = np.array(
                [(m[on] > 0.5).mean(), (m[on] < -0.5).mean()])
            print(f"scan {scan} quantEM vs {b}: agree {(m[on] > 0.5).mean():.2f}", flush=True)
        maps[f"cross_{scan}_on"] = block(on.astype(float)) > 0.5
    np.savez_compressed(OUT / "e1_maps.npz", **maps)

    phases = {}
    for scan in (23, 27):
        for sd, p, s in seeds(E1, scan)[:2]:
            phases[f"scan{scan}_s{sd}"] = p.astype(np.float32)
        phases["sampling"] = s
    np.savez_compressed(OUT / "e1_phases.npz", **phases)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
