"""Blind runs from a meaningfully wrong probe (stage 6, *_pe) against the same conditions
with the known probe and with the first blind runs (20 A defocus error).

    python pe_results.py        # writes figures/pe_results.{md,csv}

Non-uniqueness is counted by dataset as in draw_level.py.  For every stored refined probe
the overlap with the true probe, |<P, T(s) P_true>| / (|P| |P_true|), is maximized over a
sub-pixel translation s (probe and object may move together, Eq. 5), and the residual
aperture-plane phase error is the intensity-weighted rms of arg(P~ conj(P~_true)) after
removing that translation and a constant.  The starting probe gets the same two numbers.
"""
import numpy as np
import pandas as pd

import draw_level
import paths
import sim

ROWS = [  # (label, dose, known-probe sweep, first blind sweep, wrong-start blind sweep)
    ("raster, free-standing", "inf", "ss_bare_complex_inf", "rev/grid_blind_bare_inf",
     "rev/grid_blind_pe_bare_inf"),
    ("raster, free-standing", "1e5", "ss_bare_complex_1e5", "rev/grid_blind_bare_1e5",
     "rev/grid_blind_pe_bare_1e5"),
    ("raster, 2.8 nm a-C", "1e4", "rev/ss_carbon28_complex_1e4", "rev/grid_blind_carbon28_1e4",
     "rev/grid_blind_pe_carbon28_1e4"),
    ("jittered, free-standing", "inf", "rev/jit_known_bare_inf", "rev/jit_blind_bare_inf",
     "rev/jit_blind_pe_bare_inf"),
    ("jittered, free-standing", "1e5", "rev/jit_known_bare_1e5", "rev/jit_blind_bare_1e5",
     "rev/jit_blind_pe_bare_1e5"),
    ("jittered, 2.8 nm a-C", "1e4", "rev/jit_known_carbon28_1e4", "rev/jit_blind_carbon28_1e4",
     "rev/jit_blind_pe_carbon28_1e4"),
    ("N9 multislice", "1e5", None, "rev/mm_ms_blind_1e5", "rev/mm_ms_blind_pe_1e5"),
    ("N9 multislice", "2e5", "rev/mm_ms_known_2e5", "rev/mm_ms_blind_2e5",
     "rev/mm_ms_blind_pe_2e5"),
    ("N9 all three", "1e5", "rev/mm_msall_known_1e5", "rev/mm_msall_blind_1e5",
     "rev/mm_msall_blind_pe_1e5"),
    ("N9 all three", "2e5", "rev/mm_msall_known_2e5", "rev/mm_msall_blind_2e5",
     "rev/mm_msall_blind_pe_2e5"),
]


def match(P, Pt):
    """Overlap and aperture phase error of P against Pt, best sub-pixel translation."""
    r = register(P, Pt)
    return r["overlap"], r["rms"], r["shift_A"]


def register(P, Pt, step=0.05, half=1.0):
    """P moved onto Pt by the best sub-pixel translation and scaled by the least-squares
    complex factor (scale and global phase are free in ePIE), with the overlap, the
    aperture-plane phase difference (NaN outside the aperture), its intensity-weighted rms,
    and the translation in Angstrom."""
    n = P.shape[0]
    F, Ft = np.fft.fft2(P), np.fft.fft2(Pt)
    c = np.fft.ifft2(F * np.conj(Ft))
    i0 = np.array(np.unravel_index(np.argmax(np.abs(c)), c.shape), dtype=float)
    i0[i0 > n // 2] -= n
    k = np.fft.fftfreq(n)
    d = np.arange(-half, half + step / 2, step)
    X = F * np.conj(Ft)
    # the cross-correlation on a fine grid of translations around the integer peak, as one
    # matrix DFT: C[a, b] = sum_kx,ky Ex[a, kx] X[kx, ky] Ey[b, ky]
    Ex = np.exp(2j * np.pi * (i0[0] + d)[:, None] * k[None, :])
    Ey = np.exp(2j * np.pi * (i0[1] + d)[:, None] * k[None, :])
    C = np.abs(Ex @ X @ Ey.T)
    a, b = np.unravel_index(np.argmax(C), C.shape)
    v, s = C[a, b], (i0[0] + d[a], i0[1] + d[b])
    ov = v / (np.linalg.norm(F) * np.linalg.norm(Ft))
    # aperture-plane phase after the translation, intensity-weighted, constant removed
    # P(r) ~ Pt(r - s), i.e. F ~ Ft exp(-2 pi i k.s): undo the ramp
    ramp = np.exp(2j * np.pi * (k[:, None] * s[0] + k[None, :] * s[1]))
    w = np.abs(Ft) ** 2
    w = w * (w > 1e-3 * w.max())
    G = F * ramp
    z = G * np.conj(Ft)
    z0 = np.sum(w * z) / np.abs(np.sum(w * z))
    ph = np.angle(z * np.conj(z0))
    rms = float(np.sqrt(np.sum(w * ph ** 2) / np.sum(w)))
    Pr = np.fft.ifft2(G)
    Pr = Pr * np.vdot(Pr, Pt) / np.vdot(Pr, Pr)
    return dict(P=Pr, overlap=float(ov), phase=np.where(w > 0, ph, np.nan), rms=rms,
                shift_A=(float(s[0]) * sim.L / n, float(s[1]) * sim.L / n))


def probes(sweep, angle=3.0):
    out = []
    for f in sorted((paths.sweep(sweep) / "runs").glob(f"a{angle:g}_s*.npz")):
        z = np.load(f)
        if "probe" in z:
            out.append((f.name, z["probe"]))
    return out


def dataset_rate(sweeps):
    D = draw_level.datasets([s for s in sweeps if s])
    if not len(D):
        return "--", np.nan, 0
    k = int((D[:, 0] > 0).sum())
    return f"{k}/{len(D)}", D[:, 0].sum() / D[:, 1].sum(), len(D)


def main():
    angle = 3.0
    Pt = sim.make_probe(angle, sim.ANGLES[angle])
    P0 = sim.make_probe(angle, sim.ANGLES[angle] + 1500, C12=150, phi12=np.pi / 6)
    ov0, rms0, _ = match(P0, Pt)
    rows = []
    for lab, dose, known, old, pe in ROWS:
        r = dict(condition=lab, dose=dose)
        for tag, sw in (("known", known), ("blind20", old), ("blind_pe", pe)):
            ds, g, n = dataset_rate([sw]) if sw else ("--", np.nan, 0)
            r[f"{tag}_datasets"], r[f"{tag}_groups"] = ds, g
            if sw:
                q = draw_level.quality([sw])
                r[f"{tag}_acorr"], r[f"{tag}_reversed"] = q[0], q[2]
        M = [match(P, Pt) for _, P in probes(pe)]
        if M:
            ov = np.array([m[0] for m in M])
            rm = np.array([m[1] for m in M])
            sh = np.array([np.hypot(*m[2]) for m in M])
            r.update(n_probes=len(M), overlap_min=ov.min(), overlap_med=np.median(ov),
                     phase_rms_med=np.median(rm), phase_rms_max=rm.max(),
                     shift_med_A=np.median(sh))
        rows.append(r)
    T = pd.DataFrame(rows)
    out = paths.figures()
    T.to_csv(out / "pe_results.csv", index=False)

    def f(v, p=2):
        return v if isinstance(v, str) else ("--" if not np.isfinite(v) else f"{v:.{p}f}")

    md = [f"Starting probe (1500 A defocus + 150 A astigmatism error): overlap {ov0:.3f}, "
          f"aperture phase rms {rms0:.2f} rad.\n",
          "| condition | dose | known: datasets | groups | first blind: datasets | groups | "
          "wrong start: datasets | groups | \\|corr\\| | refined probe overlap (min / median) "
          "| phase rms (median / max) | shift (A) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in T.to_dict("records"):
        md.append(
            f"| {r['condition']} | {r['dose']} | {r['known_datasets']} | "
            f"{f(r['known_groups'])} | {r['blind20_datasets']} | {f(r['blind20_groups'])} | "
            f"{r['blind_pe_datasets']} | {f(r['blind_pe_groups'])} | "
            f"{f(r.get('blind_pe_acorr', np.nan), 3)} | "
            f"{f(r.get('overlap_min', np.nan), 3)} / {f(r.get('overlap_med', np.nan), 3)} | "
            f"{f(r.get('phase_rms_med', np.nan))} / {f(r.get('phase_rms_max', np.nan))} | "
            f"{f(r.get('shift_med_A', np.nan))} |")
    (out / "pe_results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
