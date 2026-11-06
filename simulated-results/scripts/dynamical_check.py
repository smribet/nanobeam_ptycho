"""Below the overlap threshold, do the patterns of a thick crystal depend on probe position?

    python dynamical_check.py

Equation (4) is derived for a transmission function.  For a crystal that is periodic in
the plane, Bloch's theorem extends it to dynamical scattering at any thickness: every
pattern is a sum of |A(k-g)|^2 |Phi_g(k-g)|^2 terms, with no dependence on the probe
position R unless two apertures overlap.  A translation of the crystal is a translation of
the probe, so this is also the statement that the translation gauge of Eq. (5) survives
multiple scattering.

The check: a perfect Au [100] slab in a laterally commensurate periodic cell, simulated
with the same multislice chain as the sweeps, with the probe placed at several positions
inside one unit cell.  The reported number is max_R ||I_R - <I>_R|| / ||<I>_R||, which
should sit at float32 round-off below the {200} threshold (4.83 mrad) and be of order
unity above it.  Writes figures/dynamical_check.json.

The threshold that matters is set by every beam in the exit wave, not only by the
zero-order Laue zone.  For Au [001] the first-order Laue zone {111} reflections have
transverse components of {110} type, sqrt(2)/a = 0.347 1/A, closer to the direct beam than
{200}; they overlap it above 3.41 mrad.  They are weakly excited (large excitation error),
so between 3.41 and 4.83 mrad the patterns depend on position only at the 1e-3 level,
while below 3.41 mrad the dependence is at round-off for every thickness.

The slices are one atomic layer (a/2) thick with each layer at a slice centre, so the
sliced model has exactly the crystal's periodicity along the beam.
"""
import json, time

import numpy as np
import abtem
from ase.build import bulk

import sim
import multislice as ms
import paths

N_CELLS = 17                       # 17 x 4.08 A = 69.4 A, close to the sweeps' field
GPTS_SIM = 680
THICKNESSES = (20.0, 50.0, 100.0)  # Angstrom
ANGLES = (3.0, 4.0, 6.0, 12.0)     # mrad; {200} overlaps above 4.83, {220} above 6.82
# probe positions inside one unit cell, in simulation pixels (0.102 A)
SHIFTS = [(0, 0), (10, 0), (0, 15), (20, 20), (7, 31), (33, 12)]


def main():
    abtem.config.set({"dask.lazy": False})
    import torch
    dev = ms.device()
    L = N_CELLS * sim.LATTICE
    s = L / GPTS_SIM
    sigma_e = abtem.core.energy.energy2sigma(sim.ENERGY)
    lam = abtem.core.energy.energy2wavelength(sim.ENERGY)
    k = np.fft.fftfreq(GPTS_SIM, s)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    band = torch.from_numpy(np.hypot(KX, KY) < 0.5 / (2 * s)).to(dev)   # antialiased
    dz = sim.LATTICE / 2                                  # one atomic layer per slice
    prop, aper = ms.propagator(GPTS_SIM, s, dz, sim.ENERGY)
    prop_t, aper_t = torch.from_numpy(prop).to(dev), torch.from_numpy(aper).to(dev)

    rows = []
    for t in THICKNESSES:
        nz = int(round(t / sim.LATTICE))
        atoms = bulk("Au", "fcc", a=sim.LATTICE, cubic=True) * (N_CELLS, N_CELLS, nz)
        atoms.positions[:, 2] += dz / 2                   # each layer at a slice centre
        atoms.cell[2, 2] = nz * sim.LATTICE
        atoms.pbc = [True, True, False]
        V = np.asarray(abtem.Potential(atoms, gpts=(GPTS_SIM, GPTS_SIM),
                                       slice_thickness=dz).build().array)
        T = ms.transmission_torch(V, sigma_e, dev)
        for a in ANGLES:
            t0 = time.time()
            pr = abtem.Probe(energy=sim.ENERGY, semiangle_cutoff=a, defocus=0.0,
                             extent=[L, L], gpts=(GPTS_SIM, GPTS_SIM))
            P0 = torch.from_numpy(np.squeeze(pr.build(
                abtem.CustomScan([[0.0, 0.0]])).array).astype(np.complex64)).to(dev)
            P = torch.stack([torch.roll(P0, sh, dims=(0, 1)) for sh in SHIFTS])
            I = ms.multislice_batch_torch(P, T, prop_t, aper_t) * band
            mean = I.mean(0)
            dev_R = ((I - mean) ** 2).sum((1, 2)).sqrt() / (mean ** 2).sum().sqrt()
            rel = float(dev_R.max())
            g200 = 2 / sim.LATTICE
            rows.append(dict(thickness_A=t, n_slices=int(V.shape[0]), alpha_mrad=a,
                             overlaps_200=bool(a * 1e-3 / lam > g200 / 2),
                             max_rel_dev=rel, seconds=round(time.time() - t0, 1)))
            print(f"t = {t/10:4.0f} nm  alpha = {a:4.1f} mrad  "
                  f"{'overlap   ' if rows[-1]['overlaps_200'] else 'no overlap'}  "
                  f"max_R |I_R - <I>| / |<I>| = {rel:.2e}", flush=True)

    out = paths.figures() / "dynamical_check.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
