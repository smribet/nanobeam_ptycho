"""Multislice forward model: transmission/propagation over an abTEM-sliced potential.

Written out explicitly rather than calling abtem.Probe.multislice so that the antialiasing,
the frozen-phonon average and the final crop are all under our control and match what the
single-slice reconstruction expects.

The grid is oversampled by 2x (0.1 A instead of 0.2 A), band-limited by a circular aperture
at half the simulation k_max, and the diffraction patterns are cropped back to the inner
half.  The reconstruction therefore sees exactly the 340^2 k-space it sees today, but the
propagation itself is free of wrap-around aliasing.

Thermal motion is frozen-phonon: independent Gaussian displacements per configuration, with
intensities -- not amplitudes -- averaged, which is what makes it incoherent.
"""
import numpy as np
import abtem

import sim

OVERSAMPLE = 2
SLICE_THICKNESS = 2.0
# rms thermal displacement, Angstrom.  Au at 300 K has B ~ 0.6 A^2 -> u ~ 0.085 A;
# carbon in an amorphous film is softer and already positionally disordered.
SIGMAS = {"Au": 0.085, "C": 0.100}


def grids(gpts=None):
    g = gpts or sim.GPTS * OVERSAMPLE
    return g, sim.L / g


def displaced(atoms, rng, sigmas=SIGMAS):
    """One frozen-phonon configuration."""
    a = atoms.copy()
    s = np.array([sigmas.get(sym, 0.09) for sym in a.get_chemical_symbols()])[:, None]
    a.positions = a.positions + rng.normal(size=a.positions.shape) * s
    return a


def slice_potential(atoms, gpts, slice_thickness=SLICE_THICKNESS):
    """Sliced projected potential, shape (n_slices, gpts, gpts), in V*Angstrom."""
    pot = abtem.Potential(atoms, gpts=(gpts, gpts), slice_thickness=slice_thickness).build()
    return np.asarray(pot.array), float(pot.sampling[0])


def _kgrid(gpts, sampling):
    k = np.fft.fftfreq(gpts, sampling)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    return KX, KY, np.hypot(KX, KY)


def propagator(gpts, sampling, dz, energy, antialias_frac=0.5):
    """Fresnel propagator with a circular antialiasing aperture already applied."""
    lam = abtem.core.energy.energy2wavelength(energy)
    _, _, k = _kgrid(gpts, sampling)
    kmax = 1.0 / (2 * sampling)
    aperture = (k < antialias_frac * kmax).astype(np.complex64)
    return (np.exp(-1j * np.pi * lam * dz * k ** 2).astype(np.complex64) * aperture,
            aperture)


def crop_centre(dp_shifted, out):
    """Inner `out` x `out` of an fftshifted pattern -- the half-k_max the crop retains."""
    n = dp_shifted.shape[-1]
    lo = (n - out) // 2
    return dp_shifted[..., lo:lo + out, lo:lo + out]


def crop_to_recon_grid(dp_unshifted, out):
    """Band-limit crop, returned in the UNSHIFTED layout the reconstruction expects.

    ePIE compares against np.fft.fft2 output, which puts k = 0 at index [0, 0].  Cropping
    requires the shifted layout, so shift, crop, and shift back -- forgetting the last step
    hands the reconstruction a pattern displaced by half the array, which it will happily
    fit to a low error while returning a meaningless object.
    """
    sh = np.fft.fftshift(dp_unshifted, axes=(-2, -1))
    return np.fft.ifftshift(crop_centre(sh, out), axes=(-2, -1))


def multislice_batch(P, V, prop, sigma_e, aperture):
    """Propagate a batch of probes (n, g, g) through the sliced potential.

    Returns |FFT|^2 on the simulation grid, unshifted.
    """
    psi = P.copy()
    n_sl = V.shape[0]
    for i in range(n_sl):
        psi *= np.exp(1j * sigma_e * V[i]).astype(np.complex64)
        if i < n_sl - 1:
            psi = np.fft.ifft2(np.fft.fft2(psi) * prop)
    return np.abs(np.fft.fft2(psi) * aperture) ** 2


def probe_batch(positions, semiangle, defocus, gpts, energy):
    """Complex probes centred at each (x, y), shape (n, gpts, gpts)."""
    P0 = centred_probe(semiangle, defocus, gpts, energy)
    s = sim.L / gpts
    out = np.empty((len(positions), gpts, gpts), dtype=np.complex64)
    for n, (px, py) in enumerate(positions):
        out[n] = np.roll(P0, (int(round(px / s)), int(round(py / s))), axis=(0, 1))
    return out


def centred_probe(semiangle, defocus, gpts, energy):
    pr = abtem.Probe(energy=energy, semiangle_cutoff=semiangle, defocus=defocus,
                     extent=[sim.L, sim.L], gpts=(gpts, gpts))
    return np.squeeze(pr.build(abtem.CustomScan([[0.0, 0.0]])).array).astype(np.complex64)


# ---------------------------------------------------------------- torch path
#
# The same transmission/propagation chain on a GPU.  Transmission functions are built once
# per configuration rather than once per batch, and probes are rolled on the device, so the
# only host transfer per batch is the cropped patterns.  `numpy_vs_torch` checks the two
# paths against each other.

def device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def transmission_torch(V, sigma_e, dev):
    import torch
    return torch.from_numpy(np.exp(1j * sigma_e * V).astype(np.complex64)).to(dev)


def probe_batch_torch(P0, positions, sampling):
    """Rolled copies of the centred probe tensor `P0`, one per (x, y) in Angstrom."""
    import torch
    return torch.stack([torch.roll(P0, (int(round(px / sampling)), int(round(py / sampling))),
                                   dims=(0, 1)) for px, py in positions])


def multislice_batch_torch(P, T, prop, aperture):
    """Torch twin of `multislice_batch`, with T = exp(i sigma V) precomputed per slice."""
    import torch
    psi = P
    for i in range(T.shape[0]):
        psi = psi * T[i]
        if i < T.shape[0] - 1:
            psi = torch.fft.ifft2(torch.fft.fft2(psi) * prop)
    F = torch.fft.fft2(psi) * aperture
    return F.real ** 2 + F.imag ** 2


def crop_to_recon_grid_torch(dp_unshifted, out):
    import torch
    sh = torch.fft.fftshift(dp_unshifted, dim=(-2, -1))
    n = sh.shape[-1]
    lo = (n - out) // 2
    return torch.fft.ifftshift(sh[..., lo:lo + out, lo:lo + out], dim=(-2, -1))


def numpy_vs_torch(atoms, angle, defocus, n_pos=8, gpts=None):
    """Max relative difference between the two paths on a few probe positions."""
    import torch
    g, s = grids(gpts)
    sigma_e = abtem.core.energy.energy2sigma(sim.ENERGY)
    prop, aper = propagator(g, s, SLICE_THICKNESS, sim.ENERGY)
    V, _ = slice_potential(atoms, g)
    pos = [(i * 7.3 % sim.L, i * 11.9 % sim.L) for i in range(n_pos)]
    ref = multislice_batch(probe_batch(pos, angle, defocus, g, sim.ENERGY), V, prop,
                           sigma_e, aper)
    dev = device()
    P0 = torch.from_numpy(centred_probe(angle, defocus, g, sim.ENERGY)).to(dev)
    out = multislice_batch_torch(probe_batch_torch(P0, pos, s),
                                 transmission_torch(V, sigma_e, dev),
                                 torch.from_numpy(prop).to(dev),
                                 torch.from_numpy(aper).to(dev)).cpu().numpy()
    return float(np.abs(out - ref).max() / np.abs(ref).max())


def coherence_samples(source_sigma=0.0, focal_sigma=0.0, n_source=5, n_focal=5):
    """Quadrature for partial coherence: [(dx, dy, ddefocus, weight)], weights summing to 1.

    Spatial: the demagnified source is a Gaussian of rms `source_sigma` (Angstrom) in
    each direction, sampled on an n_source x n_source grid out to +-2 sigma.  Temporal:
    a Gaussian focal spread of rms `focal_sigma` (Angstrom), sampled at n_focal points out
    to +-2 sigma.  Both are incoherent, so their intensities -- not amplitudes -- add.
    """
    def axis(sig, n):
        if not sig:
            return np.zeros(1), np.ones(1)
        x = np.linspace(-2 * sig, 2 * sig, n)
        w = np.exp(-x ** 2 / (2 * sig ** 2))
        return x, w / w.sum()
    xs, ws = axis(source_sigma, n_source)
    fs, wf = axis(focal_sigma, n_focal)
    return [(dx, dy, df, wx * wy * wd)
            for dx, wx in zip(xs, ws) for dy, wy in zip(xs, ws) for df, wd in zip(fs, wf)]
