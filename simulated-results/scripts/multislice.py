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
    pr = abtem.Probe(energy=energy, semiangle_cutoff=semiangle, defocus=defocus,
                     extent=[sim.L, sim.L], gpts=(gpts, gpts))
    P0 = np.squeeze(pr.build(abtem.CustomScan([[0.0, 0.0]])).array).astype(np.complex64)
    s = sim.L / gpts
    out = np.empty((len(positions), gpts, gpts), dtype=np.complex64)
    for n, (px, py) in enumerate(positions):
        out[n] = np.roll(P0, (int(round(px / s)), int(round(py / s))), axis=(0, 1))
    return out
