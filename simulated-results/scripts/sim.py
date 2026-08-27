"""Forward model and ePIE engine for the non-uniqueness sweep.

Three Au nanoparticles -- [111], [100] and a five-fold twin -- on a common field,
imaged at 300 keV with the convergence angle swept across the disk-overlap
threshold.  The probe is defocused at the larger angles so that its real-space
extent is held fixed, isolating reciprocal-space overlap as the only variable.

The data are generated with the same single-slice forward model that ePIE inverts,
so any non-uniqueness that survives is fundamental rather than model error.
"""
import numpy as np
import abtem
import ase.build
from ase.build import bulk
from ase.cluster.cubic import FaceCenteredCubic
from wulffpack import Decahedron

abtem.config.set({"dask.lazy": False})

ENERGY = 300e3
LATTICE = 4.08
L = 68.0                 # field of view, Angstrom
GPTS = 340               # 0.2 A sampling -> 49 mrad max collection angle
THICK = 36.0
THICK_SUPPORT = 44.0     # same cell, with room underneath for the carbon film
STEP = 2.0               # scan step, Angstrom

# probe defocus chosen so every angle has the same real-space extent (r80 ~ 2.8 A)
ANGLES = {3.0: 0.0, 6.0: 390.0, 12.0: 250.0}

# particle centres, and the reflection coupling each angle achieves
PARTICLES = {
    "au_111": (19.0, 46.0),
    "au_100": (49.0, 46.0),
    "decahedron": (34.0, 19.0),
}
COUPLING = {3.0: "none", 6.0: "{111},{200}", 12.0: "all"}

_REF = dict(energy=ENERGY, angles=dict(ANGLES), coupling=dict(COUPLING))


def set_energy(energy):
    """Switch accelerating voltage, holding the diffraction geometry fixed.

    Disk overlap is governed by alpha/lambda, so alpha scales with lambda.  The probe's
    real-space extent depends on lambda/alpha -- unchanged by that scaling -- and on the
    aberration function written in normalised aperture coordinates u = k*lambda/alpha,
    where the defocus term is chi = pi * defocus * alpha^2 * u^2 / lambda; holding that
    fixed requires the defocus to scale as 1/lambda.  Verified: the resulting probes match
    the 300 keV ones to four decimals at r50, r80 and r90.

    With both rules applied the probe is the *same function of position* and the disks
    overlap in the *same way* at every energy, so the interaction constant sigma is the
    only thing that changes -- which is the point of the comparison.

    Note this varies sigma within the projected-potential (single-slice) model.  That
    still contains the substrate-crystal cross terms, since exp(i*sigma*V_Au) *
    exp(i*sigma*V_C) is a product and hence a convolution in reciprocal space, but it does
    not contain genuine three-dimensional multiple scattering, which needs multislice.
    """
    global ENERGY, ANGLES, COUPLING
    r = (abtem.core.energy.energy2wavelength(energy)
         / abtem.core.energy.energy2wavelength(_REF["energy"]))
    ENERGY = energy
    ANGLES = {round(a * r, 2): d / r for a, d in _REF["angles"].items()}
    COUPLING = {round(a * r, 2): c for a, c in _REF["coupling"].items()}
    return ANGLES


def amorphous_carbon(thickness=THICK_SUPPORT, seed=10):
    """Amorphous carbon support, as a slab under the particles.

    A diamond lattice jittered by 0.5 of a bond length, which gives a diffuse ring and no
    Bragg reflections -- the point being that the support is invisible to any method that
    works from detected reflections.
    """
    n = int(np.ceil(L / 3.567))
    c = ase.build.bulk("C", cubic=True) * (n, n, 4)
    rng = np.random.default_rng(seed)
    c.positions[:] += rng.normal(size=(len(c), 3)) * 0.5 * 1.54
    c.cell = [L, L, thickness]
    c.wrap()
    del c[c.positions[:, 0] > L]
    del c[c.positions[:, 1] > L]
    c.positions[:, 2] = c.positions[:, 2] % 8 + 33
    return c


def build_sample(support=False):
    """Three Au nanoparticles: [111] single, [100] single, five-fold twin.

    With `support`, an amorphous carbon film is added underneath and the cell is
    deepened to hold it; the Au positions are unchanged either way.
    """
    def place(c, centre):
        c.cell = [L, L, THICK]
        c.center()
        c.positions[:, 0] += centre[0] - L / 2
        c.positions[:, 1] += centre[1] - L / 2
        return c

    def single(zone, twist, centre):
        c = FaceCenteredCubic("Au", [(1, 0, 0), (1, 1, 0), (1, 1, 1)], [5, 7, 4],
                              latticeconstant=LATTICE)
        if zone is not None:
            c.rotate(zone, (0, 0, 1), rotate_cell=False)
        c.rotate(twist, "z")
        return place(c, centre)

    deca = Decahedron({(1, 0, 0): 1.25, (1, 1, 1): 1.0, (1, 1, 0): 1.05},
                      primitive_structure=bulk("Au"), twin_energy=0.125,
                      natoms=400).atoms
    deca.rotate(19, "z")

    atoms = (single((1, 1, 1), 10, PARTICLES["au_111"])
             + single(None, -7, PARTICLES["au_100"])
             + place(deca, PARTICLES["decahedron"]))
    thick = THICK_SUPPORT if support else THICK
    if support:
        atoms = atoms + amorphous_carbon(thick)
    atoms.cell = [L, L, thick]
    atoms.pbc = [True, True, False]
    return atoms


def projected_object(atoms):
    """Single-slice complex transmission function and its projected potential."""
    pot = abtem.Potential(atoms, gpts=(GPTS, GPTS),
                          slice_thickness=float(atoms.cell[2, 2])).build()
    proj = pot.array[0]
    sigma = abtem.core.energy.energy2sigma(ENERGY)
    return np.exp(1j * sigma * proj).astype(np.complex64), proj, pot.sampling[0]


def make_probe(semiangle, defocus):
    """Probe wavefunction centred on the array origin."""
    pr = abtem.Probe(energy=ENERGY, semiangle_cutoff=semiangle, defocus=defocus,
                     extent=[L, L], gpts=(GPTS, GPTS))
    P = np.squeeze(pr.build(abtem.CustomScan([[0.0, 0.0]])).array)
    return P.astype(np.complex64)


def scan_pixels(sampling):
    g = np.arange(STEP / 2, L, STEP)
    return [(int(round(px / sampling)), int(round(py / sampling)))
            for px in g for py in g]


def simulate_data(P, obj, pixels):
    """Noiseless diffraction intensities, one per scan position."""
    return [np.abs(np.fft.fft2(np.roll(P, (i, j), axis=(0, 1)) * obj)).astype(np.float32)**2
            for i, j in pixels]


def apply_dose(patterns, dose, seed=0):
    """Poisson counting noise at `dose` electrons per square Angstrom of scanned area.

    Each probe position collects dose * STEP**2 electrons, distributed over the pattern in
    proportion to its intensity.  The result is rescaled back to the original total so the
    reconstruction sees the same normalisation as the noiseless case.
    """
    if dose is None:
        return patterns
    rng = np.random.default_rng(seed)
    n_e = dose * STEP ** 2
    out = []
    for p in patterns:
        tot = float(p.sum())
        out.append((rng.poisson(n_e * (p / tot)).astype(np.float32) / n_e * tot))
    return out


def best_device():
    """Fastest available torch device, or None to fall back to numpy."""
    try:
        import torch
    except ImportError:
        return None
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def orthogonalize(modes):
    """Rotate mode stack (M, N, N) to an orthogonal set ordered by occupancy.

    The modelled intensity sum_m |FFT(P O_m)|^2 is invariant under any unitary rotation
    among the modes, so this changes nothing about the fit -- it fixes a gauge.  That
    matters here because every downstream metric compares *the* object against ground
    truth, and without gauge fixing there is no unique object to compare: the modes would
    be an arbitrary mixture.  It also keeps the modes from collapsing onto one another.
    """
    M = modes.shape[0]
    flat = modes.reshape(M, -1)
    w, U = np.linalg.eigh(flat @ flat.conj().T)
    idx = np.argsort(w)[::-1]
    occ = np.clip(w[idx], 0, None)
    occ = occ / max(occ.sum(), 1e-30)
    return (U[:, idx].conj().T @ flat).reshape(modes.shape), occ


def epie(P, data, pixels, n_iter, seed, beta=0.9, callback=None, backend="auto",
         obj_model="complex", n_modes=4, ortho_every=1, info=None):
    """Single-slice ePIE with a known, fixed probe.

    The update order and the initial guess are drawn with numpy for the given seed, so
    a run is reproducible and directly comparable across backends.  Returns
    (object, error history).

    `obj_model`:
      "complex" -- the usual free complex object, 2 real unknowns per pixel.
      "phase"   -- a pure phase object O = exp(i*phi), 1 real unknown per pixel, updated
                   by gradient descent on phi directly.  With the probe held fixed there
                   is no O-P scaling ambiguity to absorb, but a free complex object can
                   still soak up whatever the forward model cannot explain -- dynamical
                   scattering, say -- as amplitude modulation.  Constraining |O| = 1
                   closes that escape route, so a model error has to show up as a phase
                   error where it can be seen.
    """
    if backend == "auto":
        backend = "torch" if best_device() else "numpy"
    if obj_model == "mixed":
        return _epie_mixed(P, data, pixels, n_iter, seed, beta, callback, backend,
                           n_modes, ortho_every, info)
    if obj_model == "phase":
        f = _epie_phase_torch if backend == "torch" else _epie_phase_numpy
        return f(P, data, pixels, n_iter, seed, beta, callback)
    if backend == "torch":
        return _epie_torch(P, data, pixels, n_iter, seed, beta, callback)
    return _epie_numpy(P, data, pixels, n_iter, seed, beta, callback)


def _epie_mixed(P, data, pixels, n_iter, seed, beta, callback, backend, n_modes,
                ortho_every, info):
    """Mixed-state OBJECT: one probe, several mutually incoherent object modes.

    Thermal motion varies the specimen and not the illumination, so the incoherence belongs
    to the object.  Intensities add across modes, amplitudes do not, which is exactly what
    a frozen-phonon average produces.
    """
    xp, dev = (np, None)
    if backend == "torch":
        import torch
        xp, dev = torch, torch.device(best_device())

    rng = np.random.default_rng(seed)
    # modes must start different or the symmetry is never broken; higher modes start weak
    O = np.stack([np.exp(1j * rng.uniform(-0.2, 0.2, size=P.shape))
                  * (1.0 if m == 0 else 0.10)
                  for m in range(n_modes)]).astype(np.complex64)
    order = np.random.default_rng(seed)
    total = float(sum(float(d.sum()) for d in data))
    history, occ = [], np.zeros(n_modes)

    if dev is not None:
        Pt = xp.from_numpy(P).to(dev)
        Om = xp.from_numpy(O).to(dev)
        amps = [xp.from_numpy(np.sqrt(np.asarray(d))).to(dev) for d in data]
        Pmax = (Pt.abs() ** 2).max()
    else:
        Pt, Om, Pmax = P, O, np.float32((np.abs(P) ** 2).max())
        amps = [np.sqrt(np.asarray(d)) for d in data]

    for _ in range(n_iter):
        err = 0.0
        for k in order.permutation(len(pixels)):
            i, j = pixels[k]
            Pr = (xp.roll(Pt, (int(i), int(j)), dims=(0, 1)) if dev is not None
                  else np.roll(Pt, (i, j), axis=(0, 1)))
            psi = Pr[None] * Om
            Psi = xp.fft.fft2(psi)
            model = (Psi.real ** 2 + Psi.imag ** 2).sum(0) if dev is not None \
                else (np.abs(Psi) ** 2).sum(0)
            amp = amps[k]
            mag = model ** 0.5
            err += float(((mag - amp) ** 2).sum())
            scale = amp / (mag + 1e-12)          # one factor, shared by every mode
            corrected = xp.fft.ifft2(Psi * scale[None])
            Om = Om + beta * (Pr.conj()[None] / Pmax) * (corrected - psi)
        if ortho_every and (len(history) + 1) % ortho_every == 0:
            arr = Om.cpu().numpy() if dev is not None else Om
            arr, occ = orthogonalize(np.asarray(arr))
            Om = xp.from_numpy(arr).to(dev) if dev is not None else arr
        history.append(float(np.sqrt(err / total)))
        if callback is not None:
            dom = (Om[0].cpu().numpy() if dev is not None else Om[0])
            callback(len(history), dom, history[-1])

    arr = Om.cpu().numpy() if dev is not None else Om
    arr, occ = orthogonalize(np.asarray(arr))
    if info is not None:
        info["occupancy"] = occ.tolist()
        info["modes"] = arr
    return arr[0].astype(np.complex64), history


def _epie_phase_numpy(P, data, pixels, n_iter, seed, beta, callback):
    _, rng = _initial_object(P, seed)
    rng2 = np.random.default_rng(seed)
    phi = rng2.uniform(-0.2, 0.2, size=P.shape).astype(np.float32)
    Pmax = np.float32((np.abs(P) ** 2).max())
    total = sum(float(d.sum()) for d in data)
    history = []

    for _ in range(n_iter):
        err = 0.0
        for k in rng.permutation(len(pixels)):
            i, j = pixels[k]
            Pr = np.roll(P, (i, j), axis=(0, 1))
            O = np.exp(1j * phi)
            psi = Pr * O
            Psi = np.fft.fft2(psi)
            amp = np.sqrt(data[k])
            err += float(np.sum((np.abs(Psi) - amp) ** 2))
            corrected = np.fft.ifft2(amp * np.exp(1j * np.angle(Psi)))
            phi = phi + beta / Pmax * np.imag(np.conj(Pr) * np.conj(O) * (corrected - psi))
        history.append(np.sqrt(err / total))
        if callback is not None:
            callback(len(history), np.exp(1j * phi), history[-1])
    return np.exp(1j * phi).astype(np.complex64), history


def _epie_phase_torch(P, data, pixels, n_iter, seed, beta, callback):
    import torch

    dev = torch.device(best_device())
    _, rng = _initial_object(P, seed)
    rng2 = np.random.default_rng(seed)
    phi = torch.from_numpy(
        rng2.uniform(-0.2, 0.2, size=P.shape).astype(np.float32)).to(dev)
    Pt = torch.from_numpy(P).to(dev)
    amps = [torch.from_numpy(np.sqrt(d)).to(dev) for d in data]
    Pmax = (Pt.abs() ** 2).max()
    total = float(sum(float(d.sum()) for d in data))
    history = []

    for _ in range(n_iter):
        err = torch.zeros((), dtype=torch.float32, device=dev)
        for k in rng.permutation(len(pixels)):
            i, j = pixels[k]
            Pr = torch.roll(Pt, (int(i), int(j)), dims=(0, 1))
            O = torch.exp(1j * phi)
            psi = Pr * O
            Psi = torch.fft.fft2(psi)
            amp = amps[k]
            err = err + ((Psi.abs() - amp) ** 2).sum()
            corrected = torch.fft.ifft2(amp * torch.exp(1j * Psi.angle()))
            phi = phi + beta / Pmax * (torch.conj(Pr) * torch.conj(O)
                                       * (corrected - psi)).imag
        history.append(float(torch.sqrt(err / total).item()))
        if callback is not None:
            callback(len(history), torch.exp(1j * phi).cpu().numpy(), history[-1])
    return torch.exp(1j * phi).cpu().numpy().astype(np.complex64), history


def _initial_object(P, seed):
    rng = np.random.default_rng(seed)
    O = np.exp(1j * rng.uniform(-0.2, 0.2, size=P.shape)).astype(np.complex64)
    return O, rng


def _epie_numpy(P, data, pixels, n_iter, seed, beta, callback):
    O, rng = _initial_object(P, seed)
    Pmax = np.float32((np.abs(P) ** 2).max())
    total = sum(float(d.sum()) for d in data)
    history = []

    for _ in range(n_iter):
        err = 0.0
        for k in rng.permutation(len(pixels)):
            i, j = pixels[k]
            Pr = np.roll(P, (i, j), axis=(0, 1))
            psi = Pr * O
            Psi = np.fft.fft2(psi)
            amp = np.sqrt(data[k])
            err += float(np.sum((np.abs(Psi) - amp) ** 2))
            corrected = np.fft.ifft2(amp * np.exp(1j * np.angle(Psi)))
            O = O + beta * np.conj(Pr) / Pmax * (corrected - psi)
        history.append(np.sqrt(err / total))
        if callback is not None:
            callback(len(history), O, history[-1])
    return O, history


def _epie_torch(P, data, pixels, n_iter, seed, beta, callback):
    import torch

    dev = torch.device(best_device())
    O_np, rng = _initial_object(P, seed)

    Pt = torch.from_numpy(P).to(dev)
    O = torch.from_numpy(O_np).to(dev)
    amps = [torch.from_numpy(np.sqrt(d)).to(dev) for d in data]
    Pmax = (Pt.abs() ** 2).max()
    total = float(sum(float(d.sum()) for d in data))
    history = []

    for _ in range(n_iter):
        err = torch.zeros((), dtype=torch.float32, device=dev)
        for k in rng.permutation(len(pixels)):
            i, j = pixels[k]
            Pr = torch.roll(Pt, (int(i), int(j)), dims=(0, 1))
            psi = Pr * O
            Psi = torch.fft.fft2(psi)
            amp = amps[k]
            err = err + ((Psi.abs() - amp) ** 2).sum()
            corrected = torch.fft.ifft2(amp * torch.exp(1j * Psi.angle()))
            O = O + beta * torch.conj(Pr) / Pmax * (corrected - psi)
        history.append(float(torch.sqrt(err / total).item()))
        if callback is not None:
            callback(len(history), O.cpu().numpy(), history[-1])
    return O.cpu().numpy(), history
