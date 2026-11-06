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
SCAN_ORIGIN = 0.0        # the scanned square, which `set_field` keeps fixed when padding
SCAN_EXTENT = None       # None -> the whole field

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


def set_field(extent):
    """Enlarge the periodic field to `extent` Angstrom at the same 0.2 A pixel size.

    The probe array spans the whole field, so this refines the reciprocal sampling by
    extent / L while leaving the maximum collection angle unchanged.  The particles are
    moved to the centre of the new field and only the original square is scanned, so the
    specimen, the probe and the scan are otherwise identical.
    """
    global L, GPTS, PARTICLES, SCAN_ORIGIN, SCAN_EXTENT
    off = (extent - L) / 2
    PARTICLES = {k: (x + off, y + off) for k, (x, y) in PARTICLES.items()}
    SCAN_ORIGIN, SCAN_EXTENT = SCAN_ORIGIN + off, SCAN_EXTENT or L
    GPTS = int(round(extent / (L / GPTS)))
    L = extent


CARBON_TOP = 33.0        # z at which the support starts, just below the particles


def amorphous_carbon(thickness=THICK_SUPPORT, seed=10):
    """Amorphous carbon support as published: a jittered diamond slab folded into 8 A.

    A diamond lattice jittered by 0.5 of a bond length, which gives a diffuse ring and no
    Bragg reflections -- the point being that the support is invisible to any method that
    works from detected reflections.

    Kept only so the published sweeps stay reproducible.  The 14.3 A slab (12 800 atoms)
    is folded modulo 8 A, so its projected density is that of ~2.8 nm of a-C at
    2.0 g/cm^3 while its local density is an unphysical 6.9 g/cm^3.  Single-slice data see
    only the projection; multislice data see the folded 3D structure.  New runs use
    `amorphous_carbon_film`.
    """
    n = int(np.ceil(L / 3.567))
    c = ase.build.bulk("C", cubic=True) * (n, n, 4)
    rng = np.random.default_rng(seed)
    c.positions[:] += rng.normal(size=(len(c), 3)) * 0.5 * 1.54
    c.cell = [L, L, thickness]
    c.wrap()
    del c[c.positions[:, 0] > L]
    del c[c.positions[:, 1] > L]
    c.positions[:, 2] = c.positions[:, 2] % 8 + CARBON_TOP
    return c


def amorphous_carbon_film(thickness, density=2.0, dmin=1.4, seed=10, batch=4000):
    """Physical amorphous carbon film of `thickness` Angstrom at `density` g/cm^3.

    Random sequential addition: uniform random positions, periodic in x and y, accepted
    only if no accepted atom lies closer than `dmin`.  At 2.0 g/cm^3 and 1.4 A the hard-core
    packing fraction is ~0.14, far below jamming, so the film fills in a few batches.
    """
    from scipy.spatial import cKDTree

    n_target = int(round(density / 12.011 * 0.602214 * L * L * thickness))
    box = np.array([L, L, thickness + 2 * dmin + 1.0])     # z padded so it never wraps
    rng = np.random.default_rng(seed)
    pos = np.empty((0, 3))
    while len(pos) < n_target:
        cand = rng.uniform((0, 0, 0), (L, L, thickness), size=(batch, 3))
        if len(pos):
            d, _ = cKDTree(pos, boxsize=box).query(cand, distance_upper_bound=dmin)
            cand = cand[~np.isfinite(d)]
        # resolve clashes inside the batch greedily, keeping the earlier candidate
        clash = cKDTree(cand, boxsize=box).query_pairs(dmin, output_type="ndarray")
        keep = np.ones(len(cand), bool)
        for i, j in clash:
            if keep[i]:
                keep[j] = False
        pos = np.vstack([pos, cand[keep]])[:n_target]
    c = ase.Atoms("C" * len(pos), positions=pos, cell=[L, L, thickness])
    c.positions[:, 2] += CARBON_TOP
    return c


def build_sample(support=False, carbon_thickness=None, scale=1.0, particles=None,
                 carbon_seed=10):
    """Three Au nanoparticles: [111] single, [100] single, five-fold twin.

    With `support`, an amorphous carbon film is added underneath and the cell is
    deepened to hold it; the Au positions are unchanged either way.  `carbon_thickness`
    (Angstrom) selects a physical film from `amorphous_carbon_film`; left as None, the
    published folded film is used.  `carbon_seed` picks the film's realisation (10 is the
    one every published and R0 run used).

    `scale` multiplies every particle's linear size (layer counts for the single crystals,
    atom count to the 1/3 for the twin) about the same centres.  Above ~1.25 neighbouring
    particles touch, so larger sizes are built one particle at a time: `particles` names
    the subset to include, at the centres in PARTICLES.
    """
    layers = np.maximum(1, np.round(np.array([5, 7, 4]) * scale)).astype(int)

    def place(c, centre):
        c.cell = [L, L, THICK]
        c.center()
        c.positions[:, 0] += centre[0] - L / 2
        c.positions[:, 1] += centre[1] - L / 2
        return c

    def single(zone, twist, centre):
        c = FaceCenteredCubic("Au", [(1, 0, 0), (1, 1, 0), (1, 1, 1)], list(layers),
                              latticeconstant=LATTICE)
        if zone is not None:
            c.rotate(zone, (0, 0, 1), rotate_cell=False)
        c.rotate(twist, "z")
        return place(c, centre)

    def decahedron(centre):
        deca = Decahedron({(1, 0, 0): 1.25, (1, 1, 1): 1.0, (1, 1, 0): 1.05},
                          primitive_structure=bulk("Au"), twin_energy=0.125,
                          natoms=int(round(400 * scale ** 3))).atoms
        deca.rotate(19, "z")
        return place(deca, centre)

    makers = {"au_111": lambda c: single((1, 1, 1), 10, c),
              "au_100": lambda c: single(None, -7, c),
              "decahedron": decahedron}
    names = list(particles or makers)
    atoms = makers[names[0]](PARTICLES[names[0]])
    for name in names[1:]:
        atoms = atoms + makers[name](PARTICLES[name])
    thick = THICK
    if support and carbon_thickness is None:
        thick = THICK_SUPPORT
        atoms = atoms + amorphous_carbon(thick, seed=carbon_seed)
    elif support and carbon_thickness > 0:
        thick = CARBON_TOP + carbon_thickness + 3.0
        atoms = atoms + amorphous_carbon_film(carbon_thickness, seed=carbon_seed)
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


def make_probe(semiangle, defocus, C12=0.0, phi12=0.0):
    """Probe wavefunction centred on the array origin; C12 (A) and phi12 (rad) add
    two-fold astigmatism, used only for the deliberately wrong starting probes."""
    extra = dict(aberrations={"C12": C12, "phi12": phi12}) if C12 else {}
    pr = abtem.Probe(energy=ENERGY, semiangle_cutoff=semiangle, defocus=defocus,
                     extent=[L, L], gpts=(GPTS, GPTS), **extra)
    P = np.squeeze(pr.build(abtem.CustomScan([[0.0, 0.0]])).array)
    return P.astype(np.complex64)


def scan_pixels(sampling, jitter=0.0, seed=0):
    """Probe positions in pixels on the regular raster, optionally perturbed.

    `jitter` (Angstrom) adds a known uniform offset in [-jitter, jitter] to each position,
    drawn once from `seed`, as in Humphry et al. (2012).  Offsets are rounded to the pixel
    grid so the forward model and the reconstruction roll the probe by the same integers.
    """
    g = np.arange(SCAN_ORIGIN + STEP / 2, SCAN_ORIGIN + (SCAN_EXTENT or L), STEP)
    pts = np.array([(px, py) for px in g for py in g])
    if jitter:
        pts = pts + np.random.default_rng(seed).uniform(-jitter, jitter, size=pts.shape)
    n = int(round(L / sampling))
    return [(int(round(px / sampling)) % n, int(round(py / sampling)) % n)
            for px, py in pts]


def simulate_data(P, obj, pixels):
    """Noiseless diffraction intensities, one per scan position."""
    return [np.abs(np.fft.fft2(np.roll(P, (i, j), axis=(0, 1)) * obj)).astype(np.float32)**2
            for i, j in pixels]


def gauge_transformed(obj, sampling, regions, mode, lo=0.10, smooth=1.0):
    """The ground truth with Eq. (6) or Eq. (5) applied to every region's lattice.

    The envelope -- spatial frequencies below `lo`, which carry the mean and the particle
    shape -- is kept.  The remainder, the lattice, is negated ('reversed') or translated
    by half of the region's first lattice vector ('shifted') inside each region's mask.
    Half of a1 changes the phase of g1 by pi but leaves g2 alone, so it is a genuine
    translation and not a disguised reversal.  Masks are softened by a Gaussian of
    `smooth` Angstrom so the edit itself introduces no sharp edge.
    """
    from scipy.ndimage import gaussian_filter
    n = obj.shape[0]
    f = np.fft.fftfreq(n, sampling)
    KX, KY = np.meshgrid(f, f, indexing="ij")
    low = np.fft.ifft2(np.fft.fft2(obj) * (np.hypot(KX, KY) < lo))
    hi = obj - low
    out = obj.astype(np.complex128)
    for r in regions:
        m = gaussian_filter(r["mask"].astype(float), smooth / sampling, mode="wrap")
        if mode == "reversed":
            new = -hi
        elif mode == "shifted":
            d = np.linalg.inv(r["G"]).T[0] / 2
            new = np.fft.ifft2(np.fft.fft2(hi) * np.exp(-2j * np.pi * (KX * d[0] + KY * d[1])))
        else:
            raise ValueError(mode)
        out = out + m * (new - hi)
    return out.astype(np.complex64)


def bin_pattern(p, b):
    """Sum b x b blocks of an unshifted pattern.

    Blocks are aligned at k = 0, so in the unshifted layout each block is contiguous in k
    -- including the ones either side of k = 0 and of the Nyquist wrap.
    """
    if b == 1:
        return p
    n = p.shape[0] // b
    return p.reshape(n, b, n, b).sum(axis=(1, 3))


def detector_mask(n, width):
    """Square detector of `width` pixels about k = 0, in the unshifted layout; None if the
    detector covers the whole grid."""
    if not width or width >= n:
        return None
    k = np.fft.fftfreq(n, 1.0 / n)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    return (np.abs(KX) < width / 2) & (np.abs(KY) < width / 2)


def misfit(P, O, data, pixels, det_bin=1, det_mask=None):
    """Normalised amplitude misfit of a given object -- the error ePIE reports -- without
    updating anything.  Evaluated on the ground truth it is the noise floor of a dataset."""
    import torch
    dev = torch.device(best_device() or "cpu")
    Pt, Ot = torch.from_numpy(P).to(dev), torch.from_numpy(O.astype(np.complex64)).to(dev)
    m = None if det_mask is None else torch.from_numpy(det_mask).to(dev)
    n = P.shape[0]
    err, total = 0.0, 0.0
    for (i, j), d in zip(pixels, data):
        F = torch.fft.fft2(torch.roll(Pt, (int(i), int(j)), dims=(0, 1)) * Ot)
        I = F.real ** 2 + F.imag ** 2
        if det_bin > 1:
            I = I.reshape(n // det_bin, det_bin, n // det_bin, det_bin).sum((1, 3))
        amp = torch.from_numpy(np.sqrt(d)).to(dev)
        r = (I.sqrt() - amp) ** 2
        err += float((r if m is None else r * m).sum())
        total += float(d.sum())
    return float(np.sqrt(err / total))


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
    out = U[:, idx].conj().T @ flat
    # each mode's global phase is free as well (the modes add in intensity), and the
    # eigenvectors come with an arbitrary one: left at pi, np.angle of the dominant mode
    # wraps across the whole field.  Fix every mode's mean phase to zero.
    out = out * np.exp(-1j * np.angle(out.mean(1, keepdims=True)))
    return out.reshape(modes.shape), occ


def epie(P, data, pixels, n_iter, seed, beta=0.9, callback=None, backend="auto",
         obj_model="complex", n_modes=4, ortho_every=1, info=None, O0=None,
         probe_update=False, det_bin=1, det_mask=None):
    """Single-slice ePIE with a known, fixed probe.

    The update order and the initial guess are drawn with numpy for the given seed, so
    a run is reproducible and directly comparable across backends.  Returns
    (object, error history).

    `O0` replaces the random initial object (the update order still follows `seed`).
    `probe_update` refines P as well -- blind ptychography, starting from the P given --
    and leaves the final probe in info["probe"].  `det_bin` and `det_mask` model the
    detector: `data` are then patterns binned by det_bin, and pixels outside det_mask
    were never recorded and are left unconstrained.  These three are implemented for the
    complex object on the torch backend, which is what the revision sweeps use.

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
    if probe_update or det_bin > 1 or det_mask is not None:
        if backend != "torch" or obj_model != "complex":
            raise NotImplementedError("probe/detector options: complex object, torch only")
        return _epie_torch(P, data, pixels, n_iter, seed, beta, callback, O0=O0,
                           probe_update=probe_update, det_bin=det_bin,
                           det_mask=det_mask, info=info)
    if O0 is not None and obj_model not in ("complex", "phase"):
        raise NotImplementedError("O0 with the mixed object model")
    if obj_model == "mixed":
        return _epie_mixed(P, data, pixels, n_iter, seed, beta, callback, backend,
                           n_modes, ortho_every, info)
    if obj_model == "phase":
        f = _epie_phase_torch if backend == "torch" else _epie_phase_numpy
        return f(P, data, pixels, n_iter, seed, beta, callback, O0=O0)
    if backend == "torch":
        return _epie_torch(P, data, pixels, n_iter, seed, beta, callback, O0=O0)
    return _epie_numpy(P, data, pixels, n_iter, seed, beta, callback, O0=O0)


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


def _epie_phase_numpy(P, data, pixels, n_iter, seed, beta, callback, O0=None):
    _, rng = _initial_object(P, seed)
    rng2 = np.random.default_rng(seed)
    phi = rng2.uniform(-0.2, 0.2, size=P.shape).astype(np.float32)
    if O0 is not None:
        phi = np.angle(O0).astype(np.float32)
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


def _epie_phase_torch(P, data, pixels, n_iter, seed, beta, callback, O0=None):
    import torch

    dev = torch.device(best_device())
    _, rng = _initial_object(P, seed)
    rng2 = np.random.default_rng(seed)
    phi0 = rng2.uniform(-0.2, 0.2, size=P.shape).astype(np.float32)
    if O0 is not None:
        phi0 = np.angle(O0).astype(np.float32)
    phi = torch.from_numpy(phi0).to(dev)
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


def _initial_object(P, seed, O0=None):
    """Random initial object and the generator that then sets the update order.

    With `O0` the random object is still drawn, so the update order for a given seed is
    the same whichever initial object is used.
    """
    rng = np.random.default_rng(seed)
    O = np.exp(1j * rng.uniform(-0.2, 0.2, size=P.shape)).astype(np.complex64)
    if O0 is not None:
        O = O0.astype(np.complex64).copy()
    return O, rng


def _epie_numpy(P, data, pixels, n_iter, seed, beta, callback, O0=None):
    O, rng = _initial_object(P, seed, O0)
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


def _epie_torch(P, data, pixels, n_iter, seed, beta, callback, O0=None,
                probe_update=False, det_bin=1, det_mask=None, info=None):
    import torch

    dev = torch.device(best_device())
    O_np, rng = _initial_object(P, seed, O0)

    Pt = torch.from_numpy(P).to(dev)
    O = torch.from_numpy(O_np).to(dev)
    amps = [torch.from_numpy(np.sqrt(d)).to(dev) for d in data]
    Pmax = (Pt.abs() ** 2).max()
    total = float(sum(float(d.sum()) for d in data))
    history = []
    plain = det_bin == 1 and det_mask is None
    mask = None if det_mask is None else torch.from_numpy(det_mask).to(dev)
    n = P.shape[0]

    def modulus(Psi, amp):
        """Impose the measured modulus per (binned) detector pixel; return the
        corrected wave and the misfit.  Unrecorded pixels keep the model's value."""
        I = Psi.real ** 2 + Psi.imag ** 2
        if det_bin > 1:
            I = I.reshape(n // det_bin, det_bin, n // det_bin, det_bin).sum((1, 3))
        mag = I.sqrt()
        s = amp / (mag + 1e-12)
        r = (mag - amp) ** 2
        if mask is not None:
            s, r = torch.where(mask, s, torch.ones_like(s)), r * mask
        if det_bin > 1:
            s = s.repeat_interleave(det_bin, 0).repeat_interleave(det_bin, 1)
        return torch.fft.ifft2(Psi * s), r.sum()

    for _ in range(n_iter):
        err = torch.zeros((), dtype=torch.float32, device=dev)
        Omax = (O.abs() ** 2).max() if probe_update else None
        for k in rng.permutation(len(pixels)):
            i, j = pixels[k]
            Pr = torch.roll(Pt, (int(i), int(j)), dims=(0, 1))
            psi = Pr * O
            Psi = torch.fft.fft2(psi)
            amp = amps[k]
            if plain:
                err = err + ((Psi.abs() - amp) ** 2).sum()
                corrected = torch.fft.ifft2(amp * torch.exp(1j * Psi.angle()))
            else:
                corrected, r = modulus(Psi, amp)
                err = err + r
            if probe_update:
                # ePIE probe step, from the object before this position's update
                dP = beta * torch.conj(O) / Omax * (corrected - psi)
                O = O + beta * torch.conj(Pr) / Pmax * (corrected - psi)
                Pt = Pt + torch.roll(dP, (-int(i), -int(j)), dims=(0, 1))
            else:
                O = O + beta * torch.conj(Pr) / Pmax * (corrected - psi)
        if probe_update:
            Pmax = (Pt.abs() ** 2).max()
        history.append(float(torch.sqrt(err / total).item()))
        if callback is not None:
            callback(len(history), O.cpu().numpy(), history[-1])
    if info is not None and probe_update:
        info["probe"] = Pt.cpu().numpy()
    return O.cpu().numpy(), history
