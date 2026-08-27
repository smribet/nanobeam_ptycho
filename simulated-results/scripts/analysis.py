"""Per-region metrics: contrast polarity, rigid offset, and its Wigner-Seitz image.

Regions are built once from ground truth and reused for every reconstruction, so the
lattice basis a shift is expressed in never changes between seeds.

A five-fold twin has no single lattice, so it is split into five wedges, each of which
is a <110> single crystal with its own basis.  The wedge orientation is found by
maximising lattice sharpness, so the cuts land on the twin planes rather than across
them.
"""
import numpy as np
from scipy.spatial import Voronoi

BAND = (0.10, 0.80)      # 1/Angstrom, the reflections the reconstruction can carry


def cross2(a, b):
    """Scalar cross product of two 2-vectors.

    Spelled out rather than left to `np.cross`, whose 2-D overload was deprecated in
    NumPy 2.0 and removed in 2.5.
    """
    return a[0] * b[1] - a[1] * b[0]


def band_limit(a, sampling, lo=BAND[0], hi=BAND[1]):
    n = a.shape[0]
    f = np.fft.fftfreq(n, sampling)
    GX, GY = np.meshgrid(f, f, indexing="ij")
    q = np.hypot(GX, GY)
    return np.real(np.fft.ifft2(np.fft.fft2(a) * ((q > lo) & (q < hi))))


def _grid(shape, sampling):
    x = np.arange(shape[0]) * sampling
    return np.meshgrid(x, x, indexing="ij")


def disc(shape, sampling, centre, radius):
    X, Y = _grid(shape, sampling)
    return np.hypot(X - centre[0], Y - centre[1]) < radius


def wedge(shape, sampling, centre, radius, k, n=5, theta0=0.0):
    """Angular sector k of n about `centre`."""
    X, Y = _grid(shape, sampling)
    dx, dy = X - centre[0], Y - centre[1]
    theta = (np.degrees(np.arctan2(dy, dx)) - theta0) % 360.0
    return ((np.hypot(dx, dy) < radius)
            & (theta >= 360.0 * k / n) & (theta < 360.0 * (k + 1) / n))


# ---------------------------------------------------------------- lattice

def lattice_vectors(truth, mask, sampling, n_scan=600):
    """Two shortest non-collinear reciprocal vectors of the masked region."""
    if mask.sum() < 50:
        return None
    patch = (truth - truth[mask].mean()) * mask
    F = np.abs(np.fft.fft2(patch))
    n = truth.shape[0]
    f = np.fft.fftfreq(n, sampling)
    GX, GY = np.meshgrid(f, f, indexing="ij")
    q = np.hypot(GX, GY)
    F[(q < BAND[0]) | (q > BAND[1])] = 0

    found = []
    for idx in np.argsort(F.ravel())[::-1][:n_scan]:
        g = np.array([GX.ravel()[idx], GY.ravel()[idx]])
        if any(np.hypot(*(g - h)) < 0.05 or np.hypot(*(g + h)) < 0.05 for h in found):
            continue
        if not found:
            found.append(g)
            continue
        if abs(cross2(found[0], g)) > 0.02:            # non-collinear
            return reduce_basis(np.stack([found[0], g]))
    return None


def reduce_basis(G):
    """Canonical (Lagrange-reduced) basis for the lattice `G` spans.

    `lattice_vectors` returns whichever two strongest non-collinear reflections it finds,
    and *which* two those are depends on relative peak heights: adding an amorphous
    support was enough to make one twin wedge report {111} + {002} where it had reported
    {111} + {111'}.  Both describe the same lattice, so Wigner-Seitz reduction is
    unaffected -- but anything that compares regions to one another (`upright`,
    `align_rotation`) needs the basis chosen the same way every time, or the wedges can
    no longer be rotated onto a common frame.

    Repeatedly subtracting the shorter vector from the longer yields the two shortest
    vectors of the lattice, which is unique up to signs and ordering.
    """
    if G is None:
        return None
    A = np.linalg.inv(G).T
    a, b = A[0].copy(), A[1].copy()
    if a @ a > b @ b:
        a, b = b, a
    for _ in range(100):
        b = b - round((a @ b) / (a @ a)) * a
        if b @ b >= a @ a:
            break
        a, b = b, a
    if cross2(a, b) < 0:                    # fix handedness so the frame is repeatable
        b = -b
    return np.linalg.inv(np.stack([a, b])).T


def lattice_sharpness(truth, mask, sampling):
    """Height of the strongest in-band reflection; used to orient the wedges."""
    if mask.sum() < 50:
        return 0.0
    patch = (truth - truth[mask].mean()) * mask
    F = np.abs(np.fft.fft2(patch))
    n = truth.shape[0]
    f = np.fft.fftfreq(n, sampling)
    GX, GY = np.meshgrid(f, f, indexing="ij")
    q = np.hypot(GX, GY)
    band = (q > BAND[0]) & (q < BAND[1])
    return float(F[band].max() / max(mask.sum(), 1))


def find_wedge_offset(truth, sampling, centre, radius, n=5, step=2.0):
    """Rotate the wedge boundaries onto the twin planes.

    A wedge straddling a twin boundary contains two lattices and yields two weaker
    reflections, so summed sharpness peaks when the cuts align with the boundaries.
    """
    best_t0, best = 0.0, -np.inf
    for t0 in np.arange(0.0, 360.0 / n, step):
        score = sum(lattice_sharpness(truth, wedge(truth.shape, sampling, centre,
                                                   radius, k, n, t0), sampling)
                    for k in range(n))
        if score > best:
            best_t0, best = float(t0), score
    return best_t0


# ---------------------------------------------------------------- offsets

def signed_offset(recon, truth, mask, sampling):
    """Rigid offset, signed peak correlation, and an ambiguity diagnostic.

    The extremum of largest magnitude is used, so a contrast-reversed region reports a
    negative correlation at its correct lattice registration rather than being
    re-registered against an unrelated positive peak.  `peak_ratio` is the weaker
    extremum over the stronger: near 0 the choice is unambiguous, near 1 it is not.
    """
    n = recon.shape[0]
    A = (recon - recon[mask].mean()) * mask
    B = (truth - truth[mask].mean()) * mask
    denom = np.sqrt(np.sum(A ** 2) * np.sum(B ** 2))
    xc = np.real(np.fft.ifft2(np.fft.fft2(A) * np.conj(np.fft.fft2(B))))

    k = np.unravel_index(np.argmax(np.abs(xc)), xc.shape)
    shift = np.array([(v if v < n // 2 else v - n) * sampling for v in k])
    hi, lo = abs(xc.max()), abs(xc.min())
    ratio = min(hi, lo) / max(hi, lo) if max(hi, lo) > 0 else 0.0
    return float(xc[k] / denom), shift, float(ratio)


def wigner_seitz(shift, G, rng=2):
    """Shortest lattice-equivalent representative of a real-space offset."""
    if G is None:
        return None, None
    A = np.linalg.inv(G).T                  # rows a1, a2 with a_i . g_j = delta_ij
    base = np.round(G @ shift)
    best, best_len = None, np.inf
    for du in range(-rng, rng + 1):
        for dv in range(-rng, rng + 1):
            cand = shift - (base + np.array([du, dv])) @ A
            if np.hypot(*cand) < best_len - 1e-12:
                best_len, best = np.hypot(*cand), cand
    return best, G @ best


def ws_polygon(G, rng=3):
    """Vertices of the Wigner-Seitz cell -- hexagonal, square, or whatever the lattice is."""
    if G is None:
        return None
    A = np.linalg.inv(G).T
    pts = np.array([i * A[0] + j * A[1]
                    for i in range(-rng, rng + 1) for j in range(-rng, rng + 1)])
    vor = Voronoi(pts)
    origin = int(np.argmin(np.hypot(pts[:, 0], pts[:, 1])))
    region = vor.regions[vor.point_region[origin]]
    if not region or -1 in region:
        return None
    v = vor.vertices[region]
    return v[np.argsort(np.arctan2(v[:, 1], v[:, 0]))]


# ---------------------------------------------------------------- driver

def build_regions(truth, sampling, particles, radius=11.0, twin="decahedron",
                  n_wedges=5, twin_radius=11.0):
    """One entry per analysed region, with its mask and lattice fixed from truth."""
    regions = []
    for name, centre in particles.items():
        if name == twin:
            t0 = find_wedge_offset(truth, sampling, centre, twin_radius, n_wedges)
            for k in range(n_wedges):
                m = wedge(truth.shape, sampling, centre, twin_radius, k, n_wedges, t0)
                regions.append((f"{name}_w{k}", m, dict(wedge_theta0=t0, wedge=k)))
        else:
            regions.append((name, disc(truth.shape, sampling, centre, radius), {}))

    out = []
    for name, mask, extra in regions:
        G = lattice_vectors(truth, mask, sampling)
        out.append(dict(name=name, mask=mask, G=G, **extra))
    return out


def summarise(recon_phase, truth, sampling, regions):
    rows = []
    for reg in regions:
        corr, shift, ratio = signed_offset(recon_phase, truth, reg["mask"], sampling)
        ws, frac = wigner_seitz(shift, reg["G"])
        rows.append(dict(
            particle=reg["name"],
            signed_corr=corr,
            reversed=bool(corr < 0),
            peak_ratio=ratio,
            shift_x=float(shift[0]), shift_y=float(shift[1]),
            ws_x=None if ws is None else float(ws[0]),
            ws_y=None if ws is None else float(ws[1]),
            frac_u=None if frac is None else float(frac[0]),
            frac_v=None if frac is None else float(frac[1]),
            has_lattice=reg["G"] is not None,
        ))
    return rows


# ------------------------------------------------- twin boundary registry

def shared_reflection(G1, G2, tol=0.06):
    """The reciprocal vector two adjacent twin segments have in common.

    Adjacent tetrahedra of a decahedron share their twin-plane reflection -- that is
    what makes them twins -- so exactly one basis vector matches between neighbours.
    """
    best, best_d = None, tol
    for g1 in G1:
        for g2 in G2:
            for sign in (1.0, -1.0):
                d = np.hypot(*(g1 - sign * g2))
                if d < best_d:
                    best_d, best = d, 0.5 * (g1 + sign * g2)
    return best


def registry_error(offset_i, offset_j, g):
    """Relative registry across a boundary, as a displacement along g in Angstrom.

    The absolute origin of each segment is unmeasurable, so only the *difference*
    between neighbours is physically meaningful -- it is what decides whether the
    reconstructed atomic arrangement at the boundary is right.  Bounded by +/- d/2.
    """
    dphi = 2 * np.pi * float(np.dot(g, np.asarray(offset_i) - np.asarray(offset_j)))
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    d = 1.0 / np.hypot(*g)
    return dphi / (2 * np.pi) * d, d


def twin_boundaries(rows, regions, prefix="decahedron_w"):
    """Registry error for each adjacent pair of wedges, from one reconstruction."""
    by_name = {r["particle"]: r for r in rows}
    G = {r["name"]: r["G"] for r in regions}
    wedges = sorted(n for n in G if n.startswith(prefix))
    out = []
    for k, name in enumerate(wedges):
        nxt = wedges[(k + 1) % len(wedges)]
        if G[name] is None or G[nxt] is None:
            continue
        g = shared_reflection(G[name], G[nxt])
        a, b = by_name.get(name), by_name.get(nxt)
        if g is None or a is None or b is None or a["ws_x"] is None or b["ws_x"] is None:
            continue
        err, d = registry_error((a["ws_x"], a["ws_y"]), (b["ws_x"], b["ws_y"]), g)
        out.append(dict(boundary=f"{k}-{(k+1) % len(wedges)}",
                        registry_err_A=err, d_shared_A=d,
                        frac_of_d=err / d,
                        sign_flip=bool((a["signed_corr"] < 0) != (b["signed_corr"] < 0)),
                        ambiguous=bool(max(a["peak_ratio"], b["peak_ratio"]) > 0.8)))
    return out


def align_rotation(G, G_ref, n=1441):
    """Rotation carrying lattice `G` onto `G_ref`, ignoring sign and basis order.

    The five twin wedges are rotated copies of one lattice, but `lattice_vectors`
    picks whichever two reflections are strongest, so wedge k's "first" basis vector
    is often wedge k-1's "second".  Fractional coordinates are therefore NOT directly
    comparable between wedges; offsets must be rotated into a common frame instead.
    """
    best, best_d = 0.0, np.inf
    for th in np.linspace(0.0, 2 * np.pi, n):
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s], [s, c]])
        d = sum(min(min(np.hypot(*(R @ g - sign * h)) for sign in (1.0, -1.0))
                    for h in G_ref) for g in G)
        if d < best_d:
            best_d, best = d, th
    return best, best_d


def cell_quantile_fn(G, n=200000, seed=0):
    """Return f(|offset|) -> fraction of the WS cell lying closer to the origin.

    Reduces an offset to a dimensionless number that is comparable across lattices of
    different size and shape: if offsets fill the cell uniformly the values are
    Uniform(0, 1), and if the reconstruction is locked they pile up at 0.  This lets
    every particle and every angle share one axis instead of a grid of scatter plots.
    """
    from matplotlib.path import Path as _Path
    poly = ws_polygon(G)
    if poly is None:
        return None
    rng = np.random.default_rng(seed)
    lim = np.abs(poly).max() * 1.1
    p = rng.uniform(-lim, lim, size=(n, 2))
    p = p[_Path(poly).contains_points(p)]
    radii = np.sort(np.hypot(p[:, 0], p[:, 1]))
    return lambda r: np.searchsorted(radii, r) / len(radii)
