"""Post-processing diagnostics for polar-cap turbulence simulations.

Field convention (matches the rest of the repo): a 2-D physical field is a
plain numpy array ``field[ix, iy]`` of shape ``(Nx, Nx)`` on a doubly-periodic
square domain of side ``L``, with ``x = ix*L/Nx`` and ``y = iy*L/Nx``
('ij' indexing). The default center for polar decompositions is
``(L/2, L/2)``.

Everything here is pure numpy, run after the fact on saved snapshots or
checkpoints -- no jax, no matplotlib, no scipy. It does not touch the
solver's own arrays or representations.
"""

import numpy as np


def polar_resample(field, L, r_max, n_r=64, n_theta=256, center=None):
    """Bilinearly resample a periodic field onto a polar (r, theta) grid.

    r_i = (i+0.5)*r_max/n_r, theta_j = 2*pi*j/n_theta. Sample points are
    wrapped periodically (mod L) in both x and y before interpolation.

    Returns (r (n_r,), theta (n_theta,), F (n_r, n_theta)).
    """
    Nx = field.shape[0]
    dx = L / Nx
    if center is None:
        cx, cy = L / 2.0, L / 2.0
    else:
        cx, cy = center

    r = (np.arange(n_r) + 0.5) * (r_max / n_r)
    theta = 2.0 * np.pi * np.arange(n_theta) / n_theta

    R, TH = np.meshgrid(r, theta, indexing='ij')
    X = np.mod(cx + R * np.cos(TH), L)
    Y = np.mod(cy + R * np.sin(TH), L)

    fx = X / dx
    fy = Y / dx
    ix0f = np.floor(fx)
    iy0f = np.floor(fy)
    tx = fx - ix0f
    ty = fy - iy0f
    ix0 = ix0f.astype(np.int64) % Nx
    iy0 = iy0f.astype(np.int64) % Nx
    ix1 = (ix0 + 1) % Nx
    iy1 = (iy0 + 1) % Nx

    f00 = field[ix0, iy0]
    f10 = field[ix1, iy0]
    f01 = field[ix0, iy1]
    f11 = field[ix1, iy1]
    F = (f00 * (1 - tx) * (1 - ty) + f10 * tx * (1 - ty)
         + f01 * (1 - tx) * ty + f11 * tx * ty)

    return r, theta, F


def azimuthal_energy_spectrum(field, L, r_max, m_max=32, n_r=64, n_theta=None,
                               center=None):
    """Azimuthal (angular Fourier) energy spectrum of a field on rings.

    Per ring i, f_m = rfft(F[i, :]) / n_theta; E_m = sum_i r_i*dr*c_m*|f_m|^2
    with c_0 = 1, c_m = 2 for m >= 1.

    Returns (m (m_max+1,), E_m (m_max+1,)).
    """
    if n_theta is None:
        n_theta = max(256, 8 * m_max)
    if m_max > n_theta // 2:
        raise ValueError(
            f"m_max={m_max} exceeds n_theta//2={n_theta // 2} (n_theta={n_theta})")

    r, _theta, F = polar_resample(field, L, r_max, n_r=n_r, n_theta=n_theta,
                                   center=center)
    dr = r_max / n_r

    f_m = np.fft.rfft(F, axis=1) / n_theta
    f_m = f_m[:, :m_max + 1]

    c = np.full(m_max + 1, 2.0)
    c[0] = 1.0

    E_m = np.sum((r * dr)[:, None] * c[None, :] * np.abs(f_m) ** 2, axis=0)
    m = np.arange(m_max + 1)
    return m, E_m


def radial_profile(field, L, r_max, n_r=64, n_theta=256, center=None):
    """Azimuthally-averaged radial profile of a field.

    Returns (r (n_r,), prof (n_r,)) with prof the theta-mean of the
    polar-resampled field per ring.
    """
    r, _theta, F = polar_resample(field, L, r_max, n_r=n_r, n_theta=n_theta,
                                   center=center)
    prof = F.mean(axis=1)
    return r, prof


def vortex_positions(zeta, L, threshold_frac=0.5, min_separation=None):
    """Locate local-maximum vortex cores in a periodic vorticity-like field.

    Candidates are grid points with zeta > threshold_frac*zeta.max() that are
    strictly greater than all 8 periodic neighbors. Candidates are visited in
    descending zeta order and kept greedily subject to a minimum periodic
    separation (default L/32) from already-kept vortices. Each kept
    candidate's position is then refined per axis with a 3-point parabolic
    fit using periodic neighbors.

    Returns an (n, 2) array of (x, y) positions, empty (0, 2) if
    zeta.max() <= 0.
    """
    zmax = zeta.max()
    if zmax <= 0:
        return np.zeros((0, 2))

    Nx = zeta.shape[0]
    dx = L / Nx
    if min_separation is None:
        min_separation = L / 32.0
    threshold = threshold_frac * zmax

    greater = np.ones_like(zeta, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            neighbor = np.roll(np.roll(zeta, -di, axis=0), -dj, axis=1)
            greater &= (zeta > neighbor)

    mask = (zeta > threshold) & greater
    idx = np.argwhere(mask)
    if idx.shape[0] == 0:
        return np.zeros((0, 2))

    vals = zeta[mask]
    order = np.argsort(-vals, kind='stable')
    idx = idx[order]

    kept_ij = []
    for i, j in idx:
        xr, yr = i * dx, j * dx
        ok = True
        for ki, kj in kept_ij:
            xk, yk = ki * dx, kj * dx
            ddx = ((xr - xk + L / 2.0) % L) - L / 2.0
            ddy = ((yr - yk + L / 2.0) % L) - L / 2.0
            if np.sqrt(ddx * ddx + ddy * ddy) < min_separation:
                ok = False
                break
        if ok:
            kept_ij.append((int(i), int(j)))

    positions = np.zeros((len(kept_ij), 2))
    for row, (i, j) in enumerate(kept_ij):
        f_center = zeta[i, j]

        f_minus = zeta[(i - 1) % Nx, j]
        f_plus = zeta[(i + 1) % Nx, j]
        denom = f_minus - 2.0 * f_center + f_plus
        ox = 0.0 if denom == 0 else 0.5 * (f_minus - f_plus) / denom
        ox = min(max(ox, -0.5), 0.5)
        positions[row, 0] = ((i + ox) * dx) % L

        f_minus = zeta[i, (j - 1) % Nx]
        f_plus = zeta[i, (j + 1) % Nx]
        denom = f_minus - 2.0 * f_center + f_plus
        oy = 0.0 if denom == 0 else 0.5 * (f_minus - f_plus) / denom
        oy = min(max(oy, -0.5), 0.5)
        positions[row, 1] = ((j + oy) * dx) % L

    return positions


def trap_mask(Nx, L, r_star, center=None):
    """Indicator mask of a single disk of radius r_star (not periodic).

    Returns an (Nx, Nx) float64 array, 1.0 where the true (unwrapped)
    Euclidean distance from center is < r_star, else 0.0.
    """
    dx = L / Nx
    if center is None:
        cx, cy = L / 2.0, L / 2.0
    else:
        cx, cy = center

    xy = np.arange(Nx) * dx
    X, Y = np.meshgrid(xy, xy, indexing='ij')
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    return (r < r_star).astype(np.float64)
