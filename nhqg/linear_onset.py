"""Linear-onset eigenvalue tool for the NHQGE vertical problem.

Assembles the dense linear operator of the (q, w, theta) system at a single
horizontal wavenumber k from the SAME discrete operators the solver uses
(G_Z, the per-field Galerkin stencils and their reductions), in reduced
coordinates: q as full Chebyshev coefficients (no BC, q_boundary='none'),
w and theta in their Galerkin bases. Its spectrum is therefore the exact
linear theory of the spatially-discretized solver -- comparing measured
step-growth against max_growth_rate isolates the time integrator and BC
assembly, for ANY vertical BC combination (this is the mixed-BC onset gate;
the stress-free sin(pi Z) analytics only cover the both-Dirichlet case).

Linearization is about the conduction state (fixed_conduction,
d(Theta_bar)/dZ = -1), matching the solver's implicit_tendency:
    dq/dt     = d(w)/dZ                          - diss_q(k)  q
    dw/dt     = c(k) d(q)/dZ + (Ra/sigma) theta  - diss_w(k)  w
    dtheta/dt = w                                - diss_th(k) theta
with c(k) = 1/(k^2 + Ld^-2) and diss_*(k) the code's per-field rates.
"""

import numpy as np


def linear_operator(cfg, grid, k):
    """Dense linear operator at horizontal wavenumber k.

    Ordering: [q (Nz+1 Chebyshev), w (Nz-1 w-Galerkin), theta (Nz-1
    Dirichlet-Galerkin)]. Returns a real ((3*Nz-1), (3*Nz-1)) matrix.
    """
    N = grid.Nz
    G = np.array(grid.G_Z)
    S_w = np.array(grid.w_stencil)
    S_th = np.array(grid.dirichlet_stencil)
    red_w = np.array(grid.w_pinv) @ np.array(grid.proj_w)
    map_w_to_th = np.array(grid.dirichlet_pinv) @ S_w

    ksq = k ** 2
    ksq_p = ksq ** cfg.hyper_order
    diss_q = cfg.nu_q * ksq_p + cfg.drag
    diss_w = cfg.nu_w * ksq_p
    diss_th = (cfg.nu_theta / cfg.sigma) * ksq_p
    c_k = 1.0 / (ksq + cfg.Ld_inv_sq)
    Ra_sigma = cfg.Ra_tilde / cfg.sigma

    nq, ng = N + 1, N - 1
    A = np.zeros((nq + 2 * ng, nq + 2 * ng))
    sq = slice(0, nq)
    sw = slice(nq, nq + ng)
    sth = slice(nq + ng, nq + 2 * ng)

    A[sq, sq] = -diss_q * np.eye(nq)
    A[sq, sw] = G @ S_w

    A[sw, sq] = red_w @ (c_k * G)
    A[sw, sw] = -diss_w * np.eye(ng)
    A[sw, sth] = red_w @ (Ra_sigma * S_th)

    A[sth, sw] = map_w_to_th
    A[sth, sth] = -diss_th * np.eye(ng)
    return A


def max_growth_rate(cfg, grid, k):
    """Largest real part of the linear spectrum at wavenumber k."""
    lam = np.linalg.eigvals(linear_operator(cfg, grid, k))
    return float(np.max(lam.real))


def critical_rayleigh(cfg, grid, k, ra_lo=1.0, ra_hi=1e4, tol=1e-6):
    """Ra_c(k): bisect Ra_tilde for max_growth_rate = 0 at this k."""
    def rate(ra):
        return max_growth_rate(cfg.with_updates(Ra_tilde=ra), grid, k)
    lo, hi = ra_lo, ra_hi
    if rate(lo) > 0 or rate(hi) < 0:
        raise ValueError("critical Ra not bracketed by [ra_lo, ra_hi]")
    while hi - lo > tol * hi:
        mid = 0.5 * (lo + hi)
        if rate(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)
