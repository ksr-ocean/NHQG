"""Grid infrastructure: CGL points, coefficient-space Chebyshev operators,
tau-method BCs, wavenumber grids, dissipation, and IMEX precomputed inverses.

Uses the Galerkin/tau approach: fields are stored as Chebyshev coefficients,
derivatives use the coefficient-space recurrence, and BCs are enforced via
tau rows (replacing the last two equations with boundary constraints).

This eliminates the collocation D_Z null mode that caused instability at
high Nz.  The IMEX shell infrastructure (precomputed dense inverses per
|k|^2 shell) is unchanged.

Single entry point: ``make_grid(cfg) -> Grid``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from nhqg.config import NHQGConfig


# ---------------------------------------------------------------------------
# Grid container
# ---------------------------------------------------------------------------

class Grid(NamedTuple):
    """All precomputed arrays needed by the solver, created once."""

    # Vertical grid (CGL points, for transforms and diagnostics)
    Z: jnp.ndarray            # (Nz+1,) CGL points in [0,1]
    xi: jnp.ndarray           # (Nz+1,) CGL points in [-1,1]
    cc_weights: jnp.ndarray   # (Nz+1,) Clenshaw-Curtis quadrature weights on [0,1]

    # Coefficient-space vertical operators
    G_Z: jnp.ndarray          # (Nz+1, Nz+1) d/dZ in coefficient space
    G_Z2: jnp.ndarray         # (Nz+1, Nz+1) d²/dZ² in coefficient space

    # Chebyshev transforms (coefficient <-> nodal)
    V: jnp.ndarray            # (Nz+1, Nz+1) coeffs -> nodal values
    V_inv: jnp.ndarray        # (Nz+1, Nz+1) nodal values -> coeffs
    V_dealias: jnp.ndarray    # (Nz_dealias+1, Nz+1) coeffs -> overresolved nodal values
    V_dealias_inv: jnp.ndarray  # (Nz_dealias+1, Nz_dealias+1) overresolved nodal -> coeffs
    V_exchange: jnp.ndarray   # (Nz_exchange, Nz+1) coeffs -> thermal work-grid values
    V_exchange_inv: jnp.ndarray  # (Nz_exchange, Nz_exchange) work-grid values -> coeffs
    dirichlet_stencil: jnp.ndarray  # (Nz+1, Nz-1) Galerkin Dirichlet -> Chebyshev
    dirichlet_pinv: jnp.ndarray     # (Nz-1, Nz+1) unique-Chebyshev left inverse
    V_exchange_dirichlet: jnp.ndarray  # (Nz_exchange, Nz-1) Dirichlet coeffs -> thermal work-grid values
    G_exchange: jnp.ndarray   # (Nz_exchange, Nz+1) mean coeffs -> work-grid d/dZ values
    exchange_weights: jnp.ndarray  # (Nz_exchange,) quadrature weights on the work grid
    mean_mass: jnp.ndarray    # (Nz+1, Nz+1) L2 mass matrix in Chebyshev coefficients
    mean_mass_inv: jnp.ndarray  # (Nz+1, Nz+1) inverse L2 mass matrix
    theta_mass: jnp.ndarray   # (Nz-1, Nz-1) L2 mass matrix in Dirichlet Galerkin coefficients
    theta_mass_inv: jnp.ndarray  # (Nz-1, Nz-1) inverse Dirichlet mass matrix
    Z_sbp: jnp.ndarray        # (Nz+1,) uniform SBP work grid on [0,1], ascending
    sbp_H: jnp.ndarray        # (Nz+1, Nz+1) diagonal SBP norm matrix
    sbp_D1: jnp.ndarray       # (Nz+1, Nz+1) 2nd-order SBP first derivative
    sbp_L: jnp.ndarray        # (Nz+1, Nz+1) 2nd-order Laplacian with zero boundary rows
    cgl_to_sbp: jnp.ndarray   # (Nz+1, Nz+1) stable nodal transfer CGL -> uniform SBP
    sbp_to_cgl: jnp.ndarray   # (Nz+1, Nz+1) stable nodal transfer uniform SBP -> CGL

    # Horizontal wavenumber grid
    kx: jnp.ndarray           # (Nx, 1) wavenumber array (full axis)
    ky: jnp.ndarray           # (1, Nk) wavenumber array (rfft axis)
    ksq: jnp.ndarray          # (Nx, Nk) |k|^2
    inv_denom: jnp.ndarray    # (Nx, Nk) 1/(|k|^2 + Ld^{-2}), k=0 -> 0
    mask_23: jnp.ndarray      # (Nx, Nk) float keep-mask for 2/3-rule horizontal dealiasing

    # Dissipation: exponential multipliers (for RK4 validator only)
    diss_q: jnp.ndarray       # (Nx, Nk) exp(-diss_rate_q * dt)
    diss_w: jnp.ndarray       # (Nx, Nk) exp(-diss_rate_w * dt)
    diss_th: jnp.ndarray      # (Nx, Nk) exp(-diss_rate_th * dt)

    # Dissipation: raw rates and IMEX alpha factors (for unified IMEX)
    diss_rate_q: jnp.ndarray  # (Nx, Nk) nu_q*|k|^{2p} + drag
    diss_rate_w: jnp.ndarray  # (Nx, Nk) nu_w*|k|^{2p}
    diss_rate_th: jnp.ndarray # (Nx, Nk) (nu_theta/sigma)*|k|^{2p}
    inv_alpha_q: jnp.ndarray  # (Nx, Nk) 1/(1 + gamma*dt*diss_rate_q)
    inv_alpha_th: jnp.ndarray # (Nx, Nk) 1/(1 + gamma*dt*diss_rate_th)

    # IMEX infrastructure (inverses per |k|^2 shell)
    imex_inv: jnp.ndarray     # (n_shells, Nz-1, Nz-1) precomputed A'^{-1} in Dirichlet basis
    q_solve: jnp.ndarray | None  # (n_shells, Nz+1, Nz+1) q-stage tau solve (q_boundary='neumann');
                                 # None for 'none' -- the solve is the scalar inv_alpha_q
    ksq_idx: jnp.ndarray      # (Nx, Nk) int32, maps (kx,ky) -> shell index

    # Tau BC projection matrices (for RK4 / post-step BC enforcement)
    proj_dirichlet: jnp.ndarray  # (Nz+1, Nz+1) projects coeffs to satisfy Dirichlet
    proj_neumann: jnp.ndarray    # (Nz+1, Nz+1) projects coeffs to satisfy Neumann

    # Scalar parameters (as 0-d arrays for JIT compatibility)
    beta: jnp.ndarray
    Ra_sigma: jnp.ndarray
    sigma: jnp.ndarray
    L: jnp.ndarray
    Ld_inv_sq: jnp.ndarray
    dt: jnp.ndarray
    gamma_imex: jnp.ndarray
    mean_temp_eps_sq: jnp.ndarray

    # Static integers (not traced by JAX)
    Nx: int
    Nk: int
    Nz: int
    Nz_gal: int
    Nz_dealias: int
    Nz_exchange: int
    Npad: int
    thermal_closure: str
    q_boundary: str
    nonlinear_advection: str
    vertical_cutoff_n: int | None
    imex_scheme: str
    vertical_dealiasing: str
    horizontal_dealiasing: str
    mean_exchange_discretization: str
    sbp_transfer_mode: str
    sbp_corrector_substeps: int
    imex_matmul_chunk: int


# ---------------------------------------------------------------------------
# Chebyshev coefficient-space differentiation matrix
# ---------------------------------------------------------------------------

def _cheb_coeff_diff_matrix(N: int, dtype=np.float64) -> np.ndarray:
    """Map Chebyshev coefficients to first-derivative coefficients.

    If f(x) = sum_{n=0}^N a_n T_n(x), this returns the matrix G such that
    b = G @ a gives f'(x) = sum_{n=0}^N b_n T_n(x).
    """
    G = np.zeros((N + 1, N + 1), dtype=dtype)

    for n in range(N + 1):
        a = np.zeros(N + 1, dtype=dtype)
        a[n] = 1.0
        b = np.zeros(N + 1, dtype=dtype)

        if N >= 1:
            b[N - 1] = 2.0 * N * a[N]
            for k in range(N - 2, 0, -1):
                b[k] = b[k + 2] + 2.0 * (k + 1) * a[k + 1]
            b[0] = (0.5 * b[2] if N >= 2 else 0.0) + a[1]

        G[:, n] = b

    return G


# ---------------------------------------------------------------------------
# Clenshaw-Curtis quadrature weights on [0, 1]
# ---------------------------------------------------------------------------

def _cc_weights(N: int, dtype=np.float64) -> np.ndarray:
    """Clenshaw-Curtis weights for N+1 CGL points, mapped to [0,1]."""
    theta = np.pi * np.arange(N + 1, dtype=dtype) / N
    w = np.zeros(N + 1, dtype=dtype)

    for j in range(N + 1):
        s = 0.0
        for k in range(1, N // 2 + 1):
            b = 1.0 if k == N // 2 else 2.0
            s += b * np.cos(2.0 * k * theta[j]) / (4.0 * k * k - 1.0)
        c_j = 2.0 if (j == 0 or j == N) else 1.0
        w[j] = (1.0 - s) / (N * c_j)

    return w


def _cheb_vandermonde_and_inverse(N: int, dtype=np.float64) -> tuple[np.ndarray, np.ndarray]:
    """Return DCT-I style Chebyshev coeff<->nodal transforms for N+1 CGL points."""
    j_idx = np.arange(N + 1, dtype=dtype)
    V = np.cos(np.outer(j_idx, j_idx) * (np.pi / N)).astype(dtype)
    c = np.ones(N + 1, dtype=dtype)
    c[0] = 2.0
    c[N] = 2.0
    inv_c = 1.0 / c
    V_inv = (2.0 / N) * np.outer(inv_c, inv_c) * V.T
    return V, V_inv


def _cheb_gauss_vandermonde_and_inverse(N: int, dtype=np.float64) -> tuple[np.ndarray, np.ndarray]:
    """Return Coral-style Chebyshev coeff<->nodal transforms on a Gauss grid.

    The work grid has ``N`` Gauss-Chebyshev points
    ``x_j = cos(pi * (j + 1/2) / N)``, with the corresponding DCT-II inverse
    pair used in Coral's overresolved nonlinear transform path.
    """
    j_idx = np.arange(N, dtype=dtype)
    n_idx = np.arange(N, dtype=dtype)
    theta = np.pi * (j_idx + 0.5) / N
    V = np.cos(np.outer(theta, n_idx)).astype(dtype)
    V_inv = ((2.0 / N) * V.T).astype(dtype)
    V_inv[0, :] *= 0.5
    return V, V_inv


def _cheb_interval_integrals(N: int, dtype=np.float64) -> np.ndarray:
    """Integrals of T_n(2z-1) over z in [0, 1] for n=0..N."""
    ints = np.zeros(N + 1, dtype=dtype)
    for n in range(N + 1):
        if n % 2 == 0:
            ints[n] = 1.0 / (1.0 - n * n) if n != 1 else 0.0
    return ints


def _piecewise_linear_interp_matrix(x_src: np.ndarray, x_dst: np.ndarray,
                                    dtype=np.float64) -> np.ndarray:
    """Return a stable piecewise-linear interpolation matrix.

    The source grid may be either ascending or descending. The target grid may
    be in any order. Rows sum to one and endpoints are preserved exactly when
    present in the source grid.
    """
    x_src = np.asarray(x_src, dtype=dtype)
    x_dst = np.asarray(x_dst, dtype=dtype)

    reverse = bool(x_src[0] > x_src[-1])
    x_work = x_src[::-1] if reverse else x_src
    M = np.zeros((x_dst.size, x_work.size), dtype=dtype)

    for i, x in enumerate(x_dst):
        if x <= x_work[0]:
            M[i, 0] = 1.0
            continue
        if x >= x_work[-1]:
            M[i, -1] = 1.0
            continue
        j = int(np.searchsorted(x_work, x, side="right") - 1)
        x0 = x_work[j]
        x1 = x_work[j + 1]
        t = (x - x0) / (x1 - x0)
        M[i, j] = 1.0 - t
        M[i, j + 1] = t

    return M[:, ::-1] if reverse else M


def _sbp2_operators(N: int, dtype=np.float64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniform-grid second-order SBP norm, first derivative, and Laplacian."""
    dz = 1.0 / N
    n = N + 1

    h = np.ones(n, dtype=dtype)
    h[0] = 0.5
    h[-1] = 0.5
    H = dz * np.diag(h)

    D1 = np.zeros((n, n), dtype=dtype)
    D1[0, 0] = -1.0 / dz
    D1[0, 1] = 1.0 / dz
    for i in range(1, N):
        D1[i, i - 1] = -0.5 / dz
        D1[i, i + 1] = 0.5 / dz
    D1[N, N - 1] = -1.0 / dz
    D1[N, N] = 1.0 / dz

    L = np.zeros((n, n), dtype=dtype)
    for i in range(1, N):
        L[i, i - 1] = 1.0 / dz ** 2
        L[i, i] = -2.0 / dz ** 2
        L[i, i + 1] = 1.0 / dz ** 2

    return H, D1, L


# ---------------------------------------------------------------------------
# Tau BC projection matrices
# ---------------------------------------------------------------------------

def _build_tau_projection(tau_rows: np.ndarray, Nz: int,
                          dtype=np.float64) -> np.ndarray:
    """Build a projection matrix that adjusts coefficients a_{N-1}, a_N
    so that the tau constraints are satisfied.

    tau_rows: (2, Nz+1) — the two BC constraint row vectors.
    Returns: (Nz+1, Nz+1) projection matrix P such that
             tau_rows @ (P @ a) = 0 for any input a.
    """
    N = Nz
    # We solve for a_{N-1}, a_N from the other coefficients:
    #   tau_rows @ a = 0
    #   [tau[0, N-1]  tau[0, N]] [a_{N-1}]   = -tau[0, :N-1] @ a[:N-1]
    #   [tau[1, N-1]  tau[1, N]] [a_{N-1}]   = -tau[1, :N-1] @ a[:N-1]
    M = tau_rows[:, N-1:N+1]  # (2, 2)
    R = tau_rows[:, :N-1]     # (2, N-1)
    M_inv = np.linalg.inv(M)
    # a_{N-1:N+1} = -M_inv @ R @ a_{:N-1}
    # Build full projection: P @ a = [a_0, ..., a_{N-2}, new_{N-1}, new_N]
    P = np.eye(N + 1, dtype=dtype)
    P[N-1, :N-1] = -(M_inv @ R)[0, :]
    P[N-1, N-1] = 0.0
    P[N-1, N] = 0.0
    P[N, :N-1] = -(M_inv @ R)[1, :]
    P[N, N-1] = 0.0
    P[N, N] = 0.0
    return P


# ---------------------------------------------------------------------------
# IMEX inverse precomputation with |k|² shell dedup (tau method)
# ---------------------------------------------------------------------------

def _build_imex_inv(G_Z: np.ndarray, dirichlet_stencil: np.ndarray,
                    dirichlet_pinv: np.ndarray, ksq_flat: np.ndarray,
                    tau_neu: np.ndarray, tau_dir: np.ndarray,
                    Ld_inv_sq: float, dt: float, gamma: float, Nz: int,
                    nu_q: float, nu_w: float, nu_theta: float,
                    sigma: float, Ra_sigma: float,
                    drag: float, hyper_order: int,
                    q_boundary: str, dtype=np.float64):
    """Precompute IMEX inverse matrices for the q-w block.

    Dirichlet BCs for w are always enforced via tau rows. For q, the solve is
    either unconstrained (Miquel-style, ``q_boundary='none'``) or uses the
    historical Neumann tau rows (``q_boundary='neumann'``).
    """
    N = Nz
    I = np.eye(N + 1, dtype=dtype)

    # P_tau: zeroes the last two rows (tau rows) of the RHS
    P_tau = I.copy()
    P_tau[N - 1, N - 1] = 0.0
    P_tau[N, N] = 0.0

    tau_neu_top = tau_neu[0]  # (N+1,) Neumann at Z=1
    tau_neu_bot = tau_neu[1]  # (N+1,) Neumann at Z=0
    tau_dir_top = tau_dir[0]  # (N+1,) Dirichlet at Z=1
    tau_dir_bot = tau_dir[1]  # (N+1,) Dirichlet at Z=0

    ksq_rounded = np.round(ksq_flat, decimals=8)
    unique_ksq, inverse_idx = np.unique(ksq_rounded, return_inverse=True)
    n_shells = len(unique_ksq)

    N_gal = N - 1
    inv_matrices = np.zeros((n_shells, N_gal, N_gal), dtype=dtype)
    # Dense per-shell q matrices exist only for the Neumann tau solve; for
    # q_boundary='none' the solve is the scalar 1/alpha_q (grid.inv_alpha_q)
    # and storing n_shells scaled identities would waste as much memory as
    # imex_inv itself.
    q_solve_matrices = (np.zeros((n_shells, N + 1, N + 1), dtype=dtype)
                        if q_boundary == 'neumann' else None)

    for s, ksq_val in enumerate(unique_ksq):
        ksq_p = ksq_val ** hyper_order
        alpha_q = 1.0 + gamma * dt * (nu_q * ksq_p + drag)
        alpha_w = 1.0 + gamma * dt * nu_w * ksq_p
        alpha_th = 1.0 + gamma * dt * (nu_theta / sigma) * ksq_p

        alpha_w_eff = alpha_w - (gamma * dt) ** 2 * Ra_sigma / alpha_th

        if q_boundary == 'neumann':
            N_q = alpha_q * I.copy()
            N_q[N - 1, :] = tau_neu_top
            N_q[N, :] = tau_neu_bot
            q_solve = np.linalg.inv(N_q) @ P_tau
            q_solve_matrices[s] = q_solve
        elif q_boundary == 'none':
            q_solve = (1.0 / alpha_q) * I
        else:
            raise ValueError(f"Unsupported q_boundary={q_boundary!r}")

        denom = ksq_val + Ld_inv_sq
        if denom == 0.0:
            A = alpha_w_eff * np.eye(N_gal, dtype=dtype)
            inv_matrices[s] = np.linalg.inv(A)
        else:
            c_k = 1.0 / denom
            B = dirichlet_pinv @ G_Z @ q_solve @ G_Z @ dirichlet_stencil
            A = alpha_w_eff * np.eye(N_gal, dtype=dtype) - (gamma * dt) ** 2 * c_k * B
            inv_matrices[s] = np.linalg.inv(A)

    return inv_matrices, q_solve_matrices, inverse_idx.astype(np.int32)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def make_grid(cfg: NHQGConfig) -> Grid:
    """Build all precomputed grid arrays from a configuration."""

    Nx, Nz, L = cfg.Nx, cfg.Nz, cfg.L
    Nk = cfg.Nk
    Npad = cfg.Npad
    Ld_inv_sq = cfg.Ld_inv_sq
    dt = cfg.dt
    fdtype = np.float64 if cfg.float_dtype == "float64" else np.float32
    build_dtype = np.float64
    N = Nz

    # ── Vertical grid ──
    j_idx = np.arange(N + 1, dtype=build_dtype)
    xi = np.cos(np.pi * j_idx / N)
    Z = 0.5 * (1.0 + xi)

    cc_w = _cc_weights(N, dtype=build_dtype)

    # ── Coefficient-space derivative operators ──
    # G_xi: d/dxi in Chebyshev coefficient space
    # G_Z = 2*G_xi: d/dZ where Z = (1+xi)/2
    G_xi = _cheb_coeff_diff_matrix(N, dtype=build_dtype)
    G_Z_np = 2.0 * G_xi
    G_Z2_np = G_Z_np @ G_Z_np

    # ── Chebyshev Vandermonde and its inverse ──
    # V[j, n] = T_n(xi_j) = cos(n*pi*j/N) — coefficients to nodal values
    V_np, V_inv_np = _cheb_vandermonde_and_inverse(N, dtype=build_dtype)

    if cfg.vertical_dealiasing == "none":
        N_dealias = N
        V_dealias_np = V_np
        V_dealias_inv_np = V_inv_np
    elif cfg.vertical_dealiasing in {"cheb_3o2", "cheb_2x"}:
        if cfg.vertical_dealiasing == "cheb_3o2":
            # Coral-style work arrays use NZAA = 3*NZ/2 on a Gauss-Chebyshev grid.
            N_dealias = max(N + 1, (3 * N) // 2)
            V_hi_np, V_hi_inv_np = _cheb_gauss_vandermonde_and_inverse(
                N_dealias, dtype=build_dtype
            )
        else:
            N_dealias = 2 * N
            V_hi_np, V_hi_inv_np = _cheb_vandermonde_and_inverse(
                N_dealias, dtype=build_dtype
            )
        V_dealias_np = V_hi_np[:, :N + 1]
        V_dealias_inv_np = V_hi_inv_np
    else:
        raise ValueError(f"Unsupported vertical_dealiasing={cfg.vertical_dealiasing!r}")

    # Dedicated Coral-style work grid for the thermal exchange pair.
    N_exchange = max(N + 1, (3 * N) // 2)
    V_exchange_full_np, V_exchange_inv_np = _cheb_gauss_vandermonde_and_inverse(
        N_exchange, dtype=build_dtype
    )
    V_exchange_np = V_exchange_full_np[:, :N + 1]

    # Dirichlet Galerkin stencil used by Coral for both-Dirichlet fields:
    # basis_j = -T_j + T_{j+2}, j = 0..N-2
    dirichlet_stencil_np = np.zeros((N + 1, N - 1), dtype=build_dtype)
    for j in range(N - 1):
        dirichlet_stencil_np[j, j] = -1.0
        dirichlet_stencil_np[j + 2, j] = 1.0
    dirichlet_unique_np = dirichlet_stencil_np[:N - 1, :]
    dirichlet_unique_inv_np = np.linalg.inv(dirichlet_unique_np)
    dirichlet_pinv_np = np.zeros((N - 1, N + 1), dtype=build_dtype)
    dirichlet_pinv_np[:, :N - 1] = dirichlet_unique_inv_np

    V_exchange_dirichlet_np = V_exchange_np @ dirichlet_stencil_np
    G_exchange_np = V_exchange_np @ G_Z_np
    exchange_ints_np = _cheb_interval_integrals(N_exchange - 1, dtype=build_dtype)
    exchange_weights_np = exchange_ints_np @ V_exchange_inv_np

    # ── Tau BC row vectors ──
    # Dirichlet: f(xi=+1)=0 → sum_n a_n = 0;  f(xi=-1)=0 → sum (-1)^n a_n = 0
    e_plus = np.ones(N + 1, dtype=build_dtype)           # T_n(+1) = 1
    e_minus = np.array([(-1.0)**n for n in range(N + 1)], dtype=build_dtype)  # T_n(-1)
    tau_dir = np.stack([e_plus, e_minus])  # (2, N+1)

    # Neumann: f'(xi=+1)=0 and f'(xi=-1)=0
    # f'(xi) = sum b_n T_n(xi), b = G_xi @ a.  f'(+1) = e_+ @ b = e_+ @ G_xi @ a
    # Use G_Z (= 2*G_xi) since d/dZ = 0 ⟺ d/dxi = 0 (factor of 2 irrelevant for = 0)
    tau_neu = np.stack([e_plus @ G_Z_np, e_minus @ G_Z_np])  # (2, N+1)

    # ── Tau projection matrices (for RK4 / post-step BC enforcement) ──
    proj_dir_np = _build_tau_projection(tau_dir, N, dtype=build_dtype)
    proj_neu_np = _build_tau_projection(tau_neu, N, dtype=build_dtype)

    # Coefficient-space L2 mass matrix on [0,1], exact under CC quadrature.
    mean_mass_np = V_np.T @ (cc_w[:, None] * V_np)
    mean_mass_inv_np = np.linalg.inv(mean_mass_np)
    theta_mass_np = dirichlet_stencil_np.T @ mean_mass_np @ dirichlet_stencil_np
    theta_mass_inv_np = np.linalg.inv(theta_mass_np)

    # ── Uniform SBP work grid and nodal transfer operators ──
    Z_sbp_np = np.linspace(0.0, 1.0, N + 1, dtype=build_dtype)
    sbp_H_np, sbp_D1_np, sbp_L_np = _sbp2_operators(N, dtype=build_dtype)
    cgl_to_sbp_linear_np = _piecewise_linear_interp_matrix(Z, Z_sbp_np, dtype=build_dtype)
    M_cc_np = np.diag(cc_w)
    if cfg.sbp_transfer_mode == "interp":
        cgl_to_sbp_np = cgl_to_sbp_linear_np
        sbp_to_cgl_np = _piecewise_linear_interp_matrix(Z_sbp_np, Z, dtype=build_dtype)
    elif cfg.sbp_transfer_mode == "mass_adjoint":
        cgl_to_sbp_np = cgl_to_sbp_linear_np
        sbp_to_cgl_np = np.linalg.solve(M_cc_np, cgl_to_sbp_np.T @ sbp_H_np)
    elif cfg.sbp_transfer_mode == "weighted_polar":
        # Start from the interpolation map, then take its weighted polar factor
        # so the transfer is simultaneously mass-compatible and exactly
        # invertible with the weighted-adjoint inverse.
        h_diag = np.diag(sbp_H_np)
        m_diag = cc_w
        H_half = np.diag(np.sqrt(h_diag))
        H_inv_half = np.diag(1.0 / np.sqrt(h_diag))
        M_half = np.diag(np.sqrt(m_diag))
        M_inv_half = np.diag(1.0 / np.sqrt(m_diag))
        A_transfer = H_half @ cgl_to_sbp_linear_np @ M_inv_half
        U_svd, _, Vt_svd = np.linalg.svd(A_transfer, full_matrices=False)
        Q_transfer = U_svd @ Vt_svd
        cgl_to_sbp_np = H_inv_half @ Q_transfer @ M_half
        sbp_to_cgl_np = np.linalg.solve(M_cc_np, cgl_to_sbp_np.T @ sbp_H_np)
    else:
        raise ValueError(f"Unsupported sbp_transfer_mode={cfg.sbp_transfer_mode!r}")

    # ── Horizontal wavenumber grid ──
    kx_1d = 2.0 * np.pi * np.fft.fftfreq(Nx, d=L / Nx)
    ky_1d = 2.0 * np.pi * np.arange(Nk) / L

    kx_2d = kx_1d[:, None]
    ky_2d = ky_1d[None, :]
    ksq_np = kx_2d ** 2 + ky_2d ** 2

    denom = ksq_np + Ld_inv_sq
    inv_denom = np.zeros_like(denom)
    np.divide(1.0, denom, out=inv_denom, where=denom > 0)

    # ── 2/3-rule dealiasing mask (used only when horizontal_dealiasing="23_rule") ──
    K_23 = Nx // 3
    _kx_idx_full = np.arange(Nx)
    _kx_int = np.where(_kx_idx_full <= Nx // 2, _kx_idx_full, _kx_idx_full - Nx)
    _ky_int = np.arange(Nk)
    mask_23_np = ((np.abs(_kx_int)[:, None] <= K_23) &
                  (_ky_int[None, :] <= K_23)).astype(np.float64)

    # ── Dissipation rates and IMEX alpha factors ──
    p = cfg.hyper_order
    if cfg.imex_scheme == "ars222":
        gamma_imex = 1.0 - 1.0 / np.sqrt(2.0)
    elif cfg.imex_scheme == "rk443":
        gamma_imex = 0.5
    else:
        raise ValueError(f"Unsupported imex_scheme={cfg.imex_scheme!r}")

    diss_rate_q_np = cfg.nu_q * ksq_np ** p + cfg.drag
    diss_rate_w_np = cfg.nu_w * ksq_np ** p
    diss_rate_th_np = (cfg.nu_theta / cfg.sigma) * ksq_np ** p

    diss_q_np = np.exp(-diss_rate_q_np * dt)
    diss_w_np = np.exp(-diss_rate_w_np * dt)
    diss_th_np = np.exp(-diss_rate_th_np * dt)

    inv_alpha_q_np = 1.0 / (1.0 + gamma_imex * dt * diss_rate_q_np)
    inv_alpha_th_np = 1.0 / (1.0 + gamma_imex * dt * diss_rate_th_np)

    # ── IMEX inverse matrices (tau method, coefficient space) ──
    ksq_flat = ksq_np.ravel()
    inv_matrices, q_solve_matrices, ksq_idx_flat = _build_imex_inv(
        G_Z_np, dirichlet_stencil_np, dirichlet_pinv_np, ksq_flat, tau_neu, tau_dir,
        Ld_inv_sq, dt, gamma_imex, Nz,
        cfg.nu_q, cfg.nu_w, cfg.nu_theta, cfg.sigma,
        cfg.Ra_tilde / cfg.sigma, cfg.drag, cfg.hyper_order,
        cfg.q_boundary, dtype=build_dtype
    )
    ksq_idx_2d = ksq_idx_flat.reshape(Nx, Nk)

    # ── Cast to target dtype ──
    def to_jax(arr, dtype=None):
        if dtype is None:
            dtype = fdtype
        return jnp.array(arr, dtype=dtype)

    return Grid(
        Z=to_jax(Z),
        xi=to_jax(xi),
        cc_weights=to_jax(cc_w),
        G_Z=to_jax(G_Z_np),
        G_Z2=to_jax(G_Z2_np),
        V=to_jax(V_np),
        V_inv=to_jax(V_inv_np),
        V_dealias=to_jax(V_dealias_np),
        V_dealias_inv=to_jax(V_dealias_inv_np),
        V_exchange=to_jax(V_exchange_np),
        V_exchange_inv=to_jax(V_exchange_inv_np),
        dirichlet_stencil=to_jax(dirichlet_stencil_np),
        dirichlet_pinv=to_jax(dirichlet_pinv_np),
        V_exchange_dirichlet=to_jax(V_exchange_dirichlet_np),
        G_exchange=to_jax(G_exchange_np),
        exchange_weights=to_jax(exchange_weights_np),
        mean_mass=to_jax(mean_mass_np),
        mean_mass_inv=to_jax(mean_mass_inv_np),
        theta_mass=to_jax(theta_mass_np),
        theta_mass_inv=to_jax(theta_mass_inv_np),
        Z_sbp=to_jax(Z_sbp_np),
        sbp_H=to_jax(sbp_H_np),
        sbp_D1=to_jax(sbp_D1_np),
        sbp_L=to_jax(sbp_L_np),
        cgl_to_sbp=to_jax(cgl_to_sbp_np),
        sbp_to_cgl=to_jax(sbp_to_cgl_np),
        kx=to_jax(kx_2d),
        ky=to_jax(ky_2d),
        mask_23=to_jax(mask_23_np),
        ksq=to_jax(ksq_np),
        inv_denom=to_jax(inv_denom),
        diss_q=to_jax(diss_q_np),
        diss_w=to_jax(diss_w_np),
        diss_th=to_jax(diss_th_np),
        diss_rate_q=to_jax(diss_rate_q_np),
        diss_rate_w=to_jax(diss_rate_w_np),
        diss_rate_th=to_jax(diss_rate_th_np),
        inv_alpha_q=to_jax(inv_alpha_q_np),
        inv_alpha_th=to_jax(inv_alpha_th_np),
        imex_inv=to_jax(inv_matrices),
        q_solve=to_jax(q_solve_matrices) if q_solve_matrices is not None else None,
        ksq_idx=jnp.array(ksq_idx_2d, dtype=jnp.int32),
        proj_dirichlet=to_jax(proj_dir_np),
        proj_neumann=to_jax(proj_neu_np),
        beta=to_jax(np.array(cfg.beta)),
        Ra_sigma=to_jax(np.array(cfg.Ra_tilde / cfg.sigma)),
        sigma=to_jax(np.array(cfg.sigma)),
        L=to_jax(np.array(L)),
        Ld_inv_sq=to_jax(np.array(Ld_inv_sq)),
        dt=to_jax(np.array(dt)),
        gamma_imex=to_jax(np.array(gamma_imex)),
        mean_temp_eps_sq=to_jax(np.array(cfg.mean_temp_eps_sq)),
        Nx=Nx,
        Nk=Nk,
        Nz=Nz,
        Nz_gal=Nz - 1,
        Nz_dealias=N_dealias,
        Nz_exchange=N_exchange,
        Npad=Npad,
        thermal_closure=cfg.thermal_closure,
        q_boundary=cfg.q_boundary,
        nonlinear_advection=cfg.nonlinear_advection,
        vertical_cutoff_n=cfg.vertical_cutoff_n,
        imex_scheme=cfg.imex_scheme,
        vertical_dealiasing=cfg.vertical_dealiasing,
        horizontal_dealiasing=cfg.horizontal_dealiasing,
        mean_exchange_discretization=cfg.mean_exchange_discretization,
        sbp_transfer_mode=cfg.sbp_transfer_mode,
        sbp_corrector_substeps=cfg.sbp_corrector_substeps,
        imex_matmul_chunk=cfg.imex_matmul_chunk,
    )
