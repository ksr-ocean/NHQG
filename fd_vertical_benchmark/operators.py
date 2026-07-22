"""Uniform-grid vertical operators and shell-precomputed IMEX matrices."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from fd_vertical_benchmark.config import FDBenchmarkConfig


class Grid(NamedTuple):
    """Precomputed arrays for the FD benchmark solver."""

    z_full: jnp.ndarray
    z_int: jnp.ndarray
    z_weights: jnp.ndarray
    norm_weights: jnp.ndarray

    D1_full: jnp.ndarray
    P_neu: jnp.ndarray
    D1_psi: jnp.ndarray
    D1_dir: jnp.ndarray
    D2_dir: jnp.ndarray

    kx: jnp.ndarray
    ky: jnp.ndarray
    ksq: jnp.ndarray
    denom: jnp.ndarray
    inv_denom: jnp.ndarray

    diss_rate_q: jnp.ndarray
    diss_rate_w: jnp.ndarray
    diss_rate_th: jnp.ndarray
    inv_alpha_q: jnp.ndarray
    inv_alpha_th: jnp.ndarray

    w_solve: jnp.ndarray
    ksq_idx: jnp.ndarray
    thbar_solve: jnp.ndarray

    dt: jnp.ndarray
    gamma_imex: jnp.ndarray
    Ra_sigma: jnp.ndarray
    sigma: jnp.ndarray
    mean_temp_eps_sq: jnp.ndarray

    Nx: int
    Nk: int
    Nz: int
    Ni: int
    Npad: int
    L: float
    thermal_closure: str
    nonlinear_advection: str
    vertical_derivative: str
    vertical_second_derivative: str
    psi_neumann_treatment: str
    psi_boundary: str
    mean_exchange: str
    sbp_corrector_substeps: int


def _first_derivative_matrix_centered2(
    n_points: int, dz: float, dtype=np.float64
) -> np.ndarray:
    """Second-order finite-difference first derivative on a uniform grid."""
    D = np.zeros((n_points, n_points), dtype=dtype)
    scale = 1.0 / (2.0 * dz)

    D[0, 0] = -3.0 * scale
    D[0, 1] = 4.0 * scale
    D[0, 2] = -1.0 * scale

    for i in range(1, n_points - 1):
        D[i, i - 1] = -scale
        D[i, i + 1] = scale

    D[-1, -3] = 1.0 * scale
    D[-1, -2] = -4.0 * scale
    D[-1, -1] = 3.0 * scale
    return D


def _first_derivative_factors_centered2(
    n_points: int, dz: float, dtype=np.float64
) -> tuple[np.ndarray, np.ndarray]:
    """Matrix factors A d = B f for the centered second-order derivative."""
    D = _first_derivative_matrix_centered2(n_points, dz, dtype=dtype)
    A = np.eye(n_points, dtype=dtype)
    return A, D


def _first_derivative_matrix_sbp42(
    n_points: int, dz: float, dtype=np.float64
) -> np.ndarray:
    """Diagonal-norm SBP(4,2) first derivative on a uniform grid."""
    if n_points < 9:
        raise ValueError("sbp42 derivative requires at least 9 grid points")

    D = np.zeros((n_points, n_points), dtype=dtype)

    D[0, 0:4] = np.array(
        [-24.0 / 17.0, 59.0 / 34.0, -4.0 / 17.0, -3.0 / 34.0],
        dtype=dtype,
    )
    D[1, 0:4] = np.array(
        [-1.0 / 2.0, 0.0, 1.0 / 2.0, 0.0],
        dtype=dtype,
    )
    D[2, 0:5] = np.array(
        [4.0 / 43.0, -59.0 / 86.0, 0.0, 59.0 / 86.0, -4.0 / 43.0],
        dtype=dtype,
    )
    D[3, 0:6] = np.array(
        [3.0 / 98.0, 0.0, -59.0 / 98.0, 0.0, 32.0 / 49.0, -4.0 / 49.0],
        dtype=dtype,
    )

    for i in range(4, n_points - 4):
        D[i, i - 2 : i + 3] = np.array(
            [1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, -1.0 / 12.0],
            dtype=dtype,
        )

    top_rows = [4, 4, 5, 6]
    for i, width in enumerate(top_rows):
        D[-1 - i, -width:] = -D[i, :width][::-1]

    return D / dz


def _first_derivative_factors_sbp42(
    n_points: int, dz: float, dtype=np.float64
) -> tuple[np.ndarray, np.ndarray]:
    """Matrix factors A d = B f for the diagonal-norm SBP(4,2) derivative."""
    D = _first_derivative_matrix_sbp42(n_points, dz, dtype=dtype)
    A = np.eye(n_points, dtype=dtype)
    return A, D


def _first_derivative_matrix_compact4(
    n_points: int, dz: float, dtype=np.float64
) -> np.ndarray:
    """Fourth-order tridiagonal compact first derivative on a uniform grid."""
    A, B = _first_derivative_factors_compact4(n_points, dz, dtype=dtype)
    return np.linalg.solve(A, B)


def _first_derivative_factors_compact4(
    n_points: int, dz: float, dtype=np.float64
) -> tuple[np.ndarray, np.ndarray]:
    """Matrix factors A d = B f for the compact fourth-order derivative."""
    if n_points < 4:
        raise ValueError("compact4 derivative requires at least 4 grid points")

    A = np.zeros((n_points, n_points), dtype=dtype)
    B = np.zeros((n_points, n_points), dtype=dtype)
    inv_dz = 1.0 / dz

    # One-sided fourth-order closure at the lower boundary:
    # f'_0 + 3 f'_1 = (-17/6 f_0 + 3/2 f_1 + 3/2 f_2 - 1/6 f_3) / dz
    A[0, 0] = 1.0
    A[0, 1] = 3.0
    B[0, 0] = (-17.0 / 6.0) * inv_dz
    B[0, 1] = (3.0 / 2.0) * inv_dz
    B[0, 2] = (3.0 / 2.0) * inv_dz
    B[0, 3] = (-1.0 / 6.0) * inv_dz

    # Interior stencil:
    # (1/4) f'_{i-1} + f'_i + (1/4) f'_{i+1} = (3/4) (f_{i+1} - f_{i-1}) / dz
    for i in range(1, n_points - 1):
        A[i, i - 1] = 0.25
        A[i, i] = 1.0
        A[i, i + 1] = 0.25
        B[i, i - 1] = (-3.0 / 4.0) * inv_dz
        B[i, i + 1] = (3.0 / 4.0) * inv_dz

    # Mirrored upper-boundary closure.
    A[-1, -2] = 3.0
    A[-1, -1] = 1.0
    B[-1, -4] = (1.0 / 6.0) * inv_dz
    B[-1, -3] = (-3.0 / 2.0) * inv_dz
    B[-1, -2] = (-3.0 / 2.0) * inv_dz
    B[-1, -1] = (17.0 / 6.0) * inv_dz

    return A, B


def _dirichlet_laplacian_interior(n_intervals: int, dz: float, dtype=np.float64) -> np.ndarray:
    """Second-order Dirichlet Laplacian on interior nodes only."""
    n_int = n_intervals - 1
    D2 = np.zeros((n_int, n_int), dtype=dtype)
    scale = 1.0 / (dz * dz)
    for i in range(n_int):
        D2[i, i] = -2.0 * scale
        if i > 0:
            D2[i, i - 1] = scale
        if i < n_int - 1:
            D2[i, i + 1] = scale
    return D2


def _second_derivative_matrix_compact4_full(
    n_points: int, dz: float, dtype=np.float64
) -> np.ndarray:
    """Fourth-order tridiagonal compact second derivative on a uniform grid."""
    if n_points < 5:
        raise ValueError("compact4 second derivative requires at least 5 grid points")

    A = np.zeros((n_points, n_points), dtype=dtype)
    B = np.zeros((n_points, n_points), dtype=dtype)
    inv_dz2 = 1.0 / (dz * dz)

    # One-sided fourth-order closure exact through degree 5:
    # f''_0 + 10 f''_1 = (145/12 f_0 - 76/3 f_1 + 29/2 f_2 - 4/3 f_3 + 1/12 f_4) / dz^2
    A[0, 0] = 1.0
    A[0, 1] = 10.0
    B[0, 0] = (145.0 / 12.0) * inv_dz2
    B[0, 1] = (-76.0 / 3.0) * inv_dz2
    B[0, 2] = (29.0 / 2.0) * inv_dz2
    B[0, 3] = (-4.0 / 3.0) * inv_dz2
    B[0, 4] = (1.0 / 12.0) * inv_dz2

    # Interior stencil:
    # (1/10) f''_{i-1} + f''_i + (1/10) f''_{i+1}
    #   = (6/5) (f_{i-1} - 2 f_i + f_{i+1}) / dz^2
    for i in range(1, n_points - 1):
        A[i, i - 1] = 0.1
        A[i, i] = 1.0
        A[i, i + 1] = 0.1
        B[i, i - 1] = (6.0 / 5.0) * inv_dz2
        B[i, i] = (-12.0 / 5.0) * inv_dz2
        B[i, i + 1] = (6.0 / 5.0) * inv_dz2

    # Mirrored upper-boundary closure.
    A[-1, -2] = 10.0
    A[-1, -1] = 1.0
    B[-1, -5] = (1.0 / 12.0) * inv_dz2
    B[-1, -4] = (-4.0 / 3.0) * inv_dz2
    B[-1, -3] = (29.0 / 2.0) * inv_dz2
    B[-1, -2] = (-76.0 / 3.0) * inv_dz2
    B[-1, -1] = (145.0 / 12.0) * inv_dz2

    return np.linalg.solve(A, B)


def _trap_weights(n_intervals: int, dz: float, dtype=np.float64) -> np.ndarray:
    """Trapezoidal-rule weights on the full vertical grid."""
    w = np.full(n_intervals + 1, dz, dtype=dtype)
    w[0] = 0.5 * dz
    w[-1] = 0.5 * dz
    return w


def _sbp42_norm_weights(n_intervals: int, dz: float, dtype=np.float64) -> np.ndarray:
    """Diagonal norm weights for the SBP(4,2) first derivative."""
    n_points = n_intervals + 1
    if n_points < 9:
        raise ValueError("sbp42 norm requires at least 9 grid points")
    w = np.ones(n_points, dtype=dtype)
    w[:4] = np.array([17.0 / 48.0, 59.0 / 48.0, 43.0 / 48.0, 49.0 / 48.0], dtype=dtype)
    w[-4:] = w[:4][::-1]
    return dz * w


def _dirichlet_extension_matrix(n_intervals: int, dtype=np.float64) -> np.ndarray:
    """Map interior Dirichlet values to the full grid by zero extension."""
    n_full = n_intervals + 1
    n_int = n_intervals - 1
    E = np.zeros((n_full, n_int), dtype=dtype)
    E[1:-1, :] = np.eye(n_int, dtype=dtype)
    return E


def _neumann_reconstruction_from_d1(D1_full: np.ndarray, dtype=np.float64) -> np.ndarray:
    """Map interior psi values to full-node values satisfying D1 psi = 0 at both boundaries."""
    n_full = D1_full.shape[0]
    n_int = n_full - 2
    if n_int < 2:
        raise ValueError("FD benchmark requires Nz >= 3")

    P = np.zeros((n_full, n_int), dtype=dtype)
    P[1:-1, :] = np.eye(n_int, dtype=dtype)
    M = np.array(
        [
            [D1_full[0, 0], D1_full[0, -1]],
            [D1_full[-1, 0], D1_full[-1, -1]],
        ],
        dtype=dtype,
    )
    R = np.array(
        [
            D1_full[0, 1:-1],
            D1_full[-1, 1:-1],
        ],
        dtype=dtype,
    )
    boundary = -np.linalg.solve(M, R)
    P[0, :] = boundary[0]
    P[-1, :] = boundary[1]
    return P


def _direct_neumann_reduction_from_factors(
    A1: np.ndarray, B1: np.ndarray, dtype=np.float64
) -> tuple[np.ndarray, np.ndarray]:
    """Build the reduced Neumann map directly from A d = B f.

    Given interior field values f_i and boundary Neumann data d_b = 0,
    solve directly for the interior derivatives d_i and the unknown boundary
    values f_b. The returned pair is:
    - P_neu: full-grid reconstruction f_full = P_neu @ f_i
    - D1_psi: reduced interior derivative map d_i = D1_psi @ f_i
    """
    n_full = A1.shape[0]
    n_int = n_full - 2
    if n_int < 2:
        raise ValueError("FD benchmark requires Nz >= 3")

    rows_b = np.array([0, n_full - 1])
    rows_i = np.arange(1, n_full - 1)

    A_bi = A1[np.ix_(rows_b, rows_i)]
    A_ii = A1[np.ix_(rows_i, rows_i)]
    B_bb = B1[np.ix_(rows_b, rows_b)]
    B_bi = B1[np.ix_(rows_b, rows_i)]
    B_ib = B1[np.ix_(rows_i, rows_b)]
    B_ii = B1[np.ix_(rows_i, rows_i)]

    M = np.block(
        [
            [A_bi, -B_bb],
            [A_ii, -B_ib],
        ]
    ).astype(dtype, copy=False)
    R = np.vstack([B_bi, B_ii]).astype(dtype, copy=False)
    X = np.linalg.solve(M, R)

    D1_psi = X[:n_int, :]
    boundary = X[n_int:, :]

    P = np.zeros((n_full, n_int), dtype=dtype)
    P[1:-1, :] = np.eye(n_int, dtype=dtype)
    P[0, :] = boundary[0]
    P[-1, :] = boundary[1]
    return P, D1_psi


def _dirichlet_second_derivative_from_full(
    D2_full: np.ndarray, E_dir: np.ndarray
) -> np.ndarray:
    """Restrict a full-grid second derivative to interior Dirichlet values."""
    return D2_full[1:-1, :] @ E_dir


def _dirichlet_second_derivative_from_d1_energy(
    D1_full: np.ndarray,
    E_dir: np.ndarray,
    norm_weights: np.ndarray,
    dtype=np.float64,
) -> np.ndarray:
    """Energy-compatible Dirichlet second derivative induced by D1_full."""
    H_full = np.diag(norm_weights.astype(dtype))
    H_int = np.diag(norm_weights[1:-1].astype(dtype))
    G = D1_full @ E_dir
    return -np.linalg.solve(H_int, G.T @ H_full @ G)


def _build_w_solve(
    B: np.ndarray,
    ksq_flat: np.ndarray,
    dt: float,
    gamma: float,
    nu_q: float,
    nu_w: float,
    nu_theta: float,
    sigma: float,
    Ra_sigma: float,
    drag: float,
    hyper_order: int,
    dtype=np.float64,
):
    """Precompute the shellwise interior solve for the coupled psi-w-theta block."""
    n_int = B.shape[0]
    I = np.eye(n_int, dtype=dtype)

    ksq_rounded = np.round(ksq_flat, decimals=8)
    unique_ksq, inverse_idx = np.unique(ksq_rounded, return_inverse=True)
    n_shells = len(unique_ksq)
    inv_mats = np.zeros((n_shells, n_int, n_int), dtype=dtype)

    for s, ksq_val in enumerate(unique_ksq):
        ksq_p = ksq_val ** hyper_order
        diss_q = nu_q * ksq_p + drag
        diss_w = nu_w * ksq_p
        diss_th = (nu_theta / sigma) * ksq_p

        alpha_q = 1.0 + gamma * dt * diss_q
        alpha_w = 1.0 + gamma * dt * diss_w
        alpha_th = 1.0 + gamma * dt * diss_th
        alpha_w_eff = alpha_w - (gamma * dt) ** 2 * Ra_sigma / alpha_th

        if ksq_val == 0.0:
            A = alpha_w_eff * I
        else:
            coeff = (gamma * dt) ** 2 * (1.0 / ksq_val) / alpha_q
            A = alpha_w_eff * I - coeff * B
        inv_mats[s] = np.linalg.inv(A)

    return inv_mats, inverse_idx.astype(np.int32)


def _stretch_map(n_intervals: int, kind: str, beta: float, dtype=np.float64):
    """Vertical coordinate map z = phi(xi) on uniform xi, with Jacobian phi'(xi).

    Returns (xi, z, dz_dxi). 'uniform' is the identity. 'tanh' clusters nodes at
    BOTH walls (z=0 and z=1) with phi'>0 everywhere (no endpoint singularity,
    unlike the literal Chebyshev/CGL map whose phi' vanishes at the ends).
    """
    xi = np.linspace(0.0, 1.0, n_intervals + 1, dtype=dtype)
    if kind == "uniform":
        return xi, xi.copy(), np.ones_like(xi)
    if kind == "tanh":
        a = float(beta)
        t_half = np.tanh(a * 0.5)
        z = 0.5 * (1.0 + np.tanh(a * (xi - 0.5)) / t_half)
        dz_dxi = (0.5 * a / t_half) * (1.0 - np.tanh(a * (xi - 0.5)) ** 2)
        return xi, z.astype(dtype), dz_dxi.astype(dtype)
    raise ValueError(f"Unsupported vertical_grid={kind!r}")


def make_grid(cfg: FDBenchmarkConfig) -> Grid:
    """Construct the (optionally stretched) vertical operators and shell matrices."""
    if cfg.Nz < 4:
        raise ValueError("FD benchmark requires Nz >= 4")

    build_dtype = np.float64
    target_dtype = np.float64 if cfg.float_dtype == "float64" else np.float32

    Nx = cfg.Nx
    Nk = cfg.Nk
    Nz = cfg.Nz
    Ni = cfg.interior_size
    Npad = cfg.Npad
    dz = cfg.dz

    z_full = np.linspace(0.0, 1.0, Nz + 1, dtype=build_dtype)
    z_int = z_full[1:-1]
    z_weights = _trap_weights(Nz, dz, dtype=build_dtype)
    E_dir = _dirichlet_extension_matrix(Nz, dtype=build_dtype)

    if cfg.vertical_derivative == "centered2":
        A1, B1 = _first_derivative_factors_centered2(Nz + 1, dz, dtype=build_dtype)
        D1_full = B1.copy()
        norm_weights = z_weights.copy()
    elif cfg.vertical_derivative == "compact4":
        A1, B1 = _first_derivative_factors_compact4(Nz + 1, dz, dtype=build_dtype)
        D1_full = np.linalg.solve(A1, B1)
        norm_weights = z_weights.copy()
    elif cfg.vertical_derivative == "sbp42":
        A1, B1 = _first_derivative_factors_sbp42(Nz + 1, dz, dtype=build_dtype)
        D1_full = B1.copy()
        norm_weights = _sbp42_norm_weights(Nz, dz, dtype=build_dtype)
    else:
        raise ValueError(f"Unsupported vertical_derivative={cfg.vertical_derivative!r}")

    # ── Vertical coordinate stretch (mapped SBP) ──
    # D1_full / norm_weights above are built on the uniform computational grid xi
    # (spacing dz=1/Nz). Map to a clustered physical grid z = phi(xi):
    #   d/dz = (1/phi') d/dxi   →   D1_z = diag(1/phi') @ D1_xi
    #   integration norm picks up phi' →  H_z = H_xi * phi'
    # The diagonal Jacobian commutes with the diagonal SBP norm, so
    # H_z D1_z + D1_z^T H_z = Bdiag is preserved: the energy estimate and the
    # balanced-exchange machine-zero conservation carry over, and D1_z stays banded.
    stretched = cfg.vertical_grid != "uniform"
    _xi, z_phys, jac = _stretch_map(Nz, cfg.vertical_grid, cfg.stretch_beta, dtype=build_dtype)
    if stretched:
        jinv = 1.0 / jac
        D1_full = jinv[:, None] * D1_full
        norm_weights = norm_weights * jac
        z_full = z_phys
        z_int = z_full[1:-1]
        z_weights = norm_weights.copy()  # SBP norm is the consistent quadrature on the mapped nodes

    # psi Neumann reconstruction. The "direct" factor path assumes the unmapped
    # operator, so on a stretched grid we use the projected path built from the
    # (mapped) D1_full.
    if cfg.psi_neumann_treatment == "direct" and not stretched:
        P_neu, D1_psi = _direct_neumann_reduction_from_factors(A1, B1, dtype=build_dtype)
    elif cfg.psi_neumann_treatment in ("projected", "direct"):
        P_neu = _neumann_reconstruction_from_d1(D1_full, dtype=build_dtype)
        D1_psi = D1_full[1:-1, :] @ P_neu
    else:
        raise ValueError(
            f"Unsupported psi_neumann_treatment={cfg.psi_neumann_treatment!r}"
        )
    D1_dir = D1_full[1:-1, 1:-1]

    # psi_boundary='none' (production q_boundary='none' analog): psi/vorticity carries
    # NO vertical BC. psi lives on the full grid; the w-coupling uses the full-grid
    # derivative D1_full[1:-1,:] (no P_neu reconstruction). dpsi/dz=0 is NOT imposed —
    # it emerges from w=0. This avoids re-introducing the spurious dq'/dz=0 Neumann
    # vorticity BC (== dpsi/dz=0) that the production solver removed.
    if cfg.psi_boundary == "none":
        D1_psi = D1_full[1:-1, :]  # (Ni, Nz+1): full-grid psi -> interior dpsi/dz
    elif cfg.psi_boundary != "neumann":
        raise ValueError(f"Unsupported psi_boundary={cfg.psi_boundary!r}")

    if stretched:
        # Energy-form Laplacian induced by the mapped D1 (dissipative in H_z).
        D2_dir = _dirichlet_second_derivative_from_d1_energy(
            D1_full, E_dir, norm_weights, dtype=build_dtype
        )
    elif cfg.vertical_second_derivative == "centered2":
        D2_dir = _dirichlet_laplacian_interior(Nz, dz, dtype=build_dtype)
    elif cfg.vertical_second_derivative == "compact4_raw":
        D2_full = _second_derivative_matrix_compact4_full(Nz + 1, dz, dtype=build_dtype)
        D2_dir = _dirichlet_second_derivative_from_full(D2_full, E_dir)
    elif cfg.vertical_second_derivative in ("from_d1_energy", "sbp42_energy"):
        D2_dir = _dirichlet_second_derivative_from_d1_energy(
            D1_full, E_dir, norm_weights, dtype=build_dtype
        )
    else:
        raise ValueError(
            f"Unsupported vertical_second_derivative={cfg.vertical_second_derivative!r}"
        )

    kx_1d = 2.0 * np.pi * np.fft.fftfreq(Nx, d=cfg.L / Nx)
    ky_1d = 2.0 * np.pi * np.arange(Nk) / cfg.L
    kx = kx_1d[:, None]
    ky = ky_1d[None, :]
    ksq = kx ** 2 + ky ** 2

    denom = ksq.copy()
    inv_denom = np.zeros_like(denom)
    np.divide(1.0, denom, out=inv_denom, where=denom > 0.0)

    gamma_imex = 1.0 - 1.0 / np.sqrt(2.0)
    p = cfg.hyper_order
    diss_rate_q = cfg.nu_q * ksq ** p + cfg.drag
    diss_rate_w = cfg.nu_w * ksq ** p
    diss_rate_th = (cfg.nu_theta / cfg.sigma) * ksq ** p
    inv_alpha_q = 1.0 / (1.0 + gamma_imex * cfg.dt * diss_rate_q)
    inv_alpha_th = 1.0 / (1.0 + gamma_imex * cfg.dt * diss_rate_th)

    if cfg.psi_boundary == "none":
        # Coupling operator with no vorticity BC: w (interior, Dirichlet) -> full-grid
        # dw/dz via E_dir, then full psi -> interior dpsi/dz via D1_full[1:-1,:].
        B = (D1_full @ D1_full @ E_dir)[1:-1, :]
    else:
        B = D1_psi @ D1_dir
    w_solve, ksq_idx = _build_w_solve(
        B,
        ksq.ravel(),
        cfg.dt,
        gamma_imex,
        cfg.nu_q,
        cfg.nu_w,
        cfg.nu_theta,
        cfg.sigma,
        cfg.Ra_tilde / cfg.sigma,
        cfg.drag,
        cfg.hyper_order,
        dtype=build_dtype,
    )
    ksq_idx = ksq_idx.reshape(Nx, Nk)

    alpha_bar = gamma_imex * cfg.dt * cfg.mean_temp_eps_sq / cfg.sigma
    A_bar = np.eye(Ni, dtype=build_dtype) - alpha_bar * D2_dir
    thbar_solve = np.linalg.inv(A_bar)

    def to_jax(arr, dtype=None):
        arr_dtype = target_dtype if dtype is None else dtype
        return jnp.array(arr, dtype=arr_dtype)

    return Grid(
        z_full=to_jax(z_full),
        z_int=to_jax(z_int),
        z_weights=to_jax(z_weights),
        norm_weights=to_jax(norm_weights),
        D1_full=to_jax(D1_full),
        P_neu=to_jax(P_neu),
        D1_psi=to_jax(D1_psi),
        D1_dir=to_jax(D1_dir),
        D2_dir=to_jax(D2_dir),
        kx=to_jax(kx),
        ky=to_jax(ky),
        ksq=to_jax(ksq),
        denom=to_jax(denom),
        inv_denom=to_jax(inv_denom),
        diss_rate_q=to_jax(diss_rate_q),
        diss_rate_w=to_jax(diss_rate_w),
        diss_rate_th=to_jax(diss_rate_th),
        inv_alpha_q=to_jax(inv_alpha_q),
        inv_alpha_th=to_jax(inv_alpha_th),
        w_solve=to_jax(w_solve),
        ksq_idx=jnp.array(ksq_idx, dtype=jnp.int32),
        thbar_solve=to_jax(thbar_solve),
        dt=to_jax(np.array(cfg.dt)),
        gamma_imex=to_jax(np.array(gamma_imex)),
        Ra_sigma=to_jax(np.array(cfg.Ra_tilde / cfg.sigma)),
        sigma=to_jax(np.array(cfg.sigma)),
        mean_temp_eps_sq=to_jax(np.array(cfg.mean_temp_eps_sq)),
        Nx=Nx,
        Nk=Nk,
        Nz=Nz,
        Ni=Ni,
        Npad=Npad,
        L=cfg.L,
        thermal_closure=cfg.thermal_closure,
        nonlinear_advection=cfg.nonlinear_advection,
        vertical_derivative=cfg.vertical_derivative,
        vertical_second_derivative=cfg.vertical_second_derivative,
        psi_neumann_treatment=cfg.psi_neumann_treatment,
        psi_boundary=cfg.psi_boundary,
        mean_exchange=cfg.mean_exchange,
        sbp_corrector_substeps=cfg.sbp_corrector_substeps,
    )
