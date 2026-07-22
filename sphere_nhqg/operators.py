"""Scalar radial operators for the stereographic spherical disk."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from scipy.special import eval_jacobi

from sphere_nhqg.geometry import inverse_conformal_factor
from sphere_nhqg.radial import RadialBasis


def _flat_laplacian_basis_values_np(basis: RadialBasis) -> np.ndarray:
    """Evaluate L_m phi_n at radial nodes for every basis function."""
    m = basis.m
    Nr = basis.Nr
    r_jet = basis.r_jet
    rho = np.asarray(basis.rho)
    xi = np.asarray(basis.xi)
    dtype = rho.dtype
    values = np.empty((Nr, Nr), dtype=dtype)

    for n in range(Nr):
        if n == 0:
            dpoly = np.zeros_like(xi, dtype=dtype)
            d2poly = np.zeros_like(xi, dtype=dtype)
        else:
            dpoly = 0.5 * (n + m + 1.0) * eval_jacobi(
                n - 1, 1.0, float(m + 1), xi
            )
            if n == 1:
                d2poly = np.zeros_like(xi, dtype=dtype)
            else:
                d2poly = 0.25 * (n + m + 1.0) * (n + m + 2.0) * eval_jacobi(
                    n - 2, 2.0, float(m + 2), xi
                )

        values[:, n] = (
            8.0
            / (r_jet * r_jet)
            * rho**m
            * ((m + 1.0) * dpoly + (1.0 + xi) * d2poly)
        )

    return values


def flat_laplacian_matrix(basis: RadialBasis) -> jnp.ndarray:
    """Dense coefficient matrix for the flat polar radial Laplacian L_m."""
    lap_values = _flat_laplacian_basis_values_np(basis)
    projection = np.asarray(basis.coeff_from_physical_nodal)
    matrix = projection @ lap_values
    return jnp.asarray(matrix, dtype=basis.xi.dtype)


def spherical_laplacian_matrix(basis: RadialBasis) -> jnp.ndarray:
    """Dense coefficient matrix for Lap_sphere = mu(r)^-1 L_m."""
    lap_values = _flat_laplacian_basis_values_np(basis)
    mu_inv = np.asarray(inverse_conformal_factor(basis.r))
    projection = np.asarray(basis.coeff_from_physical_nodal)
    matrix = projection @ (mu_inv[:, None] * lap_values)
    return jnp.asarray(matrix, dtype=basis.xi.dtype)


def helmholtz_matrix(
    basis: RadialBasis,
    Ld_inv_sq: float = 0.0,
    *,
    tau: bool = True,
) -> jnp.ndarray:
    """Dense matrix for (Lap_sphere - Ld_inv_sq) with optional jet tau row."""
    matrix = spherical_laplacian_matrix(basis) - Ld_inv_sq * jnp.eye(
        basis.Nr, dtype=basis.xi.dtype
    )
    if tau:
        matrix = matrix.at[-1, :].set(basis.jet_dirichlet_tau)
    return matrix


def helmholtz_rhs(q_coeffs: jnp.ndarray) -> jnp.ndarray:
    """Right-hand side for tau Helmholtz solve H psi = -q."""
    rhs = -q_coeffs
    return rhs.at[-1, ...].set(0.0)


def solve_helmholtz(
    q_coeffs: jnp.ndarray,
    basis: RadialBasis,
    Ld_inv_sq: float = 0.0,
) -> jnp.ndarray:
    """Solve (Lap_sphere - Ld_inv_sq) psi = -q with jet Dirichlet tau."""
    H = helmholtz_matrix(basis, Ld_inv_sq=Ld_inv_sq, tau=True)
    rhs = helmholtz_rhs(q_coeffs)
    original_shape = rhs.shape
    rhs_flat = rhs.reshape((basis.Nr, -1))
    psi_flat = jnp.linalg.solve(H, rhs_flat)
    return psi_flat.reshape(original_shape)
