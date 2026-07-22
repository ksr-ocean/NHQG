"""Jacobi/Zernike radial basis for the stereographic polar disk.

For azimuthal mode ``m``, smooth scalar fields have radial profiles

    F_m(r) = (r / r_jet)**|m| * sum_n c_n P_n^(0, |m|)(xi),
    xi = 2 (r / r_jet)**2 - 1.

The leading power enforces pole regularity by construction. This module builds
the dense transform/projection matrices needed by the first JAX prototype; it
does not yet build the spherical Laplacian or Helmholtz operators.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from scipy.special import eval_jacobi, roots_jacobi


@dataclass(frozen=True)
class RadialBasis:
    """Precomputed radial basis data for one azimuthal mode."""

    m: int
    Nr: int
    r_jet: float
    xi: jnp.ndarray
    r: jnp.ndarray
    rho: jnp.ndarray
    jacobi_weights: jnp.ndarray
    jacobi_norms: jnp.ndarray
    flat_basis_norms: jnp.ndarray
    poly_vander: jnp.ndarray
    physical_vander: jnp.ndarray
    coeff_from_poly_nodal: jnp.ndarray
    coeff_from_physical_nodal: jnp.ndarray
    derivative_vander: jnp.ndarray
    derivative_from_physical_nodal: jnp.ndarray
    jet_dirichlet_tau: jnp.ndarray
    jet_neumann_tau: jnp.ndarray


def _jax_dtype(dtype) -> jnp.dtype:
    return jnp.dtype(dtype)


def _np_dtype(dtype) -> np.dtype:
    return np.dtype(jnp.dtype(dtype).name)


def jacobi_norms(m: int, Nr: int, dtype=jnp.float64) -> jnp.ndarray:
    """Norms of P_n^(0,m) under weight (1+xi)^m on [-1, 1]."""
    if m < 0:
        raise ValueError("m must be nonnegative")
    if Nr <= 0:
        raise ValueError("Nr must be positive")

    n = jnp.arange(Nr, dtype=_jax_dtype(dtype))
    return (2.0 ** (m + 1)) / (2.0 * n + m + 1.0)


def basis_values_at_r(
    m: int,
    Nr: int,
    r_jet: float,
    r: float | np.ndarray,
    dtype=np.float64,
) -> np.ndarray:
    """Evaluate all radial basis functions at arbitrary projected radius."""
    if m < 0:
        raise ValueError("m must be nonnegative")
    if Nr <= 0:
        raise ValueError("Nr must be positive")
    if r_jet <= 0.0:
        raise ValueError("r_jet must be positive")

    np_dtype = np.dtype(dtype)
    radius = np.asarray(r, dtype=np_dtype)
    rho = radius / np_dtype.type(r_jet)
    xi = 2.0 * rho * rho - 1.0
    values = np.empty(radius.shape + (Nr,), dtype=np_dtype)
    prefactor = rho ** m
    for n in range(Nr):
        values[..., n] = prefactor * eval_jacobi(n, 0.0, float(m), xi)
    return values


def derivative_values_at_r(
    m: int,
    Nr: int,
    r_jet: float,
    r: float | np.ndarray,
    dtype=np.float64,
) -> np.ndarray:
    """Evaluate d/dr of all radial basis functions at arbitrary radius."""
    if m < 0:
        raise ValueError("m must be nonnegative")
    if Nr <= 0:
        raise ValueError("Nr must be positive")
    if r_jet <= 0.0:
        raise ValueError("r_jet must be positive")

    np_dtype = np.dtype(dtype)
    radius = np.asarray(r, dtype=np_dtype)
    rho = radius / np_dtype.type(r_jet)
    xi = 2.0 * rho * rho - 1.0
    values = np.empty(radius.shape + (Nr,), dtype=np_dtype)

    for n in range(Nr):
        poly = eval_jacobi(n, 0.0, float(m), xi)
        if n == 0:
            dpoly_dxi = np.zeros_like(radius, dtype=np_dtype)
        else:
            dpoly_dxi = 0.5 * (n + m + 1.0) * eval_jacobi(
                n - 1, 1.0, float(m + 1), xi
            )

        if m == 0:
            prefactor_derivative = np.zeros_like(radius, dtype=np_dtype)
        else:
            prefactor_derivative = (m / r_jet) * rho ** (m - 1)
        chain_derivative = (4.0 / r_jet) * rho ** (m + 1) * dpoly_dxi
        values[..., n] = prefactor_derivative * poly + chain_derivative
    return values


def make_radial_basis(
    m: int,
    Nr: int,
    r_jet: float,
    dtype=jnp.float64,
) -> RadialBasis:
    """Build transform and quadrature matrices for one azimuthal mode."""
    if m < 0:
        raise ValueError("m must be nonnegative")
    if Nr <= 0:
        raise ValueError("Nr must be positive")
    if r_jet <= 0.0:
        raise ValueError("r_jet must be positive")

    np_dtype = _np_dtype(dtype)
    jax_dtype = _jax_dtype(dtype)

    xi_np, weights_np = roots_jacobi(Nr, 0.0, float(m))
    xi_np = xi_np.astype(np_dtype)
    weights_np = weights_np.astype(np_dtype)
    rho_np = np.sqrt((1.0 + xi_np) / 2.0).astype(np_dtype)
    r_np = (float(r_jet) * rho_np).astype(np_dtype)

    poly_vander_np = np.empty((Nr, Nr), dtype=np_dtype)
    for n in range(Nr):
        poly_vander_np[:, n] = eval_jacobi(n, 0.0, float(m), xi_np)

    physical_vander_np = (rho_np[:, None] ** m) * poly_vander_np
    norms_np = np.asarray(jacobi_norms(m, Nr, dtype=dtype), dtype=np_dtype)

    coeff_from_poly_np = (poly_vander_np.T * weights_np[None, :]) / norms_np[:, None]
    rho_power_np = rho_np ** m
    coeff_from_physical_np = coeff_from_poly_np / rho_power_np[None, :]

    derivative_vander_np = derivative_values_at_r(
        m, Nr, r_jet, r_np, dtype=np_dtype
    ).reshape(Nr, Nr)
    derivative_from_physical_np = derivative_vander_np @ coeff_from_physical_np

    n_np = np.arange(Nr, dtype=np_dtype)
    jet_dirichlet_tau_np = np.ones(Nr, dtype=np_dtype)
    jet_neumann_tau_np = (
        float(m) + 2.0 * n_np * (n_np + float(m) + 1.0)
    ) / float(r_jet)

    flat_scale = float(r_jet) ** 2 / (4.0 * (2.0 ** m))
    flat_basis_norms_np = flat_scale * norms_np

    return RadialBasis(
        m=m,
        Nr=Nr,
        r_jet=float(r_jet),
        xi=jnp.asarray(xi_np, dtype=jax_dtype),
        r=jnp.asarray(r_np, dtype=jax_dtype),
        rho=jnp.asarray(rho_np, dtype=jax_dtype),
        jacobi_weights=jnp.asarray(weights_np, dtype=jax_dtype),
        jacobi_norms=jnp.asarray(norms_np, dtype=jax_dtype),
        flat_basis_norms=jnp.asarray(flat_basis_norms_np, dtype=jax_dtype),
        poly_vander=jnp.asarray(poly_vander_np, dtype=jax_dtype),
        physical_vander=jnp.asarray(physical_vander_np, dtype=jax_dtype),
        coeff_from_poly_nodal=jnp.asarray(coeff_from_poly_np, dtype=jax_dtype),
        coeff_from_physical_nodal=jnp.asarray(coeff_from_physical_np, dtype=jax_dtype),
        derivative_vander=jnp.asarray(derivative_vander_np, dtype=jax_dtype),
        derivative_from_physical_nodal=jnp.asarray(
            derivative_from_physical_np, dtype=jax_dtype
        ),
        jet_dirichlet_tau=jnp.asarray(jet_dirichlet_tau_np, dtype=jax_dtype),
        jet_neumann_tau=jnp.asarray(jet_neumann_tau_np, dtype=jax_dtype),
    )


def coefficients_to_physical(coeffs: jnp.ndarray, basis: RadialBasis) -> jnp.ndarray:
    """Evaluate radial coefficients at the basis quadrature nodes."""
    return jnp.einsum("jn,n...->j...", basis.physical_vander, coeffs)


def physical_to_coefficients(values: jnp.ndarray, basis: RadialBasis) -> jnp.ndarray:
    """Project physical nodal values back to radial coefficients."""
    return jnp.einsum("nj,j...->n...", basis.coeff_from_physical_nodal, values)


def polynomial_to_coefficients(values: jnp.ndarray, basis: RadialBasis) -> jnp.ndarray:
    """Project polynomial-part nodal values to Jacobi coefficients."""
    return jnp.einsum("nj,j...->n...", basis.coeff_from_poly_nodal, values)


def radial_derivative_from_coefficients(
    coeffs: jnp.ndarray,
    basis: RadialBasis,
) -> jnp.ndarray:
    """Evaluate radial derivative at the basis quadrature nodes."""
    return jnp.einsum("jn,n...->j...", basis.derivative_vander, coeffs)


def radial_derivative_from_physical(
    values: jnp.ndarray,
    basis: RadialBasis,
) -> jnp.ndarray:
    """Evaluate radial derivative from physical nodal values."""
    return jnp.einsum("ij,j...->i...", basis.derivative_from_physical_nodal, values)


def flat_inner_product_from_coefficients(
    a_coeffs: jnp.ndarray,
    b_coeffs: jnp.ndarray,
    basis: RadialBasis,
) -> jnp.ndarray:
    """Flat-disk radial inner product int_0^rjet a(r) b(r) r dr."""
    return jnp.einsum("n,n...,n...->...", basis.flat_basis_norms, a_coeffs, b_coeffs)
