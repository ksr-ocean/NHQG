"""Tests for scalar stereographic radial operators."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from sphere_nhqg.geometry import inverse_conformal_factor
from sphere_nhqg.operators import (
    flat_laplacian_matrix,
    helmholtz_matrix,
    solve_helmholtz,
    spherical_laplacian_matrix,
)
from sphere_nhqg.radial import (
    coefficients_to_physical,
    make_radial_basis,
    physical_to_coefficients,
)


def _as_np(value):
    return np.asarray(value)


def test_flat_laplacian_annihilates_leading_harmonic():
    for m in [0, 1, 4]:
        basis = make_radial_basis(m=m, Nr=8, r_jet=0.7)
        L = _as_np(flat_laplacian_matrix(basis))

        assert np.allclose(L[:, 0], 0.0, rtol=0.0, atol=2e-12)


def test_flat_laplacian_of_next_radial_power():
    for m in [0, 2, 5]:
        Nr = 9
        r_jet = 0.8
        basis = make_radial_basis(m=m, Nr=Nr, r_jet=r_jet)
        rho = _as_np(basis.rho)
        values = jnp.asarray(rho ** (m + 2), dtype=jnp.float64)
        coeffs = physical_to_coefficients(values, basis)

        lap_coeffs = flat_laplacian_matrix(basis) @ coeffs
        lap_values = coefficients_to_physical(lap_coeffs, basis)
        expected = (4.0 * (m + 1.0) / (r_jet * r_jet)) * rho**m

        assert np.allclose(_as_np(lap_values), expected, rtol=0.0, atol=3e-11)


def test_spherical_laplacian_is_pointwise_mu_inverse_times_flat_laplacian():
    m = 3
    Nr = 9
    r_jet = 0.6
    basis = make_radial_basis(m=m, Nr=Nr, r_jet=r_jet)
    rng = np.random.default_rng(321)
    coeffs = jnp.asarray(rng.normal(size=Nr), dtype=jnp.float64)

    flat_values = coefficients_to_physical(flat_laplacian_matrix(basis) @ coeffs, basis)
    spherical_values = coefficients_to_physical(
        spherical_laplacian_matrix(basis) @ coeffs, basis
    )
    expected = inverse_conformal_factor(basis.r) * flat_values

    assert np.allclose(_as_np(spherical_values), _as_np(expected), rtol=0.0, atol=5e-12)


def test_helmholtz_tau_solve_recovers_manufactured_dirichlet_solution():
    m = 2
    Nr = 8
    r_jet = 0.7
    Ld_inv_sq = 0.3
    basis = make_radial_basis(m=m, Nr=Nr, r_jet=r_jet)
    rng = np.random.default_rng(654)
    psi_np = rng.normal(size=Nr)
    psi_np[-1] = -np.sum(psi_np[:-1])
    psi = jnp.asarray(psi_np, dtype=jnp.float64)

    H = helmholtz_matrix(basis, Ld_inv_sq=Ld_inv_sq, tau=False)
    q = -(H @ psi)
    recovered = solve_helmholtz(q, basis, Ld_inv_sq=Ld_inv_sq)

    assert np.allclose(_as_np(recovered), psi_np, rtol=0.0, atol=2e-11)


def test_helmholtz_tau_solve_handles_batched_complex_rhs():
    m = 1
    Nr = 7
    r_jet = 0.5
    Ld_inv_sq = 0.2
    basis = make_radial_basis(m=m, Nr=Nr, r_jet=r_jet)
    rng = np.random.default_rng(987)
    psi_np = rng.normal(size=(Nr, 2, 3)) + 1j * rng.normal(size=(Nr, 2, 3))
    psi_np[-1, ...] = -np.sum(psi_np[:-1, ...], axis=0)
    psi = jnp.asarray(psi_np, dtype=jnp.complex128)

    H = helmholtz_matrix(basis, Ld_inv_sq=Ld_inv_sq, tau=False)
    q = -jnp.einsum("ij,j...->i...", H, psi)
    recovered = solve_helmholtz(q, basis, Ld_inv_sq=Ld_inv_sq)

    assert np.allclose(_as_np(recovered), psi_np, rtol=0.0, atol=3e-11)
