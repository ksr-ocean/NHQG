"""Tests for the Jacobi/Zernike radial basis."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from sphere_nhqg.radial import (
    basis_values_at_r,
    coefficients_to_physical,
    derivative_values_at_r,
    flat_inner_product_from_coefficients,
    make_radial_basis,
    physical_to_coefficients,
    radial_derivative_from_coefficients,
    radial_derivative_from_physical,
)


def _as_np(value):
    return np.asarray(value)


def test_gauss_jacobi_orthogonality_and_norms():
    for m in [0, 1, 4]:
        basis = make_radial_basis(m=m, Nr=9, r_jet=0.7)
        V = _as_np(basis.poly_vander)
        W = np.diag(_as_np(basis.jacobi_weights))
        gram = V.T @ W @ V
        expected = np.diag(_as_np(basis.jacobi_norms))

        assert np.allclose(gram, expected, rtol=0.0, atol=2e-13)


def test_physical_transform_roundtrip_batched_complex():
    basis = make_radial_basis(m=3, Nr=8, r_jet=0.6)
    rng = np.random.default_rng(123)
    coeffs_np = rng.normal(size=(8, 2, 3)) + 1j * rng.normal(size=(8, 2, 3))
    coeffs = jnp.asarray(coeffs_np, dtype=jnp.complex128)

    values = coefficients_to_physical(coeffs, basis)
    recovered = physical_to_coefficients(values, basis)

    assert np.allclose(_as_np(recovered), coeffs_np, rtol=0.0, atol=2e-12)


def test_pole_regular_values_are_built_into_basis():
    r_jet = 0.7

    m0_values = basis_values_at_r(m=0, Nr=5, r_jet=r_jet, r=0.0)
    assert np.allclose(m0_values, [(-1.0) ** n for n in range(5)])

    for m in [1, 2, 5]:
        values = basis_values_at_r(m=m, Nr=6, r_jet=r_jet, r=0.0)
        assert np.allclose(values, 0.0, rtol=0.0, atol=0.0)

        radius = 1e-5
        values = basis_values_at_r(m=m, Nr=6, r_jet=r_jet, r=radius)
        scaled = values / ((radius / r_jet) ** m)
        assert np.all(np.isfinite(scaled))


def test_radial_derivative_matches_finite_difference():
    m = 2
    Nr = 7
    r_jet = 0.8
    basis = make_radial_basis(m=m, Nr=Nr, r_jet=r_jet)

    derivative_nodes = _as_np(basis.derivative_vander)
    radii = _as_np(basis.r)
    h = 1e-6

    for j, r in enumerate(radii):
        plus = basis_values_at_r(m=m, Nr=Nr, r_jet=r_jet, r=r + h)
        minus = basis_values_at_r(m=m, Nr=Nr, r_jet=r_jet, r=r - h)
        finite_difference = (plus - minus) / (2.0 * h)

        assert np.allclose(derivative_nodes[j], finite_difference, rtol=2e-8, atol=2e-8)


def test_radial_derivative_from_coefficients_and_nodal_values_agree():
    basis = make_radial_basis(m=4, Nr=8, r_jet=0.5)
    rng = np.random.default_rng(456)
    coeffs_np = rng.normal(size=(8, 3))
    coeffs = jnp.asarray(coeffs_np, dtype=jnp.float64)

    values = coefficients_to_physical(coeffs, basis)
    derivative_from_coeffs = radial_derivative_from_coefficients(coeffs, basis)
    derivative_from_values = radial_derivative_from_physical(values, basis)

    assert np.allclose(_as_np(derivative_from_values), _as_np(derivative_from_coeffs), atol=2e-12)


def test_jet_tau_rows_match_analytic_boundary_values():
    m = 3
    Nr = 8
    r_jet = 0.7
    basis = make_radial_basis(m=m, Nr=Nr, r_jet=r_jet)

    assert np.allclose(_as_np(basis.jet_dirichlet_tau), 1.0)

    jet_values = basis_values_at_r(m=m, Nr=Nr, r_jet=r_jet, r=r_jet)
    jet_derivatives = derivative_values_at_r(m=m, Nr=Nr, r_jet=r_jet, r=r_jet)
    n = np.arange(Nr)
    expected_neumann = (m + 2.0 * n * (n + m + 1.0)) / r_jet

    assert np.allclose(jet_values, _as_np(basis.jet_dirichlet_tau))
    assert np.allclose(jet_derivatives, expected_neumann)
    assert np.allclose(_as_np(basis.jet_neumann_tau), expected_neumann)


def test_flat_inner_product_matches_high_order_quadrature():
    m = 2
    Nr = 6
    r_jet = 0.65
    basis = make_radial_basis(m=m, Nr=Nr, r_jet=r_jet)
    rng = np.random.default_rng(789)
    a = rng.normal(size=Nr)
    b = rng.normal(size=Nr)

    inner_basis = float(flat_inner_product_from_coefficients(
        jnp.asarray(a), jnp.asarray(b), basis
    ))

    x, w = np.polynomial.legendre.leggauss(512)
    r = 0.5 * r_jet * (x + 1.0)
    wr = 0.5 * r_jet * w
    values_a = basis_values_at_r(m=m, Nr=Nr, r_jet=r_jet, r=r) @ a
    values_b = basis_values_at_r(m=m, Nr=Nr, r_jet=r_jet, r=r) @ b
    inner_quad = float(np.sum(wr * values_a * values_b * r))

    assert math.isclose(inner_basis, inner_quad, rel_tol=0.0, abs_tol=2e-13)
