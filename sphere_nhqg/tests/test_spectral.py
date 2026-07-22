"""Tests for coupled azimuthal/radial spectral transforms."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from sphere_nhqg.spectral import (
    azimuthal_derivative_to_physical,
    coefficients_to_physical,
    constant_coefficients,
    dealiased_product,
    flat_jacobian_coefficients,
    flat_jacobian_physical,
    make_spectral_grid,
    physical_to_coefficients,
    radial_derivative_to_physical,
    spherical_jacobian_coefficients,
    spherical_jacobian_physical,
)
from sphere_nhqg.geometry import inverse_conformal_factor


def _as_np(value):
    return np.asarray(value)


def _random_real_field_coeffs(grid, seed: int = 0):
    rng = np.random.default_rng(seed)
    coeffs = rng.normal(size=(grid.Nm, grid.Nr, 2)) + 1j * rng.normal(
        size=(grid.Nm, grid.Nr, 2)
    )
    coeffs[0, :, :] = coeffs[0, :, :].real
    coeffs[-1, :, :] = coeffs[-1, :, :].real
    return jnp.asarray(coeffs, dtype=jnp.complex128)


def test_constant_coefficients_use_dft_normalization():
    grid = make_spectral_grid(Nphi=12, Nr=5, r_jet=0.6, Nphi_phys=18, Nr_phys=8)
    coeffs = constant_coefficients(2.5, grid)
    physical = coefficients_to_physical(coeffs, grid)

    assert coeffs[0, 0] == 2.5 * grid.Nphi
    assert np.allclose(_as_np(physical), 2.5, rtol=0.0, atol=2e-14)


def test_transform_roundtrip_on_overresolved_grid():
    grid = make_spectral_grid(Nphi=12, Nr=6, r_jet=0.7, Nphi_phys=18, Nr_phys=10)
    coeffs = _random_real_field_coeffs(grid, seed=123)

    physical = coefficients_to_physical(coeffs, grid)
    recovered = physical_to_coefficients(physical, grid)

    assert np.allclose(_as_np(recovered), _as_np(coeffs), rtol=0.0, atol=5e-11)


def test_padded_transform_preserves_bandlimited_fourier_amplitude():
    base = make_spectral_grid(Nphi=12, Nr=4, r_jet=0.5)
    padded = make_spectral_grid(Nphi=12, Nr=4, r_jet=0.5, Nphi_phys=24, Nr_phys=7)

    coeffs = jnp.zeros((base.Nm, base.Nr), dtype=jnp.complex128)
    coeffs = coeffs.at[1, 0].set(base.Nphi / 2.0)

    phi_base = np.linspace(0.0, 2.0 * np.pi, base.Nphi, endpoint=False)
    phi_padded = np.linspace(0.0, 2.0 * np.pi, padded.Nphi_phys, endpoint=False)
    physical_base = coefficients_to_physical(coeffs, base)
    physical_padded = coefficients_to_physical(coeffs, padded)

    expected_base = (base.r / base.r_jet)[:, None] * np.cos(phi_base)[None, :]
    expected_padded = (padded.r / padded.r_jet)[:, None] * np.cos(phi_padded)[None, :]

    assert np.allclose(_as_np(physical_base), _as_np(expected_base), atol=2e-14)
    assert np.allclose(_as_np(physical_padded), _as_np(expected_padded), atol=2e-14)


def test_dealiased_cos_squared_has_expected_fourier_coefficients():
    grid = make_spectral_grid(Nphi=12, Nr=4, r_jet=0.6, Nphi_phys=18, Nr_phys=8)
    cos_coeffs = jnp.zeros((grid.Nm, grid.Nr), dtype=jnp.complex128)
    cos_coeffs = cos_coeffs.at[1, 0].set(grid.Nphi / 2.0)

    product = dealiased_product(cos_coeffs, cos_coeffs, grid)
    expected = jnp.zeros_like(product)
    expected = expected.at[0, 0].set(grid.Nphi / 4.0)
    expected = expected.at[0, 1].set(grid.Nphi / 4.0)
    expected = expected.at[2, 0].set(grid.Nphi / 4.0)

    assert np.allclose(_as_np(product), _as_np(expected), rtol=0.0, atol=2e-12)


def test_dealiased_radial_axisymmetric_product_projects_to_represented_power():
    grid = make_spectral_grid(Nphi=8, Nr=5, r_jet=0.75, Nphi_phys=12, Nr_phys=9)
    rho_sq_values = (grid.r / grid.r_jet) ** 2
    physical = rho_sq_values[:, None] * jnp.ones((1, grid.Nphi_phys), dtype=jnp.float64)
    coeffs = physical_to_coefficients(physical, grid)

    product_coeffs = dealiased_product(coeffs, coeffs, grid)
    product_physical = coefficients_to_physical(product_coeffs, grid)
    expected = ((grid.r / grid.r_jet) ** 4)[:, None]

    assert np.allclose(_as_np(product_physical), _as_np(expected), rtol=0.0, atol=4e-12)


def test_physical_projection_rejects_wrong_shape():
    grid = make_spectral_grid(Nphi=8, Nr=4, r_jet=0.5)
    wrong = jnp.ones((grid.Nr_phys, grid.Nphi_phys + 1), dtype=jnp.float64)

    try:
        physical_to_coefficients(wrong, grid)
    except ValueError as exc:
        assert "values must have shape" in str(exc)
    else:
        raise AssertionError("expected shape validation to fail")


def test_derivatives_of_regular_m1_mode():
    grid = make_spectral_grid(Nphi=12, Nr=5, r_jet=0.6, Nphi_phys=18, Nr_phys=8)
    coeffs = jnp.zeros((grid.Nm, grid.Nr), dtype=jnp.complex128)
    coeffs = coeffs.at[1, 0].set(grid.Nphi / 2.0)
    phi = np.linspace(0.0, 2.0 * np.pi, grid.Nphi_phys, endpoint=False)
    rho = grid.r / grid.r_jet

    radial = radial_derivative_to_physical(coeffs, grid)
    azimuthal = azimuthal_derivative_to_physical(coeffs, grid)
    expected_radial = (1.0 / grid.r_jet) * np.cos(phi)[None, :]
    expected_azimuthal = -rho[:, None] * np.sin(phi)[None, :]

    assert np.allclose(_as_np(radial), _as_np(expected_radial), atol=3e-14)
    assert np.allclose(_as_np(azimuthal), _as_np(expected_azimuthal), atol=3e-14)


def test_flat_jacobian_matches_regular_manufactured_pair():
    grid = make_spectral_grid(Nphi=12, Nr=5, r_jet=0.7, Nphi_phys=18, Nr_phys=9)
    a = jnp.zeros((grid.Nm, grid.Nr), dtype=jnp.complex128)
    b = jnp.zeros_like(a)
    a = a.at[0, 0].set(grid.Nphi / 4.0)
    a = a.at[0, 1].set(grid.Nphi / 4.0)
    b = b.at[1, 0].set(grid.Nphi / 2.0)

    phi = np.linspace(0.0, 2.0 * np.pi, grid.Nphi_phys, endpoint=False)
    rho = grid.r / grid.r_jet
    expected = -(rho[:, None] / (grid.r_jet * grid.r_jet)) * np.sin(phi)[None, :]
    jac = flat_jacobian_physical(a, b, grid)

    assert np.allclose(_as_np(jac), _as_np(expected), rtol=0.0, atol=8e-13)


def test_spherical_jacobian_is_metric_factor_times_flat_jacobian():
    grid = make_spectral_grid(Nphi=12, Nr=5, r_jet=0.7, Nphi_phys=18, Nr_phys=9)
    a = _random_real_field_coeffs(grid, seed=11)
    b = _random_real_field_coeffs(grid, seed=12)

    flat = flat_jacobian_physical(a, b, grid)
    spherical = spherical_jacobian_physical(a, b, grid)
    expected = inverse_conformal_factor(grid.r)[:, None, None] * flat

    assert np.allclose(_as_np(spherical), _as_np(expected), rtol=0.0, atol=2e-13)


def test_self_jacobian_projects_to_zero():
    grid = make_spectral_grid(Nphi=12, Nr=5, r_jet=0.6, Nphi_phys=18, Nr_phys=9)
    a = _random_real_field_coeffs(grid, seed=21)

    jac = spherical_jacobian_coefficients(a, a, grid)

    assert np.allclose(_as_np(jac), 0.0, rtol=0.0, atol=2e-11)


def test_jacobian_antisymmetry_after_projection():
    grid = make_spectral_grid(Nphi=12, Nr=5, r_jet=0.6, Nphi_phys=18, Nr_phys=9)
    a = _random_real_field_coeffs(grid, seed=31)
    b = _random_real_field_coeffs(grid, seed=32)

    flat_sum = flat_jacobian_coefficients(a, b, grid) + flat_jacobian_coefficients(b, a, grid)
    spherical_sum = spherical_jacobian_coefficients(a, b, grid) + spherical_jacobian_coefficients(
        b, a, grid
    )

    assert np.allclose(_as_np(flat_sum), 0.0, rtol=0.0, atol=4e-11)
    assert np.allclose(_as_np(spherical_sum), 0.0, rtol=0.0, atol=4e-11)
