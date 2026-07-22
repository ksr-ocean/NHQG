"""Tests for spherical-area mean primitives."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from sphere_nhqg.geometry import cap_radius_from_latitude, coriolis_parameter
from sphere_nhqg.mean_exchange import (
    spherical_area_mean_axisymmetric,
    spherical_area_mean_weights,
)
from sphere_nhqg.radial import make_radial_basis


def _as_np(value):
    return np.asarray(value)


def test_spherical_area_mean_weights_normalize_constants():
    r_jet = float(cap_radius_from_latitude(math.radians(45.0)))
    basis = make_radial_basis(m=0, Nr=64, r_jet=r_jet)
    weights = spherical_area_mean_weights(basis)

    assert np.isclose(float(jnp.sum(weights)), 1.0, rtol=0.0, atol=3e-15)
    assert np.all(_as_np(weights) > 0.0)


def test_constant_axisymmetric_mean_is_one():
    r_jet = float(cap_radius_from_latitude(math.radians(30.0)))
    basis = make_radial_basis(m=0, Nr=64, r_jet=r_jet)
    values = jnp.ones((basis.Nr,), dtype=jnp.float64)

    assert np.isclose(float(spherical_area_mean_axisymmetric(values, basis)), 1.0, atol=3e-15)


def test_axisymmetric_mean_preserves_trailing_dimensions():
    r_jet = float(cap_radius_from_latitude(math.radians(60.0)))
    basis = make_radial_basis(m=0, Nr=48, r_jet=r_jet)
    values = basis.r[:, None] ** 2 + jnp.arange(3, dtype=jnp.float64)[None, :]
    means = spherical_area_mean_axisymmetric(values, basis)

    assert means.shape == (3,)
    assert np.all(np.diff(_as_np(means)) > 0.0)


def test_spherical_mean_of_coriolis_matches_cap_average():
    omega = 1.7
    for lat_deg in [75.0, 60.0, 45.0, 30.0]:
        phi = math.radians(lat_deg)
        r_jet = float(cap_radius_from_latitude(phi))
        basis = make_radial_basis(m=0, Nr=96, r_jet=r_jet)
        values = coriolis_parameter(basis.r, omega)
        mean_f = float(spherical_area_mean_axisymmetric(values, basis))
        expected = omega * (1.0 + math.sin(phi))

        assert np.isclose(mean_f, expected, rtol=0.0, atol=5e-14)


def test_spherical_area_mean_rejects_nonzero_m_basis():
    basis = make_radial_basis(m=1, Nr=8, r_jet=0.5)
    values = jnp.ones((basis.Nr,), dtype=jnp.float64)

    with pytest.raises(ValueError, match="m=0"):
        spherical_area_mean_axisymmetric(values, basis)
