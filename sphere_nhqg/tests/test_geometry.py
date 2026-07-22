"""Tests for stereographic polar-cap geometry."""

from __future__ import annotations

import math

import numpy as np

from sphere_nhqg.geometry import (
    cap_area_fraction_from_latitude,
    cap_radius_from_latitude,
    conformal_factor,
    coriolis_parameter,
    exact_cap_area,
    radial_coriolis_gradient,
)


def _as_float(value) -> float:
    return float(np.asarray(value))


def _quadrature_cap_area(r_jet: float, n: int = 512) -> float:
    x, w = np.polynomial.legendre.leggauss(n)
    r = 0.5 * r_jet * (x + 1.0)
    wr = 0.5 * r_jet * w
    mu = np.asarray(conformal_factor(r))
    return float(2.0 * math.pi * np.sum(wr * mu * r))


def test_cap_radius_known_latitudes():
    r_45 = _as_float(cap_radius_from_latitude(math.radians(45.0)))
    r_30 = _as_float(cap_radius_from_latitude(math.radians(30.0)))

    assert np.isclose(r_45, math.sqrt(2.0) - 1.0, rtol=0.0, atol=1e-12)
    assert np.isclose(r_30, 1.0 / math.sqrt(3.0), rtol=0.0, atol=1e-12)


def test_cap_area_matches_latitude_formula():
    for lat_deg in [75.0, 60.0, 45.0, 30.0]:
        phi = math.radians(lat_deg)
        r_jet = _as_float(cap_radius_from_latitude(phi))
        area_from_r = _as_float(exact_cap_area(r_jet))
        area_from_latitude = 4.0 * math.pi * _as_float(cap_area_fraction_from_latitude(phi))

        assert np.isclose(area_from_r, area_from_latitude, rtol=0.0, atol=2e-12)


def test_area_quadrature_matches_exact_formula():
    for lat_deg in [75.0, 60.0, 45.0, 30.0]:
        r_jet = _as_float(cap_radius_from_latitude(math.radians(lat_deg)))
        exact = _as_float(exact_cap_area(r_jet))
        quadrature = _quadrature_cap_area(r_jet)

        assert np.isclose(quadrature, exact, rtol=0.0, atol=5e-12)


def test_small_cap_area_limit_has_stereographic_factor_four():
    for r_jet in [1e-1, 1e-2, 1e-3, 1e-4]:
        area = _as_float(exact_cap_area(r_jet))
        flat_projected_area = math.pi * r_jet * r_jet
        ratio = area / flat_projected_area

        assert abs(ratio - 4.0) < 5.0 * r_jet * r_jet


def test_coriolis_profile_matches_spherical_latitude():
    omega = 1.7

    assert np.isclose(_as_float(coriolis_parameter(0.0, omega)), 2.0 * omega)
    assert np.isclose(_as_float(coriolis_parameter(1.0, omega)), 0.0)

    for lat_deg in [75.0, 60.0, 45.0, 30.0]:
        phi = math.radians(lat_deg)
        r_jet = _as_float(cap_radius_from_latitude(phi))
        expected = 2.0 * omega * math.sin(phi)

        assert np.isclose(_as_float(coriolis_parameter(r_jet, omega)), expected, atol=2e-12)


def test_radial_coriolis_gradient_matches_finite_difference():
    omega = 1.7
    h = 1e-6

    for r in [0.05, 0.2, 0.45, 0.8]:
        f_plus = _as_float(coriolis_parameter(r + h, omega))
        f_minus = _as_float(coriolis_parameter(r - h, omega))
        finite_difference = (f_plus - f_minus) / (2.0 * h)
        analytic = _as_float(radial_coriolis_gradient(r, omega))

        assert np.isclose(analytic, finite_difference, rtol=2e-9, atol=2e-9)
