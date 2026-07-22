"""Stereographic polar-cap geometry for the spherical NHQG solver.

All formulas assume a unit sphere. A dimensional planet radius can be restored
later by multiplying metric/area quantities by the appropriate powers of that
radius; keeping the first implementation nondimensional makes normalization
tests easier to audit.
"""

from __future__ import annotations

import jax.numpy as jnp


def cap_radius_from_latitude(phi_jet: float | jnp.ndarray) -> jnp.ndarray:
    """Projected disk radius for a northern jet latitude in radians."""
    phi = jnp.asarray(phi_jet)
    return jnp.tan(jnp.pi / 4.0 - 0.5 * phi)


def cap_radius_from_colatitude(theta_jet: float | jnp.ndarray) -> jnp.ndarray:
    """Projected disk radius for cap-edge colatitude in radians."""
    theta = jnp.asarray(theta_jet)
    return jnp.tan(0.5 * theta)


def stereographic_colatitude_from_radius(r: float | jnp.ndarray) -> jnp.ndarray:
    """Inverse stereographic map: projected radius to colatitude."""
    radius = jnp.asarray(r)
    return 2.0 * jnp.arctan(radius)


def latitude_from_radius(r: float | jnp.ndarray) -> jnp.ndarray:
    """Inverse stereographic map: projected radius to latitude."""
    return jnp.pi / 2.0 - stereographic_colatitude_from_radius(r)


def conformal_factor(r: float | jnp.ndarray) -> jnp.ndarray:
    """Metric conformal factor mu(r) for ds^2 = mu(r)(dr^2 + r^2 dphi^2)."""
    radius = jnp.asarray(r)
    return 4.0 / (1.0 + radius * radius) ** 2


def inverse_conformal_factor(r: float | jnp.ndarray) -> jnp.ndarray:
    """Inverse conformal factor mu(r)^-1."""
    radius = jnp.asarray(r)
    return 0.25 * (1.0 + radius * radius) ** 2


def spherical_area_density(r: float | jnp.ndarray) -> jnp.ndarray:
    """Radial area density for dA = mu(r) r dr dphi."""
    radius = jnp.asarray(r)
    return conformal_factor(radius) * radius


def exact_cap_area(r_jet: float | jnp.ndarray) -> jnp.ndarray:
    """Exact spherical area of the projected polar cap."""
    radius = jnp.asarray(r_jet)
    return 4.0 * jnp.pi * radius * radius / (1.0 + radius * radius)


def cap_area_fraction_from_latitude(phi_jet: float | jnp.ndarray) -> jnp.ndarray:
    """Fraction of the full unit sphere poleward of latitude phi_jet."""
    phi = jnp.asarray(phi_jet)
    return 0.5 * (1.0 - jnp.sin(phi))


def coriolis_parameter(
    r: float | jnp.ndarray,
    rotation_rate: float | jnp.ndarray = 1.0,
) -> jnp.ndarray:
    """Spherical Coriolis profile f(r) = 2 Omega (1-r^2)/(1+r^2)."""
    radius = jnp.asarray(r)
    omega = jnp.asarray(rotation_rate)
    return 2.0 * omega * (1.0 - radius * radius) / (1.0 + radius * radius)


def radial_coriolis_gradient(
    r: float | jnp.ndarray,
    rotation_rate: float | jnp.ndarray = 1.0,
) -> jnp.ndarray:
    """Radial derivative df/dr of the stereographic Coriolis profile."""
    radius = jnp.asarray(r)
    omega = jnp.asarray(rotation_rate)
    return -8.0 * omega * radius / (1.0 + radius * radius) ** 2


def radial_arc_length_factor(r: float | jnp.ndarray) -> jnp.ndarray:
    """Factor converting projected dr to spherical arc length."""
    return jnp.sqrt(conformal_factor(r))


def azimuthal_arc_length_factor(r: float | jnp.ndarray) -> jnp.ndarray:
    """Factor converting projected dphi to spherical azimuthal arc length."""
    radius = jnp.asarray(r)
    return jnp.sqrt(conformal_factor(radius)) * radius
