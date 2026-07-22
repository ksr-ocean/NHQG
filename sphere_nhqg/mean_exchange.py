"""Spherical-area horizontal mean primitives.

The full SBP thermal corrector will live here later. For now this module owns
the horizontal mean convention that the corrector and diagnostics must share.
"""

from __future__ import annotations

import jax.numpy as jnp

from sphere_nhqg.geometry import exact_cap_area, conformal_factor
from sphere_nhqg.radial import RadialBasis


def spherical_area_mean_weights(basis: RadialBasis) -> jnp.ndarray:
    """Weights mapping m=0 radial nodal values to spherical cap means."""
    if basis.m != 0:
        raise ValueError("spherical area means require an m=0 radial basis")

    area = exact_cap_area(basis.r_jet)
    radial_scale = basis.r_jet * basis.r_jet / 4.0
    return (2.0 * jnp.pi / area) * radial_scale * basis.jacobi_weights * conformal_factor(
        basis.r
    )


def spherical_area_mean_axisymmetric(
    values: jnp.ndarray,
    basis: RadialBasis,
) -> jnp.ndarray:
    """Compute the spherical area mean of azimuthally averaged radial data.

    ``values`` is expected to have radial node axis first and to represent the
    physical azimuthal mean F_0(r, ...), not radial coefficients.
    """
    weights = spherical_area_mean_weights(basis)
    return jnp.einsum("j,j...->...", weights, values)
