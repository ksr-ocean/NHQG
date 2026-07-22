"""Latitude masks for broad spherical QG validity/sponge envelopes."""

from __future__ import annotations

import jax.numpy as jnp


def southern_tukey_mask(
    latitudes: jnp.ndarray,
    plateau_north_edge_deg: float = -30.0,
    taper_north_edge_deg: float = 5.0,
) -> jnp.ndarray:
    """Smooth southern-hemisphere Tukey-style mask.

    The mask is one southward of ``plateau_north_edge_deg``, tapers smoothly to
    zero, and is zero northward of ``taper_north_edge_deg``.
    """
    if taper_north_edge_deg <= plateau_north_edge_deg:
        raise ValueError("taper_north_edge_deg must be north of plateau_north_edge_deg")

    lat_deg = jnp.rad2deg(latitudes)
    x = (lat_deg - plateau_north_edge_deg) / (
        taper_north_edge_deg - plateau_north_edge_deg
    )
    x = jnp.clip(x, 0.0, 1.0)
    taper = 0.5 * (1.0 + jnp.cos(jnp.pi * x))
    return jnp.where(lat_deg <= plateau_north_edge_deg, 1.0, taper)


def sponge_rate_from_mask(
    mask: jnp.ndarray,
    max_rate: float,
    power: float = 1.0,
) -> jnp.ndarray:
    """Convert a validity mask to a nonnegative sponge rate."""
    if max_rate < 0.0:
        raise ValueError("max_rate must be nonnegative")
    if power <= 0.0:
        raise ValueError("power must be positive")
    return max_rate * (1.0 - mask) ** power
