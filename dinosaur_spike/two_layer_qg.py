"""Two-layer Phillips QG building blocks on Dinosaur modal grids."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class TwoLayerState(NamedTuple):
    """Layer PV anomalies in Dinosaur modal layout."""

    q1: jax.Array
    q2: jax.Array


class TwoLayerPsi(NamedTuple):
    """Layer streamfunctions in Dinosaur modal layout."""

    psi1: jax.Array
    psi2: jax.Array


def _laplacian_eigenvalues(grid) -> jnp.ndarray:
    return jnp.asarray(grid.laplacian_eigenvalues)


def pv_from_streamfunction(
    psi: TwoLayerPsi,
    grid,
    F1: float,
    F2: float,
) -> TwoLayerState:
    """Compute two-layer PV anomalies from streamfunction coefficients."""
    lap1 = grid.laplacian(psi.psi1)
    lap2 = grid.laplacian(psi.psi2)
    q1 = lap1 + F1 * (psi.psi2 - psi.psi1)
    q2 = lap2 + F2 * (psi.psi1 - psi.psi2)
    return TwoLayerState(q1=q1, q2=q2)


def invert_streamfunction(
    state: TwoLayerState,
    grid,
    F1: float,
    F2: float,
) -> TwoLayerPsi:
    """Invert two-layer PV anomalies to streamfunctions mode-by-mode."""
    q1 = state.q1
    q2 = state.q2
    L = -_laplacian_eigenvalues(grid)
    a = -L - F1
    d = -L - F2
    b = F1
    c = F2
    det = a * d - b * c
    valid = (L > 0.0) & jnp.asarray(grid.mask)
    det_safe = jnp.where(valid, det, 1.0)

    psi1 = jnp.where(valid, (d * q1 - b * q2) / det_safe, 0.0)
    psi2 = jnp.where(valid, (-c * q1 + a * q2) / det_safe, 0.0)

    # Gauge fix the barotropic null mode. The 2x2 matrix is singular at l=0.
    zero = jnp.asarray(0.0, dtype=psi1.dtype)
    psi1 = psi1.at[..., 0, 0].set(zero)
    psi2 = psi2.at[..., 0, 0].set(zero)
    return TwoLayerPsi(psi1=psi1, psi2=psi2)


def remove_mean_pv(state: TwoLayerState) -> TwoLayerState:
    """Remove the modal mean from both layer PV anomalies."""
    zero = jnp.asarray(0.0, dtype=state.q1.dtype)
    q1 = state.q1.at[..., 0, 0].set(zero)
    q2 = state.q2.at[..., 0, 0].set(zero)
    return TwoLayerState(q1=q1, q2=q2)
