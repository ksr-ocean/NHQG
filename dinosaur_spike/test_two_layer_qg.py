"""Tests for two-layer QG modal inversion."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from dinosaur import spherical_harmonic

from dinosaur_spike.two_layer_qg import (
    TwoLayerPsi,
    TwoLayerState,
    invert_streamfunction,
    pv_from_streamfunction,
    remove_mean_pv,
)


def _grid():
    return spherical_harmonic.Grid.with_wavenumbers(
        longitude_wavenumbers=8,
        dealiasing="quadratic",
        spherical_harmonics_impl=spherical_harmonic.RealSphericalHarmonics,
        radius=1.0,
    )


def _random_modal(grid, seed: int):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=grid.modal_shape).astype(np.float32)
    values = values * grid.mask
    values[0, 0] = 0.0
    return jnp.asarray(values)


def test_two_layer_inversion_recovers_manufactured_streamfunctions():
    grid = _grid()
    psi = TwoLayerPsi(
        psi1=_random_modal(grid, 1),
        psi2=_random_modal(grid, 2),
    )
    q = pv_from_streamfunction(psi, grid, F1=0.7, F2=0.4)
    recovered = invert_streamfunction(q, grid, F1=0.7, F2=0.4)

    assert np.allclose(np.asarray(recovered.psi1), np.asarray(psi.psi1), atol=5e-6)
    assert np.allclose(np.asarray(recovered.psi2), np.asarray(psi.psi2), atol=5e-6)


def test_two_layer_inversion_sets_mean_streamfunction_gauge():
    grid = _grid()
    q = TwoLayerState(
        q1=_random_modal(grid, 3).at[0, 0].set(1.0),
        q2=_random_modal(grid, 4).at[0, 0].set(-2.0),
    )
    psi = invert_streamfunction(q, grid, F1=0.7, F2=0.4)

    assert float(psi.psi1[0, 0]) == 0.0
    assert float(psi.psi2[0, 0]) == 0.0


def test_remove_mean_pv_zeros_both_layers():
    grid = _grid()
    state = TwoLayerState(
        q1=_random_modal(grid, 5).at[0, 0].set(3.0),
        q2=_random_modal(grid, 6).at[0, 0].set(-4.0),
    )
    cleaned = remove_mean_pv(state)

    assert float(cleaned.q1[0, 0]) == 0.0
    assert float(cleaned.q2[0, 0]) == 0.0


def test_two_layer_inversion_preserves_batch_shape():
    grid = _grid()
    psi = TwoLayerPsi(
        psi1=jnp.stack([_random_modal(grid, 7), _random_modal(grid, 8)]),
        psi2=jnp.stack([_random_modal(grid, 9), _random_modal(grid, 10)]),
    )
    q = pv_from_streamfunction(psi, grid, F1=0.6, F2=0.5)
    recovered = invert_streamfunction(q, grid, F1=0.6, F2=0.5)

    assert recovered.psi1.shape == (2,) + grid.modal_shape
    assert recovered.psi2.shape == (2,) + grid.modal_shape
    assert np.allclose(np.asarray(recovered.psi1), np.asarray(psi.psi1), atol=5e-6)
    assert np.allclose(np.asarray(recovered.psi2), np.asarray(psi.psi2), atol=5e-6)
